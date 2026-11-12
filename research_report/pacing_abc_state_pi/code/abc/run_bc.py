import copy
import fcntl
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

out = Path(__file__).resolve().parent
repo = out.parents[1]
sys.path.insert(0, str(repo / "experiments/pacing_abc"))
from core import Config, Gains, PI, PredictiveBase, generate, rollout
from run import csv_write, read_json, source_hash, summarize, write_json


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def train():
    data = out / "train_prediction.csv"
    if sha(data) != manifest["sha256"]:
        raise ValueError("학습 CSV SHA-256 불일치")
    with data.open() as file:
        header = file.readline().strip().split(",")
        columns = [header.index(f"x{i}") for i in range(5 + 7 * cfg.history)]
        columns.append(header.index("target_spend_fraction"))
        table = np.loadtxt(file, delimiter=",", usecols=columns, dtype=np.float32)
    if len(table) != manifest["row_count"] or len(table) != cfg.train_days * cfg.ticks:
        raise ValueError("학습 CSV 행 수 불일치")

    cut = (cfg.train_days * 3 // 4) * cfg.ticks
    x = table[:, :-1]
    y = table[:, -1]
    scaler = StandardScaler().fit(x[:cut])
    x = scaler.transform(x)
    y_mean, y_std = float(y[:cut].mean()), float(y[:cut].std())
    z = (y - y_mean) / y_std

    linear = LinearRegression().fit(x[:cut], z[:cut])
    neural = MLPRegressor(hidden_layer_sizes=(64, 32), solver="adam",
                          random_state=42, alpha=1e-4, batch_size=1024,
                          learning_rate_init=1e-3)
    best_loss, best_epoch, best = float("inf"), 0, None
    curve = []
    for epoch in range(1, 31):
        neural.partial_fit(x[:cut], z[:cut])
        loss = float(np.mean((neural.predict(x[cut:]) - z[cut:]) ** 2))
        curve.append({"epoch": epoch, "internal_rmse": np.sqrt(loss) * y_std})
        print(f"C epoch {epoch:02d} | internal RMSE {curve[-1]['internal_rmse']:.8f}")
        if loss < best_loss:
            best_loss, best_epoch, best = loss, epoch, copy.deepcopy(neural)
        if epoch - best_epoch >= 5:
            break

    csv_write(out / "C_learning_curve.csv", curve)
    write_json(out / "prediction_metrics.json", {
        "fit_days": 750, "internal_validation_days": 250, "C_best_epoch": best_epoch,
        "B_RMSE": float(np.sqrt(np.mean((linear.predict(x[cut:]) - z[cut:]) ** 2)) * y_std),
        "C_RMSE": float(np.sqrt(best_loss) * y_std),
    })
    joblib.dump((scaler, y_mean, y_std, linear, best), out / "predictors.joblib")


def evaluate(split, models):
    dest = out / split
    dest.mkdir(exist_ok=True)
    days = list(cfg.ids(split))
    rows = {name: [] for name in models}
    for index, day in enumerate(days, 1):
        checkpoint = dest / f"day_{day}.json"
        if not checkpoint.exists():
            episode = generate(cfg, day)
            scores = {}
            for name, controller in models.items():
                scores[name], trace = rollout(episode, cfg, controller)
                if index == 1:
                    csv_write(dest / f"trace_{name}_{day}.csv", trace)
            write_json(checkpoint, scores)
        scores = read_json(checkpoint)
        if len({score["tape_sha256"] for score in scores.values()}) != 1:
            raise ValueError(f"{day}: 모델별 경매 데이터 불일치")
        for name in models:
            rows[name].append(scores[name])
        if index == 1 or index % 20 == 0 or index == len(days):
            print(f"{split}: {index}/{len(days)}일")

    csv_write(dest / "episodes.csv", [
        {"model": name, **row} for name, scores in rows.items() for row in scores
    ])
    summary = {name: summarize(scores, cfg) for name, scores in rows.items()}
    metrics = ("tracking_error", "unspent", "objective", "multiplier_tv")
    if split == "validation":
        for metric in metrics:
            if not np.isclose(summary["A"][metric]["mean"], frozen_a["validation"][metric],
                              rtol=0, atol=1e-10):
                raise ValueError(f"A의 보관된 검증 결과와 다릅니다: {metric}")

    rng = np.random.default_rng(cfg.bootstrap_seed)
    samples = rng.integers(len(days), size=(2000, len(days)))
    pairs = []
    for new, base in (("B", "A"), ("C", "B"), ("C", "A")):
        for metric in metrics:
            delta = np.array([x[metric] - y[metric] for x, y in zip(rows[new], rows[base])])
            lo, hi = np.quantile(delta[samples].mean(axis=1), [.025, .975])
            pairs.append(dict(comparison=f"{new}-{base}", metric=metric,
                              mean_difference=delta.mean(), ci95_low=lo, ci95_high=hi))
    csv_write(dest / "paired.csv", pairs)
    write_json(dest / "summary.json", summary)
    for name, result in summary.items():
        print(f"{name} | 추종 오차 {result['tracking_error']['mean']:.4%}"
              f" | 미소진율 {result['unspent']['mean']:.4%}"
              f" | 승수 변동 {result['multiplier_tv']['mean']:.6f}")


split = sys.argv[1] if len(sys.argv) > 1 else "validation"
if split not in ("validation", "test"):
    raise SystemExit("사용법: run_bc.py [validation|test]")

archive = repo / "research_report/pacing_abc_A/20261106_132615_119414/results"
common = archive / "pacing_abc_strong_v2/common_data"
a_path = archive / "pacing_abc_ki_check_v3/frozen_A.json"
protocol = read_json(common / "protocol.json")
manifest = read_json(common / "train_prediction_manifest.json")
frozen_a = read_json(a_path)
if protocol["source_sha256"] != source_hash() or protocol["numpy"] != np.__version__:
    raise SystemExit("A 실행 당시의 core.py/run.py 또는 NumPy 버전과 다릅니다.")
if frozen_a["status"] != "KI_RANGE_CHECK_PASSED":
    raise SystemExit("A의 최종 동결 계수를 확인하세요.")

params = frozen_a["parameters"]
cfg = replace(Config(**protocol["config"]), initial_multiplier=params["m0"])
gains = Gains(params["kp"], params["ki"])

lock = (out / "run.lock").open("a")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
freeze = out / "frozen_ABC.json"
predictors = out / "predictors.joblib"
if not freeze.exists():
    if split == "test":
        raise SystemExit("먼저 validation을 실행해 B·C를 학습하세요.")
    train()
signature = {"A": sha(a_path), "protocol": sha(common / "protocol.json"),
             "train_csv": manifest["sha256"], "predictors": sha(predictors),
             "runner": sha(Path(__file__)), "core": source_hash(),
             "numpy": np.__version__, "sklearn": sklearn.__version__}
if freeze.exists() and read_json(freeze) != signature:
    raise SystemExit("이 실행의 동결 파일과 현재 코드 또는 모델이 다릅니다.")
write_json(freeze, signature)
if split == "test":
    write_json(out / "TEST_ACCESSED.json", signature)

scaler, y_mean, y_std, linear, neural = joblib.load(predictors)
models = {"A": PI(cfg, gains)}
for name, predictor in (("B", linear), ("C", neural)):
    def predict(batch, model=predictor):
        return model.predict(scaler.transform(batch).astype(np.float32)) * y_std + y_mean
    models[name] = PI(cfg, gains, PredictiveBase(cfg, predict))
evaluate(split, models)
