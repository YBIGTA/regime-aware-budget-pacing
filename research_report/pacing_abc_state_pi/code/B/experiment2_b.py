import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import sklearn

out = Path(__file__).resolve().parent
run_dir = out.parent
repo = run_dir.parents[1]
sys.path.insert(0, str(repo / "experiments/pacing_abc"))
from core import Config, Gains, PI, features, generate, rollout
from run import csv_write, read_json, source_hash, write_json


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


archive = repo / "research_report/pacing_abc_A/20261106_132615_119414/results"
protocol_path = archive / "pacing_abc_strong_v2/common_data/protocol.json"
a_path = archive / "pacing_abc_ki_check_v3/frozen_A.json"
predictor_path = run_dir / "predictors.joblib"
frozen = read_json(run_dir / "frozen_ABC.json")
checks = {
    "A": sha(a_path),
    "protocol": sha(protocol_path),
    "predictors": sha(predictor_path),
    "core": source_hash(),
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
}
for key, value in checks.items():
    if frozen[key] != value:
        raise SystemExit(f"기존 ABC 실행과 일치하지 않습니다: {key}")

params = read_json(a_path)["parameters"]
cfg = replace(Config(**read_json(protocol_path)["config"]),
              initial_multiplier=params["m0"])
a_gains = Gains(params["kp"], params["ki"])
gain_scale = np.array([a_gains.kp, a_gains.ki])
grid = np.linspace(cfg.multiplier_min, cfg.multiplier_max, 101)
scaler, y_mean, y_std, linear, _ = joblib.load(predictor_path)

TRAIN_DAYS = tuple(range(400000, 400024))
VALID_DAYS = tuple(cfg.ids("validation"))
for split in ("train", "validation", "test"):
    if set(TRAIN_DAYS).intersection(cfg.ids(split)):
        raise ValueError(f"계수 학습 날짜가 기존 {split} 날짜와 겹칩니다.")


def base_and_slope(obs):
    if obs.tick == 0:
        return cfg.initial_multiplier, 0.0

    # 기존 C와 같은 입력 101개. 후보 승수 열만 달라진다.
    batch = np.repeat(features(obs, 0.0)[None, :], len(grid), axis=0)
    batch[:, 0] = grid
    normalized = scaler.transform(batch).astype(np.float32)
    predicted = np.maximum(linear.predict(normalized) * y_std + y_mean, 0)
    target = 1 / obs.ticks
    chosen = int(np.argmin(np.abs(predicted.astype(float) - target)))

    left, right = max(0, chosen - 5), min(len(grid) - 1, chosen + 5)
    slope = (float(predicted[right]) - float(predicted[left])) / (grid[right] - grid[left])
    return float(grid[chosen]), slope


def state_features(obs, base, slope):
    recent = [f for f in obs.history[-6:] if f.tick >= 0]
    recent_error = len(recent) / obs.ticks - sum(f.spend for f in recent) / obs.budget
    traffic = np.array([f.opportunities for f in recent], dtype=float)
    traffic_variation = float(traffic.std() / max(1., traffic.mean())) if len(traffic) else 0.
    return np.array([
        2 * obs.tick / obs.ticks - 1,
        2 * (1 - obs.cumulative_spend / obs.budget) - 1,
        np.tanh(20 * obs.error),
        np.tanh(50 * recent_error),
        np.tanh(traffic_variation),
        2 * base - 1,
        np.tanh(slope * obs.ticks / 10),
    ])


class GainMLP:
    """7 inputs, 4 tanh units, and 1 or 2 gain outputs."""
    def __init__(self, anchor, weights, outputs):
        self.anchor = np.asarray(anchor, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.outputs = outputs
        self.hidden = self.weights[:32].reshape(8, 4)
        self.head = self.weights[32:].reshape(4, outputs)

    def __call__(self, inputs):
        hidden = np.tanh(inputs @ self.hidden[:7] + self.hidden[7])
        adjustment = np.zeros(2)
        adjustment[:self.outputs] = np.tanh(hidden @ self.head)
        gains = np.clip(self.anchor + adjustment * gain_scale, 0., 2 * gain_scale)
        if self.outputs == 1:
            gains[1] = 0.
        return gains


class PredictivePI:
    def __init__(self, fixed, schedule=None):
        self.fixed = fixed
        self.schedule = schedule

    def reset(self):
        self.integral = 0.

    def act(self, obs):
        base, slope = base_and_slope(obs)
        if self.schedule is None:
            kp, ki = self.fixed
        else:
            kp, ki = self.schedule(state_features(obs, base, slope))

        # 현재 계수는 이번 구간의 적분 증가분에 적용한다.
        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw = base + kp * error + candidate
        lo, hi = cfg.multiplier_min, cfg.multiplier_max
        blocked = (raw > hi and error > 0) or (raw < lo and error < 0)
        if blocked:
            raw = base + kp * error + self.integral
        else:
            self.integral = candidate

        return float(np.clip(raw, lo, hi)), {
            "base": base, "raw": float(raw), "integral": float(self.integral),
            "integral_blocked": bool(blocked), "saturated": bool(raw < lo or raw > hi),
            "kp": float(kp), "ki": float(ki),
        }


train = [generate(cfg, day) for day in TRAIN_DAYS]
search_rows = []


def train_score(policy):
    return float(np.mean([rollout(ep, cfg, policy)[0]["objective"] for ep in train]))


fixed_results = []
for p in (0., 0.25, 0.5, 1., 2.):
    for i in (0., 0.1, 0.25, 0.5, 1., 2.):
        fixed = (p * a_gains.kp, i * a_gains.ki)
        loss = train_score(PredictivePI(fixed))
        fixed_results.append((loss, fixed))
        search_rows.append(dict(model="fixed_grid", trial=len(fixed_results),
                                kp=fixed[0], ki=fixed[1], objective=loss))
        print(f"고정 계수 {len(fixed_results):02d}/30 | "
              f"Kp={fixed[0]:.3f} Ki={fixed[1]:.3f} | 점수 {loss:.6%}")

pi_loss, fixed_pi = min(fixed_results)


def fit_mlp(name, anchor, anchor_loss, outputs, seed):
    rng = np.random.default_rng(seed)
    best_weights = np.concatenate([rng.normal(0, 0.5, size=32), np.zeros(4 * outputs)])
    best_loss = anchor_loss
    search_rows.append(dict(model=name, trial=0, kp=anchor[0], ki=anchor[1],
                            objective=best_loss))

    trial = 0
    for sigma in (0.10, 0.06, 0.035, 0.02):
        center = best_weights.copy()
        for direction in rng.normal(size=(3, center.size)):
            for sign in (-1, 1):
                trial += 1
                weights = center + sign * sigma * direction
                schedule = GainMLP(anchor, weights, outputs)
                loss = train_score(PredictivePI(anchor, schedule))
                search_rows.append(dict(model=name, trial=trial, kp=anchor[0],
                                        ki=anchor[1], objective=loss))
                if loss < best_loss:
                    best_loss, best_weights = loss, weights
                print(f"{name} {trial:02d}/24 | 후보 {loss:.6%} | 최저 {best_loss:.6%}")

    return GainMLP(anchor, best_weights, outputs)


state_pi = fit_mlp("B_state_PI", fixed_pi, pi_loss, outputs=2, seed=43)
csv_write(out / "training_search.csv", search_rows)
write_json(out / "frozen_experiment2.json", {
    "source": checks,
    "script_sha256": sha(Path(__file__)),
    "training_days": list(TRAIN_DAYS),
    "validation_days": list(VALID_DAYS),
    "gain_caps": (2 * gain_scale).tolist(),
    "B_fixed_PI": list(fixed_pi),
    "B_state_PI": {"anchor": list(fixed_pi), "outputs": 2, "weights": state_pi.weights.tolist()},
})

policies = {
    "A_PI": PI(cfg, a_gains),
    "B_fixed_PI": PredictivePI(fixed_pi),
    "B_state_PI": PredictivePI(fixed_pi, state_pi),
}
daily = []
for index, day in enumerate(VALID_DAYS, 1):
    episode = generate(cfg, day)
    prior = read_json(run_dir / "validation" / f"day_{day}.json")
    if prior["A"]["tape_sha256"] != episode.digest():
        raise ValueError(f"{day}: 경매 데이터가 기존 검증과 다릅니다.")

    for name, policy in policies.items():
        score, trace = rollout(episode, cfg, policy)
        row = {"model": name, **score}
        for gain, key in (("kp", "kp"), ("ki", "ki")):
            values = np.array([item.get(key, getattr(a_gains, gain)) for item in trace])
            row[f"{gain}_mean"] = float(values.mean())
            row[f"{gain}_within_day_sd"] = float(values.std())
        daily.append(row)

        if index == 1:
            csv_write(out / f"trace_{name}_{day}.csv", trace)

    if index == 1 or index % 10 == 0 or index == len(VALID_DAYS):
        print(f"검증 {index}/{len(VALID_DAYS)}일")

csv_write(out / "daily_scores.csv", daily)
metrics = ("objective", "tracking_error", "unspent", "multiplier_tv")
lookup = {(r["model"], r["day"]): r for r in daily}

# 이미 검증한 C와 직접 비교한다. 날짜마다 경매 해시도 확인한다.
import csv
c_dir = run_dir / "experiment2.3IKJDN"
c_frozen = read_json(c_dir / "frozen_experiment2.json")
if (c_frozen["source"] != checks
        or c_frozen["training_days"] != list(TRAIN_DAYS)
        or c_frozen["validation_days"] != list(VALID_DAYS)):
    raise ValueError("B와 C의 실행 환경 또는 비교 날짜가 다릅니다.")

c_rows = {}
with (c_dir / "daily_scores.csv").open(newline="") as file:
    for row in csv.DictReader(file):
        if row["model"] != "C_state_PI":
            continue
        day = int(row["day"])
        if day in c_rows:
            raise ValueError(f"C 결과에 중복 날짜가 있습니다: {day}")
        c_rows[day] = row

if set(c_rows) != set(VALID_DAYS):
    raise ValueError("C의 일별 검증 결과가 부족하거나 날짜가 다릅니다.")

for day in VALID_DAYS:
    row = c_rows[day]
    if row["tape_sha256"] != lookup["B_state_PI", day]["tape_sha256"]:
        raise ValueError(f"{day}: B와 C가 받은 경매 데이터가 다릅니다.")
    lookup["C_state_PI", day] = {key: float(row[key]) for key in metrics}


def values(name, key):
    return np.array([lookup[name, day][key] for day in VALID_DAYS])


means = []
for name in policies:
    row = {"model": name}
    for key in (*metrics, "kp_mean", "ki_mean", "kp_within_day_sd", "ki_within_day_sd"):
        row[key] = float(values(name, key).mean())
    means.append(row)
    print(f"{name:12s} | 합산 {row['objective']:.6%} | "
          f"추종 {row['tracking_error']:.6%} | 미소진 {row['unspent']:.6%}")
csv_write(out / "summary.csv", means)

pairs = (
    ("B_state_PI", "B_fixed_PI"),
    ("B_state_PI", "A_PI"),
    ("C_state_PI", "B_state_PI"),
)
rng = np.random.default_rng(cfg.bootstrap_seed)
n = len(VALID_DAYS)
sample = rng.integers(0, n, size=(20_000, n))
tail = 0.025 / len(pairs)
paired = []
for new, base in pairs:
    for metric in metrics:
        delta = values(new, metric) - values(base, metric)
        lo, hi = np.quantile(delta[sample].mean(axis=1), [tail, 1 - tail])
        paired.append(dict(new=new, baseline=base, metric=metric,
                           mean_difference=float(delta.mean()),
                           adjusted_ci95_low=float(lo), adjusted_ci95_high=float(hi)))
        if metric == "objective":
            print(f"{new} - {base} | {delta.mean()*100:+.6f}%p "
                  f"[{lo*100:+.6f}, {hi*100:+.6f}]")
csv_write(out / "paired.csv", paired)
print(f"결과 폴더  {out}")
