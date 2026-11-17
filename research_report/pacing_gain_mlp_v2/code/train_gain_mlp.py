"""B 상태별 PI 계수 신경망 재학습 (CMA-ES).

기존 B_state_PI: 입력 7 -> tanh 4 -> (Kp, Ki), 별도 24일에서 무작위 탐색 24회.
여기서는 한 번에 하나씩 바꾼다.
  E1a: 학습 방법만 CMA-ES (24일 그대로)
  E1b: E1a + 학습일 80일
  E2 : E1b + '많이 밀림'(obs.error >= 0.005) 입력 추가
학습 기준
  sum  : 추종 오차 + 미소진율
  guard: sum + 10 * max(0, 추종 오차 - 학습일 고정 PI의 추종 오차)
날짜: 학습 400000~400079, 후보 선택 400080~400119, 검증 100000~100199(마지막 1회), 테스트 미사용.

사용법 (저장소 루트):
  python research_report/pacing_gain_mlp_v2/code/train_gain_mlp.py check
  python research_report/pacing_gain_mlp_v2/code/train_gain_mlp.py anchor
  python research_report/pacing_gain_mlp_v2/code/train_gain_mlp.py train E1b sum 1 [GENS]
  python research_report/pacing_gain_mlp_v2/code/train_gain_mlp.py holdout
  python research_report/pacing_gain_mlp_v2/code/train_gain_mlp.py validate
"""
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
BUNDLE = CODE.parent
REPO = BUNDLE.parents[1]
MODEL = os.environ.get("PACING_MODEL", "B")          # B: 선형회귀 예측기, C: 신경망 예측기
assert MODEL in ("B", "C")
RESULTS = BUNDLE / ("results" if MODEL == "B" else "results_C")
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "research_report/pacing_two_state_pi/code"))
import two_state_pi as lib  # noqa: E402  (cfg, 예측기, 원래 PI 구현, 해시 점검 재사용)
from core import generate, rollout  # noqa: E402
from run import read_json, write_json  # noqa: E402

cfg = lib.cfg
THRESHOLD = 0.005
GUARD_WEIGHT = 10.0
TRAIN_ORIG = tuple(range(400000, 400024))      # 기존 B_state_PI 학습일
TRAIN = tuple(range(400000, 400080))
HOLDOUT = tuple(range(400080, 400120))
VALID = lib.VALID_DAYS
for _split in ("train", "validation", "test"):
    assert not set(TRAIN + HOLDOUT) & set(cfg.ids(_split)), _split
POP = 12
SIGMA0 = 0.15
WORKERS = int(os.environ.get("WORKERS", "10"))
import joblib  # noqa: E402
if MODEL == "B":
    PREDICTOR = lib.linear
    B_STATE = lib.b_frozen["B_state_PI"]
else:
    PREDICTOR = joblib.load(lib.predictor_path)[4]
    _c_frozen = read_json(lib.STATE / "code/C/frozen_experiment2.json")
    if _c_frozen["source"] != lib.CHECKS:
        raise SystemExit("C 실행 환경이 다릅니다")
    B_STATE = _c_frozen["C_state_PI"]                   # 이름은 B_STATE 로 두지만 C의 상태별 PI
STATE_NAME, FIXED_NAME = f"{MODEL}_state_PI", f"{MODEL}_fixed_PI"


def base_and_slope(obs):
    """기존 B/C 코드의 base_and_slope 와 같은 계산. 예측기만 MODEL 에 따라 다르다."""
    if obs.tick == 0:
        return cfg.initial_multiplier, 0.0
    batch = np.repeat(lib.features(obs, 0.0)[None, :], len(lib.grid), axis=0)
    batch[:, 0] = lib.grid
    normalized = lib.scaler.transform(batch).astype(np.float32)
    predicted = np.maximum(PREDICTOR.predict(normalized) * lib.y_std + lib.y_mean, 0)
    target = 1 / obs.ticks
    chosen = int(np.argmin(np.abs(predicted.astype(float) - target)))
    left, right = max(0, chosen - 5), min(len(lib.grid) - 1, chosen + 5)
    slope = (float(predicted[right]) - float(predicted[left])) / (lib.grid[right] - lib.grid[left])
    return float(lib.grid[chosen]), slope


class GainNet:
    """입력 n_in -> tanh 4 -> 2. n_in=7이면 기존 GainMLP와 같은 계산."""
    def __init__(self, anchor, weights, n_in):
        self.anchor = np.asarray(anchor, dtype=float)
        w = np.asarray(weights, dtype=float)
        self.n_in = n_in
        self.hidden = w[:(n_in + 1) * 4].reshape(n_in + 1, 4)
        self.head = w[(n_in + 1) * 4:].reshape(4, 2)

    def __call__(self, inputs):
        hidden = np.tanh(inputs @ self.hidden[:self.n_in] + self.hidden[self.n_in])
        return np.clip(self.anchor + np.tanh(hidden @ self.head) * lib.gain_scale, 0.0, 2 * lib.gain_scale)


class StatePI:
    """기존 PredictivePI 와 같은 PI. behind_input=True 면 8번째 입력으로 많이 밀림 여부(+1/-1)."""
    def __init__(self, anchor, weights, behind_input):
        self.n_in = 8 if behind_input else 7
        self.net = GainNet(anchor, weights, self.n_in)

    def reset(self):
        self.integral = 0.

    def act(self, obs):
        base, slope = base_and_slope(obs)
        x = lib.state_features(obs, base, slope)
        if self.n_in == 8:
            x = np.append(x, 1.0 if obs.error >= THRESHOLD else -1.0)
        kp, ki = self.net(x)
        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw = base + kp * error + candidate
        lo, hi = cfg.multiplier_min, cfg.multiplier_max
        blocked = (raw > hi and error > 0) or (raw < lo and error < 0)
        if blocked:
            raw = base + kp * error + self.integral
        else:
            self.integral = candidate
        return float(np.clip(raw, lo, hi)), {"saturated": bool(raw < lo or raw > hi),
                                             "kp": float(kp), "ki": float(ki)}


class FixedPI(StatePI):
    def __init__(self, gains):
        self.gains = tuple(gains)

    def act(self, obs):
        return _fixed_act(self, obs)


def _fixed_act(self, obs):
    base, _ = base_and_slope(obs)
    kp, ki = self.gains
    error = obs.error
    candidate = self.integral + ki * error / obs.ticks
    raw = base + kp * error + candidate
    lo, hi = cfg.multiplier_min, cfg.multiplier_max
    blocked = (raw > hi and error > 0) or (raw < lo and error < 0)
    if blocked:
        raw = base + kp * error + self.integral
    else:
        self.integral = candidate
    return float(np.clip(raw, lo, hi)), {"saturated": bool(raw < lo or raw > hi), "kp": kp, "ki": ki}


# ---------------- 병렬 평가 ----------------
_tapes = {}


def tape(day):
    if day not in _tapes:
        if len(_tapes) >= 130:
            _tapes.clear()
        _tapes[day] = generate(cfg, day)
    return _tapes[day]


def make_policy(spec):
    kind = spec[0]
    if kind == "fixed":
        return FixedPI(spec[1])
    _, anchor, weights, behind_input = spec
    return StatePI(anchor, weights, behind_input)


def _task(task):
    day, specs = task
    ep = tape(day)
    out = []
    for key, spec in specs:
        s, _ = rollout(ep, cfg, make_policy(spec))
        out.append((key, day, s["tape_sha256"], s["tracking_error"], s["unspent"], s["multiplier_tv"]))
    return out


def evaluate(pool, specs, days):
    """specs: {key: spec} -> {key: {day: (te, unspent, tv, sha)}}"""
    items = list(specs.items())
    tasks = [(d, items) for d in days]
    res = {k: {} for k in specs}
    for out in pool.imap_unordered(_task, tasks):
        for key, day, sha, te, un, tv in out:
            res[key][day] = (te, un, tv, sha)
    return res


def means(per_day):
    a = np.array([v[:3] for v in per_day.values()])
    return {"tracking_error": float(a[:, 0].mean()), "unspent": float(a[:, 1].mean()),
            "objective": float(a[:, 0].mean() + a[:, 1].mean()), "multiplier_tv": float(a[:, 2].mean())}


def loss(m, objective, te_ref):
    j = m["tracking_error"] + m["unspent"]
    if objective == "guard":
        j += GUARD_WEIGHT * max(0.0, m["tracking_error"] - te_ref)
    return j


def pool():
    return mp.get_context("spawn").Pool(WORKERS)


def append_log(name, row):
    path = RESULTS / name
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def log_time(stage, seconds, **extra):
    path = RESULTS / "timing.json"
    data = read_json(path) if path.exists() else {}
    data[stage] = {"seconds": round(seconds, 1), **extra}
    write_json(path, data)


# ---------------- 단계 ----------------
def cmd_check():
    t0 = time.time()
    stored = lib.stored_scores(STATE_NAME)
    report = {"model": MODEL}
    with pool() as p:
        spec = ("state", tuple(B_STATE["anchor"]), tuple(B_STATE["weights"]), False)
        res = evaluate(p, {"orig": spec}, VALID[:3])["orig"]
        report["reproduce_state_PI"] = [
            {"day": d, "tape_match": res[d][3] == stored[d]["tape_sha256"],
             "abs_diff_objective": abs(res[d][0] + res[d][1] - float(stored[d]["objective"]))} for d in VALID[:3]]
        assert all(r["tape_match"] and r["abs_diff_objective"] < 1e-10 for r in report["reproduce_state_PI"])
        # 8입력: 추가 입력의 가중치가 0이면 7입력과 같아야 한다
        w7 = np.asarray(B_STATE["weights"])
        w8 = np.concatenate([w7[:28], np.zeros(4), w7[28:]])
        r8 = evaluate(p, {"w8": ("state", tuple(B_STATE["anchor"]), tuple(w8), True)}, VALID[:1])["w8"]
        report["eight_input_zero_weight_equal"] = r8[VALID[0]][:2] == res[VALID[0]][:2]
        assert report["eight_input_zero_weight_equal"]
        # 처리량: 후보 12개 x 20일
        specs = {f"c{i}": ("state", tuple(B_STATE["anchor"]),
                           tuple(w7 + np.random.default_rng(i).normal(0, .1, w7.size)), False) for i in range(POP)}
        t = time.time()
        evaluate(p, specs, TRAIN[:20])
        el = time.time() - t
        report["throughput_rollouts_per_s"] = POP * 20 / el
    write_json(RESULTS / "check.json", report)
    log_time("check", time.time() - t0)
    print(json.dumps(report, indent=1))


def cmd_anchor():
    """학습 80일에서 고정 PI 30쌍 재튜닝 + 기존 B_state_PI의 학습/선택일 점수."""
    t0 = time.time()
    with pool() as p:
        specs = {f"fixed_{kp:.6f}_{ki:.6f}": ("fixed", (kp, ki)) for kp, ki in lib.PAIRS}
        res = evaluate(p, specs, TRAIN)
        rows = sorted(({"key": k, "kp": specs[k][1][0], "ki": specs[k][1][1], **means(v)} for k, v in res.items()),
                      key=lambda r: r["objective"])
        lib.write_csv(RESULTS / "anchor_fixed_train80.csv", rows)
        best = rows[0]
        orig = ("state", tuple(B_STATE["anchor"]), tuple(B_STATE["weights"]), False)
        fixed_best = ("fixed", (best["kp"], best["ki"]))
        hold = evaluate(p, {"state_PI": orig, "anchor_fixed": fixed_best}, HOLDOUT)
        train_orig = evaluate(p, {"state_PI": orig}, TRAIN)["state_PI"]
    info = {"anchor": [best["kp"], best["ki"]], "anchor_train": {k: best[k] for k in ("tracking_error", "unspent", "objective")},
            "model": MODEL, "state_PI_train": means(train_orig),
            "state_PI_holdout": means(hold["state_PI"]), "anchor_holdout": means(hold["anchor_fixed"])}
    write_json(RESULTS / "anchor.json", info)
    log_time("anchor", time.time() - t0)
    print(json.dumps(info, indent=1))


def cmd_train(exp, objective, seed, gens):
    t0 = time.time()
    import cma
    a = read_json(RESULTS / "anchor.json")
    anchor, te_ref = tuple(a["anchor"]), a["anchor_train"]["tracking_error"]
    days = TRAIN_ORIG if exp == "E1a" else TRAIN
    behind_input = exp == "E2"
    n_in = 8 if behind_input else 7
    rng = np.random.default_rng(1000 * seed + {"E1a": 1, "E1b": 2, "E2": 3}[exp])
    # 기존과 같은 초기화: 은닉층 N(0, 0.5), 출력층 0  -> 시작점은 anchor 고정 PI
    x0 = np.concatenate([rng.normal(0, 0.5, (n_in + 1) * 4), np.zeros(8)])
    es = cma.CMAEvolutionStrategy(x0, SIGMA0, {"popsize": POP, "seed": seed, "verbose": -9})
    best = (np.inf, x0, None)
    name = f"{exp}_{objective}_s{seed}"
    with pool() as p:
        if days != TRAIN:   # 벌점 기준 추종 오차는 실제 학습일에서의 고정 PI 값으로 맞춘다
            te_ref = means(evaluate(p, {"anchor": ("fixed", anchor)}, days)["anchor"])["tracking_error"]
        start = means(evaluate(p, {"x0": ("state", anchor, tuple(x0), behind_input)}, days)["x0"])
        best = (loss(start, objective, te_ref), x0, start)
        for g in range(1, gens + 1):
            xs = es.ask()
            specs = {i: ("state", anchor, tuple(x), behind_input) for i, x in enumerate(xs)}
            res = evaluate(p, specs, days)
            ms = [means(res[i]) for i in range(len(xs))]
            ls = [loss(m, objective, te_ref) for m in ms]
            es.tell(xs, ls)
            i = int(np.argmin(ls))
            if ls[i] < best[0]:
                best = (ls[i], np.array(xs[i]), ms[i])
            append_log(f"curve_{name}.csv", {"gen": g, "best_loss": best[0], "gen_min": ls[i],
                                             "gen_median": float(np.median(ls)), "sigma": es.sigma})
            if g % 10 == 0 or g == gens:
                print(f"[{name}] {g}/{gens} 최저 {best[0]:.6%} | 세대 최저 {ls[i]:.6%} | {time.time() - t0:.0f}s", flush=True)
    write_json(RESULTS / f"model_{name}.json", {
        "exp": exp, "objective": objective, "seed": seed, "gens": gens, "popsize": POP, "sigma0": SIGMA0,
        "train_days": [days[0], days[-1], len(days)], "behind_input": behind_input, "anchor": list(anchor),
        "te_ref": te_ref, "start_train": start, "best_train_loss": best[0], "best_train": best[2],
        "weights": best[1].tolist(), "mean_weights": np.asarray(es.result.xfavorite).tolist(),
    })
    log_time(f"train_{name}", time.time() - t0, rollouts=gens * POP * len(days))


def cmd_holdout():
    t0 = time.time()
    a = read_json(RESULTS / "anchor.json")
    te_ref_hold = a["anchor_holdout"]["tracking_error"]
    models = sorted(RESULTS.glob("model_*.json"))
    specs = {}
    for path in models:
        m = read_json(path)
        for which in ("weights", "mean_weights"):
            specs[f"{path.stem[6:]}|{which}"] = ("state", tuple(m["anchor"]), tuple(m[which]), m["behind_input"])
    with pool() as p:
        res = evaluate(p, specs, HOLDOUT)
    rows = []
    for key, per_day in res.items():
        name, which = key.split("|")
        exp, criterion, seed = name.split("_")
        mm = means(per_day)
        rows.append({"model": name, "weights": which, "exp": exp, "criterion": criterion, "seed": seed, **mm,
                     "loss_sum": loss(mm, "sum", te_ref_hold), "loss_guard": loss(mm, "guard", te_ref_hold)})
    rows.sort(key=lambda r: (r["criterion"], r["loss_" + r["criterion"]]))
    lib.write_csv(RESULTS / "holdout.csv", rows)
    picks = {}
    for objective in ("sum", "guard"):
        cand = [r for r in rows if r["criterion"] == objective]
        best = min(cand, key=lambda r: r[f"loss_{objective}"])
        picks[objective] = {"model": best["model"], "weights": best["weights"]}
    write_json(RESULTS / "picks.json", {"te_ref_holdout": te_ref_hold, "picks": picks,
                                        "state_PI_holdout": a["state_PI_holdout"],
                                        "anchor_holdout": a["anchor_holdout"]})
    log_time("holdout", time.time() - t0, candidates=len(specs))
    for r in rows:
        print(f"{r['model']:16s} {r['weights']:12s} 추종 {r['tracking_error']:.5%} 미소진 {r['unspent']:.5%} "
              f"합산 {r['loss_sum']:.5%} 벌점 {r['loss_guard']:.5%}")
    print(json.dumps(picks, indent=1))


def cmd_validate():
    """선택한 두 모델(학습 기준별 1개)만 검증 200일에서 1회 평가."""
    t0 = time.time()
    picks = read_json(RESULTS / "picks.json")["picks"]
    specs = {}
    for objective, pk in picks.items():
        m = read_json(RESULTS / f"model_{pk['model']}.json")
        specs[f"new_{objective}"] = ("state", tuple(m["anchor"]), tuple(m[pk["weights"]]), m["behind_input"])
    with pool() as p:
        res = evaluate(p, specs, VALID)
    stored = {name: lib.stored_scores(name) for name in (STATE_NAME, FIXED_NAME)}
    arr = {}
    for key, per_day in res.items():
        for d in VALID:
            assert per_day[d][3] == stored[STATE_NAME][d]["tape_sha256"], d
        arr[key] = {"tracking_error": np.array([per_day[d][0] for d in VALID]),
                    "unspent": np.array([per_day[d][1] for d in VALID]),
                    "multiplier_tv": np.array([per_day[d][2] for d in VALID])}
    for name, src in stored.items():
        arr[name] = {m: np.array([float(src[d][m]) for d in VALID]) for m in ("tracking_error", "unspent", "multiplier_tv")}
    for v in arr.values():
        v["objective"] = v["tracking_error"] + v["unspent"]
    table = [{"model": k, **{m: float(v[m].mean()) for m in ("tracking_error", "unspent", "objective", "multiplier_tv")}}
             for k, v in arr.items()]
    lib.write_csv(RESULTS / "validation_table.csv", table)
    rng = np.random.default_rng(lib.BOOTSTRAP_SEED)
    sample = rng.integers(0, len(VALID), size=(20_000, len(VALID)))
    paired = []
    for new in specs:
        for base in (STATE_NAME, FIXED_NAME):
            for m in ("tracking_error", "unspent", "objective", "multiplier_tv"):
                delta = arr[new][m] - arr[base][m]
                lo, hi = np.quantile(delta[sample].mean(axis=1), [0.025, 0.975])
                paired.append({"new": new, "baseline": base, "metric": m, "mean_difference": float(delta.mean()),
                               "ci95_low": float(lo), "ci95_high": float(hi),
                               "days_better": int((delta < -1e-12).sum()), "days_worse": int((delta > 1e-12).sum())})
    lib.write_csv(RESULTS / "validation_paired.csv", paired)
    log_time("validate", time.time() - t0)
    for r in table:
        print(f"{r['model']:12s} 추종 {r['tracking_error']:.6%} 미소진 {r['unspent']:.6%} 합산 {r['objective']:.6%} 변동 {r['multiplier_tv']:.6f}")
    for r in paired:
        print(f"{r['new']} - {r['baseline']} {r['metric']}: {r['mean_difference'] * 100:+.6f}%p "
              f"[{r['ci95_low'] * 100:+.6f}, {r['ci95_high'] * 100:+.6f}] 개선 {r['days_better']} 악화 {r['days_worse']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "train":
        cmd_train(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]) if len(sys.argv) > 5 else 80)
    else:
        {"check": cmd_check, "anchor": cmd_anchor, "holdout": cmd_holdout, "validate": cmd_validate}[cmd]()
