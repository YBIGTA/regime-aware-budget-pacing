"""두 상태 PI vs 튜닝한 고정 PI (B 선형회귀 예측기, 승수 상한 1.0, 기존 anti-windup).

상태: '많이 밀림' = (목표 누적 지출 - 실제 누적 지출) / 일일 예산 >= 0.005  (매 5분 입찰 직전, obs.error)
      '그 외'    = 나머지
상태가 바뀌어도 저장된 I항(Ki가 이미 곱해진 누적값)을 그대로 이어 쓴다. I항 감쇠 없음.
최종 테스트 500일(200000~)은 사용하지 않는다. 모든 날짜는 검증 100000~100199 안에서만 고른다.

사용법 (저장소 루트 기준):
  python research_report/pacing_two_state_pi/code/two_state_pi.py check
  python research_report/pacing_two_state_pi/code/two_state_pi.py search40 [N_TWO_STATE]
  python research_report/pacing_two_state_pi/code/two_state_pi.py eval200 [TOP_N]
  python research_report/pacing_two_state_pi/code/two_state_pi.py refine
  python research_report/pacing_two_state_pi/code/two_state_pi.py report
"""
import csv
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402

CODE = Path(__file__).resolve().parent
BUNDLE = CODE.parent
REPO = BUNDLE.parents[1]
RESULTS = BUNDLE / "results"
RESULTS.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO / "experiments/pacing_abc"))
from core import Config, features, generate, rollout  # noqa: E402
from run import read_json, source_hash, write_json  # noqa: E402

THRESHOLD = 0.005
SEARCH_SEED = 20261114          # 40일 선정과 두 상태 조합 표본 추출
BOOTSTRAP_SEED = 917            # cfg.bootstrap_seed 와 동일
VALID_DAYS = tuple(range(100000, 100200))
WORKERS = int(os.environ.get("WORKERS", "10"))


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


STATE = REPO / "research_report/pacing_abc_state_pi"
ARCHIVE = REPO / "research_report/pacing_abc_A/20261106_132615_119414/results"
protocol_path = ARCHIVE / "pacing_abc_strong_v2/common_data/protocol.json"
a_path = ARCHIVE / "pacing_abc_ki_check_v3/frozen_A.json"
predictor_path = STATE / "code/abc/predictors.joblib"
abc_frozen = read_json(STATE / "code/abc/frozen_ABC.json")
b_frozen = read_json(STATE / "code/B/frozen_experiment2.json")
CHECKS = {
    "A": sha(a_path), "protocol": sha(protocol_path), "predictors": sha(predictor_path),
    "core": source_hash(), "numpy": np.__version__, "sklearn": sklearn.__version__,
}
for _key, _value in CHECKS.items():
    if abc_frozen[_key] != _value or b_frozen["source"][_key] != _value:
        raise SystemExit(f"기존 ABC/B 실행 환경과 다릅니다: {_key}")

params = read_json(a_path)["parameters"]
cfg = replace(Config(**read_json(protocol_path)["config"]), initial_multiplier=params["m0"])
assert cfg.multiplier_max == 1.0 and tuple(cfg.ids("validation")) == VALID_DAYS
TEST_DAYS = set(cfg.ids("test"))
assert not TEST_DAYS & set(VALID_DAYS)
gain_scale = np.array([params["kp"], params["ki"]])
grid = np.linspace(cfg.multiplier_min, cfg.multiplier_max, 101)
scaler, y_mean, y_std, linear, _ = joblib.load(predictor_path)
KP0, KI0 = b_frozen["B_fixed_PI"]          # 49.543385..., 300.0

# 넓은 시작 격자: Kp 5개 x Ki 6개 = 30쌍 (기존 B 고정값, Ki=0, 더 작은/큰 값 포함)
KP_GRID = tuple(KP0 * m for m in (0.25, 0.5, 1.0, 2.0, 4.0))
KI_GRID = (0.0, 75.0, 150.0, 300.0, 1000.0, 3000.0)
PAIRS = tuple((kp, ki) for kp in KP_GRID for ki in KI_GRID)


# ---------------- 기존 B 코드와 동일한 기본 승수 예측 ----------------
def base_and_slope(obs):
    if obs.tick == 0:
        return cfg.initial_multiplier, 0.0
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
        2 * obs.tick / obs.ticks - 1, 2 * (1 - obs.cumulative_spend / obs.budget) - 1,
        np.tanh(20 * obs.error), np.tanh(50 * recent_error), np.tanh(traffic_variation),
        2 * base - 1, np.tanh(slope * obs.ticks / 10),
    ])


class GainMLP:
    def __init__(self, anchor, weights):
        self.anchor = np.asarray(anchor, dtype=float)
        weights = np.asarray(weights, dtype=float)
        self.hidden = weights[:32].reshape(8, 4)
        self.head = weights[32:].reshape(4, 2)

    def __call__(self, inputs):
        hidden = np.tanh(inputs @ self.hidden[:7] + self.hidden[7])
        return np.clip(self.anchor + np.tanh(hidden @ self.head) * gain_scale, 0.0, 2 * gain_scale)


class PredictivePI:
    """기존 experiment2_b.py 의 PredictivePI (참조용: 고정 PI, B_state_PI 재현)."""
    def __init__(self, fixed, schedule=None):
        self.fixed, self.schedule = fixed, schedule

    def reset(self):
        self.integral = 0.

    def act(self, obs):
        base, slope = base_and_slope(obs)
        kp, ki = self.fixed if self.schedule is None else self.schedule(state_features(obs, base, slope))
        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw = base + kp * error + candidate
        lo, hi = cfg.multiplier_min, cfg.multiplier_max
        blocked = (raw > hi and error > 0) or (raw < lo and error < 0)
        if blocked:
            raw = base + kp * error + self.integral
        else:
            self.integral = candidate
        return float(np.clip(raw, lo, hi)), {"base": base, "raw": float(raw), "integral": float(self.integral),
                                             "integral_blocked": bool(blocked), "saturated": bool(raw < lo or raw > hi),
                                             "kp": float(kp), "ki": float(ki)}


class TwoStatePI:
    """매 5분 입찰 직전 obs.error 로 상태를 정하고 그 상태의 (Kp, Ki)를 쓴다. I항은 이어 쓴다."""
    def __init__(self, behind, other, threshold=THRESHOLD):
        self.behind = (float(behind[0]), float(behind[1]))
        self.other = (float(other[0]), float(other[1]))
        self.threshold = threshold

    def reset(self):
        self.integral = 0.

    def act(self, obs):
        base, _ = base_and_slope(obs)
        error = obs.error
        is_behind = error >= self.threshold
        kp, ki = self.behind if is_behind else self.other
        integral_before = self.integral
        candidate = self.integral + ki * error / obs.ticks
        raw = base + kp * error + candidate
        lo, hi = cfg.multiplier_min, cfg.multiplier_max
        blocked = (raw > hi and error > 0) or (raw < lo and error < 0)
        if blocked:
            raw = base + kp * error + self.integral
        else:
            self.integral = candidate
        return float(np.clip(raw, lo, hi)), {
            "base": base, "raw": float(raw), "behind": bool(is_behind), "kp": kp, "ki": ki,
            "integral_before": float(integral_before), "integral": float(self.integral),
            "integral_blocked": bool(blocked), "saturated": bool(raw < lo or raw > hi),
        }


# ---------------- 병렬 실행 ----------------
_tapes = {}


def tape(day):
    if day in TEST_DAYS or day not in VALID_DAYS:
        raise ValueError(f"검증 200일 밖의 날짜: {day}")
    if day not in _tapes:
        if len(_tapes) >= 8:
            _tapes.clear()
        _tapes[day] = generate(cfg, day)
    return _tapes[day]


def run_one(ep, behind, other):
    score, rows = rollout(ep, cfg, TwoStatePI(behind, other))
    flags = np.array([r["behind"] for r in rows])
    return {
        "day": ep.day, "tape_sha256": score["tape_sha256"],
        "tracking_error": score["tracking_error"], "unspent": score["unspent"],
        "objective": score["objective"], "multiplier_tv": score["multiplier_tv"],
        "behind_fraction": float(flags.mean()),
        "switches": int(np.count_nonzero(flags[1:] != flags[:-1])),
    }


def _task(task):
    day, specs = task
    ep = tape(day)
    out = []
    for cid, behind, other in specs:
        row = run_one(ep, behind, other)
        row["cid"] = cid
        out.append(row)
    return out


def run_grid(candidates, days, label, chunk=40):
    """candidates: {cid: (behind, other)}  -> list of daily rows"""
    items = list(candidates.items())
    tasks = []
    for day in days:
        for i in range(0, len(items), chunk):
            tasks.append((day, [(cid, b, o) for cid, (b, o) in items[i:i + chunk]]))
    total = len(items) * len(days)
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for k, out in enumerate(pool.imap_unordered(_task, tasks), 1):
            rows.extend(out)
            if k % max(1, len(tasks) // 10) == 0 or k == len(tasks):
                el = time.time() - t0
                print(f"[{label}] {len(rows)}/{total} 실행 | {el:.0f}s | "
                      f"{el / len(rows) * WORKERS:.3f}s/실행(코어당)", flush=True)
    return rows, time.time() - t0


def cid_of(behind, other):
    return f"B{behind[0]:.6f}_{behind[1]:.6f}|O{other[0]:.6f}_{other[1]:.6f}"


def summarize(rows, specs):
    by = {}
    for r in rows:
        by.setdefault(r["cid"], []).append(r)
    out = []
    for cid, rs in by.items():
        behind, other = specs[cid]
        out.append({
            "cid": cid, "kind": "fixed" if behind == other else "two_state",
            "behind_kp": behind[0], "behind_ki": behind[1], "other_kp": other[0], "other_ki": other[1],
            "days": len(rs),
            "tracking_error": float(np.mean([r["tracking_error"] for r in rs])),
            "unspent": float(np.mean([r["unspent"] for r in rs])),
            "objective": float(np.mean([r["objective"] for r in rs])),
            "multiplier_tv": float(np.mean([r["multiplier_tv"] for r in rs])),
            "behind_fraction": float(np.mean([r["behind_fraction"] for r in rs])),
            "switches_per_day": float(np.mean([r["switches"] for r in rs])),
        })
    out.sort(key=lambda r: r["objective"])
    return out


def write_csv(path, rows):
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="") as f:
        return list(csv.DictReader(f))


def log_time(stage, seconds, **extra):
    path = RESULTS / "timing.json"
    data = read_json(path) if path.exists() else {}
    data[stage] = {"seconds": round(seconds, 1), **extra}
    write_json(path, data)


def specs_from_summary(rows):
    return {r["cid"]: ((float(r["behind_kp"]), float(r["behind_ki"])),
                       (float(r["other_kp"]), float(r["other_ki"]))) for r in rows}


# ---------------- 단계 ----------------
def stored_scores(model):
    rows = [r for r in read_csv(STATE / "results/daily_scores.csv") if r["model"] == model]
    return {int(r["day"]): r for r in rows}


def cmd_check():
    t0 = time.time()
    report = {"checks": CHECKS, "threshold": THRESHOLD}
    day = VALID_DAYS[0]
    ep = generate(cfg, day)

    # 1) 실행 속도 (단일 코어)
    t = time.time()
    s_fixed_ref, tr_ref = rollout(ep, cfg, PredictivePI((KP0, KI0)))
    report["seconds_per_rollout_reference_fixed"] = time.time() - t
    t = time.time()
    s_two, tr_two = rollout(ep, cfg, TwoStatePI((KP0, KI0), (KP0, KI0)))
    report["seconds_per_rollout_two_state"] = time.time() - t

    # 2) 같은 계수를 두 상태에 배정하면 고정 PI와 같은가 (여러 계수, 3일)
    same = []
    for d in VALID_DAYS[:3]:
        e = generate(cfg, d)
        for g in ((KP0, KI0), (KP0 * 2, 1000.0), (KP0 * 0.5, 0.0)):
            a, ta = rollout(e, cfg, PredictivePI(g))
            b, tb = rollout(e, cfg, TwoStatePI(g, g))
            identical = all(x["action"] == y["action"] and x["cumulative_spend"] == y["cumulative_spend"]
                            for x, y in zip(ta, tb))
            same.append({"day": d, "gains": g, "identical_trace": identical,
                         "objective_fixed": a["objective"], "objective_two_state": b["objective"]})
    report["same_gains_equals_fixed"] = same
    assert all(s["identical_trace"] for s in same), "동일 계수 재현 실패"

    # 3) 저장된 B_fixed_PI / B_state_PI 점수 재현 (환경 확인)
    fixed_stored, state_stored = stored_scores("B_fixed_PI"), stored_scores("B_state_PI")
    repro = []
    for d in VALID_DAYS[:3]:
        e = generate(cfg, d)
        f, _ = rollout(e, cfg, PredictivePI((KP0, KI0)))
        repro.append({"model": "B_fixed_PI", "day": d,
                      "tape_match": f["tape_sha256"] == fixed_stored[d]["tape_sha256"],
                      "abs_diff_objective": abs(f["objective"] - float(fixed_stored[d]["objective"]))})
    st = b_frozen["B_state_PI"]
    s, _ = rollout(ep, cfg, PredictivePI(tuple(st["anchor"]), GainMLP(st["anchor"], st["weights"])))
    repro.append({"model": "B_state_PI", "day": day,
                  "tape_match": s["tape_sha256"] == state_stored[day]["tape_sha256"],
                  "abs_diff_objective": abs(s["objective"] - float(state_stored[day]["objective"]))})
    report["stored_reproduction"] = repro
    assert all(r["tape_match"] and r["abs_diff_objective"] < 1e-10 for r in repro), repro

    # 4) 상태 분류·계수 전환·I항 연속성 (서로 다른 계수, 여러 날)
    behind, other = (KP0 * 2, 1000.0), (KP0 * 0.5, 75.0)
    checks = {"ticks": 0, "behind_ticks": 0, "switch_ticks": 0, "class_errors": 0,
              "gain_errors": 0, "integral_carry_errors": 0, "days": []}
    for d in VALID_DAYS[:5]:
        e = generate(cfg, d)
        _, rows = rollout(e, cfg, TwoStatePI(behind, other))
        prev_integral = 0.0
        prev_flag = None
        for r in rows:
            flag = r["error_before"] >= THRESHOLD
            checks["ticks"] += 1
            checks["behind_ticks"] += flag
            checks["class_errors"] += flag != r["behind"]
            exp = behind if flag else other
            checks["gain_errors"] += (r["kp"], r["ki"]) != exp
            # 직전 구간의 저장 I항이 그대로 들어와야 한다 (전환 여부와 무관)
            checks["integral_carry_errors"] += r["integral_before"] != prev_integral
            expected = prev_integral if r["integral_blocked"] else prev_integral + r["ki"] * r["error_before"] / cfg.ticks
            checks["integral_carry_errors"] += r["integral"] != expected
            if prev_flag is not None and flag != prev_flag:
                checks["switch_ticks"] += 1
            prev_integral, prev_flag = r["integral"], flag
        checks["days"].append(d)
    report["state_switch_checks"] = checks
    assert checks["class_errors"] == checks["gain_errors"] == checks["integral_carry_errors"] == 0
    assert checks["switch_ticks"] > 0, "전환이 한 번도 없어 I항 연속성을 확인하지 못했습니다"

    # 전환 직전·직후 예시 1건
    _, rows = rollout(generate(cfg, VALID_DAYS[0]), cfg, TwoStatePI(behind, other))
    for i in range(1, len(rows)):
        if rows[i]["behind"] != rows[i - 1]["behind"]:
            report["switch_example"] = [{k: rows[j][k] for k in ("tick", "error_before", "behind", "kp", "ki",
                                                                  "integral_before", "integral", "integral_blocked")}
                                        for j in (i - 1, i)]
            break

    # 5) 병렬 처리량
    specs = {cid_of(p, p): (p, p) for p in PAIRS[:20]}
    _, el = run_grid(specs, VALID_DAYS[:WORKERS], "throughput", chunk=2)
    report["parallel"] = {"workers": WORKERS, "rollouts": 20 * WORKERS, "seconds": el,
                          "rollouts_per_second": 20 * WORKERS / el}
    write_json(RESULTS / "check.json", report)
    log_time("check", time.time() - t0)
    print(json.dumps({k: report[k] for k in ("seconds_per_rollout_two_state", "parallel",
                                             "state_switch_checks", "switch_example")}, indent=1, default=str))


def cmd_search40(n_two=510):
    t0 = time.time()
    rng = np.random.default_rng(SEARCH_SEED)
    days = tuple(sorted(int(d) for d in rng.choice(VALID_DAYS, size=40, replace=False)))
    # 두 상태 조합: 대각선(=고정 PI) 제외 870개 중 균형 표본. 매 라운드 무작위 순열로
    # '많이 밀림' 쌍 30개와 '그 외' 쌍 30개를 각각 한 번씩 짝지어 각 상태에 고르게 배정.
    chosen, seen = [], set()
    if n_two >= len(PAIRS) * (len(PAIRS) - 1):   # 속도가 충분하면 870조합 전부
        chosen = [(i, j) for i in range(len(PAIRS)) for j in range(len(PAIRS)) if i != j]
    while len(chosen) < n_two:
        perm = rng.permutation(len(PAIRS))
        for i in rng.permutation(len(PAIRS)):
            j = int(perm[i])
            if i == j or (i, j) in seen:
                continue
            seen.add((i, j))
            chosen.append((int(i), j))
            if len(chosen) == n_two:
                break
    specs = {cid_of(p, p): (p, p) for p in PAIRS}
    specs.update({cid_of(PAIRS[i], PAIRS[j]): (PAIRS[i], PAIRS[j]) for i, j in chosen})
    rows, el = run_grid(specs, days, "search40")
    write_csv(RESULTS / "search40_daily.csv", rows)
    summ = summarize(rows, specs)
    write_csv(RESULTS / "search40_summary.csv", summ)
    counts_b = np.bincount([i for i, _ in chosen], minlength=30)
    counts_o = np.bincount([j for _, j in chosen], minlength=30)
    write_json(RESULTS / "search40_design.json", {
        "seed": SEARCH_SEED,
        "day_rule": "검증 100000~100199에서 np.random.default_rng(20261114).choice(200개, 40, replace=False), 정렬",
        "days": days, "kp_grid": KP_GRID, "ki_grid": KI_GRID,
        "fixed_candidates": len(PAIRS), "two_state_candidates": len(chosen),
        "two_state_design": ("대각선 제외 870조합 전부" if len(chosen) == 870 else "대각선 제외 870조합 중 무작위 순열 라운드로 균형 추출"),
        "behind_pair_count_min_max": [int(counts_b.min()), int(counts_b.max())],
        "other_pair_count_min_max": [int(counts_o.min()), int(counts_o.max())],
        "rollouts": len(rows), "seconds": el,
    })
    log_time("search40", time.time() - t0, rollouts=len(rows))
    for r in summ[:10]:
        print(r["kind"], r["cid"], f"{r['objective']:.6%}")
    best_fixed = next(r for r in summ if r["kind"] == "fixed")
    print("best fixed", best_fixed["cid"], f"{best_fixed['objective']:.6%}")


def merge_eval200(new_rows, specs, tag):
    path = RESULTS / "eval200_daily.csv"
    old = read_csv(path) if path.exists() else []
    for r in new_rows:
        r["stage"] = tag
    keep = [dict(r) for r in old if r["cid"] not in {x["cid"] for x in new_rows}]
    allrows = keep + new_rows
    write_csv(path, allrows)
    sp_path = RESULTS / "eval200_specs.json"
    all_specs = read_json(sp_path) if sp_path.exists() else {}
    all_specs.update({cid: [list(b), list(o)] for cid, (b, o) in specs.items()})
    write_json(sp_path, all_specs)
    rows = [{**r, **{k: float(r[k]) for k in ("tracking_error", "unspent", "objective", "multiplier_tv",
                                               "behind_fraction", "switches")}} for r in allrows]
    sp = {cid: (tuple(v[0]), tuple(v[1])) for cid, v in all_specs.items()}
    summ = summarize(rows, sp)
    for s in summ:
        s["complete_200"] = s["days"] == len(VALID_DAYS)
    write_csv(RESULTS / "eval200_summary.csv", summ)
    return summ


def evaluated200():
    p = RESULTS / "eval200_specs.json"
    return set(read_json(p)) if p.exists() else set()


def cmd_eval200(top_n=50):
    t0 = time.time()
    s40 = read_csv(RESULTS / "search40_summary.csv")
    two = [r for r in s40 if r["kind"] == "two_state"][:top_n]
    specs = {cid_of(p, p): (p, p) for p in PAIRS}           # 대조군: 30쌍 전부 200일
    specs.update(specs_from_summary(two))                   # 실험군: 40일 상위 top_n
    rows, el = run_grid(specs, VALID_DAYS, "eval200")
    summ = merge_eval200(rows, specs, "eval200")
    log_time("eval200", time.time() - t0, rollouts=len(rows), fixed=len(PAIRS), two_state=len(two))
    for kind in ("fixed", "two_state"):
        print(kind, [(r["cid"], f"{r['objective']:.6%}") for r in summ if r["kind"] == kind][:3])


def _neighbors(g, kp_f, ki_f, ki_zero):
    kp, ki = g
    out = set()
    for f in kp_f:
        out.add((kp * f, ki))
    if ki > 0:
        for f in ki_f:
            out.add((kp, ki * f))
    else:
        for v in ki_zero:
            out.add((kp, v))
    out.discard(g)
    return out


def cmd_refine(round_no=1):
    t0 = time.time()
    summ = read_csv(RESULTS / "eval200_summary.csv")
    sp = specs_from_summary(summ)
    best_f = next(r for r in summ if r["kind"] == "fixed")
    best_t = next(r for r in summ if r["kind"] == "two_state")
    gf = sp[best_f["cid"]][0]
    tb, to = sp[best_t["cid"]]
    if round_no == 1:
        kp_f, ki_f, ki_zero = (0.7, 0.85, 1.18, 1.4), (0.5, 0.7, 1.4, 2.0), (10.0, 25.0, 50.0)
    else:
        kp_f, ki_f, ki_zero = (0.92, 1.08), (0.85, 1.18), (5.0,)
    specs = {}
    # 대조군: 최적 고정쌍 주변 2차원 격자
    for kp in [gf[0] * f for f in (1.0, *kp_f)]:
        kis = [gf[1] * f for f in (1.0, *ki_f)] if gf[1] > 0 else [0.0, *ki_zero]
        for ki in kis:
            g = (kp, ki)
            specs[cid_of(g, g)] = (g, g)
    # 실험군: 최적 두 상태의 네 좌표를 하나씩 이동 + 두 상태 쌍의 동시 이동
    for nb in _neighbors(tb, kp_f, ki_f, ki_zero):
        specs[cid_of(nb, to)] = (nb, to)
    for no in _neighbors(to, kp_f, ki_f, ki_zero):
        specs[cid_of(tb, no)] = (tb, no)
    if round_no == 1:
        # '그 외'에 200일 최적 고정쌍, '많이 밀림'에 시작 격자 30쌍
        for p in PAIRS:
            if p != gf:
                specs[cid_of(p, gf)] = (p, gf)
    specs = {k: v for k, v in specs.items() if k not in evaluated200()}
    rows, el = run_grid(specs, VALID_DAYS, f"refine{round_no}")
    summ = merge_eval200(rows, specs, f"refine{round_no}")
    log_time(f"refine{round_no}", time.time() - t0, rollouts=len(rows), candidates=len(specs))
    for kind in ("fixed", "two_state"):
        print(kind, [(r["cid"], f"{r['objective']:.6%}") for r in summ if r["kind"] == kind][:3])


def cmd_report():
    summ = [r for r in read_csv(RESULTS / "eval200_summary.csv") if r["complete_200"] == "True"]
    daily = read_csv(RESULTS / "eval200_daily.csv")
    best_f = next(r for r in summ if r["kind"] == "fixed")
    best_t = next(r for r in summ if r["kind"] == "two_state")
    state_rows = stored_scores("B_state_PI")

    def per_day(cid):
        d = {int(r["day"]): r for r in daily if r["cid"] == cid}
        assert set(d) == set(VALID_DAYS)
        return d

    F, T = per_day(best_f["cid"]), per_day(best_t["cid"])
    for day in VALID_DAYS:
        assert F[day]["tape_sha256"] == T[day]["tape_sha256"] == state_rows[day]["tape_sha256"], day

    metrics = ("tracking_error", "unspent", "objective")
    arr = {name: {m: np.array([float(src[d][m]) for d in VALID_DAYS]) for m in metrics}
           for name, src in (("fixed", F), ("two_state", T), ("B_state_PI", state_rows))}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample = rng.integers(0, len(VALID_DAYS), size=(20_000, len(VALID_DAYS)))
    paired = []
    for new, base in (("two_state", "fixed"), ("two_state", "B_state_PI"), ("fixed", "B_state_PI")):
        for m in metrics:
            delta = arr[new][m] - arr[base][m]
            lo, hi = np.quantile(delta[sample].mean(axis=1), [0.025, 0.975])
            paired.append({"new": new, "baseline": base, "metric": m, "mean_difference": float(delta.mean()),
                           "ci95_low": float(lo), "ci95_high": float(hi),
                           "days_new_better": int((delta < -1e-12).sum()),
                           "days_new_worse": int((delta > 1e-12).sum())})
    write_csv(RESULTS / "paired.csv", paired)

    final = []
    for label, name, row in (("튜닝한 고정 PI", "fixed", best_f), ("선택한 두 상태 PI", "two_state", best_t),
                             ("기존 B_state_PI", "B_state_PI", None)):
        final.append({"model": label,
                      "tracking_error": arr[name]["tracking_error"].mean(),
                      "unspent": arr[name]["unspent"].mean(),
                      "objective": arr[name]["objective"].mean(),
                      "gains": "" if row is None else row["cid"],
                      "behind_fraction": "" if row is None else row["behind_fraction"],
                      "switches_per_day": "" if row is None else row["switches_per_day"]})
    write_csv(RESULTS / "final_table.csv", final)
    tb = [float(T[d]["behind_fraction"]) for d in VALID_DAYS]
    write_json(RESULTS / "selected.json", {
        "fixed": {"kp": float(best_f["behind_kp"]), "ki": float(best_f["behind_ki"])},
        "two_state": {"behind": {"kp": float(best_t["behind_kp"]), "ki": float(best_t["behind_ki"])},
                      "other": {"kp": float(best_t["other_kp"]), "ki": float(best_t["other_ki"])}},
        "two_state_behind_fraction_mean": float(np.mean(tb)),
        "two_state_days_with_behind": int(sum(x > 0 for x in tb)),
        "two_state_switches_per_day": float(best_t["switches_per_day"]),
        "fixed_behind_fraction_mean": float(best_f["behind_fraction"]),
        "candidates_200": {"fixed": sum(r["kind"] == "fixed" for r in summ),
                           "two_state": sum(r["kind"] == "two_state" for r in summ)},
    })
    for r in final:
        print(f"{r['model']:14s} 추종 {r['tracking_error']:.6%} 미소진 {r['unspent']:.6%} 합산 {r['objective']:.6%}")
    for p in paired:
        print(f"{p['new']}-{p['baseline']} {p['metric']}: {p['mean_difference'] * 100:+.6f}%p "
              f"[{p['ci95_low'] * 100:+.6f}, {p['ci95_high'] * 100:+.6f}] 개선 {p['days_new_better']} 악화 {p['days_new_worse']}")


if __name__ == "__main__":
    cmd = sys.argv[1]
    arg = int(sys.argv[2]) if len(sys.argv) > 2 else None
    {"check": lambda: cmd_check(),
     "search40": lambda: cmd_search40(arg or 510),
     "eval200": lambda: cmd_eval200(arg or 50),
     "refine": lambda: cmd_refine(arg or 1),
     "report": lambda: cmd_report()}[cmd]()
