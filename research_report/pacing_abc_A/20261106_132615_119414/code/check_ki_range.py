
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

from core import Config, Gains, PI, generate, rollout
from run import read_json, write_json, csv_write, source_hash

root = Path(__file__).resolve().parents[2]
previous = root / "outputs/pacing_abc_strong_v2"
out = root / "outputs/pacing_abc_ki_check_v3"
out.mkdir(parents=True, exist_ok=True)

import fcntl
lock = (out / "run.lock").open("a")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("Already running")

old = read_json(previous / "frozen_A.json")
protocol = read_json(previous / "strong_protocol.json")
if protocol["source"] != source_hash():
    raise SystemExit("core.py 또는 run.py가 변경됐습니다. 중단합니다.")
if protocol["numpy"] != np.__version__:
    raise SystemExit("NumPy 버전이 변경됐습니다. 중단합니다.")

cfg = Config(**protocol["config"])
base = old["parameters"]
ki_values = sorted(set([
    0., 100., 300., 1000., 3000., 10000.,
    30000., 100000., 300000., 1000000.,
    float(base["ki"]),
]))
dev_days = protocol["development_days"]
val_days = list(cfg.ids("validation"))
audit_days = protocol["audit_days"]

metadata = {
    "previous_protocol": old["protocol_sha256"],
    "previous_parameters": base,
    "source": source_hash(),
    "runner_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "numpy": np.__version__,
    "ki_candidates": ki_values,
    "fixed_parameters": {"kp": base["kp"], "m0": base["m0"]},
    "development_days": dev_days,
    "validation_days": val_days,
    "audit_days": audit_days,
    "selection": "Shortlist by development, select by validation J",
    "J": "tracking_error + unspent",
    "test_opened": False,
    "scope": "One-dimensional Ki range diagnostic; not global optimality",
}
meta_path = out / "protocol.json"
if meta_path.exists():
    if read_json(meta_path) != metadata:
        raise SystemExit("설정이 달라 기존 결과에 이어 쓸 수 없습니다.")
else:
    write_json(meta_path, metadata)

if (out / "DONE.json").exists():
    print((out / "DONE.json").read_text(), flush=True)
    raise SystemExit(0)

cache = out / "scores"
cache.mkdir(exist_ok=True)

@lru_cache(maxsize=256)
def episode(day):
    return generate(cfg, day)

def evaluate(ki, days):
    rows = []
    for day in days:
        path = cache / f"ki_{ki:.12g}_day_{day}.json"
        if path.exists():
            result = read_json(path)
        else:
            active = replace(cfg, initial_multiplier=base["m0"])
            result = rollout(
                episode(day), active,
                PI(active, Gains(base["kp"], ki)),
            )[0]
            write_json(path, result)
        rows.append(result)
    return rows

def average(rows):
    fields = (
        "objective", "tracking_error", "unspent", "multiplier_tv"
    )
    return {
        field: float(np.mean([row[field] for row in rows]))
        for field in fields
    }

development = []
for ki in ki_values:
    row = {"ki": ki, **average(evaluate(ki, dev_days))}
    development.append(row)
    print("DEVELOPMENT", json.dumps(row), flush=True)

csv_write(out / "ki_development.csv", development)
ranked = sorted(development, key=lambda r: (r["objective"], r["ki"]))

# 사전 규칙: 개발 상위 3개 후보와 기존 값을 검증한다.
shortlist = sorted(set(
    [row["ki"] for row in ranked[:3]] + [float(base["ki"])]
))

validation = []
for ki in shortlist:
    row = {"ki": ki, **average(evaluate(ki, val_days))}
    validation.append(row)
    print("VALIDATION", json.dumps(row), flush=True)

csv_write(out / "ki_validation.csv", validation)
selected = min(
    validation,
    key=lambda r: (r["objective"], r["tracking_error"], r["ki"]),
)
params = {**base, "ki": selected["ki"]}

# 설정 선택에 쓰지 않은 개발 점검 환경에서 동일한 날끼리 비교.
before = evaluate(float(base["ki"]), audit_days)
after = evaluate(params["ki"], audit_days)
differences = np.array([
    b["objective"] - a["objective"]
    for a, b in zip(before, after)
])
rng = np.random.default_rng(914)
samples = rng.choice(
    differences, (2000, len(differences)), replace=True
).mean(axis=1)

upper = max(ki_values)
interior = params["ki"] < upper
upper_dev = next(r for r in development if r["ki"] == upper)
selected_dev = next(
    r for r in development if r["ki"] == params["ki"]
)
bracketed = (
    interior
    and upper_dev["objective"] > selected_dev["objective"]
)
status = (
    "KI_RANGE_CHECK_PASSED"
    if bracketed else "KI_RANGE_NOT_RESOLVED"
)

report = {
    "status": status,
    "model": "A",
    "parameters": params,
    "validation": selected,
    "same_day_audit_before": average(before),
    "same_day_audit_after": average(after),
    "audit_J_difference": float(differences.mean()),
    "audit_difference_bootstrap95": list(map(
        float, np.quantile(samples, [.025, .975])
    )),
    "ki_upper_checked": upper,
    "ki_upper_worse_on_development": bracketed,
    "common_cold_start_for_ABC": params["m0"],
    "comparison_rule": (
        "Primary ABC: common selected PI gains and cold start; "
        "secondary: equal end-to-end search budgets."
    ),
    "test_opened": False,
    "scope": (
        "Completed declared finite search and Ki range check. "
        "Not proof of global optimum or real-market validity."
    ),
}
write_json(out / "frozen_A.json", report)
write_json(out / "DONE.json", report)
print("RESULT", json.dumps(report, ensure_ascii=False), flush=True)
