"""docs/v2_data_design.md의 조건대로 v2 설정 파일을 만든다.

configs/v2/v2_train_s42~s45.json, v2_validation.json, v2_test.json을 생성한다.
학습 시나리오는 시드마다 새로 뽑아, 시드 4개가 서로 다른 시장을 본다.

    .venv/bin/python experiments/data_collection/make_v2_configs.py

- 트래픽 dc, amplitude, phase는 시나리오마다 한 번 고르게 뽑아 하루 동안 유지한다.
- 경쟁 입찰 수 계수(0.5~4)와 가격 배율(0.5~2)은 로그 척도로 뽑는다. 두 변수는
  각자 120~240분 간격을 따로 뽑아 값을 바꾼다.
- 급변형은 변경 시점에 즉시 바꾼다. 점진형은 30~60분 동안 매분 로그 척도로
  일정 비율씩 바꾼 뒤 다음 변경까지 유지한다.
- 생성 난수 시드가 고정이라 다시 실행해도 같은 설정이 나온다.
"""
import json
import math
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulator.experiment_config import ExperimentConfig  # noqa: E402

CONFIG_DIR = PROJECT_ROOT / "configs" / "v2"
GENERATION_SEED = 20261104

MINUTES = 1440
CAMPAIGNS = {"count": 10, "daily_budget": 1000.0}
PI = [
    {"kp": 1.0, "ki": 0.01},
    {"kp": 5.0, "ki": 0.05},
    {"kp": 50.0, "ki": 0.5},
]

DC_RANGE = (1500.0, 4500.0)
AMPLITUDE_RANGE = (0.2, 0.9)
COMPETITION_RANGES = {
    "untracked_bid_factor": (0.5, 4.0),
    "untracked_bid_price_multiplier": (0.5, 2.0),
}
INTERVAL_RANGE = (120, 240)
RAMP_RANGE = (30, 60)

# (실험 ID, 시드, 급변형 수, 점진형 수)
SPLITS = [
    ("v2_train_s42", [42], 9, 9),
    ("v2_train_s43", [43], 9, 9),
    ("v2_train_s44", [44], 9, 9),
    ("v2_train_s45", [45], 9, 9),
    ("v2_validation", [46], 12, 12),
    ("v2_test", [47], 12, 12),
]


def _log_uniform(rng, low, high):
    return float(math.exp(rng.uniform(math.log(low), math.log(high))))


def _competition_changes(rng, name, kind):
    """한 변수의 처음 값과 {분: 값} 변경 일정을 만든다."""
    low, high = COMPETITION_RANGES[name]
    initial = _log_uniform(rng, low, high)
    changes = {}
    current = initial
    minute = 0
    while True:
        minute += int(rng.integers(INTERVAL_RANGE[0], INTERVAL_RANGE[1] + 1))
        if minute >= MINUTES:
            break
        target = _log_uniform(rng, low, high)
        if kind == "abrupt":
            changes[minute] = target
        else:
            ramp = int(rng.integers(RAMP_RANGE[0], RAMP_RANGE[1] + 1))
            for step in range(1, ramp + 1):
                at = minute + step - 1
                if at >= MINUTES:
                    break
                changes[at] = current * (target / current) ** (step / ramp)
        current = target
    return initial, changes


def make_scenario(rng, scenario_id, kind):
    traffic = {
        "dc": round(float(rng.uniform(*DC_RANGE)), 1),
        "amplitude": round(float(rng.uniform(*AMPLITUDE_RANGE)), 4),
        "phase": round(float(rng.uniform(0.0, 1.0)), 4),
    }
    initial = {}
    by_minute = {}
    for name in COMPETITION_RANGES:
        start, changes = _competition_changes(rng, name, kind)
        initial[name] = round(start, 4)
        for minute, value in changes.items():
            by_minute.setdefault(minute, {})[name] = round(value, 4)
    return {
        "id": scenario_id,
        "minutes": MINUTES,
        "campaigns": dict(CAMPAIGNS),
        "initial_condition": {"traffic": traffic, "competition": initial},
        "changes": [
            {"minute": minute, "competition": by_minute[minute]}
            for minute in sorted(by_minute)
        ],
    }


def make_config(split_index, experiment_id, seeds, abrupt, gradual):
    rng = np.random.default_rng([GENERATION_SEED, split_index])
    prefix = experiment_id.removeprefix("v2_")
    scenarios = []
    for kind, count in (("abrupt", abrupt), ("gradual", gradual)):
        for number in range(1, count + 1):
            scenario_id = f"v2_{prefix}_{kind}_{number:02d}"
            scenarios.append(make_scenario(rng, scenario_id, kind))
    return {
        "experiment_id": experiment_id,
        "seeds": seeds,
        "pi_coefficients": PI,
        "scenarios": scenarios,
    }


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for index, (experiment_id, seeds, abrupt, gradual) in enumerate(SPLITS):
        config = make_config(index, experiment_id, seeds, abrupt, gradual)
        ExperimentConfig.from_dict(config)
        runs = len(config["scenarios"]) * len(seeds) * len(PI)
        total += runs
        path = CONFIG_DIR / f"{experiment_id}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(f"{path.relative_to(PROJECT_ROOT)}: {runs} runs")
    print(f"total {total} runs")


if __name__ == "__main__":
    main()
