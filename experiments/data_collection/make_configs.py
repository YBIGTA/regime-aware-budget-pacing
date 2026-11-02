"""docs/scenario_design_proposal.md의 실행 계획대로 설정 파일을 만든다.

configs/experiment_2_test_seeds.json ~ experiment_6_shocks.json을 생성한다.

    .venv/bin/python experiments/data_collection/make_configs.py
"""
from itertools import product
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulator.experiment_config import ExperimentConfig


CONFIG_DIR = PROJECT_ROOT / "configs"

CAMPAIGNS = {"count": 10, "daily_budget": 1000.0}
PHASE = 0.5

LEVELS = {
    "dc": {"low": 1500, "base": 3000, "high": 4500},
    "amplitude": {"low": 0.2, "base": 0.6, "high": 0.9},
    "bid_factor": {"low": 0.5, "base": 2.0, "high": 4.0},
    "price_multiplier": {"low": 0.5, "base": 1.0, "high": 2.0},
}
BASE = {name: levels["base"] for name, levels in LEVELS.items()}

TRAIN_VALIDATION_SEEDS = [42, 43, 44, 45, 46]
TEST_SEEDS = list(range(47, 62))

BASIC_PI = [
    {"kp": 1.0, "ki": 0.01},
    {"kp": 5.0, "ki": 0.05},
    {"kp": 50.0, "ki": 0.5},
]
GRID_KP = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
GRID_KI_RATIO = [0.003, 0.01, 0.03]
GRID_PI = [
    {"kp": kp, "ki": round(kp * ratio, 6)}
    for kp in GRID_KP
    for ratio in GRID_KI_RATIO
]

# 기본 시나리오 1~11: 고정 그룹 중 이름을 붙인 것
BASIC_STATIC_NAMES = {
    ("base", "base", "base", "base"): "baseline_static",
    ("low", "base", "base", "base"): "traffic_low_static",
    ("high", "base", "base", "base"): "traffic_high_static",
    ("base", "low", "base", "base"): "amplitude_low_static",
    ("base", "high", "base", "base"): "amplitude_high_static",
    ("base", "base", "low", "base"): "competition_count_low_static",
    ("base", "base", "high", "base"): "competition_count_high_static",
    ("base", "base", "base", "low"): "competition_price_low_static",
    ("base", "base", "base", "high"): "competition_price_high_static",
    ("low", "low", "low", "low"): "combined_low_static",
    ("high", "high", "high", "high"): "combined_high_static",
}

# 전환 그룹: 종류별 낮음·높음 조건
TRANSITIONS = {
    "traffic": (
        {**BASE, "dc": 1500},
        {**BASE, "dc": 4500},
    ),
    "competition": (
        {**BASE, "bid_factor": 0.5, "price_multiplier": 0.5},
        {**BASE, "bid_factor": 4.0, "price_multiplier": 2.0},
    ),
    "combined": (
        {"dc": 1500, "amplitude": 0.2, "bid_factor": 0.5,
         "price_multiplier": 0.5},
        {"dc": 4500, "amplitude": 0.9, "bid_factor": 4.0,
         "price_multiplier": 2.0},
    ),
}
TRANSITION_MINUTES = {"0600": 360, "noon": 720, "1800": 1080}

# 충격 그룹: 기준 환경에서 바꾸는 값
SHOCKS = {
    "traffic_up": {"dc": 9000},
    "traffic_down": {"dc": 900},
    "price_up": {"price_multiplier": 3.0},
    "count_up": {"bid_factor": 8.0},
}
SHOCK_MINUTES = {"0500": 300, "1500": 900}
SHOCK_DURATION = 60

BASIC_ORDER = [
    "baseline_static",
    "traffic_low_static",
    "traffic_high_static",
    "amplitude_low_static",
    "amplitude_high_static",
    "competition_count_low_static",
    "competition_count_high_static",
    "competition_price_low_static",
    "competition_price_high_static",
    "combined_low_static",
    "combined_high_static",
    "traffic_low_to_high_at_noon",
    "competition_low_to_high_at_noon",
    "combined_low_to_high_at_noon",
    "combined_high_to_low_at_noon",
]


def _condition(env):
    return {
        "traffic": {
            "dc": env["dc"],
            "amplitude": env["amplitude"],
            "phase": PHASE,
        },
        "competition": {
            "untracked_bid_factor": env["bid_factor"],
            "untracked_bid_price_multiplier": env["price_multiplier"],
        },
    }


def _change(minute, env):
    condition = _condition(env)
    del condition["traffic"]["phase"]
    return {"minute": minute, **condition}


def _scenario(scenario_id, initial, changes=()):
    scenario = {
        "id": scenario_id,
        "minutes": 1440,
        "campaigns": dict(CAMPAIGNS),
        "initial_condition": _condition(initial),
    }
    if changes:
        scenario["changes"] = [_change(minute, env) for minute, env in changes]
    return scenario


def _number_slug(value):
    return f"{value:g}"


def static_scenarios():
    scenarios = []
    names = list(LEVELS)
    for levels in product(("low", "base", "high"), repeat=len(names)):
        env = {
            name: LEVELS[name][level] for name, level in zip(names, levels)
        }
        scenario_id = BASIC_STATIC_NAMES.get(levels) or (
            f"grid_dc{_number_slug(env['dc'])}"
            f"_amp{_number_slug(env['amplitude'])}"
            f"_cnt{_number_slug(env['bid_factor'])}"
            f"_px{_number_slug(env['price_multiplier'])}"
        )
        scenarios.append(_scenario(scenario_id, env))
    return scenarios


def transition_scenarios():
    scenarios = []
    for kind, (low, high) in TRANSITIONS.items():
        for direction, (start, end) in (
            ("low_to_high", (low, high)),
            ("high_to_low", (high, low)),
        ):
            for label, minute in TRANSITION_MINUTES.items():
                scenarios.append(
                    _scenario(
                        f"{kind}_{direction}_at_{label}",
                        start,
                        [(minute, end)],
                    )
                )
    return scenarios


def shock_scenarios():
    scenarios = []
    for kind, change in SHOCKS.items():
        for label, minute in SHOCK_MINUTES.items():
            scenarios.append(
                _scenario(
                    f"shock_{kind}_at_{label}",
                    BASE,
                    [
                        (minute, {**BASE, **change}),
                        (minute + SHOCK_DURATION, BASE),
                    ],
                )
            )
    return scenarios


def basic_scenarios(static, transitions):
    by_id = {scenario["id"]: scenario for scenario in static + transitions}
    return [by_id[scenario_id] for scenario_id in BASIC_ORDER]


def build_configs():
    static = static_scenarios()
    transitions = transition_scenarios()
    shocks = shock_scenarios()
    basic = basic_scenarios(static, transitions)
    return {
        "experiment_2_test_seeds": {
            "seeds": TEST_SEEDS, "pi_coefficients": BASIC_PI,
            "scenarios": basic,
        },
        "experiment_3_static": {
            "seeds": TRAIN_VALIDATION_SEEDS, "pi_coefficients": BASIC_PI,
            "scenarios": static,
        },
        "experiment_4_pi_grid": {
            "seeds": TRAIN_VALIDATION_SEEDS, "pi_coefficients": GRID_PI,
            "scenarios": basic,
        },
        "experiment_5_transitions": {
            "seeds": TRAIN_VALIDATION_SEEDS, "pi_coefficients": BASIC_PI,
            "scenarios": transitions,
        },
        "experiment_6_shocks": {
            "seeds": TRAIN_VALIDATION_SEEDS, "pi_coefficients": BASIC_PI,
            "scenarios": shocks,
        },
    }


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for experiment_id, body in build_configs().items():
        config = {"experiment_id": experiment_id, **body}
        # 실행기와 같은 검증을 통과하는지 확인한다.
        ExperimentConfig.from_dict(config)
        runs = (
            len(body["scenarios"])
            * len(body["seeds"])
            * len(body["pi_coefficients"])
        )
        total += runs
        path = CONFIG_DIR / f"{experiment_id}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(config, file, ensure_ascii=False, indent=2)
            file.write("\n")
        print(
            f"{path.relative_to(PROJECT_ROOT)}: "
            f"{len(body['scenarios'])} scenarios * "
            f"{len(body['seeds'])} seeds * "
            f"{len(body['pi_coefficients'])} PI = {runs} runs"
        )
    print(f"total={total} runs")


if __name__ == "__main__":
    main()
