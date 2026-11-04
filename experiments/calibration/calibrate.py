"""예산과 PI 계수 보정 실험.

phase capacity : 페이싱 없이(Kp=Ki=0) 예산이 충분할 때 캠페인별 하루 지출을 잰다.
phase gains    : 주어진 예산과 PI 계수 후보로 지출 추종 지표를 비교한다.

사용 예:
  .venv/bin/python experiments/calibration/calibrate.py capacity --workers 7
  .venv/bin/python experiments/calibration/calibrate.py gains \
      --budget 2000 --gains 1:0.01 5:0.05 20:0.2 --workers 7
"""

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from simulator.experiment_config import load_experiment_config
from simulator.experiment_runner import run_simulation

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "v1" / "calibration"
CONFIG_PATH = PROJECT_ROOT / "configs" / "v1" / "experiment_1_1.json"


def _scenario(scenario_id, daily_budget):
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        data = json.load(config_file)
    data["experiment_id"] = "calibration"
    data["scenarios"] = [
        scenario for scenario in data["scenarios"]
        if scenario["id"] == scenario_id
    ]
    data["scenarios"][0]["campaigns"]["daily_budget"] = daily_budget
    path = OUTPUT_DIR / f"_config_{scenario_id}_{daily_budget}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return load_experiment_config(path).scenarios[0]


def _run(job):
    phase, scenario_id, budget, kp, ki, seed = job
    name = f"{phase}__{scenario_id}__b{budget}__kp{kp}__ki{ki}__s{seed}"
    trace_path = OUTPUT_DIR / "runs" / f"{name}.csv"
    if not trace_path.exists():
        run_simulation(
            scenario=_scenario(scenario_id, budget),
            seed=seed,
            kp=kp,
            ki=ki,
            trace_path=trace_path,
            overwrite=True,
        )
    return name, summarize(pd.read_csv(trace_path))


def summarize(trace):
    rows = []
    for campaign_id, group in trace.groupby("campaign_id"):
        group = group.sort_values("minute")
        budget = group["daily_budget"].iloc[0]
        spent = group["actual_spend_cumulative"]
        exhausted = group.loc[group["remaining_budget"] <= 1e-6 * budget, "minute"]
        signal = group["pacing_signal_used"]
        rows.append(
            {
                "campaign_id": campaign_id,
                "daily_spend": spent.iloc[-1],
                "spend_ratio": spent.iloc[-1] / budget,
                "exhaust_minute": exhausted.min() if len(exhausted) else np.nan,
                "te": (
                    (group["target_spend_cumulative"] - spent).abs() / budget
                ).mean(),
                "signal_mean": signal.mean(),
                "signal_min": signal.min(),
                "signal_step_std": signal.diff().std(),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = ArgumentParser()
    parser.add_argument("phase", choices=["capacity", "gains"])
    parser.add_argument("--scenarios", nargs="+", default=[
        "baseline_static",
        "traffic_low_static",
        "traffic_high_static",
        "competition_price_high_static",
    ])
    parser.add_argument("--budget", type=float, default=1e7)
    parser.add_argument("--gains", nargs="+", default=["0:0"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()

    gains = [tuple(float(v) for v in g.split(":")) for g in args.gains]
    if args.phase == "capacity":
        gains = [(0.0, 0.0)]
    jobs = [
        (args.phase, scenario_id, args.budget, kp, ki, args.seed)
        for scenario_id in args.scenarios
        for kp, ki in gains
    ]
    print(f"{len(jobs)} runs, {args.workers} workers", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, summary in pool.map(_run, jobs):
            print(f"done {name}", flush=True)
            summary.insert(0, "run", name)
            results.append(summary)

    table = pd.concat(results, ignore_index=True)
    out = OUTPUT_DIR / f"summary_{args.phase}.csv"
    if out.exists():
        table = pd.concat([pd.read_csv(out), table]).drop_duplicates(
            ["run", "campaign_id"], keep="last"
        )
    table.to_csv(out, index=False)
    pd.set_option("display.width", 200)
    print(
        table.groupby("run")[
            ["daily_spend", "spend_ratio", "exhaust_minute", "te",
             "signal_mean", "signal_min", "signal_step_std"]
        ].median().round(4).to_string()
    )


if __name__ == "__main__":
    main()
