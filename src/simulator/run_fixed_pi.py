from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(1, str(PROJECT_ROOT / "external" / "yahoo-budget-pacing"))

from simulator.experiment_config import Scenario
from simulator.experiment_runner import run_simulation
import src.configuration as config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--campaigns", required=True, type=int)
    parser.add_argument("--minutes", required=True, type=int)
    parser.add_argument("--kp", required=True, type=float)
    parser.add_argument("--ki", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--seed", type=int, default=config.campaign_generation_seed_value
    )
    parser.add_argument("--daily-budget", type=float)
    parser.add_argument(
        "--dc", type=float, default=config.traffic_mean_cos_wave.dc
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=config.traffic_mean_cos_wave.amplitude,
    )
    parser.add_argument(
        "--phase", type=float, default=config.traffic_mean_cos_wave.phase
    )
    parser.add_argument(
        "--factor-untracked-bids",
        type=float,
        default=config.factor_untracked_bids,
    )
    parser.add_argument(
        "--untracked-bid-price-multiplier", type=float, default=1.0
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    scenario = Scenario.from_dict(
        {
            "id": args.scenario_id,
            "minutes": args.minutes,
            "campaigns": {
                "count": args.campaigns,
                "daily_budget": args.daily_budget,
            },
            "initial_condition": {
                "traffic": {
                    "dc": args.dc,
                    "amplitude": args.amplitude,
                    "phase": args.phase,
                },
                "competition": {
                    "untracked_bid_factor": args.factor_untracked_bids,
                    "untracked_bid_price_multiplier": (
                        args.untracked_bid_price_multiplier
                    ),
                },
            },
        }
    )
    result = run_simulation(
        scenario=scenario,
        seed=args.seed,
        kp=args.kp,
        ki=args.ki,
        trace_path=args.output,
        overwrite=args.overwrite,
    )
    print(f"trace_path={result['trace_path']}")
    print(f"metadata_path={result['metadata_path']}")
    print(f"rows={result['rows']}")
    print(f"validation_status={result['validation_status']}")
    if result["validation_status"] != "passed":
        print(f"validation_error={result['validation_error']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
