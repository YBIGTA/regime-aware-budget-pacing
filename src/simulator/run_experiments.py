from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulator.experiment_config import load_experiment_config
from simulator.experiment_runner import run_experiment


def parse_args():
    parser = ArgumentParser(
        description="Run scenario x seed x PI coefficient experiments"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "simulation_runs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun and overwrite runs whose trace.csv already exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="number of runs to execute in parallel processes (default: 1)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    manifest = run_experiment(
        config=config,
        output_root=args.output_dir,
        overwrite=args.overwrite,
        workers=args.workers,
    )
    if any(
        row["status"] not in ("completed", "skipped") for row in manifest
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
