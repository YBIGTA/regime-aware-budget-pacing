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
        help="overwrite files for the same deterministic run ids",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_experiment_config(args.config)
    manifest = run_experiment(
        config=config,
        output_root=args.output_dir,
        overwrite=args.overwrite,
    )
    if any(row["status"] == "failed" for row in manifest):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
