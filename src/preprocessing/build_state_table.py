from argparse import ArgumentParser
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.state_table import (
    DEFAULT_SPLITS,
    build_state_table,
    variable_definitions_markdown,
)

DOCS_PATH = PROJECT_ROOT / "docs" / "state_table_variables.md"


def _seeds(values):
    return tuple(int(value) for value in values)


def parse_args():
    parser = ArgumentParser(
        description="Build the regime analysis state table from raw traces"
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        nargs="+",
        default=[PROJECT_ROOT / "outputs" / "v1" / "simulation_runs" / "experiment_1_1"],
        help="one or more experiment folders; identical runs are used once",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "v1" / "state_table" / "experiment_1_1",
    )
    parser.add_argument(
        "--train-seeds", nargs="+", default=DEFAULT_SPLITS["train"]
    )
    parser.add_argument(
        "--validation-seeds", nargs="+", default=DEFAULT_SPLITS["validation"]
    )
    parser.add_argument("--test-seeds", nargs="*", default=())
    parser.add_argument(
        "--write-docs",
        action="store_true",
        help=f"regenerate {DOCS_PATH.relative_to(PROJECT_ROOT)} and exit",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.write_docs:
        DOCS_PATH.write_text(variable_definitions_markdown(), encoding="utf-8")
        print(f"wrote {DOCS_PATH}")
        return

    splits = {
        "train": _seeds(args.train_seeds),
        "validation": _seeds(args.validation_seeds),
    }
    if args.test_seeds:
        splits["test"] = _seeds(args.test_seeds)
    assigned = [seed for seeds in splits.values() for seed in seeds]
    if len(assigned) != len(set(assigned)):
        raise SystemExit("a seed is assigned to more than one split")

    info = build_state_table(args.experiment_dir, args.output_dir, splits)
    print(f"output_dir={args.output_dir}")
    print(f"rows={info['rows']} runs_used={info['runs_used']}")
    if info["runs_excluded"]:
        print(f"runs_excluded={len(info['runs_excluded'])}")


if __name__ == "__main__":
    main()
