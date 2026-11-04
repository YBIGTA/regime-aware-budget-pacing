"""PI 계수만 바꿔 돌린 실험(예: PI 그리드)을 제어 결과 테이블로 정리한다.

환경 변수는 PI 계수와 무관하므로 넣지 않는다. 상태 테이블의 군집화용 행과
(scenario_id, seed, minute)로 합치면 레짐 라벨을 붙일 수 있다.

출력
  control_table.parquet  캠페인 x 시나리오 x 시드 x PI x 분. 키 열 + ctrl_* + split
  run_summary.csv        run별 TE, 예산 소진율, 승수 변동성
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.state_table import (  # noqa: E402
    _CTRL_SOURCE,
    _group_runs,
    _minute_environment,
    load_runs_from_dirs,
    same_environment,
    split_for_seed,
)


def _seeds(values):
    return tuple(int(value) for value in values)


def _control_table(trace, split):
    trace = trace.sort_values(["campaign_id", "minute"], kind="stable")
    table = pd.DataFrame(
        {
            "run_id": trace["run_id"].to_numpy(),
            "scenario_id": trace["scenario_id"].to_numpy(),
            "seed": trace["environment_seed"].to_numpy(),
            "kp": trace["kp"].to_numpy(),
            "ki": trace["ki"].to_numpy(),
            "campaign_id": trace["campaign_id"].to_numpy(),
            "minute": trace["minute"].to_numpy(),
        }
    )
    for target, source in _CTRL_SOURCE.items():
        table[target] = trace[source].to_numpy()
    table["ctrl_spend_error_ratio"] = (
        table["ctrl_spend_error"] / table["ctrl_daily_budget"]
    )
    table["split"] = split
    return table


def _summary(table):
    per_campaign = table.groupby("campaign_id").agg(
        te=("ctrl_spend_error_ratio", lambda s: s.abs().mean()),
        final_spend=("ctrl_actual_spend_cumulative", "last"),
        budget=("ctrl_daily_budget", "first"),
        signal_change=(
            "ctrl_pacing_signal_used",
            lambda s: s.diff().abs().mean(),
        ),
    )
    first = table.iloc[0]
    return {
        "run_id": first["run_id"],
        "scenario_id": first["scenario_id"],
        "seed": int(first["seed"]),
        "kp": float(first["kp"]),
        "ki": float(first["ki"]),
        "split": first["split"],
        "te": per_campaign["te"].mean(),
        "budget_spent_ratio": (
            per_campaign["final_spend"] / per_campaign["budget"]
        ).mean(),
        "signal_change_per_minute": per_campaign["signal_change"].mean(),
    }


def build_control_table(experiment_dirs, output_dir, splits):
    output_dir.mkdir(parents=True, exist_ok=True)
    runs, excluded = load_runs_from_dirs(experiment_dirs)
    if excluded:
        print(f"excluded runs: {len(excluded)}", flush=True)

    table_path = output_dir / "control_table.parquet"
    partial_path = output_dir / "control_table.parquet.partial"
    writer = None
    summaries = []
    try:
        for (scenario_id, seed), group in _group_runs(runs).items():
            split = split_for_seed(seed, splits)
            reference = None
            for run in group:
                trace = pd.read_csv(run["trace_path"])
                # PI 계수가 달라도 경매 환경이 같아야 레짐 라벨을 공유할 수 있다.
                environment = _minute_environment(trace)
                if reference is None:
                    reference = environment
                elif not same_environment(reference, environment):
                    raise ValueError(
                        "environment differs between PI runs of "
                        f"{scenario_id} seed {seed}"
                    )
                table = _control_table(trace, split)
                summaries.append(_summary(table))
                arrow_table = pa.Table.from_pandas(table, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(
                        partial_path, arrow_table.schema, compression="zstd"
                    )
                writer.write_table(arrow_table)
            print(f"{scenario_id} seed {seed} ({len(group)} runs)", flush=True)
    finally:
        if writer is not None:
            writer.close()
    partial_path.replace(table_path)

    summary = pd.DataFrame(summaries).sort_values(
        ["scenario_id", "seed", "kp", "ki"]
    )
    summary.to_csv(output_dir / "run_summary.csv", index=False)
    return len(summaries)


def main():
    parser = ArgumentParser(
        description="Build a control-only table from runs that differ in PI"
    )
    parser.add_argument("--experiment-dir", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-seeds", nargs="+", default=(42, 43, 44, 45))
    parser.add_argument("--validation-seeds", nargs="+", default=(46,))
    parser.add_argument("--test-seeds", nargs="*", default=())
    args = parser.parse_args()
    splits = {
        "train": _seeds(args.train_seeds),
        "validation": _seeds(args.validation_seeds),
        "test": _seeds(args.test_seeds),
    }
    count = build_control_table(args.experiment_dir, args.output_dir, splits)
    print(f"runs={count} output={args.output_dir}")


if __name__ == "__main__":
    main()
