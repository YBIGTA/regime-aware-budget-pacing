"""원본 trace.csv를 레짐 분석용 상태 테이블로 변환한다.

한 행은 캠페인 x 시나리오 x 시드 x PI 계수 x 분이다. 환경 변수는 시나리오 x
시드마다 한 번만 계산해 같은 환경의 모든 행에 붙인다. 모든 구간 변수는 해당
분까지의 값만 사용한다.

열 이름 규칙
  env_*   군집화 입력 후보인 환경 변수 (관측 가능한 경매 환경)
  cfg_*   시나리오 설정값 (시뮬레이터 입력, 실제 운영에서는 관측 불가)
  ctrl_*  제어 결과 (PI 계수에 따라 달라짐, 군집화에 쓰지 않음)
"""

from dataclasses import dataclass
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

WINDOWS = (15, 60)
DEFAULT_SPLITS = {
    "train": (42, 43, 44, 45),
    "validation": (46,),
}

KEY_COLUMNS = [
    "run_id",
    "scenario_id",
    "seed",
    "kp",
    "ki",
    "campaign_id",
    "minute",
]


@dataclass(frozen=True)
class Column:
    name: str
    group: str
    meaning: str
    unit: str
    method: str
    cluster_use: str


def _window_columns():
    columns = []
    for window in WINDOWS:
        w = f"{window}m"
        span = f"최근 {window}분(해당 분 포함, 0분 이전은 없음)"
        columns += [
            Column(
                f"env_auctions_mean_{w}", "env",
                f"{span} 분당 경매 수 평균", "건/분",
                "actual_auctions의 구간 평균", "예",
            ),
            Column(
                f"env_auctions_trend_{w}", "env",
                f"{span} 경매 수 증감률", "비율",
                "구간 내 actual_auctions의 최소제곱 기울기 x (구간 길이-1) / "
                "구간 평균. 구간 시작부터 끝까지의 상대 변화량. 값이 1개면 0",
                "예",
            ),
            Column(
                f"env_auctions_cv_{w}", "env",
                f"{span} 경매 수 변동성", "비율",
                "구간 actual_auctions의 표준편차(ddof=0) / 구간 평균", "예",
            ),
            Column(
                f"env_comp_bids_per_auction_{w}", "env",
                f"{span} 경매 1건당 경쟁 입찰 수", "건/경매",
                "구간 competitor_bid_count_total 합 / 구간 actual_auctions 합",
                "예",
            ),
            Column(
                f"env_comp_price_mean_{w}", "env",
                f"{span} 경쟁 입찰가 평균", "통화 단위",
                "구간 전체 경쟁 입찰의 평균. 분별 평균을 입찰 수로 가중 평균",
                "예",
            ),
            Column(
                f"env_comp_price_std_{w}", "env",
                f"{span} 경쟁 입찰가 표준편차", "통화 단위",
                "분별 평균·표준편차·입찰 수로 계산한 구간 전체의 합동 표준편차"
                "(ddof=0)",
                "예",
            ),
            Column(
                f"env_comp_price_max_{w}", "env",
                f"{span} 경쟁 입찰가 최댓값", "통화 단위",
                "competitor_bid_price_max의 구간 최댓값", "예",
            ),
            Column(
                f"window_filled_{w}", "flag",
                f"최근 {window}분이 모두 찼는지 여부", "bool",
                f"minute >= {window - 1}이면 True. False인 행의 {w} 변수는 "
                "있는 분만으로 계산한 값",
                "아니오 (필터용)",
            ),
        ]
    return columns


COLUMNS = [
    Column("run_id", "key", "실행 ID", "-",
           "`<scenario>__seed<seed>__kp<kp>__ki<ki>`", "아니오"),
    Column("scenario_id", "key", "시나리오 ID", "-",
           "`configs/v1/experiment_1_1.json`의 `scenarios[].id`", "아니오"),
    Column("seed", "key", "환경 난수 시드", "-",
           "같은 시나리오·시드면 PI 계수와 무관하게 경매 입력이 같다", "아니오"),
    Column("kp", "key", "PI 비례 계수", "-", "실행 설정값", "아니오"),
    Column("ki", "key", "PI 적분 계수", "-", "실행 설정값", "아니오"),
    Column("campaign_id", "key", "캠페인 ID", "-", "campaign_0 ~", "아니오"),
    Column("minute", "key", "하루 중 분", "분", "0 ~ 1439", "아니오"),
    Column("split", "meta", "데이터 분할", "-",
           "시드로 결정. 같은 시나리오·시드는 PI와 무관하게 같은 분할",
           "아니오"),
    Column("cluster_row", "meta", "군집화용 대표 행 여부", "bool",
           "시나리오 x 시드 x 분마다 한 행(가장 작은 kp·ki, 첫 캠페인)만 True",
           "아니오 (필터용)"),
    Column("minute_of_day", "env", "하루 중 시각", "분",
           "minute과 같은 값 (0 ~ 1439)",
           "선택 (시간대로만 군집이 나뉠 수 있음, docs/state_table.md 참고)"),
    Column("env_auctions", "env", "해당 분 경매 수", "건/분",
           "actual_auctions 원값", "선택 (1분 값, 잡음이 큼)"),
    Column("env_comp_bids_per_auction", "env", "해당 분 경매 1건당 경쟁 입찰 수",
           "건/경매", "competitor_bid_count_mean 원값",
           "선택 (1분 값, 잡음이 큼)"),
    Column("env_comp_price_mean", "env", "해당 분 경쟁 입찰가 평균", "통화 단위",
           "competitor_bid_price_mean 원값", "선택 (1분 값, 잡음이 큼)"),
    *_window_columns(),
    Column("cfg_traffic_dc", "cfg", "설정된 평균 트래픽 규모", "건/분",
           "현재 적용 중인 조건의 traffic.dc",
           "금지 (시뮬레이터 정답값, 해석·검증용)"),
    Column("cfg_traffic_amplitude", "cfg", "설정된 시간대별 변동폭", "비율",
           "traffic.amplitude", "금지 (해석·검증용)"),
    Column("cfg_untracked_bid_factor", "cfg", "설정된 경쟁 입찰 수 계수", "-",
           "competition.untracked_bid_factor", "금지 (해석·검증용)"),
    Column("cfg_untracked_bid_price_multiplier", "cfg",
           "설정된 경쟁 입찰가 배율", "배",
           "competition.untracked_bid_price_multiplier", "금지 (해석·검증용)"),
    Column("cfg_condition_start_minute", "cfg", "현재 조건이 시작된 분", "분",
           "전환 시나리오에서 720이면 정오 이후 조건", "금지 (해석·검증용)"),
    Column("cfg_traffic_mean", "cfg", "설정상 기대 경매 수", "건/분",
           "코사인 트래픽 곡선의 해당 분 기댓값", "금지 (해석·검증용)"),
    Column("ctrl_daily_budget", "ctrl", "캠페인 일 예산", "통화 단위",
           "daily_budget", "아니오 (제어 결과 분석용)"),
    Column("ctrl_target_spend_cumulative", "ctrl", "목표 누적 지출", "통화 단위",
           "일 예산 x (minute+1) / 1440", "아니오"),
    Column("ctrl_actual_spend_cumulative", "ctrl", "실제 누적 지출",
           "통화 단위", "해당 분 종료 시점 누적", "아니오"),
    Column("ctrl_spend_error", "ctrl", "지출 오차", "통화 단위",
           "목표 누적 지출 - 실제 누적 지출 (양수면 덜 씀)", "아니오"),
    Column("ctrl_spend_error_ratio", "ctrl", "예산 대비 지출 오차", "비율",
           "ctrl_spend_error / 일 예산. TE는 이 값의 절댓값을 하루 평균",
           "아니오"),
    Column("ctrl_spend_this_minute", "ctrl", "해당 분 지출", "통화 단위",
           "spend_this_minute", "아니오"),
    Column("ctrl_wins_this_minute", "ctrl", "해당 분 낙찰 수", "건",
           "wins_this_minute", "아니오"),
    Column("ctrl_remaining_budget", "ctrl", "남은 예산", "통화 단위",
           "일 예산 - 실제 누적 지출", "아니오"),
    Column("ctrl_pacing_signal_used", "ctrl", "해당 분에 사용한 입찰 승수", "배",
           "원래 입찰가에 곱한 값", "아니오"),
    Column("ctrl_pacing_signal_next", "ctrl", "다음 분에 사용할 입찰 승수", "배",
           "해당 분 종료 후 PI가 계산한 값", "아니오"),
    Column("ctrl_integral_error", "ctrl", "PI 적분 상태", "비율 x 분",
           "누적한 예산 대비 오차 (승수가 상·하한에 걸린 분은 누적하지 않음)",
           "아니오"),
]

ENV_FEATURES = [
    column.name for column in COLUMNS
    if column.group == "env" and column.name != "minute_of_day"
]

_CFG_SOURCE = {
    "cfg_traffic_dc": "traffic_dc",
    "cfg_traffic_amplitude": "traffic_amplitude",
    "cfg_untracked_bid_factor": "untracked_bid_factor",
    "cfg_untracked_bid_price_multiplier": "untracked_bid_price_multiplier",
    "cfg_condition_start_minute": "condition_start_minute",
    "cfg_traffic_mean": "traffic_mean",
}
_CTRL_SOURCE = {
    "ctrl_daily_budget": "daily_budget",
    "ctrl_target_spend_cumulative": "target_spend_cumulative",
    "ctrl_actual_spend_cumulative": "actual_spend_cumulative",
    "ctrl_spend_error": "spend_error",
    "ctrl_spend_this_minute": "spend_this_minute",
    "ctrl_wins_this_minute": "wins_this_minute",
    "ctrl_remaining_budget": "remaining_budget",
    "ctrl_pacing_signal_used": "pacing_signal_used",
    "ctrl_pacing_signal_next": "pacing_signal_next",
    "ctrl_integral_error": "integral_error",
}
_ENVIRONMENT_TRACE_FIELDS = [
    "actual_auctions",
    "competitor_bid_count_total",
    "competitor_bid_count_mean",
    "competitor_bid_price_mean",
    "competitor_bid_price_std",
    "competitor_bid_price_max",
    *_CFG_SOURCE.values(),
]


def _rolling_sum(values, window):
    cumulative = np.concatenate([[0.0], np.cumsum(values, dtype=float)])
    end = np.arange(1, len(values) + 1)
    start = np.maximum(end - window, 0)
    return cumulative[end] - cumulative[start]


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    out = np.zeros_like(numerator)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def environment_features(minute_frame):
    """분 단위 환경 기록(0~마지막 분, 분당 1행)으로 환경 변수를 계산한다."""
    frame = minute_frame.sort_values("minute").reset_index(drop=True)
    minutes = frame["minute"].to_numpy()
    if not np.array_equal(minutes, np.arange(len(frame))):
        raise ValueError("minutes must be consecutive from 0")

    auctions = frame["actual_auctions"].to_numpy(dtype=float)
    bid_count = frame["competitor_bid_count_total"].to_numpy(dtype=float)
    price_mean = frame["competitor_bid_price_mean"].to_numpy(dtype=float)
    price_std = frame["competitor_bid_price_std"].to_numpy(dtype=float)
    price_max = frame["competitor_bid_price_max"]

    out = pd.DataFrame({"minute": minutes, "minute_of_day": minutes})
    out["env_auctions"] = auctions
    out["env_comp_bids_per_auction"] = frame["competitor_bid_count_mean"]
    out["env_comp_price_mean"] = price_mean

    x = minutes.astype(float)
    for window in WINDOWS:
        w = f"{window}m"
        n = _rolling_sum(np.ones_like(auctions), window)
        sum_y = _rolling_sum(auctions, window)
        mean = sum_y / n
        variance = np.maximum(
            _rolling_sum(auctions**2, window) / n - mean**2, 0.0
        )
        sum_x = _rolling_sum(x, window)
        denominator = n * _rolling_sum(x**2, window) - sum_x**2
        slope = _safe_divide(
            n * _rolling_sum(x * auctions, window) - sum_x * sum_y,
            denominator,
        )

        bids = _rolling_sum(bid_count, window)
        price_sum = _rolling_sum(price_mean * bid_count, window)
        price_square_sum = _rolling_sum(
            (price_std**2 + price_mean**2) * bid_count, window
        )
        pooled_mean = _safe_divide(price_sum, bids)
        pooled_variance = np.maximum(
            _safe_divide(price_square_sum, bids) - pooled_mean**2, 0.0
        )

        out[f"env_auctions_mean_{w}"] = mean
        out[f"env_auctions_trend_{w}"] = _safe_divide(slope * (n - 1), mean)
        out[f"env_auctions_cv_{w}"] = _safe_divide(np.sqrt(variance), mean)
        out[f"env_comp_bids_per_auction_{w}"] = _safe_divide(bids, sum_y)
        out[f"env_comp_price_mean_{w}"] = pooled_mean
        out[f"env_comp_price_std_{w}"] = np.sqrt(pooled_variance)
        out[f"env_comp_price_max_{w}"] = (
            price_max.rolling(window, min_periods=1).max().to_numpy()
        )
        out[f"window_filled_{w}"] = minutes >= window - 1

    for target, source in _CFG_SOURCE.items():
        out[target] = frame[source].to_numpy()
    return out


def _minute_environment(trace):
    environment = trace.groupby("minute", sort=True)[
        _ENVIRONMENT_TRACE_FIELDS
    ]
    if (environment.nunique() > 1).any().any():
        raise ValueError("environment values differ between campaigns")
    return environment.first().reset_index()


def split_for_seed(seed, splits):
    for name, seeds in splits.items():
        if seed in seeds:
            return name
    raise ValueError(
        f"seed {seed} is not assigned to any split; pass it explicitly"
    )


def load_runs(experiment_dir):
    """manifest.csv에서 검증을 통과한 실행만 고른다."""
    experiment_dir = Path(experiment_dir)
    with (experiment_dir / "manifest.csv").open(encoding="utf-8") as file:
        manifest = list(csv.DictReader(file))
    runs = []
    skipped = []
    for row in manifest:
        trace_path = experiment_dir / "runs" / row["run_id"] / "trace.csv"
        if (
            row["status"] in ("completed", "skipped")
            and row["validation_status"] == "passed"
            and trace_path.exists()
        ):
            runs.append(
                {
                    "run_id": row["run_id"],
                    "scenario_id": row["scenario_id"],
                    "seed": int(row["environment_seed"]),
                    "kp": float(row["kp"]),
                    "ki": float(row["ki"]),
                    "trace_path": trace_path,
                }
            )
        else:
            skipped.append(row["run_id"])
    return runs, skipped


def _group_runs(runs):
    groups = {}
    for run in runs:
        groups.setdefault((run["scenario_id"], run["seed"]), []).append(run)
    for group in groups.values():
        group.sort(key=lambda run: (run["kp"], run["ki"]))
    return dict(sorted(groups.items()))


def same_environment(a, b):
    """기기마다 다른 부동소수점 오차(1e-15 수준)는 같은 값으로 본다."""
    if list(a.columns) != list(b.columns) or len(a) != len(b):
        return False
    return bool(
        np.allclose(a.to_numpy(float), b.to_numpy(float), rtol=1e-9, atol=1e-12)
    )


def _environment_for_group(group):
    """같은 시나리오·시드의 모든 PI 실행에서 환경이 같은지 확인하고 반환한다."""
    reference = None
    for run in group:
        trace = pd.read_csv(run["trace_path"])
        environment = _minute_environment(trace)
        if reference is None:
            reference = environment
        elif not same_environment(reference, environment):
            raise ValueError(
                "environment differs between PI runs of "
                f"{run['scenario_id']} seed {run['seed']}"
            )
    return environment_features(reference)


def normalization_stats(environments):
    """학습 분할의 대표 행(시나리오 x 시드 x 분당 1행)으로 평균·표준편차를 구한다.

    구간 변수는 그 구간이 다 찬 행만 사용한다.
    """
    train = pd.concat(
        [env for env in environments if env["split"].iloc[0] == "train"],
        ignore_index=True,
    )
    if train.empty:
        raise ValueError("no train rows for normalization")
    stats = {}
    for feature in ENV_FEATURES:
        rows = train
        for window in WINDOWS:
            if feature.endswith(f"_{window}m"):
                rows = train[train[f"window_filled_{window}m"]]
        std = float(rows[feature].std(ddof=0))
        stats[feature] = {
            "mean": float(rows[feature].mean()),
            "std": std if std > 0 else 1.0,
            "rows": int(len(rows)),
        }
    return stats


def _control_rows(run, environment, stats):
    trace = pd.read_csv(run["trace_path"])
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

    environment = environment.copy()
    for feature in ENV_FEATURES:
        environment[f"{feature}_z"] = (
            environment[feature] - stats[feature]["mean"]
        ) / stats[feature]["std"]
    table = table.merge(environment, on="minute", how="left", validate="m:1")
    return table


def _ordered_columns(table):
    names = [column.name for column in COLUMNS]
    names += [f"{feature}_z" for feature in ENV_FEATURES]
    return [name for name in names if name in table.columns]


def load_runs_from_dirs(experiment_dirs):
    """여러 실험 폴더의 실행을 합친다.

    같은 run_id(시나리오·시드·PI가 같은 실행)가 여러 폴더에 있으면 결과가
    같으므로 앞 폴더의 것 하나만 쓴다.
    """
    runs = []
    excluded = []
    seen = set()
    for experiment_dir in experiment_dirs:
        dir_runs, dir_excluded = load_runs(experiment_dir)
        excluded.extend(dir_excluded)
        for run in dir_runs:
            if run["run_id"] in seen:
                continue
            seen.add(run["run_id"])
            runs.append(run)
    return runs, excluded


def build_state_table(experiment_dirs, output_dir, splits=None):
    """experiment_dirs는 실험 폴더 하나 또는 여러 개의 목록이다."""
    splits = splits or DEFAULT_SPLITS
    if isinstance(experiment_dirs, (str, Path)):
        experiment_dirs = [experiment_dirs]
    experiment_dirs = [Path(path) for path in experiment_dirs]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, excluded = load_runs_from_dirs(experiment_dirs)
    if not runs:
        raise ValueError(
            f"no usable runs in {', '.join(map(str, experiment_dirs))}"
        )
    groups = _group_runs(runs)

    environments = {}
    for (scenario_id, seed), group in groups.items():
        print(f"environment {scenario_id} seed {seed}", flush=True)
        environment = _environment_for_group(group)
        environment["split"] = split_for_seed(seed, splits)
        environments[(scenario_id, seed)] = environment
    stats = normalization_stats(list(environments.values()))

    table_path = output_dir / "state_table.parquet"
    partial_path = output_dir / "state_table.parquet.partial"
    writer = None
    row_count = 0
    try:
        for (scenario_id, seed), group in groups.items():
            for index, run in enumerate(group):
                table = _control_rows(
                    run, environments[(scenario_id, seed)], stats
                )
                first_campaign = table["campaign_id"].min()
                table["cluster_row"] = (index == 0) & (
                    table["campaign_id"] == first_campaign
                )
                table = table[_ordered_columns(table)]
                arrow_table = pa.Table.from_pandas(table, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(partial_path, arrow_table.schema)
                writer.write_table(arrow_table)
                row_count += len(table)
                print(f"rows {run['run_id']} ({len(table)})", flush=True)
    finally:
        if writer is not None:
            writer.close()
    partial_path.replace(table_path)

    with (output_dir / "normalization.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "method": "z-score: (x - mean) / std, train split cluster rows",
                "features": stats,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
        f.write("\n")

    split_rows = sorted(
        {
            (scenario_id, seed, environment["split"].iloc[0])
            for (scenario_id, seed), environment in environments.items()
        }
    )
    with (output_dir / "splits.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow(["scenario_id", "seed", "split"])
        writer_csv.writerows(split_rows)

    info = {
        "experiment_dirs": [str(path) for path in experiment_dirs],
        "rows": row_count,
        "runs_used": len(runs),
        "runs_excluded": excluded,
        "environments": len(environments),
        "splits": {name: list(seeds) for name, seeds in splits.items()},
        "windows_minutes": list(WINDOWS),
    }
    with (output_dir / "build_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return info


def variable_definitions_markdown():
    groups = {
        "key": "식별 열",
        "meta": "분할·필터 열",
        "flag": "구간 채움 표시",
        "env": "환경 변수 (군집화 입력 후보)",
        "cfg": "시나리오 설정값 (군집화 금지)",
        "ctrl": "제어 결과 (군집화에 쓰지 않음)",
    }
    lines = [
        "# 상태 테이블 변수 정의서",
        "",
        "이 파일은 `src/preprocessing/state_table.py`의 `COLUMNS`에서 자동으로",
        "생성된다. 직접 고치지 말고 코드를 고친 뒤 다시 생성한다.",
        "",
        "```bash",
        ".venv/bin/python src/preprocessing/build_state_table.py --write-docs",
        "```",
        "",
        "쓰는 방법과 담당자가 정할 항목은 [state_table.md](state_table.md)에 있다.",
        "",
    ]
    for group, title in groups.items():
        lines += [
            f"## {title}",
            "",
            "| 열 | 의미 | 단위 | 계산 방법 | 군집화 사용 |",
            "| --- | --- | --- | --- | --- |",
        ]
        for column in COLUMNS:
            if column.group == group:
                lines.append(
                    f"| `{column.name}` | {column.meaning} | {column.unit} | "
                    f"{column.method} | {column.cluster_use} |"
                )
        lines.append("")
    lines += [
        "## 정규화 열",
        "",
        "환경 변수마다 `<변수>_z` 열이 있다. 학습 분할의 `cluster_row` 행으로",
        "계산한 평균·표준편차로 정규화한 값이다. 구간 변수의 기준은 그 구간이 다",
        "찬 행(`window_filled_*`가 True)으로만 계산한다. 검증·평가 분할에도 같은",
        "기준을 적용한다. 기준값은 `normalization.json`에 있다.",
        "",
        "| 대상 | 군집화 사용 |",
        "| --- | --- |",
        "| `env_*_z` | 원 변수와 같음 |",
        "",
    ]
    return "\n".join(lines)
