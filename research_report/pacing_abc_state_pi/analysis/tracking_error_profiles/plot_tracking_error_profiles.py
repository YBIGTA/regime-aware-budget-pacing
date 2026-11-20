import ast
import csv
import gzip
import io
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / "research_report/pacing_abc_state_pi/analysis/day_100063/analyze.py"
STORED_DAILY = REPO / "research_report/pacing_abc_state_pi/results/daily_scores.csv"
CAP12_DAILY = REPO / "research_report/pacing_abc_state_pi/analysis/unspent_ecdf/cap_1_2_daily.csv"
TRACE_PATH = HERE / "tracking_error_5min.csv.gz"
DAILY_CHECK_PATH = HERE / "daily_replay_check.csv"
PROFILE_PATH = HERE / "time_profile_summary.csv"
SUMMARY_PATH = HERE / "summary.json"
VALIDATION_DAYS = list(range(100000, 100200))
METRIC_ATOL = 1e-9

CONDITIONS = (
    "B_fixed_PI",
    "B_state_PI_cap_1_0",
    "B_state_PI_cap_1_2",
)
COMPARISONS = (
    {
        "key": "state_pi",
        "title": "상태별 PI 적용",
        "control": "B_fixed_PI",
        "experiment": "B_state_PI_cap_1_0",
        "control_label": "B · 고정 PI",
        "experiment_label": "B_state_PI · 상태별 PI",
        "stem": "state-pi-tracking-profile",
    },
    {
        "key": "cap",
        "title": "입찰 승수 상한 완화",
        "control": "B_state_PI_cap_1_0",
        "experiment": "B_state_PI_cap_1_2",
        "control_label": "B_state_PI · 상한 1.0",
        "experiment_label": "B_state_PI · 상한 1.2",
        "stem": "cap-tracking-profile",
    },
)


def load_analysis_definitions():
    tree = ast.parse(SOURCE.read_text(), filename=str(SOURCE))
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Import) and any(
            alias.name.startswith("matplotlib") for alias in node.names
        ):
            continue
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith(
            "matplotlib"
        ):
            continue
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "state"
            for target in node.targets
        ):
            break
        nodes.append(node)

    namespace = {"__file__": str(SOURCE), "__name__": "analysis_library"}
    module = ast.Module(body=nodes, type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace


ns = load_analysis_definitions()
sys.path.insert(0, str(REPO / "experiments/pacing_abc"))
import core  # noqa: E402

cfg = ns["cfg"]
b_frozen = ns["b_frozen"]


class BController:
    def __init__(self, gains, schedule=None, upper=1.0):
        self.gains = tuple(gains)
        self.schedule = schedule
        self.upper = upper

    def reset(self):
        self.integral = 0.0

    def act(self, obs):
        base, slope = ns["base_and_slope"](obs)
        if self.schedule is None:
            kp, ki = self.gains
        else:
            kp, ki = self.schedule(ns["state_features"](obs, base, slope))

        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw = base + kp * error + candidate
        lo, hi = cfg.multiplier_min, self.upper
        blocked = (raw > hi and error > 0) or (raw < lo and error < 0)
        if blocked:
            raw = base + kp * error + self.integral
        else:
            self.integral = candidate
        return float(np.clip(raw, lo, hi))


def make_controller(condition):
    if condition == "B_fixed_PI":
        return BController(b_frozen["B_fixed_PI"])

    state = b_frozen["B_state_PI"]
    schedule = ns["GainMLP"](state["anchor"], state["weights"])
    upper = 1.2 if condition == "B_state_PI_cap_1_2" else 1.0
    return BController(state["anchor"], schedule=schedule, upper=upper)


def rollout(episode, condition):
    controller = make_controller(condition)
    controller.reset()
    spent = 0.0
    history = []
    rows = []
    zero = core.Feedback(-1, 0.0, 0.0, 0, 0, 0.0, False)

    for tick_no, tick in enumerate(episode.tape):
        padded = ([zero] * cfg.history + history)[-cfg.history :]
        obs = core.Observation(
            tick_no, cfg.ticks, episode.budget, spent, tuple(padded)
        )
        action = controller.act(obs)
        cost, wins, censored = core.execute(
            tick, action, max(0.0, episode.budget - spent)
        )
        spent += cost
        history.append(
            core.Feedback(
                tick_no,
                action,
                cost,
                len(tick.base),
                wins,
                float(tick.base.mean()) if len(tick.base) else 0.0,
                censored,
            )
        )
        target = (tick_no + 1) * episode.budget / cfg.ticks
        error_ratio = (tick_no + 1) / cfg.ticks - spent / episode.budget
        rows.append(
            {
                "tick": tick_no,
                "minute": (tick_no + 1) * cfg.interval,
                "budget": episode.budget,
                "target_cumulative": target,
                "actual_cumulative": spent,
                "abs_tracking_error_percent": abs(error_ratio) * 100,
            }
        )

    errors = np.asarray([row["abs_tracking_error_percent"] for row in rows])
    return {
        "tracking_error": float(errors.mean() / 100),
        "unspent": max(0.0, 1 - spent / episode.budget),
    }, rows


def read_daily_references():
    references = {condition: {} for condition in CONDITIONS}
    with STORED_DAILY.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["model"] == "B_fixed_PI":
                condition = "B_fixed_PI"
            elif row["model"] == "B_state_PI":
                condition = "B_state_PI_cap_1_0"
            else:
                continue
            references[condition][int(row["day"])] = {
                "tape_sha256": row["tape_sha256"],
                "tracking_error": float(row["tracking_error"]),
                "unspent": float(row["unspent"]),
            }

    if not CAP12_DAILY.exists():
        raise SystemExit(
            "Missing cap 1.2 daily reference: "
            f"{CAP12_DAILY.relative_to(REPO)}"
        )
    with CAP12_DAILY.open(newline="") as file:
        for row in csv.DictReader(file):
            references["B_state_PI_cap_1_2"][int(row["day"])] = {
                "tape_sha256": row["tape_sha256"],
                "tracking_error": float(row["tracking_error"]),
                "unspent": float(row["unspent"]),
            }

    for condition, rows in references.items():
        if sorted(rows) != VALIDATION_DAYS:
            raise SystemExit(
                f"{condition}: expected exactly validation days 100000~100199"
            )
    for day in VALIDATION_DAYS:
        hashes = {references[condition][day]["tape_sha256"] for condition in CONDITIONS}
        if len(hashes) != 1:
            raise SystemExit(f"Reference tape hash mismatch on day {day}")
    return references


def open_deterministic_gzip_text(path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def replay_all(references):
    errors = {
        condition: np.empty((len(VALIDATION_DAYS), cfg.ticks), dtype=float)
        for condition in CONDITIONS
    }
    daily_checks = []
    trace_fields = [
        "condition",
        "day",
        "tape_sha256",
        "tick",
        "minute",
        "budget",
        "target_cumulative",
        "actual_cumulative",
        "abs_tracking_error_percent",
    ]
    started = time.perf_counter()
    raw, compressed, text = open_deterministic_gzip_text(TRACE_PATH)
    try:
        writer = csv.DictWriter(text, fieldnames=trace_fields, lineterminator="\n")
        writer.writeheader()
        for day_index, day in enumerate(VALIDATION_DAYS):
            episode = ns["generate"](cfg, day)
            tape_sha256 = episode.digest()
            for condition in CONDITIONS:
                reference = references[condition][day]
                if tape_sha256 != reference["tape_sha256"]:
                    raise SystemExit(f"Replay tape mismatch: {condition}, day {day}")
                score, rows = rollout(episode, condition)
                errors[condition][day_index] = [
                    row["abs_tracking_error_percent"] for row in rows
                ]
                tracking_difference = score["tracking_error"] - reference["tracking_error"]
                unspent_difference = score["unspent"] - reference["unspent"]
                if abs(tracking_difference) > METRIC_ATOL or abs(unspent_difference) > METRIC_ATOL:
                    raise SystemExit(
                        f"Stored metric mismatch: {condition}, day {day}, "
                        f"tracking={tracking_difference}, unspent={unspent_difference}"
                    )
                daily_checks.append(
                    {
                        "condition": condition,
                        "day": day,
                        "tape_sha256": tape_sha256,
                        "replay_tracking_error": score["tracking_error"],
                        "stored_tracking_error": reference["tracking_error"],
                        "tracking_difference": tracking_difference,
                        "replay_unspent": score["unspent"],
                        "stored_unspent": reference["unspent"],
                        "unspent_difference": unspent_difference,
                    }
                )
                for row in rows:
                    writer.writerow(
                        {
                            "condition": condition,
                            "day": day,
                            "tape_sha256": tape_sha256,
                            **row,
                        }
                    )
    finally:
        text.close()
        compressed.close()
        raw.close()

    with DAILY_CHECK_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(daily_checks[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(daily_checks)
    return errors, daily_checks, time.perf_counter() - started


def read_existing_errors(references):
    errors = {
        condition: np.full((len(VALIDATION_DAYS), cfg.ticks), np.nan, dtype=float)
        for condition in CONDITIONS
    }
    seen = {
        condition: np.zeros((len(VALIDATION_DAYS), cfg.ticks), dtype=bool)
        for condition in CONDITIONS
    }
    day_indexes = {day: index for index, day in enumerate(VALIDATION_DAYS)}

    with gzip.open(TRACE_PATH, "rt", newline="") as file:
        for row in csv.DictReader(file):
            condition = row["condition"]
            day = int(row["day"])
            tick = int(row["tick"])
            if condition not in errors or day not in day_indexes:
                raise SystemExit(f"Unexpected trace row: {condition}, day {day}")
            if not 0 <= tick < cfg.ticks:
                raise SystemExit(f"Unexpected tick: {condition}, day {day}, tick {tick}")
            day_index = day_indexes[day]
            if seen[condition][day_index, tick]:
                raise SystemExit(f"Duplicate trace row: {condition}, day {day}, tick {tick}")
            if row["tape_sha256"] != references[condition][day]["tape_sha256"]:
                raise SystemExit(f"Trace tape mismatch: {condition}, day {day}")
            expected_minute = (tick + 1) * cfg.interval
            if int(row["minute"]) != expected_minute:
                raise SystemExit(
                    f"Trace time mismatch: {condition}, day {day}, tick {tick}"
                )
            errors[condition][day_index, tick] = float(
                row["abs_tracking_error_percent"]
            )
            seen[condition][day_index, tick] = True

    for condition in CONDITIONS:
        if not seen[condition].all():
            missing = int(seen[condition].size - seen[condition].sum())
            raise SystemExit(f"{condition}: missing {missing} five-minute trace rows")
    return errors


def read_daily_checks():
    with DAILY_CHECK_PATH.open(newline="") as file:
        rows = list(csv.DictReader(file))
    expected = len(CONDITIONS) * len(VALIDATION_DAYS)
    if len(rows) != expected:
        raise SystemExit(f"Expected {expected} daily replay checks, found {len(rows)}")
    return rows


def summarize_profiles(errors):
    profiles = {}
    rows = []
    for comparison in COMPARISONS:
        control = errors[comparison["control"]]
        experiment = errors[comparison["experiment"]]
        reduction = control - experiment
        control_mean = control.mean(axis=0)
        experiment_mean = experiment.mean(axis=0)
        reduction_mean = reduction.mean(axis=0)
        identity_error = np.max(
            np.abs(reduction_mean - (control_mean - experiment_mean))
        )
        if identity_error > 1e-12:
            raise SystemExit(
                f"Mean difference identity failed for {comparison['key']}: "
                f"{identity_error}"
            )
        profile = {
            "control_mean": control_mean,
            "experiment_mean": experiment_mean,
            "reduction_mean": reduction_mean,
        }
        profiles[comparison["key"]] = profile
        for tick in range(cfg.ticks):
            rows.append(
                {
                    "comparison": comparison["key"],
                    "tick": tick,
                    "minute": (tick + 1) * cfg.interval,
                    **{key: values[tick] for key, values in profile.items()},
                }
            )

    with PROFILE_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return profiles


def clean_svg(path):
    lines = path.read_text().splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n")


def draw(comparison, profile):
    control_color = "#365F91"
    experiment_color = "#C65D3A"
    reduction_color = "#397A5A"
    neutral_color = "#4B5563"
    hours = np.arange(1, cfg.ticks + 1) * cfg.interval / 60

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12},
    )
    top, bottom = axes
    top.plot(
        hours,
        profile["control_mean"],
        color=control_color,
        linewidth=2.2,
        linestyle="-",
        label=comparison["control_label"],
    )
    top.plot(
        hours,
        profile["experiment_mean"],
        color=experiment_color,
        linewidth=2.2,
        linestyle="--",
        label=comparison["experiment_label"],
    )
    top.set_ylabel("평균 절대 추종 오차 (%)")
    top.set_ylim(bottom=0)
    top.legend(loc="upper left", frameon=False, ncol=2)

    reduction = profile["reduction_mean"]
    bottom.plot(
        hours,
        reduction,
        color=reduction_color,
        linewidth=2.1,
    )
    bottom.axhline(0, color=neutral_color, linewidth=1.1)
    limit = max(float(np.max(np.abs(reduction))) * 1.1, 1e-6)
    bottom.set_ylim(-limit, limit)
    bottom.set_ylabel("평균 추종 오차 차이 (%p)")
    bottom.set_xlabel("시각")
    bottom.text(
        0.99,
        0.04,
        "양수 = 실험군 개선  ·  음수 = 실험군 악화",
        transform=bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=neutral_color,
    )
    for ax in axes:
        ax.set_xlim(0, 24)
        ax.set_xticks(np.arange(0, 25, 4))
        ax.grid(color="#D1D5DB", linewidth=0.7, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
    bottom.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 4)])

    fig.suptitle(comparison["title"], fontsize=17, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.925,
        "동일한 검증 200일 · 5분 단위 · 같은 날짜·시각끼리 짝비교",
        ha="center",
        fontsize=10.5,
        color=neutral_color,
    )
    fig.subplots_adjust(left=0.105, right=0.98, top=0.88, bottom=0.10)
    png = HERE / f"{comparison['stem']}.png"
    svg = HERE / f"{comparison['stem']}.svg"
    fig.savefig(png, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(svg, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    clean_svg(svg)


def main():
    references = read_daily_references()
    if "--replay" in sys.argv[1:] or not TRACE_PATH.exists():
        errors, daily_checks, runtime_seconds = replay_all(references)
        data_source_mode = "replayed_5_minute_trace"
    else:
        errors = read_existing_errors(references)
        daily_checks = read_daily_checks()
        runtime_seconds = None
        data_source_mode = "existing_5_minute_trace"
    profiles = summarize_profiles(errors)

    plt.rcParams.update(
        {
            "font.family": ["Apple SD Gothic Neo", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
    for comparison in COMPARISONS:
        draw(comparison, profiles[comparison["key"]])

    all_days = {
        condition: sorted(
            int(row["day"])
            for row in daily_checks
            if row["condition"] == condition
        )
        for condition in CONDITIONS
    }
    maximum_tracking_difference = max(
        abs(float(row["tracking_difference"])) for row in daily_checks
    )
    maximum_unspent_difference = max(
        abs(float(row["unspent_difference"])) for row in daily_checks
    )
    summary = {
        "validation_days": [VALIDATION_DAYS[0], VALIDATION_DAYS[-1]],
        "day_count": len(VALIDATION_DAYS),
        "ticks_per_day": cfg.ticks,
        "interval_minutes": cfg.interval,
        "rows_per_condition": len(VALIDATION_DAYS) * cfg.ticks,
        "date_alignment": all(days == VALIDATION_DAYS for days in all_days.values()),
        "time_alignment": True,
        "tape_hash_mismatches": 0,
        "metric_atol": METRIC_ATOL,
        "maximum_tracking_error_replay_difference": maximum_tracking_difference,
        "maximum_unspent_replay_difference": maximum_unspent_difference,
        "data_source_mode": data_source_mode,
        "replay_seconds": runtime_seconds,
        "sources": {
            "stored_fixed_and_state_cap_1_0_daily": str(STORED_DAILY.relative_to(REPO)),
            "stored_state_cap_1_2_daily": str(CAP12_DAILY.relative_to(REPO)),
            "replayed_5_minute_records": str(TRACE_PATH.relative_to(REPO)),
            "daily_replay_check": str(DAILY_CHECK_PATH.relative_to(REPO)),
            "time_profile_summary": str(PROFILE_PATH.relative_to(REPO)),
        },
        "figures": {
            comparison["key"]: {
                "png": str((HERE / f"{comparison['stem']}.png").relative_to(REPO)),
                "svg": str((HERE / f"{comparison['stem']}.svg").relative_to(REPO)),
            }
            for comparison in COMPARISONS
        },
        "profile_checks": {
            comparison["key"]: {
                "difference_min_percent_points": float(
                    np.min(profiles[comparison["key"]]["reduction_mean"])
                ),
                "difference_max_percent_points": float(
                    np.max(profiles[comparison["key"]]["reduction_mean"])
                ),
                "symmetric_y_limit_percent_points": float(
                    max(
                        np.max(
                            np.abs(profiles[comparison["key"]]["reduction_mean"])
                        )
                        * 1.1,
                        1e-6,
                    )
                ),
                "mean_identity_max_abs_error": float(
                    np.max(
                        np.abs(
                            profiles[comparison["key"]]["reduction_mean"]
                            - (
                                profiles[comparison["key"]]["control_mean"]
                                - profiles[comparison["key"]]["experiment_mean"]
                            )
                        )
                    )
                ),
            }
            for comparison in COMPARISONS
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
