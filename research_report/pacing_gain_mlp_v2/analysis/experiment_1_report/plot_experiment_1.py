"""실험 1: 기존 B_state_PI와 재학습한 합산 기준 계수 신경망 비교."""

import csv
import gzip
import importlib.util
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
BUNDLE = HERE.parents[1]
REPO = BUNDLE.parents[1]
TRAIN_SCRIPT = BUNDLE / "code/train_gain_mlp.py"
RESULTS = BUNDLE / "results"
TRACE_PATH = HERE / "experiment_1_5min.csv.gz"
DAILY_PATH = HERE / "experiment_1_daily.csv"
SUMMARY_PATH = HERE / "summary.json"
FIGURE_PATH = HERE / "experiment-1-gain-mlp-tracking.png"
CONDITIONS = ("B_state_PI", "improved_gain_mlp_sum")
METRIC_ATOL = 1e-9


def load_experiment():
    spec = importlib.util.spec_from_file_location("gain_mlp_experiment", TRAIN_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.MODEL != "B":
        raise SystemExit("Experiment 1 report must use the B linear-regression predictor")
    return module


def read_json(path):
    return json.loads(path.read_text())


def make_policies(exp):
    old = exp.B_STATE
    picks = read_json(RESULTS / "picks.json")["picks"]["sum"]
    model = read_json(RESULTS / f"model_{picks['model']}.json")
    return {
        "B_state_PI": exp.StatePI(old["anchor"], old["weights"], False),
        "improved_gain_mlp_sum": exp.StatePI(
            model["anchor"], model[picks["weights"]], model["behind_input"]
        ),
    }, picks


def open_deterministic_gzip_text(path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0)
    text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
    return raw, compressed, text


def replay(exp, policies):
    fields = [
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
    errors = {
        condition: np.empty((len(exp.VALID), exp.cfg.ticks), dtype=float)
        for condition in CONDITIONS
    }
    daily = []
    raw, compressed, text = open_deterministic_gzip_text(TRACE_PATH)
    try:
        writer = csv.DictWriter(text, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for day_index, day in enumerate(exp.VALID):
            episode = exp.generate(exp.cfg, day)
            tape_sha256 = episode.digest()
            for condition, policy in policies.items():
                score, rows = exp.rollout(episode, exp.cfg, policy)
                daily.append(
                    {
                        "condition": condition,
                        "day": day,
                        "tape_sha256": tape_sha256,
                        "tracking_error": score["tracking_error"],
                        "unspent": score["unspent"],
                    }
                )
                for tick, row in enumerate(rows):
                    absolute_error = abs(float(row["error"])) * 100
                    errors[condition][day_index, tick] = absolute_error
                    writer.writerow(
                        {
                            "condition": condition,
                            "day": day,
                            "tape_sha256": tape_sha256,
                            "tick": tick,
                            "minute": (tick + 1) * exp.cfg.interval,
                            "budget": episode.budget,
                            "target_cumulative": row["target"],
                            "actual_cumulative": row["cumulative_spend"],
                            "abs_tracking_error_percent": absolute_error,
                        }
                    )
    finally:
        text.close()
        compressed.close()
        raw.close()

    with DAILY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(daily[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(daily)
    return errors, daily


def validate_daily(exp, daily):
    stored_rows = {}
    with (RESULTS / "validation_table.csv").open(newline="") as file:
        for row in csv.DictReader(file):
            stored_rows[row["model"]] = row

    expected_names = {
        "B_state_PI": "B_state_PI",
        "improved_gain_mlp_sum": "new_sum",
    }
    checks = {}
    for condition, stored_name in expected_names.items():
        rows = [row for row in daily if row["condition"] == condition]
        if [int(row["day"]) for row in rows] != list(exp.VALID):
            raise SystemExit(f"Validation-day mismatch for {condition}")
        tracking = float(np.mean([float(row["tracking_error"]) for row in rows]))
        unspent = float(np.mean([float(row["unspent"]) for row in rows]))
        stored_tracking = float(stored_rows[stored_name]["tracking_error"])
        stored_unspent = float(stored_rows[stored_name]["unspent"])
        if (
            abs(tracking - stored_tracking) > METRIC_ATOL
            or abs(unspent - stored_unspent) > METRIC_ATOL
        ):
            raise SystemExit(f"Stored validation metric mismatch for {condition}")
        checks[condition] = {
            "tracking_error": tracking,
            "unspent": unspent,
            "stored_tracking_difference": tracking - stored_tracking,
            "stored_unspent_difference": unspent - stored_unspent,
        }

    hashes = {}
    for row in daily:
        hashes.setdefault(int(row["day"]), set()).add(row["tape_sha256"])
    if any(len(day_hashes) != 1 for day_hashes in hashes.values()):
        raise SystemExit("Control and experiment used different auction tapes")
    return checks


def draw(exp, errors):
    control = errors["B_state_PI"]
    experiment = errors["improved_gain_mlp_sum"]
    control_mean = control.mean(axis=0)
    experiment_mean = experiment.mean(axis=0)
    difference = (control - experiment).mean(axis=0)
    identity_error = float(
        np.max(np.abs(difference - (control_mean - experiment_mean)))
    )
    if identity_error > 1e-12:
        raise SystemExit(f"Mean-difference identity failed: {identity_error}")

    hours = np.arange(1, exp.cfg.ticks + 1) * exp.cfg.interval / 60
    control_color = "#365F91"
    experiment_color = "#C65D3A"
    difference_color = "#397A5A"
    neutral_color = "#4B5563"

    plt.rcParams.update(
        {
            "font.family": ["Apple SD Gothic Neo", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    fig, (top, bottom) = plt.subplots(
        2,
        1,
        figsize=(11.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.15, 1], "hspace": 0.12},
    )
    top.plot(
        hours,
        control_mean,
        color=control_color,
        linewidth=2.2,
        label="대조군 · 기존 B_state_PI",
    )
    top.plot(
        hours,
        experiment_mean,
        color=experiment_color,
        linewidth=2.2,
        linestyle="--",
        label="실험군 · 개선 계수 신경망(합산 기준)",
    )
    top.set_ylabel("평균 절대 추종 오차 (%)")
    top.set_ylim(bottom=0)
    top.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        ncol=2,
    )

    bottom.plot(hours, difference, color=difference_color, linewidth=2.1)
    bottom.axhline(0, color=neutral_color, linewidth=1.1)
    limit = max(float(np.max(np.abs(difference))) * 1.1, 1e-6)
    bottom.set_ylim(-limit, limit)
    bottom.set_ylabel("평균 추종 오차 차이 (%p)")
    bottom.set_xlabel("시각")
    bottom.text(
        0.99,
        0.04,
        "양수 = 개선 계수 신경망의 오차가 더 낮음",
        transform=bottom.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        color=neutral_color,
    )

    for axis in (top, bottom):
        axis.set_xlim(0, 24)
        axis.set_xticks(np.arange(0, 25, 4))
        axis.grid(color="#D1D5DB", linewidth=0.7, alpha=0.65)
        axis.spines[["top", "right"]].set_visible(False)
    bottom.set_xticklabels([f"{hour:02d}:00" for hour in range(0, 25, 4)])

    fig.suptitle("실험 1 · PI 계수 산출 신경망 개선", fontsize=17, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.938,
        "B 선형회귀 지출 예측기는 동일 · 검증 200일 · 5분 단위 짝비교",
        ha="center",
        fontsize=10.5,
        color=neutral_color,
    )
    fig.subplots_adjust(left=0.105, right=0.98, top=0.84, bottom=0.10)
    fig.savefig(FIGURE_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "difference_min_percent_points": float(difference.min()),
        "difference_max_percent_points": float(difference.max()),
        "symmetric_y_limit_percent_points": limit,
        "mean_identity_max_abs_error": identity_error,
    }


def main():
    exp = load_experiment()
    policies, pick = make_policies(exp)
    errors, daily = replay(exp, policies)
    daily_checks = validate_daily(exp, daily)
    profile_checks = draw(exp, errors)
    control = daily_checks["B_state_PI"]
    experiment = daily_checks["improved_gain_mlp_sum"]
    summary = {
        "control": "B_state_PI",
        "experiment": "improved_gain_mlp_sum",
        "selected_model": pick,
        "validation_days": [int(exp.VALID[0]), int(exp.VALID[-1])],
        "day_count": len(exp.VALID),
        "ticks_per_day": exp.cfg.ticks,
        "rows_per_condition": len(exp.VALID) * exp.cfg.ticks,
        "same_day_pairing": True,
        "auction_tape_hash_match": True,
        "daily_metric_checks": daily_checks,
        "control_minus_experiment_percent_points": {
            "tracking_error": (control["tracking_error"] - experiment["tracking_error"]) * 100,
            "unspent": (control["unspent"] - experiment["unspent"]) * 100,
        },
        "profile_checks": profile_checks,
        "files": {
            "trace": str(TRACE_PATH.relative_to(REPO)),
            "daily": str(DAILY_PATH.relative_to(REPO)),
            "figure": str(FIGURE_PATH.relative_to(REPO)),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
