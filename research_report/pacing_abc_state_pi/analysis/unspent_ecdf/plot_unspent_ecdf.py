import ast
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / "research_report/pacing_abc_state_pi/analysis/day_100063/analyze.py"
STORED_DAILY = REPO / "research_report/pacing_abc_state_pi/results/daily_scores.csv"
CAP12_DAILY = HERE / "cap_1_2_daily.csv"
SUMMARY = HERE / "summary.json"
PNG = HERE / "unspent-ecdf.png"
SVG = HERE / "unspent-ecdf.svg"
REPORT_PNG = HERE / "unspent-ecdf-report.png"
REPORT_SVG = HERE / "unspent-ecdf-report.svg"
APPENDIX_PNG = HERE / "unspent-ecdf-appendix.png"
APPENDIX_SVG = HERE / "unspent-ecdf-appendix.svg"
VALIDATION_DAYS = list(range(100000, 100200))
TIE_ATOL = 1e-12
THRESHOLD_RATIO = 0.001  # 0.1%


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
state = ns["b_frozen"]["B_state_PI"]


class PredictivePICap(ns["PredictivePI"]):
    def __init__(self, upper):
        super().__init__(
            tuple(state["anchor"]),
            ns["GainMLP"](state["anchor"], state["weights"]),
        )
        self.upper = upper

    def act(self, obs):
        base, slope = ns["base_and_slope"](obs)
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
        return float(np.clip(raw, lo, hi)), {
            "base": base,
            "raw": float(raw),
            "integral": float(self.integral),
            "integral_blocked": bool(blocked),
            "saturated": bool(raw < lo or raw > hi),
            "kp": float(kp),
            "ki": float(ki),
        }


def rollout_cap12(episode):
    controller = PredictivePICap(1.2)
    controller.reset()
    spent = 0.0
    history = []
    errors = []
    zero = core.Feedback(-1, 0.0, 0.0, 0, 0, 0.0, False)

    for tick_no, tick in enumerate(episode.tape):
        padded = ([zero] * cfg.history + history)[-cfg.history :]
        obs = core.Observation(
            tick_no, cfg.ticks, episode.budget, spent, tuple(padded)
        )
        action, _ = controller.act(obs)
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
        errors.append((tick_no + 1) / cfg.ticks - spent / episode.budget)

    tracking_error = float(np.abs(np.asarray(errors)).mean())
    unspent = max(0.0, 1 - spent / episode.budget)
    return {
        "day": episode.day,
        "tape_sha256": episode.digest(),
        "tracking_error": tracking_error,
        "unspent": unspent,
        "objective": tracking_error + unspent,
    }


def read_stored_daily():
    result = {"B_fixed_PI": {}, "B_state_PI": {}}
    with STORED_DAILY.open(newline="") as file:
        for row in csv.DictReader(file):
            if row["model"] in result:
                result[row["model"]][int(row["day"])] = {
                    "tape_sha256": row["tape_sha256"],
                    "tracking_error": float(row["tracking_error"]),
                    "unspent": float(row["unspent"]),
                    "objective": float(row["objective"]),
                }
    for model, rows in result.items():
        if sorted(rows) != VALIDATION_DAYS:
            raise SystemExit(f"{model}: expected validation days 100000~100199")
    for day in VALIDATION_DAYS:
        if (
            result["B_fixed_PI"][day]["tape_sha256"]
            != result["B_state_PI"][day]["tape_sha256"]
        ):
            raise SystemExit(f"Stored tape mismatch on day {day}")
    return result


def generate_cap12(stored):
    started = time.perf_counter()
    rows = []
    for day in VALIDATION_DAYS:
        episode = ns["generate"](cfg, day)
        row = rollout_cap12(episode)
        if row["tape_sha256"] != stored["B_state_PI"][day]["tape_sha256"]:
            raise SystemExit(f"Cap 1.2 tape mismatch on day {day}")
        rows.append(row)

    with CAP12_DAILY.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, time.perf_counter() - started


def paired_summary(control, experiment):
    delta = experiment - control
    return {
        "improved_days": int(np.sum(delta < -TIE_ATOL)),
        "worsened_days": int(np.sum(delta > TIE_ATOL)),
        "tied_days": int(np.sum(np.abs(delta) <= TIE_ATOL)),
        "tie_atol_ratio": TIE_ATOL,
    }


def group_summary(values):
    hits = int(np.sum(values <= THRESHOLD_RATIO))
    return {
        "mean_unspent_ratio": float(values.mean()),
        "mean_unspent_percent": float(values.mean() * 100),
        "at_or_below_0_1_percent_days": hits,
        "at_or_below_0_1_percent_fraction": hits / len(values),
        "at_or_below_0_1_percent_percent": hits / len(values) * 100,
    }


def ecdf(values):
    x = np.sort(values * 100)
    y = np.arange(1, len(x) + 1) / len(x) * 100
    return np.r_[0.0, x], np.r_[0.0, y]


def ecdf_difference(control, experiment):
    control_percent = np.sort(control * 100)
    experiment_percent = np.sort(experiment * 100)
    x = np.unique(
        np.r_[0.0, control_percent, experiment_percent, THRESHOLD_RATIO * 100, 0.2]
    )
    control_y = np.searchsorted(control_percent, x, side="right") / len(control) * 100
    experiment_y = (
        np.searchsorted(experiment_percent, x, side="right")
        / len(experiment)
        * 100
    )
    return x, experiment_y - control_y


def threshold_line(control, experiment):
    control_hits = int(np.sum(control <= THRESHOLD_RATIO))
    experiment_hits = int(np.sum(experiment <= THRESHOLD_RATIO))
    control_percent = control_hits / len(control) * 100
    experiment_percent = experiment_hits / len(experiment) * 100
    difference = experiment_percent - control_percent
    extra_days = experiment_hits - control_hits
    if extra_days:
        detail = f"{difference:+.0f}%p, {abs(extra_days)}일 더 {'많음' if extra_days > 0 else '적음'}"
    else:
        detail = f"{difference:.0f}%p"
    return (
        f"0.10% 이하: {control_percent:.0f}% → {experiment_percent:.0f}% "
        f"({detail})"
    )


def clean_svg(path):
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def style_axis(ax):
    ax.grid(color="#D1D5DB", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(0.1, color="#6B7280", linestyle=":", linewidth=1.0)
    ax.set_xlim(0, 0.2)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))


def draw_report(comparisons):
    control_color = "#365F91"
    experiment_color = "#C65D3A"
    difference_color = "#397A5A"
    neutral_color = "#4B5563"
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.5, 8.8),
        sharex="col",
        gridspec_kw={"height_ratios": [1.35, 1], "hspace": 0.14, "wspace": 0.22},
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.22)

    column_centers = (0.275, 0.755)
    for column, comparison in enumerate(comparisons):
        top = axes[0, column]
        bottom = axes[1, column]
        control = comparison["control"]
        experiment = comparison["experiment"]

        x_control, y_control = ecdf(control)
        x_experiment, y_experiment = ecdf(experiment)
        control_line = top.step(
            x_control,
            y_control,
            where="post",
            color=control_color,
            linestyle="-",
            linewidth=2.3,
            label=comparison["control_label"],
        )[0]
        experiment_line = top.step(
            x_experiment,
            y_experiment,
            where="post",
            color=experiment_color,
            linestyle="--",
            linewidth=2.3,
            label=comparison["experiment_label"],
        )[0]
        style_axis(top)
        top.set_ylim(0, 100)
        top.set_yticks(np.arange(0, 101, 20))
        top.set_ylabel("누적 시나리오 비율 (%)")

        x_difference, difference = ecdf_difference(control, experiment)
        bottom.step(
            x_difference,
            difference,
            where="post",
            color=difference_color,
            linewidth=2.1,
        )
        style_axis(bottom)
        bottom.axhline(0, color="#6B7280", linewidth=1.0)
        bottom.set_ylabel("ECDF 차이 (%p)")
        bottom.set_xlabel("최종 미소진율 (%)")

        center = column_centers[column]
        fig.text(
            center,
            0.875,
            comparison["title"],
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
        )
        fig.text(
            center,
            0.84,
            threshold_line(control, experiment),
            ha="center",
            va="center",
            fontsize=10.5,
            color=neutral_color,
        )
        fig.legend(
            [control_line, experiment_line],
            [comparison["control_label"], comparison["experiment_label"]],
            loc="upper center",
            bbox_to_anchor=(center, 0.815),
            ncol=2,
            frameon=False,
            fontsize=9.5,
        )
        paired = comparison["paired"]
        fig.text(
            center,
            0.075,
            f"평균 {control.mean() * 100:.4f}% → {experiment.mean() * 100:.4f}%\n"
            f"일별 짝비교: 개선 {paired['improved_days']}일 · "
            f"악화 {paired['worsened_days']}일 · 동률 {paired['tied_days']}일",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#374151",
        )

    fig.suptitle("최종 미소진율 누적분포 · 검증 200일", fontsize=17, fontweight="bold", y=0.97)
    fig.text(
        0.5,
        0.93,
        "전체 200일을 분모로 계산 · 표시 구간 0.00~0.20%",
        ha="center",
        fontsize=10.5,
        color=neutral_color,
    )
    fig.text(
        0.5,
        0.145,
        "ECDF 차이 = 실험군 − 대조군 · 양수는 해당 미소진율 이하로 마감한 날이 실험군에서 더 많다는 뜻",
        ha="center",
        fontsize=10,
        color=neutral_color,
    )
    fig.text(
        0.5,
        0.025,
        f"동률 기준: |실험군 − 대조군| ≤ {TIE_ATOL:.0e} (원본 비율값)",
        ha="center",
        fontsize=8.8,
        color=neutral_color,
    )
    fig.savefig(REPORT_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(REPORT_SVG, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    clean_svg(REPORT_SVG)


def draw_appendix(comparisons):
    control_color = "#365F91"
    experiment_color = "#C65D3A"
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))
    maximum = max(
        float(np.max(comparison[key]) * 100)
        for comparison in comparisons
        for key in ("control", "experiment")
    )
    for ax, comparison in zip(axes, comparisons):
        for label, values, color, linestyle in (
            (comparison["control_label"], comparison["control"], control_color, "-"),
            (comparison["experiment_label"], comparison["experiment"], experiment_color, "--"),
        ):
            x, y = ecdf(values)
            ax.step(x, y, where="post", color=color, linestyle=linestyle, linewidth=2.1, label=label)
        ax.set_title(comparison["title"], fontsize=13, fontweight="bold")
        ax.set_xlim(0, maximum * 1.02)
        ax.set_ylim(0, 100)
        ax.set_xlabel("최종 미소진율 (%)")
        ax.set_ylabel("누적 시나리오 비율 (%)")
        ax.grid(color="#D1D5DB", linewidth=0.7, alpha=0.65)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.suptitle("최종 미소진율 ECDF · 전체 범위 부록", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.075, right=0.985, top=0.82, bottom=0.25, wspace=0.22)
    fig.savefig(APPENDIX_PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(APPENDIX_SVG, bbox_inches="tight", facecolor="white", metadata={"Date": None})
    plt.close(fig)
    clean_svg(APPENDIX_SVG)


def main():
    stored = read_stored_daily()
    cap12_rows, runtime_seconds = generate_cap12(stored)
    cap12 = {row["day"]: row for row in cap12_rows}

    for day in VALIDATION_DAYS:
        if cap12[day]["tape_sha256"] != stored["B_state_PI"][day]["tape_sha256"]:
            raise SystemExit(f"Final tape check failed on day {day}")

    fixed = np.asarray([stored["B_fixed_PI"][day]["unspent"] for day in VALIDATION_DAYS])
    state_cap1 = np.asarray([stored["B_state_PI"][day]["unspent"] for day in VALIDATION_DAYS])
    state_cap12 = np.asarray([cap12[day]["unspent"] for day in VALIDATION_DAYS])

    comparisons = {
        "state_pi_vs_fixed_pi": paired_summary(fixed, state_cap1),
        "cap_1_2_vs_cap_1_0": paired_summary(state_cap1, state_cap12),
    }
    result = {
        "validation_days": [VALIDATION_DAYS[0], VALIDATION_DAYS[-1]],
        "day_count": len(VALIDATION_DAYS),
        "stored_values_are": "ratios; multiplied by 100 for percent display",
        "threshold_ratio": THRESHOLD_RATIO,
        "threshold_percent": THRESHOLD_RATIO * 100,
        "sources": {
            "fixed_and_state_cap_1_0": str(STORED_DAILY.relative_to(REPO)),
            "state_cap_1_2": str(CAP12_DAILY.relative_to(REPO)),
        },
        "tape_hash_mismatches": 0,
        "cap_1_2_replay_seconds": runtime_seconds,
        "groups": {
            "B_fixed_PI": group_summary(fixed),
            "B_state_PI_cap_1_0": group_summary(state_cap1),
            "B_state_PI_cap_1_2": group_summary(state_cap12),
        },
        "comparisons": comparisons,
        "figures": {
            "report_png": str(REPORT_PNG.relative_to(REPO)),
            "report_svg": str(REPORT_SVG.relative_to(REPO)),
            "appendix_png": str(APPENDIX_PNG.relative_to(REPO)),
            "appendix_svg": str(APPENDIX_SVG.relative_to(REPO)),
        },
    }

    expected_cap12_percent = 0.044549
    if not np.isclose(state_cap12.mean() * 100, expected_cap12_percent, atol=5e-7):
        raise SystemExit(
            "Cap 1.2 mean does not reproduce the recorded 0.044549% summary: "
            f"{state_cap12.mean() * 100:.9f}%"
        )
    SUMMARY.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": ["Apple SD Gothic Neo", "Arial Unicode MS", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "svg.fonttype": "none",
        }
    )
    report_comparisons = (
        {
            "title": "고정 PI vs 상태별 PI",
            "control_label": "B · 고정 PI",
            "experiment_label": "B_state_PI · 상태별 PI",
            "control": fixed,
            "experiment": state_cap1,
            "paired": comparisons["state_pi_vs_fixed_pi"],
        },
        {
            "title": "승수 상한 1.0 vs 1.2",
            "control_label": "B_state_PI · 상한 1.0",
            "experiment_label": "B_state_PI · 상한 1.2",
            "control": state_cap1,
            "experiment": state_cap12,
            "paired": comparisons["cap_1_2_vs_cap_1_0"],
        },
    )
    draw_report(report_comparisons)
    draw_appendix(report_comparisons)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
