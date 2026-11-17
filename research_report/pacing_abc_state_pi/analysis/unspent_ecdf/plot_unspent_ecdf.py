import ast
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / "research_report/pacing_abc_state_pi/analysis/day_100063/analyze.py"
STORED_DAILY = REPO / "research_report/pacing_abc_state_pi/results/daily_scores.csv"
CAP12_DAILY = HERE / "cap_1_2_daily.csv"
SUMMARY = HERE / "summary.json"
PNG = HERE / "unspent-ecdf.png"
SVG = HERE / "unspent-ecdf.svg"
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


def draw_panel(ax, title, control_label, experiment_label, control, experiment, paired):
    control_color = "#365F91"
    experiment_color = "#C65D3A"
    series = (
        (control_label, control, control_color, "-"),
        (experiment_label, experiment, experiment_color, "--"),
    )

    for label, values, color, linestyle in series:
        x, y = ecdf(values)
        ax.step(
            x,
            y,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=2.3,
            label=f"{label}  ·  평균 {values.mean() * 100:.4f}%",
        )

    ax.axvline(0.1, color="#6B7280", linestyle=":", linewidth=1.25)
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=10)
    ax.set_xlabel("최종 미소진율 (%)")
    ax.set_ylabel("해당 값 이하로 마감한 시나리오 (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 101)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="both", color="#D1D5DB", linewidth=0.7, alpha=0.65)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9.5)

    control_hits = int(np.sum(control <= THRESHOLD_RATIO))
    experiment_hits = int(np.sum(experiment <= THRESHOLD_RATIO))
    ax.text(
        0.025,
        0.97,
        "0.1% 이하\n"
        f"대조군  {control_hits}/200 ({control_hits / 2:.1f}%)\n"
        f"실험군  {experiment_hits}/200 ({experiment_hits / 2:.1f}%)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D1D5DB"},
    )

    inset = inset_axes(ax, width="47%", height="43%", loc="center right", borderpad=1.0)
    for _, values, color, linestyle in series:
        x, y = ecdf(values)
        inset.step(x, y, where="post", color=color, linestyle=linestyle, linewidth=1.7)
    inset.axvline(0.1, color="#6B7280", linestyle=":", linewidth=1.0)
    inset.set_xlim(0, 0.2)
    inset.set_ylim(0, 101)
    inset.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    inset.set_yticks([0, 50, 100])
    inset.tick_params(labelsize=7.5)
    inset.grid(color="#D1D5DB", linewidth=0.5, alpha=0.55)
    inset.set_title("0~0.2% 확대", fontsize=8.5, pad=3)
    mark_inset(ax, inset, loc1=2, loc2=4, fc="none", ec="#9CA3AF", lw=0.8)

    ax.text(
        0.0,
        -0.24,
        "일별 짝비교: "
        f"개선 {paired['improved_days']}일 · 악화 {paired['worsened_days']}일 · "
        f"동률 {paired['tied_days']}일  (|차이| ≤ {TIE_ATOL:.0e})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color="#374151",
    )


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
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.5))
    draw_panel(
        axes[0],
        "상태별 PI 적용",
        "B · 고정 PI",
        "B_state_PI · 상태별 PI",
        fixed,
        state_cap1,
        comparisons["state_pi_vs_fixed_pi"],
    )
    draw_panel(
        axes[1],
        "입찰 승수 상한 완화",
        "B_state_PI · 상한 1.0",
        "B_state_PI · 상한 1.2",
        state_cap1,
        state_cap12,
        comparisons["cap_1_2_vs_cap_1_0"],
    )
    fig.suptitle("최종 미소진율 누적분포 · 검증 200일", fontsize=17, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "ECDF가 왼쪽·위쪽일수록 더 많은 날이 낮은 미소진율로 마감 · 원본 비율값을 ×100해 %로 표시",
        ha="center",
        fontsize=10.5,
        color="#4B5563",
    )
    fig.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.22, wspace=0.22)
    fig.savefig(PNG, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(SVG, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
