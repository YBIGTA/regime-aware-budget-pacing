import ast
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / "research_report/pacing_abc_state_pi/analysis/day_100063/analyze.py"


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
    def __init__(self, fixed, schedule, upper):
        super().__init__(fixed, schedule)
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


def controller(cap):
    return PredictivePICap(
        tuple(state["anchor"]),
        ns["GainMLP"](state["anchor"], state["weights"]),
        cap,
    )


def execute_detailed(day, tick_no, tick, action, remaining):
    bids = tick.base * action
    eligible = (bids > tick.competitor) & (bids > 0)
    total = float(bids[eligible].sum())
    auctions = []

    if total <= remaining:
        for index, bid in enumerate(bids):
            won = bool(eligible[index])
            auctions.append(
                ((day, tick_no, index), won, float(bid) if won else 0.0)
            )
        return total, int(eligible.sum()), False, auctions

    spent = 0.0
    wins = 0
    for index, bid in enumerate(bids):
        won = bool(eligible[index]) and float(bid) <= remaining - spent
        if won:
            spent += float(bid)
            wins += 1
        auctions.append(((day, tick_no, index), won, float(bid) if won else 0.0))
    return spent, wins, True, auctions


def rollout(episode, cap):
    policy = controller(cap)
    policy.reset()
    spent = 0.0
    history = []
    rows = []
    auctions = {}
    zero = core.Feedback(-1, 0.0, 0.0, 0, 0, 0.0, False)

    for tick_no, tick in enumerate(episode.tape):
        padded = ([zero] * cfg.history + history)[-cfg.history :]
        obs = core.Observation(
            tick_no, cfg.ticks, episode.budget, spent, tuple(padded)
        )
        action, debug = policy.act(obs)
        cost, wins, censored, auction_rows = execute_detailed(
            episode.day,
            tick_no,
            tick,
            action,
            max(0.0, episode.budget - spent),
        )
        auctions.update({key: (won, payment) for key, won, payment in auction_rows})
        spent += cost
        feedback = core.Feedback(
            tick_no,
            action,
            cost,
            len(tick.base),
            wins,
            float(tick.base.mean()) if len(tick.base) else 0.0,
            censored,
        )
        history.append(feedback)
        rows.append(
            {
                "target": (tick_no + 1) * episode.budget / cfg.ticks,
                "cumulative_spend": spent,
                "error": (tick_no + 1) / cfg.ticks - spent / episode.budget,
                **asdict(feedback),
                **debug,
            }
        )

    errors = np.asarray([row["error"] for row in rows])
    return {
        "tracking_error": float(np.abs(errors).mean()),
        "unspent": max(0.0, 1 - spent / episode.budget),
    }, rows, auctions


def contribution(auctions_1, auctions_12):
    values = {"Common wins": 0.0, "New at 1.2": 0.0, "Lost at 1.2": 0.0}
    for key in auctions_1:
        won_1, paid_1 = auctions_1[key]
        won_12, paid_12 = auctions_12[key]
        if won_1 and won_12:
            values["Common wins"] += paid_12 - paid_1
        elif won_12:
            values["New at 1.2"] += paid_12
        elif won_1:
            values["Lost at 1.2"] -= paid_1
    return values


results = {}
for day in (100065, 100080):
    episode = ns["generate"](cfg, day)
    score_1, rows_1, auctions_1 = rollout(episode, 1.0)
    score_12, rows_12, auctions_12 = rollout(episode, 1.2)
    results[day] = {
        "budget": episode.budget,
        "score_1": score_1,
        "score_12": score_12,
        "rows_1": rows_1,
        "rows_12": rows_12,
        "contribution": contribution(auctions_1, auctions_12),
    }


plt.style.use("seaborn-v0_8-whitegrid")
colors = {
    "cap1": "#2563eb",
    "cap12": "#ea580c",
    "neutral": "#64748b",
    "only1": "#2563eb",
    "only12": "#ea580c",
    "total": "#111827",
}
fig, axes = plt.subplots(2, 2, figsize=(16, 9.5), constrained_layout=True)
hours = np.arange(1, cfg.ticks + 1) * cfg.interval / 60

for column, day in enumerate((100065, 100080)):
    result = results[day]
    rows_1 = result["rows_1"]
    rows_12 = result["rows_12"]
    target = np.asarray([row["target"] for row in rows_1])
    cumulative_1 = np.asarray([row["cumulative_spend"] for row in rows_1])
    cumulative_12 = np.asarray([row["cumulative_spend"] for row in rows_12])
    gap_1 = target - cumulative_1
    gap_12 = target - cumulative_12

    ax = axes[0, column]
    ax.axhline(0, color=colors["total"], linewidth=1.2, label="Target (zero gap)")
    ax.plot(hours, gap_1, color=colors["cap1"], linewidth=2.1, label="Cap 1.0")
    ax.plot(hours, gap_12, color=colors["cap12"], linewidth=2.1, label="Cap 1.2")
    ax.set_title(f"Day {day} — tracking gap over time")
    ax.set_ylabel("Target cumulative spend − actual")
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, 4))
    ax.legend(loc="upper left", frameon=False, ncol=3)

    score_1 = result["score_1"]
    score_12 = result["score_12"]
    annotation = (
        f"Final gap  {gap_1[-1]:+.3f} → {gap_12[-1]:+.3f}\n"
        f"Tracking error  {score_1['tracking_error']:.3%} → {score_12['tracking_error']:.3%}\n"
        f"Unspent  {score_1['unspent']:.3%} → {score_12['unspent']:.3%}"
    )
    ax.text(
        0.98,
        0.05,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )
    ax.text(
        0.02,
        0.03,
        "Positive = underspend   |   Negative = overspend   |   Closer to zero = better",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9,
        color=colors["neutral"],
    )

    ax = axes[1, column]
    contributions = result["contribution"]
    labels = ["Common-win\npayment delta", "1.2-only\nspend", "1.0-only\nspend (−)", "Total\nspend delta"]
    values = list(contributions.values())
    values.append(sum(values))
    bar_colors = [colors["neutral"], colors["only12"], colors["only1"], colors["total"]]
    bars = ax.bar(labels, values, color=bar_colors, width=0.68)
    ax.axhline(0, color=colors["neutral"], linewidth=1.0)
    ax.bar_label(bars, labels=[f"{value:+.3f}" for value in values], padding=4, fontsize=10)
    ax.set_title("Auction-level decomposition of total spend difference")
    ax.set_ylabel("Contribution to 1.2 − 1.0 spend")

for ax in axes.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", visible=False)

axes[0, 0].set_xlabel("Hour of day")
axes[0, 1].set_xlabel("Hour of day")
fig.suptitle(
    "B_state_PI — bid multiplier cap 1.0 vs 1.2\n"
    "Same auction tape, budget path, predictor, and PI settings",
    fontsize=18,
    fontweight="bold",
)
fig.savefig(HERE / "cap-comparison.png", dpi=180, bbox_inches="tight")
