"""Replay B_state_PI with cap 1.0 / 1.2 on days 100065, 100080 and save 5-minute traces.

Reuses the rollout code in plot.py (everything before its plotting section) so the
numbers come from the same replay the draft figure used.
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
source = (HERE / "plot.py").read_text()
replay_source = source.split("results = {}")[0]
ns = {"__file__": str(HERE / "plot.py"), "__name__": "cap_replay"}
exec(compile(replay_source, str(HERE / "plot.py"), "exec"), ns)

lib = ns["ns"]
cfg = ns["cfg"]
summary = {}
trace_rows = []

for day in (100065, 100080):
    episode = lib["generate"](cfg, day)
    score_1, rows_1, auctions_1 = ns["rollout"](episode, 1.0)
    score_12, rows_12, auctions_12 = ns["rollout"](episode, 1.2)

    parts = {"common": [0.0, 0], "only12": [0.0, 0], "only1": [0.0, 0]}
    for key, (won_1, paid_1) in auctions_1.items():
        won_12, paid_12 = auctions_12[key]
        if won_1 and won_12:
            parts["common"][0] += paid_12 - paid_1
            parts["common"][1] += 1
        elif won_12:
            parts["only12"][0] += paid_12
            parts["only12"][1] += 1
        elif won_1:
            parts["only1"][0] -= paid_1
            parts["only1"][1] += 1

    budget = episode.budget
    for tick, (r1, r12) in enumerate(zip(rows_1, rows_12)):
        trace_rows.append({
            "day": day,
            "tick": tick,
            "hour": (tick + 1) * cfg.interval / 60,
            "budget": budget,
            "target_cumulative": r1["target"],
            "cumulative_cap1_0": r1["cumulative_spend"],
            "cumulative_cap1_2": r12["cumulative_spend"],
            "spend_cap1_0": r1["spend"],
            "spend_cap1_2": r12["spend"],
            "wins_cap1_0": r1["wins"],
            "wins_cap1_2": r12["wins"],
            "multiplier_cap1_0": r1["action"],
            "multiplier_cap1_2": r12["action"],
        })

    total = sum(v[0] for v in parts.values())
    summary[day] = {
        "budget": budget,
        "score_cap1_0": score_1,
        "score_cap1_2": score_12,
        "common_payment_delta": parts["common"][0],
        "common_wins": parts["common"][1],
        "only12_spend": parts["only12"][0],
        "only12_wins": parts["only12"][1],
        "only1_spend_neg": parts["only1"][0],
        "only1_wins": parts["only1"][1],
        "total_spend_delta": total,
        "spend_delta_check": rows_12[-1]["cumulative_spend"] - rows_1[-1]["cumulative_spend"],
        "net_win_change": parts["only12"][1] - parts["only1"][1],
    }

with open(HERE / "traces_5min.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(trace_rows[0]))
    writer.writeheader()
    writer.writerows(trace_rows)
(HERE / "decomposition.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
