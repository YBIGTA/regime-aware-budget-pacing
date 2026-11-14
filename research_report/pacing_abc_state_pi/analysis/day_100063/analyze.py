import csv
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import sklearn


DAY = 100063
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "experiments/pacing_abc"))

from core import Config, Gains, features, generate, rollout  # noqa: E402
from run import csv_write, read_json, source_hash, write_json  # noqa: E402


def sha(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


bundle = REPO / "research_report/pacing_abc_state_pi"
archive = REPO / "research_report/pacing_abc_A/20261106_132615_119414/results"
protocol_path = archive / "pacing_abc_strong_v2/common_data/protocol.json"
a_path = archive / "pacing_abc_ki_check_v3/frozen_A.json"
predictor_path = bundle / "code/abc/predictors.joblib"
abc_frozen_path = bundle / "code/abc/frozen_ABC.json"
b_frozen_path = bundle / "code/B/frozen_experiment2.json"
daily_scores_path = bundle / "results/daily_scores.csv"

checks = {
    "A": sha(a_path),
    "protocol": sha(protocol_path),
    "predictors": sha(predictor_path),
    "core": source_hash(),
    "numpy": np.__version__,
    "sklearn": sklearn.__version__,
}
abc_frozen = read_json(abc_frozen_path)
b_frozen = read_json(b_frozen_path)
for key, value in checks.items():
    if abc_frozen[key] != value or b_frozen["source"][key] != value:
        raise SystemExit(f"Frozen artifact mismatch: {key}")

params = read_json(a_path)["parameters"]
cfg = replace(
    Config(**read_json(protocol_path)["config"]),
    initial_multiplier=params["m0"],
)
a_gains = Gains(params["kp"], params["ki"])
gain_scale = np.array([a_gains.kp, a_gains.ki])
grid = np.linspace(cfg.multiplier_min, cfg.multiplier_max, 101)
scaler, y_mean, y_std, linear, _ = joblib.load(predictor_path)


def base_and_slope(obs):
    if obs.tick == 0:
        return cfg.initial_multiplier, 0.0

    batch = np.repeat(features(obs, 0.0)[None, :], len(grid), axis=0)
    batch[:, 0] = grid
    normalized = scaler.transform(batch).astype(np.float32)
    predicted = np.maximum(linear.predict(normalized) * y_std + y_mean, 0)
    target = 1 / obs.ticks
    chosen = int(np.argmin(np.abs(predicted.astype(float) - target)))
    left = max(0, chosen - 5)
    right = min(len(grid) - 1, chosen + 5)
    slope = (float(predicted[right]) - float(predicted[left])) / (
        grid[right] - grid[left]
    )
    return float(grid[chosen]), slope


def state_features(obs, base, slope):
    recent = [feedback for feedback in obs.history[-6:] if feedback.tick >= 0]
    recent_error = (
        len(recent) / obs.ticks
        - sum(feedback.spend for feedback in recent) / obs.budget
    )
    traffic = np.array(
        [feedback.opportunities for feedback in recent], dtype=float
    )
    traffic_variation = (
        float(traffic.std() / max(1.0, traffic.mean())) if len(traffic) else 0.0
    )
    return np.array(
        [
            2 * obs.tick / obs.ticks - 1,
            2 * (1 - obs.cumulative_spend / obs.budget) - 1,
            np.tanh(20 * obs.error),
            np.tanh(50 * recent_error),
            np.tanh(traffic_variation),
            2 * base - 1,
            np.tanh(slope * obs.ticks / 10),
        ]
    )


class GainMLP:
    def __init__(self, anchor, weights):
        self.anchor = np.asarray(anchor, dtype=float)
        weights = np.asarray(weights, dtype=float)
        self.hidden = weights[:32].reshape(8, 4)
        self.head = weights[32:].reshape(4, 2)

    def __call__(self, inputs):
        hidden = np.tanh(inputs @ self.hidden[:7] + self.hidden[7])
        adjustment = np.tanh(hidden @ self.head)
        return np.clip(
            self.anchor + adjustment * gain_scale,
            0.0,
            2 * gain_scale,
        )


class PredictivePI:
    def __init__(self, fixed, schedule):
        self.fixed = fixed
        self.schedule = schedule

    def reset(self):
        self.integral = 0.0

    def act(self, obs):
        base, slope = base_and_slope(obs)
        kp, ki = self.schedule(state_features(obs, base, slope))
        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw = base + kp * error + candidate
        lo, hi = cfg.multiplier_min, cfg.multiplier_max
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


state = b_frozen["B_state_PI"]
controller = PredictivePI(
    tuple(state["anchor"]),
    GainMLP(state["anchor"], state["weights"]),
)
episode = generate(cfg, DAY)
score, trace = rollout(episode, cfg, controller)

with daily_scores_path.open(newline="") as file:
    stored = [
        row
        for row in csv.DictReader(file)
        if row["model"] == "B_state_PI" and int(row["day"]) == DAY
    ]
if len(stored) != 1:
    raise SystemExit(f"Expected one stored row for day {DAY}, found {len(stored)}")
stored = stored[0]

if score["tape_sha256"] != stored["tape_sha256"]:
    raise SystemExit("Replayed auction tape does not match the stored result")
for metric in ("tracking_error", "unspent", "objective", "multiplier_tv"):
    if not np.isclose(score[metric], float(stored[metric]), rtol=0, atol=1e-10):
        raise SystemExit(
            f"Replayed {metric} differs: {score[metric]} != {stored[metric]}"
        )

csv_write(HERE / "trace.csv", trace)

hours = np.array([(row["tick"] + 1) * cfg.interval / 60 for row in trace])
target = np.array([row["target"] for row in trace])
actual = np.array([row["cumulative_spend"] for row in trace])
gap = target - actual
max_index = int(np.argmax(np.abs(gap)))
max_gap = float(gap[max_index])
final_gap = float(gap[-1])
recovered_fraction = 1 - abs(final_gap) / abs(max_gap)
direction = "below target" if max_gap > 0 else "above target"


def first_recovery_hour(fraction):
    for index in range(max_index + 1, len(gap)):
        if abs(gap[index]) <= abs(max_gap) * fraction:
            return float(hours[index])
    return None


first_crossing_hour = next(
    (
        float(hours[index])
        for index in range(max_index + 1, len(gap))
        if gap[index] * max_gap < 0
    ),
    None,
)

summary = {
    "day": DAY,
    "model": "B_state_PI",
    "tape_sha256": score["tape_sha256"],
    "budget": episode.budget,
    "tracking_error": score["tracking_error"],
    "unspent": score["unspent"],
    "objective": score["objective"],
    "max_absolute_gap": abs(max_gap),
    "max_gap_fraction_of_budget": abs(max_gap) / episode.budget,
    "max_gap_direction": direction,
    "max_gap_tick": int(trace[max_index]["tick"]),
    "max_gap_hour": float(hours[max_index]),
    "final_gap": final_gap,
    "final_gap_fraction_of_budget": final_gap / episode.budget,
    "gap_recovered_fraction": recovered_fraction,
    "recovery_hours": {
        "50_percent_of_peak_gap": first_recovery_hour(0.50),
        "25_percent_of_peak_gap": first_recovery_hour(0.25),
        "10_percent_of_peak_gap": first_recovery_hour(0.10),
        "1_percent_of_peak_gap": first_recovery_hour(0.01),
    },
    "first_target_crossing_hour_after_peak": first_crossing_hour,
    "stored_metrics_reproduced": True,
}
write_json(HERE / "summary.json", summary)

plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(12, 6.75))
ax.plot(
    hours,
    target,
    color="#6b7280",
    linewidth=2.0,
    linestyle="--",
    label="Target cumulative spend",
)
ax.plot(
    hours,
    actual,
    color="#2563eb",
    linewidth=2.2,
    label="Actual cumulative spend",
)
ax.vlines(
    hours[max_index],
    min(target[max_index], actual[max_index]),
    max(target[max_index], actual[max_index]),
    color="#dc2626",
    linewidth=2.2,
    label="Maximum absolute gap",
)
ax.scatter(
    [hours[max_index], hours[max_index]],
    [target[max_index], actual[max_index]],
    color="#dc2626",
    s=34,
    zorder=5,
)
ax.annotate(
    f"Max gap: {abs(max_gap):,.1f} ({abs(max_gap) / episode.budget:.2%})\n"
    f"{direction} at {hours[max_index]:.2f} h",
    xy=(hours[max_index], (target[max_index] + actual[max_index]) / 2),
    xytext=(max(0.5, hours[max_index] - 7), target[max_index] * 0.72),
    arrowprops={"arrowstyle": "->", "color": "#dc2626"},
    color="#991b1b",
)
ax.annotate(
    f"Final gap: {final_gap:,.1f} ({final_gap / episode.budget:.3%})\n"
    f"Recovered {recovered_fraction:.1%} of peak gap",
    xy=(hours[-1], actual[-1]),
    xytext=(16.2, actual[-1] * 0.83),
    arrowprops={"arrowstyle": "->", "color": "#2563eb"},
    color="#1e3a8a",
)
ax.set_title(
    "B + state-dependent PI — validation day 100063\n"
    f"Budget {episode.budget:,.1f}  |  mean tracking error "
    f"{score['tracking_error']:.3%}  |  final unspent {score['unspent']:.3%}",
    loc="left",
    pad=12,
)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Cumulative spend")
ax.set_xlim(0, 24.15)
ax.set_ylim(bottom=0)
ax.legend(loc="upper left", frameon=False)
fig.tight_layout()
fig.savefig(HERE / "cumulative-spend.png", dpi=180, bbox_inches="tight")
plt.close(fig)

print(json.dumps(summary, ensure_ascii=False, indent=2))
