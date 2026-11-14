import ast
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / "research_report/pacing_abc_state_pi/analysis/day_100063/analyze.py"
ALPHAS = (0.1, 0.2, 0.3, 0.5)


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
scaler = ns["scaler"]
y_mean = ns["y_mean"]
y_std = ns["y_std"]
linear = ns["linear"]


def predict_spend(obs, action):
    inputs = core.features(obs, action)[None, :]
    normalized = scaler.transform(inputs).astype(np.float32)
    fraction = float(linear.predict(normalized)[0] * y_std + y_mean)
    return max(fraction, 0.0) * obs.budget


class DecayPI:
    def __init__(self, mode, alpha):
        self.mode = mode
        self.alpha = alpha
        self.schedule = ns["GainMLP"](state["anchor"], state["weights"])

    def reset(self):
        self.integral = 0.0

    def standard_step(self, obs, base, kp, ki):
        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw_candidate = base + kp * error + candidate
        blocked = (raw_candidate > 1.0 and error > 0) or (
            raw_candidate < 0.0 and error < 0
        )
        integral = self.integral if blocked else candidate
        raw = base + kp * error + integral
        action = float(np.clip(raw, 0.0, 1.0))
        return integral, raw, action, blocked, raw_candidate

    def act(self, obs):
        base, slope = ns["base_and_slope"](obs)
        kp, ki = self.schedule(ns["state_features"](obs, base, slope))
        integral_before = self.integral
        standard_i, standard_raw, standard_action, blocked, raw_candidate = (
            self.standard_step(obs, base, kp, ki)
        )

        decayed_i = standard_i
        decayed_raw = standard_raw
        decayed_action = standard_action
        predicted_spend = None
        required_spend = (obs.tick + 1) * obs.budget / obs.ticks - obs.cumulative_spend
        prediction_allows_decay = False
        skipped_because_behind = self.mode == "predictive_gated" and obs.error > 0

        if standard_i > 0 and self.mode != "control":
            decayed_i = (1 - self.alpha) * standard_i
            decayed_raw = base + kp * obs.error + decayed_i
            decayed_action = float(np.clip(decayed_raw, 0.0, 1.0))
            if self.mode in ("predictive", "predictive_gated") and not skipped_because_behind:
                predicted_spend = predict_spend(obs, decayed_action)
                prediction_allows_decay = predicted_spend >= required_spend

        decay_triggered = standard_i > 0 and (
            self.mode == "fixed"
            or (
                self.mode in ("predictive", "predictive_gated")
                and prediction_allows_decay
            )
        )
        if decay_triggered:
            self.integral = decayed_i
            raw = decayed_raw
            action = decayed_action
        else:
            self.integral = standard_i
            raw = standard_raw
            action = standard_action

        return action, {
            "base": base,
            "raw": float(raw),
            "raw_candidate": float(raw_candidate),
            "integral_before": float(integral_before),
            "integral_standard": float(standard_i),
            "integral": float(self.integral),
            "integral_blocked": bool(blocked),
            "saturated": bool(raw < 0.0 or raw > 1.0),
            "kp": float(kp),
            "ki": float(ki),
            "decay_triggered": bool(decay_triggered),
            "predicted_spend_after_decay": predicted_spend,
            "required_spend": float(required_spend),
            "prediction_allows_decay": bool(prediction_allows_decay),
            "skipped_because_behind": bool(skipped_because_behind),
        }


def rollout(episode, mode, alpha):
    controller = DecayPI(mode, alpha)
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
        action, debug = controller.act(obs)
        cost, wins, censored = core.execute(
            tick, action, max(0.0, episode.budget - spent)
        )
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
                "error_before": obs.error,
                "error": (tick_no + 1) / cfg.ticks - spent / episode.budget,
                **asdict(feedback),
                **debug,
            }
        )

    errors = np.asarray([row["error"] for row in rows])
    actions = np.asarray([row["action"] for row in rows])
    score = {
        "day": episode.day,
        "tape_sha256": episode.digest(),
        "tracking_error": float(np.abs(errors).mean()),
        "unspent": max(0.0, 1 - spent / episode.budget),
        "objective": float(np.abs(errors).mean()) + max(0.0, 1 - spent / episode.budget),
        "multiplier_tv": float(np.abs(np.diff(actions)).mean()),
        "max_ahead": float(np.maximum(-errors, 0).max()),
        "decay_ticks": sum(row["decay_triggered"] for row in rows),
    }
    return score, rows


def clock(tick, end=False):
    minute = (tick + int(end)) * cfg.interval
    return f"{minute // 60:02d}:{minute % 60:02d}"


def ranges(ticks):
    ticks = sorted(set(ticks))
    if not ticks:
        return []
    result = []
    start = previous = ticks[0]
    for tick in ticks[1:]:
        if tick == previous + 1:
            previous = tick
            continue
        result.append(f"{clock(start)}-{clock(previous, end=True)}")
        start = previous = tick
    result.append(f"{clock(start)}-{clock(previous, end=True)}")
    return result


def inspect_day(day):
    episode = ns["generate"](cfg, day)
    output = {}
    for mode, alpha in [("control", 0.0)] + [
        (mode, alpha) for alpha in ALPHAS for mode in ("fixed", "predictive")
    ]:
        score, rows = rollout(episode, mode, alpha)
        triggers = [row for row in rows if row["decay_triggered"]]
        max_index = int(np.argmax([-row["error"] for row in rows]))
        first_crossing = next(
            (index for index, row in enumerate(rows) if row["error"] < 0), None
        )
        recovery_crossing = next(
            (index for index in range(168, len(rows)) if rows[index]["error"] < 0),
            None,
        )
        recovery_triggers = [row for row in triggers if row["tick"] >= 168]
        first = triggers[0] if triggers else None
        key = "control" if mode == "control" else f"{mode}_{alpha}"
        output[key] = {
            "score": score,
            "trigger_count": len(triggers),
            "trigger_ranges": ranges([row["tick"] for row in triggers]),
            "first_trigger": None
            if first is None
            else {
                "interval": f"{clock(first['tick'])}-{clock(first['tick'], end=True)}",
                "integral_before": first["integral_before"],
                "integral_standard": first["integral_standard"],
                "integral_after": first["integral"],
                "action": first["action"],
                "predicted_spend_after_decay": first[
                    "predicted_spend_after_decay"
                ],
                "required_spend": first["required_spend"],
            },
            "first_target_crossing": None
            if first_crossing is None
            else clock(first_crossing, end=True),
            "recovery_crossing": None
            if recovery_crossing is None
            else {
                "time": clock(recovery_crossing, end=True),
                "overshoot_amount": max(
                    -rows[recovery_crossing]["error"] * episode.budget, 0.0
                ),
                "integral": rows[recovery_crossing]["integral"],
                "action": rows[recovery_crossing]["action"],
                "base": rows[recovery_crossing]["base"],
                "p_term": rows[recovery_crossing]["kp"]
                * rows[recovery_crossing]["error_before"],
                "decay_triggered": rows[recovery_crossing]["decay_triggered"],
            },
            "first_recovery_decay": None
            if not recovery_triggers
            else {
                "interval": f"{clock(recovery_triggers[0]['tick'])}-{clock(recovery_triggers[0]['tick'], end=True)}",
                "integral_before": recovery_triggers[0]["integral_before"],
                "integral_after": recovery_triggers[0]["integral"],
                "action": recovery_triggers[0]["action"],
            },
            "max_overshoot": {
                "time": clock(max_index, end=True),
                "fraction_of_budget": score["max_ahead"],
                "amount": score["max_ahead"] * episode.budget,
                "integral": rows[max_index]["integral"],
                "action": rows[max_index]["action"],
            },
            "window_14_00_to_19_50": {
                "integral_start": rows[168]["integral"],
                "integral_end": rows[237]["integral"],
                "cap_ticks": int(
                    sum(
                        np.isclose(row["action"], 1.0, rtol=0, atol=1e-12)
                        for row in rows[168:238]
                    )
                ),
                "decay_ticks": int(
                    sum(row["decay_triggered"] for row in rows[168:238])
                ),
            },
        }
    return output


def validation_experiment():
    baseline = {}
    with ns["daily_scores_path"].open(newline="") as file:
        for row in csv.DictReader(file):
            if row["model"] == "B_state_PI":
                baseline[int(row["day"])] = row

    configurations = [("control", 0.0)] + [
        (mode, alpha) for alpha in ALPHAS for mode in ("fixed", "predictive")
    ]
    scores = {configuration: [] for configuration in configurations}
    tape_mismatches = []
    started = time.perf_counter()

    for day in cfg.ids("validation"):
        episode = ns["generate"](cfg, day)
        for mode, alpha in configurations:
            score, _ = rollout(episode, mode, alpha)
            scores[mode, alpha].append(score)
            if mode == "control" and score["tape_sha256"] != baseline[day]["tape_sha256"]:
                tape_mismatches.append(day)

    result = {}
    for mode, alpha in configurations:
        values = scores[mode, alpha]
        key = "control" if mode == "control" else f"{mode}_{alpha}"
        result[key] = {
            metric: float(np.mean([score[metric] for score in values]))
            for metric in (
                "max_ahead",
                "tracking_error",
                "unspent",
                "objective",
                "multiplier_tv",
                "decay_ticks",
            )
        }

    control = result["control"]
    eligible = []
    for key, value in result.items():
        if key == "control":
            continue
        value["tracking_error_delta"] = value["tracking_error"] - control["tracking_error"]
        value["unspent_delta"] = value["unspent"] - control["unspent"]
        value["max_ahead_delta"] = value["max_ahead"] - control["max_ahead"]
        if (
            value["tracking_error"] <= control["tracking_error"] + 1e-12
            and value["unspent"] <= control["unspent"] + 1e-12
        ):
            eligible.append(key)

    recommendation = min(eligible, key=lambda key: result[key]["max_ahead"]) if eligible else None
    return {
        "days": cfg.validation_days,
        "tape_mismatches": tape_mismatches,
        "runtime_seconds": time.perf_counter() - started,
        "results": result,
        "eligible_without_mean_tracking_or_unspent_worsening": eligible,
        "recommendation": recommendation,
    }


def summarize_trigger_group(events, total_count):
    prediction_errors = np.asarray(
        [event["predicted_spend"] - event["actual_spend"] for event in events]
    )
    normalized_errors = np.asarray(
        [
            (event["predicted_spend"] - event["actual_spend"])
            / event["budget"]
            for event in events
        ]
    )
    reached = sum(event["reached_next_target"] for event in events)
    return {
        "count": len(events),
        "share_of_all_triggers": len(events) / total_count,
        "mean_predicted_spend": float(
            np.mean([event["predicted_spend"] for event in events])
        ),
        "mean_actual_spend": float(
            np.mean([event["actual_spend"] for event in events])
        ),
        "mean_prediction_minus_actual": float(prediction_errors.mean()),
        "mae": float(np.abs(prediction_errors).mean()),
        "mean_normalized_prediction_minus_actual": float(normalized_errors.mean()),
        "normalized_mae": float(np.abs(normalized_errors).mean()),
        "reached_next_target_count": reached,
        "reached_next_target_rate": reached / len(events),
    }


def diagnose_predictive_alpha_0_1():
    baseline = {}
    with ns["daily_scores_path"].open(newline="") as file:
        for row in csv.DictReader(file):
            if row["model"] == "B_state_PI":
                baseline[int(row["day"])] = row

    events = []
    tape_mismatches = []
    started = time.perf_counter()
    for day in cfg.ids("validation"):
        episode = ns["generate"](cfg, day)
        score, rows = rollout(episode, "predictive", 0.1)
        if score["tape_sha256"] != baseline[day]["tape_sha256"]:
            tape_mismatches.append(day)
        for row in rows:
            if not row["decay_triggered"]:
                continue
            predicted = row["predicted_spend_after_decay"]
            if predicted is None:
                raise AssertionError("Triggered predictive decay has no prediction")
            events.append(
                {
                    "day": day,
                    "tick": row["tick"],
                    "hour": row["tick"] * cfg.interval // 60,
                    "budget": episode.budget,
                    "behind_before": row["error_before"] > 0,
                    "predicted_spend": predicted,
                    "actual_spend": row["spend"],
                    "reached_next_target": row["error"] <= 0,
                    "shortfall_after": max(row["error"] * episode.budget, 0.0),
                }
            )

    behind = [event for event in events if event["behind_before"]]
    at_or_ahead = [event for event in events if not event["behind_before"]]
    false_recovery = [
        event for event in behind if not event["reached_next_target"]
    ]
    shortfalls = np.asarray([event["shortfall_after"] for event in false_recovery])

    hourly = []
    for hour in range(24):
        hour_behind = [event for event in behind if event["hour"] == hour]
        hour_false = [event for event in false_recovery if event["hour"] == hour]
        hourly.append(
            {
                "hour": hour,
                "behind_trigger_count": len(hour_behind),
                "false_recovery_count": len(hour_false),
                "false_recovery_rate": len(hour_false) / len(hour_behind)
                if hour_behind
                else None,
                "total_shortfall": float(
                    sum(event["shortfall_after"] for event in hour_false)
                ),
            }
        )

    six_hour_blocks = []
    for start in (0, 6, 12, 18):
        block = [
            event for event in false_recovery if start <= event["hour"] < start + 6
        ]
        six_hour_blocks.append(
            {
                "time": f"{start:02d}:00-{start + 6:02d}:00",
                "count": len(block),
                "share": len(block) / len(false_recovery),
                "total_shortfall": float(
                    sum(event["shortfall_after"] for event in block)
                ),
            }
        )

    window_counts = []
    for start in range(22):
        count = sum(
            event["hour"] in range(start, start + 3) for event in false_recovery
        )
        window_counts.append((count, start))
    peak_count, peak_start = max(window_counts)

    return {
        "days": cfg.validation_days,
        "tape_mismatches": tape_mismatches,
        "runtime_seconds": time.perf_counter() - started,
        "total_triggers": len(events),
        "mean_triggers_per_day": len(events) / cfg.validation_days,
        "groups": {
            "behind_before": summarize_trigger_group(behind, len(events)),
            "at_or_ahead_before": summarize_trigger_group(at_or_ahead, len(events)),
        },
        "behind_predicted_to_recover_but_failed": {
            "count": len(false_recovery),
            "share_of_behind_triggers": len(false_recovery) / len(behind),
            "affected_days": len({event["day"] for event in false_recovery}),
            "total_shortfall": float(shortfalls.sum()),
            "mean_shortfall": float(shortfalls.mean()),
            "median_shortfall": float(np.median(shortfalls)),
            "max_shortfall": float(shortfalls.max()),
            "mean_shortfall_fraction_of_budget": float(
                np.mean(
                    [
                        event["shortfall_after"] / event["budget"]
                        for event in false_recovery
                    ]
                )
            ),
            "mean_prediction_minus_actual": float(
                np.mean(
                    [
                        event["predicted_spend"] - event["actual_spend"]
                        for event in false_recovery
                    ]
                )
            ),
            "peak_three_hour_window": {
                "time": f"{peak_start:02d}:00-{peak_start + 3:02d}:00",
                "count": peak_count,
                "share": peak_count / len(false_recovery),
            },
            "six_hour_blocks": six_hour_blocks,
            "hourly": hourly,
        },
    }


def compare_gated_predictive_alpha_0_1():
    baseline = {}
    with ns["daily_scores_path"].open(newline="") as file:
        for row in csv.DictReader(file):
            if row["model"] == "B_state_PI":
                baseline[int(row["day"])] = row

    configurations = (
        ("control", 0.0),
        ("predictive", 0.1),
        ("predictive_gated", 0.1),
    )
    scores = {configuration: [] for configuration in configurations}
    tape_mismatches = []
    maximum_control_metric_difference = 0.0
    started = time.perf_counter()

    for day in cfg.ids("validation"):
        episode = ns["generate"](cfg, day)
        for mode, alpha in configurations:
            score, _ = rollout(episode, mode, alpha)
            scores[mode, alpha].append(score)
            if score["tape_sha256"] != baseline[day]["tape_sha256"]:
                tape_mismatches.append((day, mode))
            if mode == "control":
                for metric in (
                    "tracking_error",
                    "unspent",
                    "objective",
                    "multiplier_tv",
                ):
                    maximum_control_metric_difference = max(
                        maximum_control_metric_difference,
                        abs(score[metric] - float(baseline[day][metric])),
                    )

    metrics = ("max_ahead", "tracking_error", "unspent", "decay_ticks")
    names = {
        ("control", 0.0): "control",
        ("predictive", 0.1): "predictive_0.1",
        ("predictive_gated", 0.1): "predictive_gated_0.1",
    }
    summaries = {}
    arrays = {}
    for configuration in configurations:
        key = names[configuration]
        arrays[key] = {
            metric: np.asarray(
                [score[metric] for score in scores[configuration]], dtype=float
            )
            for metric in metrics
        }
        summaries[key] = {
            metric: float(arrays[key][metric].mean()) for metric in metrics
        }

    rng = np.random.default_rng(cfg.bootstrap_seed)
    samples = rng.integers(cfg.validation_days, size=(2000, cfg.validation_days))
    comparisons = {}
    for new, old in (
        ("predictive_gated_0.1", "predictive_0.1"),
        ("predictive_gated_0.1", "control"),
    ):
        comparison = {}
        for metric in metrics[:-1]:
            delta = arrays[new][metric] - arrays[old][metric]
            sampled_means = delta[samples].mean(axis=1)
            lo, hi = np.quantile(sampled_means, (0.025, 0.975))
            comparison[metric] = {
                "mean_delta": float(delta.mean()),
                "ci95_mean_delta": [float(lo), float(hi)],
                "improved_days": int(np.sum(delta < -1e-12)),
                "worsened_days": int(np.sum(delta > 1e-12)),
                "tied_days": int(np.sum(np.abs(delta) <= 1e-12)),
            }
        comparisons[f"{new}_minus_{old}"] = comparison

    return {
        "days": cfg.validation_days,
        "tape_mismatches": tape_mismatches,
        "maximum_control_metric_difference": maximum_control_metric_difference,
        "runtime_seconds": time.perf_counter() - started,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def main():
    if "--compare-gated-alpha-0.1" in sys.argv:
        print(
            json.dumps(
                {
                    "gated_predictive_alpha_0.1": compare_gated_predictive_alpha_0_1()
                },
                ensure_ascii=False,
            )
        )
        return
    if "--diagnose-alpha-0.1" in sys.argv:
        print(
            json.dumps(
                {"predictive_alpha_0.1_diagnostic": diagnose_predictive_alpha_0_1()},
                ensure_ascii=False,
            )
        )
        return
    day_result = inspect_day(100103)
    print(json.dumps({"day_100103": day_result}, ensure_ascii=False))
    validation = validation_experiment()
    print(json.dumps({"validation_200_days": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
