import ast
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SOURCE = REPO / "research_report/pacing_abc_state_pi/analysis/day_100063/analyze.py"
MODES = ("fixed", "state", "time", "time_and_spend")


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
frozen = ns["b_frozen"]
fixed_gains = tuple(frozen["B_fixed_PI"])
state = frozen["B_state_PI"]
state_start_tick = cfg.ticks - 120 // cfg.interval


class GainSelectionPI:
    def __init__(self, mode):
        self.mode = mode
        self.schedule = ns["GainMLP"](state["anchor"], state["weights"])

    def reset(self):
        self.integral = 0.0

    def choose_state_gains(self, obs):
        if self.mode == "state":
            return True
        if self.mode == "time":
            return obs.tick >= state_start_tick
        if self.mode == "time_and_spend":
            return obs.tick >= state_start_tick and obs.error > 0
        return False

    def act(self, obs):
        base, slope = ns["base_and_slope"](obs)
        use_state = self.choose_state_gains(obs)
        if use_state:
            kp, ki = self.schedule(ns["state_features"](obs, base, slope))
        else:
            kp, ki = fixed_gains

        integral_before = self.integral
        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw = base + kp * error + candidate
        blocked = (raw > 1.0 and error > 0) or (raw < 0.0 and error < 0)
        if blocked:
            raw = base + kp * error + self.integral
        else:
            self.integral = candidate

        return float(np.clip(raw, 0.0, 1.0)), {
            "base": base,
            "raw": float(raw),
            "integral_before": float(integral_before),
            "integral": float(self.integral),
            "integral_blocked": bool(blocked),
            "saturated": bool(raw < 0.0 or raw > 1.0),
            "kp": float(kp),
            "ki": float(ki),
            "use_state_gains": bool(use_state),
        }


def run_day(day, mode):
    episode = ns["generate"](cfg, day)
    score, trace = core.rollout(episode, cfg, GainSelectionPI(mode))
    score["state_gain_ticks"] = sum(row["use_state_gains"] for row in trace)
    return score, trace


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


def inspect_transition_day(day=100103):
    result = {}
    for mode in MODES:
        score, trace = run_day(day, mode)
        transition_rows = []
        previous_mode = trace[0]["use_state_gains"]
        maximum_continuity_residual = 0.0
        for index, row in enumerate(trace):
            expected_integral = 0.0 if index == 0 else trace[index - 1]["integral"]
            residual = row["integral_before"] - expected_integral
            maximum_continuity_residual = max(
                maximum_continuity_residual, abs(residual)
            )
            if index and row["use_state_gains"] != previous_mode:
                transition_rows.append(
                    {
                        "interval": f"{clock(row['tick'])}-{clock(row['tick'], end=True)}",
                        "from": "state" if previous_mode else "fixed",
                        "to": "state" if row["use_state_gains"] else "fixed",
                        "error_before": row["error_before"],
                        "integral_previous_tick": trace[index - 1]["integral"],
                        "integral_before": row["integral_before"],
                        "integral_after": row["integral"],
                        "kp": row["kp"],
                        "ki": row["ki"],
                    }
                )
            previous_mode = row["use_state_gains"]

        result[mode] = {
            "score": score,
            "state_gain_ticks": score["state_gain_ticks"],
            "state_gain_ranges": ranges(
                [row["tick"] for row in trace if row["use_state_gains"]]
            ),
            "transitions": transition_rows,
            "maximum_integral_continuity_residual": maximum_continuity_residual,
        }
    return result


def paired_comparison(new_values, base_values, samples):
    delta = new_values - base_values
    sampled_means = delta[samples].mean(axis=1)
    lo, hi = np.quantile(sampled_means, (0.025, 0.975))
    return {
        "mean_delta": float(delta.mean()),
        "ci95_mean_delta": [float(lo), float(hi)],
        "improved_days": int(np.sum(delta < -1e-12)),
        "worsened_days": int(np.sum(delta > 1e-12)),
        "tied_days": int(np.sum(np.abs(delta) <= 1e-12)),
    }


def validation_experiment():
    stored = {}
    with ns["daily_scores_path"].open(newline="") as file:
        for row in csv.DictReader(file):
            if row["model"] in ("B_fixed_PI", "B_state_PI"):
                stored[row["model"], int(row["day"])] = row

    scores = {mode: [] for mode in MODES}
    tape_mismatches = []
    maximum_stored_metric_difference = 0.0
    started = time.perf_counter()
    for day in cfg.ids("validation"):
        expected_tape = stored["B_state_PI", day]["tape_sha256"]
        for mode in MODES:
            score, _ = run_day(day, mode)
            scores[mode].append(score)
            if score["tape_sha256"] != expected_tape:
                tape_mismatches.append((day, mode))
            stored_model = (
                "B_fixed_PI"
                if mode == "fixed"
                else "B_state_PI"
                if mode == "state"
                else None
            )
            if stored_model:
                for metric in (
                    "tracking_error",
                    "unspent",
                    "objective",
                    "multiplier_tv",
                ):
                    maximum_stored_metric_difference = max(
                        maximum_stored_metric_difference,
                        abs(
                            score[metric]
                            - float(stored[stored_model, day][metric])
                        ),
                    )

    metrics = ("tracking_error", "unspent", "state_gain_ticks")
    arrays = {
        mode: {
            metric: np.asarray([score[metric] for score in scores[mode]], dtype=float)
            for metric in metrics
        }
        for mode in MODES
    }
    summaries = {
        mode: {metric: float(values.mean()) for metric, values in arrays[mode].items()}
        for mode in MODES
    }

    rng = np.random.default_rng(cfg.bootstrap_seed)
    samples = rng.integers(cfg.validation_days, size=(2000, cfg.validation_days))
    comparisons = {}
    for new, base in (("time_and_spend", "time"), ("time_and_spend", "state")):
        comparisons[f"{new}_minus_{base}"] = {
            metric: paired_comparison(
                arrays[new][metric], arrays[base][metric], samples
            )
            for metric in ("tracking_error", "unspent")
        }

    return {
        "days": cfg.validation_days,
        "fixed_gains": {"kp": fixed_gains[0], "ki": fixed_gains[1]},
        "state_start_tick": state_start_tick,
        "state_start_time": clock(state_start_tick),
        "tape_mismatches": tape_mismatches,
        "maximum_stored_metric_difference": maximum_stored_metric_difference,
        "runtime_seconds": time.perf_counter() - started,
        "summaries": summaries,
        "comparisons": comparisons,
    }


def main():
    if "--inspect-day" in sys.argv:
        print(json.dumps({"day_100103": inspect_transition_day()}, ensure_ascii=False))
        return
    print(json.dumps({"validation_200_days": validation_experiment()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
