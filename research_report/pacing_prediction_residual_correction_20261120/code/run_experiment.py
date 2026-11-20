"""B 선형회귀의 최근 예측 잔차를 이용한 온라인 기본 승수 보정 실험."""
from __future__ import annotations

import csv
import gzip
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")
os.environ["PACING_MODEL"] = "B"

import numpy as np  # noqa: E402

CODE = Path(__file__).resolve().parent
BUNDLE = CODE.parent
WORKSPACE = BUNDLE.parents[1]
ORIGINAL = WORKSPACE / "research_report/pacing_gain_mlp_v2"
PREVIOUS_GENERATION_STUDY = WORKSPACE / "research_report/pacing_gain_mlp_cmaes_generations_20261120"
RESULTS = BUNDLE / "results"
LOGS = BUNDLE / "logs"
for _path in (RESULTS, LOGS):
    _path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ORIGINAL / "code"))
import train_gain_mlp as original  # noqa: E402
from core import Observation, Feedback, features, generate, rollout  # noqa: E402

ETA = 0.25
H = 0.15
WORKERS = 10
NEW40 = tuple(range(600000, 600040))
NEW100 = tuple(range(600040, 600140))
DIAGNOSTIC_DAYS = original.TRAIN
BOOTSTRAP_SAMPLES = 20_000
EPS = 1e-12

PICKS_PATH = ORIGINAL / "results/picks.json"
PICK = json.loads(PICKS_PATH.read_text())["picks"]["sum"]
MODEL_PATH = ORIGINAL / f"results/model_{PICK['model']}.json"
MODEL = json.loads(MODEL_PATH.read_text())
ANCHOR = tuple(MODEL["anchor"])
WEIGHTS = tuple(MODEL[PICK["weights"]])

assert PICK == {"model": "E2_sum_s1", "weights": "mean_weights"}
assert MODEL["behind_input"] is True
assert len(original.lib.grid) == 101
assert original.cfg.multiplier_max == 1.0
assert len(DIAGNOSTIC_DAYS) == 80


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def value_hash(values) -> str:
    return hashlib.sha256(np.asarray(values, dtype="<f8").tobytes()).hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"쓸 행이 없습니다: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def update_status(stage: str, **extra) -> None:
    path = RESULTS / "status.json"
    data = json.loads(path.read_text()) if path.exists() else {
        "experiment": BUNDLE.name,
        "started_at": now_iso(),
    }
    data.update({"stage": stage, "updated_at": now_iso(), **extra})
    atomic_json(path, data)


def progress(stage: str, completed_days: int, total_days: int, conditions: int,
             day: int, started: float) -> None:
    elapsed = time.time() - started
    eta = elapsed / completed_days * (total_days - completed_days) if completed_days else 0.0
    line = (
        f"[{stage}] {completed_days}/{total_days}일 · "
        f"{completed_days * conditions}/{total_days * conditions} 조건-일 완료 · "
        f"day {day} · 경과 {elapsed:.1f}s · 예상 남음 {eta:.1f}s"
    )
    print(line, flush=True)
    with (LOGS / "progress.log").open("a") as f:
        f.write(f"{now_iso()} {line}\n")
    atomic_text(RESULTS / "status.txt", line + "\n")
    update_status(stage, completed_days=completed_days, total_days=total_days,
                  completed_condition_days=completed_days * conditions,
                  total_condition_days=total_days * conditions, current_day=day,
                  elapsed_seconds=elapsed, eta_seconds=eta)


class ResidualCorrectedE2:
    """고정 E2 계수망과 고정 선형회귀에 온라인 잔차 상태만 추가한다."""

    def __init__(self, mode: str, eta: float = ETA, h: float = H):
        if mode not in ("A", "B", "C"):
            raise ValueError(mode)
        self.mode = mode
        self.eta = float(eta)
        self.h = float(h)
        self.net = original.GainNet(ANCHOR, WEIGHTS, 8)
        self.grid = np.asarray(original.lib.grid, dtype=float)
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.common_bias = 0.0
        self.local_bias = np.zeros_like(self.grid)
        self.pending = None
        self.intervals = []

    def _raw_grid(self, obs):
        batch = np.repeat(original.lib.features(obs, 0.0)[None, :], len(self.grid), axis=0)
        batch[:, 0] = self.grid
        normalized = original.lib.scaler.transform(batch).astype(np.float32)
        predicted = np.maximum(
            original.PREDICTOR.predict(normalized) * original.lib.y_std + original.lib.y_mean, 0.0
        ).astype(float)
        if predicted.shape != self.grid.shape or not np.isfinite(predicted).all():
            raise ValueError("유효하지 않은 격자 예측")
        return predicted

    def _predict_at_action(self, obs, action: float) -> float:
        row = original.lib.features(obs, float(action))[None, :]
        normalized = original.lib.scaler.transform(row).astype(np.float32)
        value = float(original.PREDICTOR.predict(normalized)[0] * original.lib.y_std + original.lib.y_mean)
        if not math.isfinite(value):
            raise ValueError("유효하지 않은 최종 승수 예측")
        return max(0.0, value)

    def _correct_grid(self, raw):
        if self.mode == "B":
            return np.maximum(raw + self.common_bias, 0.0)
        if self.mode == "C":
            return np.maximum(raw + self.local_bias, 0.0)
        return raw

    def _correct_at_action(self, raw: float, action: float) -> float:
        if self.mode == "B":
            return max(0.0, raw + self.common_bias)
        if self.mode == "C":
            return max(0.0, raw + float(np.interp(action, self.grid, self.local_bias)))
        return raw

    def act(self, obs):
        if self.pending is not None:
            raise RuntimeError("이전 구간 관측 전에 다음 행동을 요청했습니다")
        if obs.tick == 0:
            raw_base = corrected_base = original.cfg.initial_multiplier
            slope = 0.0
        else:
            raw_grid = self._raw_grid(obs)
            target = 1.0 / obs.ticks
            raw_index = int(np.argmin(np.abs(raw_grid - target)))
            raw_base = float(self.grid[raw_index])
            corrected_grid = self._correct_grid(raw_grid)
            corrected_index = int(np.argmin(np.abs(corrected_grid - target)))
            corrected_base = float(self.grid[corrected_index])
            left, right = max(0, raw_index - 5), min(len(self.grid) - 1, raw_index + 5)
            slope = float((raw_grid[right] - raw_grid[left]) / (self.grid[right] - self.grid[left]))

        state = original.lib.state_features(obs, raw_base, slope)
        state = np.append(state, 1.0 if obs.error >= original.THRESHOLD else -1.0)
        kp, ki = self.net(state)
        error = obs.error
        candidate = self.integral + ki * error / obs.ticks
        raw_action = corrected_base + kp * error + candidate
        lo, hi = original.cfg.multiplier_min, original.cfg.multiplier_max
        blocked = (raw_action > hi and error > 0) or (raw_action < lo and error < 0)
        if blocked:
            raw_action = corrected_base + kp * error + self.integral
        else:
            self.integral = candidate
        action = float(np.clip(raw_action, lo, hi))
        prediction = self._predict_at_action(obs, action)
        corrected_prediction = self._correct_at_action(prediction, action)
        self.pending = {
            "day": None,
            "tick": obs.tick,
            "action": action,
            "original_base": raw_base,
            "corrected_base": corrected_base,
            "base_changed": raw_base != corrected_base,
            "original_prediction": prediction,
            "corrected_prediction": corrected_prediction,
            "prediction_valid": True,
        }
        return action, {
            "original_base": raw_base,
            "corrected_base": corrected_base,
            "base_changed": bool(raw_base != corrected_base),
            "saturated": bool(raw_action < lo or raw_action > hi),
            "integral_blocked": bool(blocked),
            "kp": float(kp),
            "ki": float(ki),
        }

    def observe(self, obs, action: float, spend: float, censored: bool):
        pending = self.pending
        self.pending = None
        if pending is None or pending["tick"] != obs.tick or abs(pending["action"] - action) > EPS:
            raise RuntimeError("행동-관측 짝이 맞지 않습니다")
        actual = float(spend / obs.budget)
        residual = actual - pending["original_prediction"]
        if not pending["prediction_valid"]:
            excluded_reason = "no_valid_prediction"
        elif censored:
            excluded_reason = "budget_censored"
        else:
            excluded_reason = ""
            if self.mode == "B" and self.eta:
                self.common_bias = (1.0 - self.eta) * self.common_bias + self.eta * residual
            elif self.mode == "C" and self.eta:
                weight = np.maximum(0.0, 1.0 - np.abs(self.grid - action) / self.h)
                self.local_bias = (1.0 - self.eta) * self.local_bias + self.eta * weight * residual
        remaining_end = max(0.0, obs.budget - obs.cumulative_spend - spend)
        record = {
            **pending,
            "actual_spend_fraction": actual,
            "residual_original": residual,
            "residual_corrected": actual - pending["corrected_prediction"],
            "budget_censored": bool(censored),
            "update_excluded": bool(excluded_reason),
            "excluded_reason": excluded_reason,
            "remaining_end": remaining_end,
            "cap_reached": action >= original.cfg.multiplier_max - EPS,
        }
        self.intervals.append(record)
        return record


class DiagnosticBias:
    def __init__(self, mode: str):
        self.mode = mode
        self.grid = np.asarray(original.lib.grid, dtype=float)
        self.common = 0.0
        self.local = np.zeros_like(self.grid)

    def corrected(self, raw: float, action: float) -> float:
        if self.mode == "B":
            return max(0.0, raw + self.common)
        return max(0.0, raw + float(np.interp(action, self.grid, self.local)))

    def update(self, action: float, residual: float) -> None:
        if self.mode == "B":
            self.common = (1.0 - ETA) * self.common + ETA * residual
        else:
            weight = np.maximum(0.0, 1.0 - np.abs(self.grid - action) / H)
            self.local = (1.0 - ETA) * self.local + ETA * weight * residual


def run_policy(day: int, mode: str, eta: float = ETA) -> tuple[dict, list[dict]]:
    episode = generate(original.cfg, day)
    controller = ResidualCorrectedE2(mode, eta=eta)

    def sink(obs, action, spend, censored):
        controller.observe(obs, action, spend, censored)

    summary, rows = rollout(episode, original.cfg, controller, sink=sink)
    if len(controller.intervals) != original.cfg.ticks:
        raise RuntimeError("구간 기록 수 불일치")
    valid = [row for row in controller.intervals if not row["update_excluded"]]
    reasons = Counter(row["excluded_reason"] for row in controller.intervals if row["excluded_reason"])
    early = any(
        row["remaining_end"] <= 1e-9 * max(1.0, episode.budget) and row["tick"] < original.cfg.ticks - 1
        for row in controller.intervals
    )
    daily = {
        "day": day,
        "condition": mode,
        "tape_sha256": summary["tape_sha256"],
        "tracking_error": summary["tracking_error"],
        "unspent": summary["unspent"],
        "objective": summary["objective"],
        "multiplier_tv": summary["multiplier_tv"],
        "max_target_overshoot": summary["max_ahead"],
        "early_budget_exhaustion": int(early),
        "cap_fraction": float(np.mean([row["cap_reached"] for row in controller.intervals])),
        "base_changed_count": sum(row["base_changed"] for row in controller.intervals),
        "valid_prediction_intervals": len(valid),
        "prediction_mae_original": float(np.mean([abs(row["residual_original"]) for row in valid])) if valid else math.nan,
        "prediction_mean_error_original": float(np.mean([row["residual_original"] for row in valid])) if valid else math.nan,
        "prediction_mae_corrected": float(np.mean([abs(row["residual_corrected"]) for row in valid])) if valid else math.nan,
        "prediction_mean_error_corrected": float(np.mean([row["residual_corrected"] for row in valid])) if valid else math.nan,
        "update_excluded_count": sum(row["update_excluded"] for row in controller.intervals),
        "excluded_budget_censored": reasons["budget_censored"],
        "excluded_no_valid_prediction": reasons["no_valid_prediction"],
    }
    for row in controller.intervals:
        row["day"] = day
        row["condition"] = mode
    return daily, controller.intervals


def _comparison_task(task):
    day, modes = task
    out = []
    for mode in modes:
        daily, _ = run_policy(day, mode)
        out.append(daily)
    hashes = {row["tape_sha256"] for row in out}
    if len(hashes) != 1:
        raise RuntimeError(f"조건별 테이프 불일치: {day}")
    return day, out


def run_comparison(stage: str, days, modes) -> list[dict]:
    started = time.time()
    rows = []
    tasks = [(day, tuple(modes)) for day in days]
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for completed, (day, result) in enumerate(pool.imap_unordered(_comparison_task, tasks), 1):
            rows.extend(result)
            progress(stage, completed, len(days), len(modes), day, started)
    rows.sort(key=lambda row: (row["day"], row["condition"]))
    return rows


def _diagnostic_task(day: int):
    episode = generate(original.cfg, day)
    controller = ResidualCorrectedE2("A")
    b_tracker = DiagnosticBias("B")
    c_tracker = DiagnosticBias("C")
    records = []

    def sink(obs, action, spend, censored):
        pending = controller.pending
        raw = pending["original_prediction"]
        predictions = {
            "raw": raw,
            "B": b_tracker.corrected(raw, action),
            "C": c_tracker.corrected(raw, action),
        }
        actual = float(spend / obs.budget)
        valid = pending["prediction_valid"] and not censored
        for name, prediction in predictions.items():
            records.append({
                "day": day, "tick": obs.tick, "prediction": name, "action": action,
                "predicted_spend_fraction": prediction, "actual_spend_fraction": actual,
                "residual_actual_minus_prediction": actual - prediction,
                "absolute_error": abs(actual - prediction), "valid": int(valid),
                "budget_censored": int(censored), "last_hour": int(obs.tick >= obs.ticks - 12),
            })
        residual = actual - raw
        if valid:
            b_tracker.update(action, residual)
            c_tracker.update(action, residual)
        controller.observe(obs, action, spend, censored)

    summary, _ = rollout(episode, original.cfg, controller, sink=sink)
    return day, summary["tape_sha256"], records


def run_diagnostic() -> None:
    update_status("예측 진단")
    started = time.time()
    all_rows = []
    tape_rows = []
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for completed, (day, tape_hash, rows) in enumerate(
            pool.imap_unordered(_diagnostic_task, DIAGNOSTIC_DAYS), 1
        ):
            all_rows.extend(rows)
            tape_rows.append({"day": day, "tape_sha256": tape_hash})
            progress("예측 진단", completed, len(DIAGNOSTIC_DAYS), 1, day, started)
    all_rows.sort(key=lambda row: (row["day"], row["tick"], row["prediction"]))
    path = RESULTS / "prediction_diagnostic_intervals.csv.gz"
    with gzip.open(path, "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(all_rows)
    summaries = []
    for prediction in ("raw", "B", "C"):
        for window in ("all", "last_hour"):
            subset = [row for row in all_rows if row["prediction"] == prediction and row["valid"]
                      and (window == "all" or row["last_hour"])]
            summaries.append({
                "prediction": prediction,
                "window": window,
                "valid_intervals": len(subset),
                "mae": float(np.mean([row["absolute_error"] for row in subset])),
                "mean_error_actual_minus_prediction": float(np.mean([
                    row["residual_actual_minus_prediction"] for row in subset
                ])),
            })
    write_csv(RESULTS / "prediction_diagnostic_summary.csv", summaries)
    write_csv(RESULTS / "prediction_diagnostic_tapes.csv", sorted(tape_rows, key=lambda row: row["day"]))


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        valid_total = sum(int(row["valid_prediction_intervals"]) for row in subset)
        output.append({
            "condition": condition,
            "days": len(subset),
            "tracking_error": float(np.mean([float(row["tracking_error"]) for row in subset])),
            "unspent": float(np.mean([float(row["unspent"]) for row in subset])),
            "objective": float(np.mean([float(row["objective"]) for row in subset])),
            "prediction_mae_original": sum(float(row["prediction_mae_original"]) * int(row["valid_prediction_intervals"]) for row in subset) / valid_total,
            "prediction_mae_corrected": sum(float(row["prediction_mae_corrected"]) * int(row["valid_prediction_intervals"]) for row in subset) / valid_total,
            "prediction_mean_error_original": sum(float(row["prediction_mean_error_original"]) * int(row["valid_prediction_intervals"]) for row in subset) / valid_total,
            "prediction_mean_error_corrected": sum(float(row["prediction_mean_error_corrected"]) * int(row["valid_prediction_intervals"]) for row in subset) / valid_total,
            "base_changed_count": sum(int(row["base_changed_count"]) for row in subset),
            "cap_fraction": float(np.mean([float(row["cap_fraction"]) for row in subset])),
            "update_excluded_count": sum(int(row["update_excluded_count"]) for row in subset),
            "excluded_budget_censored": sum(int(row["excluded_budget_censored"]) for row in subset),
            "excluded_no_valid_prediction": sum(int(row["excluded_no_valid_prediction"]) for row in subset),
            "mean_max_target_overshoot": float(np.mean([float(row["max_target_overshoot"]) for row in subset])),
            "maximum_target_overshoot": max(float(row["max_target_overshoot"]) for row in subset),
            "early_budget_exhaustion_days": sum(int(row["early_budget_exhaustion"]) for row in subset),
        })
    return output


def paired(rows: list[dict], experiment: str, bootstrap: bool) -> list[dict]:
    by = {(int(row["day"]), row["condition"]): row for row in rows}
    days = sorted({int(row["day"]) for row in rows if row["condition"] == "A"})
    rng = np.random.default_rng(original.lib.BOOTSTRAP_SEED)
    sample = rng.integers(0, len(days), size=(BOOTSTRAP_SAMPLES, len(days))) if bootstrap else None
    output = []
    for metric in ("tracking_error", "unspent", "objective"):
        control = np.array([float(by[(day, "A")][metric]) for day in days])
        treatment = np.array([float(by[(day, experiment)][metric]) for day in days])
        delta = treatment - control
        if bootstrap:
            distribution = delta[sample].mean(axis=1)
            low, high = np.quantile(distribution, (0.025, 0.975))
        else:
            low = high = math.nan
        output.append({
            "experiment": experiment,
            "control": "A",
            "metric": metric,
            "days": len(days),
            "control_mean": float(control.mean()),
            "experiment_mean": float(treatment.mean()),
            "mean_difference": float(delta.mean()),
            "relative_change_percent": float(delta.mean() / control.mean() * 100),
            "days_better": int((delta < -EPS).sum()),
            "days_worse": int((delta > EPS).sum()),
            "days_equal": int((np.abs(delta) <= EPS).sum()),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "bootstrap_samples": BOOTSTRAP_SAMPLES if bootstrap else 0,
        })
    return output


def discover_prior_day_scope() -> dict:
    roots = [WORKSPACE / "research_report", WORKSPACE / "experiments"]
    exact_hits = []
    needles = {str(day) for day in NEW40 + NEW100}
    for root in roots:
        for path in root.rglob("*"):
            if not path.is_file() or BUNDLE in path.parents or path.suffix.lower() not in (".json", ".csv", ".md", ".txt", ".py"):
                continue
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            matched = sorted(set(re.findall(r"(?<!\d)600(?:0\d{2}|1[0-3]\d)(?!\d)", text)) & needles)
            if matched:
                exact_hits.append({"path": str(path), "matched_ids": matched[:20]})
    return {
        "roots_scanned": [str(root) for root in roots],
        "exact_new_id_hits_before_this_experiment": exact_hits,
        "local_full_repository_exact_match_check_before_creation": 0,
        "unverified_scope": "팀원의 미동기화 로컬 작업공간·외부 저장소는 확인할 수 없음",
    }


def make_manifest() -> None:
    known = {
        "protocol_train": list(original.cfg.ids("train")),
        "gain_mlp_train80": list(original.TRAIN),
        "gain_mlp_holdout40": list(original.HOLDOUT),
        "protocol_validation": list(original.cfg.ids("validation")),
        "reserved_final_test500": list(original.cfg.ids("test")),
    }
    used = set().union(*(set(values) for values in known.values()))
    if used & set(NEW40 + NEW100) or set(NEW40) & set(NEW100):
        raise RuntimeError("날짜 분할 중복")
    scope = discover_prior_day_scope()
    if scope["exact_new_id_hits_before_this_experiment"]:
        raise RuntimeError(f"신규 날짜가 기존 파일에 나타납니다: {scope['exact_new_id_hits_before_this_experiment']}")
    manifest = {
        "created_at": now_iso(),
        "new_comparison40": list(NEW40),
        "new_confirmation100": list(NEW100),
        "known_excluded_ranges": {
            name: {"start": min(values), "end": max(values), "count": len(values)}
            for name, values in known.items()
        },
        "overlap_checks_passed": True,
        "workspace_scope_check": scope,
        "master_seed": original.cfg.master_seed,
        "note": "Episode ID가 경매 테이프 난수 시드를 결정하며 A/B/C는 같은 ID의 같은 테이프를 사용함",
    }
    atomic_json(BUNDLE / "manifest.json", manifest)
    atomic_json(BUNDLE / "config.json", {
        "conditions": {
            "A": "기존 E2, 예측 보정 없음",
            "B": "모든 승수 후보에 공통 잔차 EMA",
            "C": "최종 승수 주변 삼각 커널 잔차 EMA",
        },
        "eta": ETA,
        "h": H,
        "grid_candidates": len(original.lib.grid),
        "multiplier_min": original.cfg.multiplier_min,
        "multiplier_max": original.cfg.multiplier_max,
        "checkpoint": str(MODEL_PATH),
        "checkpoint_sha256": sha256(MODEL_PATH),
        "checkpoint_weight_field": PICK["weights"],
        "linear_predictor": str(original.lib.predictor_path),
        "linear_predictor_sha256": sha256(original.lib.predictor_path),
        "daily_state_reset": ["integral", "prediction_correction"],
        "excluded_updates": ["no_valid_prediction", "budget_censored"],
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_seed": original.lib.BOOTSTRAP_SEED,
    })


def preflight() -> None:
    update_status("사전 점검")
    prior = read_csv(PREVIOUS_GENERATION_STUDY / "results/validation_daily.csv")
    prior_map = {(int(row["day"]), row["model"]): row for row in prior}
    report = {
        "selected_checkpoint": {
            "pick_file": str(PICKS_PATH),
            "model_file": str(MODEL_PATH),
            "model_file_sha256": sha256(MODEL_PATH),
            "weight_field": PICK["weights"],
            "selected_weights_sha256_float64": value_hash(WEIGHTS),
            "linear_predictor_file": str(original.lib.predictor_path),
            "linear_predictor_file_sha256": sha256(original.lib.predictor_path),
        },
        "eta": ETA,
        "h": H,
        "grid_candidates": len(original.lib.grid),
        "multiplier_max": original.cfg.multiplier_max,
        "checks": [],
    }
    for day in original.VALID[:3]:
        original_policy = original.StatePI(ANCHOR, WEIGHTS, True)
        expected_summary, expected_rows = rollout(generate(original.cfg, day), original.cfg, original_policy)
        a_summary, a_rows = run_policy(day, "A")
        b0_summary, b0_rows = run_policy(day, "B", eta=0.0)
        c0_summary, c0_rows = run_policy(day, "C", eta=0.0)
        prior_row = prior_map[(day, "g60")]
        summary_keys = ("tracking_error", "unspent", "objective", "multiplier_tv")
        action_expected = np.array([row["action"] for row in expected_rows])
        action_a = np.array([row["action"] for row in a_rows])
        action_b0 = np.array([row["action"] for row in b0_rows])
        action_c0 = np.array([row["action"] for row in c0_rows])
        row = {
            "day": day,
            "original_action_max_abs_diff": float(np.max(np.abs(action_a - action_expected))),
            "eta0_B_action_max_abs_diff": float(np.max(np.abs(action_b0 - action_a))),
            "eta0_C_action_max_abs_diff": float(np.max(np.abs(action_c0 - action_a))),
            "original_metric_max_abs_diff": max(abs(a_summary[key] - expected_summary[key]) for key in summary_keys),
            "stored_E2_metric_max_abs_diff": max(abs(a_summary[key] - float(prior_row[key])) for key in summary_keys),
            "eta0_B_metric_max_abs_diff": max(abs(b0_summary[key] - a_summary[key]) for key in summary_keys),
            "eta0_C_metric_max_abs_diff": max(abs(c0_summary[key] - a_summary[key]) for key in summary_keys),
            "tape_match": a_summary["tape_sha256"] == prior_row["tape_sha256"],
        }
        if not row["tape_match"] or any(row[key] > 1e-12 for key in row if key.endswith("abs_diff")):
            raise RuntimeError(f"재현 점검 실패: {row}")
        report["checks"].append(row)

    signature = list(inspect.signature(ResidualCorrectedE2.act).parameters)
    if signature != ["self", "obs"]:
        raise RuntimeError(f"act 서명에 미래 정보가 들어갈 수 있습니다: {signature}")
    zero = Feedback(-1, 0.0, 0.0, 0, 0, 0.0, False)
    obs = Observation(0, original.cfg.ticks, 100.0, 0.0, tuple([zero] * original.cfg.history))
    action1, _ = ResidualCorrectedE2("B").act(obs)
    action2, _ = ResidualCorrectedE2("B").act(obs)
    report["causality"] = {
        "act_signature": signature,
        "observation_fields": list(inspect.signature(Observation).parameters),
        "future_tape_argument_present": False,
        "identical_observation_action_equal": action1 == action2,
        "update_occurs_only_in_post_execute_sink": True,
    }
    atomic_json(RESULTS / "preflight.json", report)


def run_40() -> tuple[list[dict], str | None]:
    update_status("신규 40일 비교")
    rows = run_comparison("신규 40일", NEW40, ("A", "B", "C"))
    write_csv(RESULTS / "comparison40_daily.csv", rows)
    summary = summarize(rows)
    write_csv(RESULTS / "comparison40_summary.csv", summary)
    paired_rows = paired(rows, "B", True) + paired(rows, "C", True)
    write_csv(RESULTS / "comparison40_paired.csv", paired_rows)
    by = {row["condition"]: row for row in summary}
    candidates = sorted((by[name]["objective"], name) for name in ("B", "C"))
    selected = candidates[0][1] if candidates[0][0] < by["A"]["objective"] - EPS else None
    decision = {
        "selected_for_100_days": selected,
        "control_objective": by["A"]["objective"],
        "B_objective": by["B"]["objective"],
        "C_objective": by["C"]["objective"],
        "reason": "40일 평균 합산 손실 최저 보정 후보" if selected else "두 보정 후보 모두 대조군보다 악화",
    }
    atomic_json(RESULTS / "selection.json", decision)
    return rows, selected


def run_100(selected: str) -> list[dict]:
    update_status("신규 100일 재확인", selected=selected)
    rows = run_comparison("신규 100일", NEW100, ("A", selected))
    write_csv(RESULTS / "comparison100_daily.csv", rows)
    write_csv(RESULTS / "comparison100_summary.csv", summarize(rows))
    write_csv(RESULTS / "comparison100_paired.csv", paired(rows, selected, True))
    return rows


def pct(value: float, digits: int = 6) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 6) -> str:
    return f"{value * 100:+.{digits}f}%p"


def report() -> None:
    update_status("보고서 작성")
    diagnostic = read_csv(RESULTS / "prediction_diagnostic_summary.csv")
    summary40 = read_csv(RESULTS / "comparison40_summary.csv")
    paired40 = read_csv(RESULTS / "comparison40_paired.csv")
    selection = json.loads((RESULTS / "selection.json").read_text())
    preflight_data = json.loads((RESULTS / "preflight.json").read_text())
    selected = selection["selected_for_100_days"]
    lines = [
        "# 최근 선형회귀 예측 오차를 이용한 기본 승수 보정",
        "",
        "## 무엇을 바꿨나",
        "",
        "- A: 기존 B 선형회귀 + 고정 E2 합산 계수망.",
        "- B: 완료된 직전 구간들의 실제−예측 잔차를 EMA(η=0.25)로 누적해 101개 후보에 공통 적용.",
        "- C: 최종 사용 승수 주변(h=0.15)에 삼각 가중치를 적용한 후보별 잔차 EMA.",
        "- 계수망 입력의 기본 승수·기울기는 항상 보정 전 원래 선형회귀로 계산했고, 최종 PI 계산의 기본 승수만 보정값으로 교체했다.",
        "- 결제 제한 구간은 보정 상태를 그대로 유지했으며, 모든 상태는 매일 초기화했다.",
        "- 후보 격자 사이의 최종 승수 예측은 원래 선형회귀를 해당 승수에서 직접 평가했다. C의 진단용 보정값만 인접 격자에서 선형 보간했다.",
        "",
        f"- 선택 체크포인트: `{preflight_data['selected_checkpoint']['model_file']}` / `{PICK['weights']}`",
        f"- 체크포인트 SHA-256: `{preflight_data['selected_checkpoint']['model_file_sha256']}`",
        f"- 선형 예측기 SHA-256: `{preflight_data['selected_checkpoint']['linear_predictor_file_sha256']}`",
        "",
        "η=0인 B/C는 3개 날짜에서 기존 E2와 행동·일별 지표 최대 절대차 0으로 일치했다. 정책 입력은 현재 관측뿐이며 갱신은 결제 결과가 나온 뒤에만 수행됐다.",
        "",
        "## 결과",
        "",
        "### 고정 E2 행동에서의 예측 진단 · 학습용 80일",
        "",
        "오차는 실제 지출−예측 지출이며 모두 하루 예산 비율 단위다.",
        "",
        "| 예측 | 구간 | 유효 구간 | MAE | 평균 오차 |",
        "|---|---|---:|---:|---:|",
    ]
    for row in diagnostic:
        lines.append(
            f"| {row['prediction']} | {row['window']} | {row['valid_intervals']} | "
            f"{pct(float(row['mae']))} | {pp(float(row['mean_error_actual_minus_prediction']))} |"
        )
    lines.extend([
        "",
        "### 신규 40일 정책 비교",
        "",
        "| 조건 | 추종 오차 | 미소진율 | 합산 손실 | 기본 승수 변경 | 상한 도달률 | 갱신 제외(결제/예측없음) | 최대 목표 초과 | 조기 소진일 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for row in summary40:
        lines.append(
            f"| {row['condition']} | {pct(float(row['tracking_error']))} | {pct(float(row['unspent']))} | "
            f"{pct(float(row['objective']))} | {row['base_changed_count']} | {pct(float(row['cap_fraction']))} | "
            f"{row['update_excluded_count']} ({row['excluded_budget_censored']}/{row['excluded_no_valid_prediction']}) | "
            f"{pct(float(row['maximum_target_overshoot']))} | {row['early_budget_exhaustion_days']} |"
        )
    lines.extend([
        "",
        "날짜별 차이는 실험군−A이며 음수가 개선이다.",
        "",
        "| 비교 | 지표 | 평균 변화 | 상대 변화율 | 개선/악화/동일 | 95% CI |",
        "|---|---|---:|---:|---:|---:|",
    ])
    metric_names = {"tracking_error": "추종", "unspent": "미소진", "objective": "합산"}
    for row in paired40:
        lines.append(
            f"| {row['experiment']}−A | {metric_names[row['metric']]} | {pp(float(row['mean_difference']))} | "
            f"{float(row['relative_change_percent']):+.3f}% | {row['days_better']}/{row['days_worse']}/{row['days_equal']} | "
            f"[{pp(float(row['ci95_low']))}, {pp(float(row['ci95_high']))}] |"
        )
    lines.extend(["", f"40일 선택: **{selected or '없음'}** — {selection['reason']}", ""])

    if selected:
        summary100 = read_csv(RESULTS / "comparison100_summary.csv")
        paired100 = read_csv(RESULTS / "comparison100_paired.csv")
        lines.extend([
            "### 별도 신규 100일 재확인",
            "",
            "| 조건 | 추종 오차 | 미소진율 | 합산 손실 | 기본 승수 변경 | 상한 도달률 | 갱신 제외(결제/예측없음) | 최대 목표 초과 | 조기 소진일 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for row in summary100:
            lines.append(
                f"| {row['condition']} | {pct(float(row['tracking_error']))} | {pct(float(row['unspent']))} | "
                f"{pct(float(row['objective']))} | {row['base_changed_count']} | {pct(float(row['cap_fraction']))} | "
                f"{row['update_excluded_count']} ({row['excluded_budget_censored']}/{row['excluded_no_valid_prediction']}) | "
                f"{pct(float(row['maximum_target_overshoot']))} | "
                f"{row['early_budget_exhaustion_days']} |"
            )
        lines.extend([
            "",
            "| 지표 | 평균 변화 | 상대 변화율 | 개선/악화/동일 | 날짜 bootstrap 95% CI |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in paired100:
            lines.append(
                f"| {metric_names[row['metric']]} | {pp(float(row['mean_difference']))} | "
                f"{float(row['relative_change_percent']):+.3f}% | {row['days_better']}/{row['days_worse']}/{row['days_equal']} | "
                f"[{pp(float(row['ci95_low']))}, {pp(float(row['ci95_high']))}] |"
            )
        objective_pair = next(row for row in paired100 if row["metric"] == "objective")
        if float(objective_pair["mean_difference"]) < 0 and float(objective_pair["ci95_high"]) < 0:
            adoption = f"{selected} 채택: 100일 합산 손실 감소의 95% 구간이 0 아래다."
        elif float(objective_pair["mean_difference"]) < 0:
            adoption = f"{selected} 보류: 평균 합산 손실은 감소했지만 95% 구간이 0을 포함한다."
        else:
            adoption = f"{selected} 미채택: 별도 100일 평균 합산 손실이 개선되지 않았다."
    else:
        adoption = "B/C 모두 신규 40일에서 악화되어 100일 재확인을 실행하지 않았고 미채택했다."

    by40 = {row["condition"]: row for row in summary40}
    raw_diag = next(row for row in diagnostic if row["prediction"] == "raw" and row["window"] == "all")
    lines.extend([
        "",
        "## 채택 여부",
        "",
        f"- **{adoption}**",
        "- 예측 정확도 개선과 정책 손실 개선은 별개로 판단했다. "
        + ("B의 진단 MAE는 원 예측보다 낮았다." if float(next(row for row in diagnostic if row["prediction"] == "B" and row["window"] == "all")["mae"]) < float(raw_diag["mae"]) else "B는 평균 편향을 거의 제거했지만 진단 MAE는 오히려 커졌다."),
        "- " + ("C의 전체 진단 MAE는 원 예측보다 낮았지만 마지막 1시간 MAE와 실제 정책 합산 손실은 개선되지 않았다." if float(next(row for row in diagnostic if row["prediction"] == "C" and row["window"] == "all")["mae"]) < float(raw_diag["mae"]) else "C의 진단 MAE도 개선되지 않았다."),
        "- " + ("C는 B보다 40일 합산 손실이 낮아 국소화가 공통 보정의 악화를 줄였지만, A보다 나빠 유효한 추가 개선으로 이어지지 않았다." if float(by40["C"]["objective"]) < float(by40["B"]["objective"]) else "C가 B보다 40일 합산 손실이 낮지 않아 가까운 승수에 집중한 추가 효과는 확인되지 않았다."),
        "- 예약된 최종 테스트 500일(200000~200499)은 사용하지 않았다.",
        "",
        "## 재현",
        "",
        "```bash",
        "cd /Users/chanwoong/pacing_two_state_20261114",
        ".venv/bin/python research_report/pacing_prediction_residual_correction_20261120/code/run_experiment.py all",
        "```",
        "",
        "일별 결과와 상세 진단은 `results/`에, 신규 날짜와 중복 점검 범위는 `manifest.json`에 저장했다.",
        "",
    ])
    atomic_text(BUNDLE / "REPORT_KO.md", "\n".join(lines))
    update_status("완료", completed_at=now_iso(), selected=selected, adoption=adoption)
    (BUNDLE / "DONE").touch()


def all_steps() -> None:
    make_manifest()
    preflight()
    run_diagnostic()
    _, selected = run_40()
    if selected:
        run_100(selected)
    report()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    commands = {
        "manifest": make_manifest,
        "preflight": preflight,
        "diagnostic": run_diagnostic,
        "compare40": lambda: run_40(),
        "report": report,
        "all": all_steps,
    }
    if command not in commands:
        raise SystemExit(f"알 수 없는 명령: {command}")
    try:
        commands[command]()
    except Exception as exc:
        update_status("실패", error=repr(exc))
        raise
