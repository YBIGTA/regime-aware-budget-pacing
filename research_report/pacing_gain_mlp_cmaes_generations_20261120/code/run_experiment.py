"""E2 합산 CMA-ES의 60/120/240세대 비교 실험.

원본 pacing_gain_mlp_v2를 읽기 전용으로 재사용한다. seed마다 한 번만 240세대까지
학습하고 60/120/240세대에서 동일 경로의 체크포인트를 저장한다.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import random
import shutil
import sys
import time
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
sys.path.insert(0, str(ORIGINAL / "code"))
import train_gain_mlp as base  # noqa: E402

RESULTS = BUNDLE / "results"
LOGS = BUNDLE / "logs"
CHECKPOINTS = RESULTS / "checkpoints"
for _path in (RESULTS, LOGS, CHECKPOINTS):
    _path.mkdir(parents=True, exist_ok=True)

SEEDS = (1, 2, 3)
TARGETS = (60, 120, 240)
POP = 12
TRAIN_DAYS = base.TRAIN
HOLDOUT_DAYS = base.HOLDOUT
VALID_DAYS = base.VALID
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = base.lib.BOOTSTRAP_SEED
EPS = 1e-12

assert base.MODEL == "B"
assert base.POP == POP
assert base.WORKERS == 10
assert base.cfg.multiplier_max == 1.0
assert len(TRAIN_DAYS) == 80 and len(HOLDOUT_DAYS) == 40 and len(VALID_DAYS) == 200


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_json(path: Path):
    return json.loads(path.read_text())


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def fmt_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def append_csv(path: Path, row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row), lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerow(row)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"쓸 행이 없습니다: {path}")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def record_extra(row: dict) -> None:
    """단계별 추가 실행 집계를 재실행해도 중복 없이 보존한다."""
    path = RESULTS / "extra_execution.csv"
    rows = []
    if path.exists():
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
    rows = [old for old in rows if (old["stage"], str(old["seed"])) != (row["stage"], str(row["seed"]))]
    rows.append(row)
    write_csv(path, rows)


def load_status() -> dict:
    path = RESULTS / "status.json"
    if path.exists():
        return read_json(path)
    return {
        "experiment": BUNDLE.name,
        "started_at": now_iso(),
        "stage": "준비",
        "workers": base.WORKERS,
        "seeds": {str(seed): {"generation": 0, "state": "대기"} for seed in SEEDS},
    }


def update_status(stage: str, **extra) -> None:
    status = load_status()
    status.update({"stage": stage, "updated_at": now_iso(), **extra})
    atomic_json(RESULTS / "status.json", status)


def progress_lines(seed: int, generation: int, gen_min: float, best_loss: float,
                   elapsed: float, seed_eta: float) -> list[str]:
    candidate1 = (
        "추가후보 1 · 120세대 목표 · 완료"
        if generation >= 120 else
        f"추가후보 1 · 120세대 목표 · {generation}/120세대 · {generation / 120:.1%}"
    )
    candidate2 = (
        "추가후보 2 · 240세대 목표 · 완료"
        if generation >= 240 else
        f"추가후보 2 · 240세대 목표 · {generation}/240세대 · {generation / 240:.1%}"
    )
    return [
        f"seed {seed}/3",
        candidate1,
        candidate2,
        f"현재 세대 최고 학습 손실 {gen_min:.6%} / 지금까지 최고 학습 손실 {best_loss:.6%}",
        f"경과 시간 {fmt_duration(elapsed)} / seed 예상 남은 시간 {fmt_duration(seed_eta)}",
    ]


def publish_progress(seed: int, generation: int, gen_min: float, best_loss: float,
                     elapsed: float, seed_eta: float) -> None:
    lines = progress_lines(seed, generation, gen_min, best_loss, elapsed, seed_eta)
    text = "\n".join(lines) + "\n"
    print("\n" + text, flush=True)
    with (LOGS / f"seed{seed}.log").open("a") as f:
        f.write(f"[{now_iso()}]\n{text}\n")
    with (LOGS / "progress.log").open("a") as f:
        f.write(f"[{now_iso()}]\n{text}\n")
    atomic_text(RESULTS / "status.txt", text)
    status = load_status()
    status["stage"] = "학습"
    status["updated_at"] = now_iso()
    status["active_seed"] = seed
    status["seeds"][str(seed)] = {
        "generation": generation,
        "state": "완료" if generation == 240 else "학습 중",
        "best_loss": best_loss,
        "generation_min": gen_min,
        "elapsed_seconds": elapsed,
        "eta_seconds": seed_eta,
    }
    atomic_json(RESULTS / "status.json", status)


def spec_for(weights, anchor, behind_input=True):
    return ("state", tuple(anchor), tuple(weights), behind_input)


def preflight() -> None:
    update_status("사전 점검")
    started = time.time()
    stored = base.lib.stored_scores(base.STATE_NAME)
    state = base.B_STATE
    report = {
        "started_at": now_iso(),
        "workspace": str(WORKSPACE),
        "original_bundle": str(ORIGINAL),
        "workers": base.WORKERS,
        "cpu_logical": os.cpu_count(),
        "python": sys.version,
        "numpy": np.__version__,
        "training_script_sha256": sha256(ORIGINAL / "code/train_gain_mlp.py"),
        "original_anchor_sha256": sha256(ORIGINAL / "results/anchor.json"),
        "original_e2_sum_s1_sha256": sha256(ORIGINAL / "results/model_E2_sum_s1.json"),
        "multiplier_max": base.cfg.multiplier_max,
        "anti_windup_implementation": "원본 StatePI.act 재사용",
        "rollouts": 244,
    }
    with base.pool() as pool:
        original_spec = spec_for(state["weights"], state["anchor"], False)
        reproduced = base.evaluate(pool, {"original": original_spec}, VALID_DAYS[:3])["original"]
        rows = []
        for day in VALID_DAYS[:3]:
            objective = reproduced[day][0] + reproduced[day][1]
            rows.append({
                "day": day,
                "tape_match": reproduced[day][3] == stored[day]["tape_sha256"],
                "abs_diff_objective": abs(objective - float(stored[day]["objective"])),
            })
        report["reproduce_state_pi"] = rows
        if not all(row["tape_match"] and row["abs_diff_objective"] < 1e-10 for row in rows):
            raise RuntimeError("기존 B_state_PI 재현 점검 실패")

        w7 = np.asarray(state["weights"])
        w8 = np.concatenate([w7[:28], np.zeros(4), w7[28:]])
        eight = base.evaluate(pool, {"eight": spec_for(w8, state["anchor"], True)}, VALID_DAYS[:1])["eight"]
        report["eight_input_zero_weight_equal"] = eight[VALID_DAYS[0]][:2] == reproduced[VALID_DAYS[0]][:2]
        if not report["eight_input_zero_weight_equal"]:
            raise RuntimeError("E2 8입력 영가중치 점검 실패")

        specs = {
            f"candidate_{i}": spec_for(
                w7 + np.random.default_rng(i).normal(0, 0.1, w7.size), state["anchor"], False
            )
            for i in range(POP)
        }
        throughput_start = time.time()
        base.evaluate(pool, specs, TRAIN_DAYS[:20])
        throughput_seconds = time.time() - throughput_start
        report["throughput_seconds_240_rollouts"] = throughput_seconds
        report["throughput_rollouts_per_second"] = 240 / throughput_seconds
        report["estimated_seconds_per_240_generations"] = throughput_seconds * 4 * 240

    report["seconds"] = time.time() - started
    report["finished_at"] = now_iso()
    atomic_json(RESULTS / "preflight.json", report)
    update_status("사전 점검 완료", preflight=report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def checkpoint_path(seed: int) -> Path:
    return CHECKPOINTS / f"resume_seed{seed}.pkl"


def save_checkpoint(seed: int, generation: int, es, best, start_metrics: dict,
                    elapsed: float, x0_rng_state: dict) -> None:
    payload = {
        "format_version": 1,
        "seed": seed,
        "generation": generation,
        "es": es,
        "best": best,
        "start_metrics": start_metrics,
        "elapsed_seconds": elapsed,
        "numpy_random_state": np.random.get_state(),
        "python_random_state": random.getstate(),
        "x0_rng_state": x0_rng_state,
        "saved_at": now_iso(),
    }
    path = checkpoint_path(seed)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if generation in TARGETS:
        shutil.copy2(path, CHECKPOINTS / f"seed{seed}_gen{generation:03d}.pkl")


def load_checkpoint(seed: int):
    path = checkpoint_path(seed)
    if not path.exists():
        return None
    with path.open("rb") as f:
        payload = pickle.load(f)
    if payload["seed"] != seed:
        raise RuntimeError(f"체크포인트 seed 불일치: {path}")
    np.random.set_state(payload["numpy_random_state"])
    random.setstate(payload["python_random_state"])
    return payload


def normalize_curve(seed: int, generation: int) -> None:
    path = RESULTS / f"curve_seed{seed}.csv"
    if not path.exists():
        return
    with path.open(newline="") as f:
        rows = [row for row in csv.DictReader(f) if int(row["generation"]) <= generation]
    if rows:
        write_csv(path, rows)
    else:
        path.unlink()


def save_model(seed: int, generation: int, es, best, start_metrics: dict, elapsed: float,
               anchor, te_ref: float) -> None:
    model = {
        "experiment": "E2",
        "objective": "sum",
        "seed": seed,
        "generation": generation,
        "popsize": POP,
        "sigma0": base.SIGMA0,
        "train_days": [TRAIN_DAYS[0], TRAIN_DAYS[-1], len(TRAIN_DAYS)],
        "behind_input": True,
        "anchor": list(anchor),
        "te_ref": te_ref,
        "start_train": start_metrics,
        "best_train_loss": float(best[0]),
        "best_train": best[2],
        "weights": np.asarray(best[1]).tolist(),
        "mean_weights": np.asarray(es.result.xfavorite).tolist(),
        "sigma": float(es.sigma),
        "elapsed_seconds": elapsed,
        "candidate_evaluations": generation * POP,
        "training_day_simulations": generation * POP * len(TRAIN_DAYS),
        "saved_at": now_iso(),
    }
    atomic_json(RESULTS / f"model_g{generation}_s{seed}.json", model)
    if generation == 60 and seed in (1, 2):
        old = read_json(ORIGINAL / f"results/model_E2_sum_s{seed}.json")
        comparison = {
            "seed": seed,
            "generation": 60,
            "best_loss_abs_diff": abs(model["best_train_loss"] - old["best_train_loss"]),
            "weights_max_abs_diff": float(np.max(np.abs(np.asarray(model["weights"]) - np.asarray(old["weights"])))),
            "mean_weights_max_abs_diff": float(np.max(np.abs(np.asarray(model["mean_weights"]) - np.asarray(old["mean_weights"])))),
        }
        comparison["exact_path_reproduced"] = all(comparison[key] < 1e-12 for key in (
            "best_loss_abs_diff", "weights_max_abs_diff", "mean_weights_max_abs_diff"
        ))
        atomic_json(RESULTS / f"reproduce_original_60_s{seed}.json", comparison)
        if not comparison["exact_path_reproduced"]:
            raise RuntimeError(f"seed {seed}의 기존 60세대 경로를 재현하지 못했습니다: {comparison}")


def train_seed(seed: int) -> None:
    import cma

    anchor_info = read_json(ORIGINAL / "results/anchor.json")
    anchor = tuple(anchor_info["anchor"])
    te_ref = float(anchor_info["anchor_train"]["tracking_error"])
    checkpoint = load_checkpoint(seed)
    if checkpoint:
        generation = int(checkpoint["generation"])
        es = checkpoint["es"]
        best = checkpoint["best"]
        start_metrics = checkpoint["start_metrics"]
        elapsed_before = float(checkpoint["elapsed_seconds"])
        x0_rng_state = checkpoint["x0_rng_state"]
        normalize_curve(seed, generation)
        print(f"seed {seed}: {generation}세대 체크포인트에서 재개", flush=True)
    else:
        generation = 0
        rng = np.random.default_rng(1000 * seed + 3)
        x0 = np.concatenate([rng.normal(0, 0.5, 36), np.zeros(8)])
        x0_rng_state = rng.bit_generator.state
        es = cma.CMAEvolutionStrategy(x0, base.SIGMA0, {"popsize": POP, "seed": seed, "verbose": -9})
        setup_start = time.time()
        with base.pool() as pool:
            start_metrics = base.means(base.evaluate(
                pool, {"x0": spec_for(x0, anchor, True)}, TRAIN_DAYS
            )["x0"])
        setup_seconds = time.time() - setup_start
        best = (base.loss(start_metrics, "sum", te_ref), x0, start_metrics)
        elapsed_before = 0.0
        record_extra({
            "stage": "training_start_candidate",
            "seed": seed,
            "models": 1,
            "days": len(TRAIN_DAYS),
            "rollouts": len(TRAIN_DAYS),
            "seconds": setup_seconds,
        })
        save_checkpoint(seed, 0, es, best, start_metrics, 0.0, x0_rng_state)

    if generation >= 240:
        print(f"seed {seed}: 이미 240세대 완료", flush=True)
        return

    segment_start = time.time()
    with base.pool() as pool:
        for current in range(generation + 1, 241):
            xs = es.ask()
            specs = {i: spec_for(x, anchor, True) for i, x in enumerate(xs)}
            evaluated = base.evaluate(pool, specs, TRAIN_DAYS)
            metrics = [base.means(evaluated[i]) for i in range(len(xs))]
            losses = [base.loss(metric, "sum", te_ref) for metric in metrics]
            es.tell(xs, losses)
            best_index = int(np.argmin(losses))
            if losses[best_index] < best[0]:
                best = (losses[best_index], np.asarray(xs[best_index]), metrics[best_index])
            elapsed = elapsed_before + time.time() - segment_start
            seconds_per_gen = elapsed / current if current else 0.0
            eta = seconds_per_gen * (240 - current)
            row = {
                "generation": current,
                "best_loss": float(best[0]),
                "generation_min": float(losses[best_index]),
                "generation_median": float(np.median(losses)),
                "sigma": float(es.sigma),
                "elapsed_seconds": elapsed,
            }
            append_csv(RESULTS / f"curve_seed{seed}.csv", row)
            save_checkpoint(seed, current, es, best, start_metrics, elapsed, x0_rng_state)
            if current in TARGETS:
                save_model(seed, current, es, best, start_metrics, elapsed, anchor, te_ref)
            publish_progress(seed, current, losses[best_index], best[0], elapsed, eta)


def train_all() -> None:
    update_status("학습")
    started = time.time()
    for seed in SEEDS:
        train_seed(seed)
    seconds = time.time() - started
    atomic_json(RESULTS / "training_run.json", {
        "finished_at": now_iso(),
        "wall_seconds_this_invocation": seconds,
        "seeds": list(SEEDS),
        "targets": list(TARGETS),
        "candidate_evaluations_per_seed": 240 * POP,
        "training_day_simulations_per_seed": 240 * POP * len(TRAIN_DAYS),
        "training_day_simulations_total": len(SEEDS) * 240 * POP * len(TRAIN_DAYS),
    })
    update_status("학습 완료")


def holdout() -> None:
    update_status("후보 선택 40일")
    started = time.time()
    specs = {}
    metadata = {}
    for generation in TARGETS:
        for seed in SEEDS:
            model = read_json(RESULTS / f"model_g{generation}_s{seed}.json")
            for weight_kind in ("weights", "mean_weights"):
                key = f"g{generation}_s{seed}|{weight_kind}"
                specs[key] = spec_for(model[weight_kind], model["anchor"], True)
                metadata[key] = (generation, seed, weight_kind, model)
    with base.pool() as pool:
        evaluated = base.evaluate(pool, specs, HOLDOUT_DAYS)
    rows = []
    daily_rows = []
    for key, per_day in evaluated.items():
        generation, seed, weight_kind, model = metadata[key]
        summary = base.means(per_day)
        rows.append({
            "generation": generation,
            "seed": seed,
            "weights": weight_kind,
            "best_train_loss": model["best_train_loss"],
            **summary,
        })
        for day in HOLDOUT_DAYS:
            te, unspent, tv, tape_hash = per_day[day]
            daily_rows.append({
                "generation": generation, "seed": seed, "weights": weight_kind, "day": day,
                "tracking_error": te, "unspent": unspent, "objective": te + unspent,
                "multiplier_tv": tv, "tape_sha256": tape_hash,
            })
    rows.sort(key=lambda row: (row["generation"], row["seed"], row["objective"]))
    write_csv(RESULTS / "holdout_table.csv", rows)
    write_csv(RESULTS / "holdout_daily.csv", daily_rows)

    seed_summary = []
    for generation in TARGETS:
        for seed in SEEDS:
            candidates = [row for row in rows if row["generation"] == generation and row["seed"] == seed]
            seed_summary.append(min(candidates, key=lambda row: row["objective"]))
    write_csv(RESULTS / "holdout_seed_summary.csv", seed_summary)
    picks = {}
    for generation in TARGETS:
        candidates = [row for row in rows if row["generation"] == generation]
        best = min(candidates, key=lambda row: row["objective"])
        picks[str(generation)] = {
            "generation": generation,
            "seed": best["seed"],
            "weights": best["weights"],
            "holdout": {key: best[key] for key in ("tracking_error", "unspent", "objective", "multiplier_tv")},
        }
    atomic_json(RESULTS / "picks.json", picks)
    seconds = time.time() - started
    record_extra({
        "stage": "holdout_selection",
        "seed": "all",
        "models": len(specs),
        "days": len(HOLDOUT_DAYS),
        "rollouts": len(specs) * len(HOLDOUT_DAYS),
        "seconds": seconds,
    })
    update_status("후보 선택 완료", picks=picks)
    print(json.dumps(picks, ensure_ascii=False, indent=2), flush=True)


def bootstrap_rows(arrays: dict[str, dict[str, np.ndarray]]) -> list[dict]:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sample = rng.integers(0, len(VALID_DAYS), size=(BOOTSTRAP_SAMPLES, len(VALID_DAYS)))
    rows = []
    for generation in (120, 240):
        new_name = f"g{generation}"
        for metric in ("tracking_error", "unspent", "objective"):
            delta = arrays[new_name][metric] - arrays["g60"][metric]
            distribution = delta[sample].mean(axis=1)
            lo, hi = np.quantile(distribution, (0.025, 0.975))
            rows.append({
                "comparison": f"{generation}-60",
                "metric": metric,
                "mean_difference": float(delta.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "days_better": int((delta < -EPS).sum()),
                "days_worse": int((delta > EPS).sum()),
                "days_equal": int((np.abs(delta) <= EPS).sum()),
                "bootstrap_samples": BOOTSTRAP_SAMPLES,
            })
    return rows


def validate() -> None:
    update_status("검증 200일")
    started = time.time()
    picks = read_json(RESULTS / "picks.json")
    specs = {}
    for generation in TARGETS:
        pick = picks[str(generation)]
        model = read_json(RESULTS / f"model_g{generation}_s{pick['seed']}.json")
        specs[f"g{generation}"] = spec_for(model[pick["weights"]], model["anchor"], True)
    original_pick = read_json(ORIGINAL / "results/picks.json")["picks"]["sum"]
    original_model = read_json(ORIGINAL / f"results/model_{original_pick['model']}.json")
    specs["original_E2_sum"] = spec_for(
        original_model[original_pick["weights"]], original_model["anchor"], original_model["behind_input"]
    )

    with base.pool() as pool:
        evaluated = base.evaluate(pool, specs, VALID_DAYS)
    stored = base.lib.stored_scores(base.STATE_NAME)
    arrays = {}
    table = []
    daily_rows = []
    for name, per_day in evaluated.items():
        for day in VALID_DAYS:
            if per_day[day][3] != stored[day]["tape_sha256"]:
                raise RuntimeError(f"검증 경매 테이프 불일치: {name}, {day}")
        arrays[name] = {
            "tracking_error": np.array([per_day[day][0] for day in VALID_DAYS]),
            "unspent": np.array([per_day[day][1] for day in VALID_DAYS]),
            "multiplier_tv": np.array([per_day[day][2] for day in VALID_DAYS]),
        }
        arrays[name]["objective"] = arrays[name]["tracking_error"] + arrays[name]["unspent"]
        table.append({
            "model": name,
            **{metric: float(arrays[name][metric].mean())
               for metric in ("tracking_error", "unspent", "objective", "multiplier_tv")},
        })
        for index, day in enumerate(VALID_DAYS):
            daily_rows.append({
                "model": name, "day": day,
                "tracking_error": arrays[name]["tracking_error"][index],
                "unspent": arrays[name]["unspent"][index],
                "objective": arrays[name]["objective"][index],
                "multiplier_tv": arrays[name]["multiplier_tv"][index],
                "tape_sha256": per_day[day][3],
            })
    table.sort(key=lambda row: (row["model"] != "g60", row["model"]))
    write_csv(RESULTS / "validation_table.csv", table)
    write_csv(RESULTS / "validation_daily.csv", daily_rows)
    paired = bootstrap_rows(arrays)
    write_csv(RESULTS / "validation_paired.csv", paired)
    seconds = time.time() - started
    record_extra({
        "stage": "validation",
        "seed": "representatives",
        "models": len(specs),
        "days": len(VALID_DAYS),
        "rollouts": len(specs) * len(VALID_DAYS),
        "seconds": seconds,
    })
    update_status("검증 완료")
    for row in table:
        print(
            f"{row['model']:15s} 추종 {row['tracking_error']:.6%} 미소진 {row['unspent']:.6%} "
            f"합산 {row['objective']:.6%}", flush=True
        )


def svg_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def training_curve_svg() -> None:
    width, height = 1000, 620
    left, right, top, bottom = 105, 35, 70, 85
    series = {}
    for seed in SEEDS:
        with (RESULTS / f"curve_seed{seed}.csv").open(newline="") as f:
            series[seed] = [(int(row["generation"]), float(row["best_loss"]) * 100) for row in csv.DictReader(f)]
    all_y = [value for values in series.values() for _, value in values]
    y_min, y_max = min(all_y), max(all_y)
    padding = max((y_max - y_min) * 0.08, 0.0005)
    y_min -= padding
    y_max += padding

    def x(value):
        return left + (value / 240) * (width - left - right)

    def y(value):
        return top + (y_max - value) / (y_max - y_min) * (height - top - bottom)

    colors = {1: "#2563eb", 2: "#dc2626", 3: "#059669"}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Apple SD Gothic Neo",sans-serif;fill:#111827}</style>',
        '<text x="500" y="35" text-anchor="middle" font-size="24" font-weight="700">세대별 최고 학습 손실 · E2 합산</text>',
    ]
    for tick in np.linspace(y_min, y_max, 6):
        yy = y(float(tick))
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-12}" y="{yy+5:.1f}" text-anchor="end" font-size="14">{tick:.4f}%</text>')
    for tick in (0, 60, 120, 180, 240):
        xx = x(tick)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{xx:.1f}" y="{height-bottom+28}" text-anchor="middle" font-size="14">{tick}</text>')
    for target in TARGETS:
        xx = x(target)
        parts.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" stroke="#6b7280" stroke-dasharray="5 5"/>')
    for seed, values in series.items():
        points = " ".join(f"{x(gen):.2f},{y(value):.2f}" for gen, value in values)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colors[seed]}" stroke-width="2.5"/>')
        lx = 690 + (seed - 1) * 90
        parts.append(f'<line x1="{lx}" y1="55" x2="{lx+24}" y2="55" stroke="{colors[seed]}" stroke-width="3"/>')
        parts.append(f'<text x="{lx+30}" y="60" font-size="14">seed {seed}</text>')
    parts.extend([
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#111827"/>',
        f'<text x="{(left+width-right)/2:.1f}" y="{height-25}" text-anchor="middle" font-size="16">CMA-ES 세대</text>',
        f'<text x="25" y="{(top+height-bottom)/2:.1f}" text-anchor="middle" font-size="16" transform="rotate(-90 25 {(top+height-bottom)/2:.1f})">최고 학습 손실 (%)</text>',
        '</svg>',
    ])
    atomic_text(RESULTS / "training_curve.svg", "\n".join(parts) + "\n")


def pct(value: float, digits: int = 6) -> str:
    return f"{value * 100:.{digits}f}%"


def pp(value: float, digits: int = 6) -> str:
    return f"{value * 100:+.{digits}f}%p"


def report() -> None:
    update_status("결과 정리")
    training_curve_svg()
    with (RESULTS / "holdout_seed_summary.csv").open(newline="") as f:
        holdout_rows = list(csv.DictReader(f))
    with (RESULTS / "validation_table.csv").open(newline="") as f:
        validation_rows = list(csv.DictReader(f))
    with (RESULTS / "validation_paired.csv").open(newline="") as f:
        paired_rows = list(csv.DictReader(f))
    with (RESULTS / "extra_execution.csv").open(newline="") as f:
        extra_rows = list(csv.DictReader(f))
    picks = read_json(RESULTS / "picks.json")
    validation = {row["model"]: row for row in validation_rows}

    lines = [
        "# CMA-ES 세대 수 비교 실험 결과",
        "",
        f"- 완료 시각: {now_iso()}",
        f"- 실행 위치: `{WORKSPACE}` (맥미니, 10코어)",
        "- 대상: B 선형회귀 지출 예측기 + E2 PI 계수 신경망, 합산 손실",
        "- 학습: seed 1·2·3, 후보 12개, 동일 경로로 240세대까지 실행 후 60/120/240 체크포인트",
        "- 데이터: 학습 80일(400000~400079), 후보 선택 40일(400080~400119), 검증 200일(100000~100199)",
        "- 최종 테스트 500일은 사용하지 않음",
        "",
        "## 조건·seed별 학습 및 후보 선택 40일",
        "",
        "각 seed에서 학습 중 최고 가중치와 CMA-ES 평균 가중치를 같은 40일에 평가하고, 합산 손실이 낮은 쪽을 적었다.",
        "",
        "| 세대 | seed | 학습 최고 손실 | 선택 가중치 | 40일 추종 오차 | 40일 미소진율 | 40일 합산 |",
        "|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in holdout_rows:
        lines.append(
            f"| {row['generation']} | {row['seed']} | {pct(float(row['best_train_loss']))} | "
            f"{row['weights']} | {pct(float(row['tracking_error']))} | {pct(float(row['unspent']))} | "
            f"{pct(float(row['objective']))} |"
        )
    lines.extend([
        "",
        "조건별 대표 모델은 40일 합산 손실 최저 후보로 정했다.",
        "",
        "| 조건 | 대표 seed | 가중치 | 40일 합산 |",
        "|---:|---:|---|---:|",
    ])
    for generation in TARGETS:
        pick = picks[str(generation)]
        lines.append(
            f"| {generation}세대 | {pick['seed']} | {pick['weights']} | {pct(pick['holdout']['objective'])} |"
        )

    lines.extend([
        "",
        "## 검증 200일",
        "",
        "| 모델 | 평균 추종 오차 | 최종 미소진율 | 합산 손실 |",
        "|---|---:|---:|---:|",
    ])
    order = ("g60", "g120", "g240", "original_E2_sum")
    labels = {"g60": "60세대", "g120": "120세대", "g240": "240세대", "original_E2_sum": "기존 E2 합산 참고"}
    for name in order:
        row = validation[name]
        lines.append(
            f"| {labels[name]} | {pct(float(row['tracking_error']))} | {pct(float(row['unspent']))} | "
            f"{pct(float(row['objective']))} |"
        )

    lines.extend([
        "",
        "## 60세대 대비 변화와 같은 날짜 짝비교",
        "",
        "차이는 추가 세대 모델 − 60세대 모델이다. 음수면 개선이다.",
        "",
        "| 비교 | 지표 | 평균 차이 | 95% 부트스트랩 CI | 개선/악화/동일 일수 |",
        "|---|---|---:|---:|---:|",
    ])
    metric_labels = {"tracking_error": "추종 오차", "unspent": "미소진율", "objective": "합산 손실"}
    for row in paired_rows:
        lines.append(
            f"| {row['comparison']}세대 | {metric_labels[row['metric']]} | {pp(float(row['mean_difference']))} | "
            f"[{pp(float(row['ci95_low']))}, {pp(float(row['ci95_high']))}] | "
            f"{row['days_better']}/{row['days_worse']}/{row['days_equal']} |"
        )

    g60 = validation["g60"]
    paired_lookup = {(row["comparison"], row["metric"]): row for row in paired_rows}
    lines.extend(["", "## 판단", ""])
    for generation in (120, 240):
        current = validation[f"g{generation}"]
        te_delta = float(current["tracking_error"]) - float(g60["tracking_error"])
        unspent_delta = float(current["unspent"]) - float(g60["unspent"])
        objective_delta = float(current["objective"]) - float(g60["objective"])
        if te_delta < -EPS and unspent_delta < -EPS:
            verdict = "두 주요 지표가 모두 개선됐다."
        elif te_delta > EPS or unspent_delta > EPS:
            worsened = []
            if te_delta > EPS:
                worsened.append("추종 오차")
            if unspent_delta > EPS:
                worsened.append("미소진율")
            verdict = f"주요 지표 중 {', '.join(worsened)}가 악화됐다."
        else:
            verdict = "두 주요 지표 중 적어도 하나는 실질적으로 동일했다."
        objective_pair = paired_lookup[(f"{generation}-60", "objective")]
        if float(objective_pair["ci95_high"]) < 0:
            generalization = "합산 손실의 95% 구간이 0 아래여서 별도 200일에서도 개선됐다고 판단한다."
        elif float(objective_pair["ci95_low"]) > 0:
            generalization = "합산 손실의 95% 구간이 0 위여서 별도 200일에서 악화됐다고 판단한다."
        else:
            generalization = "합산 손실의 95% 구간이 0을 포함해 별도 200일 개선은 확인되지 않았다."
        lines.append(
            f"- {generation}세대: 추종 {pp(te_delta)}, 미소진 {pp(unspent_delta)}, 합산 {pp(objective_delta)}. "
            f"{verdict} {generalization}"
        )

    target_times = {}
    for generation in TARGETS:
        target_times[generation] = sum(
            read_json(RESULTS / f"model_g{generation}_s{seed}.json")["elapsed_seconds"] for seed in SEEDS
        )
    actual_training_seconds = target_times[240]
    base_rollouts = len(SEEDS) * 60 * POP * len(TRAIN_DAYS)
    lines.extend([
        "",
        "## 실행 시간과 계산량",
        "",
        f"- 실제 연속 학습 시간(seed 누적): {fmt_duration(actual_training_seconds)}",
        f"- 60세대까지: {base_rollouts:,} 학습일 시뮬레이션, seed 누적 {fmt_duration(target_times[60])}",
        f"- 120세대까지: {len(SEEDS)*120*POP*len(TRAIN_DAYS):,}회, 60세대 대비 +{len(SEEDS)*60*POP*len(TRAIN_DAYS):,}회, seed 누적 {fmt_duration(target_times[120])}",
        f"- 240세대까지: {len(SEEDS)*240*POP*len(TRAIN_DAYS):,}회, 60세대 대비 +{len(SEEDS)*180*POP*len(TRAIN_DAYS):,}회, seed 누적 {fmt_duration(target_times[240])}",
        "- 같은 경로를 공유했으므로 60세대와 120세대를 별도로 다시 학습하지 않았다.",
        "",
        "추가 계산량 대비 검증 합산 손실 변화:",
        "",
    ])
    for generation in (120, 240):
        extra_rollouts = len(SEEDS) * (generation - 60) * POP * len(TRAIN_DAYS)
        objective_delta = float(validation[f"g{generation}"]["objective"]) - float(g60["objective"])
        relative = -objective_delta / float(g60["objective"]) * 100
        efficiency = -objective_delta * 100 * 100_000 / extra_rollouts
        lines.append(
            f"- {generation}세대: 추가 {extra_rollouts:,}회, 합산 {pp(objective_delta)}, "
            f"상대 개선률 {relative:+.3f}%, 추가 10만 회당 개선 {efficiency:+.6f}%p"
        )
    lines.extend([
        "",
        "후보 선택·검증 등 별도 실행:",
        "",
        "| 단계 | 모델 수 | 날짜 수 | 실행 수 | 시간 |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in extra_rows:
        lines.append(
            f"| {row['stage']} (seed {row['seed']}) | {row['models']} | {row['days']} | {row['rollouts']} | "
            f"{fmt_duration(float(row['seconds']))} |"
        )
    lines.extend([
        "",
        "## 산출물",
        "",
        "- `training_curve.svg`: seed별 세대 최고 학습 손실",
        "- `holdout_table.csv`, `holdout_daily.csv`: 후보 선택 40일 결과",
        "- `validation_table.csv`, `validation_daily.csv`: 검증 200일 결과",
        "- `validation_paired.csv`: 같은 날짜 짝비교와 20,000회 부트스트랩 CI",
        "- `checkpoints/`: 매 seed 재개 상태와 60/120/240세대 보존 체크포인트",
        "- `../logs/`: seed별 진행 로그와 전체 진행 로그",
        "",
    ])
    atomic_text(BUNDLE / "REPORT_KO.md", "\n".join(lines))
    update_status("완료", completed_at=now_iso(), report=str(BUNDLE / "REPORT_KO.md"))
    (BUNDLE / "DONE").touch()


def all_steps() -> None:
    if not (RESULTS / "preflight.json").exists():
        preflight()
    train_all()
    holdout()
    validate()
    report()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "all"
    commands = {
        "preflight": preflight,
        "train": train_all,
        "holdout": holdout,
        "validate": validate,
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
