"""Frozen twelve-model final evaluation. No fitting, tuning or candidate selection.

preflight uses archived development/confirmation days only. final explicitly
opens 200000..200499 after persisting protocol and access marker. Historical
episode guards are never changed. Every process generates one common tape/day.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[key] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BUNDLE = Path(__file__).resolve().parents[1]
OUT = BUNDLE / "results"
VERSION = "final-500-v1"
BASE = "d4e4296607a882a81bea227408f58a4e1361c9b2"
PRIMARY = ["tracking_error", "unspent", "multiplier_tv"]
METRICS = PRIMARY + ["objective", "max_ahead", "saturation_fraction"]
CASES = [200000, 200249, 200499]
BOOTSTRAPS = 20000
BOOT_SEED = 20261121
CTX = None


def now():
    return datetime.now(timezone.utc).isoformat()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def load_context():
    global CTX
    if CTX is not None:
        return CTX
    sys.path.insert(0, str(ROOT / "research_report/pacing_asymmetric_pi/code"))
    import env
    from controller import AsymmetricPI
    rt = env.rt
    sys.path.insert(0, str(ROOT / "research_report/pacing_ki_common/code"))
    from verify_export import verify
    integrity = verify()
    predictors = __import__("joblib").load(rt.STATE / "code/abc/predictors.joblib")
    definitions = {}
    source = rt.GAIN / "code/train_gain_mlp.py"
    for label, predictor in (("B", predictors[3]), ("C", predictors[4])):
        namespace = dict(np=np, lib=rt.lib, cfg=rt.cfg, PREDICTOR=predictor, THRESHOLD=.005)
        rt.definitions(source, ["base_and_slope", "GainNet", "StatePI", "FixedPI", "_fixed_act"], namespace)
        definitions[label] = namespace
    CTX = SimpleNamespace(rt=rt, env=env, cls=AsymmetricPI, definitions=definitions, integrity=integrity)
    return CTX


def specifications():
    c = load_context()
    rt = c.rt
    specs = {"A_PI": {"kind": "A", "gains": [rt.params["kp"], rt.params["ki"]]}}
    for label in ("B", "C"):
        frozen = read(rt.STATE / f"code/{label}/frozen_experiment2.json")
        specs[label + "_fixed_PI"] = {"kind": "fixed", "predictor": label, "gains": frozen[label + "_fixed_PI"]}
        state = frozen[label + "_state_PI"]
        assert state["outputs"] == 2
        specs[label + "_state_PI"] = {"kind": "state", "predictor": label,
            "anchor": state["anchor"], "weights": state["weights"], "behind_input": False}
        folder = rt.GAIN / ("results" if label == "B" else "results_C")
        picks = read(folder / "picks.json")["picks"]
        for objective in ("sum", "guard"):
            pick = picks[objective]
            model = read(folder / f"model_{pick['model']}.json")
            specs[f"{label}_CMA_{objective}"] = {"kind": "state", "predictor": label,
                "anchor": model["anchor"], "weights": model[pick["weights"]], "behind_input": True}
    parent = c.env.policies()
    assert specs["B_CMA_sum"]["weights"] == parent["CMA"]["weights"]
    assert specs["B_CMA_sum"]["anchor"] == parent["CMA"]["anchor"]
    for name in ("G", "CMA_A05", "G_A05"):
        specs[name] = {"kind": "asymmetric", "spec": parent[name]}
    assert len(specs) == 12
    return specs


def make_policy(spec):
    c = load_context()
    if spec["kind"] == "A":
        return c.rt.core.PI(c.rt.cfg, c.rt.core.Gains(*spec["gains"]))
    if spec["kind"] == "asymmetric":
        return c.cls(spec["spec"])
    namespace = c.definitions[spec["predictor"]]
    if spec["kind"] == "fixed":
        return namespace["FixedPI"](spec["gains"])
    return namespace["StatePI"](spec["anchor"], spec["weights"], spec["behind_input"])


def evaluate_day(day, specs, *, final=False):
    c = load_context()
    allowed = set(range(200000, 200500)) if final else {100000, 100199, 560000, 560999, 630000}
    if day not in allowed:
        raise ValueError(f"Date not authorized in this phase: {day}")
    if final and (not (OUT / "protocol.json").exists() or not (OUT / "TEST_OPENED.json").exists()):
        raise ValueError("Final-test protocol and access marker required before generation")
    ep = c.rt.core.generate(c.rt.cfg, day)
    tape = ep.digest()
    rows, traces = [], []
    for name, spec in specs.items():
        score, trace = c.rt.core.rollout(ep, c.rt.cfg, make_policy(spec))
        assert score["tape_sha256"] == tape
        assert all(np.isfinite(score[m]) for m in METRICS)
        row = {"model": name, **score,
            "remaining_pre_final": trace[-2]["remaining"] / ep.budget,
            "final_spend": trace[-1]["spend"] / ep.budget,
            "last2h_tracking_error": float(np.mean([abs(r["error"]) for r in trace[-24:]]))}
        assert abs(row["remaining_pre_final"] - row["final_spend"] - row["unspent"]) < 1e-10
        rows.append(row)
        if final and day in CASES:
            for r in trace:
                traces.append({"model": name, "day": day, "tick": r["tick"], "budget": ep.budget,
                    "target": r["target"], "cumulative_spend": r["cumulative_spend"],
                    "error": r["error"], "action": r["action"], "base": r.get("base"),
                    "kp": r.get("kp", spec.get("gains", [None, None])[0]),
                    "ki": r.get("ki", spec.get("gains", [None, None])[1]),
                    "factor": r.get("factor"), "target_rate": r.get("target_rate"),
                    "integral_blocked": r.get("integral_blocked"), "saturated": r["saturated"]})
    return {"day": day, "rows": rows, "traces": traces}


def preflight():
    c, specs = load_context(), specifications()
    assert list(c.rt.cfg.ids("test")) == list(range(200000, 200500))
    scores = {}
    for day in (100000, 100199, 560000, 560999):
        result = evaluate_day(day, specs)
        scores[day] = {r["model"]: r for r in result["rows"]}
    checks = 0
    tape_matches, tape_differences, max_metric_difference = 0, [], 0.
    golden = [(c.rt.STATE / "results/daily_scores.csv", {"A_PI": "A_PI", "B_fixed_PI": "B_fixed_PI",
               "C_fixed_PI": "C_fixed_PI", "B_state_PI": "B_state_PI", "C_state_PI": "C_state_PI"}, {100000, 100199}),
              (c.env.BUNDLE / "results/fresh_daily.csv", {"CMA": "B_CMA_sum", "G": "G",
               "CMA_A05": "CMA_A05", "G_A05": "G_A05"}, {560000, 560999})]
    for path, mapping, days in golden:
        with path.open(encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                day = int(row["day"])
                if day not in days or row["model"] not in mapping:
                    continue
                new = scores[day][mapping[row["model"]]]
                if new["tape_sha256"] == row["tape_sha256"]:
                    tape_matches += 1
                else:
                    tape_differences.append({"day": day, "model": row["model"],
                        "archived": row["tape_sha256"], "current": new["tape_sha256"]})
                for metric in PRIMARY + ["objective"]:
                    max_metric_difference = max(max_metric_difference, abs(new[metric] - float(row[metric])))
                    np.testing.assert_allclose(new[metric], float(row[metric]), rtol=0, atol=1e-9,
                                               err_msg=f"{day} {row['model']} {metric}")
                checks += 1
    assert checks == 18
    # Added input and all four selected CMA models use the archived definitions.
    for label in ("B", "C"):
        for objective in ("sum", "guard"):
            sp = specs[f"{label}_CMA_{objective}"]
            assert len(sp["weights"]) == 44
    # Future-tape perturbation must leave previous actions unchanged.
    from dataclasses import replace
    ep = c.rt.core.generate(c.rt.cfg, 630000)
    changed = replace(ep, tape=ep.tape[:50] + tuple(replace(t, competitor=t.competitor * 5) for t in ep.tape[50:]))
    for name in ("B_CMA_sum", "C_CMA_guard", "G_A05"):
        _, a = c.rt.core.rollout(ep, c.rt.cfg, make_policy(specs[name]))
        _, b = c.rt.core.rollout(changed, c.rt.cfg, make_policy(specs[name]))
        np.testing.assert_array_equal([r["action"] for r in a[:50]], [r["action"] for r in b[:50]])
    # The historical final-test guards remain sealed after our imports.
    for day in (200000, 200499):
        try:
            c.rt.episode(day)
        except ValueError:
            pass
        else:
            raise AssertionError("Historical seal changed")
    return {"status": "passed", "checked_utc": now(), "golden_policy_days": checks,
            "future_independence_models": 3, "models_loaded": len(specs), "final_test_used": False,
            "integrity": c.integrity, "bitwise_matching_archived_tapes": tape_matches,
            "archived_tape_hash_differences": tape_differences,
            "max_archived_metric_difference": max_metric_difference,
            "replay_tolerance": 1e-9,
            "portability": "Archived tape hashes may differ across platform math libraries. Metrics must agree within tolerance; final paired models always use exactly the same newly generated tape. No bitwise archive reproduction claim."}


def frozen_design():
    c = load_context()
    # Track all existing files in these source bundles, excluding runtime caches.
    folders = [ROOT / "experiments/pacing_abc", c.rt.STATE, c.rt.GAIN,
               ROOT / "research_report/pacing_ki_common", c.env.BUNDLE,
               ROOT / "research_report/pacing_deadline_pi", ROOT / "research_report/pacing_sensitivity_hypotheses"]
    paths = {Path(__file__).resolve()}
    for folder in folders:
        paths.update(p for p in folder.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and p.suffix not in (".pyc", ".tmp") and ".git" not in p.parts)
    paths.update([c.rt.ARCHIVE / "pacing_abc_ki_check_v3/frozen_A.json",
                  c.rt.ARCHIVE / "pacing_abc_strong_v2/common_data/protocol.json"])
    versions = {name: importlib.metadata.version(name) for name in
                ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
    return {"version": VERSION, "source_base": BASE, "config": asdict(c.rt.cfg),
        "days": list(range(200000, 200500)), "models": specifications(), "primary_model": "G_A05",
        "primary_baseline": "B_CMA_sum", "primary_metrics": PRIMARY,
        "primary_success": "All three mean differences below zero; simultaneous support requires all adjusted upper bounds below zero.",
        "primary_intervals": "Bonferroni across TE, UR, V: 98.3333333333% marginal percentile intervals",
        "other_intervals": "95% descriptive intervals; no model selection on this final set",
        "bootstrap": BOOTSTRAPS, "bootstrap_seed": BOOT_SEED, "trace_days": CASES,
        "bootstrap_draws": "Shared multinomial weights over paired whole days for all policies and metrics",
        "python": platform.python_version(), "platform": platform.platform(), "packages": versions,
        "sources": {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)},
        "scope": "One frozen model per stored pick; no retraining or training-seed robustness experiment. Uncalibrated synthetic in-distribution days."}


def worker(task):
    day, specs, design_hash = task
    result = evaluate_day(day, specs, final=True)
    result["design_sha256"] = design_hash
    return result


def lock_run():
    import fcntl
    path = ROOT / "outputs/pacing-final-20261121.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a")
    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return handle


def bootstrap_means(values):
    rng = np.random.default_rng(BOOT_SEED)
    means = np.empty((BOOTSTRAPS, values.shape[1]))
    for start in range(0, BOOTSTRAPS, 250):
        n = min(250, BOOTSTRAPS - start)
        weights = rng.multinomial(len(values), np.full(len(values), 1. / len(values)), size=n)
        means[start:start + n] = weights @ values / len(values)
    return means


def report(protocol):
    design = protocol["design"]
    names = list(design["models"])
    rows = []
    for day in design["days"]:
        result = read(OUT / f"checkpoints/day_{day}.json")
        validate_checkpoint(result, day, protocol["design_sha256"], names)
        rows.extend(result["rows"])
    rows.sort(key=lambda r: (names.index(r["model"]), r["day"]))
    write_csv(OUT / "daily.csv", rows)
    by_model = {name: [r for r in rows if r["model"] == name] for name in names}
    summary = []
    for name, rs in by_model.items():
        row = {"model": name, "days": len(rs)}
        for metric in METRICS + ["remaining_pre_final", "final_spend", "last2h_tracking_error"]:
            values = np.array([r[metric] for r in rs])
            row[metric] = float(values.mean())
            row[metric + "_p95"] = float(np.quantile(values, .95))
            row[metric + "_worst10_mean"] = float(np.sort(values)[-50:].mean())
        summary.append(row)
    write_csv(OUT / "summary.csv", summary)
    data = np.array([[by_model[name][i][metric] for name in names for metric in METRICS]
                     for i in range(len(design["days"]))])
    resampled = bootstrap_means(data).reshape(BOOTSTRAPS, len(names), len(METRICS))
    pairs = [(name, "A_PI") for name in names if name != "A_PI"]
    pairs += [(name, "B_CMA_sum") for name in names if name not in ("A_PI", "B_CMA_sum")]
    pairs += [("B_CMA_sum", "B_state_PI"), ("B_CMA_guard", "B_state_PI"),
              ("C_CMA_sum", "C_state_PI"), ("C_CMA_guard", "C_state_PI"),
              ("G_A05", "G"), ("G_A05", "CMA_A05")]
    comparisons = []
    for new, base in dict.fromkeys(pairs):
        for metric in METRICS:
            a = np.array([r[metric] for r in by_model[base]])
            b = np.array([r[metric] for r in by_model[new]])
            delta = b - a
            adjusted = (new, base) == ("G_A05", "B_CMA_sum") and metric in PRIMARY
            alpha = .05 / 3 if adjusted else .05
            sampled_delta = resampled[:, names.index(new), METRICS.index(metric)] - resampled[:, names.index(base), METRICS.index(metric)]
            low, high = (float(v) for v in np.quantile(sampled_delta, [alpha / 2, 1 - alpha / 2]))
            comparisons.append({"new": new, "baseline": base, "metric": metric,
                "difference": float(delta.mean()), "ci_low": low, "ci_high": high,
                "interval_level": 1 - .05 / 3 if adjusted else .95,
                "relative_reduction_pct": 100 * float(a.mean() - b.mean()) / float(a.mean()) if a.mean() else None,
                "improved_days": int((delta < -1e-12).sum()), "worsened_days": int((delta > 1e-12).sum()),
                "tied_days": int((np.abs(delta) <= 1e-12).sum()),
                "scope": "confirmatory" if adjusted else "descriptive"})
    write_csv(OUT / "paired.csv", comparisons)
    primary = [r for r in comparisons if r["scope"] == "confirmatory"]
    save(OUT / "conclusion.json", {"primary_model": "G_A05", "baseline": "B_CMA_sum",
        "all_three_means_reduced": all(r["difference"] < 0 for r in primary),
        "all_three_adjusted_upper_bounds_below_zero": all(r["ci_high"] < 0 for r in primary),
        "comparisons": primary, "candidate_changed_after_test": False, "final_test_used": True})
    lines = ["# 광고 예산 페이싱: 최종 500일 평가", "", "고정한 12개 모델을 동일한 경매 테이프에서 평가했다. 재학습·후보 재선택은 수행하지 않았다.",
        "", "|모델|추종 오차 (%)|미소진율 (%)|합산 손실 (%)|승수 변동|", "|---|---:|---:|---:|---:|"]
    for r in summary:
        lines.append(f"|{r['model']}|{100*r['tracking_error']:.6f}|{100*r['unspent']:.6f}|{100*r['objective']:.6f}|{r['multiplier_tv']:.6f}|")
    lines += ["", "## 사전 지정 주 비교: G_A05 − B_CMA_sum", "", "TE·UR 차이는 %p, V는 원 단위이다. 세 지표에 Bonferroni 보정한 구간을 사용한다.",
              "", "|지표|평균 차이|보정 구간|상대 감소율 (%)|", "|---|---:|---|---:|"]
    for r in primary:
        scale = 100 if r["metric"] != "multiplier_tv" else 1
        lines.append(f"|{r['metric']}|{scale*r['difference']:+.7f}|[{scale*r['ci_low']:+.7f}, {scale*r['ci_high']:+.7f}]|{r['relative_reduction_pct']:+.3f}|")
    lines += ["", "## 자료", "", "- daily.csv: 모델별·날짜별 성능과 경매 해시", "- summary.csv: 평균, 95백분위수, 최악 10% 평균",
              "- paired.csv: 같은 날짜 간 대응 비교 및 신뢰구간", "- traces/: 결과를 보기 전에 지정한 세 날짜의 5분 기록",
              "- protocol.json / TEST_OPENED.json: 모델 동결과 최종 테스트 접근 기록",
              "", "## 사전 재현 확인", "", f"기존 자료 18개 모델·날짜 조합의 지표 차이는 최대 {protocol['preflight']['max_archived_metric_difference']:.3g}였다.",
              f"이전 경매 테이프 해시가 다른 조합은 {len(protocol['preflight']['archived_tape_hash_differences'])}개였다. 플랫폼 간 비트 단위 재현과 수치 허용오차 내 지표 재현을 구분한다. 이번 최종 비교의 모든 모델은 날짜별로 동일한 테이프를 사용했다.",
              "", "## 해석 범위", "", "실제 시장 자료로 교정하지 않은 합성 first-price 환경이다. 학습 시드 반복 실험, CMA-ES와 동일 계산량 무작위 탐색의 비교는 포함하지 않았다.",
              "최종 결과를 본 뒤 같은 500일로 모델을 고르거나 수정하지 않는다. 다른 비교의 95% 구간은 기술적 비교이며 다중 비교를 보정하지 않았다.", ""]
    (OUT / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:len(summary) + 6]), flush=True)


def validate_checkpoint(result, day, design_hash, names):
    if result["day"] != day or result["design_sha256"] != design_hash:
        raise ValueError("Checkpoint belongs to another frozen run")
    if [r["model"] for r in result["rows"]] != names:
        raise ValueError("Missing or repeated models in checkpoint")
    tapes = {r["tape_sha256"] for r in result["rows"]}
    if len(tapes) != 1 or any(r["day"] != day for r in result["rows"]):
        raise ValueError("Unpaired auction days")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["preflight", "final"])
    parser.add_argument("--workers", type=int, default=min(6, max(1, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    lock = lock_run()
    if args.mode == "preflight":
        result = preflight()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    subprocess.run(["git", "merge-base", "--is-ancestor", BASE, "HEAD"], cwd=ROOT, check=True)
    OUT.mkdir(parents=True, exist_ok=True)
    design = frozen_design()
    design_hash = digest(design)
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = read(protocol_path)
        if protocol["design"] != design or protocol["design_sha256"] != design_hash:
            raise ValueError("Frozen code, inputs or environment changed. Original final results are preserved.")
    else:
        if any(OUT.iterdir()):
            raise ValueError("Results directory must be empty before initial freeze")
        check = preflight()
        protocol = {"frozen_utc": now(), "design": design, "design_sha256": design_hash, "preflight": check}
        save(protocol_path, protocol)
    if (OUT / "DONE.json").exists():
        done = read(OUT / "DONE.json")
        if done["design_sha256"] != design_hash:
            raise ValueError("Completed result has a different protocol")
        for rel, expected in done["artifacts"].items():
            if sha(OUT / rel) != expected:
                raise ValueError(f"Saved result changed: {rel}")
        print("Completed final result verified; no test episodes regenerated.", flush=True)
        return
    marker = OUT / "TEST_OPENED.json"
    if marker.exists():
        if read(marker)["design_sha256"] != design_hash:
            raise ValueError("Test access belongs to another protocol")
    else:
        save(marker, {"opened_utc": now(), "design_sha256": design_hash, "days": design["days"],
            "note": "Explicit final-test access; no parent episode guards modified. Resume only with identical frozen design."})
    names = list(design["models"])
    pending = []
    for day in design["days"]:
        path = OUT / f"checkpoints/day_{day}.json"
        if path.exists():
            validate_checkpoint(read(path), day, design_hash, names)
        else:
            pending.append((day, design["models"], design_hash))
    completed = 500 - len(pending)
    print(f"Final evaluation: {completed}/500 days saved; {len(names)} frozen models, workers={args.workers}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        for result in pool.map(worker, pending, chunksize=1):
            day = result["day"]
            validate_checkpoint(result, day, design_hash, names)
            if result["traces"]:
                write_csv(OUT / f"traces/day_{day}.csv", result["traces"])
            result.pop("traces")
            save(OUT / f"checkpoints/day_{day}.json", result)
            completed += 1
            if completed % 10 == 0 or completed == 500:
                print(f"Saved {completed}/500 days", flush=True)
    report(protocol)
    if frozen_design() != design:
        raise ValueError("Frozen inputs changed while running; not publishing completion")
    artifacts = {str(p.relative_to(OUT)): sha(p) for p in sorted(OUT.rglob("*")) if p.is_file() and p.name != "DONE.json"}
    save(OUT / "DONE.json", {"status": "completed", "completed_utc": now(), "design_sha256": design_hash,
        "days": 500, "models": names, "rollouts": 500 * len(names), "artifacts": artifacts, "final_test_used": True})
    print(f"Saved final report: {OUT / 'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
