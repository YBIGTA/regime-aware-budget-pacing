"""Train search, validation selection, development-only reporting."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import optuna
import pandas as pd
import controller_diagnostic_v2 as d

ROOT = Path(__file__).resolve().parent


def fast(ep, p, bias, aw):
    B = ep["budget"]
    remaining, integral, previous = B, 0., bias
    errors, outputs, values, spends, remainders = [], [], [], [], []
    hits = 0

    for t, tick in enumerate(ep["ticks"]):
        e = t / 48 - (B - remaining) / B
        candidate = integral + e / 48
        raw = bias + p["kp"] * e + p["ki"] * candidate
        lo, hi = max(-6., previous - .5), min(4., previous + .5)

        if aw == "clamp":
            if (raw > hi and e > 0) or (raw < lo and e < 0):
                raw = bias + p["kp"] * e + p["ki"] * integral
            else:
                integral = candidate

        u = min(hi, max(lo, raw))
        if aw == "backcalc":
            integral = candidate + .5 * (u - raw) / p["ki"]

        hits += int(
            t > 0 and abs(abs(u - previous) - .5) <= 1e-6 + .5e-5
        )
        count = (
            int(np.searchsorted(
                tick["threshold"], math.exp(u), side="right"
            ))
            if remaining >= .1 else 0
        )
        spend = float(tick["cost_cum"][count])
        value = float(tick["value_cum"][count])

        if spend > remaining:
            order = np.random.default_rng(
                tick["seed"]
            ).permutation(tick["n"])
            selected = order[order < count]
            count = int(np.searchsorted(
                np.cumsum(tick["cost"][selected]),
                remaining, side="right"
            ))
            selected = selected[:count]
            spend = float(tick["cost"][selected].sum())
            value = float(tick["value"][selected].sum())

        remaining = max(0., remaining - spend)
        errors.append(abs((t + 1) / 48 - (B - remaining) / B))
        outputs.append(u)
        values.append(value)
        spends.append(spend)
        remainders.append(remaining)
        previous = u

    te = float(np.mean(errors))
    tv = float(np.abs(np.diff(outputs)).mean())
    nz = np.flatnonzero(np.asarray(spends) > 0)

    return {
        "te": te,
        "multiplier_tv": tv,
        "unspent": remaining / B,
        "expected_conversions": float(sum(values)),
        "objective": te + .01 * tv + .2 * remaining / B,
        "rate_hit_fraction": hits / 47,
        "initial_zero_ticks": int(nz[0]) if len(nz) else 48,
        "exhausted_by_90pct": float(
            any(r < .1 for r in remainders[:43])
        ),
        "remaining_at_75pct": remainders[35] / B,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=60)
    ap.add_argument("--name", default="global_retune_v3")
    args = ap.parse_args()
    if args.trials < 20:
        raise ValueError("Use at least 20 trials per variant.")

    lab = d.load_lab()
    d.self_test(lab)

    base = ROOT / "runs/full_pilot"
    old = json.loads((base / "fixed.json").read_text())
    split = json.loads((base / "split.json").read_text())
    seed = int(json.loads(
        (base / "config.json").read_text()
    ).get("seed", 42))

    periods = set(sum(split.values(), []))
    episodes, identities = [], set()
    files = sorted((ROOT / "data").glob("pack-*/*.parquet"))
    started = time.monotonic()

    for i, path in enumerate(files, 1):
        print(
            f"Preparing {i}/{len(files)}: {path.name}",
            flush=True
        )
        df = pd.read_parquet(path)
        if int(df.deliveryPeriodIndex.iloc[0]) not in periods:
            continue

        ep = lab.prepare_episode(df, seed)
        key = (ep["period"], ep["ad"])
        if key in identities:
            raise ValueError(f"Duplicate episode: {key}")

        identities.add(key)
        episodes.append(ep)
        del df

    train = [
        e for e in episodes if e["period"] in split["train"]
    ]
    val = [
        e for e in episodes if e["period"] in split["validation"]
    ]
    biases, calibration = d.initial_bias(train)

    expected = {(p, a) for p in periods for a in biases}
    if not train or not val or identities != expected:
        raise ValueError("Missing or unbalanced data.")

    methods = {
        "global_clamp": (False, "clamp"),
        "calibrated_clamp": (True, "clamp"),
        "global_backcalc": (False, "backcalc"),
        "calibrated_backcalc": (True, "backcalc"),
    }

    def bias_for(ep, p, calibrated):
        center = biases[ep["ad"]] if calibrated else 0.
        return float(np.clip(center + p["offset"], -6, 4))

    def evaluate(group, p, calibrated, aw):
        rows = [
            fast(e, p, bias_for(e, p, calibrated), aw)
            for e in group
        ]
        return {
            k: float(np.mean([r[k] for r in rows]))
            for k in rows[0]
        }

    reference = {
        "kp": old["kp"],
        "ki": old["ki"],
        "offset": old["bias"],
    }

    # 高速版と既存版の数値一致を確認
    for ep in episodes:
        for calibrated, aw in [
            (False, "clamp"),
            (True, "clamp"),
            (True, "backcalc"),
        ]:
            bias = bias_for(ep, reference, calibrated)
            quick = fast(ep, reference, bias, aw)
            slow, _ = d.replay(ep, reference, bias, aw)
            for k in [
                "te", "multiplier_tv", "unspent",
                "expected_conversions", "objective",
            ]:
                assert np.isclose(
                    quick[k], slow[k], atol=1e-10
                ), k

    print("Accelerated scoring parity passed.", flush=True)

    baseline = {
        k: evaluate(g, reference, False, "clamp")
        for k, g in [("train", train), ("validation", val)]
    }

    out = ROOT / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    config = {
        "trials_per_variant": args.trials,
        "seed": seed,
        "split": split,
        "reference": reference,
        "initial_bias": biases,
        "calibration": calibration,
        "bounds": [-6, 4],
        "rate": .5,
        "tracking": .5,
        "value_floor": .98,
        "baseline": baseline,
        "hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path(__file__),
                ROOT / "run.py",
                ROOT / "controller_diagnostic_v2.py",
            ]
        },
        "data_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (ROOT / "data").glob("pack-*/manifest.json")
        },
    }

    config_path = out / "config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != json.loads(
            json.dumps(config)
        ):
            raise ValueError(
                "Configuration changed. Use a new --name."
            )

    lab.save(config_path, config)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    chosen = {
        "original_reference": {
            "parameters": reference,
            "calibrated": False,
            "antiwindup": "clamp",
            "validation_objective": baseline[
                "validation"
            ]["objective"],
        }
    }

    for name, (calibrated, aw) in methods.items():
        study = optuna.create_study(
            direction="minimize",
            study_name=name,
            load_if_exists=True,
            storage="sqlite:///" + str(
                (out / (name + ".sqlite3")).resolve()
            ),
            sampler=optuna.samplers.TPESampler(
                seed=seed, n_startup_trials=20
            ),
        )

        if not study.trials:
            offset = 0. if calibrated else old["bias"]
            study.enqueue_trial({
                "kp": old["kp"],
                "ki": old["ki"],
                "offset": offset,
            })
            for kp in [.03, .3, 3., 30.]:
                for ki in [.03, .3, 3., 30.]:
                    study.enqueue_trial({
                        "kp": kp, "ki": ki, "offset": offset
                    })

        print(
            f"\n{name}: target {args.trials} trials",
            flush=True
        )

        def objective(trial):
            p = {
                "kp": trial.suggest_float(
                    "kp", .01, 100, log=True
                ),
                "ki": trial.suggest_float(
                    "ki", .01, 100, log=True
                ),
                "offset": trial.suggest_float(
                    "offset", -5, 2
                ),
            }
            result = evaluate(train, p, calibrated, aw)
            trial.set_user_attr("train_metrics", result)
            ratio = result["expected_conversions"] / max(
                baseline["train"]["expected_conversions"], 1e-12
            )
            return (
                result["objective"]
                + 10 * max(0., .98 - ratio)
            )

        def progress(s, trial):
            if (trial.number + 1) % 10 == 0:
                print(
                    f"{name}: {trial.number + 1}/{args.trials}",
                    flush=True
                )

        done = sum(
            t.state == optuna.trial.TrialState.COMPLETE
            for t in study.trials
        )
        study.optimize(
            objective,
            n_trials=max(0, args.trials - done),
            callbacks=[progress],
        )

        eligible = []
        for trial in study.trials:
            if trial.state != optuna.trial.TrialState.COMPLETE:
                continue

            tr = trial.user_attrs["train_metrics"]
            if tr["expected_conversions"] < (
                .98 * baseline["train"]["expected_conversions"]
            ):
                continue

            vr = evaluate(val, trial.params, calibrated, aw)
            if vr["expected_conversions"] >= (
                .98
                * baseline["validation"]["expected_conversions"]
            ):
                eligible.append((
                    vr["objective"],
                    trial.number,
                    trial.params,
                    vr,
                ))

        study.trials_dataframe().to_csv(
            out / (name + "_trials.csv"), index=False
        )
        print(
            f"{name}: feasible candidates={len(eligible)}",
            flush=True
        )

        if eligible:
            score, number, p, vr = min(
                eligible, key=lambda x: (x[0], x[1])
            )
            chosen[name] = {
                "parameters": p,
                "calibrated": calibrated,
                "antiwindup": aw,
                "trial": number,
                "validation_objective": score,
                "validation_metrics": vr,
            }

        lab.save(out / "candidate_controllers.json", chosen)

    # 개발 기간을 평가하기 전에 선택 결과를 확정.
    winner = min(
        chosen, key=lambda n: chosen[n]["validation_objective"]
    )
    lab.save(out / "selected_controller.json", {
        "method": winner,
        **chosen[winner],
        "initial_bias": biases,
        "note": (
            "Validation selection; >=98% mean expected "
            "conversions on both train and validation."
        ),
    })

    records, traces = [], []
    for name, c in chosen.items():
        p = c["parameters"]
        for ep in episodes:
            group = next(
                k for k, v in split.items()
                if ep["period"] in v
            )
            if group == "test":
                group = "development"

            bias = bias_for(ep, p, c["calibrated"])
            m = fast(ep, p, bias, c["antiwindup"])
            records.append({
                "split": group,
                "method": name,
                "period": ep["period"],
                "advertiser": ep["ad"],
                **m,
            })
            _, f = d.replay(ep, p, bias, c["antiwindup"])
            traces.append(f.assign(split=group, method=name))

    frame = pd.DataFrame(records)
    frame.to_csv(out / "episode_metrics.csv", index=False)
    pd.concat(traces, ignore_index=True).to_csv(
        out / "traces.csv", index=False
    )

    cols = [
        "te", "expected_conversions", "unspent", "objective",
        "rate_hit_fraction", "initial_zero_ticks",
        "exhausted_by_90pct", "remaining_at_75pct",
    ]
    summary = (
        frame.groupby(["split", "method", "period"])[cols]
        .mean()
        .groupby(["split", "method"])
        .mean()
    )
    summary.to_csv(out / "summary.csv")

    print("\nSELECTED USING VALIDATION:", winner)
    print(json.dumps(chosen[winner], indent=2))
    print("\nRETUNED SUMMARY:")
    print(summary.round(6).to_string())
    print(f"\nElapsed: {time.monotonic() - started:.1f}s")
    print("Results:", out.resolve())


if __name__ == "__main__":
    main()
