"""Existing AuctionNet data: PI initialization and anti-windup diagnostics."""
import argparse
import hashlib
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def load_lab():
    spec = importlib.util.spec_from_file_location(
        "auctionnet_original", ROOT / "run.py"
    )
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    return lab


def initial_bias(episodes):
    # 학습 기간의 첫 4틱만 사용해 광고주별 초기 승수 추정.
    estimates, audit = {}, []
    for ep in episodes:
        ticks = ep["ticks"][:4]
        threshold = np.concatenate([t["threshold"] for t in ticks])
        cost = np.concatenate([t["cost"] for t in ticks])
        order = np.argsort(threshold, kind="stable")
        threshold = threshold[order]
        total = np.cumsum(cost[order])

        target = ep["budget"] * len(ticks) / 48
        index = int(np.searchsorted(total, target, side="left"))
        alpha = threshold[min(index, len(threshold) - 1)]
        bias = float(np.clip(np.log(max(alpha, 1e-300)), -6, 4))

        estimates.setdefault(ep["ad"], []).append(bias)
        audit.append({
            "period": ep["period"],
            "advertiser": ep["ad"],
            "alpha": math.exp(bias),
            "target": target,
            "target_available": bool(total[-1] >= target),
        })

    return {
        ad: float(np.median(values))
        for ad, values in estimates.items()
    }, audit


def replay(ep, params, bias, antiwindup="clamp", rate=.5, tracking=.5):
    B = ep["budget"]
    remaining, integral, previous = B, 0., bias
    kp, ki = params["kp"], params["ki"]
    rows = []

    for t, tick in enumerate(ep["ticks"]):
        error = t / 48 - (B - remaining) / B
        low = max(-6., previous - rate)
        high = min(4., previous + rate)
        candidate = integral + error / 48
        raw = bias + kp * error + ki * candidate
        blocked = False

        if antiwindup == "clamp":
            blocked = (
                (raw > high and error > 0)
                or (raw < low and error < 0)
            )
            if blocked:
                raw = bias + kp * error + ki * integral
            else:
                integral = candidate

        u = float(np.clip(raw, low, high))

        if antiwindup == "backcalc":
            # 변화량 제한과 출력 상하한을 모두 적용한 값으로 보정.
            integral = candidate + tracking * (u - raw) / ki

        alpha = math.exp(u)
        count = (
            int(np.searchsorted(tick["threshold"], alpha, side="right"))
            if remaining >= .1 else 0
        )
        spend = float(tick["cost_cum"][count])
        value = float(tick["value_cum"][count])

        if spend > remaining:
            # 기존 run.py와 동일한 예산 부족 처리 및 난수.
            order = np.random.default_rng(
                tick["seed"]
            ).permutation(tick["n"])
            selected = order[order < count]
            cumulative = np.cumsum(tick["cost"][selected])
            count = int(np.searchsorted(
                cumulative, remaining, side="right"
            ))
            selected = selected[:count]
            spend = float(tick["cost"][selected].sum())
            value = float(tick["value"][selected].sum())

        before = remaining
        remaining = max(0., remaining - spend)

        rows.append({
            "period": ep["period"],
            "advertiser": ep["ad"],
            "tick": t,
            "alpha": alpha,
            "log_alpha": u,
            "kp": kp,
            "ki": ki,
            "bias": bias,
            "error_before": error,
            "raw_output": raw,
            "applied_output": u,
            "integral": integral,
            "integral_blocked": int(blocked),
            "rate_hit": int(
                t > 0 and np.isclose(
                    abs(u - previous), rate, atol=1e-6
                )
            ),
            "spend": spend,
            "remaining_before": before,
            "remaining": remaining,
            "value": value,
            "error": (t + 1) / 48 - (B - remaining) / B,
        })
        previous = u

    f = pd.DataFrame(rows)
    te = float(f.error.abs().mean())
    tv = float(f.log_alpha.diff().abs().iloc[1:].mean())
    positives = f.index[f.spend > 0].tolist()
    depleted = f.index[f.remaining < .1].tolist()

    metric = {
        "period": ep["period"],
        "advertiser": ep["ad"],
        "te": te,
        "multiplier_tv": tv,
        "unspent": remaining / B,
        "expected_conversions": float(f.value.sum()),
        "objective": te + .01 * tv + .2 * remaining / B,
        "rate_hit_fraction": float(f.rate_hit.iloc[1:].mean()),
        "integral_blocked_fraction": float(
            f.integral_blocked.mean()
        ),
        "initial_zero_ticks": positives[0] if positives else 48,
        "max_abs_error": float(f.error.abs().max()),
        "early_exhausted": int(
            bool(depleted) and depleted[0] < 47
        ),
    }
    return metric, f


def self_test(lab):
    # 예산 부족을 포함한 여러 조건에서 기존 계산과 동일한지 검사.
    for B in [15., 100., 2000.]:
        ticks = []
        for t in range(48):
            threshold = np.array([.05, .25, .7, 1.5, 3., 8.])
            cost = (
                np.array([.1, .4, 1., 3., 8., 12.])
                * (1 + .4 * (t % 3))
            )
            value = np.arange(1., 7.) / 20
            ticks.append({
                "threshold": threshold,
                "cost": cost,
                "value": value,
                "cost_cum": np.r_[0, np.cumsum(cost)],
                "value_cum": np.r_[0, np.cumsum(value)],
                "n": 6,
                "seed": 100 + t,
                "features": [0.] * 11,
                "ref_win": 0.,
                "ref_spend": 0.,
                "sensitivity": 0.,
            })

        ep = {"period": 7, "ad": 1, "budget": B, "ticks": ticks}
        p = {"bias": -3.353, "kp": 39.911, "ki": 60.941}

        old, traces = lab.simulate(ep, p, trace=True)
        new, f = replay(ep, p, p["bias"])

        for key in [
            "te", "multiplier_tv", "unspent",
            "expected_conversions", "objective",
        ]:
            assert np.isclose(old[key], new[key], atol=1e-12), key

        for key in ["alpha", "spend", "remaining", "error"]:
            assert np.allclose(
                [r[key] for r in traces], f[key], atol=1e-12
            ), key

        for mode in ["clamp", "backcalc"]:
            _, f = replay(ep, p, 0., mode)
            assert np.isfinite(
                f.select_dtypes("number").to_numpy()
            ).all()
            assert f.log_alpha.between(-6, 4).all()
            assert (
                f.log_alpha.diff().abs().dropna().max()
                <= .5 + 1e-12
            )
            assert f.spend.sum() <= B + 1e-9

    print("Self-check passed.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    lab = load_lab()
    self_test(lab)
    if args.self_test:
        return

    base = ROOT / "runs" / "full_pilot"
    params = json.loads((base / "fixed.json").read_text())
    split = json.loads((base / "split.json").read_text())

    config_path = base / "config.json"
    original_config = (
        json.loads(config_path.read_text())
        if config_path.exists() else {}
    )
    seed = int(original_config.get("seed", 42))

    periods = set(sum(split.values(), []))
    files = sorted((ROOT / "data").glob("pack-*/*.parquet"))
    if not files:
        raise RuntimeError("Existing Parquet data missing.")

    episodes, identities = [], set()
    started = time.monotonic()

    for i, path in enumerate(files, 1):
        print(
            f"Preparing {i}/{len(files)}: {path.name}",
            flush=True,
        )
        df = pd.read_parquet(path)
        if int(df.deliveryPeriodIndex.iloc[0]) not in periods:
            continue

        ep = lab.prepare_episode(df, seed)
        identity = (ep["period"], ep["ad"])
        if identity in identities:
            raise RuntimeError(f"Duplicate episode: {identity}")

        identities.add(identity)
        episodes.append(ep)
        del df

    train = [
        e for e in episodes
        if e["period"] in split["train"]
    ]
    biases, calibration = initial_bias(train)

    expected = {
        (period, ad)
        for period in periods
        for ad in biases
    }
    if identities != expected:
        raise RuntimeError("Incomplete period/advertiser coverage.")

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    out = ROOT / "runs" / ("controller_diagnostic_v2_" + stamp)
    out.mkdir(parents=True, exist_ok=False)

    lab.save(out / "config.json", {
        "fixed": params,
        "original_split": split,
        "seed": seed,
        "train_only_initial_bias": biases,
        "calibration": calibration,
        "tracking_gain_per_tick": .5,
        "code_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "run_py_sha256": hashlib.sha256(
            (ROOT / "run.py").read_bytes()
        ).hexdigest(),
        "data_manifests": {
            str(p.relative_to(ROOT)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in (ROOT / "data").glob("pack-*/manifest.json")
        },
        "note": (
            "Diagnostic ablation; inherited gains NOT re-tuned. "
            "Former test periods are development data."
        ),
    })

    methods = [
        ("legacy", False, "clamp", .5),
        ("init_only", True, "clamp", .5),
        ("backcalc_only", False, "backcalc", .5),
        ("init_backcalc", True, "backcalc", .5),
        ("init_backcalc_rate1", True, "backcalc", 1.),
    ]
    metrics, traces = [], []

    for ep in episodes:
        group = next(
            k for k, values in split.items()
            if ep["period"] in values
        )
        if group == "test":
            group = "development"

        for method, calibrated, aw, rate in methods:
            bias = (
                biases[ep["ad"]]
                if calibrated else params["bias"]
            )
            m, f = replay(ep, params, bias, aw, rate)
            metrics.append({
                **m, "method": method, "split": group
            })
            traces.append(f.assign(method=method, split=group))

    metrics = pd.DataFrame(metrics)
    legacy = metrics[
        metrics.method == "legacy"
    ].set_index(["period", "advertiser"])

    for i, r in metrics.iterrows():
        b = legacy.loc[(r["period"], r["advertiser"])]
        metrics.loc[i, "te_change"] = r.te - b.te
        metrics.loc[i, "value_change"] = (
            r.expected_conversions - b.expected_conversions
        )

    numeric = [
        c for c in metrics.select_dtypes("number").columns
        if c not in ["period", "advertiser"]
    ]
    summary = (
        metrics.groupby(["split", "method", "period"])[numeric]
        .mean()
        .groupby(["split", "method"])
        .mean()
    )
    combined = pd.concat(traces, ignore_index=True)

    metrics.to_csv(out / "episode_metrics.csv", index=False)
    combined.to_csv(out / "traces.csv", index=False)
    summary.to_csv(out / "summary.csv")

    old_path = base / "test_traces.csv"
    if old_path.exists():
        keys = ["period", "advertiser", "tick"]
        old = pd.read_csv(old_path)
        old = old[old.method == "fixed"].sort_values(keys)
        new = combined[
            (combined.method == "legacy")
            & (combined.split == "development")
        ].sort_values(keys)

        assert np.array_equal(old[keys], new[keys])
        for col in ["alpha", "spend", "remaining", "error"]:
            assert np.allclose(
                old[col], new[col], rtol=1e-9, atol=1e-9
            ), f"Original result mismatch: {col}"

        print(
            "Existing full_pilot traces reproduced.",
            flush=True,
        )

    print("\nTRAIN-ONLY INITIAL MULTIPLIERS:")
    print({
        ad: round(math.exp(b), 5)
        for ad, b in biases.items()
    })

    print("\nDIAGNOSTIC SUMMARY (no re-tuning):")
    columns = [
        "te", "expected_conversions", "unspent",
        "initial_zero_ticks", "rate_hit_fraction",
        "integral_blocked_fraction", "early_exhausted",
    ]
    print(summary[columns].round(6).to_string())
    print(f"\nElapsed: {time.monotonic() - started:.1f}s")
    print("Results:", out.resolve())


if __name__ == "__main__":
    main()
