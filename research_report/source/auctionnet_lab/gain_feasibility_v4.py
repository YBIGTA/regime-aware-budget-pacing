"""Matched P ablation and local gain feasibility diagnostics."""
import hashlib
import json
import math
from pathlib import Path
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import controller_diagnostic_v2 as d

ROOT = Path(__file__).resolve().parent


def roll(ep, p, bias, start=0, end=48, state=None):
    B = ep["budget"]
    remaining, integral, previous = (
        state if state is not None else (B, 0., bias)
    )
    rows, states = [], {}

    for t in range(start, end):
        states[t] = (remaining, integral, previous)
        tick = ep["ticks"][t]
        e = t / 48 - (B - remaining) / B
        candidate = integral + e / 48
        lo, hi = max(-6., previous - .5), min(4., previous + .5)
        raw = bias + p["kp"] * e + p["ki"] * candidate

        if (raw > hi and e > 0) or (raw < lo and e < 0):
            raw = bias + p["kp"] * e + p["ki"] * integral
        else:
            integral = candidate

        u = min(hi, max(lo, raw))
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
        rows.append({
            "tick": t,
            "alpha": math.exp(u),
            "spend": spend,
            "remaining": remaining,
            "error": (t + 1) / 48 - (B - remaining) / B,
            "value": value,
            "du": abs(u - previous),
        })
        previous = u

    states[end] = (remaining, integral, previous)
    return pd.DataFrame(rows), states


def main():
    lab = d.load_lab()
    d.self_test(lab)

    base = ROOT / "runs/global_retune_v3"
    c = json.loads(
        (base / "selected_controller.json").read_text()
    )
    cfg = json.loads((base / "config.json").read_text())

    if c["antiwindup"] != "clamp":
        raise ValueError(
            "This diagnostic expects the selected clamp controller."
        )

    p = c["parameters"]
    split = cfg["split"]
    wanted = set(split["train"] + split["validation"])
    biases = {
        int(k): v for k, v in c["initial_bias"].items()
    }

    stamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S_%fZ"
    )
    out = ROOT / "runs" / ("gain_feasibility_v4_" + stamp)
    out.mkdir(parents=True)

    lab.save(out / "protocol.json", {
        "source_controller": c,
        "horizons": [6, 12],
        "anchors": [6, 12, 18, 24, 30, 36],
        "selection": (
            "Oracle uses future outcomes; "
            "NOT a deployable policy or performance result."
        ),
        "scope": "Train and validation only.",
        "p_ablation": (
            "Same kp/bias; ki=0. "
            "P-only has NOT been independently optimized."
        ),
        "code_hash": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "source_hashes": {
            name: hashlib.sha256(
                (base / name).read_bytes()
            ).hexdigest()
            for name in [
                "selected_controller.json", "config.json"
            ]
        },
    })

    pool = sorted(set(
        (float(kp), float(ki))
        for kp in [.25, 1., 4., 16., 64., p["kp"]]
        for ki in [0., .03, .3, 3., 30., p["ki"]]
    ))
    baseline_id = pool.index((p["kp"], p["ki"]))

    episodes, windows = [], []
    identities = set()
    files = sorted(
        (ROOT / "data").glob("pack-*/*.parquet")
    )

    for path in files:
        match = re.fullmatch(
            r"p(\d+)-a(\d+)\.parquet", path.name
        )
        if not match or int(match[1]) not in wanted:
            continue

        print("Preparing:", path.name, flush=True)
        ep = lab.prepare_episode(
            pd.read_parquet(path), cfg["seed"]
        )
        identity = (ep["period"], ep["ad"])
        if identity in identities:
            raise ValueError("Duplicate episode.")
        identities.add(identity)

        group = (
            "train" if ep["period"] in split["train"]
            else "validation"
        )
        center = biases[ep["ad"]] if c["calibrated"] else 0.
        bias = float(np.clip(center + p["offset"], -6, 4))

        full, states = roll(ep, p, bias)
        slow, _ = d.replay(ep, p, bias, "clamp")
        assert np.isclose(
            full.error.abs().mean(), slow["te"]
        )
        assert np.isclose(
            full.value.sum(), slow["expected_conversions"]
        )

        for name, gains in [
            ("selected_PI", p),
            ("matched_P", dict(p, ki=0.)),
        ]:
            f, _ = roll(ep, gains, bias)
            episodes.append({
                "split": group,
                "period": ep["period"],
                "advertiser": ep["ad"],
                "method": name,
                "te": float(f.error.abs().mean()),
                "expected_conversions": float(f.value.sum()),
                "unspent": float(
                    f.remaining.iloc[-1] / ep["budget"]
                ),
            })

        for start in [6, 12, 18, 24, 30, 36]:
            for horizon in [6, 12]:
                end = start + horizon
                if end > 48 or states[start][0] < .1:
                    continue

                prefix_state = states[start]
                same, _ = roll(
                    ep, p, bias, start, end, prefix_state
                )
                cols = ["alpha", "spend", "remaining", "error"]
                assert np.allclose(
                    same[cols], full.iloc[start:end][cols]
                )

                # 意思決定時点より前の情報だけを特徴量に使用
                old = ep["ticks"][start - 1]["features"]
                features = {
                    "time_left": (48 - start) / 48,
                    "remaining_ratio": (
                        prefix_state[0] / ep["budget"]
                    ),
                    "error_before": (
                        start / 48
                        - (ep["budget"] - prefix_state[0])
                        / ep["budget"]
                    ),
                    "log_alpha_before": prefix_state[2],
                    **dict(zip([
                        "log_volume", "volume_change", "volume_cv",
                        "value_median", "value_spread",
                        "price_value_median", "price_value_q90",
                        "price_value_change", "price_shift",
                        "value_shift", "volume_shift",
                    ], old)),
                }

                for candidate_id, (kp, ki) in enumerate(pool):
                    f, _ = roll(
                        ep, {"kp": kp, "ki": ki},
                        bias, start, end, prefix_state
                    )
                    local_j = float(
                        f.error.abs().mean() + .01 * f.du.mean()
                    )
                    windows.append({
                        "split": group,
                        "period": ep["period"],
                        "advertiser": ep["ad"],
                        "start": start,
                        "horizon": horizon,
                        "candidate": candidate_id,
                        "kp": kp,
                        "ki": ki,
                        "local_j": local_j,
                        "value": float(f.value.sum()),
                        **features,
                    })

    expected = {
        (period, ad) for period in wanted for ad in biases
    }
    if identities != expected:
        raise ValueError("Incomplete period/advertiser coverage.")

    e = pd.DataFrame(episodes)
    e.to_csv(out / "p_ablation_episodes.csv", index=False)
    summary = (
        e.groupby(["split", "method", "period"])[
            ["te", "expected_conversions", "unspent"]
        ].mean()
        .groupby(["split", "method"]).mean()
    )
    summary.to_csv(out / "p_ablation_summary.csv")

    w = pd.DataFrame(windows)
    keys = ["split", "period", "advertiser", "start", "horizon"]
    b = w[w.candidate == baseline_id][
        keys + ["local_j", "value"]
    ].rename(columns={
        "local_j": "baseline_j",
        "value": "baseline_value",
    })
    w = w.merge(b, on=keys, validate="many_to_one")

    w["feasible"] = (
        w.value + 1e-12 >= .98 * w.baseline_value
    )
    w["search_loss"] = (
        w.local_j
        + 10 * np.maximum(
            0, .98 * w.baseline_value - w.value
        ) / np.maximum(w.baseline_value, 1e-12)
    )

    # 공통 계수는 학습 구간에서만 선택.
    common = (
        w[w.split == "train"]
        .groupby(["horizon", "candidate"]).search_loss.mean()
    )
    common_ids = {
        h: int(common.loc[h].idxmin()) for h in [6, 12]
    }
    w.to_csv(out / "candidate_windows.csv", index=False)

    # 각 구간에서 기대 전환 98% 이상을 유지하는 후보 중 최적 선택.
    # 동률이면 기존 계수를 우선해 무의미한 전환 집계를 방지.
    feasible = w[w.feasible].copy()
    best = feasible.groupby(keys).local_j.transform("min")
    feasible = feasible[
        feasible.local_j <= best + 1e-10
    ].copy()
    feasible["changes_pair"] = (
        feasible.candidate != baseline_id
    )
    oracle = (
        feasible.sort_values(["changes_pair", "candidate"])
        .drop_duplicates(keys)
    )
    oracle.to_csv(out / "oracle_windows.csv", index=False)

    reports = []
    for (group, horizon), block in w.groupby(
        ["split", "horizon"]
    ):
        local = oracle[
            (oracle.split == group)
            & (oracle.horizon == horizon)
        ]
        baseline = block[block.candidate == baseline_id]
        shared = block[
            block.candidate == common_ids[horizon]
        ]
        reports.append({
            "split": group,
            "horizon": horizon,
            "windows": len(local),
            "baseline_j": baseline.local_j.mean(),
            "common_train_selected_j": shared.local_j.mean(),
            "oracle_j": local.local_j.mean(),
            "oracle_gain_pct": 100 * (
                1 - local.local_j.mean()
                / baseline.local_j.mean()
            ),
            "common_value_pass_fraction": (
                shared.feasible.mean()
            ),
            "oracle_changes_gain_fraction": (
                (local.candidate != baseline_id).mean()
            ),
            "oracle_distinct_gain_pairs": (
                local.candidate.nunique()
            ),
        })

    report = pd.DataFrame(reports)
    report.to_csv(out / "local_summary.csv", index=False)

    print("\nP ABLATION (matched coefficients, not independent tuning):")
    print(summary.round(6).to_string())
    print("\nLOCAL GAIN UPPER BOUND (future-informed diagnostic only):")
    print(report.round(5).to_string(index=False))
    print("\nResults:", out.resolve())


if __name__ == "__main__":
    main()
