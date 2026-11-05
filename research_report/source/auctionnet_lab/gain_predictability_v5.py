"""Past-feature nearest-neighbor gain selection; local-window evaluation."""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

FEATURES = [
    "time_left", "remaining_ratio", "error_before",
    "log_alpha_before",
    "log_volume", "volume_change", "volume_cv",
    "value_median", "value_spread",
    "price_value_median", "price_value_q90",
    "price_value_change", "price_shift",
    "value_shift", "volume_shift",
]

KEYS = ["period", "advertiser", "start"]


def prepare(frame, candidates):
    grouped = frame.groupby(KEYS, sort=True)
    sizes = grouped.size()
    if not (sizes == len(candidates)).all():
        raise ValueError("Incomplete candidate coverage.")

    # Each feature must describe the state before choosing a candidate.
    if (grouped[FEATURES].nunique(dropna=False) > 1).any().any():
        raise ValueError("Features differ between candidates.")

    meta = grouped.first()
    order = meta.index

    def matrix(column):
        return (
            frame.pivot(index=KEYS, columns="candidate", values=column)
            .reindex(index=order, columns=candidates)
            .to_numpy(dtype=float)
        )

    x = meta[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Features contain NaN or infinity.")

    return meta.reset_index(), x, {
        name: matrix(name)
        for name in ["local_j", "search_loss", "value", "feasible"]
    }


def predict(x_train, y_train, x_query, neighbors):
    # Fit scaling using only the training fold.
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-8] = 1.0

    train_z = (x_train - center) / scale
    query_z = (x_query - center) / scale

    distance = (
        (query_z[:, None, :] - train_z[None, :, :]) ** 2
    ).mean(axis=2)

    k = min(neighbors, len(x_train))
    nearest = np.argsort(distance, axis=1)[:, :k]

    # Uniform neighbors provide a simple first predictive baseline.
    return y_train[nearest].mean(axis=1)


def choose(predicted, baseline, threshold):
    selected = predicted.argmin(axis=1)
    improvement = -predicted[np.arange(len(selected)), selected]
    selected[improvement <= threshold] = baseline
    return selected


def period_mean(values, periods):
    return float(
        pd.DataFrame({"period": periods, "value": values})
        .groupby("period").value.mean().mean()
    )


def evaluate(meta, arrays, selected, baseline, split, horizon, method):
    rows = np.arange(len(meta))
    score = arrays["local_j"][rows, selected]
    base_score = arrays["local_j"][:, baseline]
    value = arrays["value"][rows, selected]
    base_value = arrays["value"][:, baseline]
    feasible = arrays["feasible"][rows, selected]
    periods = meta["period"].to_numpy()

    mean_score = period_mean(score, periods)
    mean_base = period_mean(base_score, periods)

    report = {
        "split": split,
        "horizon_hours": horizon / 2,
        "method": method,
        "windows": len(meta),
        "local_j": mean_score,
        "gain_vs_PI_pct": 100 * (1 - mean_score / mean_base),
        "value_pass_fraction": period_mean(feasible, periods),
        "switch_fraction": period_mean(selected != baseline, periods),
        "gain_at_least_2pct_fraction": period_mean(
            score <= .98 * base_score, periods
        ),
    }

    detail = meta[KEYS].copy()
    detail["split"] = split
    detail["horizon"] = horizon
    detail["method"] = method
    detail["candidate"] = selected
    detail["local_j"] = score
    detail["baseline_j"] = base_score
    detail["value"] = value
    detail["baseline_value"] = base_value
    detail["value_pass"] = feasible.astype(bool)
    return report, detail


def main():
    sources = [
        path for path in ROOT.glob("runs/gain_feasibility_v4_*")
        if (path / "candidate_windows.csv").is_file()
        and (path / "protocol.json").is_file()
    ]
    if not sources:
        raise FileNotFoundError("No completed v4 result found.")

    source = max(sources, key=lambda path: path.name)
    protocol = json.loads((source / "protocol.json").read_text())
    parameters = protocol["source_controller"]["parameters"]

    frame = pd.read_csv(source / "candidate_windows.csv")
    required = set(
        FEATURES + KEYS
        + ["split", "horizon", "candidate", "kp", "ki",
           "local_j", "search_loss", "value", "feasible"]
    )
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {required - set(frame.columns)}")

    if not frame["feasible"].isin([True, False, 0, 1]).all():
        raise ValueError("Unrecognized feasible column.")

    pairs = (
        frame[["candidate", "kp", "ki"]]
        .drop_duplicates()
        .sort_values("candidate")
    )
    if pairs.candidate.duplicated().any():
        raise ValueError("Candidate IDs map to multiple gain pairs.")

    candidates = pairs.candidate.astype(int).tolist()
    matching = pairs[
        np.isclose(pairs.kp, parameters["kp"])
        & np.isclose(pairs.ki, parameters["ki"])
    ]
    if len(matching) != 1:
        raise ValueError("Cannot identify the baseline gain pair.")
    baseline = candidates.index(int(matching.iloc[0].candidate))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = ROOT / "runs" / ("gain_predictability_v5_" + stamp)
    out.mkdir(parents=True)

    reports, details, tuning_rows = [], [], []
    settings = {}

    print("Source:", source.resolve(), flush=True)

    for horizon in [6, 12]:
        block = frame[frame.horizon == horizon]
        train_meta, train_x, train_a = prepare(
            block[block.split == "train"], candidates
        )
        val_meta, val_x, val_a = prepare(
            block[block.split == "validation"], candidates
        )
        periods = train_meta.period.to_numpy()
        unique_periods = np.unique(periods)
        if len(unique_periods) < 3:
            raise ValueError("At least three training periods are required.")

        # Predict relative penalized loss against the selected PI.
        # Future outcomes are training labels, never query features.
        denominator = train_a["local_j"][:, baseline] + .01
        target = (
            train_a["search_loss"]
            - train_a["search_loss"][:, [baseline]]
        ) / denominator[:, None]

        trials = []
        for neighbors in [5, 10, 20]:
            oof = np.empty_like(target)
            for held_out in unique_periods:
                fit = periods != held_out
                query = ~fit
                oof[query] = predict(
                    train_x[fit], target[fit],
                    train_x[query], neighbors
                )

            for threshold in [0., .02, .05, .10]:
                selected = choose(oof, baseline, threshold)
                losses = target[np.arange(len(selected)), selected]
                score = period_mean(losses, periods)
                trials.append(
                    (score, -threshold, -neighbors,
                     neighbors, threshold, selected.copy())
                )
                tuning_rows.append({
                    "horizon": horizon,
                    "neighbors": neighbors,
                    "threshold": threshold,
                    "train_oof_relative_penalized_loss": score,
                })

        # Include keeping PI everywhere as an explicit option.
        best = min(trials, key=lambda row: row[:3])
        enabled = bool(best[0] < 0.)
        neighbors, threshold = best[3], best[4]
        oof_selected = (
            best[5] if enabled
            else np.full(len(train_meta), baseline, dtype=int)
        )

        predicted = predict(train_x, target, val_x, neighbors)
        val_selected = (
            choose(predicted, baseline, threshold)
            if enabled
            else np.full(len(val_meta), baseline, dtype=int)
        )

        # Select one common gain pair using training periods only.
        common = int(np.argmin([
            period_mean(train_a["search_loss"][:, i], periods)
            for i in range(len(candidates))
        ]))

        settings[str(horizon)] = {
            "enabled_by_train_cv": enabled,
            "neighbors": neighbors,
            "minimum_predicted_relative_gain": threshold,
            "train_oof_score": best[0] if enabled else 0.,
            "common_candidate": candidates[common],
        }

        for name, meta, arrays, selected in [
            ("train_oof", train_meta, train_a, oof_selected),
            ("validation", val_meta, val_a, val_selected),
        ]:
            oracle_loss = np.where(
                arrays["feasible"].astype(bool),
                arrays["local_j"], np.inf
            )
            oracle = oracle_loss.argmin(axis=1)

            for method, picks in [
                ("selected_PI", np.full(len(meta), baseline, dtype=int)),
                ("common_train_selected",
                 np.full(len(meta), common, dtype=int)),
                ("past_feature_selector", selected),
                ("future_oracle", oracle),
            ]:
                report, detail = evaluate(
                    meta, arrays, picks, baseline,
                    name, horizon, method
                )
                detail["candidate"] = [
                    candidates[i] for i in detail["candidate"]
                ]
                reports.append(report)
                details.append(detail)

        # Save the fitted selector inputs for later continuous replay.
        np.savez_compressed(
            out / f"selector_h{horizon}.npz",
            x_train=train_x,
            relative_loss=target,
            candidate_ids=np.asarray(candidates),
            kp=pairs.kp.to_numpy(),
            ki=pairs.ki.to_numpy(),
            features=np.asarray(FEATURES),
            baseline_position=np.asarray(baseline),
        )

    summary = pd.DataFrame(reports)
    summary.to_csv(out / "summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(
        out / "selected_windows.csv", index=False
    )
    pd.DataFrame(tuning_rows).to_csv(
        out / "train_cv_search.csv", index=False
    )
    (out / "protocol.json").write_text(
        json.dumps({
            "source": str(source.resolve()),
            "features": FEATURES,
            "settings": settings,
            "validation_used_for_tuning": False,
            "scope": "Local windows from baseline states; not continuous replay.",
            "weighting": "Equal period weighting.",
            "cv": "Leave one training period out.",
        }, indent=2),
        encoding="utf-8"
    )

    print("\nTRAIN-SELECTED SETTINGS:")
    print(json.dumps(settings, indent=2))
    print("\nLOCAL WINDOW RESULTS:")
    print(summary.round(5).to_string(index=False))
    print("\nResults:", out.resolve())


if __name__ == "__main__":
    main()
