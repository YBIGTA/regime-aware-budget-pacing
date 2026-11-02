from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from preprocessing.state_table import (
    COLUMNS,
    ENV_FEATURES,
    build_state_table,
    environment_features,
)
from simulator.experiment_config import ExperimentConfig
from simulator.experiment_runner import run_experiment


def make_minute_frame(minutes=80, seed=0):
    """분마다 실제 경쟁 입찰가 표본을 만들고, trace와 같은 요약 통계를 함께 반환한다."""
    rng = np.random.default_rng(seed)
    auctions = rng.integers(50, 200, size=minutes)
    bids = []
    rows = []
    for minute in range(minutes):
        count = int(rng.integers(0, 40)) if minute % 7 else 0
        prices = rng.lognormal(-2, 1, size=count)
        bids.append(prices)
        rows.append(
            {
                "minute": minute,
                "actual_auctions": auctions[minute],
                "competitor_bid_count_total": count,
                "competitor_bid_count_mean": count / auctions[minute],
                "competitor_bid_price_mean": prices.mean() if count else 0.0,
                "competitor_bid_price_std": prices.std() if count else 0.0,
                "competitor_bid_price_max": prices.max() if count else 0.0,
                "traffic_dc": 100.0,
                "traffic_amplitude": 0.5,
                "untracked_bid_factor": 1.0,
                "untracked_bid_price_multiplier": 1.0,
                "condition_start_minute": 0,
                "traffic_mean": 100.0,
            }
        )
    return pd.DataFrame(rows), bids


class TestEnvironmentFeatures(unittest.TestCase):
    def test_window_features_match_direct_computation(self):
        frame, bids = make_minute_frame()
        features = environment_features(frame)
        auctions = frame["actual_auctions"].to_numpy(dtype=float)

        for window in (15, 60):
            w = f"{window}m"
            for minute in (0, 1, 13, 14, 40, 59, 60, 79):
                start = max(0, minute - window + 1)
                y = auctions[start:minute + 1]
                x = np.arange(start, minute + 1)
                prices = np.concatenate(bids[start:minute + 1])
                row = features.iloc[minute]

                self.assertAlmostEqual(row[f"env_auctions_mean_{w}"], y.mean())
                self.assertAlmostEqual(
                    row[f"env_auctions_cv_{w}"], y.std() / y.mean()
                )
                expected_trend = (
                    np.polyfit(x, y, 1)[0] * (len(y) - 1) / y.mean()
                    if len(y) > 1 else 0.0
                )
                self.assertAlmostEqual(
                    row[f"env_auctions_trend_{w}"], expected_trend
                )
                self.assertAlmostEqual(
                    row[f"env_comp_bids_per_auction_{w}"],
                    len(prices) / y.sum(),
                )
                if len(prices):
                    self.assertAlmostEqual(
                        row[f"env_comp_price_mean_{w}"], prices.mean()
                    )
                    self.assertAlmostEqual(
                        row[f"env_comp_price_std_{w}"], prices.std()
                    )
                    self.assertAlmostEqual(
                        row[f"env_comp_price_max_{w}"], prices.max()
                    )
                self.assertEqual(
                    row[f"window_filled_{w}"], minute >= window - 1
                )

    def test_features_use_only_past_and_current_minutes(self):
        frame, _ = make_minute_frame()
        changed = frame.copy()
        future = changed["minute"] > 30
        changed.loc[future, "actual_auctions"] *= 10
        changed.loc[future, "competitor_bid_price_mean"] *= 10
        changed.loc[future, "competitor_bid_price_max"] *= 10

        original = environment_features(frame).iloc[:31]
        modified = environment_features(changed).iloc[:31]
        pd.testing.assert_frame_equal(original, modified)

    def test_every_feature_is_documented(self):
        documented = {column.name for column in COLUMNS}
        features = environment_features(make_minute_frame()[0])
        self.assertEqual(set(features.columns) - documented, set())
        self.assertTrue(set(ENV_FEATURES) <= documented)


class TestBuildStateTable(unittest.TestCase):
    def test_build_from_batch_output(self):
        config = ExperimentConfig.from_dict(
            {
                "experiment_id": "state-test",
                "seeds": [1, 2, 3],
                "pi_coefficients": [
                    {"kp": 0.1, "ki": 0.01},
                    {"kp": 0.2, "ki": 0.02},
                ],
                "scenarios": [
                    {
                        "id": "scenario-a",
                        "minutes": 20,
                        "campaigns": {"count": 2, "daily_budget": 10.0},
                        "initial_condition": {
                            "traffic": {"dc": 3, "amplitude": 0},
                            "competition": {"untracked_bid_factor": 1},
                        },
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_experiment(config, temp_dir)
            output_dir = Path(temp_dir) / "state"
            info = build_state_table(
                Path(temp_dir) / "state-test",
                output_dir,
                {"train": (1, 2), "validation": (3,)},
            )
            table = pd.read_parquet(output_dir / "state_table.parquet")

        # 시드 3개 x PI 2개 x 캠페인 2개 x 20분
        self.assertEqual(info["rows"], 3 * 2 * 2 * 20)
        self.assertEqual(len(table), info["rows"])
        self.assertFalse(
            table.duplicated(
                ["scenario_id", "seed", "kp", "ki", "campaign_id", "minute"]
            ).any()
        )

        cluster = table[table["cluster_row"]]
        self.assertEqual(len(cluster), 3 * 20)
        self.assertFalse(
            cluster.duplicated(["scenario_id", "seed", "minute"]).any()
        )

        self.assertEqual(
            table.groupby("seed")["split"].unique().map(tuple).to_dict(),
            {1: ("train",), 2: ("train",), 3: ("validation",)},
        )

        environment_columns = [c for c in table.columns if c.startswith("env_")]
        per_minute = table.groupby(["seed", "minute"])[environment_columns]
        self.assertTrue((per_minute.nunique(dropna=False) <= 1).all().all())

        self.assertTrue(
            np.allclose(
                table["ctrl_spend_error_ratio"],
                table["ctrl_spend_error"] / table["ctrl_daily_budget"],
            )
        )

    def test_multiple_experiment_dirs_are_merged_without_duplicates(self):
        def config(experiment_id, seeds):
            return ExperimentConfig.from_dict(
                {
                    "experiment_id": experiment_id,
                    "seeds": seeds,
                    "pi_coefficients": [{"kp": 0.1, "ki": 0.01}],
                    "scenarios": [
                        {
                            "id": "scenario-a",
                            "minutes": 3,
                            "campaigns": {"count": 1, "daily_budget": 10.0},
                            "initial_condition": {
                                "traffic": {"dc": 1, "amplitude": 0},
                                "competition": {"untracked_bid_factor": 0},
                            },
                        }
                    ],
                }
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            run_experiment(config("first", [1]), temp_dir)
            # 시드 1은 두 폴더에 모두 있다.
            run_experiment(config("second", [1, 2]), temp_dir)
            output_dir = Path(temp_dir) / "state"
            info = build_state_table(
                [Path(temp_dir) / "first", Path(temp_dir) / "second"],
                output_dir,
                {"train": (1,), "test": (2,)},
            )
            table = pd.read_parquet(output_dir / "state_table.parquet")

        self.assertEqual(info["runs_used"], 2)
        self.assertEqual(len(table), 2 * 3)
        self.assertEqual(
            table.groupby("seed")["split"].unique().map(tuple).to_dict(),
            {1: ("train",), 2: ("test",)},
        )

    def test_unassigned_seed_is_rejected(self):
        config = ExperimentConfig.from_dict(
            {
                "experiment_id": "split-test",
                "seeds": [1, 9],
                "pi_coefficients": [{"kp": 0.1, "ki": 0.01}],
                "scenarios": [
                    {
                        "id": "scenario-a",
                        "minutes": 3,
                        "campaigns": {"count": 1, "daily_budget": 10.0},
                        "initial_condition": {
                            "traffic": {"dc": 1, "amplitude": 0},
                            "competition": {"untracked_bid_factor": 0},
                        },
                    }
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            run_experiment(config, temp_dir)
            with self.assertRaisesRegex(ValueError, "seed 9"):
                build_state_table(
                    Path(temp_dir) / "split-test",
                    Path(temp_dir) / "state",
                    {"train": (1,)},
                )


if __name__ == "__main__":
    unittest.main()
