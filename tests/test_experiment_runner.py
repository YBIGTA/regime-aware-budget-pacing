import csv
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from simulator.experiment_config import (
    CompetitionCondition,
    ExperimentConfig,
    Scenario,
)
from simulator.experiment_runner import (
    ScenarioServingSystem,
    run_experiment,
    run_simulation,
    validate_trace,
)


def make_scenario(scenario_id="changing"):
    return Scenario.from_dict(
        {
            "id": scenario_id,
            "minutes": 4,
            "campaigns": {"count": 2, "daily_budget": 10.0},
            "initial_condition": {
                "traffic": {"dc": 2, "amplitude": 0, "phase": 0.5},
                "competition": {
                    "untracked_bid_factor": 1,
                    "untracked_bid_price_multiplier": 0.5,
                },
            },
            "changes": [
                {
                    "minute": 2,
                    "traffic": {"dc": 5},
                    "competition": {
                        "untracked_bid_factor": 3,
                        "untracked_bid_price_multiplier": 2,
                    },
                }
            ],
        }
    )


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


class TestExperimentConfig(unittest.TestCase):
    def test_condition_schedule_resolves_partial_updates(self):
        scenario = make_scenario()

        rows = scenario.condition_rows()

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["start_minute"], 0)
        self.assertEqual(rows[0]["end_minute"], 1)
        self.assertEqual(rows[1]["start_minute"], 2)
        self.assertEqual(rows[1]["end_minute"], 3)
        self.assertEqual(rows[1]["traffic_dc"], 5.0)
        self.assertEqual(rows[1]["traffic_amplitude"], 0.0)
        self.assertEqual(rows[1]["untracked_bid_factor"], 3.0)

    def test_invalid_change_minute_is_rejected(self):
        data = {
            "id": "invalid",
            "minutes": 3,
            "campaigns": {"count": 1},
            "initial_condition": {
                "traffic": {"dc": 1, "amplitude": 0},
                "competition": {"untracked_bid_factor": 1},
            },
            "changes": [{"minute": 3, "traffic": {"dc": 2}}],
        }

        with self.assertRaises(ValueError):
            Scenario.from_dict(data)


class TestSimulationRunner(unittest.TestCase):
    def test_competition_price_multiplier_scales_same_base_bid(self):
        first = ScenarioServingSystem(
            competition_rng=np.random.default_rng(10)
        )
        second = ScenarioServingSystem(
            competition_rng=np.random.default_rng(10)
        )
        first.set_competition(CompetitionCondition(1.0, 1.0))
        second.set_competition(CompetitionCondition(1.0, 2.0))

        first_bid = first._generate_untracked_bids(1)[0].amount
        second_bid = second._generate_untracked_bids(1)[0].amount

        self.assertAlmostEqual(second_bid, first_bid * 2)

    def test_competition_factor_controls_untracked_bid_count(self):
        serving_system = ScenarioServingSystem(
            competition_rng=np.random.default_rng(10)
        )
        serving_system.set_competition(CompetitionCondition(2.5, 1.0))

        bids = serving_system._generate_untracked_bids(4)

        self.assertEqual(len(bids), 10)

    def test_competition_statistics_record_observed_bids(self):
        serving_system = ScenarioServingSystem(
            competition_rng=np.random.default_rng(10)
        )
        serving_system.set_competition(CompetitionCondition(1.0, 1.0))
        serving_system.start_minute()

        bids = serving_system._generate_untracked_bids(2)
        bids += serving_system._generate_untracked_bids(1)
        statistics = serving_system.get_competition_statistics(
            auction_count=2
        )
        amounts = np.array([bid.amount for bid in bids])

        self.assertEqual(statistics["competitor_bid_count_total"], 3)
        self.assertEqual(statistics["competitor_bid_count_mean"], 1.5)
        self.assertAlmostEqual(
            statistics["competitor_bid_price_mean"], amounts.mean()
        )
        self.assertAlmostEqual(
            statistics["competitor_bid_price_std"], amounts.std()
        )
        self.assertAlmostEqual(
            statistics["competitor_bid_price_max"], amounts.max()
        )

    def test_same_scenario_and_seed_reproduces_trace(self):
        scenario = make_scenario()
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.csv"
            second = Path(temp_dir) / "second.csv"
            run_simulation(
                scenario, 42, 0.1, 0.01, first, run_id="same-run"
            )
            run_simulation(
                scenario, 42, 0.1, 0.01, second, run_id="same-run"
            )

            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_condition_change_is_written_to_trace(self):
        scenario = make_scenario()
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.csv"
            run_simulation(scenario, 7, 0.1, 0.01, trace)

            rows = read_csv(trace)

        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]["daily_budget"], "10.0")
        self.assertEqual(rows[0]["condition_start_minute"], "0")
        self.assertEqual(rows[-1]["condition_start_minute"], "2")
        self.assertEqual(rows[-1]["traffic_dc"], "5.0")
        self.assertEqual(rows[-1]["untracked_bid_factor"], "3.0")
        self.assertEqual(rows[0]["traffic_mean"], "2.0")
        self.assertEqual(rows[-1]["traffic_mean"], "5.0")
        self.assertIn("competitor_bid_count_total", rows[0])
        self.assertIn("competitor_bid_price_mean", rows[0])

    def test_trace_validation_detects_incorrect_remaining_budget(self):
        scenario = make_scenario()
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.csv"
            run_simulation(scenario, 7, 0.1, 0.01, trace)
            rows = read_csv(trace)
            rows[0]["remaining_budget"] = "999"
            with trace.open("w", newline="", encoding="utf-8") as output:
                writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            validation = validate_trace(trace, scenario)

        self.assertEqual(validation["status"], "failed")
        self.assertIn("remaining budget mismatch", validation["error"])

    def test_last_minute_keeps_daily_totals_before_yahoo_resets_them(self):
        scenario = Scenario.from_dict(
            {
                "id": "full-day",
                "minutes": 1440,
                "campaigns": {"count": 1, "daily_budget": 10},
                "initial_condition": {
                    "traffic": {"dc": 1, "amplitude": 0},
                    "competition": {"untracked_bid_factor": 0},
                },
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            trace = Path(temp_dir) / "trace.csv"
            run_simulation(scenario, 3, 0.1, 0.01, trace)
            rows = read_csv(trace)

        self.assertEqual(rows[-1]["target_spend_cumulative"], "10.0")
        self.assertGreaterEqual(
            float(rows[-1]["actual_spend_cumulative"]),
            float(rows[-2]["actual_spend_cumulative"]),
        )

    def test_pi_coefficients_share_the_same_auction_opportunities(self):
        scenario = make_scenario()
        with tempfile.TemporaryDirectory() as temp_dir:
            first = Path(temp_dir) / "first.csv"
            second = Path(temp_dir) / "second.csv"
            run_simulation(scenario, 99, 0.0, 0.0, first)
            run_simulation(scenario, 99, 0.8, 0.1, second)

            first_rows = read_csv(first)
            second_rows = read_csv(second)

        first_auctions = [row["actual_auctions"] for row in first_rows]
        second_auctions = [row["actual_auctions"] for row in second_rows]
        self.assertEqual(first_auctions, second_auctions)

    def test_batch_runs_full_scenario_seed_coefficient_product(self):
        scenario_a = make_scenario("scenario-a")
        scenario_b = make_scenario("scenario-b")
        config = ExperimentConfig.from_dict(
            {
                "experiment_id": "batch-test",
                "seeds": [1, 2],
                "pi_coefficients": [
                    {"kp": 0.1, "ki": 0.01},
                    {"kp": 0.2, "ki": 0.02},
                ],
                "scenarios": [
                    {
                        "id": scenario_a.scenario_id,
                        "minutes": scenario_a.minutes,
                        "campaigns": {
                            "count": scenario_a.campaigns.count,
                            "daily_budget": scenario_a.campaigns.daily_budget,
                        },
                        "initial_condition": {
                            "traffic": {"dc": 1, "amplitude": 0},
                            "competition": {"untracked_bid_factor": 0},
                        },
                    },
                    {
                        "id": scenario_b.scenario_id,
                        "minutes": scenario_b.minutes,
                        "campaigns": {
                            "count": scenario_b.campaigns.count,
                            "daily_budget": scenario_b.campaigns.daily_budget,
                        },
                        "initial_condition": {
                            "traffic": {"dc": 1, "amplitude": 0},
                            "competition": {"untracked_bid_factor": 0},
                        },
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = run_experiment(config, temp_dir)
            experiment_dir = Path(temp_dir) / "batch-test"

            self.assertEqual(len(manifest), 8)
            self.assertTrue(
                all(row["status"] == "completed" for row in manifest)
            )
            self.assertTrue(
                all(
                    row["validation_status"] == "passed"
                    for row in manifest
                )
            )
            self.assertTrue((experiment_dir / "manifest.csv").exists())
            self.assertTrue((experiment_dir / "conditions.csv").exists())
            self.assertEqual(
                len(list((experiment_dir / "runs").glob("*/trace.csv"))),
                8,
            )

    def test_batch_skips_existing_traces_and_reruns_incomplete_ones(self):
        scenario = make_scenario("scenario-a")
        config = ExperimentConfig.from_dict(
            {
                "experiment_id": "resume-test",
                "seeds": [1],
                "pi_coefficients": [
                    {"kp": 0.1, "ki": 0.01},
                    {"kp": 0.2, "ki": 0.02},
                ],
                "scenarios": [
                    {
                        "id": scenario.scenario_id,
                        "minutes": scenario.minutes,
                        "campaigns": {
                            "count": scenario.campaigns.count,
                            "daily_budget": scenario.campaigns.daily_budget,
                        },
                        "initial_condition": {
                            "traffic": {"dc": 1, "amplitude": 0},
                            "competition": {"untracked_bid_factor": 0},
                        },
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            first = run_experiment(config, temp_dir)
            kept_trace = Path(first[0]["trace_path"])
            kept_mtime = kept_trace.stat().st_mtime_ns
            interrupted_trace = Path(first[1]["trace_path"])
            expected = interrupted_trace.read_bytes()
            interrupted_trace.rename(
                interrupted_trace.with_name("trace.csv.partial")
            )

            second = run_experiment(config, temp_dir)

            self.assertEqual(
                [row["status"] for row in second], ["skipped", "completed"]
            )
            self.assertEqual(second[0]["validation_status"], "passed")
            self.assertEqual(kept_trace.stat().st_mtime_ns, kept_mtime)
            self.assertEqual(interrupted_trace.read_bytes(), expected)
            self.assertFalse(
                interrupted_trace.with_name("trace.csv.partial").exists()
            )

    def test_parallel_workers_reproduce_sequential_traces(self):
        scenario = make_scenario("scenario-a")
        config = ExperimentConfig.from_dict(
            {
                "experiment_id": "parallel-test",
                "seeds": [1, 2],
                "pi_coefficients": [{"kp": 0.1, "ki": 0.01}],
                "scenarios": [
                    {
                        "id": scenario.scenario_id,
                        "minutes": scenario.minutes,
                        "campaigns": {
                            "count": scenario.campaigns.count,
                            "daily_budget": scenario.campaigns.daily_budget,
                        },
                        "initial_condition": {
                            "traffic": {"dc": 2, "amplitude": 0},
                            "competition": {"untracked_bid_factor": 1},
                        },
                    },
                ],
            }
        )
        with tempfile.TemporaryDirectory() as sequential_dir, \
                tempfile.TemporaryDirectory() as parallel_dir:
            sequential = run_experiment(config, sequential_dir, workers=1)
            parallel = run_experiment(config, parallel_dir, workers=2)

            self.assertEqual(
                [row["status"] for row in parallel], ["completed"] * 2
            )
            self.assertEqual(
                [row["run_id"] for row in parallel],
                [row["run_id"] for row in sequential],
            )
            for sequential_row, parallel_row in zip(sequential, parallel):
                self.assertEqual(
                    Path(parallel_row["trace_path"]).read_bytes(),
                    Path(sequential_row["trace_path"]).read_bytes(),
                )

if __name__ == "__main__":
    unittest.main()
