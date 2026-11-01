from dataclasses import asdict
import csv
import json
import math
from pathlib import Path
import random
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TEAM_SRC = PROJECT_ROOT / "src"
YAHOO_ROOT = PROJECT_ROOT / "external" / "yahoo-budget-pacing"
sys.path.insert(0, str(TEAM_SRC))
sys.path.insert(1, str(YAHOO_ROOT))

from controller.fixed_pi import FixedPI
from simulator.experiment_config import ExperimentConfig, Scenario
import src.configuration as yahoo_config
from src import constants, system_generation_utils
from src.constants import AuctionType
from src.system.auction import AuctionFP
from src.system.bid import Bid
from src.system.clock import Clock
from src.system.daily_cosine import DailyCosineWave
from src.system.marketplace import Marketplace
from src.system.serving_system import ServingSystem


TRACE_FIELDS = [
    "run_id",
    "scenario_id",
    "environment_seed",
    "kp",
    "ki",
    "minute",
    "condition_start_minute",
    "traffic_dc",
    "traffic_amplitude",
    "traffic_phase",
    "untracked_bid_factor",
    "untracked_bid_price_multiplier",
    "traffic_mean",
    "actual_auctions",
    "competitor_bid_count_total",
    "competitor_bid_count_mean",
    "competitor_bid_price_mean",
    "competitor_bid_price_std",
    "competitor_bid_price_max",
    "campaign_id",
    "daily_budget",
    "target_spend_cumulative",
    "actual_spend_cumulative",
    "spend_error",
    "spend_this_minute",
    "wins_this_minute",
    "remaining_budget",
    "pacing_signal_used",
    "pacing_signal_next",
    "integral_error",
]


CONDITION_FIELDS = [
    "scenario_id",
    "start_minute",
    "end_minute",
    "campaign_count",
    "campaign_daily_budget",
    "traffic_dc",
    "traffic_amplitude",
    "traffic_phase",
    "untracked_bid_factor",
    "untracked_bid_price_multiplier",
]


MANIFEST_FIELDS = [
    "run_id",
    "scenario_id",
    "environment_seed",
    "kp",
    "ki",
    "status",
    "rows",
    "trace_path",
    "metadata_path",
    "validation_status",
    "validation_error",
    "error",
]


class ScenarioMarketplace(Marketplace):
    def __init__(self, *args, environment_rng, **kwargs):
        super().__init__(*args, **kwargs)
        self.environment_rng = environment_rng

    def set_traffic(self, traffic):
        self.traffic_mean_cos_wave = DailyCosineWave(
            dc=traffic.dc,
            amplitude=traffic.amplitude,
            phase=traffic.phase,
        )

    def _sample_current_num_of_auctions(self):
        mean = self._calculate_current_mean_num_of_auctions()
        return int(self.environment_rng.poisson(mean))

    def _generate_auction(self, user_properties_probabilities):
        if self.auction_type != AuctionType.FP:
            raise NotImplementedError
        user_properties = {
            feature: self.environment_rng.choice(
                list(yahoo_config.user_properties[feature].keys()),
                p=user_properties_probabilities[feature],
            )
            for feature in user_properties_probabilities
        }
        return AuctionFP(user_properties=user_properties)


class ScenarioServingSystem(ServingSystem):
    def __init__(self, *args, competition_rng, **kwargs):
        super().__init__(*args, **kwargs)
        self.competition_rng = competition_rng
        self.untracked_bid_factor = 0.0
        self.untracked_bid_price_multiplier = 1.0
        self.start_minute()

    def start_minute(self):
        self.competitor_bid_count_total = 0
        self.competitor_bid_price_sum = 0.0
        self.competitor_bid_price_sum_squares = 0.0
        self.competitor_bid_price_max = 0.0

    def set_competition(self, competition):
        self.untracked_bid_factor = competition.untracked_bid_factor
        self.untracked_bid_price_multiplier = (
            competition.untracked_bid_price_multiplier
        )

    def _calculate_number_of_untracked_bids(self, num_relevant_campaigns):
        return round(self.untracked_bid_factor * num_relevant_campaigns)

    def _generate_untracked_bids(self, num_relevant_campaigns):
        num_untracked_bids = self._calculate_number_of_untracked_bids(
            num_relevant_campaigns
        )
        if num_untracked_bids == 0:
            return []
        if self.untracked_bids_cache.size < num_untracked_bids:
            sampled_batch_size = max(
                num_untracked_bids, yahoo_config.bid_sampling_batch_size
            )
            self.untracked_bids_cache = (
                yahoo_config.untracked_bids_distribution.rvs(
                    size=sampled_batch_size,
                    random_state=self.competition_rng,
                )
            )
        sampled_bids = self.untracked_bids_cache[-num_untracked_bids:]
        self.untracked_bids_cache = self.untracked_bids_cache[
            :-num_untracked_bids
        ]
        sampled_bids = sampled_bids * self.untracked_bid_price_multiplier
        self.competitor_bid_count_total += sampled_bids.size
        self.competitor_bid_price_sum += float(sampled_bids.sum())
        self.competitor_bid_price_sum_squares += float(
            np.square(sampled_bids).sum()
        )
        self.competitor_bid_price_max = max(
            self.competitor_bid_price_max,
            float(sampled_bids.max()),
        )
        return [
            Bid(campaign_id=f"untracked_campaign_{index}", amount=amount)
            for index, amount in enumerate(sampled_bids)
        ]

    def get_competition_statistics(self, auction_count):
        count = self.competitor_bid_count_total
        if count == 0:
            price_mean = 0.0
            price_std = 0.0
        else:
            price_mean = self.competitor_bid_price_sum / count
            variance = max(
                self.competitor_bid_price_sum_squares / count
                - price_mean**2,
                0.0,
            )
            price_std = math.sqrt(variance)
        return {
            "competitor_bid_count_total": count,
            "competitor_bid_count_mean": (
                count / auction_count if auction_count else 0.0
            ),
            "competitor_bid_price_mean": price_mean,
            "competitor_bid_price_std": price_std,
            "competitor_bid_price_max": self.competitor_bid_price_max,
        }


def _set_campaign_daily_budget(campaigns, daily_budget):
    if daily_budget is None:
        return
    bids_distribution = (
        system_generation_utils.generate_bid_distribution_for_budget(
            daily_budget
        )
    )
    max_bid = bids_distribution.mean() * yahoo_config.max_bid_factor_of_bids_mean
    for campaign in campaigns:
        campaign.daily_budget = daily_budget
        campaign.total_budget = daily_budget * yahoo_config.num_days_to_simulate
        campaign.bids_distribution = bids_distribution
        campaign.max_bid = max_bid
        campaign.bids_cache = np.array([])


def _float_slug(value):
    return format(value, ".12g").replace("-", "m").replace(".", "p")


def make_run_id(scenario_id, seed, kp, ki):
    return (
        f"{scenario_id}__seed{seed}__kp{_float_slug(kp)}"
        f"__ki{_float_slug(ki)}"
    )


def validate_trace(trace_path, scenario: Scenario):
    errors = []

    def add_error(message):
        if message not in errors and len(errors) < 20:
            errors.append(message)

    with Path(trace_path).open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        missing_fields = set(TRACE_FIELDS) - set(reader.fieldnames or [])
        rows = list(reader)

    if missing_fields:
        add_error(
            "missing columns: " + ", ".join(sorted(missing_fields))
        )
        return {
            "status": "failed",
            "error": "; ".join(errors),
        }

    expected_row_count = scenario.minutes * scenario.campaigns.count
    if len(rows) != expected_row_count:
        add_error(
            f"expected {expected_row_count} rows, found {len(rows)}"
        )

    expected_conditions = {}
    active_index = 0
    for minute in range(scenario.minutes):
        if (
            active_index + 1 < len(scenario.schedule)
            and scenario.schedule[active_index + 1].minute == minute
        ):
            active_index += 1
        expected_conditions[minute] = scenario.schedule[active_index]

    rows_by_campaign = {}
    environment_by_minute = {}
    seen_keys = set()
    for row in rows:
        try:
            minute = int(row["minute"])
            campaign_id = row["campaign_id"]
            key = (campaign_id, minute)
            if key in seen_keys:
                add_error(f"duplicate row: campaign={campaign_id}, minute={minute}")
            seen_keys.add(key)
            rows_by_campaign.setdefault(campaign_id, []).append(row)

            if not 0 <= minute < scenario.minutes:
                add_error(f"minute out of range: {minute}")
                continue
            scheduled = expected_conditions[minute]
            condition = scheduled.condition
            expected_values = {
                "condition_start_minute": scheduled.minute,
                "traffic_dc": condition.traffic.dc,
                "traffic_amplitude": condition.traffic.amplitude,
                "traffic_phase": condition.traffic.phase,
                "untracked_bid_factor": (
                    condition.competition.untracked_bid_factor
                ),
                "untracked_bid_price_multiplier": (
                    condition.competition.untracked_bid_price_multiplier
                ),
            }
            for field, expected in expected_values.items():
                if not math.isclose(
                    float(row[field]),
                    float(expected),
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    add_error(f"condition mismatch: {field} at minute {minute}")

            environment_values = tuple(
                row[field]
                for field in (
                    "actual_auctions",
                    "competitor_bid_count_total",
                    "competitor_bid_count_mean",
                    "competitor_bid_price_mean",
                    "competitor_bid_price_std",
                    "competitor_bid_price_max",
                )
            )
            if minute in environment_by_minute:
                if environment_by_minute[minute] != environment_values:
                    add_error(
                        f"environment values differ between campaigns at minute {minute}"
                    )
            else:
                environment_by_minute[minute] = environment_values
                auction_count = int(row["actual_auctions"])
                bid_count = int(row["competitor_bid_count_total"])
                bid_count_mean = float(row["competitor_bid_count_mean"])
                expected_mean = (
                    bid_count / auction_count if auction_count else 0.0
                )
                if not math.isclose(
                    bid_count_mean,
                    expected_mean,
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    add_error(
                        f"competitor bid count mean mismatch at minute {minute}"
                    )
                price_mean = float(row["competitor_bid_price_mean"])
                price_std = float(row["competitor_bid_price_std"])
                price_max = float(row["competitor_bid_price_max"])
                if bid_count == 0 and any(
                    value != 0.0
                    for value in (price_mean, price_std, price_max)
                ):
                    add_error(
                        f"competitor price statistics must be zero at minute {minute}"
                    )
                if price_std < 0 or price_max < price_mean:
                    add_error(
                        f"invalid competitor price statistics at minute {minute}"
                    )
        except (KeyError, TypeError, ValueError) as error:
            add_error(f"invalid row value: {error}")

    if len(rows_by_campaign) != scenario.campaigns.count:
        add_error(
            f"expected {scenario.campaigns.count} campaigns, "
            f"found {len(rows_by_campaign)}"
        )

    expected_minutes = set(range(scenario.minutes))
    for campaign_id, campaign_rows in rows_by_campaign.items():
        actual_minutes = {int(row["minute"]) for row in campaign_rows}
        if actual_minutes != expected_minutes:
            add_error(f"missing or extra minutes for campaign {campaign_id}")
            continue
        previous_spend = 0.0
        for row in sorted(campaign_rows, key=lambda item: int(item["minute"])):
            minute = int(row["minute"])
            daily_budget = float(row["daily_budget"])
            cumulative_spend = float(row["actual_spend_cumulative"])
            spend_this_minute = float(row["spend_this_minute"])
            remaining_budget = float(row["remaining_budget"])
            if cumulative_spend + 1e-9 < previous_spend:
                add_error(
                    f"cumulative spend decreased: campaign={campaign_id}, minute={minute}"
                )
            if not math.isclose(
                cumulative_spend - previous_spend,
                spend_this_minute,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                add_error(
                    f"minute spend mismatch: campaign={campaign_id}, minute={minute}"
                )
            if not math.isclose(
                daily_budget - cumulative_spend,
                remaining_budget,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                add_error(
                    f"remaining budget mismatch: campaign={campaign_id}, minute={minute}"
                )
            previous_spend = cumulative_spend

    return {
        "status": "passed" if not errors else "failed",
        "error": "; ".join(errors),
    }


def run_simulation(
    scenario: Scenario,
    seed: int,
    kp: float,
    ki: float,
    trace_path,
    metadata_path=None,
    run_id=None,
    overwrite=False,
):
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**32
    ):
        raise ValueError("seed must be an integer between 0 and 2**32 - 1")
    trace_path = Path(trace_path)
    if trace_path.exists() and not overwrite:
        raise FileExistsError(f"trace already exists: {trace_path}")
    if metadata_path is None:
        metadata_path = trace_path.with_suffix(".metadata.json")
    metadata_path = Path(metadata_path)
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"metadata already exists: {metadata_path}")
    if run_id is None:
        run_id = make_run_id(scenario.scenario_id, seed, kp, ki)

    random.seed(seed)
    np.random.seed(seed)
    yahoo_config.campaign_generation_seed_value = seed
    Clock.reset()

    campaigns = system_generation_utils.generate_campaigns(
        scenario.campaigns.count
    )
    _set_campaign_daily_budget(
        campaigns, scenario.campaigns.daily_budget
    )
    pacing_system = FixedPI(kp=kp, ki=ki)

    seed_sequence = np.random.SeedSequence(seed)
    environment_seed, competition_seed = seed_sequence.spawn(2)
    serving_system = ScenarioServingSystem(
        pacing_system=pacing_system,
        tracked_campaigns=campaigns,
        competition_rng=np.random.default_rng(competition_seed),
    )
    initial = scenario.schedule[0].condition
    marketplace = ScenarioMarketplace(
        serving_system=serving_system,
        auction_type=yahoo_config.auction_type,
        traffic_mean_cos_wave=DailyCosineWave(
            dc=initial.traffic.dc,
            amplitude=initial.traffic.amplitude,
            phase=initial.traffic.phase,
        ),
        environment_rng=np.random.default_rng(environment_seed),
    )
    serving_system.set_competition(initial.competition)

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_by_minute = {
        scheduled.minute: scheduled for scheduled in scenario.schedule
    }
    active = scenario.schedule[0]
    row_count = 0

    with trace_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=TRACE_FIELDS)
        writer.writeheader()

        for minute in range(scenario.minutes):
            if minute in schedule_by_minute:
                active = schedule_by_minute[minute]
                marketplace.set_traffic(active.condition.traffic)
                serving_system.set_competition(active.condition.competition)

            traffic_mean = marketplace._calculate_current_mean_num_of_auctions()
            auctions = marketplace._generate_auctions()
            serving_system.start_minute()
            spend_before = {
                campaign.id: campaign.spent_today() for campaign in campaigns
            }
            wins_before = {
                campaign.id: campaign.num_auctions_won_today()
                for campaign in campaigns
            }
            signal_before = {
                campaign.id: pacing_system.get_pacing_signal(campaign.id)
                for campaign in campaigns
            }

            marketplace._run_auctions(auctions)
            competition_statistics = (
                serving_system.get_competition_statistics(len(auctions))
            )
            spend_after = {
                campaign.id: campaign.spent_today() for campaign in campaigns
            }
            wins_after = {
                campaign.id: campaign.num_auctions_won_today()
                for campaign in campaigns
            }
            serving_system.end_iteration()

            condition = active.condition
            for campaign in campaigns:
                state = pacing_system.get_state(campaign.id)
                actual_spend = spend_after[campaign.id]
                target_spend = (
                    campaign.daily_budget
                    * (Clock.minute_in_day() + 1)
                    / constants.num_minutes_in_day
                )
                writer.writerow(
                    {
                        "run_id": run_id,
                        "scenario_id": scenario.scenario_id,
                        "environment_seed": seed,
                        "kp": kp,
                        "ki": ki,
                        "minute": minute,
                        "condition_start_minute": active.minute,
                        "traffic_dc": condition.traffic.dc,
                        "traffic_amplitude": condition.traffic.amplitude,
                        "traffic_phase": condition.traffic.phase,
                        "untracked_bid_factor": (
                            condition.competition.untracked_bid_factor
                        ),
                        "untracked_bid_price_multiplier": (
                            condition.competition.untracked_bid_price_multiplier
                        ),
                        "traffic_mean": traffic_mean,
                        "actual_auctions": len(auctions),
                        **competition_statistics,
                        "campaign_id": campaign.id,
                        "daily_budget": campaign.daily_budget,
                        "target_spend_cumulative": target_spend,
                        "actual_spend_cumulative": actual_spend,
                        "spend_error": target_spend - actual_spend,
                        "spend_this_minute": (
                            actual_spend - spend_before[campaign.id]
                        ),
                        "wins_this_minute": (
                            wins_after[campaign.id] - wins_before[campaign.id]
                        ),
                        "remaining_budget": (
                            campaign.daily_budget - actual_spend
                        ),
                        "pacing_signal_used": signal_before[campaign.id],
                        "pacing_signal_next": state.pacing_signal,
                        "integral_error": state.integral_error,
                    }
                )
                row_count += 1
            Clock.advance()

    validation = validate_trace(trace_path, scenario)
    metadata = {
        "run_id": run_id,
        "scenario_id": scenario.scenario_id,
        "environment_seed": seed,
        "kp": kp,
        "ki": ki,
        "rows": row_count,
        "trace_path": str(trace_path),
        "scenario": asdict(scenario),
        "condition_table": scenario.condition_rows(),
        "validation": validation,
    }
    with metadata_path.open("w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, ensure_ascii=False, indent=2)
        metadata_file.write("\n")

    return {
        "run_id": run_id,
        "scenario_id": scenario.scenario_id,
        "environment_seed": seed,
        "kp": kp,
        "ki": ki,
        "status": (
            "completed"
            if validation["status"] == "passed"
            else "validation_failed"
        ),
        "rows": row_count,
        "trace_path": str(trace_path),
        "metadata_path": str(metadata_path),
        "validation_status": validation["status"],
        "validation_error": validation["error"],
        "error": "",
    }


def _write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(config: ExperimentConfig, output_root, overwrite=False):
    experiment_dir = Path(output_root) / config.experiment_id
    manifest_path = experiment_dir / "manifest.csv"
    conditions_path = experiment_dir / "conditions.csv"
    runs_path = experiment_dir / "runs"
    if (
        not overwrite
        and (manifest_path.exists() or runs_path.exists())
    ):
        raise FileExistsError(
            f"experiment output already exists: {experiment_dir}"
        )
    experiment_dir.mkdir(parents=True, exist_ok=True)

    condition_rows = [
        row
        for scenario in config.scenarios
        for row in scenario.condition_rows()
    ]
    _write_csv(conditions_path, CONDITION_FIELDS, condition_rows)

    total_runs = (
        len(config.scenarios)
        * len(config.seeds)
        * len(config.pi_coefficients)
    )
    manifest_rows = []
    run_number = 0
    for scenario in config.scenarios:
        for seed in config.seeds:
            for coefficients in config.pi_coefficients:
                run_number += 1
                run_id = make_run_id(
                    scenario.scenario_id,
                    seed,
                    coefficients.kp,
                    coefficients.ki,
                )
                run_dir = experiment_dir / "runs" / run_id
                trace_path = run_dir / "trace.csv"
                metadata_path = run_dir / "metadata.json"
                print(f"[{run_number}/{total_runs}] {run_id}")
                try:
                    result = run_simulation(
                        scenario=scenario,
                        seed=seed,
                        kp=coefficients.kp,
                        ki=coefficients.ki,
                        trace_path=trace_path,
                        metadata_path=metadata_path,
                        run_id=run_id,
                        overwrite=overwrite,
                    )
                except Exception as error:
                    result = {
                        "run_id": run_id,
                        "scenario_id": scenario.scenario_id,
                        "environment_seed": seed,
                        "kp": coefficients.kp,
                        "ki": coefficients.ki,
                        "status": "failed",
                        "rows": 0,
                        "trace_path": str(trace_path),
                        "metadata_path": str(metadata_path),
                        "validation_status": "not_run",
                        "validation_error": "",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    print(f"failed: {result['error']}")
                if result["status"] == "validation_failed":
                    print(
                        "validation failed: "
                        f"{result['validation_error']}"
                    )
                manifest_rows.append(result)
                _write_csv(manifest_path, MANIFEST_FIELDS, manifest_rows)

    completed = sum(row["status"] == "completed" for row in manifest_rows)
    print(f"experiment_dir={experiment_dir}")
    print(f"completed={completed} failed={total_runs - completed}")
    return manifest_rows
