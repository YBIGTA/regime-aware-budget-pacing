from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import csv
import json
import math
import multiprocessing
from pathlib import Path
import random
import sys
import time

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
    progress=None,
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
    # 완료 전에는 임시 파일에 쓰고 마지막에 이름을 바꾼다. trace.csv가 있으면
    # 항상 끝까지 실행되고 metadata까지 저장된 결과라는 뜻이다.
    partial_trace_path = trace_path.with_name(trace_path.name + ".partial")

    with partial_trace_path.open(
        "w", newline="", encoding="utf-8"
    ) as output_file:
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
            if progress is not None:
                progress(minute + 1, scenario.minutes)

    validation = validate_trace(partial_trace_path, scenario)
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
    partial_trace_path.replace(trace_path)

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


def _format_duration(seconds):
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    return f"{minutes}m{seconds:02d}s"


class BatchProgress:
    """배치 전체와 현재 run의 진행률을 터미널에 출력한다.

    터미널에서는 한 줄을 갱신하고, 로그 파일로 리다이렉트된 경우에는
    run마다 10% 단위로 줄을 추가한다.
    """

    def __init__(self, total_minutes, stream=None, refresh_seconds=0.5):
        self.total_minutes = total_minutes
        self.stream = stream if stream is not None else sys.stdout
        self.interactive = self.stream.isatty()
        self.refresh_seconds = refresh_seconds
        self.done_minutes = 0
        self.started_at = time.monotonic()
        self._last_print = 0.0
        self._last_decile = -1
        self._label = ""

    def start_run(self, label):
        self._label = label
        self._run_started_at = time.monotonic()
        self._last_decile = -1

    def update(self, minute, minutes):
        now = time.monotonic()
        finished = minute == minutes
        if self.interactive:
            if not finished and now - self._last_print < self.refresh_seconds:
                return
        else:
            decile = minute * 10 // minutes
            if decile == self._last_decile:
                return
            self._last_decile = decile
        self._last_print = now

        done = self.done_minutes + minute
        overall = done / self.total_minutes if self.total_minutes else 1.0
        elapsed = now - self.started_at
        eta = elapsed * (1 - overall) / overall if overall > 0 else 0.0
        line = (
            f"{self._label} minute {minute}/{minutes} "
            f"({minute / minutes:5.1%}) | overall {overall:5.1%} "
            f"elapsed {_format_duration(elapsed)} "
            f"ETA {_format_duration(eta)}"
        )
        if self.interactive:
            self.stream.write("\r\033[K" + line)
        else:
            self.stream.write(line + "\n")
        self.stream.flush()

    def finish_run(self, minutes):
        self.done_minutes += minutes
        if self.interactive:
            self.stream.write("\n")
        self.stream.write(
            f"  done in {_format_duration(time.monotonic() - self._run_started_at)}\n"
        )
        self.stream.flush()


def _skipped_result(run_id, scenario, seed, coefficients, trace_path,
                    metadata_path):
    validation = {"status": "unknown", "error": ""}
    rows = ""
    if metadata_path.exists():
        with metadata_path.open(encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        validation = metadata.get("validation", validation)
        rows = metadata.get("rows", "")
    return {
        "run_id": run_id,
        "scenario_id": scenario.scenario_id,
        "environment_seed": seed,
        "kp": coefficients.kp,
        "ki": coefficients.ki,
        "status": "skipped",
        "rows": rows,
        "trace_path": str(trace_path),
        "metadata_path": str(metadata_path),
        "validation_status": validation["status"],
        "validation_error": validation["error"],
        "error": "",
    }


def _failed_result(run_id, scenario, seed, coefficients, trace_path,
                   metadata_path, error):
    return {
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


def _execute_run(scenario, seed, coefficients, run_id, run_dir,
                 progress=None):
    """run 하나를 실행하고 (manifest 행, 소요 시간)을 반환한다.

    병렬 실행 시 worker 프로세스에서 호출되므로 모듈 최상위에 둔다.
    """
    trace_path = run_dir / "trace.csv"
    metadata_path = run_dir / "metadata.json"
    started_at = time.monotonic()
    try:
        # trace.csv가 없으면 이전 실행이 중간에 멈춘 것이므로
        # 남아 있는 metadata나 임시 파일은 덮어쓴다.
        result = run_simulation(
            scenario=scenario,
            seed=seed,
            kp=coefficients.kp,
            ki=coefficients.ki,
            trace_path=trace_path,
            metadata_path=metadata_path,
            run_id=run_id,
            overwrite=True,
            progress=progress,
        )
    except Exception as error:
        result = _failed_result(
            run_id, scenario, seed, coefficients,
            trace_path, metadata_path, error,
        )
    return result, time.monotonic() - started_at


def _print_run_problems(result, stream=None):
    stream = stream if stream is not None else sys.stdout
    if result["status"] == "failed":
        stream.write(f"failed: {result['error']}\n")
    if result["status"] == "validation_failed":
        stream.write(f"validation failed: {result['validation_error']}\n")
    stream.flush()


def run_experiment(config: ExperimentConfig, output_root, overwrite=False,
                   workers=1):
    """모든 조합을 실행한다.

    overwrite=False이면 trace.csv가 이미 있는 run은 다시 생성하지 않는다.
    중단된 배치를 같은 명령으로 다시 실행하면 남은 run만 이어서 실행된다.
    workers가 2 이상이면 run을 프로세스 workers개로 나눠 동시에 실행한다.
    run마다 시드로 난수를 초기화하므로 결과는 순차 실행과 같다.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    experiment_dir = Path(output_root) / config.experiment_id
    manifest_path = experiment_dir / "manifest.csv"
    conditions_path = experiment_dir / "conditions.csv"
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
    planned_runs = []
    for scenario in config.scenarios:
        for seed in config.seeds:
            for coefficients in config.pi_coefficients:
                run_id = make_run_id(
                    scenario.scenario_id,
                    seed,
                    coefficients.kp,
                    coefficients.ki,
                )
                run_dir = experiment_dir / "runs" / run_id
                planned_runs.append(
                    (scenario, seed, coefficients, run_id, run_dir)
                )

    # manifest 행은 완료 순서와 관계없이 계획 순서대로 기록한다.
    results = [None] * len(planned_runs)
    pending = []
    for index, (scenario, seed, coefficients, run_id, run_dir) in (
        enumerate(planned_runs)
    ):
        trace_path = run_dir / "trace.csv"
        if not overwrite and trace_path.exists():
            print(f"[{index + 1}/{total_runs}] {run_id} skipped (trace.csv exists)")
            results[index] = _skipped_result(
                run_id, scenario, seed, coefficients,
                trace_path, run_dir / "metadata.json",
            )
        else:
            pending.append(index)

    def write_manifest():
        _write_csv(
            manifest_path,
            MANIFEST_FIELDS,
            [row for row in results if row is not None],
        )

    write_manifest()
    if workers == 1:
        _run_sequential(planned_runs, pending, results, total_runs,
                        write_manifest)
    else:
        _run_parallel(planned_runs, pending, results, workers,
                      write_manifest)

    manifest_rows = results
    completed = sum(row["status"] == "completed" for row in manifest_rows)
    skipped = sum(row["status"] == "skipped" for row in manifest_rows)
    failed = total_runs - completed - skipped
    print(f"experiment_dir={experiment_dir}")
    print(f"completed={completed} skipped={skipped} failed={failed}")
    return manifest_rows


def _run_sequential(planned_runs, pending, results, total_runs,
                    write_manifest):
    pending_minutes = sum(planned_runs[index][0].minutes for index in pending)
    progress = BatchProgress(total_minutes=pending_minutes)
    for index in pending:
        scenario, seed, coefficients, run_id, run_dir = planned_runs[index]
        label = f"[{index + 1}/{total_runs}] {run_id}"
        print(label)
        progress.start_run(label)
        result, _ = _execute_run(
            scenario, seed, coefficients, run_id, run_dir,
            progress=progress.update,
        )
        progress.finish_run(scenario.minutes)
        _print_run_problems(result)
        results[index] = result
        write_manifest()


def _run_parallel(planned_runs, pending, results, workers, write_manifest,
                  stream=None):
    """run 단위로 프로세스 풀에 나눠 실행한다.

    분 단위 진행률 대신 run이 끝날 때마다 한 줄을 출력한다.
    """
    stream = stream if stream is not None else sys.stdout
    pending_minutes = sum(planned_runs[index][0].minutes for index in pending)
    done_minutes = 0
    started_at = time.monotonic()
    stream.write(f"running {len(pending)} runs with {workers} workers\n")
    stream.flush()
    # fork는 macOS와 numpy 조합에서 불안정할 수 있어 spawn을 쓴다.
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(_execute_run, *planned_runs[index]): index
            for index in pending
        }
        for finished, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            scenario, seed, coefficients, run_id, run_dir = planned_runs[index]
            try:
                result, duration = future.result()
            except Exception as error:
                # worker 프로세스가 비정상 종료된 경우
                result = _failed_result(
                    run_id, scenario, seed, coefficients,
                    run_dir / "trace.csv", run_dir / "metadata.json", error,
                )
                duration = 0.0
            results[index] = result
            write_manifest()

            done_minutes += scenario.minutes
            overall = done_minutes / pending_minutes if pending_minutes else 1.0
            elapsed = time.monotonic() - started_at
            eta = elapsed * (1 - overall) / overall if overall > 0 else 0.0
            stream.write(
                f"[{finished}/{len(pending)}] {run_id} {result['status']} "
                f"in {_format_duration(duration)} | overall {overall:5.1%} "
                f"elapsed {_format_duration(elapsed)} "
                f"ETA {_format_duration(eta)}\n"
            )
            stream.flush()
            _print_run_problems(result, stream)
