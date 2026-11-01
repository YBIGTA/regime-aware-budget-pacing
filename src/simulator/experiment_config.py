from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re


SAFE_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def _require_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_id(value, name):
    if not isinstance(value, str) or not SAFE_ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{name} must contain only letters, numbers, '.', '_' or '-'"
        )
    return value


@dataclass(frozen=True)
class TrafficCondition:
    dc: float
    amplitude: float
    phase: float = 0.5

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("traffic must be an object")
        dc = _require_number(data.get("dc"), "traffic.dc")
        amplitude = _require_number(
            data.get("amplitude"), "traffic.amplitude"
        )
        phase = _require_number(data.get("phase", 0.5), "traffic.phase")
        if dc <= 0:
            raise ValueError("traffic.dc must be positive")
        if not 0 <= amplitude <= 1:
            raise ValueError("traffic.amplitude must be between 0 and 1")
        if not 0 <= phase <= 1:
            raise ValueError("traffic.phase must be between 0 and 1")
        return cls(dc=dc, amplitude=amplitude, phase=phase)


@dataclass(frozen=True)
class CompetitionCondition:
    untracked_bid_factor: float
    untracked_bid_price_multiplier: float = 1.0

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("competition must be an object")
        factor = _require_number(
            data.get("untracked_bid_factor"),
            "competition.untracked_bid_factor",
        )
        price_multiplier = _require_number(
            data.get("untracked_bid_price_multiplier", 1.0),
            "competition.untracked_bid_price_multiplier",
        )
        if factor < 0:
            raise ValueError(
                "competition.untracked_bid_factor must be non-negative"
            )
        if price_multiplier <= 0:
            raise ValueError(
                "competition.untracked_bid_price_multiplier must be positive"
            )
        return cls(
            untracked_bid_factor=factor,
            untracked_bid_price_multiplier=price_multiplier,
        )


@dataclass(frozen=True)
class EnvironmentCondition:
    traffic: TrafficCondition
    competition: CompetitionCondition

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("initial_condition must be an object")
        return cls(
            traffic=TrafficCondition.from_dict(data.get("traffic")),
            competition=CompetitionCondition.from_dict(
                data.get("competition")
            ),
        )

    def with_updates(self, data):
        if not isinstance(data, dict):
            raise ValueError("condition change must be an object")
        traffic_data = asdict(self.traffic)
        competition_data = asdict(self.competition)
        if "traffic" in data:
            if not isinstance(data["traffic"], dict):
                raise ValueError("change.traffic must be an object")
            traffic_data.update(data["traffic"])
        if "competition" in data:
            if not isinstance(data["competition"], dict):
                raise ValueError("change.competition must be an object")
            competition_data.update(data["competition"])
        return EnvironmentCondition(
            traffic=TrafficCondition.from_dict(traffic_data),
            competition=CompetitionCondition.from_dict(competition_data),
        )


@dataclass(frozen=True)
class CampaignCondition:
    count: int
    daily_budget: float | None = None

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("campaigns must be an object")
        count = data.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("campaigns.count must be a positive integer")
        daily_budget = data.get("daily_budget")
        if daily_budget is not None:
            daily_budget = _require_number(
                daily_budget, "campaigns.daily_budget"
            )
            if daily_budget <= 0:
                raise ValueError("campaigns.daily_budget must be positive")
        return cls(count=count, daily_budget=daily_budget)


@dataclass(frozen=True)
class ScheduledCondition:
    minute: int
    condition: EnvironmentCondition


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    minutes: int
    campaigns: CampaignCondition
    schedule: tuple[ScheduledCondition, ...]

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("each scenario must be an object")
        scenario_id = _validate_id(data.get("id"), "scenario.id")
        minutes = data.get("minutes")
        if (
            isinstance(minutes, bool)
            or not isinstance(minutes, int)
            or not 1 <= minutes <= 1440
        ):
            raise ValueError("scenario.minutes must be between 1 and 1440")

        campaigns = CampaignCondition.from_dict(data.get("campaigns"))
        initial = EnvironmentCondition.from_dict(
            data.get("initial_condition")
        )
        schedule = [ScheduledCondition(minute=0, condition=initial)]
        current = initial
        seen_minutes = {0}
        previous_minute = 0
        for change in data.get("changes", []):
            if not isinstance(change, dict):
                raise ValueError("each condition change must be an object")
            minute = change.get("minute")
            if (
                isinstance(minute, bool)
                or not isinstance(minute, int)
                or not 1 <= minute < minutes
            ):
                raise ValueError(
                    "change.minute must be after 0 and before scenario.minutes"
                )
            if minute in seen_minutes or minute < previous_minute:
                raise ValueError(
                    "condition changes must have unique, increasing minutes"
                )
            updates = {
                key: value
                for key, value in change.items()
                if key != "minute"
            }
            if not updates:
                raise ValueError("condition change must update a condition")
            current = current.with_updates(updates)
            schedule.append(
                ScheduledCondition(minute=minute, condition=current)
            )
            seen_minutes.add(minute)
            previous_minute = minute

        return cls(
            scenario_id=scenario_id,
            minutes=minutes,
            campaigns=campaigns,
            schedule=tuple(schedule),
        )

    def condition_rows(self):
        rows = []
        for index, scheduled in enumerate(self.schedule):
            end_minute = (
                self.schedule[index + 1].minute - 1
                if index + 1 < len(self.schedule)
                else self.minutes - 1
            )
            condition = scheduled.condition
            rows.append(
                {
                    "scenario_id": self.scenario_id,
                    "start_minute": scheduled.minute,
                    "end_minute": end_minute,
                    "campaign_count": self.campaigns.count,
                    "campaign_daily_budget": self.campaigns.daily_budget,
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
            )
        return rows


@dataclass(frozen=True)
class PICoefficients:
    kp: float
    ki: float

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("each PI coefficient must be an object")
        kp = _require_number(data.get("kp"), "pi_coefficients.kp")
        ki = _require_number(data.get("ki"), "pi_coefficients.ki")
        if kp < 0 or ki < 0:
            raise ValueError("PI coefficients must be non-negative")
        return cls(kp=kp, ki=ki)


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    seeds: tuple[int, ...]
    pi_coefficients: tuple[PICoefficients, ...]
    scenarios: tuple[Scenario, ...]

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict):
            raise ValueError("experiment config must be an object")
        experiment_id = _validate_id(
            data.get("experiment_id"), "experiment_id"
        )

        seeds = data.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise ValueError("seeds must be a non-empty list")
        if any(
            isinstance(seed, bool)
            or not isinstance(seed, int)
            or not 0 <= seed < 2**32
            for seed in seeds
        ):
            raise ValueError(
                "every seed must be an integer between 0 and 2**32 - 1"
            )
        if len(set(seeds)) != len(seeds):
            raise ValueError("seeds must not contain duplicates")

        coefficients_data = data.get("pi_coefficients")
        if not isinstance(coefficients_data, list) or not coefficients_data:
            raise ValueError("pi_coefficients must be a non-empty list")
        coefficients = tuple(
            PICoefficients.from_dict(item) for item in coefficients_data
        )
        coefficient_pairs = {(item.kp, item.ki) for item in coefficients}
        if len(coefficient_pairs) != len(coefficients):
            raise ValueError("pi_coefficients must not contain duplicates")

        scenarios_data = data.get("scenarios")
        if not isinstance(scenarios_data, list) or not scenarios_data:
            raise ValueError("scenarios must be a non-empty list")
        scenarios = tuple(Scenario.from_dict(item) for item in scenarios_data)
        scenario_ids = [scenario.scenario_id for scenario in scenarios]
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("scenario ids must be unique")

        return cls(
            experiment_id=experiment_id,
            seeds=tuple(seeds),
            pi_coefficients=coefficients,
            scenarios=scenarios,
        )


def load_experiment_config(path):
    path = Path(path)
    with path.open(encoding="utf-8") as config_file:
        data = json.load(config_file)
    return ExperimentConfig.from_dict(data)
