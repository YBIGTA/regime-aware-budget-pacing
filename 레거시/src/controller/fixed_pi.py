from dataclasses import dataclass, field

from src import constants
from src.system.budget_pacing.pacing_system_interface import PacingSystemInterface
from src.system.clock import Clock


@dataclass
class PIState:
    daily_budget: float
    pacing_signal: float = 1.0
    actual_spend: float = 0.0
    integral_error: float = 0.0
    history: list[dict] = field(default_factory=list)


class FixedPI(PacingSystemInterface):
    def __init__(
        self,
        kp: float,
        ki: float,
        min_signal: float = 0.0,
        max_signal: float = 1.0,
    ):
        if kp < 0 or ki < 0:
            raise ValueError("kp and ki must be non-negative")
        if min_signal < 0 or max_signal <= min_signal:
            raise ValueError("invalid pacing signal bounds")

        self.kp = kp
        self.ki = ki
        self.min_signal = min_signal
        self.max_signal = max_signal
        self.states: dict[str, PIState] = {}

    def add_campaign(self, campaign):
        self.states[campaign.id] = PIState(daily_budget=campaign.daily_budget)

    def get_pacing_signal(self, campaign_id):
        return self.states[campaign_id].pacing_signal

    def end_iteration(self, campaign_id, spend_since_last_iteration):
        state = self.states[campaign_id]
        state.actual_spend += spend_since_last_iteration

        minute = Clock.minute_in_day()
        target_fraction = (minute + 1) / constants.num_minutes_in_day
        actual_fraction = state.actual_spend / state.daily_budget
        error = target_fraction - actual_fraction

        candidate_integral = state.integral_error + error
        raw_signal = 1.0 + self.kp * error + self.ki * candidate_integral
        next_signal = min(self.max_signal, max(self.min_signal, raw_signal))

        if self.min_signal < raw_signal < self.max_signal:
            state.integral_error = candidate_integral

        state.pacing_signal = next_signal
        state.history.append(
            {
                "minute": minute,
                "target_fraction": target_fraction,
                "actual_fraction": actual_fraction,
                "error": error,
                "integral_error": state.integral_error,
                "next_signal": next_signal,
            }
        )

    def get_state(self, campaign_id):
        return self.states[campaign_id]

    def get_pacing_statistics(self, campaign_id):
        return {"pi_history": self.states[campaign_id].history}

    def get_global_pacing_statistics(self):
        return {}