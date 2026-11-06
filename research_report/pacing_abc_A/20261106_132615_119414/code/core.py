"""Shared causal first-price benchmark. NumPy only; no regime or RL code."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Protocol
import hashlib
import json
import math
import numpy as np

VERSION = 'abc-first-price-v1'

@dataclass(frozen=True)
class Config:
    master_seed: int = 20261105
    minutes: int = 1440
    interval: int = 5
    history: int = 12
    multiplier_min: float = 0.0
    multiplier_max: float = 1.0
    initial_multiplier: float = 0.5
    bootstrap_seed: int = 917
    train_start: int = 1000
    train_days: int = 40
    validation_start: int = 100000
    validation_days: int = 20
    test_start: int = 200000
    test_days: int = 60
    trials: int = 40
    arrival_min: float = 4.0  # opportunities/minute
    arrival_max: float = 12.0
    budget_fraction_min: float = 0.08
    budget_fraction_max: float = 0.25

    def __post_init__(self):
        if self.minutes <= 0 or self.interval <= 0 or self.minutes % self.interval:
            raise ValueError('Horizon must be divisible by interval')
        if self.history < 1 or self.trials < 2:
            raise ValueError('history>=1 and trials>=2 required')
        if not (0 <= self.multiplier_min < self.multiplier_max <= 1):
            raise ValueError('Require 0 <= lower < upper <= 1')
        if not self.multiplier_min <= self.initial_multiplier <= self.multiplier_max:
            raise ValueError('Initial multiplier out of bounds')
        if not (0 < self.arrival_min <= self.arrival_max):
            raise ValueError('Invalid arrival range')
        if not (0 < self.budget_fraction_min <= self.budget_fraction_max < 1):
            raise ValueError('Invalid budget fraction')
        groups = [set(self.ids(s)) for s in ('train', 'validation', 'test')]
        if any(not g or min(g) < 0 for g in groups):
            raise ValueError('Nonempty nonnegative episode IDs required')
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError('Episode splits overlap')

    @property
    def ticks(self):
        return self.minutes // self.interval

    def ids(self, split):
        if split not in ('train', 'validation', 'test'):
            raise ValueError(split)
        return range(getattr(self, split+'_start'),
                     getattr(self, split+'_start')+getattr(self, split+'_days'))

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

@dataclass(frozen=True)
class Tick:
    base: np.ndarray
    competitor: np.ndarray

@dataclass(frozen=True)
class Episode:
    day: int
    budget: float
    tape: tuple[Tick, ...]

    def digest(self):
        h = hashlib.sha256(np.asarray([self.day, self.budget], dtype='<f8').tobytes())
        for tick in self.tape:
            h.update(np.asarray([len(tick.base)], dtype='<i8').tobytes())
            h.update(tick.base.astype('<f8').tobytes())
            h.update(tick.competitor.astype('<f8').tobytes())
        return h.hexdigest()


def generate(cfg: Config, day: int) -> Episode:
    """Smooth + correlated noise + optional finite shock; ranges are uncalibrated."""
    rng = np.random.default_rng(np.random.SeedSequence([cfg.master_seed, day]))
    n = cfg.ticks
    arrival = rng.uniform(cfg.arrival_min, cfg.arrival_max)
    base_scale = rng.uniform(.5, 2.)
    comp_ratio = rng.uniform(.4, 1.1)
    phase = rng.uniform(0, 2*np.pi)
    amplitude = rng.uniform(0, .65)
    comp_amplitude = rng.uniform(0, .35)
    comp_phase = rng.uniform(0, 2*np.pi)
    budget_fraction = rng.uniform(cfg.budget_fraction_min, cfg.budget_fraction_max)
    # Budget uses sampled ex-ante parameters, never realized future tape totals.
    budget = cfg.minutes*arrival*base_scale*math.exp(.35**2/2)*budget_fraction
    begin = int(rng.uniform(.2, .7)*n)
    end = min(n, begin+max(1, int(rng.uniform(.1, .4)*n)))
    supply_shock = rng.uniform(.35, 1.8) if rng.random() < .7 else 1.
    comp_shock = rng.uniform(.65, 1.6) if rng.random() < .7 else 1.
    supply_noise = comp_noise = 0.
    tape = []
    for t in range(n):
        supply_noise = .9*supply_noise+rng.normal(0, .04)
        comp_noise = .9*comp_noise+rng.normal(0, .03)
        shock = begin <= t < end
        lam = arrival*cfg.interval*(1+amplitude*np.cos(2*np.pi*t/n+phase))
        lam *= math.exp(supply_noise)*(supply_shock if shock else 1.)
        count = int(rng.poisson(lam))
        base = base_scale*rng.lognormal(0, .35, count)
        level = comp_ratio*(1+comp_amplitude*np.cos(2*np.pi*t/n+comp_phase))
        level *= math.exp(comp_noise)*(comp_shock if shock else 1.)
        competitor = base_scale*level*rng.lognormal(0, .45, count)
        base.setflags(write=False)
        competitor.setflags(write=False)
        tape.append(Tick(base, competitor))
    return Episode(day, budget, tuple(tape))

@dataclass(frozen=True)
class Feedback:
    tick: int
    action: float
    spend: float
    opportunities: int
    wins: int
    base_mean: float
    budget_censored: bool

@dataclass(frozen=True)
class Observation:
    # Created before interval t. History contains only completed intervals.
    tick: int
    ticks: int
    budget: float
    cumulative_spend: float
    history: tuple[Feedback, ...]

    @property
    def error(self):
        return self.tick/self.ticks-self.cumulative_spend/self.budget

    @property
    def target_increment(self):
        return self.budget/self.ticks


def features(obs: Observation, candidate: float) -> np.ndarray:
    """Common B/C input schema; no current/future counts or competing prices.

    Fitting scalers must use training rows only. No advertiser/day ID input.
    """
    x = [candidate, obs.tick/obs.ticks, math.log(obs.budget),
         obs.cumulative_spend/obs.budget, obs.error]
    # Oldest -> newest, fixed length 12 set by caller through feature schema.
    for f in obs.history:
        x.extend([float(f.tick >= 0), f.action, f.spend/obs.budget, math.log1p(f.opportunities),
                  f.wins/max(1, f.opportunities), f.base_mean,
                  float(f.budget_censored)])
    return np.asarray(x, dtype=float)

class BaseProvider(Protocol):
    def __call__(self, obs: Observation) -> float: ...

@dataclass(frozen=True)
class Gains:
    kp: float
    ki: float

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.kp, self.ki)):
            raise ValueError('Finite nonnegative gains required')

class PI:
    """A/B/C all use THIS implementation, differing only in base provider."""
    def __init__(self, cfg: Config, gains: Gains, provider: BaseProvider | None = None):
        self.cfg, self.gains, self.provider = cfg, gains, provider
        self.integral = 0.

    def reset(self):
        self.integral = 0.

    def act(self, obs: Observation):
        base = self.cfg.initial_multiplier if self.provider is None else float(self.provider(obs))
        lo, hi = self.cfg.multiplier_min, self.cfg.multiplier_max
        if not math.isfinite(base) or not lo <= base <= hi:
            raise ValueError('Invalid base-provider output')
        error = obs.error
        candidate = self.integral+error/obs.ticks
        raw = base+self.gains.kp*error+self.gains.ki*candidate
        blocked = (raw > hi and error > 0) or (raw < lo and error < 0)
        if blocked:
            raw = base+self.gains.kp*error+self.gains.ki*self.integral
        else:
            self.integral = candidate
        action = float(np.clip(raw, lo, hi))
        return action, dict(base=base, integral=self.integral, raw=raw,
                            integral_blocked=blocked, saturated=raw < lo or raw > hi)

class PredictiveBase:
    """B/C adapter only; predictive models themselves are deliberately not fitted here.

    predictor(X) returns normalized next-interval spend for each candidate.
    Both B/C MUST use identical features and grid. No online updates in v1.
    """
    def __init__(self, cfg, predictor):
        self.cfg, self.predictor = cfg, predictor
        self.grid = np.linspace(cfg.multiplier_min, cfg.multiplier_max, 101)

    def __call__(self, obs):
        if obs.tick == 0:
            return self.cfg.initial_multiplier  # identical cold start
        x = np.stack([features(obs, float(m)) for m in self.grid])
        y = np.asarray(self.predictor(x), dtype=float)
        if y.shape != self.grid.shape or not np.isfinite(y).all():
            raise ValueError('Predictor must return 101 finite normalized spends')
        # Deterministic ties favor lower bid; no extra hidden error compensation.
        cost = np.abs(np.maximum(y, 0)-1/obs.ticks)
        return float(self.grid[int(np.argmin(cost))])


def execute(tick: Tick, action: float, remaining: float):
    """Chronological first-price auction. Can't pay => skip, no bid clipping.

    Equal bids lose. No shuffling, rationing RNG, or policy-dependent randomness.
    """
    bids = tick.base*action
    eligible = (bids > tick.competitor) & (bids > 0)
    total = float(bids[eligible].sum())
    if total <= remaining:
        return total, int(eligible.sum()), False
    spent = 0.
    wins = 0
    for payment in bids[eligible]:
        if payment <= remaining-spent:
            spent += float(payment)
            wins += 1
    return spent, wins, True


def capacity_diagnostic(ep, cfg):
    # Necessary capacity check, NOT proof that a causal controller can track.
    amounts = [float((tick.base*cfg.multiplier_max)[
        tick.base*cfg.multiplier_max > tick.competitor].sum()) for tick in ep.tape]
    shortfall = np.maximum(np.arange(1, cfg.ticks+1)/cfg.ticks-
                           np.cumsum(amounts)/ep.budget, 0)
    return float(shortfall.max())


def rollout(ep: Episode, cfg: Config, controller: PI, sink=None):
    if len(ep.tape) != cfg.ticks or ep.budget <= 0:
        raise ValueError('Invalid episode')
    controller.reset()
    spent = 0.
    history = []
    rows = []
    zero = Feedback(-1, 0., 0., 0, 0, 0., False)
    for t, tick in enumerate(ep.tape):
        padded = ([zero]*cfg.history+history)[-cfg.history:]
        obs = Observation(t, cfg.ticks, ep.budget, spent, tuple(padded))
        action, debug = controller.act(obs)
        if not math.isfinite(action) or not cfg.multiplier_min <= action <= cfg.multiplier_max:
            raise ValueError('Invalid action')
        cost, wins, censored = execute(tick, action, max(0., ep.budget-spent))
        if sink is not None:
            sink(obs, action, cost, censored)
        spent += cost
        if spent > ep.budget+1e-9*max(1, ep.budget):
            raise AssertionError('Overspend')
        f = Feedback(t, action, cost, len(tick.base), wins,
                     float(tick.base.mean()) if len(tick.base) else 0., censored)
        history.append(f)
        end_error = (t+1)/cfg.ticks-spent/ep.budget
        rows.append(dict(day=ep.day, budget=ep.budget, target=(t+1)*ep.budget/cfg.ticks,
                         cumulative_spend=spent, remaining=max(0., ep.budget-spent),
                         error_before=obs.error, error=end_error, **asdict(f), **debug))
    errors = np.asarray([r['error'] for r in rows])
    actions = np.asarray([r['action'] for r in rows])
    te = float(np.abs(errors).mean())
    unspent = max(0., 1-spent/ep.budget)
    summary = dict(day=ep.day, tape_sha256=ep.digest(), tracking_error=te,
                   unspent=unspent, objective=te+unspent,
                   multiplier_tv=float(np.abs(np.diff(actions)).mean()) if len(actions)>1 else 0.,
                   max_ahead=float(np.maximum(-errors, 0).max()),
                   saturation_fraction=float(np.mean([r['saturated'] for r in rows])),
                   capacity_shortfall=capacity_diagnostic(ep, cfg),
                   budget_censored_ticks=sum(r['budget_censored'] for r in rows))
    return summary, rows
