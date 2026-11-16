"""Finite-horizon empirical family-wise false-retirement calibration.

A replicate fails if ANY still-good sleeve receives a retirement proposal.
Counting proposals conservatively models an IC that approves every proposal.
Intervals quantify Monte Carlo uncertainty, not a lifetime guarantee.
"""
from dataclasses import dataclass, replace
from typing import Sequence, Protocol
import numpy as np
from scipy.stats import beta
from .decay import finite
from .monitor import Policy, MonitorState, evaluate, monthly_inputs, monthly_schedule
from .states import Value, Lifecycle, ScheduleCalendar

@dataclass(frozen=True)
class Rate:
    events: int
    replicates: int
    rate: float
    interval_95: tuple[float, float]

@dataclass(frozen=True)
class Calibration:
    policy: Policy
    calibration: Rate
    validation: Rate
    target_met_on_calibration: bool
    months: int
    family_size: int
    calibration_seeds: tuple[int, ...]
    validation_seeds: tuple[int, ...]
    target: float = .05
    min_validation: int = 2000
    false_pause: Rate | None = None

    @property
    def passed(self):
        return self.target_met_on_calibration and self.validation.replicates >= self.min_validation and self.validation.interval_95[1] <= self.target

    @property
    def status(self):
        return 'CALIBRATED' if self.passed else 'NOT_CALIBRATED'

    def report(self):
        r = self.validation
        return (f'Under the registered null set, over a {self.months}-month horizon and a family of '
                f'{self.family_size} sleeves, the simulated family false-retirement rate on an independent '
                f'validation set was {r.events}/{r.replicates} with 95% interval {r.interval_95}. '
                f'This is an empirical calibration, not a bound. {self.status}. '
                f'False-pause rate: {self.false_pause}.')


class NullModel(Protocol):
    def sample(self, rng, months, family_size, policy): ...


@dataclass(frozen=True)
class GaussianAR1:
    correlation: float = .2
    rho: float = 0.

    def sample(self, rng, months, family_size, policy):
        finite(self.correlation, self.rho)
        if not 0 <= self.correlation <= 1 or not -1 < self.rho < 1:
            raise ValueError('invalid Gaussian null')
        def innovation():
            return np.sqrt(self.correlation)*rng.normal()+np.sqrt(1-self.correlation)*rng.normal(size=family_size)
        noise = innovation()
        rows = []
        for _ in range(months):
            noise = self.rho*noise+np.sqrt(1-self.rho**2)*innovation()
            rows.append(policy.mu_p+policy.sigma*noise)
        return np.asarray(rows)


@dataclass(frozen=True)
class BlockBootstrap:
    history: tuple
    block: int = 6

    def sample(self, rng, months, family_size, policy):
        history = np.asarray(self.history, dtype=float)
        if history.ndim == 1:
            history = history[:,None]
        if history.ndim != 2 or len(history)<2 or history.shape[1] not in (1,family_size) or not np.isfinite(history).all() or type(self.block) is not int or not 1 <= self.block <= len(history):
            raise ValueError('invalid block history')
        # Center the empirical innovations on the registered still-good mean.
        history = history-history.mean(axis=0)+policy.mu_p
        indices = np.concatenate([(rng.integers(len(history))+np.arange(self.block)) % len(history)
                                  for _ in range(int(np.ceil(months/self.block)))])[:months]
        return np.broadcast_to(history[indices], (months,family_size)).copy()



def binomial_rate(events: int, replicates: int) -> Rate:
    if type(events) is not int or type(replicates) is not int or not 0 <= events <= replicates or replicates < 1:
        raise ValueError('invalid binomial counts')
    lo = 0. if events == 0 else float(beta.ppf(.025, events, replicates-events+1))
    hi = 1. if events == replicates else float(beta.ppf(.975, events+1, replicates-events))
    return Rate(events, replicates, events/replicates, (lo, hi))


def false_event_rate(policy, seeds, *, event='RETIRE', months=60, family_size=5,
                     correlation=.2, rho=0., null=None):
    policy.validate()
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('unique nonnegative seeds required')
    if event not in ('RETIRE','PAUSE') or type(months) is not int or type(family_size) is not int or min(months,family_size)<1 or policy.mu_p <= policy.breakeven:
        raise ValueError('invalid still-good null')
    model = null if null is not None else GaussianAR1(correlation,rho)
    failures = 0
    calendar = ScheduleCalendar()
    for seed in seeds:
        samples = np.asarray(model.sample(np.random.default_rng(seed),months,family_size,policy))
        if samples.shape != (months,family_size) or not np.isfinite(samples).all():
            raise ValueError('invalid null output')
        states = [MonitorState() for _ in range(family_size)]
        failed = False
        for row in samples:
            for i,x in enumerate(row):
                result = evaluate(states[i], monthly_inputs(Value(float(x))), calendar, policy)
                states[i] = result.state
                failed |= ('RETIRE' in result.proposals if event == 'RETIRE' else result.state.lifecycle == Lifecycle.PAUSE)
            if failed:
                break
        failures += int(failed)
    return binomial_rate(failures,len(seeds))


def false_retire_rate(policy, seeds, **kwargs):
    return false_event_rate(policy,seeds,event='RETIRE',**kwargs)


def calibrate(policy: Policy, calibration_seeds: Sequence[int], validation_seeds: Sequence[int], *,
              multipliers: Sequence[float] = (1., 1.5, 2., 3.), target: float = .05,
              months: int = 60, family_size: int = 5, correlation: float = .2,
              rho: float = 0., min_validation: int = 2000, null: NullModel | None = None) -> Calibration:
    """Select on calibration only, then evaluate the frozen policy once."""
    if type(min_validation) is not int or min_validation < 1:
        raise ValueError("invalid minimum validation count")
    finite(target, multipliers)
    if set(calibration_seeds) & set(validation_seeds) or not 0 < target < 1 or not multipliers or any(m < 1 for m in multipliers):
        raise ValueError('need disjoint seed sets and valid search grid')
    kwargs = dict(months=months, family_size=family_size, correlation=correlation, rho=rho, null=null)
    for factor in sorted(multipliers):
        selected = replace(policy, reduce_h=policy.reduce_h*factor, pause_h=policy.pause_h*factor,
                           hard_pause_net=policy.hard_pause_net/factor,
                           retire_positive=policy.retire_positive/factor,
                           retire_months=int(np.ceil(policy.retire_months*factor)))
        rate = false_retire_rate(selected, calibration_seeds, **kwargs)
        if rate.rate <= target:
            break
    validation = false_retire_rate(selected, validation_seeds, **kwargs)
    return Calibration(selected, rate, validation, rate.rate <= target, months, family_size,
                       tuple(calibration_seeds), tuple(validation_seeds), target, min_validation,
                       false_event_rate(selected, validation_seeds, event="PAUSE", **kwargs))
