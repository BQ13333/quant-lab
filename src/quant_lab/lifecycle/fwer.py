"""Finite-horizon empirical family-wise false-retirement calibration.

A replicate fails if ANY still-good sleeve receives a retirement proposal.
Counting proposals conservatively models an IC that approves every proposal.
Intervals quantify Monte Carlo uncertainty, not a lifetime guarantee.
"""
from dataclasses import dataclass, replace
from typing import Sequence
import numpy as np
from scipy.stats import beta
from .decay import finite
from .monitor import Policy, MonitorState, evaluate, monthly_inputs, monthly_schedule
from .states import Value

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


def binomial_rate(events: int, replicates: int) -> Rate:
    if type(events) is not int or type(replicates) is not int or not 0 <= events <= replicates or replicates < 1:
        raise ValueError('invalid binomial counts')
    lo = 0. if events == 0 else float(beta.ppf(.025, events, replicates-events+1))
    hi = 1. if events == replicates else float(beta.ppf(.975, events+1, replicates-events))
    return Rate(events, replicates, events/replicates, (lo, hi))


def false_retire_rate(policy: Policy, seeds: Sequence[int], *, months: int = 60,
                      family_size: int = 5, correlation: float = .2, rho: float = 0.) -> Rate:
    """Gaussian still-good null, cross-sleeve equicorrelation and AR(1) noise.

    Operational inputs are healthy; this measures statistical triggers only.
    psi must be registered in policy for the selected AR process.
    """
    policy.validate(); finite(correlation, rho)
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or s < 0 for s in seeds):
        raise ValueError('unique nonnegative seeds required')
    if type(months) is not int or type(family_size) is not int or min(months, family_size) < 1 or not 0 <= correlation <= 1 or not -1 < rho < 1 or policy.mu_p <= policy.breakeven:
        raise ValueError('invalid still-good null')
    failures = 0
    for seed in seeds:
        rng = np.random.default_rng(seed)
        states = [MonitorState() for _ in range(family_size)]
        noise = np.sqrt(correlation)*rng.normal()+np.sqrt(1-correlation)*rng.normal(size=family_size)
        failed = False
        for _ in range(months):
            innovations = np.sqrt(correlation)*rng.normal()+np.sqrt(1-correlation)*rng.normal(size=family_size)
            noise = rho*noise+np.sqrt(1-rho*rho)*innovations
            for i, x in enumerate(policy.mu_p+policy.sigma*noise):
                result = evaluate(states[i], monthly_inputs(Value(float(x))), monthly_schedule(), policy)
                states[i] = result.state
                failed |= 'RETIRE' in result.proposals
            if failed:
                break
        failures += int(failed)
    return binomial_rate(failures, len(seeds))


def calibrate(policy: Policy, calibration_seeds: Sequence[int], validation_seeds: Sequence[int], *,
              multipliers: Sequence[float] = (1., 1.5, 2., 3.), target: float = .05,
              months: int = 60, family_size: int = 5, correlation: float = .2,
              rho: float = 0.) -> Calibration:
    """Select on calibration only, then evaluate the frozen policy once."""
    finite(target, multipliers)
    if set(calibration_seeds) & set(validation_seeds) or not 0 < target < 1 or not multipliers or any(m < 1 for m in multipliers):
        raise ValueError('need disjoint seed sets and valid search grid')
    kwargs = dict(months=months, family_size=family_size, correlation=correlation, rho=rho)
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
                       tuple(calibration_seeds), tuple(validation_seeds))
