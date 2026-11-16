"""Synthetic lab demonstration, not evidence of achievable market returns."""
from dataclasses import dataclass, replace
import numpy as np
import pandas as pd
from .allocator import DiscountedThompson, Sleeve
from .decay import finite
from .monitor import MonitorState, Policy, evaluate, monthly_inputs, monthly_schedule
from .states import Lifecycle, Value, Permission, ScheduleCalendar

@dataclass(frozen=True)
class Simulation:
    path: pd.DataFrame
    strategies: pd.DataFrame


def simulate(*, months: int = 120, birth_rate: float = 1., stop_births: int | None = None,
             prior_mean: float = .10, prior_sd: float = .02, half_life: float = 12.,
             noise_sd: float = .05, seed: int = 0) -> Simulation:
    """Poisson arrivals; lognormal half-lives; independent monthly IC noise.

    Allocate before observing each month's rewards. Retirement proposals are
    recorded as proposals only, without approval. All candidate streams are
    pre-generated so paired birth-stop experiments share counterfactual noise.
    Path IR is conditional expected annualised ICIR; realised IC is also saved.
    """
    finite(birth_rate, prior_mean, prior_sd, half_life, noise_sd)
    if type(months) is not int or months < 1 or birth_rate < 0 or min(prior_mean, half_life, noise_sd) <= 0 or prior_sd < 0 or (stop_births is not None and (type(stop_births) is not int or stop_births < 0)):
        raise ValueError('invalid simulation parameters')
    rng = np.random.default_rng(seed)
    arrivals = np.repeat(np.arange(months), rng.poisson(birth_rate, months))
    births = np.r_[0, arrivals]
    n = len(births)
    initial = np.maximum(.001, rng.normal(prior_mean, prior_sd, n))
    halves = half_life*rng.lognormal(0., .3, n)
    noise = rng.normal(0., noise_sd, (months, n))
    policy = Policy(mu_p=prior_mean, tau=max(prior_sd, .02), sigma=noise_sd, breakeven=.01)
    states = [MonitorState(lifecycle=Lifecycle.RAMP) for _ in range(n)]
    bandit = DiscountedThompson(n, sigma=noise_sd, prior_mean=prior_mean, prior_sd=max(prior_sd, .02), seed=seed)
    weights = np.zeros(n)
    rows = []
    for t in range(months):
        active = (births <= t) & ((births < stop_births) if stop_births is not None else True)
        sleeves = [Sleeve.from_monitor(states[i], permission=Permission.OK if active[i] else Permission.BLOCK) for i in range(n)]
        # Full turnover budget allows mandatory retirements; allocator itself
        # supports tighter budgets and reports infeasible combinations.
        weights = bandit.allocate(weights, sleeves, max_turnover=2.)
        means = initial*np.exp(-np.log(2)*np.maximum(0, t-births)/halves)
        rewards = means+noise[t]
        expected = float(weights@means)
        sd = noise_sd*float(np.linalg.norm(weights))
        rows.append(dict(month=t, portfolio_ic=float(weights@rewards), expected_ic=expected,
                         portfolio_ir=expected/sd*np.sqrt(12) if sd else 0., invested=float(weights.sum()),
                         live=int(sum(active[i] and states[i].lifecycle != Lifecycle.RETIRED for i in range(n)))))
        bandit.update([float(rewards[i]) if active[i] else None for i in range(n)])
        for i in np.flatnonzero(active):
            result = evaluate(states[i], monthly_inputs(Value(float(rewards[i]))), ScheduleCalendar(), policy,
                              approve_retirement=False)
            states[i] = result.state
    return Simulation(pd.DataFrame(rows), pd.DataFrame(dict(birth=births, initial_ic=initial, half_life=halves)))
