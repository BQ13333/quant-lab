"""Renewal planning, with explicit correlation and discovery assumptions."""
import math
from typing import Mapping, Sequence
from scipy.stats import norm
from .decay import finite
from quant_lab.validation.overfit import deflated_sharpe_probability


def live_strategies(birth_rate: float, mean_lifetime: float) -> float:
    finite(birth_rate, mean_lifetime)
    if birth_rate < 0 or mean_lifetime <= 0:
        raise ValueError('invalid renewal parameters')
    return birth_rate*mean_lifetime


def survival(age: float, half_life: float) -> float:
    finite(age, half_life)
    if age < 0 or half_life <= 0:
        raise ValueError('invalid survival parameters')
    return math.exp(-math.log(2)*age/half_life)


def family_survival(age: float, half_lives: Mapping[str, float]) -> dict[str, float]:
    return {family: survival(age, half) for family, half in half_lives.items()}


def portfolio_ir(n: float, r: float, rho: float = 0.) -> float:
    finite(n, r, rho)
    if n < 0 or not -1 <= rho <= 1 or (n > 0 and 1+(n-1)*rho <= 0):
        raise ValueError('invalid equicorrelation/strategy count')
    return r*math.sqrt(n/(1+(n-1)*rho)) if n else 0.


def required_birth_rate(target_ir: float, r: float, rho: float, mean_lifetime: float) -> float:
    finite(target_ir, r, rho, mean_lifetime)
    if target_ir < 0 or r <= 0 or not 0 <= rho < 1 or mean_lifetime <= 0:
        raise ValueError('invalid target assumptions')
    a = (target_ir/r)**2
    if a*rho >= 1:
        raise ValueError('target exceeds correlation ceiling')
    return a*(1-rho)/(1-a*rho)/mean_lifetime


def expected_maximum(n_trials: int) -> float:
    """Gaussian expected maximum approximation used in deflated Sharpe."""
    if type(n_trials) is not int or n_trials < 1:
        raise ValueError('positive trial count required')
    if n_trials == 1:
        return 0.
    gamma = .5772156649015329
    return float((1-gamma)*norm.isf(1/n_trials)+gamma*norm.isf(1/(n_trials*math.e)))


def ideas_needed(births: float, pass_rate: float, n_trials: int = 1,
                 sharpe_se: float = 1.) -> float:
    """Planning approximation: shift the Gaussian pass threshold by E[max]*SE.

    pass_rate is an unadjusted Gaussian-tail probability. This is a funnel
    assumption, not an approval test; use discovery_probability on real returns.
    """
    finite(births, pass_rate, sharpe_se)
    if births < 0 or not 0 < pass_rate <= 1 or sharpe_se < 0:
        raise ValueError('invalid funnel inputs')
    haircut = expected_maximum(n_trials)*sharpe_se
    adjusted = float(norm.sf(norm.isf(pass_rate)+haircut))
    if adjusted == 0:
        raise ValueError('discovery probability underflows')
    return births/adjusted


def discovery_probability(returns: Sequence[float], trial_sharpes: Sequence[float]) -> float:
    """Reuse the house DSR implementation; ledger supplies the complete family."""
    finite(returns, trial_sharpes)
    return deflated_sharpe_probability(returns, trial_sharpes)


def worked_example() -> dict[str, float]:
    lifetime = 3/math.log(2)
    n = live_strategies(2., lifetime)
    return dict(mean_lifetime=lifetime, live=n, portfolio_ir=portfolio_ir(n, .5, .1),
                ideas=ideas_needed(2., .1, 100, .1))
