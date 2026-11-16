"""Deterministic decay diagnostics; gaps must be explicit, never imputed."""
from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy.special import logsumexp, ndtr
from scipy.stats import linregress, norm


def finite(*values: object) -> None:
    try:
        valid = all(np.isfinite(v).all() for v in values)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError('inputs must be finite numbers')


@dataclass(frozen=True)
class Posterior:
    mean: float
    sd: float
    n_eff: float
    p_positive: float
    p_net: float


def normal_posterior(xs: Sequence[float], mu_p: float, tau: float, sigma: float,
                     breakeven: float = 0., rho: float = 0., psi: float | None = None) -> Posterior:
    """Normal-Normal update with frozen AR(1) variance inflation, capped at n."""
    finite(xs, mu_p, tau, sigma, breakeven, rho)
    if tau <= 0 or sigma <= 0 or not -1 < rho < 1:
        raise ValueError('invalid scale or AR coefficient')
    psi = min(1., (1-rho)/(1+rho)) if psi is None else psi
    finite(psi)
    if not 0 < psi <= 1:
        raise ValueError('psi must be in (0,1]')
    n = len(xs) * psi
    precision = 1/tau**2 + n/sigma**2
    mean = (mu_p/tau**2 + (n * float(np.mean(xs))/sigma**2 if len(xs) else 0))/precision
    sd = precision**-.5
    finite(mean, sd)
    return Posterior(mean, sd, n, float(ndtr(mean/sd)), float(ndtr((mean-breakeven)/sd)))


def cusum(previous: float, observation: float | None, mu: float, sigma: float,
          k: float = .15, h: float = 5.) -> tuple[float, bool]:
    """None means a scheduled gap; NaN is always an error."""
    finite(previous, mu, sigma, k, h)
    if previous < 0 or sigma <= 0 or k < 0 or h <= 0:
        raise ValueError('invalid CUSUM parameters')
    if observation is not None:
        finite(observation)
        previous = max(0., previous - (observation-mu)/sigma-k)
    finite(previous)
    return previous, previous > h


@dataclass(frozen=True)
class ChangePoint:
    run_length: tuple[float, ...]
    probability_recent: float


def bocpd(xs: Sequence[float], mu: float = 0., tau: float = 1., sigma: float = 1.,
          hazard: float = 1/60, window: int = 12) -> ChangePoint:
    """Adams–MacKay recursion, known variance and unknown Normal mean.

    Run length zero denotes a boundary AFTER the current datum. Its probability
    alone equals the constant hazard; recent probability sums r < window.
    """
    finite(xs, mu, tau, sigma, hazard, window)
    if min(tau, sigma) <= 0 or not 0 < hazard < 1 or not isinstance(window, int) or window < 1:
        raise ValueError('invalid BOCPD parameters')
    logp = np.array([0.]); means = np.array([mu]); variances = np.array([tau*tau])
    for x in xs:
        joint = logp + norm.logpdf(x, means, np.sqrt(variances+sigma*sigma))
        new = np.r_[logsumexp(joint)+np.log(hazard), joint+np.log1p(-hazard)]
        logp = new-logsumexp(new)
        updated = 1/(1/variances+1/sigma**2)
        means = np.r_[mu, updated*(means/variances+x/sigma**2)]
        variances = np.r_[tau*tau, updated]
    p = np.exp(logp)
    return ChangePoint(tuple(p), float(p[:window].sum()))


@dataclass(frozen=True)
class DecayFit:
    slope: float
    standard_error: float
    half_life: float | None
    amplitude: float


def exponential_half_life(horizons: Sequence[float], ic: Sequence[float]) -> DecayFit:
    """Log-linear diagnostic on positive marginal IC/rates, not cumulative IC."""
    finite(horizons, ic)
    x, y = np.asarray(horizons), np.asarray(ic)
    if len(x) != len(y) or len(x) < 3 or np.any(y <= 0) or np.any(np.diff(x) <= 0):
        raise ValueError('need ordered horizons and positive rates')
    fit = linregress(x, np.log(y))
    return DecayFit(float(fit.slope), float(fit.stderr),
                    float(-np.log(2)/fit.slope) if fit.slope < 0 else None, float(np.exp(fit.intercept)))


def rolling_decay_slope(xs: Sequence[float], window: int = 12) -> tuple[float, float]:
    """OLS slope and iid standard error on the last window of calendar IC."""
    finite(xs, window)
    if not isinstance(window, int) or window < 3 or len(xs) < window:
        raise ValueError('insufficient window')
    fit = linregress(np.arange(window), np.asarray(xs)[-window:])
    return float(fit.slope), float(fit.stderr)
