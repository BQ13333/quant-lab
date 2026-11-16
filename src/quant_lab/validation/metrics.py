from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd


def _clean_returns(returns: pd.Series | Sequence[float]) -> pd.Series:
    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    return series


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _normal_ppf(probability: float) -> float:
    """Acklam inverse normal approximation."""

    if not 0.0 < probability < 1.0:
        raise ValueError("probability must be between 0 and 1")

    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    low = 0.02425
    high = 1.0 - low

    if probability < low:
        q = math.sqrt(-2.0 * math.log(probability))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if probability <= high:
        q = probability - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    q = math.sqrt(-2.0 * math.log(1.0 - probability))
    return -(
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
        / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    )


def sharpe_ratio(returns: pd.Series | Sequence[float], periods_per_year: int = 252) -> float:
    series = _clean_returns(returns)
    if len(series) < 2:
        return 0.0
    vol = series.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return 0.0
    return float(series.mean() / vol * math.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series | Sequence[float], periods_per_year: int = 252) -> float:
    """Annualised Sortino ratio using downside deviation about zero.

    Downside deviation is ``sqrt(mean(min(r, 0)^2))`` over the full sample,
    including zeros and positive returns, not the sample standard deviation
    of the negative observations only.
    """

    series = _clean_returns(returns)
    if len(series) < 2:
        return 0.0
    downside_dev = math.sqrt(float(np.mean(np.minimum(series.to_numpy(dtype=float), 0.0) ** 2)))
    if not math.isfinite(downside_dev) or downside_dev == 0.0:
        mean = float(series.mean())
        if mean > 0.0:
            return math.inf
        return 0.0
    return float(series.mean() / downside_dev * math.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series | Sequence[float]) -> float:
    """Peak-to-trough drawdown, including a loss on the first period.

    Equity is started at 1.0 *before* applying the first return, so a path
    that opens with a loss (for example ``[-0.2, 0.1]``) reports -0.20 rather
    than 0.0.
    """

    series = _clean_returns(returns)
    if series.empty:
        return 0.0
    equity = np.concatenate(([1.0], (1.0 + series.to_numpy(dtype=float)).cumprod()))
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    return float(np.min(drawdown))


def calmar_ratio(returns: pd.Series | Sequence[float], periods_per_year: int = 252) -> float:
    series = _clean_returns(returns)
    if series.empty:
        return 0.0
    annual_return = series.mean() * periods_per_year
    drawdown = abs(max_drawdown(series))
    if drawdown == 0:
        return 0.0
    return float(annual_return / drawdown)


def probabilistic_sharpe_ratio(
    returns: pd.Series | Sequence[float],
    benchmark_sr: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Estimate P(true Sharpe > benchmark Sharpe).

    The calculation follows the Bailey and Lopez de Prado probabilistic Sharpe
    adjustment for non-normal returns. Inputs and outputs use annualized Sharpe
    units, while the test statistic is computed in per-period units.
    """

    series = pd.Series(returns, dtype=float)
    n = len(series)
    if n < 4 or not np.isfinite(series).all() or not np.isfinite(benchmark_sr) or not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("PSR requires finite returns, benchmark and at least four observations")
    if not np.isfinite(series.std(ddof=1)) or series.std(ddof=1) <= 0:
        raise ValueError("PSR requires positive finite variance")

    sr_period = sharpe_ratio(series, periods_per_year=1)
    benchmark_period = benchmark_sr / math.sqrt(periods_per_year)
    series_skew = float(series.skew())
    series_kurtosis = float(series.kurt()) + 3.0

    adjustment = 1.0 - series_skew * sr_period + ((series_kurtosis - 1.0) / 4.0) * sr_period**2
    if not np.isfinite([sr_period, series_skew, series_kurtosis, adjustment]).all() or adjustment <= 0:
        raise ValueError("PSR requires finite moments and positive adjustment")
    denominator = math.sqrt(adjustment)
    z_score = (sr_period - benchmark_period) * math.sqrt(n - 1) / denominator
    return float(_normal_cdf(z_score))


def deflated_sharpe_ratio(
    returns: pd.Series | Sequence[float],
    trial_sharpes: Sequence[float],
    periods_per_year: int = 252,
    *,
    n_trials: int | None = None,
) -> float:
    """Estimate confidence using observed dispersion and a separate charged count."""

    if not np.isfinite(periods_per_year) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive and finite")
    valid_trials = np.asarray(trial_sharpes, dtype=float)
    trials = len(valid_trials) if n_trials is None else n_trials
    if type(trials) is not int or trials < max(1, len(valid_trials)):
        raise ValueError("charged trials must cover observed trials")
    if valid_trials.ndim != 1 or not np.isfinite(valid_trials).all():
        raise ValueError("trial Sharpes must be finite")
    if trials == 1 and len(valid_trials) == 1:
        return probabilistic_sharpe_ratio(returns, periods_per_year=periods_per_year)
    if len(valid_trials) < 2:
        raise ValueError("multiple trials require observed dispersion evidence")
    trials_period = valid_trials / math.sqrt(periods_per_year)
    trial_std = float(np.std(trials_period, ddof=1))
    if not np.isfinite(trial_std) or trial_std <= 0:
        raise ValueError("multiple trials require positive finite dispersion")

    euler_gamma = 0.5772156649015329
    expected_max_noise_sr = trial_std * (
        (1.0 - euler_gamma) * _normal_ppf(1.0 - 1.0 / trials)
        + euler_gamma * _normal_ppf(1.0 - 1.0 / (trials * math.e))
    )
    benchmark_annual = expected_max_noise_sr * math.sqrt(periods_per_year)
    return probabilistic_sharpe_ratio(
        returns,
        benchmark_sr=benchmark_annual,
        periods_per_year=periods_per_year,
    )
