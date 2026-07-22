from __future__ import annotations

import numpy as np
import pandas as pd


def daily_rebalance(weights: pd.DataFrame) -> pd.DataFrame:
    """Identity rebalance: use target weights every day."""
    return weights.copy()


def weekly_rebalance(weights: pd.DataFrame, day_of_week: int = 4) -> pd.DataFrame:
    """Hold weights constant between specified day-of-week rebalances (default Friday)."""
    if not isinstance(weights.index, pd.DatetimeIndex):
        weights = weights.copy()
        weights.index = pd.to_datetime(weights.index)

    is_target = weights.index.dayofweek == day_of_week
    if not is_target.any():
        # Fallback to the last trading day of each ISO week
        week_periods = weights.index.to_period("W")
        is_target = ~week_periods.duplicated(keep="last")

    rebalanced = weights.loc[is_target]
    result = rebalanced.reindex(weights.index, method="ffill")
    return result.fillna(0.0)


def monthly_rebalance(
    weights: pd.DataFrame, day_of_month: int | None = None
) -> pd.DataFrame:
    """Hold weights constant between monthly rebalances.

    If day_of_month is None, rebalance on the last day of each month.
    """
    if not isinstance(weights.index, pd.DatetimeIndex):
        weights = weights.copy()
        weights.index = pd.to_datetime(weights.index)

    if day_of_month is not None:
        is_target = weights.index.day == day_of_month
        if not is_target.any():
            # Fallback to last day of each month
            is_target = ~weights.index.to_period("M").duplicated(keep="last")
    else:
        # Last day of each month
        is_target = ~weights.index.to_period("M").duplicated(keep="last")

    rebalanced = weights.loc[is_target]
    result = rebalanced.reindex(weights.index, method="ffill")
    return result.fillna(0.0)


def no_trade_band_rebalance(
    target_weights: pd.DataFrame,
    current_weights: pd.DataFrame,
    band: float = 0.02,
) -> pd.DataFrame:
    """Only rebalance when the deviation from current exceeds the band."""
    diff = (target_weights - current_weights).abs()
    should_trade = diff > band
    return target_weights.where(should_trade, current_weights)


def rank_hysteresis_rebalance(
    current_scores: pd.DataFrame | pd.Series,
    previous_scores: pd.DataFrame | pd.Series,
    threshold: float = 0.1,
) -> pd.DataFrame | pd.Series:
    """Only trade when the percentile rank change exceeds the threshold."""
    if isinstance(current_scores, pd.DataFrame):
        current_ranks = current_scores.rank(axis=1, pct=True, method="average")
        previous_ranks = previous_scores.rank(axis=1, pct=True, method="average")
    else:
        current_ranks = current_scores.rank(pct=True, method="average")
        previous_ranks = previous_scores.rank(pct=True, method="average")

    rank_change = (current_ranks - previous_ranks).abs()
    should_trade = rank_change > threshold
    return current_scores.where(should_trade, previous_scores)


def _portfolio_returns(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute portfolio returns from weights and asset returns."""
    aligned_returns = returns.reindex(weights.index)
    shifted_weights = weights.shift(1).fillna(0.0)
    return (shifted_weights * aligned_returns).sum(axis=1)


def compare_rebalance_frequencies(
    strategies: dict[str, pd.DataFrame],
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """Compare strategy performance under daily, weekly, and monthly rebalance."""
    results = []

    for name, target_weights in strategies.items():
        for freq, rebalance_fn in (
            ("daily", daily_rebalance),
            ("weekly", weekly_rebalance),
            ("monthly", monthly_rebalance),
        ):
            rebalanced = rebalance_fn(target_weights)
            port_returns = _portfolio_returns(rebalanced, returns)
            turnover = rebalanced.diff().abs().sum(axis=1).mean()

            total_return = (1 + port_returns).prod() - 1
            volatility = port_returns.std() * np.sqrt(252)
            sharpe = (
                port_returns.mean() / port_returns.std() * np.sqrt(252)
                if port_returns.std() > 0
                else 0.0
            )
            cum_returns = (1 + port_returns).cumprod()
            max_dd = (cum_returns / cum_returns.cummax() - 1).min()

            results.append(
                {
                    "strategy": name,
                    "freq": freq,
                    "total_return": total_return,
                    "volatility": volatility,
                    "sharpe": sharpe,
                    "max_drawdown": max_dd,
                    "mean_turnover": turnover,
                }
            )

    return pd.DataFrame(results)
