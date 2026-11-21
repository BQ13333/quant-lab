from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from quant_lab.validation.metrics import max_drawdown, sharpe_ratio


def cost_stress(
    returns: pd.Series | Sequence[float],
    weights: pd.DataFrame | None = None,
    cost_grid: tuple[float, ...] = (0, 5, 10, 15, 25, 50, 100),
) -> pd.DataFrame:
    """Return net Sharpe, return, and max drawdown at each cost level (bps)."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    rows: list[dict[str, float]] = []

    for cost in cost_grid:
        if weights is not None and not weights.empty and cost > 0:
            turnover = weights.diff().abs().sum(axis=1).mean()
            cost_returns = series - (turnover * cost / 10000)
        else:
            cost_returns = series
        rows.append(
            {
                "cost_bps": cost,
                "net_sharpe": sharpe_ratio(cost_returns),
                "net_return": float(cost_returns.mean() * 252),
                "max_drawdown": max_drawdown(cost_returns),
            }
        )

    return pd.DataFrame(rows)


def liquidity_stress(
    returns: pd.Series | Sequence[float],
    weights: pd.DataFrame,
    adv: pd.Series,
    capital_grid: tuple[float, ...] = (1e6, 1e7, 1e8, 1e9),
) -> pd.DataFrame:
    """Capacity metrics: how much capital can be deployed before liquidity breaks."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    avg_abs_weight = weights.abs().mean().mean() if not weights.empty else 0.0
    rows: list[dict[str, float]] = []

    for capital in capital_grid:
        if avg_abs_weight > 0 and adv.sum() > 0:
            participation = (capital * avg_abs_weight) / adv.sum()
        else:
            participation = 0.0
        rows.append(
            {
                "capital": capital,
                "avg_participation": float(participation),
                "net_sharpe": float(sharpe_ratio(series)),
                "gross_return": float(series.mean() * 252),
            }
        )

    return pd.DataFrame(rows)


def fx_stress(
    returns: pd.Series | Sequence[float],
    fx_returns: pd.Series | Sequence[float],
    stress_levels: tuple[float, ...] = (0.9, 0.95, 1.0, 1.05, 1.1),
) -> pd.DataFrame:
    """Hedged/unhedged scenarios under FX stress multipliers."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    fx = pd.Series(fx_returns, dtype=float).reindex(series.index).fillna(0.0)
    rows: list[dict[str, float]] = []

    for level in stress_levels:
        stressed = series + fx * (level - 1.0)
        rows.append(
            {
                "fx_stress_level": level,
                "unhedged_return": float(series.mean() * 252),
                "hedged_return": float(stressed.mean() * 252),
                "unhedged_sharpe": sharpe_ratio(series),
                "hedged_sharpe": sharpe_ratio(stressed),
                "max_drawdown": max_drawdown(stressed),
            }
        )

    return pd.DataFrame(rows)


def country_crisis_slices(
    returns: pd.Series | Sequence[float],
    weights: pd.DataFrame,
    group_map: pd.Series,
    crisis_countries: Sequence[str],
) -> dict[str, pd.Series]:
    """Return per-crisis performance when a given country is excluded."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    results: dict[str, pd.Series] = {}

    for country in crisis_countries:
        mask = ~group_map.isin([country])
        if hasattr(weights, "columns"):
            cols = [c for c in weights.columns if c in mask.index and mask.get(c, True)]
            sub_weights = weights[cols] if cols else weights
        else:
            sub_weights = weights
        sub_returns = series.loc[sub_weights.index] if not sub_weights.empty else series
        results[country] = pd.Series(
            {
                "mean_return": float(sub_returns.mean() * 252),
                "sharpe": sharpe_ratio(sub_returns),
                "max_drawdown": max_drawdown(sub_returns),
            }
        )

    return results


def monthly_stability(returns: pd.Series | Sequence[float]) -> pd.DataFrame:
    """Monthly return stats, best/worst month, and streak analysis."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return pd.DataFrame()

    if not hasattr(series.index, "month"):
        series.index = pd.date_range(start="2000-01-01", periods=len(series), freq="B")

    monthly = series.groupby([series.index.year, series.index.month]).sum()
    if monthly.empty:
        return pd.DataFrame()

    monthly.index = pd.to_datetime(
        [f"{y}-{m:02d}-01" for y, m in monthly.index]
    )

    positive = (monthly > 0).astype(int)
    streaks = positive.groupby((positive != positive.shift()).cumsum()).sum()
    pos_streaks = streaks[streaks > 0]
    neg_streaks = streaks[streaks == 0]
    neg_streaks = neg_streaks.abs()

    return pd.DataFrame(
        {
            "mean_monthly": [float(monthly.mean())],
            "std_monthly": [float(monthly.std(ddof=1))],
            "best_month": [float(monthly.max())],
            "worst_month": [float(monthly.min())],
            "pct_positive_months": [float((monthly > 0).mean())],
            "longest_positive_streak": [int(pos_streaks.max()) if not pos_streaks.empty else 0],
            "longest_negative_streak": [int(neg_streaks.max()) if not neg_streaks.empty else 0],
        }
    )


def regime_stability(
    returns: pd.Series | Sequence[float], regimes: pd.Series | Sequence
) -> pd.DataFrame:
    """Performance per regime label."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    regime_series = pd.Series(regimes)
    if hasattr(regime_series, "index") and len(regime_series.index) == len(series):
        regime_series.index = series.index
    aligned = pd.concat([series.rename("returns"), regime_series.rename("regime")], axis=1).dropna()
    if aligned.empty:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    for label, group in aligned.groupby("regime"):
        rows.append(
            {
                "regime": label,
                "n_days": len(group),
                "mean_return": float(group["returns"].mean() * 252),
                "sharpe": sharpe_ratio(group["returns"]),
                "max_drawdown": max_drawdown(group["returns"]),
            }
        )

    return pd.DataFrame(rows)


def bootstrap_confidence_interval(
    returns: pd.Series | Sequence[float],
    statistic: str = "sharpe",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for a chosen statistic."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 2:
        return (0.0, 0.0, 0.0)

    rng = np.random.default_rng(42)
    bootstraps: list[float] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(series.values, size=len(series), replace=True)
        sample_series = pd.Series(sample)
        if statistic == "sharpe":
            bootstraps.append(sharpe_ratio(sample_series))
        elif statistic == "mean":
            bootstraps.append(float(sample_series.mean() * 252))
        elif statistic == "drawdown":
            bootstraps.append(max_drawdown(sample_series))
        else:
            bootstraps.append(sharpe_ratio(sample_series))

    point = bootstraps[0] if statistic == "sharpe" else sharpe_ratio(series)
    lower = float(np.percentile(bootstraps, ((1 - confidence) / 2) * 100))
    upper = float(np.percentile(bootstraps, (1 - (1 - confidence) / 2) * 100))
    return (lower, upper, point)


def no_single_country_dominance(
    returns: pd.Series | Sequence[float],
    weights: pd.DataFrame,
    group_map: pd.Series,
    max_country_contribution: float = 0.5,
) -> bool:
    """True if no single country contributes more than max_country_contribution of absolute PnL."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if weights.empty or group_map.empty:
        return True

    abs_pnl = (weights.abs().sum(axis=1).reindex(series.index).fillna(0.0) * series.abs()).sum()
    if abs_pnl == 0:
        return True

    for country in group_map.unique():
        country_assets = group_map[group_map == country].index
        if len(country_assets) == 0:
            continue
        country_cols = [c for c in country_assets if c in weights.columns]
        if not country_cols:
            continue
        country_pnl = (
            weights[country_cols].abs().sum(axis=1).reindex(series.index).fillna(0.0) * series.abs()
        ).sum()
        if country_pnl / abs_pnl > max_country_contribution:
            return False

    return True


def no_single_month_dominance(
    returns: pd.Series | Sequence[float],
    max_month_contribution: float = 0.3,
) -> bool:
    """True if no single month contributes more than max_month_contribution of total absolute return."""

    series = pd.Series(returns, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return True

    if not hasattr(series.index, "month"):
        series.index = pd.date_range(start="2000-01-01", periods=len(series), freq="B")

    monthly = series.groupby([series.index.year, series.index.month]).sum().abs()
    total = monthly.sum()
    if total == 0:
        return True

    return bool((monthly / total).max() <= max_month_contribution)
