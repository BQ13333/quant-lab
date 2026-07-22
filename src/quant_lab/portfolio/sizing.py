from __future__ import annotations

import numpy as np
import pandas as pd


def scores_to_weights(scores: pd.DataFrame, gross_exposure: float = 1.0) -> pd.DataFrame:
    """Convert cross-sectional scores into dollar-neutral target weights."""

    demeaned = scores.sub(scores.mean(axis=1), axis=0)
    gross = demeaned.abs().sum(axis=1).replace(0.0, np.nan)
    weights = demeaned.div(gross, axis=0).fillna(0.0) * gross_exposure
    return weights


def scores_to_long_only_weights(
    scores: pd.DataFrame,
    gross_exposure: float = 1.0,
    top_fraction: float = 0.33,
) -> pd.DataFrame:
    """Convert scores into long-only weights on the top-ranked names."""

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")

    ranks = scores.rank(axis=1, pct=True, method="first")
    threshold = 1.0 - top_fraction
    selected = ranks.ge(threshold) if threshold <= 0.0 else ranks.gt(threshold)
    raw = ranks.where(selected, 0.0).clip(lower=0.0)
    gross = raw.sum(axis=1).replace(0.0, np.nan)
    return raw.div(gross, axis=0).fillna(0.0) * gross_exposure


def _cap_and_redistribute(row: pd.Series, gross_exposure: float, max_weight: float) -> pd.Series:
    """Cap one long-only row and redistribute excess by original rank weight."""

    capped = pd.Series(0.0, index=row.index, dtype=float)
    positive = row.replace([np.inf, -np.inf], np.nan).dropna().clip(lower=0.0)
    positive = positive[positive > 0.0]
    if positive.empty or gross_exposure <= 0.0:
        return capped

    remaining_scores = positive.copy()
    remaining_gross = float(gross_exposure)
    tolerance = 1e-12
    for _ in range(len(positive) + 1):
        if remaining_scores.empty or remaining_gross <= tolerance:
            break
        score_sum = remaining_scores.sum()
        if score_sum <= 0.0 or pd.isna(score_sum):
            break
        proposed = remaining_scores / score_sum * remaining_gross
        over_cap = proposed > max_weight + tolerance
        if not over_cap.any():
            capped.loc[proposed.index] = proposed.clip(upper=max_weight)
            break

        cap_names = proposed[over_cap].index
        capped.loc[cap_names] = max_weight
        remaining_gross -= max_weight * len(cap_names)
        remaining_scores = remaining_scores.drop(cap_names)

    return capped


def scores_to_long_only_capped_weights(
    scores: pd.DataFrame,
    gross_exposure: float = 1.0,
    top_fraction: float = 0.33,
    max_weight: float = 0.05,
) -> pd.DataFrame:
    """Convert scores into long-only top-ranked weights with a per-name cap."""

    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")
    uncapped = scores_to_long_only_weights(
        scores,
        gross_exposure=gross_exposure,
        top_fraction=top_fraction,
    )
    active = uncapped.clip(lower=0.0).fillna(0.0)
    capped = pd.DataFrame(0.0, index=active.index, columns=active.columns)
    remaining_gross = pd.Series(float(gross_exposure), index=active.index)
    tolerance = 1e-12

    for _ in range(len(active.columns) + 1):
        active_sum = active.sum(axis=1).replace(0.0, np.nan)
        proposed = active.div(active_sum, axis=0).mul(remaining_gross, axis=0).fillna(0.0)
        over_cap = proposed > max_weight + tolerance
        if not bool(over_cap.any().any()):
            capped = capped + proposed.clip(upper=max_weight)
            break

        newly_capped = over_cap & active.gt(0.0)
        if not bool(newly_capped.any().any()):
            capped = capped + proposed.clip(upper=max_weight)
            break

        capped = capped + newly_capped.astype(float) * max_weight
        remaining_gross = (remaining_gross - newly_capped.sum(axis=1) * max_weight).clip(lower=0.0)
        active = active.mask(newly_capped, 0.0)

    return capped.fillna(0.0)


def scores_to_group_neutral_weights(
    scores: pd.DataFrame,
    groups: pd.Series,
    gross_exposure: float = 1.0,
) -> pd.DataFrame:
    """Convert scores into dollar-neutral weights with equal gross per group."""

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    aligned_groups = groups.reindex(scores.columns)
    valid_groups = [group for group in aligned_groups.dropna().unique()]
    if not valid_groups:
        return weights

    group_gross = gross_exposure / len(valid_groups)
    for group in valid_groups:
        columns = aligned_groups[aligned_groups == group].index
        weights.loc[:, columns] = scores_to_weights(scores.loc[:, columns], gross_exposure=group_gross)
    return weights


def scores_to_group_neutral_long_only_weights(
    scores: pd.DataFrame,
    groups: pd.Series,
    gross_exposure: float = 1.0,
    top_fraction: float = 0.33,
) -> pd.DataFrame:
    """Convert scores into long-only weights with equal gross per group."""

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    aligned_groups = groups.reindex(scores.columns)
    valid_groups = [group for group in aligned_groups.dropna().unique()]
    if not valid_groups:
        return weights

    group_gross = gross_exposure / len(valid_groups)
    for group in valid_groups:
        columns = aligned_groups[aligned_groups == group].index
        weights.loc[:, columns] = scores_to_long_only_weights(
            scores.loc[:, columns],
            gross_exposure=group_gross,
            top_fraction=top_fraction,
        )
    return weights


def scores_to_group_neutral_long_only_capped_weights(
    scores: pd.DataFrame,
    groups: pd.Series,
    gross_exposure: float = 1.0,
    top_fraction: float = 0.33,
    max_weight: float = 0.05,
) -> pd.DataFrame:
    """Convert scores into long-only equal-group weights with a per-name cap."""

    if max_weight <= 0.0:
        raise ValueError("max_weight must be positive")

    weights = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    aligned_groups = groups.reindex(scores.columns)
    valid_groups = [group for group in aligned_groups.dropna().unique()]
    if not valid_groups:
        return weights

    group_gross = gross_exposure / len(valid_groups)
    for group in valid_groups:
        columns = aligned_groups[aligned_groups == group].index
        weights.loc[:, columns] = scores_to_long_only_capped_weights(
            scores.loc[:, columns],
            gross_exposure=group_gross,
            top_fraction=top_fraction,
            max_weight=max_weight,
        )
    return weights


def scores_to_turnover_constrained_weights(
    scores: pd.DataFrame,
    previous_weights: pd.DataFrame,
    max_turnover: float = 0.5,
) -> pd.DataFrame:
    """Optimize weights subject to a turnover constraint from previous weights."""
    from quant_lab.portfolio.optimizers import turnover_constrained_optimizer

    return turnover_constrained_optimizer(scores, previous_weights, max_turnover=max_turnover)


def scores_to_no_trade_band_weights(
    scores: pd.DataFrame,
    previous_weights: pd.DataFrame,
    band: float = 0.05,
) -> pd.DataFrame:
    """Only rebalance if weight change exceeds band; otherwise keep previous weights."""
    from quant_lab.portfolio.optimizers import no_trade_band_optimizer

    return no_trade_band_optimizer(scores, previous_weights, band=band)


def downsample_rebalance(
    weights: pd.DataFrame,
    frequency: str = "M",
) -> pd.DataFrame:
    """Downsample daily weights to monthly or weekly rebalancing (hold constant between)."""
    from quant_lab.portfolio.optimizers import rebalance_schedule

    if frequency not in {"D", "W", "M"}:
        raise ValueError("frequency must be one of 'D', 'W', 'M'")
    return rebalance_schedule(weights, frequency=frequency)


def volatility_target(
    weights: pd.DataFrame,
    asset_returns: pd.DataFrame,
    target_volatility: float,
    lookback: int = 63,
    periods_per_year: int = 252,
    max_leverage: float = 3.0,
) -> pd.DataFrame:
    """Scale weights to an ex-ante annualized volatility target."""

    shifted_weights = weights.shift(1).fillna(0.0)
    portfolio_returns = (shifted_weights * asset_returns).sum(axis=1)
    min_periods = max(5, min(20, lookback))
    realized_vol = portfolio_returns.rolling(lookback, min_periods=min_periods).std().shift(1) * np.sqrt(periods_per_year)
    scale = (target_volatility / realized_vol).clip(upper=max_leverage).replace([np.inf, -np.inf], np.nan)
    return weights.mul(scale.ffill().fillna(1.0), axis=0)
