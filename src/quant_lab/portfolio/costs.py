from __future__ import annotations

import numpy as np
import pandas as pd


# ASEAN country cost schedule (decimal terms, e.g. 0.0015 = 15 bps)
_COUNTRY_SCHEDULE: dict[str, float] = {
    "Singapore": 0.0015,
    "Indonesia": 0.0025,
    "Malaysia": 0.0020,
}


def _trades(weights: pd.DataFrame, starting_holdings: pd.Series | None = None) -> pd.DataFrame:
    """Absolute trades against explicit holdings, defaulting to cash."""
    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        raise ValueError("weights must be finite")
    start = pd.Series(0.0, index=weights.columns) if starting_holdings is None else starting_holdings.reindex(weights.columns)
    if not np.isfinite(start.to_numpy(dtype=float)).all():
        raise ValueError("starting holdings must be complete and finite")
    trades = weights.diff()
    if len(weights):
        trades.iloc[0] = weights.iloc[0] - start
    return trades.abs()


def _validated_adv(adv: pd.Series, weights: pd.DataFrame, capital: float) -> pd.Series:
    aligned = adv.reindex(weights.columns)
    if not np.isfinite(capital) or capital <= 0 or not np.isfinite(aligned.to_numpy(dtype=float)).all() or (aligned <= 0).any():
        raise ValueError("capacity requires positive finite capital and complete positive finite ADV")
    return aligned


def linear_transaction_cost(weights: pd.DataFrame, cost_bps: float, *, starting_holdings: pd.Series | None = None) -> pd.Series:
    """Estimate daily cost from absolute weight turnover and one-way bps cost."""

    turnover = _trades(weights, starting_holdings).sum(axis=1)
    return turnover * (cost_bps / 10_000.0)


def capacity_score(
    weights: pd.DataFrame,
    average_daily_dollar_volume: pd.Series,
    capital: float,
    *, starting_holdings: pd.Series | None = None,
) -> pd.Series:
    """Return participation as target dollar trade divided by ADV."""

    dollar_trades = _trades(weights, starting_holdings).mul(capital)
    aligned_adv = _validated_adv(average_daily_dollar_volume, weights, capital)
    return dollar_trades.div(aligned_adv, axis=1).max(axis=1).fillna(0.0)


def country_cost_schedule(country: str, tier: str = "stock") -> float:
    """Return one-way transaction cost in decimal terms for a given country and tier."""
    if country not in _COUNTRY_SCHEDULE:
        raise ValueError(f"unsupported country: {country}")
    if tier not in {"stock", "etf"}:
        raise ValueError(f"unsupported tier: {tier}")
    if tier == "etf":
        return 0.0005  # 5 bps
    return _COUNTRY_SCHEDULE[country]


def bps_cost_grid() -> list[float]:
    """Return a grid of cost assumptions in decimal terms."""
    return [0.0, 0.0005, 0.001, 0.0015, 0.0025, 0.005, 0.01]


def adv_participation_cost(
    weights: pd.DataFrame,
    adv: pd.Series,
    capital: float = 1e8,
    max_participation: float = 0.1,
    *, starting_holdings: pd.Series | None = None,
) -> pd.Series:
    """Compute daily ADV-based penalty cost when participation exceeds max_participation.

    Dollar trade = |weight diff| * capital.
    participation = trade / adv.
    If participation > max_participation:
        penalty = (participation / max_participation - 1) * 50 bps (in decimal).
    """
    dollar_trade = _trades(weights, starting_holdings).mul(capital)
    aligned_adv = _validated_adv(adv, weights, capital)
    if not np.isfinite(max_participation) or max_participation <= 0:
        raise ValueError("max_participation must be positive and finite")
    participation = dollar_trade.div(aligned_adv, axis=1)
    excess = participation / max_participation - 1.0
    penalty = excess.clip(lower=0.0) * 0.005  # 50 bps in decimal
    return penalty.sum(axis=1).fillna(0.0)


def spread_proxy_from_ohlcv(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
) -> pd.DataFrame:
    """Estimate daily spread proxy from OHLCV data using (high - low) / close."""
    return (high - low) / close.replace(0.0, pd.NA)


def slippage_cost(
    weights: pd.DataFrame,
    spreads: pd.DataFrame,
    participation: float = 0.05,
    *, starting_holdings: pd.Series | None = None,
) -> pd.Series:
    """Compute daily slippage cost from spread proxy and weight turnover.

    slippage = spread * participation * |weight diff|.
    """
    weight_diff = _trades(weights, starting_holdings)
    aligned_spreads = spreads.reindex(weights.index).reindex(weights.columns, axis=1)
    slippage = aligned_spreads * participation * weight_diff
    return slippage.sum(axis=1).fillna(0.0)


def total_cost(
    weights: pd.DataFrame,
    cost_bps: float,
    spreads: pd.DataFrame | None = None,
    adv: pd.Series | None = None,
    capital: float | None = None,
    country_schedule: dict[str, float] | None = None,
    *, starting_holdings: pd.Series | None = None,
) -> tuple[pd.Series, dict[str, pd.Series]]:
    """Combine linear cost, ADV penalty, and slippage into total daily cost.

    cost_bps is expected in decimal terms (e.g. 0.0015 for 15 bps).
    Returns total daily cost Series and a dict of named components.
    """
    components: dict[str, pd.Series] = {}

    turnover = _trades(weights, starting_holdings).sum(axis=1)

    if country_schedule is not None:
        per_asset_turnover = _trades(weights, starting_holdings)
        country_cost = per_asset_turnover.copy()
        for asset in country_cost.columns:
            if asset not in country_schedule or not np.isfinite(country_schedule[asset]) or country_schedule[asset] < 0:
                raise ValueError(f'missing or invalid asset schedule: {asset}')
            country_cost[asset] *= country_schedule[asset]
        components["linear"] = country_cost.sum(axis=1)
        components["country"] = components["linear"].copy()
    else:
        components["linear"] = turnover * cost_bps

    if adv is not None and capital is not None:
        adv_cost = adv_participation_cost(weights, adv, capital=capital, starting_holdings=starting_holdings)
        components["adv_penalty"] = adv_cost
    else:
        components["adv_penalty"] = pd.Series(0.0, index=weights.index)

    if spreads is not None:
        slip = slippage_cost(weights, spreads, starting_holdings=starting_holdings)
        components["slippage"] = slip
    else:
        components["slippage"] = pd.Series(0.0, index=weights.index)

    total = (
        components["linear"]
        + components["adv_penalty"]
        + components["slippage"]
    )
    return total, components


def turnover_decomposition(
    weights: pd.DataFrame,
    metadata: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Decompose turnover by model, country, month, and top asset contributors.

    Returns a DataFrame indexed by date with columns:
    - total_turnover
    - by_month
    - by_model (if MultiIndex columns or metadata has 'model')
    - by_country (if metadata has 'country')
    - by_asset (top contributors)
    """
    turnover = weights.diff().abs().fillna(weights.abs())
    total_turnover = turnover.sum(axis=1)

    result = pd.DataFrame(
        {
            "total_turnover": total_turnover,
            "by_month": pd.to_datetime(weights.index).to_period("M").astype(str),
        },
        index=weights.index,
    )
    result.index.name = "date"

    # by_model from MultiIndex columns or metadata
    if isinstance(turnover.columns, pd.MultiIndex):
        model_groups = {}
        for model in turnover.columns.get_level_values(0).unique():
            model_assets = turnover.loc[:, turnover.columns.get_level_values(0) == model]
            model_groups[model] = model_assets.sum(axis=1)
        by_model = pd.DataFrame(model_groups)
        result["by_model"] = by_model.apply(
            lambda row: ", ".join(
                f"{k}: {v:.4f}" for k, v in row.items() if v > 0
            ),
            axis=1,
        )
    elif metadata is not None and "model" in metadata.columns:
        model_groups = {}
        for model in metadata["model"].dropna().unique():
            assets = metadata[metadata["model"] == model].index
            common_assets = [a for a in assets if a in turnover.columns]
            if common_assets:
                model_groups[model] = turnover[common_assets].sum(axis=1)
        if model_groups:
            by_model = pd.DataFrame(model_groups)
            result["by_model"] = by_model.apply(
                lambda row: ", ".join(
                    f"{k}: {v:.4f}" for k, v in row.items() if v > 0
                ),
                axis=1,
            )

    if metadata is not None and "country" in metadata.columns:
        country_groups = {}
        for country in metadata["country"].dropna().unique():
            assets = metadata[metadata["country"] == country].index
            common_assets = [a for a in assets if a in turnover.columns]
            if common_assets:
                country_groups[country] = turnover[common_assets].sum(axis=1)
        if country_groups:
            by_country = pd.DataFrame(country_groups)
            result["by_country"] = by_country.apply(
                lambda row: ", ".join(
                    f"{k}: {v:.4f}" for k, v in row.items() if v > 0
                ),
                axis=1,
            )

    # Top asset contributors per day
    top_n = 3
    top_contributors = turnover.apply(
        lambda row: ", ".join(
            f"{k}: {v:.4f}" for k, v in row.nlargest(top_n).items() if v > 0
        ),
        axis=1,
    )
    result["by_asset"] = top_contributors

    return result
