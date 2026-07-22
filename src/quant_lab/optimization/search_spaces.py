from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSearchSpace:
    """Hyperparameter space for alpha model selection."""

    lookbacks: tuple[int, ...] = (21, 63, 126, 252)
    vol_lookbacks: tuple[int, ...] = (21, 63, 126)

    def to_dict(self) -> dict[str, tuple[Any, ...]]:
        return {
            "lookback": self.lookbacks,
            "vol_lookback": self.vol_lookbacks,
        }


@dataclass
class AllocatorSearchSpace:
    """Hyperparameter space for portfolio allocator selection."""

    weighting: tuple[str, ...] = (
        "dollar_neutral",
        "long_only",
        "group_neutral_dollar_neutral",
        "group_neutral_long_only",
    )
    top_fraction: tuple[float, ...] = (0.10, 0.20, 0.33)
    rebalance_frequency: tuple[str, ...] = ("D", "W", "M")

    def to_dict(self) -> dict[str, tuple[Any, ...]]:
        return {
            "weighting": self.weighting,
            "top_fraction": self.top_fraction,
            "rebalance_frequency": self.rebalance_frequency,
        }


@dataclass
class PortfolioSearchSpace:
    """Hyperparameter space for portfolio-level constraints."""

    target_volatility: tuple[float, ...] = (0.08, 0.10, 0.12, 0.15)
    max_leverage: tuple[float, ...] = (1.5, 2.0, 3.0)
    max_country_weight: tuple[float, ...] = (0.30, 0.40, 0.50)
    max_asset_weight: tuple[float, ...] = (0.10, 0.15, 0.20)

    def to_dict(self) -> dict[str, tuple[Any, ...]]:
        return {
            "target_volatility": self.target_volatility,
            "max_leverage": self.max_leverage,
            "max_country_weight": self.max_country_weight,
            "max_asset_weight": self.max_asset_weight,
        }


@dataclass
class CostSearchSpace:
    """Hyperparameter space for transaction cost modelling."""

    cost_bps: tuple[int, ...] = (0, 5, 10, 15, 25, 50)
    no_trade_band: tuple[float, ...] = (0.0, 0.01, 0.02, 0.05)
    turnover_penalty: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0)

    def to_dict(self) -> dict[str, tuple[Any, ...]]:
        return {
            "cost_bps": self.cost_bps,
            "no_trade_band": self.no_trade_band,
            "turnover_penalty": self.turnover_penalty,
        }


@dataclass
class LiquiditySearchSpace:
    """Hyperparameter space for liquidity filtering."""

    min_adv_usd: tuple[float, ...] = (1e5, 1e6, 5e6)
    min_traded_days: tuple[int, ...] = (21, 42, 63)
    max_missing_ratio: tuple[float, ...] = (0.1, 0.2, 0.3)

    def to_dict(self) -> dict[str, tuple[Any, ...]]:
        return {
            "min_adv_usd": self.min_adv_usd,
            "min_traded_days": self.min_traded_days,
            "max_missing_ratio": self.max_missing_ratio,
        }


@dataclass
class RegimeSearchSpace:
    """Hyperparameter space for regime detection."""

    regime_threshold: tuple[float, ...] = (0.33, 0.50)
    trend_lookback: tuple[int, ...] = (63, 126)
    vol_lookback: tuple[int, ...] = (21, 63)

    def to_dict(self) -> dict[str, tuple[Any, ...]]:
        return {
            "regime_threshold": self.regime_threshold,
            "trend_lookback": self.trend_lookback,
            "vol_lookback": self.vol_lookback,
        }
