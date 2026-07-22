from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_lab.contracts import SignalFrame
from quant_lab.models.base import AlphaModel
from quant_lab.models.baselines import (
    LowVolatilityModel,
    MeanReversionModel,
    MomentumModel,
    VolatilityAdjustedMomentumModel,
    _zscore_cross_section,
)
from quant_lab.regimes import VolTrendRegimeDetector


# ---------------------------------------------------------------------------
# Extended variants of existing baseline models
# ---------------------------------------------------------------------------

@dataclass
class LowVolatilityModel126(LowVolatilityModel):
    lookback: int = 126


@dataclass
class LowVolatilityModel252(LowVolatilityModel):
    lookback: int = 252


@dataclass
class MomentumModel21(MomentumModel):
    lookback: int = 21


@dataclass
class MomentumModel126(MomentumModel):
    lookback: int = 126


@dataclass
class MomentumModel252(MomentumModel):
    lookback: int = 252


@dataclass
class VolatilityAdjustedMomentumModel126(VolatilityAdjustedMomentumModel):
    lookback: int = 126


@dataclass
class VolatilityAdjustedMomentumModelVol21(VolatilityAdjustedMomentumModel):
    volatility_lookback: int = 21


@dataclass
class MeanReversionModel21(MeanReversionModel):
    lookback: int = 21


@dataclass
class MeanReversionModel63(MeanReversionModel):
    lookback: int = 63


# ---------------------------------------------------------------------------
# New zoo models
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesMomentumModel(AlphaModel):
    """Time-series momentum: score = sign of past return per asset, optionally scaled by vol."""

    lookback: int = 63
    scale_by_vol: bool = False
    vol_lookback: int = 63

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        momentum = prices.pct_change(self.lookback, fill_method=None)
        scores = np.sign(momentum)
        if self.scale_by_vol:
            returns = prices.pct_change(fill_method=None)
            vol = returns.rolling(
                self.vol_lookback, min_periods=max(5, self.vol_lookback // 3)
            ).std()
            scores = scores.div(vol.replace(0.0, np.nan))
        scores = _zscore_cross_section(scores).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "time_series_momentum",
                "lookback": self.lookback,
                "scale_by_vol": self.scale_by_vol,
                "vol_lookback": self.vol_lookback,
            },
        )


@dataclass
class TrendLowVolEnsembleModel(AlphaModel):
    """Equal-weight ensemble of a trend model and a low-vol model."""

    trend_model: AlphaModel = field(
        default_factory=lambda: MomentumModel(name="trend")
    )
    lowvol_model: AlphaModel = field(
        default_factory=lambda: LowVolatilityModel(name="lowvol")
    )

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        trend_signal = self.trend_model.predict(prices, start, end)
        lowvol_signal = self.lowvol_model.predict(prices, start, end)
        scores = (trend_signal.scores + lowvol_signal.scores) / 2.0
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "trend_low_vol_ensemble",
                "trend_model": self.trend_model.name,
                "lowvol_model": self.lowvol_model.name,
            },
        )


@dataclass
class BreakoutModel(AlphaModel):
    """Donchian channel breakout: score = +1 if close > highest high, -1 if close < lowest low."""

    lookback: int = 20

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        highest_high = prices.shift(1).rolling(
            self.lookback, min_periods=max(5, self.lookback // 3)
        ).max()
        lowest_low = prices.shift(1).rolling(
            self.lookback, min_periods=max(5, self.lookback // 3)
        ).min()
        scores = pd.DataFrame(
            np.where(
                prices > highest_high,
                1.0,
                np.where(prices < lowest_low, -1.0, 0.0),
            ),
            index=prices.index,
            columns=prices.columns,
        )
        scores = _zscore_cross_section(scores).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={"family": "breakout", "lookback": self.lookback},
        )


@dataclass
class VolumeConfirmedMomentumModel(AlphaModel):
    """Momentum score weighted by relative volume (volume / rolling ADV)."""

    lookback: int = 63
    volume_lookback: int = 63
    volume: pd.DataFrame | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if self.volume is None:
            raise ValueError("volume must be provided to VolumeConfirmedMomentumModel")
        momentum = prices.pct_change(self.lookback, fill_method=None)
        adv = self.volume.rolling(
            self.volume_lookback, min_periods=max(5, self.volume_lookback // 3)
        ).mean()
        rel_volume = self.volume.div(adv.replace(0.0, np.nan))
        raw = momentum * rel_volume
        scores = _zscore_cross_section(raw).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "volume_confirmed_momentum",
                "lookback": self.lookback,
                "volume_lookback": self.volume_lookback,
            },
        )


@dataclass
class VolatilityCompressionModel(AlphaModel):
    """Score = -1 on vol compression, +1 on vol expansion, z-scored cross-section."""

    vol_lookback: int = 21
    compression_lookback: int = 63

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        returns = prices.pct_change(fill_method=None)
        current_vol = returns.rolling(
            self.vol_lookback, min_periods=max(5, self.vol_lookback // 3)
        ).std()
        vol_lower = current_vol.rolling(self.compression_lookback).quantile(0.25)
        vol_upper = current_vol.rolling(self.compression_lookback).quantile(0.75)
        scores = pd.DataFrame(
            np.where(
                current_vol < vol_lower,
                -1.0,
                np.where(current_vol > vol_upper, 1.0, 0.0),
            ),
            index=prices.index,
            columns=prices.columns,
        )
        scores = _zscore_cross_section(scores).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "volatility_compression",
                "vol_lookback": self.vol_lookback,
                "compression_lookback": self.compression_lookback,
            },
        )


@dataclass
class CountryRelativeMomentumModel(AlphaModel):
    """Momentum relative to country average, then z-scored globally."""

    lookback: int = 63
    groups: pd.Series | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if self.groups is None:
            raise ValueError("groups must be provided to CountryRelativeMomentumModel")
        momentum = prices.pct_change(self.lookback, fill_method=None)
        aligned_groups = self.groups.reindex(prices.columns)
        country_mean = momentum.T.groupby(aligned_groups).transform("mean").T
        relative = momentum - country_mean
        scores = _zscore_cross_section(relative).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "country_relative_momentum",
                "lookback": self.lookback,
            },
        )


@dataclass
class WithinCountryRankingModel(AlphaModel):
    """Z-score momentum within each country."""

    lookback: int = 63
    groups: pd.Series | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if self.groups is None:
            raise ValueError("groups must be provided to WithinCountryRankingModel")
        momentum = prices.pct_change(self.lookback, fill_method=None)
        aligned_groups = self.groups.reindex(prices.columns)

        def _zscore_group(x: pd.Series) -> pd.Series:
            std = x.std()
            if std == 0 or pd.isna(std):
                return pd.Series(0.0, index=x.index)
            return (x - x.mean()) / std

        scores = momentum.T.groupby(aligned_groups).transform(_zscore_group).T
        scores = scores.iloc[start:end].fillna(0.0)
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "within_country_ranking",
                "lookback": self.lookback,
            },
        )


@dataclass
class CrossCountryRankingModel(AlphaModel):
    """Rank countries by average momentum, then assign country rank to stocks."""

    lookback: int = 63
    groups: pd.Series | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if self.groups is None:
            raise ValueError("groups must be provided to CrossCountryRankingModel")
        momentum = prices.pct_change(self.lookback, fill_method=None)
        aligned_groups = self.groups.reindex(prices.columns)
        country_momentum = momentum.T.groupby(aligned_groups).mean().T
        country_zscore = _zscore_cross_section(country_momentum)
        scores = pd.DataFrame(
            index=prices.index, columns=prices.columns, dtype=float
        )
        for col in prices.columns:
            country = aligned_groups.get(col)
            if country is not None and country in country_zscore.columns:
                scores[col] = country_zscore[country]
            else:
                scores[col] = 0.0
        scores = scores.iloc[start:end].fillna(0.0)
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "cross_country_ranking",
                "lookback": self.lookback,
            },
        )


@dataclass
class RegimeConditionedModel(AlphaModel):
    """Wraps another model and only emits signals when the regime matches."""

    model: AlphaModel | None = field(default=None)
    regime: str | None = field(default=None)
    detector: VolTrendRegimeDetector = field(default_factory=VolTrendRegimeDetector)
    hold_previous: bool = False

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if self.model is None:
            raise ValueError("model must be provided to RegimeConditionedModel")
        if self.regime is None:
            raise ValueError("regime must be provided to RegimeConditionedModel")
        regimes = self.detector.describe(prices)
        model_signal = self.model.predict(prices, start, end)
        scores = model_signal.scores.copy()
        window_regimes = regimes.iloc[start:end]
        mask = window_regimes.apply(
            lambda r: self.regime in r if pd.notna(r) else False
        )
        if not mask.any():
            # No rows match the regime
            scores[:] = 0.0
        else:
            scores.loc[~mask] = 0.0
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "regime_conditioned",
                "regime": self.regime,
                "wrapped_model": self.model.name,
            },
        )
