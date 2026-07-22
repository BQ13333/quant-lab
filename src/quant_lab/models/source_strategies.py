from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

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
from quant_lab.models.zoo import BreakoutModel, TimeSeriesMomentumModel, TrendLowVolEnsembleModel


@dataclass(frozen=True)
class SourceStrategySpec:
    strategy_id: str
    family: str
    source_title: str
    source_file: str
    source_section: str
    implementation_status: str
    data_requirements: tuple[str, ...]
    model_builder: Callable[[str], AlphaModel] | None = None


def _source_metadata(spec: SourceStrategySpec, extra: dict[str, object] | None = None) -> dict[str, object]:
    metadata: dict[str, object] = {
        "family": spec.family,
        "source_strategy_id": spec.strategy_id,
        "source_title": spec.source_title,
        "source_file": spec.source_file,
        "source_section": spec.source_section,
        "data_requirements": ",".join(spec.data_requirements),
    }
    if extra:
        metadata.update(extra)
    return metadata


def _moving_average(prices: pd.DataFrame, lookback: int, method: str) -> pd.DataFrame:
    if method == "ema":
        return prices.ewm(span=lookback, min_periods=max(5, lookback // 3), adjust=False).mean()
    if method == "sma":
        return prices.rolling(lookback, min_periods=max(5, lookback // 3)).mean()
    raise ValueError("method must be 'sma' or 'ema'")


@dataclass
class SourceMomentumModel(MomentumModel):
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        metadata = dict(signal.metadata)
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, {"lookback": self.lookback}))
        return SignalFrame(signal.scores, self.name, self.horizon, metadata)


@dataclass
class SourceMeanReversionModel(MeanReversionModel):
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        metadata = dict(signal.metadata)
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, {"lookback": self.lookback}))
        return SignalFrame(signal.scores, self.name, self.horizon, metadata)


@dataclass
class SourceLowVolatilityModel(LowVolatilityModel):
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        metadata = dict(signal.metadata)
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, {"lookback": self.lookback}))
        return SignalFrame(signal.scores, self.name, self.horizon, metadata)


@dataclass
class SourceVolatilityAdjustedMomentumModel(VolatilityAdjustedMomentumModel):
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        metadata = dict(signal.metadata)
        if self.source_spec is not None:
            metadata.update(
                _source_metadata(
                    self.source_spec,
                    {"lookback": self.lookback, "volatility_lookback": self.volatility_lookback},
                )
            )
        return SignalFrame(signal.scores, self.name, self.horizon, metadata)


@dataclass
class SourceBreakoutModel(BreakoutModel):
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        metadata = dict(signal.metadata)
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, {"lookback": self.lookback}))
        return SignalFrame(signal.scores, self.name, self.horizon, metadata)


@dataclass
class SourceTimeSeriesMomentumModel(TimeSeriesMomentumModel):
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        metadata = dict(signal.metadata)
        if self.source_spec is not None:
            metadata.update(
                _source_metadata(
                    self.source_spec,
                    {
                        "lookback": self.lookback,
                        "scale_by_vol": self.scale_by_vol,
                        "vol_lookback": self.vol_lookback,
                    },
                )
            )
        return SignalFrame(signal.scores, self.name, self.horizon, metadata)


@dataclass
class SourceTrendLowVolEnsembleModel(TrendLowVolEnsembleModel):
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        metadata = dict(signal.metadata)
        if self.source_spec is not None:
            metadata.update(
                _source_metadata(
                    self.source_spec,
                    {
                        "family": "trend_low_vol_ensemble",
                        "trend_model": self.trend_model.name,
                        "lowvol_model": self.lowvol_model.name,
                    },
                )
            )
        return SignalFrame(signal.scores, self.name, self.horizon, metadata)


@dataclass
class MovingAverageCrossoverModel(AlphaModel):
    fast: int = 21
    slow: int = 63
    method: str = "sma"
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if self.fast >= self.slow:
            raise ValueError("fast lookback must be smaller than slow lookback")
        fast_ma = _moving_average(prices, self.fast, self.method)
        slow_ma = _moving_average(prices, self.slow, self.method)
        returns = prices.pct_change(fill_method=None)
        vol = returns.rolling(self.slow, min_periods=max(5, self.slow // 3)).std()
        raw = fast_ma.sub(slow_ma).div(prices.replace(0.0, np.nan)).div(vol.replace(0.0, np.nan))
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {
            "family": "moving_average_crossover",
            "fast": self.fast,
            "slow": self.slow,
            "method": self.method,
        }
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class TripleMovingAverageModel(AlphaModel):
    fast: int = 10
    medium: int = 40
    slow: int = 120
    method: str = "sma"
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if not self.fast < self.medium < self.slow:
            raise ValueError("lookbacks must satisfy fast < medium < slow")
        fast_ma = _moving_average(prices, self.fast, self.method)
        medium_ma = _moving_average(prices, self.medium, self.method)
        slow_ma = _moving_average(prices, self.slow, self.method)
        raw = (fast_ma > medium_ma).astype(float) + (medium_ma > slow_ma).astype(float) - 1.0
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {
            "family": "triple_moving_average",
            "fast": self.fast,
            "medium": self.medium,
            "slow": self.slow,
            "method": self.method,
        }
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class EWMACModel(AlphaModel):
    fast: int = 16
    slow: int = 64
    vol_lookback: int = 64
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if self.fast >= self.slow:
            raise ValueError("fast lookback must be smaller than slow lookback")
        fast_ema = _moving_average(prices, self.fast, "ema")
        slow_ema = _moving_average(prices, self.slow, "ema")
        returns = prices.pct_change(fill_method=None)
        vol = returns.rolling(self.vol_lookback, min_periods=max(5, self.vol_lookback // 3)).std()
        raw = fast_ema.sub(slow_ema).div(prices.replace(0.0, np.nan)).div(vol.replace(0.0, np.nan))
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {
            "family": "ewmac_trend",
            "fast": self.fast,
            "slow": self.slow,
            "vol_lookback": self.vol_lookback,
        }
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class ChannelPositionModel(AlphaModel):
    lookback: int = 55
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        high = prices.shift(1).rolling(self.lookback, min_periods=max(5, self.lookback // 3)).max()
        low = prices.shift(1).rolling(self.lookback, min_periods=max(5, self.lookback // 3)).min()
        width = high.sub(low).replace(0.0, np.nan)
        raw = prices.sub(low).div(width).mul(2.0).sub(1.0)
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {"family": "channel_position", "lookback": self.lookback}
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class MovingAverageDistanceReversionModel(AlphaModel):
    lookback: int = 63
    method: str = "sma"
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        ma = _moving_average(prices, self.lookback, self.method)
        returns = prices.pct_change(fill_method=None)
        vol = returns.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).std()
        distance = prices.div(ma.replace(0.0, np.nan)).sub(1.0).div(vol.replace(0.0, np.nan))
        scores = _zscore_cross_section(-distance).iloc[start:end]
        metadata = {"family": "moving_average_reversion", "lookback": self.lookback, "method": self.method}
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class BollingerReversionModel(AlphaModel):
    lookback: int = 63
    entry_z: float = 1.0
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        ma = prices.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).mean()
        std = prices.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).std()
        z = prices.sub(ma).div(std.replace(0.0, np.nan))
        raw = -z.where(z.abs() >= self.entry_z, 0.0)
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {"family": "bollinger_reversion", "lookback": self.lookback, "entry_z": self.entry_z}
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class ResidualMomentumModel(AlphaModel):
    lookback: int = 126
    beta_lookback: int = 126
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        returns = prices.pct_change(fill_method=None).fillna(0.0)
        market = returns.mean(axis=1)
        market_var = market.rolling(self.beta_lookback, min_periods=max(5, self.beta_lookback // 3)).var()
        beta = returns.rolling(self.beta_lookback, min_periods=max(5, self.beta_lookback // 3)).cov(market)
        beta = beta.div(market_var.replace(0.0, np.nan), axis=0).fillna(0.0)
        residual = returns.sub(beta.mul(market, axis=0), axis=0)
        raw = residual.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).sum()
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {
            "family": "residual_momentum",
            "lookback": self.lookback,
            "beta_lookback": self.beta_lookback,
        }
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class ResidualMeanReversionModel(ResidualMomentumModel):
    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        signal = super().predict(prices, start, end)
        scores = -signal.scores
        metadata = dict(signal.metadata)
        metadata["family"] = "residual_mean_reversion"
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


def _rolling_log_slope(values: np.ndarray) -> float:
    if np.isnan(values).any():
        return np.nan
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    y = np.log(np.maximum(values.astype(float), 1e-12))
    y -= y.mean()
    denom = np.dot(x, x)
    if denom == 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)


@dataclass
class RegressionTrendModel(AlphaModel):
    lookback: int = 90
    vol_lookback: int = 63
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        slope = prices.rolling(self.lookback, min_periods=self.lookback).apply(_rolling_log_slope, raw=True)
        returns = prices.pct_change(fill_method=None)
        vol = returns.rolling(self.vol_lookback, min_periods=max(5, self.vol_lookback // 3)).std()
        raw = slope.mul(252.0).div(vol.replace(0.0, np.nan))
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {"family": "regression_trend", "lookback": self.lookback, "vol_lookback": self.vol_lookback}
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class ReturnSkewnessModel(AlphaModel):
    lookback: int = 126
    prefer_positive_skew: bool = True
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        returns = prices.pct_change(fill_method=None)
        skew = returns.rolling(self.lookback, min_periods=max(20, self.lookback // 3)).skew()
        raw = skew if self.prefer_positive_skew else -skew
        scores = _zscore_cross_section(raw).iloc[start:end]
        metadata = {
            "family": "return_skewness",
            "lookback": self.lookback,
            "prefer_positive_skew": self.prefer_positive_skew,
        }
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


@dataclass
class MultiHorizonBlendModel(AlphaModel):
    models: tuple[AlphaModel, ...] = field(default_factory=tuple)
    source_spec: SourceStrategySpec | None = field(default=None, repr=False)

    def fit(self, prices: pd.DataFrame, start: int, end: int) -> "MultiHorizonBlendModel":
        for model in self.models:
            model.fit(prices, start, end)
        return self

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if not self.models:
            raise ValueError("models must not be empty")
        parts = [model.predict(prices, start, end).scores for model in self.models]
        scores = sum(parts) / len(parts)
        scores = _zscore_cross_section(scores)
        metadata = {
            "family": "alpha_combo",
            "component_models": ",".join(model.name for model in self.models),
        }
        if self.source_spec is not None:
            metadata.update(_source_metadata(self.source_spec, metadata))
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=metadata)


def build_source_strategy_specs() -> list[SourceStrategySpec]:
    """Return strategy specs with public bibliographic provenance metadata."""

    provenance = "docs/STRATEGY_PROVENANCE.md"
    kakushadze = provenance
    carver = provenance
    kaufman = provenance
    clenow = provenance
    finding = provenance
    grinold = provenance

    return [
        SourceStrategySpec(
            "kakushadze_price_momentum",
            "momentum",
            "151 Trading Strategies",
            kakushadze,
            "3.1 Price-Momentum",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "kakushadze_low_volatility_anomaly",
            "low_volatility",
            "151 Trading Strategies",
            kakushadze,
            "3.4 Low-Volatility Anomaly",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "kakushadze_residual_momentum",
            "residual_momentum",
            "151 Trading Strategies",
            kakushadze,
            "3.7 Residual Momentum",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "kakushadze_mean_reversion",
            "mean_reversion",
            "151 Trading Strategies",
            kakushadze,
            "3.9-3.10 Mean-Reversion",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "kakushadze_moving_average",
            "moving_average",
            "151 Trading Strategies",
            kakushadze,
            "3.11-3.13 Moving Averages",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "kakushadze_channel",
            "channel",
            "151 Trading Strategies",
            kakushadze,
            "3.15 Channel",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "kaufman_trend_systems",
            "trend_following",
            "Trading Systems and Methods",
            kaufman,
            "Trend, moving average, breakout and channel systems",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "carver_ewmac_and_risk_framework",
            "ewmac_trend",
            "Systematic Trading",
            carver,
            "EWMAC rule and volatility targeting framework",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "clenow_regression_trend",
            "regression_trend",
            "Following the Trend",
            clenow,
            "Diversified trend-following regression-style trend",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "worldquant_alpha_process",
            "alpha_combo",
            "Finding Alphas",
            finding,
            "Alpha design, robustness, correlation and alpha combinations",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "grinold_kahn_information_ratio",
            "portfolio_construction",
            "Advances in Active Portfolio Management",
            grinold,
            "Information ratio, breadth, transfer coefficient and active portfolio construction",
            "implemented",
            ("adj_close",),
        ),
        SourceStrategySpec(
            "kakushadze_earnings_momentum",
            "earnings_momentum",
            "151 Trading Strategies",
            kakushadze,
            "3.2 Earnings-Momentum",
            "requires_pit_fundamentals",
            ("earnings_release_date", "standardized_unexpected_earnings"),
        ),
        SourceStrategySpec(
            "kakushadze_value",
            "value",
            "151 Trading Strategies",
            kakushadze,
            "3.3 Value",
            "requires_pit_fundamentals",
            ("book_to_price", "earnings_yield", "sales_to_price"),
        ),
        SourceStrategySpec(
            "kakushadze_pairs_trading",
            "pairs_trading",
            "151 Trading Strategies",
            kakushadze,
            "3.8 Pairs Trading",
            "requires_cointegration_engine",
            ("adj_close", "stable_security_master"),
        ),
        SourceStrategySpec(
            "finding_alphas_price_volume",
            "price_volume",
            "Finding Alphas",
            finding,
            "Equity price-volume and alpha design chapters",
            "requires_volume_panel",
            ("adj_close", "volume", "adv"),
        ),
    ]


def _spec(strategy_id: str) -> SourceStrategySpec:
    specs = {spec.strategy_id: spec for spec in build_source_strategy_specs()}
    return specs[strategy_id]


def build_source_strategy_model_suite(prefix: str = "source") -> list[AlphaModel]:
    """Build a broad source-derived OHLCV-compatible strategy suite."""

    models: list[AlphaModel] = []
    mom = _spec("kakushadze_price_momentum")
    lowvol = _spec("kakushadze_low_volatility_anomaly")
    residual = _spec("kakushadze_residual_momentum")
    reversion = _spec("kakushadze_mean_reversion")
    ma = _spec("kakushadze_moving_average")
    channel = _spec("kakushadze_channel")
    kaufman = _spec("kaufman_trend_systems")
    carver = _spec("carver_ewmac_and_risk_framework")
    clenow = _spec("clenow_regression_trend")
    finding = _spec("worldquant_alpha_process")
    grinold = _spec("grinold_kahn_information_ratio")

    for lookback in (21, 63, 126, 252):
        models.append(SourceMomentumModel(name=f"{prefix}_src_mom_{lookback}", lookback=lookback, source_spec=mom))
    for lookback in (21, 63, 126, 252):
        models.append(
            SourceLowVolatilityModel(name=f"{prefix}_src_lowvol_{lookback}", lookback=lookback, source_spec=lowvol)
        )
    for lookback, vol_lookback in ((21, 21), (63, 21), (63, 63), (126, 63), (252, 126)):
        models.append(
            SourceVolatilityAdjustedMomentumModel(
                name=f"{prefix}_src_vam_{lookback}_vol{vol_lookback}",
                lookback=lookback,
                volatility_lookback=vol_lookback,
                source_spec=mom,
            )
        )
    for lookback in (5, 10, 21, 63):
        models.append(
            SourceMeanReversionModel(name=f"{prefix}_src_rev_{lookback}", lookback=lookback, source_spec=reversion)
        )
    for lookback, beta_lookback in ((63, 126), (126, 126), (252, 252)):
        models.append(
            ResidualMomentumModel(
                name=f"{prefix}_src_resid_mom_{lookback}_beta{beta_lookback}",
                lookback=lookback,
                beta_lookback=beta_lookback,
                source_spec=residual,
            )
        )
    for lookback, beta_lookback in ((21, 126), (63, 126)):
        models.append(
            ResidualMeanReversionModel(
                name=f"{prefix}_src_resid_rev_{lookback}_beta{beta_lookback}",
                lookback=lookback,
                beta_lookback=beta_lookback,
                source_spec=reversion,
            )
        )
    for fast, slow, method in ((10, 40, "sma"), (21, 63, "sma"), (21, 126, "sma"), (16, 64, "ema"), (32, 128, "ema")):
        models.append(
            MovingAverageCrossoverModel(
                name=f"{prefix}_src_ma_{method}_{fast}_{slow}",
                fast=fast,
                slow=slow,
                method=method,
                source_spec=ma,
            )
        )
    for fast, medium, slow in ((5, 20, 60), (10, 40, 120), (21, 63, 189)):
        models.append(
            TripleMovingAverageModel(
                name=f"{prefix}_src_triple_ma_{fast}_{medium}_{slow}",
                fast=fast,
                medium=medium,
                slow=slow,
                source_spec=ma,
            )
        )
    for lookback in (20, 55, 126):
        models.append(
            SourceBreakoutModel(
                name=f"{prefix}_src_breakout_{lookback}",
                lookback=lookback,
                source_spec=kaufman,
            )
        )
        models.append(ChannelPositionModel(name=f"{prefix}_src_channel_{lookback}", lookback=lookback, source_spec=channel))
    for lookback in (21, 63, 126):
        models.append(
            MovingAverageDistanceReversionModel(
                name=f"{prefix}_src_ma_rev_{lookback}",
                lookback=lookback,
                source_spec=reversion,
            )
        )
    for lookback, entry_z in ((21, 1.0), (63, 1.0), (126, 1.5)):
        models.append(
            BollingerReversionModel(
                name=f"{prefix}_src_boll_rev_{lookback}_z{str(entry_z).replace('.', '')}",
                lookback=lookback,
                entry_z=entry_z,
                source_spec=kaufman,
            )
        )
    for fast, slow in ((8, 32), (16, 64), (32, 128)):
        models.append(EWMACModel(name=f"{prefix}_src_ewmac_{fast}_{slow}", fast=fast, slow=slow, source_spec=carver))
    for lookback in (63, 126):
        models.append(
            SourceTimeSeriesMomentumModel(
                name=f"{prefix}_src_tsmom_{lookback}_vol",
                lookback=lookback,
                scale_by_vol=True,
                source_spec=carver,
            )
        )
    for lookback in (63, 126, 189):
        models.append(
            RegressionTrendModel(
                name=f"{prefix}_src_regtrend_{lookback}",
                lookback=lookback,
                source_spec=clenow,
            )
        )
    for lookback, positive in ((63, True), (126, True), (126, False)):
        label = "pos" if positive else "neg"
        models.append(
            ReturnSkewnessModel(
                name=f"{prefix}_src_skew_{label}_{lookback}",
                lookback=lookback,
                prefer_positive_skew=positive,
                source_spec=kaufman,
            )
        )
    for lookback in (63, 126, 252):
        models.append(
            SourceTrendLowVolEnsembleModel(
                name=f"{prefix}_src_trend_lowvol_{lookback}",
                trend_model=SourceMomentumModel(name=f"{prefix}_src_combo_mom_{lookback}", lookback=lookback, source_spec=finding),
                lowvol_model=SourceLowVolatilityModel(
                    name=f"{prefix}_src_combo_lowvol_{lookback}", lookback=lookback, source_spec=grinold
                ),
                source_spec=finding,
            )
        )
    models.append(
        MultiHorizonBlendModel(
            name=f"{prefix}_src_alpha_combo_mom_lowvol_resid",
            models=(
                SourceMomentumModel(name=f"{prefix}_src_combo_mom_126", lookback=126, source_spec=finding),
                SourceLowVolatilityModel(name=f"{prefix}_src_combo_lowvol_126", lookback=126, source_spec=grinold),
                ResidualMomentumModel(name=f"{prefix}_src_combo_resid_126", lookback=126, source_spec=residual),
            ),
            source_spec=finding,
        )
    )

    return models


def source_strategy_manifest() -> pd.DataFrame:
    rows = []
    for spec in build_source_strategy_specs():
        rows.append(
            {
                "strategy_id": spec.strategy_id,
                "family": spec.family,
                "source_title": spec.source_title,
                "source_file": spec.source_file,
                "source_section": spec.source_section,
                "implementation_status": spec.implementation_status,
                "data_requirements": ",".join(spec.data_requirements),
            }
        )
    return pd.DataFrame(rows)
