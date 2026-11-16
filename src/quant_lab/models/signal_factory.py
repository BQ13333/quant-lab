from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_lab.models.base import AlphaModel
from quant_lab.models.baselines import (
    LowVolatilityModel,
    MeanReversionModel,
    MomentumModel,
    VolatilityAdjustedMomentumModel,
)
from quant_lab.models.zoo import (
    BreakoutModel,
    CountryRelativeMomentumModel,
    CrossCountryRankingModel,
    TimeSeriesMomentumModel,
    TrendLowVolEnsembleModel,
    WithinCountryRankingModel,
)


ALLOWED_MARKETS = {'MY', 'CN', 'HK'}

def require_admission_markets(markets):
    if not markets or not set(markets) <= ALLOWED_MARKETS:
        raise ValueError('out-of-scope admission market')


SignalProfile = Literal["lite", "slow", "full"]


@dataclass(frozen=True)
class SignalSpec:
    name: str
    family: str
    horizon_days: int
    turnover_bias: Literal["low", "medium", "high"]
    description: str


@dataclass(frozen=True)
class SignalFactoryConfig:
    prefix: str = "signal"
    profile: SignalProfile = "slow"
    include_group_models: bool = True
    groups: pd.Series | None = None
    admission_markets: tuple[str, ...] | None = None


def build_signal_models(config: SignalFactoryConfig) -> list[AlphaModel]:
    """Build a named model suite from one repeatable configuration.

    The factory is intentionally conservative. `lite` matches the fast global
    runner. `slow` emphasizes lower-turnover horizons. `full` includes shorter
    research probes but should be held to stricter multiple-testing penalties.
    """

    if config.profile not in {'lite','slow','full'}:
        raise ValueError('unknown signal profile')
    if config.admission_markets is not None:
        require_admission_markets(config.admission_markets)
    prefix = config.prefix
    models: list[AlphaModel] = []

    if config.profile == "lite":
        models.extend(
            [
                MomentumModel(name=f"{prefix}_mom_21", lookback=21),
                MomentumModel(name=f"{prefix}_mom_63", lookback=63),
                VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_63", lookback=63, volatility_lookback=63),
                MeanReversionModel(name=f"{prefix}_rev_5", lookback=5),
                LowVolatilityModel(name=f"{prefix}_lowvol_63", lookback=63),
            ]
        )
    else:
        models.extend(
            [
                MomentumModel(name=f"{prefix}_mom_63", lookback=63),
                MomentumModel(name=f"{prefix}_mom_126", lookback=126),
                MomentumModel(name=f"{prefix}_mom_252", lookback=252),
                VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_126", lookback=126, volatility_lookback=63),
                VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_252", lookback=252, volatility_lookback=126),
                TimeSeriesMomentumModel(name=f"{prefix}_tsmom_126", lookback=126, scale_by_vol=True),
                TimeSeriesMomentumModel(name=f"{prefix}_tsmom_252", lookback=252, scale_by_vol=True),
                LowVolatilityModel(name=f"{prefix}_lowvol_126", lookback=126),
                LowVolatilityModel(name=f"{prefix}_lowvol_252", lookback=252),
                BreakoutModel(name=f"{prefix}_breakout_55", lookback=55),
                TrendLowVolEnsembleModel(
                    name=f"{prefix}_trend_lowvol_126",
                    trend_model=MomentumModel(name=f"{prefix}_trend_mom_126", lookback=126),
                    lowvol_model=LowVolatilityModel(name=f"{prefix}_trend_lowvol_leg_126", lookback=126),
                ),
            ]
        )

    if config.profile == "full":
        models.extend(
            [
                MomentumModel(name=f"{prefix}_mom_21", lookback=21),
                MeanReversionModel(name=f"{prefix}_rev_5", lookback=5),
                MeanReversionModel(name=f"{prefix}_rev_21", lookback=21),
                VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_63", lookback=63, volatility_lookback=21),
            ]
        )

    if config.include_group_models and config.groups is not None and not config.groups.empty:
        models.extend(
            [
                CountryRelativeMomentumModel(name=f"{prefix}_group_rel_mom_126", lookback=126, groups=config.groups),
                WithinCountryRankingModel(name=f"{prefix}_within_group_mom_126", lookback=126, groups=config.groups),
                CrossCountryRankingModel(name=f"{prefix}_cross_group_mom_126", lookback=126, groups=config.groups),
            ]
        )

    return models


def signal_manifest(models: list[AlphaModel]) -> pd.DataFrame:
    """Return a compact manifest for a model list."""

    rows = []
    for model in models:
        lookback = getattr(model, "lookback", None)
        family = model.__class__.__name__
        rows.append(
            {
                "model_name": model.name,
                "class_name": family,
                "lookback": lookback,
                "horizon": model.horizon,
            }
        )
    return pd.DataFrame(rows)


def default_signal_specs() -> list[SignalSpec]:
    return [
        SignalSpec("momentum_63", "momentum", 63, "medium", "Quarterly cross-sectional momentum."),
        SignalSpec("momentum_126", "momentum", 126, "low", "Six-month cross-sectional momentum."),
        SignalSpec("momentum_252", "momentum", 252, "low", "Twelve-month cross-sectional momentum."),
        SignalSpec("vol_adjusted_momentum", "trend", 126, "low", "Momentum scaled by realized volatility."),
        SignalSpec("time_series_momentum", "trend", 126, "low", "Own-asset trend with volatility scaling."),
        SignalSpec("low_volatility", "defensive", 126, "low", "Preference for lower realized volatility."),
        SignalSpec("trend_low_vol_ensemble", "ensemble", 126, "low", "Equal blend of trend and defensive legs."),
        SignalSpec("group_relative_momentum", "relative_value", 126, "low", "Momentum measured against region or country peers."),
    ]
