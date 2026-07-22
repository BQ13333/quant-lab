from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_lab.contracts import SignalFrame
from quant_lab.models.base import AlphaModel


def _zscore_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0.0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


@dataclass
class MomentumModel(AlphaModel):
    lookback: int = 63

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        raw = prices.pct_change(self.lookback, fill_method=None)
        scores = _zscore_cross_section(raw).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={"family": "momentum", "lookback": self.lookback},
        )


@dataclass
class MeanReversionModel(AlphaModel):
    lookback: int = 5

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        raw = -prices.pct_change(self.lookback, fill_method=None)
        scores = _zscore_cross_section(raw).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={"family": "mean_reversion", "lookback": self.lookback},
        )


@dataclass
class VolatilityAdjustedMomentumModel(AlphaModel):
    lookback: int = 63
    volatility_lookback: int = 63

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        returns = prices.pct_change(fill_method=None)
        momentum = prices.pct_change(self.lookback, fill_method=None)
        volatility = returns.rolling(self.volatility_lookback, min_periods=max(5, self.volatility_lookback // 3)).std().shift(1)
        raw = momentum.div(volatility.replace(0.0, np.nan))
        scores = _zscore_cross_section(raw).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "volatility_adjusted_momentum",
                "lookback": self.lookback,
                "volatility_lookback": self.volatility_lookback,
            },
        )


@dataclass
class LowVolatilityModel(AlphaModel):
    lookback: int = 63

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        returns = prices.pct_change(fill_method=None)
        raw = -returns.rolling(self.lookback, min_periods=max(5, self.lookback // 3)).std().shift(1)
        scores = _zscore_cross_section(raw).iloc[start:end]
        return SignalFrame(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={"family": "low_volatility", "lookback": self.lookback},
        )
