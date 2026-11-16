from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


class RegimeValue(str, Enum):
    UNKNOWN = "UNKNOWN"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    LOW_VOL = "low_vol"
    MID_VOL = "mid_vol"
    HIGH_VOL = "high_vol"
    HIGH_CORR = "high_corr"
    NORMAL_CORR = "normal_corr"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RegimeState:
    trend: RegimeValue
    volatility: RegimeValue
    correlation: RegimeValue

    @property
    def label(self) -> str:
        if RegimeValue.UNKNOWN in (self.trend, self.volatility, self.correlation):
            return RegimeValue.UNKNOWN.value
        return f"{self.trend}_{self.volatility}_{self.correlation}"


@dataclass
class VolTrendRegimeDetector:
    trend_lookback: int = 126
    volatility_lookback: int = 63
    correlation_lookback: int = 63

    def describe(self, prices: pd.DataFrame) -> pd.Series:
        returns = prices.pct_change().fillna(0.0)
        market = returns.mean(axis=1)

        # Equal-weight, daily rebalanced market index is independent of price units.
        market_index = (1.0 + market).cumprod()
        trend_signal = market_index.pct_change(self.trend_lookback)
        realized_vol = market.rolling(self.volatility_lookback).std() * np.sqrt(252)
        avg_corr = returns.rolling(self.correlation_lookback).corr().groupby(level=0).mean().mean(axis=1)

        vol_low = realized_vol.expanding(min_periods=self.volatility_lookback).quantile(0.33)
        vol_high = realized_vol.expanding(min_periods=self.volatility_lookback).quantile(0.67)
        corr_high = avg_corr.expanding(min_periods=self.correlation_lookback).quantile(0.67)

        labels = []
        for idx in prices.index:
            trend = RegimeValue.TREND_UP if trend_signal.loc[idx] > 0 else RegimeValue.TREND_DOWN
            if not np.isfinite(trend_signal.loc[idx]):
                trend = RegimeValue.UNKNOWN

            vol_value = realized_vol.loc[idx]
            if not np.isfinite([vol_value, vol_low.loc[idx], vol_high.loc[idx]]).all():
                volatility = RegimeValue.UNKNOWN
            elif vol_value <= vol_low.loc[idx]:
                volatility = RegimeValue.LOW_VOL
            elif vol_value >= vol_high.loc[idx]:
                volatility = RegimeValue.HIGH_VOL
            else:
                volatility = RegimeValue.MID_VOL

            corr_value = avg_corr.loc[idx]
            if not np.isfinite([corr_value, corr_high.loc[idx]]).all():
                correlation = RegimeValue.UNKNOWN
            elif corr_value >= corr_high.loc[idx]:
                correlation = RegimeValue.HIGH_CORR
            else:
                correlation = RegimeValue.NORMAL_CORR

            labels.append(RegimeState(trend, volatility, correlation).label)

        return pd.Series(labels, index=prices.index, name="regime")
