from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeState:
    trend: str
    volatility: str
    correlation: str

    @property
    def label(self) -> str:
        return f"{self.trend}_{self.volatility}_{self.correlation}"


@dataclass
class VolTrendRegimeDetector:
    trend_lookback: int = 126
    volatility_lookback: int = 63
    correlation_lookback: int = 63

    def describe(self, prices: pd.DataFrame) -> pd.Series:
        returns = prices.pct_change().fillna(0.0)
        market = returns.mean(axis=1)

        trend_signal = prices.mean(axis=1).pct_change(self.trend_lookback)
        realized_vol = market.rolling(self.volatility_lookback).std() * np.sqrt(252)
        avg_corr = returns.rolling(self.correlation_lookback).corr().groupby(level=0).mean().mean(axis=1)

        vol_low = realized_vol.expanding(min_periods=self.volatility_lookback).quantile(0.33)
        vol_high = realized_vol.expanding(min_periods=self.volatility_lookback).quantile(0.67)
        corr_high = avg_corr.expanding(min_periods=self.correlation_lookback).quantile(0.67)

        labels = []
        for idx in prices.index:
            trend = "trend_up" if trend_signal.loc[idx] > 0 else "trend_down"
            if pd.isna(trend_signal.loc[idx]):
                trend = "unknown_trend"

            vol_value = realized_vol.loc[idx]
            if pd.isna(vol_value):
                volatility = "unknown_vol"
            elif vol_value <= vol_low.loc[idx]:
                volatility = "low_vol"
            elif vol_value >= vol_high.loc[idx]:
                volatility = "high_vol"
            else:
                volatility = "mid_vol"

            corr_value = avg_corr.loc[idx]
            if pd.isna(corr_value):
                correlation = "unknown_corr"
            elif corr_value >= corr_high.loc[idx]:
                correlation = "high_corr"
            else:
                correlation = "normal_corr"

            labels.append(RegimeState(trend, volatility, correlation).label)

        return pd.Series(labels, index=prices.index, name="regime")
