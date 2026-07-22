from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MarketPanel:
    """Aligned market data for a research universe."""

    prices: pd.DataFrame
    volumes: pd.DataFrame | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.prices.empty:
            raise ValueError("prices must not be empty")
        if not isinstance(self.prices.index, pd.DatetimeIndex):
            raise TypeError("prices index must be a DatetimeIndex")
        if self.prices.index.has_duplicates:
            raise ValueError("prices index must not contain duplicates")
        if not self.prices.index.is_monotonic_increasing:
            raise ValueError("prices index must be sorted")
        if not np.isfinite(self.prices.to_numpy(dtype=float)).all():
            raise ValueError("prices must be finite")
        if self.volumes is not None and self.volumes.shape != self.prices.shape:
            raise ValueError("volumes must match prices shape")

    @property
    def returns(self) -> pd.DataFrame:
        self.validate()
        return self.prices.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


@dataclass(frozen=True)
class SignalFrame:
    """Model output scores aligned to a market universe."""

    scores: pd.DataFrame
    model_name: str
    horizon: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate_against(self, prices: pd.DataFrame) -> None:
        if self.scores.empty:
            raise ValueError("scores must not be empty")
        missing_columns = set(self.scores.columns) - set(prices.columns)
        if missing_columns:
            raise ValueError(f"scores contain unknown assets: {sorted(missing_columns)}")
        if not self.scores.index.isin(prices.index).all():
            raise ValueError("scores index must be a subset of the price index")
