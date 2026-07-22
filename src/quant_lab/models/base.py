from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from quant_lab.contracts import SignalFrame


@dataclass
class AlphaModel(ABC):
    """Base interface for all alpha models in the tournament."""

    name: str
    horizon: str = "1d"

    def fit(self, prices: pd.DataFrame, start: int, end: int) -> "AlphaModel":
        return self

    @abstractmethod
    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        """Return model scores for rows [start, end)."""
