from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

from quant_lab.contracts import SignalFrame
from quant_lab.models.base import AlphaModel
from quant_lab.regimes import VolTrendRegimeDetector


@dataclass
class EqualWeightEnsembleModel(AlphaModel):
    """Average z-scored signals from a list of models with equal weight."""

    models: list[AlphaModel] = field(default_factory=list)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if not self.models:
            empty = pd.DataFrame(
                0.0, index=prices.iloc[start:end].index, columns=prices.columns
            )
            return SignalFrame(
                scores=empty,
                model_name=self.name,
                horizon=self.horizon,
                metadata={"family": "equal_weight_ensemble", "models": []},
            )
        signals = []
        for model in self.models:
            signal = model.predict(prices, start, end)
            signal.validate_against(prices)
            if not signal.scores.index.equals(prices.iloc[start:end].index) or not signal.scores.columns.equals(prices.columns):
                raise ValueError("ensemble signals must align exactly")
            signals.append(signal.scores)
        combined = sum(signals) / len(signals)
        if not np.isfinite(combined.to_numpy(dtype=float)).all():
            raise ValueError("combined ensemble scores must be finite")
        return SignalFrame(
            scores=combined,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "equal_weight_ensemble",
                "models": [m.name for m in self.models],
            },
        )


@dataclass
class WeightedEnsembleModel(AlphaModel):
    """Linear combination of signals with finite signed, unnormalized coefficients.

    Empty models and coefficients explicitly represent cash/no signal.
    """

    models: list[AlphaModel] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if len(self.models) != len(self.weights) or not np.isfinite(self.weights).all():
            raise ValueError("weights must be finite and match models exactly")
        if not self.models:
            empty = pd.DataFrame(
                0.0, index=prices.iloc[start:end].index, columns=prices.columns
            )
            return SignalFrame(
                scores=empty,
                model_name=self.name,
                horizon=self.horizon,
                metadata={"family": "weighted_ensemble", "models": []},
            )
        if len(self.models) != len(self.weights):
            raise ValueError(
                f"models ({len(self.models)}) and weights ({len(self.weights)}) must have the same length"
            )
        signals = []
        for model in self.models:
            signal = model.predict(prices, start, end)
            signal.validate_against(prices)
            if not signal.scores.index.equals(prices.iloc[start:end].index) or not signal.scores.columns.equals(prices.columns):
                raise ValueError("ensemble signals must align exactly")
            signals.append(signal.scores)
        combined = pd.DataFrame(0.0, index=signals[0].index, columns=signals[0].columns)
        for s, w in zip(signals, self.weights):
            combined += s * w
        if not np.isfinite(combined.to_numpy(dtype=float)).all():
            raise ValueError("combined ensemble scores must be finite")
        return SignalFrame(
            scores=combined,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "weighted_ensemble",
                "models": [m.name for m in self.models],
                "weights": list(self.weights),
            },
        )


@dataclass
class RegimeWeightedEnsembleModel(AlphaModel):
    """Switch model based on the current regime detected by VolTrendRegimeDetector."""

    models: dict[str, AlphaModel] = field(default_factory=dict)
    detector: VolTrendRegimeDetector = field(default_factory=VolTrendRegimeDetector)

    def _match_regime(self, label: str, pattern: str) -> bool:
        return pattern in label

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        regimes = self.detector.describe(prices).iloc[start:end]
        # Pre-compute all model signals
        model_signals: dict[str, pd.DataFrame] = {}
        for label, model in self.models.items():
            model_signals[label] = model.predict(prices, start, end).scores

        # Build combined signal row by row based on regime
        combined = pd.DataFrame(
            0.0, index=regimes.index, columns=prices.columns
        )
        for idx in regimes.index:
            regime_label = regimes.loc[idx]
            if pd.isna(regime_label):
                continue
            matched = False
            for pattern, scores in model_signals.items():
                if self._match_regime(regime_label, pattern):
                    combined.loc[idx] = scores.loc[idx]
                    matched = True
                    break
            if not matched:
                combined.loc[idx] = 0.0
        if not np.isfinite(combined.to_numpy(dtype=float)).all():
            raise ValueError("combined ensemble scores must be finite")
        return SignalFrame(
            scores=combined,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "regime_weighted_ensemble",
                "regimes": list(self.models.keys()),
            },
        )
