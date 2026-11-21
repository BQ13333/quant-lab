from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

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
        signals = [m.predict(prices, start, end).scores for m in self.models]
        combined = sum(signals) / len(signals)
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
    """Weighted average of z-scored signals from a list of models."""

    models: list[AlphaModel] = field(default_factory=list)
    weights: list[float] = field(default_factory=list)

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if not self.models or not self.weights:
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
        signals = [m.predict(prices, start, end).scores for m in self.models]
        combined = pd.DataFrame(0.0, index=signals[0].index, columns=signals[0].columns)
        for s, w in zip(signals, self.weights):
            combined += s * w
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
        model_signals: dict[str, pd.DataFrame] = {}
        for label, model in self.models.items():
            model_signals[label] = model.predict(prices, start, end).scores

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
        return SignalFrame(
            scores=combined,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "regime_weighted_ensemble",
                "regimes": list(self.models.keys()),
            },
        )
