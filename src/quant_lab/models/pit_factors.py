from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import pandas as pd

from quant_lab.contracts import SignalFrame
from quant_lab.models.base import AlphaModel
from quant_lab.models.baselines import _zscore_cross_section


@dataclass
class PITFactorModel(AlphaModel):
    """Cross-sectional alpha model built from leakage-QA-passed PIT factor matrices."""

    factor_matrices: Mapping[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    factor_weights: Mapping[str, float] = field(default_factory=dict)
    feature_set: str = "pit"
    source_artifact: str = ""

    def predict(self, prices: pd.DataFrame, start: int, end: int) -> SignalFrame:
        if not self.factor_weights:
            scores = pd.DataFrame(0.0, index=prices.iloc[start:end].index, columns=prices.columns)
            return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=self._metadata())

        aligned_scores = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
        total_abs_weight = 0.0
        for factor, weight in self.factor_weights.items():
            if factor not in self.factor_matrices:
                continue
            matrix = (
                self.factor_matrices[factor]
                .reindex(index=prices.index, columns=prices.columns)
                .astype(float)
            )
            aligned_scores = aligned_scores.add(_zscore_cross_section(matrix).mul(float(weight)), fill_value=0.0)
            total_abs_weight += abs(float(weight))

        if total_abs_weight > 0.0:
            aligned_scores = aligned_scores.div(total_abs_weight)
        scores = _zscore_cross_section(aligned_scores).iloc[start:end]
        return SignalFrame(scores=scores, model_name=self.name, horizon=self.horizon, metadata=self._metadata())

    def _metadata(self) -> dict[str, object]:
        return {
            "family": "pit_factor",
            "feature_set": self.feature_set,
            "pit_features": ",".join(self.factor_weights.keys()),
            "pit_weights": ",".join(f"{key}:{value:g}" for key, value in self.factor_weights.items()),
            "source_artifact": self.source_artifact,
            "readiness": "research_mode_leakage_qa_passed",
        }


def build_pit_factor_model_suite(
    factor_matrices: Mapping[str, pd.DataFrame],
    family_columns: Mapping[str, tuple[str, ...]],
    prefix: str = "pit",
    source_artifact: str = "",
) -> list[PITFactorModel]:
    """Build interpretable PIT family and composite models for ablation."""

    models: list[PITFactorModel] = []
    for family in ("liquidity", "fundamentals", "ibes", "events"):
        columns = tuple(col for col in family_columns.get(family, ()) if col in factor_matrices)
        if not columns:
            continue
        models.append(
            PITFactorModel(
                name=f"{prefix}_{family}",
                factor_matrices=factor_matrices,
                factor_weights={column: 1.0 for column in columns},
                feature_set=family,
                source_artifact=source_artifact,
            )
        )

    all_columns = tuple(
        col
        for family in ("fundamentals", "ibes", "events", "liquidity")
        for col in family_columns.get(family, ())
        if col in factor_matrices
    )
    if all_columns:
        models.append(
            PITFactorModel(
                name=f"{prefix}_all_equal",
                factor_matrices=factor_matrices,
                factor_weights={column: 1.0 for column in all_columns},
                feature_set="all_equal",
                source_artifact=source_artifact,
            )
        )

    quality_revision_columns = [
        column
        for column in (
            "pit_fundamentals__book_to_price",
            "pit_ibes__eps_yield",
            "pit_ibes__eps_revision_63d",
            "pit_ibes__revenue_revision_63d",
            "pit_ibes__recommendation_score",
        )
        if column in factor_matrices
    ]
    if quality_revision_columns:
        models.append(
            PITFactorModel(
                name=f"{prefix}_quality_revision",
                factor_matrices=factor_matrices,
                factor_weights={column: 1.0 for column in quality_revision_columns},
                feature_set="quality_revision",
                source_artifact=source_artifact,
            )
        )

    value_liquidity_columns = [
        column
        for column in (
            "pit_fundamentals__book_to_price",
            "pit_liquidity__log_market_cap",
            "pit_liquidity__free_float_pct",
        )
        if column in factor_matrices
    ]
    if value_liquidity_columns:
        models.append(
            PITFactorModel(
                name=f"{prefix}_value_liquidity",
                factor_matrices=factor_matrices,
                factor_weights={column: 1.0 for column in value_liquidity_columns},
                feature_set="value_liquidity",
                source_artifact=source_artifact,
            )
        )
    return models
