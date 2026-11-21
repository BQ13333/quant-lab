from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PIT_FEATURE_FAMILY_COLUMNS: dict[str, tuple[str, ...]] = {
    "liquidity": (
        "pit_liquidity__log_market_cap",
        "pit_liquidity__free_float_pct",
    ),
    "fundamentals": (
        "pit_fundamentals__book_to_price",
    ),
    "ibes": (
        "pit_ibes__eps_yield",
        "pit_ibes__eps_revision_63d",
        "pit_ibes__revenue_revision_63d",
        "pit_ibes__recommendation_score",
    ),
    "events": (
        "pit_events__earnings_recency_30d",
    ),
}


@dataclass(frozen=True)
class PITFeatureBundle:
    artifact_dir: Path
    manifest: dict
    panel: pd.DataFrame
    feature_quality: pd.DataFrame
    leakage_summary: pd.DataFrame


@dataclass(frozen=True)
class PITFactorMatrixBundle:
    matrices: dict[str, pd.DataFrame]
    family_columns: dict[str, tuple[str, ...]]
    feature_quality: pd.DataFrame
    source_manifest: dict


def load_pit_feature_bundle(artifact_dir: str | Path, require_pass: bool = True) -> PITFeatureBundle:
    """Load a leakage-QA-passed PIT feature artifact."""

    root = Path(artifact_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing PIT QA manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if require_pass and manifest.get("status") != "pass":
        raise ValueError(f"PIT QA artifact did not pass leakage gate: {manifest.get('status')}")
    if require_pass and int(manifest.get("total_leakage_violations", 0)) != 0:
        raise ValueError("PIT QA artifact has leakage violations")

    artifacts = manifest.get("artifacts", {})
    panel_path = Path(str(artifacts.get("decision_lagged_features", root / "decision_lagged_features.parquet")))
    quality_path = Path(str(artifacts.get("feature_quality", root / "feature_quality.csv")))
    leakage_path = Path(str(artifacts.get("leakage_summary", root / "leakage_summary.csv")))
    if not panel_path.exists():
        raise FileNotFoundError(f"missing PIT decision panel: {panel_path}")

    panel = pd.read_parquet(panel_path)
    feature_quality = pd.read_csv(quality_path, low_memory=False) if quality_path.exists() else pd.DataFrame()
    leakage_summary = pd.read_csv(leakage_path, low_memory=False) if leakage_path.exists() else pd.DataFrame()
    return PITFeatureBundle(
        artifact_dir=root,
        manifest=manifest,
        panel=panel,
        feature_quality=feature_quality,
        leakage_summary=leakage_summary,
    )


def build_default_pit_factor_matrices(
    panel: pd.DataFrame,
    prices: pd.DataFrame,
    revision_window: int = 63,
    event_window: int = 30,
) -> PITFactorMatrixBundle:
    """Create model-facing PIT factor matrices aligned to a price matrix."""

    required = {"requested_ric", "decision_date"}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"PIT panel missing required columns: {sorted(missing)}")
    if prices.empty:
        raise ValueError("prices must not be empty")

    aligned_index = prices.index
    aligned_columns = prices.columns.astype(str)
    price_matrix = prices.copy()
    price_matrix.columns = aligned_columns

    market_cap = _pivot_numeric(panel, "liquidity_market_cap_free_float__market_cap", aligned_index, aligned_columns)
    free_float = _pivot_numeric(panel, "liquidity_market_cap_free_float__free_float", aligned_index, aligned_columns)
    book_value = _pivot_numeric(panel, "fundamentals_pit__book_value", aligned_index, aligned_columns)
    eps_estimate = _pivot_numeric(panel, "ibes_estimates_revisions__eps_estimate", aligned_index, aligned_columns)
    revenue_estimate = _pivot_numeric(panel, "ibes_estimates_revisions__revenue_estimate", aligned_index, aligned_columns)
    recommendation = _pivot_numeric(panel, "ibes_estimates_revisions__recommendation", aligned_index, aligned_columns)
    earnings_release = _pivot_datetime(panel, "earnings_events_actuals__earnings_release", aligned_index, aligned_columns)

    matrices: dict[str, pd.DataFrame] = {
        "pit_liquidity__log_market_cap": np.log(market_cap.where(market_cap > 0.0)),
        "pit_liquidity__free_float_pct": free_float,
        "pit_fundamentals__book_to_price": book_value.div(price_matrix.replace(0.0, np.nan)),
        "pit_ibes__eps_yield": eps_estimate.div(price_matrix.replace(0.0, np.nan)),
        "pit_ibes__eps_revision_63d": _revision_proxy(eps_estimate, revision_window),
        "pit_ibes__revenue_revision_63d": _revision_proxy(revenue_estimate, revision_window),
        "pit_ibes__recommendation_score": -recommendation,
        "pit_events__earnings_recency_30d": _event_recency_score(earnings_release, event_window),
    }
    matrices = {name: _sanitize_matrix(matrix).reindex(index=aligned_index, columns=aligned_columns) for name, matrix in matrices.items()}
    quality = _factor_matrix_quality(matrices)
    source_manifest = {
        "factor_count": len(matrices),
        "families": {family: list(columns) for family, columns in PIT_FEATURE_FAMILY_COLUMNS.items()},
        "revision_window": revision_window,
        "event_window": event_window,
        "warning": "Derived factors are research-mode features built only from leakage-QA-passed PIT panel columns.",
    }
    return PITFactorMatrixBundle(
        matrices=matrices,
        family_columns=PIT_FEATURE_FAMILY_COLUMNS,
        feature_quality=quality,
        source_manifest=source_manifest,
    )


def select_factor_columns(
    families: Sequence[str],
    family_columns: dict[str, tuple[str, ...]] | None = None,
) -> tuple[str, ...]:
    """Expand family names into model-facing factor column names."""

    columns_by_family = family_columns or PIT_FEATURE_FAMILY_COLUMNS
    if not families:
        raise ValueError("at least one PIT feature family is required")
    selected: list[str] = []
    for family in families:
        key = family.strip().lower()
        if key == "all":
            for values in columns_by_family.values():
                selected.extend(values)
            continue
        if key not in columns_by_family:
            raise ValueError(f"unknown PIT feature family: {family}")
        selected.extend(columns_by_family[key])
    return tuple(dict.fromkeys(selected))


def _pivot_numeric(
    panel: pd.DataFrame,
    column: str,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    if column not in panel.columns:
        return pd.DataFrame(np.nan, index=index, columns=columns)
    data = panel[["decision_date", "requested_ric", column]].copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"], errors="coerce")
    data["requested_ric"] = data["requested_ric"].astype(str)
    data[column] = pd.to_numeric(data[column], errors="coerce")
    wide = data.pivot_table(index="decision_date", columns="requested_ric", values=column, aggfunc="last")
    return wide.reindex(index=index, columns=columns)


def _pivot_datetime(
    panel: pd.DataFrame,
    column: str,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    if column not in panel.columns:
        return pd.DataFrame(pd.NaT, index=index, columns=columns)
    data = panel[["decision_date", "requested_ric", column]].copy()
    data["decision_date"] = pd.to_datetime(data["decision_date"], errors="coerce")
    data["requested_ric"] = data["requested_ric"].astype(str)
    data[column] = pd.to_datetime(data[column], errors="coerce")
    wide = data.pivot_table(index="decision_date", columns="requested_ric", values=column, aggfunc="last")
    return wide.reindex(index=index, columns=columns)


def _revision_proxy(values: pd.DataFrame, window: int) -> pd.DataFrame:
    lagged = values.shift(window)
    denom = lagged.abs().replace(0.0, np.nan)
    return values.sub(lagged).div(denom)


def _event_recency_score(event_dates: pd.DataFrame, window: int) -> pd.DataFrame:
    scores = pd.DataFrame(np.nan, index=event_dates.index, columns=event_dates.columns)
    for column in event_dates.columns:
        event_series = pd.to_datetime(event_dates[column], errors="coerce")
        days = (event_dates.index.to_series(index=event_dates.index) - event_series).dt.days
        score = (window - days).clip(lower=0, upper=window).div(window)
        scores[column] = score.where(days.ge(0))
    return scores


def _sanitize_matrix(matrix: pd.DataFrame) -> pd.DataFrame:
    return matrix.replace([np.inf, -np.inf], np.nan)


def _factor_matrix_quality(matrices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, matrix in sorted(matrices.items()):
        total = int(matrix.size)
        non_null = int(matrix.notna().sum().sum())
        rows.append(
            {
                "factor": name,
                "rows": int(matrix.shape[0]),
                "columns": int(matrix.shape[1]),
                "non_null_cells": non_null,
                "coverage": float(non_null / max(1, total)),
                "first_non_null_date": matrix.notna().any(axis=1).idxmax() if matrix.notna().any(axis=1).any() else pd.NaT,
                "last_non_null_date": matrix.index[matrix.notna().any(axis=1)].max()
                if matrix.notna().any(axis=1).any()
                else pd.NaT,
            }
        )
    return pd.DataFrame(rows).sort_values("factor").reset_index(drop=True)
