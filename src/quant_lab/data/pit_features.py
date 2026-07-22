from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


DEFAULT_DATASET_LAG_DAYS: dict[str, int] = {
    "liquidity_market_cap_free_float": 1,
    "fundamentals_pit": 90,
    "ibes_estimates_revisions": 1,
    "earnings_events_actuals": 1,
}


@dataclass(frozen=True)
class PITLaggedFeatureArtifacts:
    run_id: str
    artifact_dir: Path
    features_parquet_path: Path
    features_csv_path: Path
    quality_path: Path
    manifest_path: Path
    features: pd.DataFrame
    quality: pd.DataFrame


def read_pit_history_partitions(root: str | Path) -> pd.DataFrame:
    """Read PIT history parquet files from a backfill artifact root."""

    path = Path(root)
    search_root = path / "history" if (path / "history").exists() else path
    files = sorted(search_root.rglob("*.parquet"))
    if not files:
        return pd.DataFrame()
    frames = [pd.read_parquet(file) for file in files]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def monthly_decision_dates(start_date: str, end_date: str, month_end: bool = True) -> pd.DatetimeIndex:
    freq = "ME" if month_end else "MS"
    return pd.date_range(start=pd.Timestamp(start_date), end=pd.Timestamp(end_date), freq=freq)


def build_lagged_pit_feature_panel(
    history: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp | str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lag_days_by_dataset: dict[str, int] | None = None,
    max_staleness_days_by_dataset: dict[str, int] | None = None,
    entity_col: str = "requested_ric",
) -> pd.DataFrame:
    """Build a decision-safe wide feature panel from PIT history rows."""

    if history.empty:
        return pd.DataFrame()
    required = {entity_col, "asof_date", "dataset_id", "field_role", "value"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"history missing required columns: {sorted(missing)}")

    data = history.copy()
    if "inside_requested_window" in data.columns:
        data = data[data["inside_requested_window"].fillna(True).astype(bool)].copy()
    if data.empty:
        return pd.DataFrame()

    lag_days_by_dataset = {**DEFAULT_DATASET_LAG_DAYS, **(lag_days_by_dataset or {})}
    max_staleness_days_by_dataset = max_staleness_days_by_dataset or {}
    data["asof_date"] = pd.to_datetime(data["asof_date"], errors="coerce")
    data = data.dropna(subset=[entity_col, "asof_date", "dataset_id", "field_role"]).copy()
    data["lag_days"] = data["dataset_id"].astype(str).map(lag_days_by_dataset).fillna(30).astype(int)
    data["source_available_date"] = data["asof_date"] + pd.to_timedelta(data["lag_days"], unit="D")

    if decision_dates is None:
        start = pd.Timestamp(start_date) if start_date else data["source_available_date"].min()
        end = pd.Timestamp(end_date) if end_date else data["source_available_date"].max()
        decision_dates = monthly_decision_dates(start.date().isoformat(), end.date().isoformat())
    decisions = _decision_grid(data, decision_dates=decision_dates, entity_col=entity_col)
    features = decisions.copy()

    meta = _latest_entity_metadata(data, entity_col=entity_col)
    if not meta.empty:
        features = features.merge(meta, on=entity_col, how="left")

    for (dataset_id, field_role), group in data.groupby(["dataset_id", "field_role"], dropna=False, sort=True):
        feature_name = _feature_name(dataset_id, field_role)
        joined = _merge_group_asof(
            decisions=decisions,
            events=group,
            entity_col=entity_col,
            feature_name=feature_name,
        )
        max_staleness = max_staleness_days_by_dataset.get(str(dataset_id))
        if max_staleness is not None and f"{feature_name}__source_asof_date" in joined.columns:
            stale_days = (
                pd.to_datetime(joined["decision_date"])
                - pd.to_datetime(joined[f"{feature_name}__source_asof_date"])
            ).dt.days
            stale = stale_days.gt(max_staleness)
            joined.loc[stale, [feature_name, f"{feature_name}__source_asof_date", f"{feature_name}__source_available_date"]] = pd.NA
        features = features.merge(
            joined[[entity_col, "decision_date", feature_name, f"{feature_name}__source_asof_date", f"{feature_name}__source_available_date"]],
            on=[entity_col, "decision_date"],
            how="left",
        )

    assert_lagged_feature_panel_no_leakage(features)
    return features.sort_values([entity_col, "decision_date"]).reset_index(drop=True)


def assert_lagged_feature_panel_no_leakage(features: pd.DataFrame, decision_col: str = "decision_date") -> None:
    if features.empty:
        return
    if decision_col not in features.columns:
        raise ValueError(f"missing decision column: {decision_col}")
    decisions = pd.to_datetime(features[decision_col], errors="coerce")
    source_cols = [col for col in features.columns if col.endswith("__source_available_date")]
    for col in source_cols:
        available = pd.to_datetime(features[col], errors="coerce")
        violations = available.notna() & available.gt(decisions)
        if violations.any():
            examples = features.loc[violations, [decision_col, col]].head(5).to_dict("records")
            raise ValueError(f"point-in-time leakage detected in {col}: {examples}")


def summarize_lagged_feature_panel(features: pd.DataFrame, entity_col: str = "requested_ric") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if features.empty:
        return pd.DataFrame(columns=["feature", "rows", "non_null_rows", "coverage", "first_source_asof", "last_source_asof"])
    total_rows = len(features)
    feature_cols = [
        col
        for col in features.columns
        if "__" in col and not col.endswith("__source_asof_date") and not col.endswith("__source_available_date")
    ]
    for feature in feature_cols:
        source_asof_col = f"{feature}__source_asof_date"
        values = features[feature]
        rows.append(
            {
                "feature": feature,
                "entities": int(features[entity_col].nunique()) if entity_col in features else 0,
                "rows": int(total_rows),
                "non_null_rows": int(values.notna().sum()),
                "coverage": float(values.notna().sum() / max(1, total_rows)),
                "first_source_asof": pd.to_datetime(features[source_asof_col], errors="coerce").min()
                if source_asof_col in features
                else pd.NaT,
                "last_source_asof": pd.to_datetime(features[source_asof_col], errors="coerce").max()
                if source_asof_col in features
                else pd.NaT,
            }
        )
    return pd.DataFrame(rows).sort_values("feature").reset_index(drop=True)


def write_lagged_pit_feature_artifacts(
    history: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    decision_dates: Sequence[pd.Timestamp | str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lag_days_by_dataset: dict[str, int] | None = None,
    max_staleness_days_by_dataset: dict[str, int] | None = None,
) -> PITLaggedFeatureArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    features = build_lagged_pit_feature_panel(
        history,
        decision_dates=decision_dates,
        start_date=start_date,
        end_date=end_date,
        lag_days_by_dataset=lag_days_by_dataset,
        max_staleness_days_by_dataset=max_staleness_days_by_dataset,
    )
    quality = summarize_lagged_feature_panel(features)

    features_parquet_path = artifact_dir / "lagged_features.parquet"
    features_csv_path = artifact_dir / "lagged_features.csv"
    quality_path = artifact_dir / "feature_quality.csv"
    manifest_path = artifact_dir / "manifest.json"
    features.to_parquet(features_parquet_path, index=False)
    features.to_csv(features_csv_path, index=False)
    quality.to_csv(quality_path, index=False)
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "features": quality["feature"].tolist() if "feature" in quality else [],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "lag_days_by_dataset": {**DEFAULT_DATASET_LAG_DAYS, **(lag_days_by_dataset or {})},
        "warning": "Features are lagged by conservative dataset policies; keep source audit columns for leakage checks.",
        "artifacts": {
            "lagged_features_parquet": str(features_parquet_path.resolve()),
            "lagged_features_csv": str(features_csv_path.resolve()),
            "feature_quality": str(quality_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PITLaggedFeatureArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        features_parquet_path=features_parquet_path,
        features_csv_path=features_csv_path,
        quality_path=quality_path,
        manifest_path=manifest_path,
        features=features,
        quality=quality,
    )


def _decision_grid(
    data: pd.DataFrame,
    decision_dates: Sequence[pd.Timestamp | str],
    entity_col: str,
) -> pd.DataFrame:
    entities = sorted(data[entity_col].dropna().astype(str).unique())
    dates = pd.to_datetime(pd.Index(decision_dates), errors="coerce").dropna().sort_values().unique()
    grid = pd.MultiIndex.from_product([entities, dates], names=[entity_col, "decision_date"]).to_frame(index=False)
    return grid


def _latest_entity_metadata(data: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    meta_cols = [
        entity_col,
        "refinitiv_ric",
        "sid",
        "ticker",
        "local_symbol",
        "name",
        "country",
        "exchange",
        "currency",
    ]
    available = [col for col in meta_cols if col in data.columns]
    if entity_col not in available:
        return pd.DataFrame()
    return (
        data.sort_values("asof_date")
        .dropna(subset=[entity_col])
        .drop_duplicates(entity_col, keep="last")[available]
        .reset_index(drop=True)
    )


def _merge_group_asof(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    entity_col: str,
    feature_name: str,
) -> pd.DataFrame:
    event_cols = [entity_col, "source_available_date", "asof_date", "value"]
    right = events[event_cols].copy()
    right[entity_col] = right[entity_col].astype(str)
    right = right.sort_values([entity_col, "source_available_date"]).reset_index(drop=True)
    parts = []
    for entity, left_group in decisions.groupby(entity_col, dropna=False, sort=False):
        right_group = right[right[entity_col].eq(str(entity))]
        if right_group.empty:
            out = left_group.copy()
            out[feature_name] = pd.NA
            out[f"{feature_name}__source_asof_date"] = pd.NaT
            out[f"{feature_name}__source_available_date"] = pd.NaT
            parts.append(out)
            continue
        joined = pd.merge_asof(
            left_group.sort_values("decision_date"),
            right_group.sort_values("source_available_date"),
            left_on="decision_date",
            right_on="source_available_date",
            direction="backward",
            suffixes=("", "_event"),
        )
        event_entity_col = f"{entity_col}_event"
        if event_entity_col in joined.columns:
            joined = joined.drop(columns=[event_entity_col])
        joined = joined.rename(
            columns={
                "value": feature_name,
                "asof_date": f"{feature_name}__source_asof_date",
                "source_available_date": f"{feature_name}__source_available_date",
            }
        )
        parts.append(joined)
    if not parts:
        return decisions.copy()
    return pd.concat(parts, ignore_index=True, sort=False)


def _feature_name(dataset_id: object, field_role: object) -> str:
    return f"{_slugish(dataset_id)}__{_slugish(field_role)}"


def _slugish(value: object) -> str:
    text = str(value).strip().lower()
    slug = "".join(ch if ch.isalnum() else "_" for ch in text)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"
