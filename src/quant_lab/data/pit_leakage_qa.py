from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from quant_lab.data.pit_features import (
    DEFAULT_DATASET_LAG_DAYS,
    assert_lagged_feature_panel_no_leakage,
    build_lagged_pit_feature_panel,
    summarize_lagged_feature_panel,
)


@dataclass(frozen=True)
class PITLeakageQAArtifacts:
    run_id: str
    artifact_dir: Path
    decision_panel_path: Path
    feature_quality_path: Path
    leakage_summary_path: Path
    leakage_violations_path: Path
    decision_calendar_summary_path: Path
    summary_path: Path
    manifest_path: Path
    decision_panel: pd.DataFrame
    feature_quality: pd.DataFrame
    leakage_summary: pd.DataFrame
    leakage_violations: pd.DataFrame


def load_l2_backtest_decision_calendar(
    l2_artifact_dir: str | Path,
    universe_tier: str = "u2",
    include_countries: Sequence[str] | None = ("Malaysia", "Singapore", "Indonesia"),
    price_col: str = "adj_close",
    min_rows: int | None = None,
    max_missing_ratio: float | None = None,
    ffill_limit: int = 5,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Rebuild the price-matrix decision calendar used by the L2 backtest loop.

    The returned rows are one security per model decision date, filtered through
    the L2 tier file and the same broad price coverage gates used by the
    cross-market backtest loader.
    """

    root = Path(l2_artifact_dir)
    manifest_path = root / "manifest.json"
    tiers_path = root / "eligible_universe_tiers.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing L2 manifest: {manifest_path}")
    if not tiers_path.exists():
        raise FileNotFoundError(f"missing L2 universe tiers: {tiers_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    long_ohlcv_root = Path(str(manifest["long_ohlcv_root"]))
    min_rows = int(min_rows if min_rows is not None else manifest.get("min_rows", 2500))
    max_missing_ratio = float(max_missing_ratio if max_missing_ratio is not None else manifest.get("max_missing_ratio", 0.25))

    tiers = pd.read_csv(tiers_path, low_memory=False)
    selected = _select_tier_universe(tiers, universe_tier=universe_tier, include_countries=include_countries)
    selected_tickers = set(selected["ticker"].astype(str))
    if not selected_tickers:
        raise ValueError(f"no tickers selected for universe_tier={universe_tier}")

    columns = [
        "date",
        "ticker",
        "sid",
        "local_symbol",
        "name",
        "country",
        "exchange",
        "currency",
        price_col,
    ]
    long_ohlcv = pd.read_parquet(long_ohlcv_root, columns=columns)
    long_ohlcv = long_ohlcv[long_ohlcv["ticker"].astype(str).isin(selected_tickers)].copy()
    if long_ohlcv.empty:
        raise ValueError("no OHLCV rows after L2 tier filter")

    long_ohlcv["date"] = pd.to_datetime(long_ohlcv["date"], errors="coerce")
    long_ohlcv[price_col] = pd.to_numeric(long_ohlcv[price_col], errors="coerce")
    long_ohlcv = long_ohlcv.dropna(subset=["date", "ticker", price_col]).copy()
    if long_ohlcv.empty:
        raise ValueError("no priced OHLCV rows after date/ticker/price cleanup")

    prices = (
        long_ohlcv.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
        .astype(float)
    )
    prices = prices.ffill(limit=ffill_limit)
    eligible = prices.notna().sum().ge(min_rows) & prices.isna().mean().le(max_missing_ratio)
    prices = prices.loc[:, eligible].dropna(how="all")
    if prices.empty:
        raise ValueError("no eligible price columns after coverage filters")

    if start_date:
        prices = prices.loc[prices.index >= pd.Timestamp(start_date)]
    if end_date:
        prices = prices.loc[prices.index <= pd.Timestamp(end_date)]
    if prices.empty:
        raise ValueError("no decision dates after requested date filter")

    metadata = _latest_metadata(long_ohlcv)
    metadata = metadata[metadata["ticker"].astype(str).isin(prices.columns.astype(str))].copy()
    metadata = selected.merge(metadata, on="ticker", how="left", suffixes=("_tier", ""))
    metadata = metadata[metadata["ticker"].astype(str).isin(prices.columns.astype(str))].copy()
    metadata = metadata.drop_duplicates("ticker", keep="last")

    grid = pd.MultiIndex.from_product(
        [sorted(prices.columns.astype(str)), prices.index],
        names=["requested_ric", "decision_date"],
    ).to_frame(index=False)
    decision_meta = metadata.rename(
        columns={
            "ticker": "requested_ric",
            "sid": "decision_sid",
            "local_symbol": "decision_local_symbol",
            "name": "decision_name",
            "country": "decision_country",
            "exchange": "decision_exchange",
            "currency": "decision_currency",
        }
    )
    keep_meta = [
        "requested_ric",
        "decision_sid",
        "decision_local_symbol",
        "decision_name",
        "decision_country",
        "decision_exchange",
        "decision_currency",
    ]
    keep_meta = [col for col in keep_meta if col in decision_meta.columns]
    decisions = grid.merge(decision_meta[keep_meta], on="requested_ric", how="left")
    decisions["universe_tier"] = universe_tier
    return decisions.sort_values(["requested_ric", "decision_date"]).reset_index(drop=True)


def build_decision_lagged_pit_panel(
    history: pd.DataFrame,
    decisions: pd.DataFrame,
    lag_days_by_dataset: dict[str, int] | None = None,
    max_staleness_days_by_dataset: dict[str, int] | None = None,
    entity_col: str = "requested_ric",
    decision_col: str = "decision_date",
) -> pd.DataFrame:
    """Build lagged PIT features on an externally supplied decision calendar."""

    if history.empty:
        raise ValueError("history is empty")
    if decisions.empty:
        raise ValueError("decisions is empty")
    required = {entity_col, decision_col}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"decisions missing required columns: {sorted(missing)}")

    calendar = decisions.copy()
    calendar[entity_col] = calendar[entity_col].astype(str)
    calendar[decision_col] = pd.to_datetime(calendar[decision_col], errors="coerce")
    calendar = calendar.dropna(subset=[entity_col, decision_col]).drop_duplicates([entity_col, decision_col]).copy()
    if calendar.empty:
        raise ValueError("decision calendar is empty after cleanup")

    feature_dates = sorted(calendar[decision_col].unique())
    features = build_lagged_pit_feature_panel(
        history,
        decision_dates=feature_dates,
        lag_days_by_dataset=lag_days_by_dataset,
        max_staleness_days_by_dataset=max_staleness_days_by_dataset,
        entity_col=entity_col,
    )
    if features.empty:
        return calendar

    panel = calendar.merge(features, on=[entity_col, decision_col], how="left", suffixes=("_decision", ""))
    assert_lagged_feature_panel_no_leakage(panel, decision_col=decision_col)
    return panel.sort_values([entity_col, decision_col]).reset_index(drop=True)


def summarize_pit_leakage(panel: pd.DataFrame, decision_col: str = "decision_date") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return leakage summary rows and the violating decision-feature rows."""

    violation_columns = [
        "feature",
        "requested_ric",
        decision_col,
        "decision_country",
        "feature_value",
        "source_asof_date",
        "source_available_date",
    ]
    if panel.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "feature",
                "rows",
                "feature_non_null_rows",
                "feature_coverage",
                "source_available_non_null_rows",
                "leakage_violations",
                "min_available_lag_days",
                "median_available_lag_days",
                "max_available_lag_days",
                "min_asof_staleness_days",
                "median_asof_staleness_days",
                "max_asof_staleness_days",
                "first_source_asof",
                "last_source_asof",
                "status",
            ]
        )
        return empty_summary, pd.DataFrame(columns=violation_columns)

    decisions = pd.to_datetime(panel[decision_col], errors="coerce")
    summary_rows: list[dict[str, Any]] = []
    violation_frames: list[pd.DataFrame] = []
    source_cols = [col for col in panel.columns if col.endswith("__source_available_date")]
    for source_col in sorted(source_cols):
        feature = source_col.removesuffix("__source_available_date")
        asof_col = f"{feature}__source_asof_date"
        value_non_null = panel[feature].notna() if feature in panel.columns else pd.Series(False, index=panel.index)
        available = pd.to_datetime(panel[source_col], errors="coerce")
        asof = pd.to_datetime(panel[asof_col], errors="coerce") if asof_col in panel.columns else pd.Series(pd.NaT, index=panel.index)
        available_non_null = available.notna()
        violations = available_non_null & available.gt(decisions)
        available_lag_days = (decisions[available_non_null] - available[available_non_null]).dt.days
        asof_staleness_days = (decisions[asof.notna()] - asof[asof.notna()]).dt.days
        summary_rows.append(
            {
                "feature": feature,
                "rows": int(len(panel)),
                "feature_non_null_rows": int(value_non_null.sum()),
                "feature_coverage": float(value_non_null.sum() / max(1, len(panel))),
                "source_available_non_null_rows": int(available_non_null.sum()),
                "leakage_violations": int(violations.sum()),
                "min_available_lag_days": _safe_stat(available_lag_days, "min"),
                "median_available_lag_days": _safe_stat(available_lag_days, "median"),
                "max_available_lag_days": _safe_stat(available_lag_days, "max"),
                "min_asof_staleness_days": _safe_stat(asof_staleness_days, "min"),
                "median_asof_staleness_days": _safe_stat(asof_staleness_days, "median"),
                "max_asof_staleness_days": _safe_stat(asof_staleness_days, "max"),
                "first_source_asof": asof.min(),
                "last_source_asof": asof.max(),
                "status": "fail" if violations.any() else "pass",
            }
        )
        if violations.any():
            bad = pd.DataFrame(
                {
                    "feature": feature,
                    "requested_ric": panel.loc[violations, "requested_ric"]
                    if "requested_ric" in panel.columns
                    else pd.NA,
                    decision_col: panel.loc[violations, decision_col],
                    "decision_country": panel.loc[violations, "decision_country"]
                    if "decision_country" in panel.columns
                    else pd.NA,
                    "feature_value": panel.loc[violations, feature] if feature in panel.columns else pd.NA,
                    "source_asof_date": asof.loc[violations],
                    "source_available_date": available.loc[violations],
                }
            )
            violation_frames.append(bad)

    summary = pd.DataFrame(summary_rows).sort_values("feature").reset_index(drop=True)
    violations_df = (
        pd.concat(violation_frames, ignore_index=True, sort=False)[violation_columns]
        if violation_frames
        else pd.DataFrame(columns=violation_columns)
    )
    return summary, violations_df


def write_pit_leakage_qa_artifacts(
    history: pd.DataFrame,
    decisions: pd.DataFrame,
    artifact_root: str | Path,
    run_id: str,
    lag_days_by_dataset: dict[str, int] | None = None,
    max_staleness_days_by_dataset: dict[str, int] | None = None,
    entity_col: str = "requested_ric",
    decision_col: str = "decision_date",
) -> PITLeakageQAArtifacts:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    panel = build_decision_lagged_pit_panel(
        history,
        decisions,
        lag_days_by_dataset=lag_days_by_dataset,
        max_staleness_days_by_dataset=max_staleness_days_by_dataset,
        entity_col=entity_col,
        decision_col=decision_col,
    )
    feature_quality = summarize_lagged_feature_panel(panel, entity_col=entity_col)
    leakage_summary, leakage_violations = summarize_pit_leakage(panel, decision_col=decision_col)
    calendar_summary = _decision_calendar_summary(decisions, panel, entity_col=entity_col, decision_col=decision_col)

    decision_panel_path = artifact_dir / "decision_lagged_features.parquet"
    feature_quality_path = artifact_dir / "feature_quality.csv"
    leakage_summary_path = artifact_dir / "leakage_summary.csv"
    leakage_violations_path = artifact_dir / "leakage_violations.csv"
    decision_calendar_summary_path = artifact_dir / "decision_calendar_summary.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"

    panel.to_parquet(decision_panel_path, index=False)
    feature_quality.to_csv(feature_quality_path, index=False)
    leakage_summary.to_csv(leakage_summary_path, index=False)
    leakage_violations.to_csv(leakage_violations_path, index=False)
    calendar_summary.to_csv(decision_calendar_summary_path, index=False)

    total_violations = int(leakage_summary["leakage_violations"].sum()) if not leakage_summary.empty else 0
    summary_path.write_text(
        _summary_md(run_id, panel, feature_quality, leakage_summary, calendar_summary, total_violations),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "scope": "pit_leakage_qa_against_backtest_calendar",
        "decision_rows": int(len(panel)),
        "decision_entities": int(panel[entity_col].nunique()) if entity_col in panel else 0,
        "decision_start": str(pd.to_datetime(panel[decision_col], errors="coerce").min()) if decision_col in panel else "",
        "decision_end": str(pd.to_datetime(panel[decision_col], errors="coerce").max()) if decision_col in panel else "",
        "features": feature_quality["feature"].tolist() if "feature" in feature_quality else [],
        "total_leakage_violations": total_violations,
        "status": "pass" if total_violations == 0 else "fail",
        "lag_days_by_dataset": {**DEFAULT_DATASET_LAG_DAYS, **(lag_days_by_dataset or {})},
        "max_staleness_days_by_dataset": max_staleness_days_by_dataset or {},
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "decision_lagged_features": str(decision_panel_path.resolve()),
            "feature_quality": str(feature_quality_path.resolve()),
            "leakage_summary": str(leakage_summary_path.resolve()),
            "leakage_violations": str(leakage_violations_path.resolve()),
            "decision_calendar_summary": str(decision_calendar_summary_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return PITLeakageQAArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        decision_panel_path=decision_panel_path,
        feature_quality_path=feature_quality_path,
        leakage_summary_path=leakage_summary_path,
        leakage_violations_path=leakage_violations_path,
        decision_calendar_summary_path=decision_calendar_summary_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        decision_panel=panel,
        feature_quality=feature_quality,
        leakage_summary=leakage_summary,
        leakage_violations=leakage_violations,
    )


def _select_tier_universe(
    tiers: pd.DataFrame,
    universe_tier: str,
    include_countries: Sequence[str] | None,
) -> pd.DataFrame:
    if "ticker" not in tiers.columns:
        raise ValueError("L2 tier file missing ticker column")
    selected = tiers.copy()
    if include_countries:
        if "country" not in selected.columns:
            raise ValueError("L2 tier file missing country column")
        countries = {country.strip().lower() for country in include_countries}
        selected = selected[selected["country"].astype(str).str.strip().str.lower().isin(countries)].copy()

    tier = universe_tier.strip().lower()
    if tier != "all":
        tier_col = {"u2": "u2_backtest_eligible", "u3": "u3_liquid_proxy"}.get(tier, tier)
        if tier_col not in selected.columns:
            raise ValueError(f"L2 tier file missing universe tier column: {tier_col}")
        selected = selected[selected[tier_col].fillna(False).astype(bool)].copy()

    keep = [
        "ticker",
        "sid",
        "local_symbol",
        "name",
        "country",
        "exchange",
        "currency",
    ]
    keep = [col for col in keep if col in selected.columns]
    return selected[keep].drop_duplicates("ticker").reset_index(drop=True)


def _latest_metadata(long_ohlcv: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ticker",
        "sid",
        "local_symbol",
        "name",
        "country",
        "exchange",
        "currency",
        "date",
    ]
    available = [col for col in cols if col in long_ohlcv.columns]
    metadata = long_ohlcv[available].sort_values("date").drop_duplicates("ticker", keep="last")
    return metadata.drop(columns=["date"], errors="ignore").reset_index(drop=True)


def _decision_calendar_summary(
    decisions: pd.DataFrame,
    panel: pd.DataFrame,
    entity_col: str,
    decision_col: str,
) -> pd.DataFrame:
    decision_dates = pd.to_datetime(decisions[decision_col], errors="coerce")
    panel_dates = pd.to_datetime(panel[decision_col], errors="coerce") if decision_col in panel else pd.Series(dtype="datetime64[ns]")
    return pd.DataFrame(
        [
            {
                "input_decision_rows": int(len(decisions)),
                "input_decision_entities": int(decisions[entity_col].nunique()) if entity_col in decisions else 0,
                "input_first_decision_date": decision_dates.min(),
                "input_last_decision_date": decision_dates.max(),
                "panel_rows": int(len(panel)),
                "panel_entities": int(panel[entity_col].nunique()) if entity_col in panel else 0,
                "panel_first_decision_date": panel_dates.min(),
                "panel_last_decision_date": panel_dates.max(),
            }
        ]
    )


def _safe_stat(values: pd.Series, name: str) -> float | None:
    cleaned = pd.to_numeric(values, errors="coerce").dropna()
    if cleaned.empty:
        return None
    if name == "min":
        return float(cleaned.min())
    if name == "median":
        return float(cleaned.median())
    if name == "max":
        return float(cleaned.max())
    raise ValueError(f"unsupported stat: {name}")


def _summary_md(
    run_id: str,
    panel: pd.DataFrame,
    feature_quality: pd.DataFrame,
    leakage_summary: pd.DataFrame,
    calendar_summary: pd.DataFrame,
    total_violations: int,
) -> str:
    status = "PASS" if total_violations == 0 else "FAIL"
    entity_count = int(panel["requested_ric"].nunique()) if "requested_ric" in panel else 0
    decision_dates = pd.to_datetime(panel["decision_date"], errors="coerce") if "decision_date" in panel else pd.Series(dtype="datetime64[ns]")
    feature_count = int(len(feature_quality)) if not feature_quality.empty else 0
    min_coverage = float(feature_quality["coverage"].min()) if "coverage" in feature_quality and not feature_quality.empty else 0.0
    lines = [
        "# PIT Leakage QA Report",
        "",
        f"Run ID: `{run_id}`",
        f"Leakage status: `{status}`",
        f"Total leakage violations: `{total_violations}`",
        "",
        "## Decision Calendar",
        "",
        f"- Decision-security rows: `{len(panel):,}`",
        f"- Securities: `{entity_count:,}`",
        f"- Decision range: `{decision_dates.min()}` to `{decision_dates.max()}`",
        "",
        "## Feature Readiness",
        "",
        f"- Feature columns checked: `{feature_count:,}`",
        f"- Minimum feature coverage: `{min_coverage:.2%}`",
        "- Audit basis: source available dates must be on or before each model decision date.",
        "- Features still require separate source-semantic approval before production alpha use.",
        "",
        "## Calendar Summary",
        "",
        _markdown_table(calendar_summary),
        "",
        "## Leakage Summary",
        "",
        _markdown_table(leakage_summary),
        "",
    ]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).fillna("")
    columns = list(clipped.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |"
        for _, row in clipped.iterrows()
    ]
    return "\n".join([header, separator, *rows])
