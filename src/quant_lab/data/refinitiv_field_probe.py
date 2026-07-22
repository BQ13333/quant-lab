from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


REFINITIV_PRICE_FIELD_CANDIDATES: dict[str, list[str]] = {
    "Open": ["OPEN_PRC", "TR.PriceOpen", "TR.OPENPRICE"],
    "High": ["HIGH_1", "TR.PriceHigh", "TR.HIGHPRICE"],
    "Low": ["LOW_1", "TR.PriceLow", "TR.LOWPRICE"],
    "Close": ["TRDPRC_1", "TR.PriceClose", "TR.CLOSEPRICE", "CLOSE_PRC", "CF_CLOSE", "HST_CLOSE"],
    "Volume": ["ACVOL_UNS", "TR.AccumulatedVolume", "TR.Volume", "CF_VOLUME"],
}


@dataclass(frozen=True)
class RefinitivFieldProbeResult:
    artifact_dir: Path
    sample_path: Path
    results_path: Path
    summary_path: Path
    recommended_path: Path
    field_catalog_path: Path
    validation_path: Path
    manifest_path: Path
    results: pd.DataFrame
    recommended: pd.DataFrame


def default_refinitiv_probe_samples() -> pd.DataFrame:
    """Representative liquid RICs for first-pass field validation."""

    return pd.DataFrame(
        [
            {"country": "United States", "exchange": "NYSE", "ticker": "IBM.N", "refinitiv_ric": "IBM.N", "name": "IBM"},
            {"country": "United States", "exchange": "NASDAQ", "ticker": "MSFT.O", "refinitiv_ric": "MSFT.O", "name": "Microsoft"},
            {"country": "Japan", "exchange": "JPX", "ticker": "7203.T", "refinitiv_ric": "7203.T", "name": "Toyota Motor"},
            {"country": "Hong Kong", "exchange": "HKEX", "ticker": "0700.HK", "refinitiv_ric": "0700.HK", "name": "Tencent"},
            {"country": "South Korea", "exchange": "KRX", "ticker": "005930.KS", "refinitiv_ric": "005930.KS", "name": "Samsung Electronics"},
            {"country": "Singapore", "exchange": "SGX", "ticker": "DBS", "refinitiv_ric": "DBSM.SI", "name": "DBS Group"},
            {"country": "Malaysia", "exchange": "Bursa Malaysia", "ticker": "MAYBANK", "refinitiv_ric": "MBBM.KL", "name": "Maybank"},
            {"country": "Indonesia", "exchange": "IDX", "ticker": "BBRI", "refinitiv_ric": "BBRI.JK", "name": "Bank Rakyat Indonesia"},
        ]
    )


def sample_refinitiv_universe(
    listings: pd.DataFrame | None = None,
    sample_per_market: int = 1,
    ric_col: str = "refinitiv_ric",
    max_samples: int | None = None,
) -> pd.DataFrame:
    defaults = default_refinitiv_probe_samples()
    if listings is None or listings.empty or ric_col not in listings.columns:
        return defaults.head(max_samples).copy() if max_samples else defaults

    data = listings[listings[ric_col].notna()].copy()
    group_cols = [col for col in ("country", "exchange") if col in data.columns]
    if not group_cols:
        sampled = data.head(sample_per_market).copy()
    else:
        sampled = (
            data.sort_values(group_cols + ["ticker"])
            .groupby(group_cols, dropna=False, group_keys=False)
            .head(sample_per_market)
            .copy()
        )
    combined = pd.concat([defaults, sampled], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(ric_col, keep="first").reset_index(drop=True)
    if max_samples is None or max_samples <= 0 or len(combined) <= max_samples:
        return combined

    default_rics = set(defaults[ric_col].dropna().astype(str))
    is_default = combined[ric_col].astype(str).isin(default_rics)
    core = combined[is_default].head(max_samples)
    remaining_slots = max(0, max_samples - len(core))
    extra = combined[~is_default].head(remaining_slots)
    return pd.concat([core, extra], ignore_index=True, sort=False)


def _non_null_stats(history: pd.DataFrame) -> dict[str, Any]:
    if history.empty:
        return {
            "rows": 0,
            "columns": 0,
            "non_null_values": 0,
            "coverage": 0.0,
            "first_available": pd.NaT,
            "last_available": pd.NaT,
            "last_value": pd.NA,
            "returned_columns": "",
        }
    non_null = history.notna()
    any_row = non_null.any(axis=1)
    values = pd.Series(history.to_numpy().ravel()).dropna()
    return {
        "rows": int(history.shape[0]),
        "columns": int(history.shape[1]),
        "non_null_values": int(non_null.sum().sum()),
        "coverage": float(non_null.sum().sum() / max(1, history.shape[0] * history.shape[1])),
        "first_available": history.index[any_row].min() if any_row.any() else pd.NaT,
        "last_available": history.index[any_row].max() if any_row.any() else pd.NaT,
        "last_value": values.iloc[-1] if not values.empty else pd.NA,
        "returned_columns": "|".join(str(col) for col in history.columns),
    }


def _count_errors(values: pd.Series) -> int:
    return int(values.fillna("").astype(str).str.strip().ne("").sum())


def probe_refinitiv_fields(
    ld: Any,
    samples: pd.DataFrame,
    fields_by_role: dict[str, list[str]] | None = None,
    interval: str = "1D",
    count: int = 60,
    sleep_seconds: float = 0.05,
    progress_every: int = 0,
) -> pd.DataFrame:
    fields_by_role = fields_by_role or REFINITIV_PRICE_FIELD_CANDIDATES
    rows: list[dict[str, Any]] = []
    total_calls = len(samples) * sum(len(fields) for fields in fields_by_role.values())
    completed_calls = 0
    for sample in samples.itertuples(index=False):
        sample_dict = sample._asdict()
        ric = sample_dict.get("refinitiv_ric") or sample_dict.get("ticker")
        if pd.isna(ric) or not str(ric).strip():
            continue
        for role, fields in fields_by_role.items():
            for field in fields:
                row = {
                    "country": sample_dict.get("country"),
                    "exchange": sample_dict.get("exchange"),
                    "ticker": sample_dict.get("ticker"),
                    "name": sample_dict.get("name"),
                    "refinitiv_ric": str(ric),
                    "role": role,
                    "field": field,
                    "interval": interval,
                    "count": count,
                    "error": "",
                }
                try:
                    history = ld.get_history(universe=[str(ric)], fields=[field], interval=interval, count=count)
                    row.update(_non_null_stats(history))
                except Exception as exc:  # pragma: no cover - depends on live vendor state
                    row.update(_non_null_stats(pd.DataFrame()))
                    row["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(row)
                completed_calls += 1
                if progress_every > 0 and (completed_calls % progress_every == 0 or completed_calls == total_calls):
                    print(
                        f"Probed {completed_calls}/{total_calls}: {ric} {role} {field}",
                        file=sys.stderr,
                        flush=True,
                    )
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)
    return pd.DataFrame(rows)


def summarize_refinitiv_field_probe(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    grouped = (
        results.groupby(["country", "exchange", "role", "field"], dropna=False)
        .agg(
            tested_rics=("refinitiv_ric", "nunique"),
            successful_rics=("non_null_values", lambda values: int((values > 0).sum())),
            avg_coverage=("coverage", "mean"),
            max_coverage=("coverage", "max"),
            total_non_null=("non_null_values", "sum"),
            errors=("error", _count_errors),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["country", "exchange", "role", "successful_rics", "avg_coverage", "total_non_null"],
        ascending=[True, True, True, False, False, False],
    ).reset_index(drop=True)


def recommend_refinitiv_field_map(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = []
    for (country, exchange, role), group in summary.groupby(["country", "exchange", "role"], dropna=False):
        ranked = group.sort_values(
            ["successful_rics", "avg_coverage", "total_non_null", "errors"],
            ascending=[False, False, False, True],
        )
        best = ranked.iloc[0].to_dict()
        best["status"] = "recommended" if best["successful_rics"] > 0 and best["avg_coverage"] > 0 else "no_usable_field"
        rows.append(best)
    return pd.DataFrame(rows).sort_values(["country", "exchange", "role"]).reset_index(drop=True)


def build_refinitiv_ric_validation_manifest(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize whether each candidate RIC returned any historical price data."""

    if results.empty:
        return pd.DataFrame()
    data = results.copy()
    for col in ("first_available", "last_available"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], errors="coerce")
    sample_cols = ["country", "exchange", "ticker", "name", "refinitiv_ric", "interval", "count"]
    existing_sample_cols = [col for col in sample_cols if col in data.columns]
    manifest = (
        data.groupby(existing_sample_cols, dropna=False)
        .agg(
            tested_fields=("field", "nunique"),
            successful_fields=("non_null_values", lambda values: int((values > 0).sum())),
            total_non_null_values=("non_null_values", "sum"),
            max_coverage=("coverage", "max"),
            avg_coverage=("coverage", "mean"),
            errors=("error", _count_errors),
            first_available=("first_available", "min"),
            last_available=("last_available", "max"),
        )
        .reset_index()
    )
    manifest["status"] = manifest["successful_fields"].map(
        lambda value: "validated_price_history" if value > 0 else "no_history_for_candidate_ric"
    )
    return manifest.sort_values(
        ["status", "country", "exchange", "refinitiv_ric"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def build_refinitiv_field_catalog(recommended: pd.DataFrame) -> pd.DataFrame:
    """Convert recommended field probe output into a compact market field catalog."""

    if recommended.empty:
        return pd.DataFrame()
    catalog = recommended[recommended["status"].eq("recommended")].copy()
    if catalog.empty:
        return pd.DataFrame()
    catalog["data_family"] = "price_history"
    catalog["field_role"] = catalog["role"]
    catalog["market_key"] = catalog["country"].fillna("") + "|" + catalog["exchange"].fillna("")
    catalog["validation_status"] = "live_probe_recommended"
    keep_cols = [
        "market_key",
        "country",
        "exchange",
        "data_family",
        "field_role",
        "field",
        "successful_rics",
        "avg_coverage",
        "max_coverage",
        "total_non_null",
        "errors",
        "validation_status",
    ]
    return catalog[keep_cols].sort_values(["country", "exchange", "field_role"]).reset_index(drop=True)


def write_refinitiv_field_probe_artifacts(
    artifact_root: str | Path,
    run_id: str,
    samples: pd.DataFrame,
    results: pd.DataFrame,
    session_name: str | None,
    interval: str,
    count: int,
) -> RefinitivFieldProbeResult:
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_refinitiv_field_probe(results)
    recommended = recommend_refinitiv_field_map(summary)
    field_catalog = build_refinitiv_field_catalog(recommended)
    validation = build_refinitiv_ric_validation_manifest(results)

    sample_path = artifact_dir / "sample_universe.csv"
    results_path = artifact_dir / "field_probe_results.csv"
    summary_path = artifact_dir / "field_probe_summary.csv"
    recommended_path = artifact_dir / "recommended_field_map.csv"
    field_catalog_path = artifact_dir / "field_catalog.csv"
    validation_path = artifact_dir / "ric_validation_manifest.csv"
    manifest_path = artifact_dir / "manifest.json"
    samples.to_csv(sample_path, index=False)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    recommended.to_csv(recommended_path, index=False)
    field_catalog.to_csv(field_catalog_path, index=False)
    validation.to_csv(validation_path, index=False)
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "session_name": session_name,
        "interval": interval,
        "count": count,
        "samples": int(len(samples)),
        "probes": int(len(results)),
        "recommended_rows": int(len(recommended)),
        "field_catalog_path": str(field_catalog_path),
        "field_catalog_rows": int(len(field_catalog)),
        "ric_validation_path": str(validation_path),
        "validated_rics": int(validation["status"].eq("validated_price_history").sum()) if "status" in validation else 0,
        "unresolved_candidate_rics": int(validation["status"].eq("no_history_for_candidate_ric").sum())
        if "status" in validation
        else 0,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return RefinitivFieldProbeResult(
        artifact_dir=artifact_dir,
        sample_path=sample_path,
        results_path=results_path,
        summary_path=summary_path,
        recommended_path=recommended_path,
        field_catalog_path=field_catalog_path,
        validation_path=validation_path,
        manifest_path=manifest_path,
        results=results,
        recommended=recommended,
    )
