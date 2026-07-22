from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.data_lake import _slug, default_data_lake_root, wide_ohlcv_to_long, write_ohlcv_long_partitions
from quant_lab.data.vendor_adapters import REFINITIV_OHLCV_FIELD_MAP, RefinitivRetryableProviderError
from quant_lab.data.yfinance_loader import DownloadResult


VALIDATED_REFINITIV_RIC_STATUSES = {
    "validated_price_history",
    "security_master_discovery_candidate",
    "independently_validated_security_master",
}
PAYLOAD_HISTORICAL_STATUSES = {"completed", "completed_partial"}
STRUCTURAL_HISTORICAL_STATUSES = {"completed_prelisting", "completed_postretirement"}
RESUMABLE_HISTORICAL_STATUSES = PAYLOAD_HISTORICAL_STATUSES | STRUCTURAL_HISTORICAL_STATUSES


@dataclass(frozen=True)
class HistoricalBackfillRun:
    run_id: str
    artifact_dir: Path
    plan_path: Path
    manifest_path: Path
    long_ohlcv_root: Path
    manifest: pd.DataFrame


def parse_country_filter(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip().casefold() for item in value.split(",") if item.strip()}


def historical_date_windows(start_date: str, end_date: str, window_years: int = 5) -> list[tuple[str, str]]:
    """Return non-overlapping inclusive date windows."""

    if window_years <= 0:
        raise ValueError("window_years must be positive")
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    windows: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        window_end = min(cursor + pd.DateOffset(years=window_years) - pd.Timedelta(days=1), end)
        windows.append((cursor.date().isoformat(), window_end.date().isoformat()))
        cursor = window_end + pd.Timedelta(days=1)
    return windows


def refinitiv_universe_from_symbol_map(
    symbol_map_csv: str | Path,
    include_countries: set[str] | None = None,
    exclude_countries: set[str] | None = None,
    max_tickers: int | None = None,
) -> pd.DataFrame:
    """Load accepted Refinitiv RICs from a symbol map with country filters."""

    symbol_map = pd.read_csv(symbol_map_csv, low_memory=False)
    required_cols = {"refinitiv_ric", "refinitiv_symbol_status"}
    missing = required_cols - set(symbol_map.columns)
    if missing:
        raise ValueError(f"Refinitiv symbol-map CSV is missing columns: {sorted(missing)}")
    universe = symbol_map[
        symbol_map["refinitiv_symbol_status"].isin(VALIDATED_REFINITIV_RIC_STATUSES) & symbol_map["refinitiv_ric"].notna()
    ].copy()
    if universe.empty:
        raise ValueError("Refinitiv symbol-map CSV has no validated symbols.")

    include_countries = include_countries or set()
    exclude_countries = exclude_countries or set()
    if include_countries or exclude_countries:
        if "country" not in universe.columns:
            raise ValueError("Refinitiv country filtering requires a country column in the symbol map.")
        country = universe["country"].astype("string").str.strip().str.casefold()
        if include_countries:
            universe = universe.loc[country.isin(include_countries)].copy()
            country = universe["country"].astype("string").str.strip().str.casefold()
        if exclude_countries:
            universe = universe.loc[~country.isin(exclude_countries)].copy()
        if universe.empty:
            raise ValueError("Refinitiv country filters removed every validated symbol.")

    if "yahoo_ticker" not in universe.columns:
        universe["yahoo_ticker"] = universe["refinitiv_ric"]
    else:
        universe["source_yahoo_ticker"] = universe["yahoo_ticker"]
        universe["yahoo_ticker"] = universe["refinitiv_ric"]
    universe["refinitiv_validation_status"] = universe["refinitiv_symbol_status"]
    universe["refinitiv_symbol_map_path"] = str(symbol_map_csv)
    universe = universe.sort_values([col for col in ("country", "exchange", "ticker", "refinitiv_ric") if col in universe.columns])
    if max_tickers is not None:
        if max_tickers <= 0:
            return universe.head(0).copy()
        universe = universe.head(max_tickers).copy()
    return universe.reset_index(drop=True)


def refinitiv_field_map_from_catalog(field_catalog: pd.DataFrame, country: object, exchange: object) -> dict[str, str] | None:
    """Resolve an OHLCV field map for a country/exchange, with safe country fallback."""

    if field_catalog.empty:
        return None
    required = {"country", "exchange", "field", "field_role"}
    if not required.issubset(field_catalog.columns):
        return None
    country_value = str(country)
    exchange_value = str(exchange)
    country_catalog = field_catalog[field_catalog["country"].astype(str).eq(country_value)].copy()
    market = country_catalog[country_catalog["exchange"].astype(str).eq(exchange_value)].copy()
    if market.empty and not country_catalog.empty:
        role_fields = country_catalog.groupby("field_role")["field"].nunique(dropna=True)
        if not role_fields.empty and role_fields.le(1).all():
            market = country_catalog.copy()
    if market.empty:
        return None
    field_map = dict(zip(market["field_role"].astype(str), market["field"].astype(str), strict=False))
    return field_map or None


def build_refinitiv_historical_plan(
    universe: pd.DataFrame,
    root: str | Path,
    run_id: str,
    start_date: str,
    end_date: str,
    batch_size: int = 10,
    window_years: int = 5,
    interval: str = "1d",
) -> pd.DataFrame:
    """Create a deterministic country/exchange/batch/window extraction plan."""

    if universe.empty:
        raise ValueError("universe must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    windows = historical_date_windows(start_date=start_date, end_date=end_date, window_years=window_years)
    group_cols = [col for col in ("country", "exchange") if col in universe.columns]
    if not group_cols:
        grouped = [(("unknown", "unknown"), universe.copy())]
        group_cols = ["country", "exchange"]
    else:
        grouped = list(universe.groupby(group_cols, dropna=False, sort=True))

    root_path = Path(root)
    batch_root = root_path / "prices" / run_id
    rows: list[dict[str, object]] = []
    ownership_columns = [col for col in ("country", "exchange") if col in universe.columns]
    ownership = universe[[*ownership_columns, "refinitiv_ric"]].drop_duplicates()
    if ownership["refinitiv_ric"].duplicated().any():
        conflicting = sorted(ownership.loc[ownership["refinitiv_ric"].duplicated(False), "refinitiv_ric"].astype(str).unique())
        raise ValueError(f"Refinitiv RICs span more than one market ownership group: {conflicting[:20]}")
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        meta = dict(zip(group_cols, key, strict=False))
        country = meta.get("country", "unknown")
        exchange = meta.get("exchange", "unknown")
        country_slug = _slug(country)
        exchange_slug = _slug(exchange)
        group = (
            group.sort_values([col for col in ("ticker", "refinitiv_ric") if col in group.columns])
            .drop_duplicates("refinitiv_ric", keep="first")
            .reset_index(drop=True)
        )
        for batch_no, start_idx in enumerate(range(0, len(group), batch_size), start=1):
            batch = group.iloc[start_idx : start_idx + batch_size].copy()
            tickers = batch["refinitiv_ric"].dropna().astype(str).tolist()
            if not tickers:
                continue
            for window_no, (window_start, window_end) in enumerate(windows, start=1):
                output_path = (
                    batch_root
                    / f"country={country_slug}"
                    / f"exchange={exchange_slug}"
                    / f"batch={batch_no:04d}"
                    / f"window={window_no:03d}_{window_start}_{window_end}"
                    / "prices.parquet"
                )
                plan_row_id = f"{country_slug}-{exchange_slug}-{batch_no:04d}-{window_no:03d}"
                rows.append(
                    {
                        "plan_row_id": plan_row_id,
                        "run_id": run_id,
                        "country": country,
                        "exchange": exchange,
                        "country_slug": country_slug,
                        "exchange_slug": exchange_slug,
                        "batch_no": batch_no,
                        "window_no": window_no,
                        "start_date": window_start,
                        "end_date": window_end,
                        "interval": interval,
                        "requested_tickers": len(tickers),
                        "tickers": ",".join(tickers),
                        "local_symbols": ",".join(batch["ticker"].dropna().astype(str).tolist()) if "ticker" in batch.columns else "",
                        "output_path": str(output_path),
                        "coverage_path": str(output_path.with_suffix(".coverage.csv")),
                        "ohlcv_path": str(output_path.with_suffix(".ohlcv.parquet")),
                        "long_ohlcv_root": str(root_path / "prices_long" / run_id / "ohlcv"),
                        "status": "pending",
                    }
                )
    plan = pd.DataFrame(rows)
    if plan.empty:
        raise ValueError("historical plan is empty")
    return plan


def write_refinitiv_historical_plan(
    symbol_map_csv: str | Path,
    root: str | Path,
    run_id: str,
    start_date: str,
    end_date: str,
    artifact_root: str | Path = "artifacts/historical_backfill",
    include_countries: set[str] | None = None,
    exclude_countries: set[str] | None = None,
    max_tickers: int | None = None,
    batch_size: int = 10,
    window_years: int = 5,
    interval: str = "1d",
) -> tuple[Path, pd.DataFrame]:
    universe = refinitiv_universe_from_symbol_map(
        symbol_map_csv,
        include_countries=include_countries,
        exclude_countries=exclude_countries,
        max_tickers=max_tickers,
    )
    plan = build_refinitiv_historical_plan(
        universe=universe,
        root=root,
        run_id=run_id,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        window_years=window_years,
        interval=interval,
    )
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_dir / "historical_plan.csv"
    plan.to_csv(plan_path, index=False)
    return plan_path, plan


def execute_refinitiv_historical_plan(
    plan: pd.DataFrame,
    root: str | Path,
    run_id: str,
    downloader: Callable[..., DownloadResult],
    universe: pd.DataFrame,
    artifact_dir: str | Path,
    field_catalog: pd.DataFrame | None = None,
    max_plan_rows: int | None = None,
    resume: bool = True,
    provider: str = "refinitiv",
    adjustment: str = "vendor_default",
    data_tier: str = "bronze",
    exclude_plan_row_ids: set[str] | None = None,
    include_plan_row_ids: set[str] | None = None,
) -> HistoricalBackfillRun:
    """Execute a historical plan row-by-row and persist progress after each row."""

    if plan.empty:
        raise ValueError("plan must not be empty")
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    plan_path = artifact_path / "historical_plan.csv"
    manifest_path = artifact_path / "historical_manifest.csv"
    long_ohlcv_root = Path(root) / "prices_long" / run_id / "ohlcv"
    field_catalog = field_catalog if field_catalog is not None else pd.DataFrame()
    fetched_at_utc = datetime.now(timezone.utc).isoformat()

    existing = pd.read_csv(manifest_path, low_memory=False) if resume and manifest_path.exists() else pd.DataFrame()
    completed = set(existing.loc[_resumable_historical_rows(existing), "plan_row_id"].astype(str)) if not existing.empty else set()
    manifest_payload_ids: set[str] = set()
    if not existing.empty and {"plan_row_id", "status", "long_ohlcv_rows"}.issubset(existing.columns):
        payload_rows = pd.to_numeric(existing["long_ohlcv_rows"], errors="coerce").fillna(0)
        payload_mask = existing["status"].astype(str).isin(PAYLOAD_HISTORICAL_STATUSES) & payload_rows.gt(0)
        manifest_payload_ids = set(existing.loc[payload_mask, "plan_row_id"].astype(str))
    lake_payload_ids = _existing_long_batch_ids(long_ohlcv_root) if resume else set()
    orphan_payload_ids = sorted(lake_payload_ids - manifest_payload_ids)
    integrity_halt = (
        {
            "error_type": "OrphanLongPayload",
            "error": "Long OHLCV payload exists without a durable completed manifest row.",
            "orphan_payload_count": len(orphan_payload_ids),
            "orphan_plan_row_ids": orphan_payload_ids,
        }
        if orphan_payload_ids
        else None
    )
    excluded = {str(plan_row_id) for plan_row_id in (exclude_plan_row_ids or set()) if str(plan_row_id)}
    included = {str(plan_row_id) for plan_row_id in (include_plan_row_ids or set()) if str(plan_row_id)}
    manifest_rows = existing.to_dict("records") if not existing.empty else []

    work_plan = plan.copy()
    pending_mask = ~work_plan["plan_row_id"].astype(str).isin(completed | excluded)
    if included:
        pending_mask = pending_mask & work_plan["plan_row_id"].astype(str).isin(included)
    if max_plan_rows is not None:
        if max_plan_rows <= 0:
            raise ValueError("max_plan_rows must be positive")
        pending = work_plan[pending_mask].head(max_plan_rows)
    else:
        pending = work_plan[pending_mask]
    if integrity_halt is not None:
        pending = pending.head(0)

    universe_by_ric = universe.copy()
    universe_by_ric["refinitiv_ric"] = universe_by_ric["refinitiv_ric"].astype(str)

    provider_halt: dict[str, str] | None = None
    for _, row in pending.iterrows():
        stop_after_row = False
        tickers = [ticker for ticker in str(row["tickers"]).split(",") if ticker]
        batch_universe = universe_by_ric[universe_by_ric["refinitiv_ric"].isin(tickers)].copy()
        batch_universe["yahoo_ticker"] = batch_universe["refinitiv_ric"]
        output_path = Path(str(row["output_path"]))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        field_map = refinitiv_field_map_from_catalog(field_catalog, row["country"], row["exchange"])
        if field_map is None and "validated_close_field" in batch_universe.columns:
            close_fields = batch_universe["validated_close_field"].dropna().astype(str).unique().tolist()
            if len(close_fields) == 1:
                field_map = {**REFINITIV_OHLCV_FIELD_MAP, "Close": close_fields[0]}

        started_at = datetime.now(timezone.utc).isoformat()
        try:
            result = downloader(
                tickers=tickers,
                start=str(row["start_date"]),
                end=str(row["end_date"]),
                interval=str(row["interval"]),
                output_path=output_path,
                ohlcv=True,
                field_map=field_map,
            )
            long = pd.DataFrame()
            long_rows = 0
            outside_window_rows_removed = 0
            canonical_coverage_path = output_path.with_suffix(".canonical_coverage.csv")
            if result.ohlcv is not None and not result.ohlcv.empty:
                long = wide_ohlcv_to_long(
                    result.ohlcv,
                    listings=batch_universe,
                    provider=provider,
                    interval=str(row["interval"]),
                    adjustment=adjustment,
                    data_tier=data_tier,
                    run_id=run_id,
                    batch_id=str(row["plan_row_id"]),
                    fetched_at_utc=fetched_at_utc,
                )
                raw_long_rows = int(len(long))
                long = _clip_long_to_listing_windows(long, batch_universe)
                outside_window_rows_removed = raw_long_rows - int(len(long))
                if not long.empty:
                    write_ohlcv_long_partitions(long, root=root, run_id=run_id)
                    long_rows = int(long.shape[0])
            canonical_coverage = _canonical_coverage(long, tickers)
            _atomic_to_csv(canonical_coverage, canonical_coverage_path)
            provider_failure_count = len(result.failures)
            failure_count = int(canonical_coverage["rows"].eq(0).sum())
            price_rows = int(result.prices.shape[0])
            price_columns = int(result.prices.shape[1])
            prelisting, first_trade_date_min = _prelisting_evidence(
                batch_universe,
                window_end=str(row["end_date"]),
                requested_tickers=tickers,
            )
            postretirement, retire_date_max = _postretirement_evidence(
                batch_universe,
                window_start=str(row["start_date"]),
                requested_tickers=tickers,
            )
            if (price_columns <= 0 or long_rows <= 0) and prelisting:
                status = "completed_prelisting"
                error_type = ""
                error = ""
                structural_reason = "all_requested_tickers_first_trade_after_window"
            elif (price_columns <= 0 or long_rows <= 0) and postretirement:
                status = "completed_postretirement"
                error_type = ""
                error = ""
                structural_reason = "all_requested_tickers_retired_before_window"
            elif price_columns <= 0 or long_rows <= 0:
                status = "failed_empty"
                error_type = "EmptyHistory"
                error = "No usable OHLCV output was returned for this plan row."
                structural_reason = ""
            else:
                status = "completed" if failure_count == 0 else "completed_partial"
                error_type = ""
                error = ""
                structural_reason = ""
            manifest_row = {
                **row.to_dict(),
                "status": status,
                "price_rows": price_rows,
                "price_columns": price_columns,
                "failures": failure_count,
                "provider_failures": provider_failure_count,
                "long_ohlcv_rows": long_rows,
                "outside_listing_window_rows_removed": outside_window_rows_removed,
                "canonical_coverage_path": str(canonical_coverage_path),
                "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": error_type,
                "error": error,
                "structural_reason": structural_reason,
                "first_trade_date_min": first_trade_date_min,
                "retire_date_max": retire_date_max,
            }
        except RefinitivRetryableProviderError as exc:  # pragma: no cover - provider-dependent path.
            manifest_row = {
                **row.to_dict(),
                "status": "failed_retryable_provider",
                "price_rows": 0,
                "price_columns": 0,
                "failures": len(tickers),
                "provider_failures": len(tickers),
                "long_ohlcv_rows": 0,
                "outside_listing_window_rows_removed": 0,
                "canonical_coverage_path": "",
                "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "structural_reason": "",
                "first_trade_date_min": "",
                "retire_date_max": "",
            }
            provider_halt = {
                "plan_row_id": str(row["plan_row_id"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            stop_after_row = True
        except Exception as exc:  # pragma: no cover - provider-dependent path.
            manifest_row = {
                **row.to_dict(),
                "status": "failed",
                "price_rows": 0,
                "price_columns": 0,
                "failures": len(tickers),
                "provider_failures": len(tickers),
                "long_ohlcv_rows": 0,
                "outside_listing_window_rows_removed": 0,
                "canonical_coverage_path": "",
                "started_at_utc": started_at,
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "structural_reason": "",
                "first_trade_date_min": "",
                "retire_date_max": "",
            }
        manifest_rows = [item for item in manifest_rows if str(item.get("plan_row_id")) != str(row["plan_row_id"])]
        manifest_rows.append(manifest_row)
        _atomic_to_csv(
            pd.DataFrame(manifest_rows).sort_values(["country", "exchange", "batch_no", "window_no"]),
            manifest_path,
        )
        if stop_after_row:
            break

    final_manifest = pd.read_csv(manifest_path, low_memory=False) if manifest_path.exists() else pd.DataFrame(manifest_rows)
    _atomic_to_csv(plan, plan_path)
    completed_mask = _resumable_historical_rows(final_manifest)
    failed_mask = (
        final_manifest["status"].astype(str).str.startswith("failed", na=False)
        if not final_manifest.empty and "status" in final_manifest
        else pd.Series(dtype=bool)
    )
    run_manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_path),
        "plan_path": str(plan_path),
        "manifest_path": str(manifest_path),
        "long_ohlcv_root": str(long_ohlcv_root),
        "planned_rows": int(plan.shape[0]),
        "executed_rows": int(len(final_manifest)),
        "completed_rows": int(completed_mask.sum()) if not final_manifest.empty else 0,
        "failed_rows": int(failed_mask.sum()) if not final_manifest.empty else 0,
        "empty_rows": int(final_manifest["status"].astype(str).eq("failed_empty").sum())
        if not final_manifest.empty and "status" in final_manifest
        else 0,
        "excluded_rows": int(len(excluded)),
        "included_rows": int(len(included)),
        "resume": bool(resume),
        "halted_early": provider_halt is not None or integrity_halt is not None,
        "provider_halt": provider_halt,
        "integrity_halt": integrity_halt,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (artifact_path / "historical_run_manifest.json").write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    return HistoricalBackfillRun(
        run_id=run_id,
        artifact_dir=artifact_path,
        plan_path=plan_path,
        manifest_path=manifest_path,
        long_ohlcv_root=long_ohlcv_root,
        manifest=final_manifest,
    )


def default_historical_lake_root(project_root: str | Path = ".") -> Path:
    return default_data_lake_root(project_root)


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _existing_long_batch_ids(root: Path) -> set[str]:
    """Stream batch identifiers from an existing Parquet lake without loading price rows."""

    if not root.is_dir() or not next(root.rglob("*.parquet"), None):
        return set()
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    dataset = ds.dataset(root, format="parquet", partitioning="hive", exclude_invalid_files=True)
    if "batch_id" not in dataset.schema.names:
        raise ValueError(f"existing OHLCV lake has no batch_id column: {root}")
    batch_ids: set[str] = set()
    for batch in dataset.scanner(columns=["batch_id"], batch_size=65_536, use_threads=True).to_batches():
        values = pc.unique(batch.column("batch_id")).to_pylist()
        batch_ids.update(str(value) for value in values if value is not None and str(value))
    return batch_ids


def _clip_long_to_listing_windows(long: pd.DataFrame, listings: pd.DataFrame) -> pd.DataFrame:
    """Keep rows inside the validated identity windows for each requested RIC."""

    if long.empty or listings.empty or "ticker" not in long or "refinitiv_ric" not in listings:
        return long
    start_col = "listing_start" if "listing_start" in listings else "first_trade_date" if "first_trade_date" in listings else None
    end_col = "listing_end" if "listing_end" in listings else "retire_date" if "retire_date" in listings else None
    if start_col is None:
        return long

    dates = pd.to_datetime(long["date"], errors="coerce").dt.normalize()
    keep = pd.Series(False, index=long.index)
    for ric, windows in listings.groupby(listings["refinitiv_ric"].astype(str), dropna=False):
        ticker_rows = long["ticker"].astype(str).eq(str(ric))
        if not ticker_rows.any():
            continue
        valid_window_found = False
        for window in windows.to_dict("records"):
            start = pd.to_datetime(window.get(start_col), errors="coerce")
            if pd.isna(start):
                continue
            end = pd.to_datetime(window.get(end_col), errors="coerce") if end_col else pd.NaT
            right_open = _truthy_scalar(window.get("right_open")) if "right_open" in windows else pd.isna(end)
            window_rows = dates.ge(pd.Timestamp(start).normalize())
            if not right_open and pd.notna(end):
                window_rows &= dates.le(pd.Timestamp(end).normalize())
            keep |= ticker_rows & window_rows
            valid_window_found = True
        if not valid_window_found:
            keep |= ticker_rows
    known_rics = set(listings["refinitiv_ric"].dropna().astype(str))
    keep |= ~long["ticker"].astype(str).isin(known_rics)
    return long.loc[keep].reset_index(drop=True)


def _canonical_coverage(long: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    if long.empty:
        counts: dict[str, int] = {}
        starts: dict[str, Any] = {}
        ends: dict[str, Any] = {}
    else:
        normalized = long.copy()
        normalized["ticker"] = normalized["ticker"].astype(str)
        normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
        grouped = normalized.groupby("ticker", dropna=False)["date"]
        counts = grouped.size().astype(int).to_dict()
        starts = grouped.min().to_dict()
        ends = grouped.max().to_dict()
    return pd.DataFrame(
        {
            "ticker": tickers,
            "rows": [int(counts.get(ticker, 0)) for ticker in tickers],
            "start": [starts.get(ticker, pd.NaT) for ticker in tickers],
            "end": [ends.get(ticker, pd.NaT) for ticker in tickers],
        }
    )


def _truthy_scalar(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def _resumable_historical_rows(manifest: pd.DataFrame) -> pd.Series:
    """Rows that should be skipped by resume because they produced usable payloads."""

    if manifest.empty or "status" not in manifest.columns or "plan_row_id" not in manifest.columns:
        return pd.Series(False, index=manifest.index)
    status = manifest["status"].astype(str)
    payload_status_ok = status.isin(PAYLOAD_HISTORICAL_STATUSES)
    structural_status_ok = status.isin(STRUCTURAL_HISTORICAL_STATUSES)
    payload_checks: list[pd.Series] = []
    if "price_columns" in manifest.columns:
        payload_checks.append(pd.to_numeric(manifest["price_columns"], errors="coerce").fillna(0).gt(0))
    if "long_ohlcv_rows" in manifest.columns:
        payload_checks.append(pd.to_numeric(manifest["long_ohlcv_rows"], errors="coerce").fillna(0).gt(0))
    if not payload_checks:
        return payload_status_ok | structural_status_ok
    payload_ok = payload_checks[0]
    for check in payload_checks[1:]:
        payload_ok = payload_ok & check
    return (payload_status_ok & payload_ok) | structural_status_ok


def _prelisting_evidence(
    batch_universe: pd.DataFrame,
    window_end: str,
    requested_tickers: list[str],
) -> tuple[bool, str]:
    """Prove that an empty plan row predates every requested security's first trade."""

    if not requested_tickers or "first_trade_date" not in batch_universe.columns:
        return False, ""
    ric = batch_universe.get("refinitiv_ric", pd.Series(dtype="string")).astype(str)
    evidence = batch_universe.loc[ric.isin(requested_tickers), ["refinitiv_ric", "first_trade_date"]].copy()
    if evidence["refinitiv_ric"].astype(str).nunique() != len(set(requested_tickers)):
        return False, ""
    first_trade_dates = pd.to_datetime(evidence["first_trade_date"], errors="coerce").dt.normalize()
    if first_trade_dates.isna().any():
        return False, ""
    is_prelisting = bool(first_trade_dates.gt(pd.Timestamp(window_end).normalize()).all())
    first_trade_date_min = first_trade_dates.min().date().isoformat() if is_prelisting else ""
    return is_prelisting, first_trade_date_min


def _postretirement_evidence(
    batch_universe: pd.DataFrame,
    window_start: str,
    requested_tickers: list[str],
) -> tuple[bool, str]:
    """Prove that an empty plan row follows every requested security's retirement."""

    if not requested_tickers or "retire_date" not in batch_universe.columns:
        return False, ""
    ric = batch_universe.get("refinitiv_ric", pd.Series(dtype="string")).astype(str)
    evidence = batch_universe.loc[ric.isin(requested_tickers), ["refinitiv_ric", "retire_date"]].copy()
    if evidence["refinitiv_ric"].astype(str).nunique() != len(set(requested_tickers)):
        return False, ""
    retire_dates = pd.to_datetime(evidence["retire_date"], errors="coerce").dt.normalize()
    if retire_dates.isna().any():
        return False, ""
    is_postretirement = bool(retire_dates.lt(pd.Timestamp(window_start).normalize()).all())
    retire_date_max = retire_dates.max().date().isoformat() if is_postretirement else ""
    return is_postretirement, retire_date_max
