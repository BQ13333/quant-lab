from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class JapanOHLCVCanonicalizationResult:
    run_id: str
    artifact_dir: Path
    data_root: Path
    manifest_path: Path
    quality_path: Path
    hash_inventory_path: Path
    manifest: dict[str, Any]


def canonicalize_japan_ohlcv(
    *,
    raw_root: str | Path,
    listing_windows_path: str | Path,
    historical_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_ohlcv_canonical",
    run_id: str,
) -> JapanOHLCVCanonicalizationResult:
    """Publish a bounded-memory, listing-window canonical OHLCV candidate."""

    raw_root = Path(raw_root)
    listing_windows_path = Path(listing_windows_path)
    historical_dir = Path(historical_dir)
    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists():
        raise FileExistsError(f"immutable canonical OHLCV artifact already exists: {artifact_dir}")
    if not raw_root.is_dir() or not listing_windows_path.is_file():
        raise FileNotFoundError("raw OHLCV root and listing-window evidence are required")

    run_manifest_path = historical_dir / "historical_run_manifest.json"
    terminal_path = historical_dir / "historical_manifest.csv"
    plan_path = historical_dir / "historical_plan.csv"
    if not all(path.is_file() for path in (run_manifest_path, terminal_path, plan_path)):
        raise FileNotFoundError("historical plan, terminal manifest, and run manifest are required")

    windows = _normalize_windows(_read_table(listing_windows_path))
    if windows.empty or not windows["pit_usable"].all():
        raise ValueError("listing-window evidence is empty or contains non-PIT-usable rows")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    terminal = pd.read_csv(terminal_path, low_memory=False)
    plan = pd.read_csv(plan_path, low_memory=False)

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=artifact_dir.parent))
    data_root = temporary / "ohlcv"
    quality_rows: list[dict[str, Any]] = []
    source_batch_counts: dict[str, int] = {}
    canonical_batch_counts: dict[str, int] = {}
    canonical_ticker_counts: dict[tuple[str, str], int] = {}
    try:
        month_dirs = sorted(path for path in raw_root.rglob("month_partition=*") if path.is_dir())
        if not month_dirs:
            raise ValueError(f"raw OHLCV root has no month partitions: {raw_root}")
        for month_dir in month_dirs:
            source_files = sorted(month_dir.glob("*.parquet"))
            if not source_files:
                continue
            source = pd.concat((pd.read_parquet(path) for path in source_files), ignore_index=True)
            canonical, outside_rows, unknown_rows = _canonicalize_frame(source, windows)
            _accumulate_batch_counts(source_batch_counts, source)
            _accumulate_batch_counts(canonical_batch_counts, canonical)
            _accumulate_ticker_counts(canonical_ticker_counts, canonical)
            duplicate_rows = int(canonical.duplicated(["ticker", "date"]).sum())
            if duplicate_rows:
                raise ValueError(f"duplicate canonical RIC/date rows in {month_dir}: {duplicate_rows}")
            relative = month_dir.relative_to(raw_root)
            output_dir = data_root / relative
            output_dir.mkdir(parents=True, exist_ok=True)
            if not canonical.empty:
                canonical.to_parquet(output_dir / "part-00000.parquet", index=False, compression="zstd")
            quality_rows.append(
                {
                    "partition": str(relative),
                    "source_files": len(source_files),
                    "source_rows": len(source),
                    "canonical_rows": len(canonical),
                    "outside_listing_window_rows": outside_rows,
                    "unknown_ric_rows": unknown_rows,
                    "duplicate_canonical_rows": duplicate_rows,
                }
            )

        quality = pd.DataFrame(quality_rows)
        if quality.empty:
            raise ValueError("no canonical partitions were produced")
        quality_path = temporary / "partition_quality.csv"
        quality.to_csv(quality_path, index=False)
        output_files = sorted(data_root.rglob("*.parquet"))
        output_rows = sum(int(value) for value in quality["canonical_rows"])
        source_rows = sum(int(value) for value in quality["source_rows"])
        outside_rows = sum(int(value) for value in quality["outside_listing_window_rows"])
        unknown_rows = sum(int(value) for value in quality["unknown_ric_rows"])
        duplicate_rows = sum(int(value) for value in quality["duplicate_canonical_rows"])
        verified_rows = sum(len(pd.read_parquet(path, columns=["ticker"])) for path in output_files)
        if verified_rows != output_rows:
            raise ValueError(f"canonical read-back mismatch: expected {output_rows}, found {verified_rows}")

        planned = _integer(run_manifest.get("planned_rows"), -1)
        completed = _integer(run_manifest.get("completed_rows"), -1)
        failed = _integer(run_manifest.get("failed_rows"), -1)
        extraction_complete = bool(
            planned > 0
            and planned == len(plan) == len(terminal) == completed
            and failed == 0
            and not terminal["status"].astype(str).str.startswith("failed").any()
        )
        quality_passed = unknown_rows == 0 and duplicate_rows == 0 and source_rows == output_rows + outside_rows
        promotion_eligible = extraction_complete and quality_passed

        canonical_plan_path = temporary / "historical_plan.csv"
        plan.to_csv(canonical_plan_path, index=False)
        canonical_terminal = terminal.copy()
        source_terminal_rows = (
            canonical_terminal["long_ohlcv_rows"]
            if "long_ohlcv_rows" in canonical_terminal
            else pd.Series(0, index=canonical_terminal.index)
        )
        canonical_terminal["source_long_ohlcv_rows"] = pd.to_numeric(
            source_terminal_rows, errors="coerce"
        ).fillna(0).astype(int)
        canonical_terminal["long_ohlcv_rows"] = (
            canonical_terminal["plan_row_id"].astype(str).map(canonical_batch_counts).fillna(0).astype(int)
        )
        canonical_terminal["outside_listing_window_rows_removed"] = (
            canonical_terminal["plan_row_id"].astype(str).map(source_batch_counts).fillna(0).astype(int)
            - canonical_terminal["long_ohlcv_rows"]
        )
        coverage_root = temporary / "coverage"
        coverage_root.mkdir(parents=True, exist_ok=True)
        original_failures = (
            canonical_terminal["failures"]
            if "failures" in canonical_terminal
            else pd.Series(0, index=canonical_terminal.index)
        )
        canonical_terminal["provider_failures"] = pd.to_numeric(
            original_failures, errors="coerce"
        ).fillna(0).astype(int)
        for index, row in canonical_terminal.iterrows():
            plan_row_id = str(row["plan_row_id"])
            tickers = [value for value in str(row.get("tickers", "")).split(",") if value]
            counts = [int(canonical_ticker_counts.get((plan_row_id, ticker), 0)) for ticker in tickers]
            coverage = pd.DataFrame({"ticker": tickers, "rows": counts})
            coverage_path = coverage_root / f"{plan_row_id}.csv"
            coverage.to_csv(coverage_path, index=False)
            canonical_terminal.at[index, "canonical_coverage_path"] = str(Path("coverage") / coverage_path.name)
            missing = sum(value == 0 for value in counts)
            canonical_terminal.at[index, "failures"] = missing
            original_status = str(row.get("status", ""))
            if original_status.startswith("failed"):
                continue
            payload_rows = int(canonical_terminal.at[index, "long_ohlcv_rows"])
            if payload_rows > 0:
                canonical_terminal.at[index, "status"] = "completed" if missing == 0 else "completed_partial"
                continue
            reason = _structural_empty_reason(row, tickers, windows)
            if reason:
                canonical_terminal.at[index, "status"] = reason
                canonical_terminal.at[index, "structural_reason"] = f"canonical_{reason}"
            else:
                canonical_terminal.at[index, "status"] = "failed_empty_canonical"
                canonical_terminal.at[index, "error_type"] = "UnexplainedCanonicalEmpty"
                canonical_terminal.at[index, "error"] = (
                    "No PIT-valid OHLCV rows and no full-window lifecycle explanation."
                )
        canonical_terminal_path = temporary / "historical_manifest.csv"
        canonical_terminal.to_csv(canonical_terminal_path, index=False)
        canonical_run = {
            **run_manifest,
            "run_id": run_id,
            "source_run_id": run_manifest.get("run_id", ""),
            "long_ohlcv_root": "ohlcv",
            "planned_rows": len(plan),
            "executed_rows": len(canonical_terminal),
            "completed_rows": completed,
            "failed_rows": failed,
            "canonical": True,
            "listing_window_clipped": True,
            "canonical_rows": output_rows,
            "outside_listing_window_rows_removed": outside_rows,
            "promotion_eligible": promotion_eligible,
        }
        canonical_run_path = temporary / "historical_run_manifest.json"
        canonical_run_path.write_text(json.dumps(canonical_run, indent=2), encoding="utf-8")

        inventory_files = [
            run_manifest_path,
            terminal_path,
            plan_path,
            listing_windows_path,
            quality_path,
            canonical_plan_path,
            canonical_terminal_path,
            canonical_run_path,
            *sorted(coverage_root.glob("*.csv")),
            *output_files,
        ]
        inventory = {
            "algorithm": "sha256",
            "immutable": True,
            "coverage_complete": promotion_eligible,
            "files": {
                _inventory_key(path, temporary): _sha256(path)
                for path in inventory_files
            },
        }
        inventory_path = temporary / "hash_inventory.json"
        inventory_path.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
        manifest = {
            "run_id": run_id,
            "artifact_type": "japan_listing_window_canonical_ohlcv",
            "status": "pass" if promotion_eligible else "pass_partial_candidate",
            "evidence_scope": "full" if promotion_eligible else "partial_candidate",
            "promotion_eligible": promotion_eligible,
            "scorecard_eligible": promotion_eligible,
            "source_extraction_complete": extraction_complete,
            "quality_passed": quality_passed,
            "source_rows": source_rows,
            "canonical_rows": output_rows,
            "outside_listing_window_rows_removed": outside_rows,
            "unknown_ric_rows": unknown_rows,
            "duplicate_canonical_rows": duplicate_rows,
            "partition_count": len(quality),
            "output_file_count": len(output_files),
            "source_run_id": run_manifest.get("run_id", ""),
            "source_plan_rows": len(plan),
            "source_terminal_rows": len(terminal),
            "source_completed_rows": completed,
            "source_failed_rows": failed,
            "artifacts": {
                "ohlcv_root": "ohlcv",
                "partition_quality": "partition_quality.csv",
                "hash_inventory": "hash_inventory.json",
                "historical_plan": "historical_plan.csv",
                "historical_manifest": "historical_manifest.csv",
                "historical_run_manifest": "historical_run_manifest.json",
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, artifact_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return JapanOHLCVCanonicalizationResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        data_root=artifact_dir / "ohlcv",
        manifest_path=artifact_dir / "manifest.json",
        quality_path=artifact_dir / "partition_quality.csv",
        hash_inventory_path=artifact_dir / "hash_inventory.json",
        manifest=manifest,
    )


def _canonicalize_frame(source: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    if source.empty:
        return source.copy(), 0, 0
    required = {"ticker", "date"}
    if not required.issubset(source.columns):
        raise ValueError(f"OHLCV partition is missing columns: {sorted(required - set(source.columns))}")
    data = source.copy().reset_index(drop=True)
    data["_source_row_id"] = data.index
    data["ticker"] = data["ticker"].astype(str)
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    if data["date"].isna().any():
        raise ValueError("OHLCV partition contains invalid dates")
    joined = data[["_source_row_id", "ticker", "date"]].merge(windows, how="left", left_on="ticker", right_on="ric")
    known = joined["ric"].notna()
    in_window = known & joined["date"].ge(joined["start"])
    in_window &= joined["right_open"] | joined["date"].le(joined["end"])
    keep_ids = set(joined.loc[in_window, "_source_row_id"].astype(int))
    known_ids = set(joined.loc[known, "_source_row_id"].astype(int))
    all_ids = set(data["_source_row_id"].astype(int))
    unknown_ids = all_ids - known_ids
    canonical = data.loc[data["_source_row_id"].isin(keep_ids)].drop(columns="_source_row_id")
    canonical = canonical.sort_values(["ticker", "date"]).reset_index(drop=True)
    outside_rows = len(all_ids - keep_ids - unknown_ids)
    return canonical, outside_rows, len(unknown_ids)


def _normalize_windows(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"refinitiv_ric", "listing_start", "listing_end", "pit_usable"}
    if not required.issubset(frame.columns):
        raise ValueError(f"listing-window evidence is missing columns: {sorted(required - set(frame.columns))}")
    result = pd.DataFrame(
        {
            "ric": frame["refinitiv_ric"].astype(str),
            "start": pd.to_datetime(frame["listing_start"], errors="coerce").dt.normalize(),
            "end": pd.to_datetime(frame["listing_end"], errors="coerce").dt.normalize(),
            "right_open": (
                frame["right_open"] if "right_open" in frame else pd.Series(False, index=frame.index)
            ).map(_truthy),
            "pit_usable": frame["pit_usable"].map(_truthy),
            "window_end_known_at": pd.to_datetime(
                frame["window_end_known_at"]
                if "window_end_known_at" in frame
                else pd.Series(pd.NaT, index=frame.index),
                errors="coerce",
                utc=True,
            ),
        }
    )
    if result[["ric", "start", "end"]].isna().any().any():
        raise ValueError("listing-window evidence contains invalid keys or dates")
    return result.drop_duplicates().reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() in {".parquet", ".pq"} else pd.read_csv(path, low_memory=False)


def _accumulate_batch_counts(target: dict[str, int], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if "batch_id" not in frame:
        raise ValueError("OHLCV payload is missing batch_id lineage")
    counts = frame["batch_id"].dropna().astype(str).value_counts()
    for batch_id, count in counts.items():
        target[str(batch_id)] = target.get(str(batch_id), 0) + int(count)


def _accumulate_ticker_counts(target: dict[tuple[str, str], int], frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    if not {"batch_id", "ticker"}.issubset(frame.columns):
        raise ValueError("OHLCV payload is missing batch_id/ticker lineage")
    counts = frame.dropna(subset=["batch_id", "ticker"]).groupby(["batch_id", "ticker"]).size()
    for (batch_id, ticker), count in counts.items():
        key = (str(batch_id), str(ticker))
        target[key] = target.get(key, 0) + int(count)


def _structural_empty_reason(row: pd.Series, tickers: list[str], windows: pd.DataFrame) -> str:
    if not tickers:
        return ""
    start = pd.to_datetime(row.get("start_date"), errors="coerce")
    end = pd.to_datetime(row.get("end_date"), errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return ""
    relevant = windows.loc[windows["ric"].isin(tickers)]
    if set(relevant["ric"].astype(str)) != set(tickers) or not relevant["pit_usable"].all():
        return ""
    if relevant.groupby("ric")["start"].min().gt(pd.Timestamp(end).normalize()).all():
        return "completed_prelisting"
    closed = ~relevant["right_open"]
    known = relevant["window_end_known_at"].notna()
    if (
        closed.all()
        and known.all()
        and relevant.groupby("ric")["end"].max().lt(pd.Timestamp(start).normalize()).all()
    ):
        return "completed_postretirement"
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "pass", "passed", "complete"}


def _integer(value: Any, default: int) -> int:
    try:
        return default if value is None or pd.isna(value) else int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_key(path: Path, artifact_temp: Path) -> str:
    try:
        return str(path.relative_to(artifact_temp))
    except ValueError:
        return str(path.resolve())
