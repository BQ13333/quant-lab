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
class OHLCVRowEqualityResult:
    run_id: str
    artifact_dir: Path
    passed: bool
    manifest_path: Path
    checks_path: Path
    manifest: dict[str, Any]


def validate_ohlcv_row_equality(
    *,
    source_root: str | Path,
    compacted_root: str | Path,
    artifact_root: str | Path,
    run_id: str,
) -> OHLCVRowEqualityResult:
    """Independently prove source/compacted Parquet equality in bounded yearly slices."""

    source_root = Path(source_root)
    compacted_root = Path(compacted_root)
    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists():
        raise FileExistsError(f"immutable row-equality artifact already exists: {artifact_dir}")
    if not source_root.is_dir() or not compacted_root.is_dir():
        raise FileNotFoundError("source and compacted OHLCV roots are required")

    source_years = _year_directories(source_root)
    compacted_years = _year_directories(compacted_root)
    all_years = sorted(set(source_years) | set(compacted_years))
    if not all_years:
        raise ValueError("no year partitions were found")

    rows: list[dict[str, Any]] = []
    for year in all_years:
        source_files = sorted(source_years.get(year, Path()).rglob("*.parquet")) if year in source_years else []
        compacted_files = sorted(compacted_years.get(year, Path()).rglob("*.parquet")) if year in compacted_years else []
        source = _canonical_frame(_read_files(source_files))
        compacted = _canonical_frame(_read_files(compacted_files))
        columns_equal = list(source.columns) == list(compacted.columns)
        rows_equal = len(source) == len(compacted)
        source_hash = _frame_hash(source)
        compacted_hash = _frame_hash(compacted)
        exact_equal = bool(columns_equal and rows_equal and source_hash == compacted_hash)
        if exact_equal:
            try:
                pd.testing.assert_frame_equal(source, compacted, check_dtype=False, check_exact=True)
            except AssertionError:
                exact_equal = False
        rows.append(
            {
                "year": year,
                "source_files": len(source_files),
                "compacted_files": len(compacted_files),
                "source_rows": len(source),
                "compacted_rows": len(compacted),
                "columns_equal": columns_equal,
                "rows_equal": rows_equal,
                "source_row_sha256": source_hash,
                "compacted_row_sha256": compacted_hash,
                "exact_equal": exact_equal,
            }
        )

    checks = pd.DataFrame(rows)
    passed = bool(checks["exact_equal"].all())
    manifest = {
        "run_id": run_id,
        "artifact_type": "ohlcv_source_compaction_row_equality",
        "status": "pass" if passed else "failed_validation",
        "row_equality_passed": passed,
        "scorecard_eligible": passed,
        "source_root": str(source_root.resolve()),
        "compacted_root": str(compacted_root.resolve()),
        "year_count": len(checks),
        "source_files": int(checks["source_files"].sum()),
        "compacted_files": int(checks["compacted_files"].sum()),
        "source_rows": int(checks["source_rows"].sum()),
        "compacted_rows": int(checks["compacted_rows"].sum()),
        "failed_years": checks.loc[~checks["exact_equal"], "year"].astype(str).tolist(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=artifact_dir.parent))
    try:
        checks_path = temporary / "year_checks.csv"
        manifest_path = temporary / "manifest.json"
        checks.to_csv(checks_path, index=False)
        manifest["artifact_sha256"] = {"year_checks": _sha256(checks_path)}
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, artifact_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return OHLCVRowEqualityResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        passed=passed,
        manifest_path=artifact_dir / "manifest.json",
        checks_path=artifact_dir / "year_checks.csv",
        manifest=manifest,
    )


def _year_directories(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in root.rglob("year_partition=*"):
        if path.is_dir():
            year = path.name.split("=", 1)[1]
            if year in result:
                raise ValueError(f"multiple year ownership directories for {year} under {root}")
            result[year] = path
    return result


def _read_files(files: list[Path]) -> pd.DataFrame:
    if not files:
        return pd.DataFrame()
    return pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)


def _canonical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    data = frame.drop(columns=[column for column in frame if column.endswith("_partition")], errors="ignore").copy()
    if "date" in data:
        data["date"] = pd.to_datetime(data["date"], errors="raise").dt.tz_localize(None)
    columns = sorted(data.columns)
    data = data[columns]
    sort_keys = [column for column in ("ticker", "date", "batch_id") if column in data]
    if not sort_keys:
        raise ValueError("OHLCV row equality requires ticker/date or batch lineage keys")
    return data.sort_values(sort_keys, kind="mergesort", na_position="first").reset_index(drop=True)


def _frame_hash(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"").hexdigest()
    values = pd.util.hash_pandas_object(frame, index=False, categorize=True).to_numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
