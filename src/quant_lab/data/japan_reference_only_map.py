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
class JapanReferenceOnlyMapResult:
    artifact_dir: Path
    symbol_map_path: Path
    manifest_path: Path
    symbol_map: pd.DataFrame
    manifest: dict[str, Any]


def build_japan_reference_only_map(
    *,
    active_map_dir: str | Path,
    full_reference_map_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
) -> JapanReferenceOnlyMapResult:
    active_dir = Path(active_map_dir)
    reference_dir = Path(full_reference_map_dir)
    output_dir = Path(artifact_root) / run_id
    if output_dir.exists():
        raise FileExistsError(f"immutable reference-only map already exists: {output_dir}")
    active_path = active_dir / "symbol_map.csv"
    reference_path = reference_dir / "symbol_map.csv"
    active_manifest_path = active_dir / "manifest.json"
    reference_manifest_path = reference_dir / "manifest.json"
    for path in (active_path, reference_path, active_manifest_path, reference_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    active_manifest = _read_json(active_manifest_path)
    reference_manifest = _read_json(reference_manifest_path)
    if not _validated_map_manifest(active_manifest, "active"):
        raise ValueError("active map is not independently admitted full evidence")
    if not _validated_map_manifest(reference_manifest, "reference"):
        raise ValueError("full reference map is not independently admitted full evidence")

    active = pd.read_csv(active_path, low_memory=False)
    reference = pd.read_csv(reference_path, low_memory=False)
    key_columns = _window_key_columns(active, reference)
    active_keys = _key_index(active, key_columns)
    reference_keys = _key_index(reference, key_columns)
    missing = active_keys - reference_keys
    if missing:
        raise ValueError(f"active map has {len(missing)} windows absent from full reference map")
    active_key_series = active[key_columns].astype(str).agg("\x1f".join, axis=1)
    reference_key_series = reference[key_columns].astype(str).agg("\x1f".join, axis=1)
    reference_only = reference.loc[~reference_key_series.isin(set(active_key_series))].copy()
    if reference_only.empty:
        raise ValueError("reference-only map is empty")
    overlap = _key_index(reference_only, key_columns) & active_keys
    union = _key_index(reference_only, key_columns) | active_keys
    if overlap or union != reference_keys:
        raise ValueError("active/reference-only partition does not exactly reconstruct full reference map")
    if reference_only.duplicated(key_columns).any():
        raise ValueError("reference-only map has duplicate listing windows")
    interval_overlap_rows = _interval_overlap_rows(active, reference_only)
    if interval_overlap_rows:
        raise ValueError(
            f"active/reference-only maps have {interval_overlap_rows} overlapping RIC/date intervals"
        )
    reference_only = reference_only.sort_values(key_columns).reset_index(drop=True)

    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    try:
        symbol_map_path = temporary / "symbol_map.csv"
        reference_only.to_csv(symbol_map_path, index=False)
        manifest = {
            "run_id": run_id,
            "artifact_type": "validated_jpx_reference_only_symbol_map",
            "status": "pass",
            "validation_status": "pass",
            "evidence_scope": "full",
            "scorecard_eligible": True,
            "map_role": "reference",
            "partition_role": "reference_only",
            "independent_audit": False,
            "derived_from_independently_validated_maps": True,
            "source_checksums_verified": True,
            "mapping_coverage": 1.0,
            "unresolved_rows": 0,
            "overlap_conflict_rows": 0,
            "duplicate_window_rows": 0,
            "active_window_rows": len(active),
            "full_reference_window_rows": len(reference),
            "reference_only_window_rows": len(reference_only),
            "reference_only_unique_rics": int(reference_only["refinitiv_ric"].nunique()),
            "partition_union_rows": len(active) + len(reference_only),
            "partition_reconstructs_full_reference": True,
            "active_reference_window_overlap_rows": 0,
            "active_reference_interval_overlap_rows": interval_overlap_rows,
            "key_columns": key_columns,
            "input_sha256": {
                str(active_path.resolve()): _sha256(active_path),
                str(reference_path.resolve()): _sha256(reference_path),
                str(active_manifest_path.resolve()): _sha256(active_manifest_path),
                str(reference_manifest_path.resolve()): _sha256(reference_manifest_path),
            },
            "symbol_map_sha256": _sha256(symbol_map_path),
            "production_ready": False,
            "ibkr_ready": False,
            "live_execution_ready": False,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return JapanReferenceOnlyMapResult(
        artifact_dir=output_dir,
        symbol_map_path=output_dir / "symbol_map.csv",
        manifest_path=output_dir / "manifest.json",
        symbol_map=reference_only,
        manifest=manifest,
    )


def _window_key_columns(active: pd.DataFrame, reference: pd.DataFrame) -> list[str]:
    candidates = ["stable_listing_id", "refinitiv_ric", "listing_start", "listing_end"]
    columns = [column for column in candidates if column in active and column in reference]
    if not {"refinitiv_ric", "listing_start", "listing_end"}.issubset(columns):
        raise ValueError("symbol maps lack exact listing-window keys")
    return columns


def _key_index(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
    if frame[columns].isna().any().any():
        raise ValueError("symbol map contains null listing-window keys")
    return set(frame[columns].astype(str).itertuples(index=False, name=None))


def _validated_map_manifest(manifest: dict[str, Any], role: str) -> bool:
    return bool(
        manifest.get("status") == "pass"
        and manifest.get("validation_status") == "pass"
        and manifest.get("evidence_scope") == "full"
        and manifest.get("scorecard_eligible") is True
        and manifest.get("source_checksums_verified") is True
        and manifest.get("map_role") == role
        and int(manifest.get("unresolved_rows", 0) or 0) == 0
        and int(manifest.get("overlap_conflict_rows", 0) or 0) == 0
    )


def _interval_overlap_rows(active: pd.DataFrame, reference: pd.DataFrame) -> int:
    left = active[["refinitiv_ric", "listing_start", "listing_end"]].copy()
    right = reference[["refinitiv_ric", "listing_start", "listing_end"]].copy()
    for frame in (left, right):
        frame["listing_start"] = pd.to_datetime(frame["listing_start"], errors="coerce", utc=True)
        frame["listing_end"] = pd.to_datetime(frame["listing_end"], errors="coerce", utc=True)
        if frame[["listing_start", "listing_end"]].isna().any().any():
            raise ValueError("symbol map has invalid listing interval dates")
    joined = left.merge(right, on="refinitiv_ric", suffixes=("_active", "_reference"))
    overlap = joined["listing_start_active"].le(joined["listing_end_reference"])
    overlap &= joined["listing_start_reference"].le(joined["listing_end_active"])
    return int(overlap.sum())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
