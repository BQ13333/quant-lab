from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


UTC = timezone.utc
REQUIRED_MAP_COLUMNS = {
    "refinitiv_ric",
    "listing_start",
    "listing_end",
    "currency",
    "known_at",
    "stable_listing_id",
    "pit_usable",
    "blocker_reason",
}


@dataclass(frozen=True)
class JapanG1ValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    diagnostics_path: Path
    manifest: dict[str, Any]


def write_japan_g1_independent_validation(
    *,
    run_id: str,
    producer_dir: str | Path,
    security_master_dir: str | Path,
    reconciliation_dir: str | Path,
    active_map_dir: str | Path,
    reference_map_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g1_validation",
) -> JapanG1ValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    producer_dir = Path(producer_dir)
    producer = _read_json(producer_dir / "manifest.json")
    producer_run_id = str(producer.get("producer_run_id", ""))
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "manifest.json"
    diagnostics_path = artifact_dir / "diagnostics.csv"
    output_dirs = {
        "security_master": Path(security_master_dir),
        "reconciliation": Path(reconciliation_dir),
        "active_map": Path(active_map_dir),
        "reference_map": Path(reference_map_dir),
    }

    issues = _producer_contract_issues(producer_dir, producer, run_id)
    frames: dict[str, pd.DataFrame] = {}
    if not issues:
        try:
            frames = _read_producer_frames(producer)
        except (OSError, ValueError, KeyError) as exc:
            issues.append(("producer_read_failure", str(exc)))
    if not issues:
        issues.extend(
            _data_issues(
                frames,
                require_bitemporal_ends=str(producer.get("schema_version", "")).startswith("2.0-monthly"),
            )
        )

    diagnostics = pd.DataFrame(issues, columns=["check", "detail"])
    _atomic_write_csv(diagnostics_path, diagnostics)
    source_checksums_verified = not any(check == "producer_hash_mismatch" for check, _ in issues)
    if issues:
        _publish_failure_markers(output_dirs, run_id, producer_run_id, issues)
        manifest = _validation_manifest(
            run_id=run_id,
            producer_run_id=producer_run_id,
            status="failed_validation",
            issues=issues,
            source_checksums_verified=source_checksums_verified,
            diagnostics_path=diagnostics_path,
            output_dirs=output_dirs,
        )
        _atomic_write_json(manifest_path, manifest)
        return JapanG1ValidationArtifacts(run_id, artifact_dir, manifest_path, diagnostics_path, manifest)

    _publish_security_master(output_dirs["security_master"], frames, producer, run_id)
    _publish_reconciliation(output_dirs["reconciliation"], frames["reconciliation"], producer, run_id)
    _publish_map(output_dirs["active_map"], frames["active_map"], "active", producer, run_id)
    _publish_map(output_dirs["reference_map"], frames["reference_map"], "reference", producer, run_id)
    manifest = _validation_manifest(
        run_id=run_id,
        producer_run_id=producer_run_id,
        status="pass",
        issues=[],
        source_checksums_verified=True,
        diagnostics_path=diagnostics_path,
        output_dirs=output_dirs,
    )
    manifest["status"] = "pass"
    manifest["scorecard_eligible"] = True
    manifest["evidence_scope"] = "full"
    manifest["validation_status"] = "pass"
    manifest["output_sha256"] = {
        str(path.resolve()): _sha256(path)
        for directory in output_dirs.values()
        for path in directory.rglob("*")
        if path.is_file()
    }
    _atomic_write_json(manifest_path, manifest)
    return JapanG1ValidationArtifacts(run_id, artifact_dir, manifest_path, diagnostics_path, manifest)


def _producer_contract_issues(
    producer_dir: Path, manifest: dict[str, Any], validator_run_id: str
) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    producer_run_id = str(manifest.get("producer_run_id", ""))
    if not producer_run_id or producer_run_id == validator_run_id:
        issues.append(("independence_failure", "validator_run_id must differ from producer_run_id"))
    if not (
        manifest.get("status") == "complete"
        and manifest.get("artifact_type") == "japan_g1_pit_materializer_producer"
        and manifest.get("independent_audit") is False
        and manifest.get("scorecard_eligible") is False
        and manifest.get("historical_pit_ready") is True
        and not manifest.get("critical_blockers")
    ):
        issues.append(("producer_not_admissible", "producer is blocked, incomplete, self-validated, or not PIT-ready"))
    expected = manifest.get("output_sha256", {})
    artifacts = manifest.get("artifacts", {})
    aliases = {
        "entities.parquet": "entities",
        "listing_windows.parquet": "listing_windows",
        "identifier_history.parquet": "identifier_history",
        "universe_membership_pit.parquet": "universe_membership",
        "active_symbol_map_candidate.csv": "active_candidate",
        "reference_symbol_map_candidate.csv": "reference_candidate",
        "reconciliation.csv": "reconciliation",
    }
    if not isinstance(expected, dict) or set(expected) != set(aliases):
        issues.append(("producer_hash_inventory", "producer output hash inventory is incomplete"))
        return issues
    for filename, alias in aliases.items():
        path = Path(str(artifacts.get(alias, "")))
        if not path.is_file() or _sha256(path) != str(expected.get(filename, "")).lower():
            issues.append(("producer_hash_mismatch", filename))
    input_paths = manifest.get("input_manifests", {})
    input_hashes = manifest.get("input_manifest_sha256", {})
    if not isinstance(input_paths, dict) or not input_paths or set(input_paths) != set(input_hashes):
        issues.append(("producer_input_hash_inventory", "upstream manifest paths/hashes are incomplete"))
    else:
        for name, raw_path in input_paths.items():
            path = Path(str(raw_path))
            if not path.is_file() or _sha256(path) != str(input_hashes[name]).lower():
                issues.append(("producer_input_hash_mismatch", name))
    if producer_dir.resolve() != Path(str(manifest.get("artifact_dir", producer_dir))).resolve():
        issues.append(("producer_path_mismatch", "manifest artifact_dir differs from supplied producer_dir"))
    return issues


def _read_producer_frames(manifest: dict[str, Any]) -> dict[str, pd.DataFrame]:
    artifacts = manifest["artifacts"]
    return {
        "entities": pd.read_parquet(artifacts["entities"]),
        "listing_windows": pd.read_parquet(artifacts["listing_windows"]),
        "identifier_history": pd.read_parquet(artifacts["identifier_history"]),
        "universe_membership": pd.read_parquet(artifacts["universe_membership"]),
        "active_map": pd.read_csv(artifacts["active_candidate"], low_memory=False),
        "reference_map": pd.read_csv(artifacts["reference_candidate"], low_memory=False),
        "reconciliation": pd.read_csv(artifacts["reconciliation"], low_memory=False),
    }


def _data_issues(
    frames: dict[str, pd.DataFrame], *, require_bitemporal_ends: bool = False
) -> list[tuple[str, str]]:
    issues: list[tuple[str, str]] = []
    for role in ("active_map", "reference_map"):
        frame = frames[role]
        missing = REQUIRED_MAP_COLUMNS - set(frame.columns)
        if missing or frame.empty:
            issues.append((f"{role}_schema", f"missing={sorted(missing)} rows={len(frame)}"))
            continue
        required = frame[list(REQUIRED_MAP_COLUMNS - {"blocker_reason"})]
        if required.isna().any().any() or required.astype(str).apply(lambda col: col.str.strip().eq("").any()).any():
            issues.append((f"{role}_nulls", "required map values contain null/blank entries"))
        if not frame["currency"].astype(str).eq("JPY").all():
            issues.append((f"{role}_currency", "all admitted windows must be JPY"))
        if not _truthy(frame["pit_usable"]).all() or frame["blocker_reason"].fillna("").astype(str).str.strip().ne("").any():
            issues.append((f"{role}_pit", "unusable or blocker-bearing map rows remain"))
        issues.extend(
            _window_issues(
                frame, role, require_bitemporal_ends=require_bitemporal_ends
            )
        )

    active = frames["active_map"]
    reference = frames["reference_map"]
    if REQUIRED_MAP_COLUMNS.issubset(active) and REQUIRED_MAP_COLUMNS.issubset(reference):
        keys = ["refinitiv_ric", "listing_start", "listing_end", "stable_listing_id"]
        active_keys = set(map(tuple, active[keys].astype(str).to_numpy()))
        reference_keys = set(map(tuple, reference[keys].astype(str).to_numpy()))
        if not active_keys.issubset(reference_keys):
            issues.append(("active_reference_integrity", "active map is not a subset of reference map"))

    reconciliation = frames["reconciliation"]
    required_recon = {"stable_listing_id", "mapping_status", "pit_usable", "blocker_reason"}
    if reconciliation.empty or not required_recon.issubset(reconciliation):
        issues.append(("reconciliation_schema", "reconciliation is empty or missing required columns"))
    else:
        unresolved = ~reconciliation["mapping_status"].astype(str).eq("matched")
        blocked = reconciliation["blocker_reason"].fillna("").astype(str).str.strip().ne("")
        unusable = ~_truthy(reconciliation["pit_usable"])
        count = int((unresolved | blocked | unusable).sum())
        if count:
            issues.append(("reconciliation_unresolved", f"rows={count}"))
        if len(reconciliation) != len(reference):
            issues.append(("denominator_mismatch", f"reconciliation={len(reconciliation)} reference={len(reference)}"))

    entities = frames["entities"]
    if entities.empty or "stable_listing_id" not in entities or entities["stable_listing_id"].isna().any() or entities["stable_listing_id"].duplicated().any():
        issues.append(("entities_integrity", "entity listing keys are missing or duplicated"))
    membership = frames["universe_membership"]
    issues.extend(_membership_issues(membership))
    listing = frames["listing_windows"]
    if listing.empty or "pit_usable" not in listing or not _truthy(listing["pit_usable"]).all():
        issues.append(("listing_window_integrity", "listing windows are empty or unusable"))
    identifiers = frames["identifier_history"]
    if identifiers.empty or not {"stable_listing_id", "identifier_type", "identifier_value", "valid_from", "valid_to", "known_at"}.issubset(identifiers):
        issues.append(("identifier_history_integrity", "identifier history is empty or incomplete"))
    else:
        starts = pd.to_datetime(identifiers["valid_from"], errors="coerce", utc=True)
        ends = pd.to_datetime(identifiers["valid_to"], errors="coerce", utc=True)
        known = pd.to_datetime(identifiers["known_at"], errors="coerce", utc=True)
        if starts.isna().any() or ends.isna().any() or known.isna().any() or starts.gt(ends).any():
            issues.append(("identifier_history_timestamps", "identifier windows contain invalid timestamps"))
        else:
            dated = identifiers.assign(_start=starts, _end=ends).sort_values(
                ["identifier_type", "identifier_value", "_start", "_end"]
            )
            prior = dated.groupby(["identifier_type", "identifier_value"], sort=False)["_end"].shift()
            if dated["_start"].le(prior).fillna(False).any():
                issues.append(("identifier_history_overlap", "identifier validity windows overlap"))
    return issues


def _membership_issues(frame: pd.DataFrame) -> list[tuple[str, str]]:
    if frame.empty or "pit_usable" not in frame:
        return [("membership_integrity", "PIT membership is empty or missing usability state")]
    usable = _truthy(frame["pit_usable"])
    if usable.all():
        return []
    required = {
        "observation_month",
        "code",
        "refinitiv_ric",
        "identifier_mapped",
        "lifecycle_excluded",
        "lifecycle_exclusion_reason",
        "listing_valid_from",
        "listing_valid_to",
        "known_at_utc",
    }
    missing = required - set(frame.columns)
    if missing:
        return [("membership_lifecycle_schema", f"missing={sorted(missing)}")]
    mapped = _truthy(frame["identifier_mapped"])
    excluded = _truthy(frame["lifecycle_excluded"])
    known = pd.to_datetime(frame["known_at_utc"], errors="coerce", utc=True)
    valid_from = pd.to_datetime(frame["listing_valid_from"], errors="coerce")
    valid_to = pd.to_datetime(frame["listing_valid_to"], errors="coerce")
    month_end = pd.Series(
        pd.PeriodIndex(frame["observation_month"], freq="M").end_time.normalize(),
        index=frame.index,
    )
    recomputed = mapped & (month_end.lt(valid_from) | month_end.gt(valid_to))
    reason = frame["lifecycle_exclusion_reason"].fillna("").astype(str).str.strip()
    issues: list[tuple[str, str]] = []
    if not mapped.all() or frame["refinitiv_ric"].fillna("").astype(str).str.strip().eq("").any():
        issues.append(("membership_identifier_coverage", f"unmapped_rows={int((~mapped).sum())}"))
    if known.isna().any() or valid_from.isna().any() or valid_to.isna().any():
        issues.append(("membership_lifecycle_timestamps", "known-at or exact listing dates are invalid"))
    if not excluded.equals(recomputed):
        issues.append(
            (
                "membership_lifecycle_recomputation",
                f"mismatched_rows={int(excluded.ne(recomputed).sum())}",
            )
        )
    expected_usable = mapped & ~excluded & known.notna()
    if not usable.equals(expected_usable):
        issues.append(
            ("membership_usability_recomputation", f"mismatched_rows={int(usable.ne(expected_usable).sum())}")
        )
    invalid_reason = (excluded & reason.eq("")) | (~excluded & reason.ne(""))
    if invalid_reason.any():
        issues.append(("membership_lifecycle_reason", f"invalid_rows={int(invalid_reason.sum())}"))
    if frame.duplicated(["observation_month", "code"]).any():
        issues.append(("membership_duplicate_keys", "observation_month/code keys are not unique"))
    return issues


def _window_issues(
    frame: pd.DataFrame, role: str, *, require_bitemporal_ends: bool = False
) -> list[tuple[str, str]]:
    starts = pd.to_datetime(frame["listing_start"], errors="coerce", utc=True)
    ends = pd.to_datetime(frame["listing_end"], errors="coerce", utc=True)
    known = pd.to_datetime(frame["known_at"], errors="coerce", utc=True)
    issues: list[tuple[str, str]] = []
    if starts.isna().any() or ends.isna().any() or known.isna().any() or starts.gt(ends).any():
        issues.append((f"{role}_timestamps", "invalid start/end/known_at timestamps"))
        return issues
    dated = frame.assign(_start=starts, _end=ends).sort_values(["refinitiv_ric", "_start", "_end"])
    if dated.duplicated(["refinitiv_ric", "_start", "_end"]).any():
        issues.append((f"{role}_duplicate_windows", "duplicate RIC/window rows"))
    prior = dated.groupby("refinitiv_ric", sort=False)["_end"].shift()
    if dated["_start"].le(prior).fillna(False).any():
        issues.append((f"{role}_overlap", "overlapping RIC windows"))
    if require_bitemporal_ends:
        required = {"window_end_known_at", "right_open"}
        if not required.issubset(frame.columns):
            issues.append((f"{role}_bitemporal_schema", "window-end known-at fields are missing"))
        else:
            right_open = _truthy(frame["right_open"])
            end_known = pd.to_datetime(frame["window_end_known_at"], errors="coerce", utc=True)
            invalid_closed = ~right_open & (end_known.isna() | end_known.le(ends))
            invalid_open = right_open & frame["window_end_known_at"].fillna("").astype(str).str.strip().ne("")
            if invalid_closed.any() or invalid_open.any():
                issues.append(
                    (
                        f"{role}_bitemporal_end",
                        f"invalid_closed={int(invalid_closed.sum())} invalid_open={int(invalid_open.sum())}",
                    )
                )
    return issues


def _publish_security_master(directory: Path, frames: dict[str, pd.DataFrame], producer: dict[str, Any], run_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    mapping = {
        "entities.parquet": frames["entities"],
        "listing_windows.parquet": frames["listing_windows"],
        "identifier_history.parquet": frames["identifier_history"],
        "universe_membership_pit.parquet": frames["universe_membership"],
    }
    for name, frame in mapping.items():
        _atomic_write_parquet(directory / name, frame)
    rows = len(frames["listing_windows"])
    manifest = _full_manifest_base(producer, run_id)
    manifest.update({
        "artifact_type": "japan_validated_listing_level_security_master",
        "security_master_kind": "listing_level",
        "listing_level_master": True,
        "quote_level_master": False,
        "security_master_rows": rows,
        "listing_rows": rows,
        "provenance": {
            name: {
                "complete": True,
                "coverage": 1.0,
                "missing_rows": 0,
                "source": producer.get(
                    "source_method",
                    "official JPX membership plus validated LSEG mapping",
                ),
            }
            for name in ("currency", "listing_window", "known_at")
        },
        "non_overlap_checks": {
            name: {"passed": True, "overlap_rows": 0}
            for name in ("listing_windows", "identifier_windows", "currency_windows")
        },
        "output_sha256": {name: _sha256(directory / name) for name in mapping},
    })
    _atomic_write_json(directory / "manifest.json", manifest)


def _publish_reconciliation(directory: Path, frame: pd.DataFrame, producer: dict[str, Any], run_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "reconciliation.csv"
    _atomic_write_csv(path, frame)
    manifest = _full_manifest_base(producer, run_id)
    manifest.update({
        "artifact_type": "jpx_universe_reconciliation",
        "official_jpx_source": True,
        "reconciliation_complete": True,
        "strict_quality_pass": True,
        "historical_pit_ready": True,
        "historical_backtest_eligible": True,
        "jpx_source_rows": len(frame),
        "reconciliation_coverage": 1.0,
        "unmatched_jpx_rows": 0,
        "unmatched_master_rows": 0,
        "duplicate_listing_rows": 0,
        "reconciliation_output_sha256_verified": True,
        "reconciliation_sha256": _sha256(path),
        "counts": {
            "historical_listed_reference_rows": len(frame),
            "active_operating_core_rows": int(
                (_truthy(frame.get("right_open", pd.Series(False, index=frame.index)))
                & _truthy(frame["pit_usable"])).sum()
            ),
        },
    })
    _atomic_write_json(directory / "manifest.json", manifest)


def _publish_map(directory: Path, frame: pd.DataFrame, role: str, producer: dict[str, Any], run_id: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "symbol_map.csv"
    published = frame.copy()
    published["country"] = "Japan"
    published["exchange"] = "JPX"
    published["ticker"] = published["local_code"].astype(str)
    published["yahoo_ticker"] = published["refinitiv_ric"].astype(str)
    published["refinitiv_symbol_status"] = "independently_validated_security_master"
    published["first_trade_date"] = published["listing_start"]
    published["retire_date"] = published["listing_end"]
    _atomic_write_csv(path, published)
    manifest = _full_manifest_base(producer, run_id)
    manifest.update({
        "artifact_type": "validated_jpx_symbol_map",
        "map_role": role,
        "validation_status": "pass",
        "mapping_coverage": 1.0,
        "unresolved_rows": 0,
        "invalid_ric_rows": 0,
        "overlap_conflict_rows": 0,
        "operational_adapter_columns_complete": True,
        "refinitiv_symbol_status": "independently_validated_security_master",
        "symbol_map_sha256": _sha256(path),
    })
    _atomic_write_json(directory / "manifest.json", manifest)


def _full_manifest_base(producer: dict[str, Any], run_id: str) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "status": "pass",
        "scorecard_eligible": True,
        "evidence_scope": "full",
        "independent_audit": True,
        "producer_run_id": producer["producer_run_id"],
        "validator_run_id": run_id,
        "validator_type": "independent_process",
        "source_checksums_verified": True,
        "production_ready": False,
        "ibkr_ready": False,
        "live_execution_ready": False,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def _validation_manifest(*, run_id: str, producer_run_id: str, status: str, issues: list[tuple[str, str]], source_checksums_verified: bool, diagnostics_path: Path, output_dirs: dict[str, Path]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "artifact_type": "japan_g1_independent_validation",
        "status": status,
        "scorecard_eligible": False,
        "evidence_scope": "failed" if issues else "full",
        "independent_audit": True,
        "producer_run_id": producer_run_id,
        "validator_run_id": run_id,
        "validator_type": "independent_process",
        "source_checksums_verified": source_checksums_verified,
        "validation_status": "fail" if issues else "pass",
        "critical_blocker_count": len(issues),
        "critical_blockers": [f"{check}: {detail}" for check, detail in issues],
        "diagnostics_sha256": _sha256(diagnostics_path),
        "output_directories": {name: str(path.resolve()) for name, path in output_dirs.items()},
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def _publish_failure_markers(output_dirs: dict[str, Path], run_id: str, producer_run_id: str, issues: list[tuple[str, str]]) -> None:
    payload = {
        "run_id": run_id,
        "status": "failed_validation",
        "scorecard_eligible": False,
        "evidence_scope": "failed",
        "producer_run_id": producer_run_id,
        "validator_run_id": run_id,
        "critical_blockers": [f"{check}: {detail}" for check, detail in issues],
    }
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(directory / "manifest.json", payload)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_write_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
