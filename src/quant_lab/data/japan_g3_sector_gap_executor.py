from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.japan_g3_sector_snapshot import (
    SECTOR_FIELDS,
    build_sector_routing_profile,
)
from quant_lab.data.vendor_adapters import (
    RefinitivRetryableProviderError,
    is_refinitiv_retryable_provider_error,
)


@dataclass(frozen=True)
class JapanG3SectorGapExecutionResult:
    artifact_dir: Path
    manifest_path: Path
    row_manifest_path: Path
    profile_path: Path
    manifest: dict[str, Any]
    row_manifest: pd.DataFrame
    profile: pd.DataFrame


def execute_japan_g3_sector_gap_plan(
    client: Any,
    *,
    gap_plan_dir: str | Path,
    expected_plan_sha256: str,
    baseline_sector_dir: str | Path,
    g1_validation_dir: str | Path,
    g2_admission_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
    max_plan_rows: int = 1,
) -> JapanG3SectorGapExecutionResult:
    if max_plan_rows <= 0:
        raise ValueError("max_plan_rows must be positive")
    gap_dir = Path(gap_plan_dir)
    plan_path = gap_dir / "sector_gap_plan.csv"
    gap_manifest_path = gap_dir / "manifest.json"
    baseline_path = Path(baseline_sector_dir) / "sector_routing_profile.csv"
    for path in (plan_path, gap_manifest_path, baseline_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if _sha256(plan_path) != expected_plan_sha256.lower():
        raise ValueError("Japan G3 sector-gap plan SHA-256 mismatch")
    gap_manifest = _read_json(gap_manifest_path)
    plan = pd.read_csv(plan_path, low_memory=False)
    if not (
        gap_manifest.get("status") == "complete_plan"
        and int(gap_manifest.get("selected_rics", 0)) > 0
        and int(gap_manifest.get("missing_rics", -1)) >= 0
        and int(gap_manifest.get("plan_rows", 0)) == len(plan)
        and gap_manifest.get("denominator_reconciles") is True
    ):
        raise ValueError("Japan G3 sector-gap denominator is not admitted")
    if not _g1_admitted(_read_json(Path(g1_validation_dir) / "manifest.json")):
        raise PermissionError("independent Japan G1 admission is required")
    if not _g2_admitted(_read_json(Path(g2_admission_dir) / "manifest.json")):
        raise PermissionError("strict Japan G2 aggregate admission is required")

    baseline = pd.read_csv(baseline_path, low_memory=False)
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    staged_plan = artifact_dir / "sector_gap_plan.csv"
    if staged_plan.exists() and staged_plan.read_bytes() != plan_path.read_bytes():
        raise ValueError("staged Japan G3 sector-gap plan drift")
    if not staged_plan.exists():
        staged_plan.write_bytes(plan_path.read_bytes())
    lock = artifact_dir / ".writer.lock"
    try:
        os.mkdir(lock)
    except FileExistsError as exc:
        raise RuntimeError(f"Japan G3 sector writer lock exists: {lock}") from exc
    row_manifest_path = artifact_dir / "row_manifest.csv"
    profile_path = artifact_dir / "sector_routing_profile.csv"
    manifest_path = artifact_dir / "manifest.json"
    columns = [
        "plan_row_id", "status", "requested_rics", "profile_rics", "output_path",
        "started_at_utc", "completed_at_utc", "error_type", "error",
    ]
    existing = (
        pd.read_csv(row_manifest_path, low_memory=False)
        if row_manifest_path.exists()
        else pd.DataFrame(columns=columns)
    )
    completed_ids = set(
        existing.loc[existing["status"].eq("completed"), "plan_row_id"].astype(str)
    )
    pending = plan.loc[~plan["plan_row_id"].astype(str).isin(completed_ids)].head(max_plan_rows)
    records = existing.to_dict("records")
    provider_halt: dict[str, str] | None = None
    try:
        for row in pending.to_dict("records"):
            row_id = str(row["plan_row_id"])
            rics = [item for item in str(row["ric_list"]).split(",") if item]
            started = datetime.now(timezone.utc).isoformat()
            output = artifact_dir / "batches" / f"{row_id}.parquet"
            try:
                frame = client.get_data(
                    universe=rics,
                    fields=[field for _, field in SECTOR_FIELDS],
                )
                profile = build_sector_routing_profile(frame)
                returned = set(profile.get("ric", pd.Series(dtype=str)).astype(str))
                complete = returned == set(rics)
                if not profile.empty:
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temp = output.with_suffix(".tmp.parquet")
                    profile.to_parquet(temp, index=False)
                    os.replace(temp, output)
                result = {
                    "plan_row_id": row_id,
                    "status": "completed" if complete else "failed_partial_coverage",
                    "requested_rics": len(rics),
                    "profile_rics": len(returned),
                    "output_path": str(output.resolve()) if not profile.empty else "",
                    "started_at_utc": started,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": "" if complete else "PartialCoverage",
                    "error": "" if complete else f"returned {len(returned)}/{len(rics)} RICs",
                }
            except Exception as exc:  # pragma: no cover - live provider path
                retryable = isinstance(exc, RefinitivRetryableProviderError) or is_refinitiv_retryable_provider_error(exc)
                result = {
                    "plan_row_id": row_id,
                    "status": "failed_retryable_provider" if retryable else "failed",
                    "requested_rics": len(rics),
                    "profile_rics": 0,
                    "output_path": "",
                    "started_at_utc": started,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                if retryable:
                    provider_halt = {
                        "plan_row_id": row_id,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
            records = [item for item in records if str(item.get("plan_row_id")) != row_id]
            records.append(result)
            _atomic_csv(pd.DataFrame(records, columns=columns), row_manifest_path)
            if provider_halt is not None:
                break
    finally:
        lock.rmdir()

    row_manifest = pd.DataFrame(records, columns=columns)
    completed = row_manifest.loc[row_manifest["status"].eq("completed")]
    parts = [baseline]
    for raw_path in completed["output_path"].dropna().astype(str):
        path = Path(raw_path)
        if path.is_file():
            parts.append(pd.read_parquet(path))
    combined = pd.concat(parts, ignore_index=True, sort=False).drop_duplicates("ric", keep="last")
    combined = combined.sort_values("ric").reset_index(drop=True)
    _atomic_csv(combined, profile_path)
    failed_rows = int(row_manifest["status"].astype(str).str.startswith("failed").sum())
    complete = bool(
        len(completed) == len(plan)
        and failed_rows == 0
        and combined["ric"].nunique() == int(gap_manifest["selected_rics"])
    )
    manifest = {
        "run_id": run_id,
        "status": "complete_current_routing" if complete else "completed_partial",
        "scope": "full_denominator_current_sector_routing_only",
        "evidence_scope": "current_snapshot",
        "scorecard_eligible": False,
        "partial": not complete,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "requested_rics": int(gap_manifest["selected_rics"]),
        "profile_rics": int(combined["ric"].nunique()),
        "coverage": float(combined["ric"].nunique() / int(gap_manifest["selected_rics"])),
        "planned_rows": len(plan),
        "completed_rows": len(completed),
        "failed_rows": failed_rows,
        "halted_early": provider_halt is not None,
        "provider_halt": provider_halt,
        "point_in_time_sector_claimed": False,
        "plan_sha256": _sha256(plan_path),
        "baseline_sha256": _sha256(baseline_path),
        "artifact_sha256": {
            "row_manifest": _sha256(row_manifest_path),
            "sector_routing_profile": _sha256(profile_path),
        },
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return JapanG3SectorGapExecutionResult(
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        row_manifest_path=row_manifest_path,
        profile_path=profile_path,
        manifest=manifest,
        row_manifest=row_manifest,
        profile=combined,
    )


def _g1_admitted(manifest: dict[str, Any]) -> bool:
    independent = manifest.get("independent_validation") is True or bool(
        manifest.get("independent_audit") is True
        and manifest.get("validator_type") == "independent_process"
    )
    return bool(
        manifest.get("status") in {"pass", "complete"}
        and manifest.get("scorecard_eligible") is True
        and manifest.get("evidence_scope") == "full"
        and independent
        and manifest.get("source_checksums_verified") is True
        and int(manifest.get("critical_blocker_count", 0) or 0) == 0
    )


def _g2_admitted(manifest: dict[str, Any]) -> bool:
    return bool(
        manifest.get("status") == "pass"
        and manifest.get("admission_passed") is True
        and manifest.get("g2_complete") is True
        and manifest.get("scorecard_eligible") is True
        and manifest.get("source_checksums_verified") is True
        and int(manifest.get("critical_blocker_count", -1)) == 0
        and int(manifest.get("check_count", -1)) == int(manifest.get("passed_check_count", -2))
    )


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


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)
