from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


UTC = timezone.utc
PROMOTION_POLICY = {
    "alpha": False,
    "historical_backtest": False,
    "production": False,
    "ibkr": False,
    "live_trading": False,
}


@dataclass(frozen=True)
class JapanG1SuccessorVendorPlanArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    plan_path: Path
    manifest: dict[str, Any]


def write_japan_g1_successor_vendor_plan(
    *,
    run_id: str,
    transition_evidence_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g1_successor_vendor_plan",
    expected_lookup_rows: int = 12,
) -> JapanG1SuccessorVendorPlanArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    source = Path(transition_evidence_dir).resolve()
    source_manifest_path = source / "manifest.json"
    source_manifest = _read_json(source_manifest_path)
    candidates_path = source / "admitted_transition_candidates.csv"
    expected_hash = source_manifest.get("output_sha256", {}).get(candidates_path.name, "")
    counts = source_manifest.get("counts", {})
    if not (
        source_manifest.get("artifact_type")
        == "japan_g1_official_corporate_transition_evidence_candidate"
        and source_manifest.get("status") == "complete_candidate"
        and source_manifest.get("denominator_valid") is True
        and source_manifest.get("scorecard_eligible") is False
        and source_manifest.get("independent_validation_required") is True
        and int(counts.get("transition_denominator", 0)) == 66
        and int(counts.get("event_segments_admitted_candidate", 0)) == 66
        and int(counts.get("unresolved_candidates", -1)) == 0
        and candidates_path.is_file()
        and expected_hash
        and _sha256(candidates_path) == expected_hash
    ):
        raise ValueError("corporate-transition evidence contract invalid")

    candidates = pd.read_csv(candidates_path, dtype=str).fillna("")
    admitted = candidates["event_segment_admitted"].str.lower().isin({"true", "1", "yes"})
    missing_ric = candidates["successor_primary_rics"].str.strip().eq("")
    plan = candidates.loc[admitted & missing_ric].copy()
    if len(plan) != expected_lookup_rows:
        raise ValueError(
            f"successor vendor lookup denominator mismatch: {len(plan)} != {expected_lookup_rows}"
        )
    if plan["successor_code"].duplicated().any() or not plan["successor_code"].map(
        lambda value: bool(re.fullmatch(r"[0-9A-Z]{4}", value))
    ).all():
        raise ValueError("successor vendor plan contains duplicate or malformed codes")
    if not plan["evidence_class"].isin({"public_event_only", "tpm_official"}).all():
        raise ValueError("successor vendor plan contains an unexpected evidence class")

    plan = plan.rename(columns={"successor_code": "local_code"})
    keep = [
        "local_code",
        "old_code",
        "evidence_class",
        "successor_issue_name",
        "official_listing_date",
        "listing_source_url",
        "old_event_date",
        "old_event_source_url",
        "name_match_basis",
        "gap_days",
    ]
    plan = plan[keep].sort_values("local_code").reset_index(drop=True)
    plan.insert(
        0,
        "vendor_plan_row_id",
        [
            "JG1SV-" + hashlib.sha256(
                f"{row.old_code}|{row.local_code}|{row.official_listing_date}".encode()
            ).hexdigest()[:16]
            for row in plan.itertuples(index=False)
        ],
    )
    plan["requested_country_code"] = "JPN"
    plan["requested_state_buckets"] = "active|inactive"
    plan["lookup_status"] = "planned_not_executed"

    root = (Path(artifact_root) / run_id).resolve()
    if root == source or _is_relative_to(root, source):
        raise ValueError("output must be separate from immutable source evidence")
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"single-writer lock exists: {lock}") from exc
    try:
        plan_path = root / "targeted_successor_code_plan.csv"
        _atomic_csv(plan_path, plan)
        manifest = {
            "schema_version": "1.0-official-event-backed-vendor-plan",
            "run_id": run_id,
            "artifact_type": "japan_g1_successor_vendor_lookup_plan",
            "status": "planned_not_executed",
            "evidence_scope": "official_event_backed_successor_codes_without_quote_chain_rows",
            "scorecard_eligible": False,
            "independent_validation_required": True,
            "refinitiv_called": False,
            "counts": {
                "transition_denominator": 66,
                "successor_vendor_lookup_rows": len(plan),
                "public_event_only_rows": int(plan["evidence_class"].eq("public_event_only").sum()),
                "tpm_rows": int(plan["evidence_class"].eq("tpm_official").sum()),
            },
            "transition_evidence_manifest": str(source_manifest_path),
            "transition_evidence_manifest_sha256": _sha256(source_manifest_path),
            "transition_candidates_sha256": _sha256(candidates_path),
            "artifacts": {"targeted_successor_code_plan": str(plan_path)},
            "output_sha256": {plan_path.name: _sha256(plan_path)},
            "critical_blockers": [
                "bounded_refinitiv_successor_lookup_not_executed",
                "independent_transition_validation_not_executed",
            ],
            "promotion_policy": PROMOTION_POLICY,
            "generated_at_utc": datetime.now(UTC).isoformat(),
        }
        manifest_path = root / "manifest.json"
        _atomic_json(manifest_path, manifest)
        return JapanG1SuccessorVendorPlanArtifacts(
            run_id, root, manifest_path, plan_path, manifest
        )
    finally:
        lock.rmdir()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_bytes(path, frame.to_csv(index=False).encode("utf-8"))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
