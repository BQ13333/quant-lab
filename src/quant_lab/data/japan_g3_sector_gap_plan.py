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

from quant_lab.data.japan_g3_sector_snapshot import SECTOR_FIELDS


@dataclass(frozen=True)
class JapanG3SectorGapPlanResult:
    artifact_dir: Path
    missing_rics_path: Path
    plan_path: Path
    manifest_path: Path
    missing_rics: pd.DataFrame
    plan: pd.DataFrame
    manifest: dict[str, Any]


def build_japan_g3_sector_gap_plan(
    *,
    pit_plan_dir: str | Path,
    existing_sector_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
    batch_size: int = 25,
) -> JapanG3SectorGapPlanResult:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    plan_dir = Path(pit_plan_dir)
    sector_dir = Path(existing_sector_dir)
    selected_path = plan_dir / "selected_universe.csv"
    plan_manifest_path = plan_dir / "manifest.json"
    profile_path = sector_dir / "sector_routing_profile.csv"
    sector_manifest_path = sector_dir / "manifest.json"
    for path in (selected_path, plan_manifest_path, profile_path, sector_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    plan_manifest = _read_json(plan_manifest_path)
    if not (
        plan_manifest.get("status") == "complete"
        and plan_manifest.get("scorecard_eligible") is True
        and plan_manifest.get("g1_independent_validation_admitted") is True
        and plan_manifest.get("point_in_time_plan") is True
    ):
        raise ValueError("G3 PIT plan is not admitted full-denominator evidence")
    selected = pd.read_csv(selected_path, low_memory=False)
    selected_rics = set(selected["refinitiv_ric"].dropna().astype(str))
    existing = pd.read_csv(profile_path, low_memory=False)
    existing_rics = set(existing.get("ric", pd.Series(dtype=str)).dropna().astype(str))
    outside = existing_rics - selected_rics
    if outside:
        raise ValueError(f"existing sector profile has {len(outside)} RICs outside the PIT plan")
    missing = sorted(selected_rics - existing_rics)
    missing_frame = pd.DataFrame({"refinitiv_ric": missing})
    rows: list[dict[str, Any]] = []
    fields = ",".join(field for _, field in SECTOR_FIELDS)
    for offset in range(0, len(missing), batch_size):
        chunk = missing[offset : offset + batch_size]
        rows.append(
            {
                "plan_row_id": f"jpn-g3-sector-gap-{offset // batch_size + 1:04d}",
                "dataset_id": "fundamentals_sector_routing",
                "country": "Japan",
                "batch_number": offset // batch_size + 1,
                "ric_count": len(chunk),
                "ric_list": ",".join(chunk),
                "source_fields": fields,
                "status": "queued_after_g2_vendor_health",
            }
        )
    gap_plan = pd.DataFrame(rows)
    if missing and gap_plan.empty:
        raise ValueError("sector gap plan is empty despite missing RICs")

    output_dir = Path(artifact_root) / run_id
    if output_dir.exists():
        raise FileExistsError(f"immutable sector gap plan already exists: {output_dir}")
    root = Path(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=root))
    try:
        missing_path = temporary / "missing_rics.csv"
        gap_plan_path = temporary / "sector_gap_plan.csv"
        missing_frame.to_csv(missing_path, index=False)
        gap_plan.to_csv(gap_plan_path, index=False)
        manifest = {
            "run_id": run_id,
            "artifact_type": "japan_g3_sector_gap_plan",
            "status": "complete_plan",
            "evidence_scope": "full_denominator_gap_plan",
            "scorecard_eligible": False,
            "research_only": True,
            "production_ready": False,
            "live_execution_ready": False,
            "ibkr_ready": False,
            "selected_rics": len(selected_rics),
            "existing_profile_rics": len(existing_rics),
            "missing_rics": len(missing),
            "plan_rows": len(gap_plan),
            "batch_size": batch_size,
            "denominator_reconciles": len(existing_rics) + len(missing) == len(selected_rics),
            "existing_rics_outside_plan": 0,
            "input_sha256": {
                str(selected_path.resolve()): _sha256(selected_path),
                str(plan_manifest_path.resolve()): _sha256(plan_manifest_path),
                str(profile_path.resolve()): _sha256(profile_path),
                str(sector_manifest_path.resolve()): _sha256(sector_manifest_path),
            },
            "artifact_sha256": {
                "missing_rics": _sha256(missing_path),
                "sector_gap_plan": _sha256(gap_plan_path),
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return JapanG3SectorGapPlanResult(
        artifact_dir=output_dir,
        missing_rics_path=output_dir / "missing_rics.csv",
        plan_path=output_dir / "sector_gap_plan.csv",
        manifest_path=output_dir / "manifest.json",
        missing_rics=missing_frame,
        plan=gap_plan,
        manifest=manifest,
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
