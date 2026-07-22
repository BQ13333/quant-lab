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

from quant_lab.data.corporate_actions_pit import build_corporate_action_event_plan


JAPAN_LIQUIDITY_PROBE_FIELDS: tuple[tuple[str, str], ...] = (
    ("market_cap", "TR.CompanyMarketCap"),
    ("free_float", "TR.FreeFloatPct"),
    ("shares_outstanding", "TR.SharesOutstanding"),
)


@dataclass(frozen=True)
class JapanG2VendorPlanResult:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    selected_universe_path: Path
    corporate_action_plan_path: Path
    liquidity_probe_queue_path: Path
    manifest: dict[str, Any]


def build_japan_g2_vendor_plan(
    *,
    active_symbol_map_path: str | Path,
    reference_symbol_map_path: str | Path,
    artifact_root: str | Path,
    run_id: str,
    start_date: str = "2006-01-01",
    end_date: str = "2026-07-06",
    batch_size: int = 25,
    corporate_action_artifact_root: str | Path = "artifacts/refinitiv_corporate_actions_events",
    corporate_action_run_id: str = "REFINITIV-JPN-G2-CORPACT-20060101-20260706-V1",
) -> JapanG2VendorPlanResult:
    active_path = Path(active_symbol_map_path)
    reference_path = Path(reference_symbol_map_path)
    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists():
        raise FileExistsError(f"immutable Japan G2 vendor plan already exists: {artifact_dir}")
    if not active_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("validated active and reference symbol maps are required")

    active = pd.read_csv(active_path, low_memory=False)
    reference = pd.read_csv(reference_path, low_memory=False)
    selected = _selected_universe(active, reference)
    if selected.empty:
        raise ValueError("validated Japan G2 vendor universe is empty")
    corporate_plan = build_corporate_action_event_plan(
        selected_universe=selected,
        artifact_root=corporate_action_artifact_root,
        run_id=corporate_action_run_id,
        start_date=start_date,
        end_date=end_date,
        tier="u2",
        batch_size=batch_size,
    )
    liquidity_probe = _liquidity_probe_queue(selected, start_date, end_date)

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=artifact_dir.parent))
    try:
        selected_path = temporary / "selected_universe.csv"
        corporate_path = temporary / "corporate_action_plan.csv"
        liquidity_path = temporary / "liquidity_probe_queue.csv"
        field_probe_path = temporary / "field_probe_queue.csv"
        requirements_path = temporary / "dataset_requirements.csv"
        batches_path = temporary / "extraction_batches.csv"
        selected.to_csv(selected_path, index=False)
        corporate_plan.to_csv(corporate_path, index=False)
        liquidity_probe.to_csv(liquidity_path, index=False)
        liquidity_probe.to_csv(field_probe_path, index=False)
        pd.DataFrame(
            [
                {
                    "dataset_id": "liquidity_market_cap_free_float",
                    "owner_agent": "liquidity_adv_agent",
                    "source_family": "refinitiv_reference_history_plus_ohlcv",
                    "cadence": "monthly_history",
                    "point_in_time_key": "asof_date,source_loaded_at_utc",
                }
            ]
        ).to_csv(requirements_path, index=False)
        pd.DataFrame(
            columns=["batch_id", "dataset_id", "country", "tier", "ric_list", "start_date", "end_date"]
        ).to_csv(batches_path, index=False)
        output_hashes = {
            path.name: _sha256(path)
            for path in (
                selected_path,
                corporate_path,
                liquidity_path,
                field_probe_path,
                requirements_path,
                batches_path,
            )
        }
        manifest = {
            "run_id": run_id,
            "artifact_type": "japan_g2_vendor_execution_plan",
            "status": "planning_complete_vendor_execution_blocked",
            "planning_complete": True,
            "vendor_execution_performed": False,
            "promotion_eligible": False,
            "scorecard_eligible": False,
            "country": "Japan",
            "exchange": "JPX",
            "start_date": start_date,
            "end_date": end_date,
            "unique_reference_rics": len(selected),
            "active_rics": int(selected["is_active_role"].sum()),
            "reference_only_rics": int((~selected["is_active_role"]).sum()),
            "corporate_action_plan_rows": len(corporate_plan),
            "corporate_action_run_id": corporate_action_run_id,
            "corporate_action_artifact_root": str(Path(corporate_action_artifact_root)),
            "corporate_action_event_families": sorted(corporate_plan["event_family"].unique()),
            "liquidity_probe_rows": len(liquidity_probe),
            "liquidity_history_plan_ready": False,
            "critical_blockers": [
                "fresh Refinitiv health evidence is required before corporate-action extraction",
                "Japan-specific liquidity history fields must pass bounded route probes before historical planning",
            ],
            "input_sha256": {
                str(active_path.resolve()): _sha256(active_path),
                str(reference_path.resolve()): _sha256(reference_path),
            },
            "output_sha256": output_hashes,
            "artifacts": {
                "selected_universe": "selected_universe.csv",
                "corporate_action_plan": "corporate_action_plan.csv",
                "liquidity_probe_queue": "liquidity_probe_queue.csv",
                "field_probe_queue": "field_probe_queue.csv",
                "dataset_requirements": "dataset_requirements.csv",
                "extraction_batches": "extraction_batches.csv",
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temporary, artifact_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return JapanG2VendorPlanResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=artifact_dir / "manifest.json",
        selected_universe_path=artifact_dir / "selected_universe.csv",
        corporate_action_plan_path=artifact_dir / "corporate_action_plan.csv",
        liquidity_probe_queue_path=artifact_dir / "liquidity_probe_queue.csv",
        manifest=manifest,
    )


def _selected_universe(active: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    required = {"refinitiv_ric", "country", "exchange", "pit_usable"}
    for name, frame in (("active", active), ("reference", reference)):
        if not required.issubset(frame.columns):
            raise ValueError(f"{name} symbol map is missing columns: {sorted(required - set(frame.columns))}")
        if not frame["pit_usable"].map(_truthy).all():
            raise ValueError(f"{name} symbol map contains non-PIT-usable rows")
    active_rics = set(active["refinitiv_ric"].dropna().astype(str))
    rows: list[dict[str, Any]] = []
    for ric, group in reference.groupby(reference["refinitiv_ric"].astype(str), sort=True):
        first = group.sort_values([column for column in ("listing_start", "local_code") if column in group]).iloc[0]
        rows.append(
            {
                "country": str(first["country"]),
                "exchange": str(first["exchange"]),
                "ticker": str(first.get("local_code", first.get("ticker", ""))),
                "refinitiv_ric": str(ric),
                "currency": str(first.get("currency", "JPY")),
                "selected_for_u2": True,
                "selected_for_u3": True,
                "has_refinitiv_ric": True,
                "is_active_role": str(ric) in active_rics,
                "listing_window_count": len(group),
                "pit_usable": True,
            }
        )
    return pd.DataFrame(rows).sort_values("refinitiv_ric").reset_index(drop=True)


def _liquidity_probe_queue(selected: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    samples = selected.sort_values(["is_active_role", "refinitiv_ric"], ascending=[False, True]).head(3)
    rows: list[dict[str, Any]] = []
    for role, field in JAPAN_LIQUIDITY_PROBE_FIELDS:
        for ric in samples["refinitiv_ric"].astype(str):
            rows.append(
                {
                    "dataset_id": "liquidity_market_cap_free_float",
                    "owner_agent": "liquidity_adv_agent",
                    "source_family": "refinitiv_reference_history_plus_ohlcv",
                    "field_role": role,
                    "candidate_field": field,
                    "required": True,
                    "candidate_rank": 1,
                    "refinitiv_ric": ric,
                    "interval": "1M",
                    "start_date": start_date,
                    "end_date": end_date,
                    "status": "blocked_vendor_health",
                }
            )
    return pd.DataFrame(rows)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes", "pass", "passed", "complete"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
