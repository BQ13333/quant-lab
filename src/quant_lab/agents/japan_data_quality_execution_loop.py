from __future__ import annotations

import hashlib
import json
import os
import shlex
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd

from quant_lab.agents.japan_data_quality_loop import (
    JapanDataQualityPaths,
    default_japan_data_quality_paths,
    run_japan_data_quality_loop,
)
from quant_lab.data.historical_backfill import build_refinitiv_historical_plan


@dataclass(frozen=True)
class JapanExecutionLoopConfig:
    run_id: str
    artifact_root: Path = Path("artifacts/japan_data_quality_execution_loop")
    project_root: Path = Path(".")
    scorecard_run_id: str = "JPN-DATA-QUALITY-LOOP-20260711"
    start_date: str = "2006-01-01"
    end_date: str = "2026-07-06"
    batch_size: int = 20
    active_max_plan_rows: int = 5
    reference_max_plan_rows: int = 5
    refinitiv_session: str = "desktop.workspace"
    vendor_health_evidence_path: Path = Path("artifacts/vendor_health/japan_refinitiv_health.json")
    vendor_health_max_age_minutes: int = 60
    g1_current_enrichment_dir: Path = Path(
        "artifacts/japan_g1_enrichment/"
        "JPX-UNIVERSE-RECONCILIATION-V2-20260711-G1-CURRENT-ENRICHMENT-V2"
    )
    listing_valid_time_dir: Path = Path(
        "artifacts/japan_listing_valid_time/"
        "JPX-UNIVERSE-RECONCILIATION-V2-20260711-LISTING-VALID-TIME-BASELINE"
    )
    jquants_listed_info_dir: Path = Path(
        "artifacts/jquants_listed_info/JQUANTS-JPN-LISTED-INFO-20080507-20260706-V2"
    )
    jquants_listed_info_run_id: str = "JQUANTS-JPN-LISTED-INFO-20080507-20260706-V2"
    jquants_max_dates: int = 5
    jquants_listed_shares_dir: Path = Path(
        "artifacts/jquants_listed_shares/JQUANTS-JPN-LISTED-SHARES-20080507-20260706-V1"
    )
    jquants_listed_shares_run_id: str = "JQUANTS-JPN-LISTED-SHARES-20080507-20260706-V1"
    g1_materializer_dir: Path = Path(
        "artifacts/japan_g1_monthly_materializer/JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1"
    )
    g1_materializer_run_id: str = "JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1"
    g1_validator_dir: Path = Path(
        "artifacts/japan_g1_validation/JPN-G1-MONTHLY-INDEPENDENT-VALIDATION-20260712-V1"
    )
    jpx_public_events_dir: Path = Path(
        "artifacts/jpx_public_listing_events/JPX-PUBLIC-LISTING-EVENTS-20260712"
    )
    jpx_public_crosscheck_dir: Path = Path(
        "artifacts/jpx_public_event_reconciliation/JPX-PUBLIC-EVENT-REFINITIV-CROSSCHECK-V2-20260712"
    )
    jpx_monthly_quotations_dir: Path = Path(
        "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2"
    )
    jpx_monthly_membership_qa_dir: Path = Path(
        "artifacts/japan_combined_monthly_membership_qa/JPN-COMBINED-MONTHLY-MEMBERSHIP-QA-200607-202606-V1"
    )
    g3_pit_plan_dir: Path = Path("artifacts/pit_data_plan/PIT-DATA-PLAN-JPN-U2-20260712-V3")
    g3_route_registry_dir: Path = Path("artifacts/japan_g3_route_registry/JPN-G3-ROUTE-REGISTRY-20260712")
    g3_fundamentals_qa_dir: Path = Path(
        "artifacts/refinitiv_japan_g3_fundamentals/REFINITIV-JPN-G3-FUNDAMENTALS-QA-20260712"
    )
    g3_sector_snapshot_dir: Path = Path(
        "artifacts/refinitiv_current_snapshot/REFINITIV-JPN-G3-SECTOR-ROUTING-20260712"
    )
    g3_sector_gap_dir: Path = Path(
        "artifacts/japan_g3_sector_gap_plan/JPN-G3-SECTOR-GAP-20260712-V2"
    )
    g3_sector_full_dir: Path = Path(
        "artifacts/refinitiv_current_snapshot/REFINITIV-JPN-G3-SECTOR-ROUTING-FULL-20260712-V2"
    )
    g3_ibes_events_qa_dir: Path = Path(
        "artifacts/refinitiv_japan_g3_ibes_events/REFINITIV-JPN-G3-IBES-EVENTS-QA-20260712"
    )
    g3_execution_plan_dir: Path = Path(
        "artifacts/japan_g3_execution_plan/JPN-G3-EXECUTION-PLAN-20260712-V2"
    )
    writer_lock_marker: Path = Path("artifacts/historical_backfill/.japan_ohlcv_writer.lock")
    active_historical_run_id: str = "REFINITIV-JPN-HISTORICAL-20Y-JPX-ACTIVE-G1V2"
    reference_historical_run_id: str = "REFINITIV-JPN-HISTORICAL-20Y-JPX-REFERENCE-ONLY-G1V4"
    g2_vendor_plan_dir: Path = Path(
        "artifacts/japan_g2_vendor_plan/JPN-G2-VENDOR-PLAN-20260712-V2"
    )
    corporate_action_max_plan_rows: int = 5
    liquidity_max_plan_rows: int = 5
    liquidity_field_probe_run_id: str = "REFINITIV-JPN-G2-LIQUIDITY-FIELD-PROBE-20260712-V1"
    liquidity_historical_probe_run_id: str = "REFINITIV-JPN-G2-LIQUIDITY-HISTORY-PROBE-20260712-V1"
    liquidity_historical_run_id: str = "REFINITIV-PIT-HISTORICAL-LIQUIDITY-JPN-U2-20260712-V2"
    active_canonical_run_id: str = "JPN-ACTIVE-G1V2-CANONICAL-FULL-20260712-V1"
    reference_canonical_run_id: str = "JPN-REFERENCE-ONLY-G1V4-CANONICAL-FULL-20260712-V1"
    active_postprocess_tag: str = "JPN-ACTIVE-G1V2-FULL-20260712-V1"
    reference_postprocess_tag: str = "JPN-REFERENCE-ONLY-G1V4-FULL-20260712-V1"
    price_union_run_id: str = "JPN-G2-PRICE-UNION-20260712-V1"
    adjusted_returns_run_id: str = "ADJRET-RECON-JPN-U2-20260712-V2"
    component_inventory_run_id: str = "JPN-G2-COMPONENT-HASHES-20260712-V1"
    strict_g2_admission_run_id: str = "JPN-G2-STRICT-ADMISSION-20260712-V3"
    g4_halt_history_dir: Path = Path(
        "artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V16"
    )
    g4_halt_validation_dir: Path = Path(
        "artifacts/japan_g4_halt_history_validation/"
        "JPN-G4-HALTS-INDEPENDENT-VALIDATION-200607-202606-V16-R1"
    )
    g4_halt_denominator_recovery_dir: Path = Path(
        "artifacts/japan_g4_halt_denominator_recovery/"
        "JPN-G4-HALT-DENOMINATOR-RECOVERY-20260712-V2"
    )
    g4_halt_denominator_validation_dir: Path = Path(
        "artifacts/japan_g4_halt_denominator_recovery_validation/"
        "JPN-G4-HALT-DENOMINATOR-INDEPENDENT-VALIDATION-20260712-V2"
    )
    g4_halt_report_events_dir: Path = Path(
        "artifacts/japan_g4_halt_report_events/"
        "JPN-G4-HALT-REPORT-EVENTS-20260712-V2"
    )
    g4_halt_report_events_validation_dir: Path = Path(
        "artifacts/japan_g4_halt_report_events_validation/"
        "JPN-G4-HALT-REPORT-EVENTS-INDEPENDENT-VALIDATION-20260712-V3"
    )
    g4_pre2013_price_limit_dir: Path = Path(
        "artifacts/japan_pre2013_price_limit_archive/"
        "JPN-PRE2013-PRICE-LIMIT-ARCHIVE-200607-201308-V2"
    )
    g4_return_normal_dir: Path = Path(
        "artifacts/jpx_return_to_normal_reconstruction/"
        "JPX-RETURN-NORMAL-202008-202606-PARTIAL-20260712-V3"
    )
    g4_return_normal_validation_dir: Path = Path(
        "artifacts/jpx_return_to_normal_reconstruction_validation/"
        "JPX-RETURN-NORMAL-INDEPENDENT-VALIDATION-20260712-V3"
    )
    g4_market_rule_validation_dir: Path = Path(
        "artifacts/japan_market_rule_history_validation/"
        "JPN-MARKET-RULE-HISTORY-INDEPENDENT-VALIDATION-20260712-V2-R1"
    )
    g4_tick_assignment_validation_dir: Path = Path(
        "artifacts/japan_tick_assignment_history_validation/"
        "JPN-TICK-ASSIGNMENT-INDEPENDENT-VALIDATION-20260712-V3"
    )
    g4_tca_validation_dir: Path = Path(
        "artifacts/japan_g4_tca_calibration_validation/"
        "JPN-G4-TCA-INDEPENDENT-VALIDATION-20260712-V1"
    )
    field_catalog_path: Path | None = None
    min_security_master_rows: int = 5_000
    min_active_rows: int = 3_000
    min_reference_rows: int = 1_000
    as_of_utc: datetime | None = None
    paths: JapanDataQualityPaths | None = None


@dataclass(frozen=True)
class JapanExecutionLoopArtifacts:
    run_id: str
    artifact_dir: Path
    agent_roster_path: Path
    execution_queue_path: Path
    execution_state_path: Path
    quarantine_path: Path
    historical_quarantine_path: Path
    active_symbol_map_path: Path
    reference_symbol_map_path: Path
    active_plan_path: Path
    reference_plan_path: Path
    summary_path: Path
    manifest_path: Path
    writer_lock_marker: Path
    execution_queue: pd.DataFrame
    quarantine: pd.DataFrame


def latest_japan_execution_loop_run_id() -> str:
    return datetime.now(timezone.utc).strftime("JPN-DATA-QUALITY-EXECUTION-LOOP-%Y%m%d-%H%M%S")


def write_japan_execution_loop_artifacts(config: JapanExecutionLoopConfig) -> JapanExecutionLoopArtifacts:
    _validate_config(config)
    root = config.project_root
    paths = config.paths or default_japan_data_quality_paths(root)
    now = (config.as_of_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    artifact_dir = _under_root(root, config.artifact_root) / config.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scorecard = _run_scorecard(config, paths)
    security_manifest = _read_json(paths.security_master_dir / "manifest.json")
    universe_manifest = _read_json(paths.universe_reconciliation_dir / "manifest.json")
    active_map_manifest = _read_json(paths.active_symbol_map_dir / "manifest.json")
    reference_map_manifest = _read_json(paths.reference_symbol_map_dir / "manifest.json")
    active_universe = _read_csv(paths.active_symbol_map_dir / "symbol_map.csv")
    reference_universe = _read_csv(paths.reference_symbol_map_dir / "symbol_map.csv")
    active_universe, reference_universe, universe_state = _reconcile_universe(
        config,
        security_manifest,
        universe_manifest,
        active_map_manifest,
        reference_map_manifest,
        active_universe,
        reference_universe,
    )

    active_symbol_map_path = artifact_dir / "active_symbol_map.csv"
    reference_symbol_map_path = artifact_dir / "reference_symbol_map.csv"
    if universe_state["eligible"]:
        _atomic_write_csv(active_universe, active_symbol_map_path)
        _atomic_write_csv(reference_universe, reference_symbol_map_path)

    active_dir = root / "artifacts/historical_backfill" / config.active_historical_run_id
    reference_dir = root / "artifacts/historical_backfill" / config.reference_historical_run_id
    active_plan_path = active_dir / "historical_plan.csv"
    reference_plan_path = reference_dir / "historical_plan.csv"
    plan_states = {
        "active": _stage_immutable_plan(
            config, active_universe, config.active_historical_run_id, active_plan_path
        ),
        "reference": _stage_immutable_plan(
            config, reference_universe, config.reference_historical_run_id, reference_plan_path
        ),
    }

    active_manifest_path = active_dir / "historical_manifest.csv"
    active_manifest_source = active_manifest_path if active_manifest_path.is_file() else paths.historical_dir / "historical_manifest.csv"
    active_detected = _nonusable_rows(_read_csv(active_manifest_source), now)
    reference_detected = _nonusable_rows(_read_csv(reference_dir / "historical_manifest.csv"), now)
    historical_quarantine_path = active_dir / "quarantined_plan_rows.csv"
    reference_quarantine_path = reference_dir / "quarantined_plan_rows.csv"
    legacy_quarantine_path = paths.historical_dir / "quarantined_plan_rows.csv"
    if not historical_quarantine_path.exists() and active_manifest_source != active_manifest_path and legacy_quarantine_path.is_file():
        _atomic_write_csv(_read_csv(legacy_quarantine_path), historical_quarantine_path)
    active_quarantine = _atomic_merge_quarantine(historical_quarantine_path, active_detected)
    reference_quarantine = _atomic_merge_quarantine(reference_quarantine_path, reference_detected)
    quarantine = pd.concat(
        [
            active_quarantine.assign(lane="active"),
            reference_quarantine.assign(lane="reference"),
        ],
        ignore_index=True,
    )
    quarantine_path = artifact_dir / "historical_quarantine.csv"
    _atomic_write_csv(quarantine, quarantine_path)

    health_path = _under_root(root, config.vendor_health_evidence_path)
    vendor_health = _vendor_health_state(health_path, now, config.vendor_health_max_age_minutes)
    active_recovery = _active_recovery_state(root, config.active_historical_run_id)
    g2_vendor = _g2_vendor_state(config)
    g2_postprocess = _g2_postprocess_state(config)
    g4_partial_evidence = _g4_partial_evidence_state(config)
    current_enrichment = _current_enrichment_state(
        _read_json(_under_root(root, config.g1_current_enrichment_dir) / "manifest.json")
    )
    valid_time = _valid_time_state(
        _read_json(_under_root(root, config.listing_valid_time_dir) / "manifest.json")
    )
    jquants_listed_info = _jquants_listed_info_state(
        _read_json(_under_root(root, config.jquants_listed_info_dir) / "manifest.json")
    )
    jquants_listed_shares = _jquants_listed_shares_state(
        _read_json(_under_root(root, config.jquants_listed_shares_dir) / "manifest.json")
    )
    g1_materializer = _g1_materializer_state(
        _read_json(_under_root(root, config.g1_materializer_dir) / "manifest.json")
    )
    g1_validator = _g1_validator_state(
        _read_json(_under_root(root, config.g1_validator_dir) / "manifest.json")
    )
    jpx_public_events = _jpx_public_events_state(
        _read_json(_under_root(root, config.jpx_public_events_dir) / "manifest.json")
    )
    jpx_public_crosscheck = _jpx_public_crosscheck_state(
        _read_json(_under_root(root, config.jpx_public_crosscheck_dir) / "manifest.json")
    )
    jpx_monthly_quotations = _jpx_monthly_quotations_state(
        _read_json(_under_root(root, config.jpx_monthly_quotations_dir) / "manifest.json")
    )
    jpx_monthly_membership_qa = _jpx_monthly_membership_qa_state(
        _read_json(_under_root(root, config.jpx_monthly_membership_qa_dir) / "manifest.json")
    )
    writer_lock_marker = _under_root(root, config.writer_lock_marker)
    lock_state = {
        "marker_path": str(writer_lock_marker.resolve()),
        "marker_exists": writer_lock_marker.exists(),
        "single_writer_required": True,
    }
    lanes = {lane.lane_id: lane for lane in scorecard.lanes}
    queue = _execution_queue(
        config=config,
        paths=paths,
        lanes=lanes,
        scorecard_accepted=scorecard.accepted,
        universe_state=universe_state,
        vendor_health=vendor_health,
        current_enrichment=current_enrichment,
        valid_time=valid_time,
        jquants_listed_info=jquants_listed_info,
        jquants_listed_shares=jquants_listed_shares,
        g1_materializer=g1_materializer,
        g1_validator=g1_validator,
        jpx_public_events=jpx_public_events,
        jpx_public_crosscheck=jpx_public_crosscheck,
        jpx_monthly_quotations=jpx_monthly_quotations,
        jpx_monthly_membership_qa=jpx_monthly_membership_qa,
        lock_state=lock_state,
        plan_states=plan_states,
        active_recovery=active_recovery,
        g2_vendor=g2_vendor,
        g2_postprocess=g2_postprocess,
        g4_partial_evidence=g4_partial_evidence,
        active_symbol_map_path=active_symbol_map_path,
        reference_symbol_map_path=reference_symbol_map_path,
        active_quarantine_path=historical_quarantine_path,
        reference_quarantine_path=reference_quarantine_path,
    )

    agent_roster_path = artifact_dir / "agent_roster.csv"
    execution_queue_path = artifact_dir / "execution_queue.csv"
    execution_state_path = artifact_dir / "execution_state.json"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"
    _atomic_write_csv(_agent_roster(), agent_roster_path)
    _atomic_write_csv(queue, execution_queue_path)

    state = {
        "run_id": config.run_id,
        "generated_at_utc": now.isoformat(),
        "mode": "queue_generation_only",
        "live_extraction_executed": False,
        "universe": universe_state,
        "vendor_health": vendor_health,
        "g1_current_enrichment": current_enrichment,
        "g1_listing_valid_time": valid_time,
        "jquants_listed_info": jquants_listed_info,
        "jquants_listed_shares": jquants_listed_shares,
        "g1_materializer": g1_materializer,
        "g1_validator": g1_validator,
        "jpx_public_events": jpx_public_events,
        "jpx_public_crosscheck": jpx_public_crosscheck,
        "jpx_monthly_quotations": jpx_monthly_quotations,
        "jpx_monthly_membership_qa": jpx_monthly_membership_qa,
        "writer_lock": lock_state,
        "plans": plan_states,
        "active_quota_recovery": active_recovery,
        "g2_vendor": g2_vendor,
        "g2_postprocess": g2_postprocess,
        "g4_partial_evidence": g4_partial_evidence,
        "quarantines": {
            "active": str(historical_quarantine_path.resolve()),
            "reference": str(reference_quarantine_path.resolve()),
            "active_rows": len(active_quarantine),
            "reference_rows": len(reference_quarantine),
        },
        "phase_dependencies": {
            "G1": [],
            "G2": ["G1", "VENDOR_HEALTH"],
            "G3": ["G2"],
            "G4": ["G3"],
            "G5": ["G4"],
        },
        "promotion_policy": {
            "alpha": False,
            "production": False,
            "ibkr": False,
            "live_trading": False,
        },
    }
    _atomic_write_json(state, execution_state_path)
    _atomic_write_text(
        summary_path,
        _summary(config, scorecard.overall_score, scorecard.accepted, vendor_health, universe_state, queue),
    )
    manifest = {
        "run_id": config.run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "country": "Japan",
        "country_code": "JPN",
        "exchange_scope": "JPX",
        "queue_rows": int(len(queue)),
        "quarantine_rows": int(len(quarantine)),
        "scorecard_run_id": config.scorecard_run_id,
        "scorecard_accepted": bool(scorecard.accepted),
        "live_extraction_executed": False,
        "alpha_promotion_allowed": False,
        "production_promotion_allowed": False,
        "ibkr_promotion_allowed": False,
        "generated_at_utc": now.isoformat(),
        "artifact_sha256": {
            "agent_roster": _sha256(agent_roster_path),
            "execution_queue": _sha256(execution_queue_path),
            "execution_state": _sha256(execution_state_path),
            "historical_quarantine": _sha256(historical_quarantine_path),
            "reference_historical_quarantine": _sha256(reference_quarantine_path),
            "summary": _sha256(summary_path),
        },
        "artifacts": {
            "agent_roster": str(agent_roster_path.resolve()),
            "execution_queue": str(execution_queue_path.resolve()),
            "execution_state": str(execution_state_path.resolve()),
            "historical_quarantine": str(historical_quarantine_path.resolve()),
            "reference_historical_quarantine": str(reference_quarantine_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    _atomic_write_json(manifest, manifest_path)
    return JapanExecutionLoopArtifacts(
        run_id=config.run_id,
        artifact_dir=artifact_dir,
        agent_roster_path=agent_roster_path,
        execution_queue_path=execution_queue_path,
        execution_state_path=execution_state_path,
        quarantine_path=quarantine_path,
        historical_quarantine_path=historical_quarantine_path,
        active_symbol_map_path=active_symbol_map_path,
        reference_symbol_map_path=reference_symbol_map_path,
        active_plan_path=active_plan_path,
        reference_plan_path=reference_plan_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        writer_lock_marker=writer_lock_marker,
        execution_queue=queue,
        quarantine=quarantine,
    )


def _validate_config(config: JapanExecutionLoopConfig) -> None:
    positive = {
        "batch_size": config.batch_size,
        "active_max_plan_rows": config.active_max_plan_rows,
        "reference_max_plan_rows": config.reference_max_plan_rows,
        "corporate_action_max_plan_rows": config.corporate_action_max_plan_rows,
        "liquidity_max_plan_rows": config.liquidity_max_plan_rows,
        "vendor_health_max_age_minutes": config.vendor_health_max_age_minutes,
        "jquants_max_dates": config.jquants_max_dates,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(f"Japan execution limits must be positive: {', '.join(invalid)}")


def _run_scorecard(config: JapanExecutionLoopConfig, paths: JapanDataQualityPaths) -> Any:
    try:
        return run_japan_data_quality_loop(
            paths=paths,
            artifact_root=config.project_root / "artifacts/institutional_data_quality",
            run_id=config.scorecard_run_id,
            project_root=config.project_root,
        )
    except NameError as exc:
        lane_ids = (
            "g1_security_master_universe",
            "g2_prices_actions_liquidity",
            "g3_pit_fundamentals_ibes_events",
            "g4_fx_calendar_halts_tca",
            "g5_leakage_bias_reproducibility",
        )
        lanes = tuple(
            SimpleNamespace(
                lane_id=lane_id,
                status="fail",
                critical_blockers=(f"Japan scorecard dependency error: {exc}",),
                evidence="scorecard execution failed closed",
            )
            for lane_id in lane_ids
        )
        return SimpleNamespace(overall_score=0.0, accepted=False, lanes=lanes)


def _reconcile_universe(
    config: JapanExecutionLoopConfig,
    security_manifest: dict[str, Any],
    manifest: dict[str, Any],
    active_map_manifest: dict[str, Any],
    reference_map_manifest: dict[str, Any],
    active: pd.DataFrame,
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    reasons: list[str] = []
    security_rows = max(
        int(security_manifest.get(key, 0) or 0)
        for key in ("security_master_rows", "listing_rows", "entity_rows")
    )
    if security_rows < config.min_security_master_rows:
        reasons.append("japan_security_master_below_minimum")
    if manifest.get("scorecard_eligible") is not True:
        reasons.append("jpx_reconciliation_not_scorecard_eligible")
    if manifest.get("strict_quality_pass") is not True:
        reasons.append("jpx_reconciliation_strict_quality_failed")
    if manifest.get("historical_pit_ready") is not True:
        reasons.append("jpx_historical_pit_membership_not_ready")
    if manifest.get("historical_backtest_eligible") is not True:
        reasons.append("jpx_historical_backtest_scope_not_eligible")

    required = {"refinitiv_ric", "currency", "known_at", "listing_start", "listing_end"}
    for role, frame, map_manifest, minimum in (
        ("active", active, active_map_manifest, config.min_active_rows),
        ("reference", reference, reference_map_manifest, config.min_reference_rows),
    ):
        if not (
            map_manifest.get("status") in {"complete", "pass"}
            and map_manifest.get("scorecard_eligible") is True
            and map_manifest.get("evidence_scope") == "full"
            and map_manifest.get("map_role") == role
            and map_manifest.get("validation_status") == "pass"
        ):
            reasons.append(f"{role}_jpx_map_not_independently_validated")
        if frame.empty or not required.issubset(frame.columns):
            reasons.append(f"{role}_jpx_map_missing_validated_columns")
            continue
        if len(frame) < minimum:
            reasons.append(f"{role}_jpx_universe_below_minimum")
        if frame[list(required)].isna().any().any():
            reasons.append(f"{role}_jpx_map_has_null_required_values")
        if not frame["currency"].astype(str).eq("JPY").all():
            reasons.append(f"{role}_jpx_map_has_non_jpy_currency")
        if not _non_overlapping_symbol_map_windows(frame):
            reasons.append(f"invalid_or_overlapping_{role}_jpx_windows")

    if reasons:
        active = active.head(0).copy()
        reference = reference.head(0).copy()
    active_rows = len(active)
    reference_rows = len(reference)
    return active.reset_index(drop=True), reference.reset_index(drop=True), {
        "eligible": not reasons,
        "exchange_scope": "JPX",
        "active_rows": active_rows,
        "reference_rows": reference_rows,
        "full_reference_source_rows": int(
            manifest.get("counts", {}).get("historical_listed_reference_rows", reference_rows)
        ),
        "security_master_rows": security_rows,
        "reasons": reasons,
    }


def _non_overlapping_symbol_map_windows(frame: pd.DataFrame) -> bool:
    required = {"refinitiv_ric", "listing_start", "listing_end"}
    if frame.empty or not required.issubset(frame.columns):
        return False
    starts = pd.to_datetime(frame["listing_start"], errors="coerce", utc=True)
    ends = pd.to_datetime(frame["listing_end"], errors="coerce", utc=True)
    if starts.isna().any() or ends.isna().any() or starts.gt(ends).any():
        return False
    dated = frame.assign(_start=starts, _end=ends).sort_values(
        ["refinitiv_ric", "_start", "_end"]
    )
    if dated.duplicated(["refinitiv_ric", "_start", "_end"]).any():
        return False
    previous_end = dated.groupby("refinitiv_ric", sort=False)["_end"].shift()
    return not bool(dated["_start"].le(previous_end).fillna(False).any())


def _stage_immutable_plan(
    config: JapanExecutionLoopConfig,
    universe: pd.DataFrame,
    run_id: str,
    plan_path: Path,
) -> dict[str, Any]:
    checksum_path = plan_path.with_suffix(".sha256")
    if universe.empty:
        return {
            "run_id": run_id,
            "plan_path": str(plan_path.resolve()),
            "checksum_path": str(checksum_path.resolve()),
            "sha256": None,
            "rows": 0,
            "valid": False,
            "reason": "universe_not_eligible",
        }
    desired = build_refinitiv_historical_plan(
        universe=universe,
        root=config.project_root / "artifacts/data_lake",
        run_id=run_id,
        start_date=config.start_date,
        end_date=config.end_date,
        batch_size=config.batch_size,
        window_years=5,
        interval="1d",
    )
    desired_bytes = desired.to_csv(index=False).encode("utf-8")
    desired_sha = hashlib.sha256(desired_bytes).hexdigest()
    expected_sha = checksum_path.read_text(encoding="utf-8").strip().split()[0] if checksum_path.exists() else None
    current_sha = _sha256(plan_path) if plan_path.exists() else None

    valid = True
    reason = "checksum_verified"
    if expected_sha and (expected_sha != desired_sha or current_sha != expected_sha):
        valid = False
        reason = "immutable_plan_checksum_mismatch"
    elif plan_path.exists() and current_sha != desired_sha:
        valid = False
        reason = "existing_plan_differs_from_generated_policy"
    else:
        if not plan_path.exists():
            _atomic_write_bytes(plan_path, desired_bytes)
        if not checksum_path.exists():
            _atomic_write_text(checksum_path, desired_sha + "\n")
        current_sha = _sha256(plan_path)
        valid = current_sha == desired_sha
        if not valid:
            reason = "plan_changed_during_staging"
    return {
        "run_id": run_id,
        "plan_path": str(plan_path.resolve()),
        "checksum_path": str(checksum_path.resolve()),
        "sha256": expected_sha or desired_sha,
        "current_sha256": current_sha,
        "rows": int(len(desired)),
        "valid": bool(valid),
        "reason": reason,
    }


def _vendor_health_state(path: Path, now: datetime, max_age_minutes: int) -> dict[str, Any]:
    evidence = _read_json(path)
    symbol = str(evidence.get("symbol", evidence.get("probe_symbol", evidence.get("universe", ""))))
    status = str(evidence.get("status", "missing")).casefold()
    timestamp_text = evidence.get("observed_at_utc", evidence.get("checked_at_utc", evidence.get("generated_at_utc")))
    observed: datetime | None = None
    if timestamp_text:
        try:
            observed = datetime.fromisoformat(str(timestamp_text).replace("Z", "+00:00"))
            if observed.tzinfo is None:
                observed = None
            else:
                observed = observed.astimezone(timezone.utc)
        except ValueError:
            observed = None
    age_minutes = (now - observed).total_seconds() / 60 if observed else None
    fresh = age_minutes is not None and -5 <= age_minutes <= max_age_minutes
    row_value = next((evidence[key] for key in ("row_count", "rows", "price_rows") if key in evidence), None)
    payload_ok = row_value is None or int(row_value) > 0
    valid = status in {"healthy", "pass", "passed"} and symbol == "7203.T" and fresh and payload_ok
    reasons: list[str] = []
    if status not in {"healthy", "pass", "passed"}:
        reasons.append("health_status_not_healthy")
    if symbol != "7203.T":
        reasons.append("probe_symbol_must_be_7203.T")
    if not fresh:
        reasons.append("health_evidence_missing_or_stale")
    if not payload_ok:
        reasons.append("health_probe_returned_no_rows")
    return {
        "valid": valid,
        "status": status,
        "probe_symbol": symbol,
        "observed_at_utc": observed.isoformat() if observed else None,
        "age_minutes": round(age_minutes, 3) if age_minutes is not None else None,
        "max_age_minutes": max_age_minutes,
        "evidence_path": str(path.resolve()),
        "reasons": reasons,
    }


def _g2_vendor_state(config: JapanExecutionLoopConfig) -> dict[str, Any]:
    root = config.project_root
    plan_dir = _under_root(root, config.g2_vendor_plan_dir)
    plan_manifest = _read_json(plan_dir / "manifest.json")
    output_sha256 = plan_manifest.get("output_sha256", {})
    hashes_valid = bool(output_sha256) and all(
        (plan_dir / str(name)).is_file()
        and _sha256(plan_dir / str(name)) == str(expected).lower()
        for name, expected in output_sha256.items()
    )
    input_sha256 = plan_manifest.get("input_sha256", {})
    input_hashes_valid = bool(input_sha256) and all(
        Path(str(name)).is_file()
        and _sha256(Path(str(name))) == str(expected).lower()
        for name, expected in input_sha256.items()
    )
    plan_valid = bool(
        hashes_valid
        and input_hashes_valid
        and plan_manifest.get("artifact_type") == "japan_g2_vendor_execution_plan"
        and plan_manifest.get("planning_complete") is True
        and int(plan_manifest.get("unique_reference_rics", 0) or 0) == 5_752
        and int(plan_manifest.get("corporate_action_plan_rows", 0) or 0) == 462
        and int(plan_manifest.get("liquidity_probe_rows", 0) or 0) == 9
    )

    corporate_run_id = str(
        plan_manifest.get("corporate_action_run_id")
        or "REFINITIV-JPN-G2-CORPACT-20060101-20260706-V1"
    )
    corporate_root = _under_root(
        root,
        Path(
            str(
                plan_manifest.get("corporate_action_artifact_root")
                or "artifacts/refinitiv_corporate_actions_events"
            )
        ),
    )
    corporate_dir = corporate_root / corporate_run_id
    corporate_manifest = _read_json(corporate_dir / "manifest.json")
    corporate_plan_rows = int(plan_manifest.get("corporate_action_plan_rows", 0) or 0)
    corporate_complete = bool(
        corporate_manifest.get("status") == "complete"
        and corporate_manifest.get("official_source") is True
        and int(corporate_manifest.get("planned_rows", -1)) == corporate_plan_rows
        and int(corporate_manifest.get("completed_rows", -1)) == corporate_plan_rows
        and int(corporate_manifest.get("failed_rows", -1)) == 0
    )

    field_probe_dir = (
        root / "artifacts/refinitiv_pit_field_probe" / config.liquidity_field_probe_run_id
    )
    field_probe_manifest = _read_json(field_probe_dir / "manifest.json")
    field_probe_results = _read_csv(field_probe_dir / "field_probe_results.csv")
    field_probe_samples = _read_csv(field_probe_dir / "sample_universe.csv")
    accepted_fields = _read_csv(field_probe_dir / "accepted_field_map.csv")
    required_roles = {"market_cap", "free_float", "shares_outstanding"}
    accepted_roles = set(accepted_fields.get("field_role", pd.Series(dtype=str)).astype(str))
    expected_samples = {"1301.T", "130A.T", "131A.T"}
    field_probe_complete = bool(
        int(field_probe_manifest.get("error_fields", -1)) == 0
        and int(field_probe_manifest.get("probe_rows", -1)) == 9 == len(field_probe_results)
        and set(field_probe_manifest.get("datasets", []))
        == {"liquidity_market_cap_free_float"}
        and not field_probe_results.empty
        and field_probe_results.get("status", pd.Series(dtype=str)).astype(str).eq("validated").all()
        and set(
            field_probe_samples.get("refinitiv_ric", pd.Series(dtype=str)).astype(str)
        )
        == expected_samples
        and len(accepted_fields) == 3
        and accepted_roles == required_roles
        and pd.to_numeric(
            accepted_fields.get("non_null_values", pd.Series(dtype=float)), errors="coerce"
        ).gt(0).all()
    )

    historical_probe_dir = (
        root
        / "artifacts/refinitiv_pit_historical_probe"
        / config.liquidity_historical_probe_run_id
    )
    history_map = _read_csv(historical_probe_dir / "history_field_map.csv")
    historical_results = _read_csv(historical_probe_dir / "historical_probe_results.csv")
    historical_roles = set(history_map.get("field_role", pd.Series(dtype=str)).astype(str))
    historical_probe_complete = bool(
        len(history_map) == 3
        and historical_roles == required_roles
        and len(historical_results) == 3
        and historical_results.get("status", pd.Series(dtype=str))
        .astype(str)
        .eq("historical_validated")
        .all()
        and historical_results.get("interval", pd.Series(dtype=str)).astype(str).eq("1M").all()
        and historical_results.get("start_date", pd.Series(dtype=str))
        .astype(str)
        .eq(config.start_date)
        .all()
        and historical_results.get("end_date", pd.Series(dtype=str))
        .astype(str)
        .eq(config.end_date)
        .all()
        and pd.to_numeric(
            historical_results.get("inside_requested_rows", pd.Series(dtype=float)),
            errors="coerce",
        ).gt(0).all()
        and pd.to_numeric(
            historical_results.get("outside_requested_rows", pd.Series(dtype=float)),
            errors="coerce",
        ).eq(0).all()
    )

    liquidity_dir = (
        root
        / "artifacts/refinitiv_pit_historical_backfill"
        / config.liquidity_historical_run_id
    )
    liquidity_manifest = _read_json(liquidity_dir / "manifest.json")
    liquidity_terminal = _read_csv(liquidity_dir / "historical_manifest.csv")
    liquidity_quality = _read_csv(liquidity_dir / "historical_quality.csv")
    liquidity_planned = int(liquidity_manifest.get("planned_rows", 0) or 0)
    liquidity_complete = bool(
        liquidity_manifest.get("halted_early") is False
        and liquidity_planned > 0
        and int(liquidity_manifest.get("executed_manifest_rows", -1)) == liquidity_planned
        and int(liquidity_manifest.get("completed_rows", -1)) == liquidity_planned
        and int(liquidity_manifest.get("empty_rows", -1)) == 0
        and int(liquidity_manifest.get("failed_rows", -1)) == 0
        and len(liquidity_terminal) == liquidity_planned
        and liquidity_terminal.get("status", pd.Series(dtype=str)).astype(str).eq("completed").all()
        and pd.to_numeric(
            liquidity_terminal.get("history_rows", pd.Series(dtype=float)), errors="coerce"
        ).gt(0).all()
        and pd.to_numeric(
            liquidity_terminal.get("outside_requested_rows", pd.Series(dtype=float)),
            errors="coerce",
        ).eq(0).all()
        and not liquidity_quality.empty
        and pd.to_numeric(
            liquidity_quality.get("min_coverage", pd.Series(dtype=float)), errors="coerce"
        ).eq(1.0).all()
        and pd.to_numeric(
            liquidity_quality.get("avg_coverage", pd.Series(dtype=float)), errors="coerce"
        ).eq(1.0).all()
    )
    sample_rics = (
        _read_csv(plan_dir / "liquidity_probe_queue.csv")
        .get("refinitiv_ric", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    return {
        "plan_dir": str(plan_dir),
        "plan_valid": plan_valid,
        "plan_hashes_valid": hashes_valid,
        "input_hashes_valid": input_hashes_valid,
        "selected_universe_path": str(plan_dir / "selected_universe.csv"),
        "corporate_action_plan_path": str(plan_dir / "corporate_action_plan.csv"),
        "corporate_action_plan_sha256": str(
            output_sha256.get("corporate_action_plan.csv", "")
        ),
        "corporate_action_run_id": corporate_run_id,
        "corporate_action_artifact_root": str(corporate_root),
        "corporate_action_dir": str(corporate_dir),
        "corporate_action_complete": corporate_complete,
        "liquidity_sample_rics": sample_rics,
        "liquidity_field_probe_dir": str(field_probe_dir),
        "liquidity_field_probe_complete": field_probe_complete,
        "liquidity_historical_probe_dir": str(historical_probe_dir),
        "liquidity_historical_probe_complete": historical_probe_complete,
        "liquidity_dir": str(liquidity_dir),
        "liquidity_complete": liquidity_complete,
    }


def _g2_postprocess_state(config: JapanExecutionLoopConfig) -> dict[str, Any]:
    root = config.project_root
    active = _postprocess_lane_state(
        config, role="active", canonical_run_id=config.active_canonical_run_id,
        tag=config.active_postprocess_tag,
    )
    reference = _postprocess_lane_state(
        config, role="reference", canonical_run_id=config.reference_canonical_run_id,
        tag=config.reference_postprocess_tag,
    )
    price_union_dir = root / "artifacts/japan_g2_price_union" / config.price_union_run_id
    union_manifest = _read_json(price_union_dir / "manifest.json")
    union_rows = int(union_manifest.get("output_rows", 0) or 0)
    price_union_complete = bool(
        active["complete"]
        and reference["complete"]
        and union_manifest.get("status") == "complete"
        and union_manifest.get("exact_rows_unique") is True
        and union_manifest.get("keys_unique") is True
        and union_manifest.get("source_file_hashes_verified") is True
        and union_manifest.get("input_manifest_hashes_verified") is True
        and union_rows == active["rows"] + reference["rows"]
        and _sha256_checksums_valid(price_union_dir, price_union_dir / "checksums.sha256")
    )
    adjusted_dir = (
        root / "artifacts/adjusted_return_reconstruction" / config.adjusted_returns_run_id
    )
    adjusted = _read_json(adjusted_dir / "manifest.json")
    adjusted_complete = bool(
        adjusted.get("status") == "pass"
        and adjusted.get("production_adjusted_return_ready") is True
        and int(adjusted.get("event_rows", 0) or 0) > 0
        and int(adjusted.get("event_rows", -1)) == int(adjusted.get("valid_event_rows", -2))
        == int(adjusted.get("matched_event_rows", -3))
        and float(adjusted.get("matched_event_share", -1.0)) == 1.0
        and int(adjusted.get("quarantined_event_rows", -1)) == 0
        and int(adjusted.get("ambiguous_factor_rows", -1)) == 0
        and float(adjusted.get("quarantine_share", -1.0)) == 0.0
        and int(adjusted.get("daily_adjusted_return_rows", -1)) == union_rows
    )
    component_inventory_path = (
        root / "artifacts/japan_g2_hash_inventory" / config.component_inventory_run_id
        / "component_hashes.json"
    )
    component_inventory_complete = _inventory_file_valid(component_inventory_path)
    admission_dir = (
        root / "artifacts/japan_g2_admission" / config.strict_g2_admission_run_id
    )
    admission = _read_json(admission_dir / "manifest.json")
    contract = _read_csv(admission_dir / "component_contract.csv")
    admission_complete = bool(
        admission.get("status") == "pass"
        and admission.get("admission_passed") is True
        and admission.get("g2_complete") is True
        and admission.get("scorecard_eligible") is True
        and admission.get("source_checksums_verified") is True
        and admission.get("window_coverage_complete") is True
        and admission.get("raw_compacted_parity") is True
        and admission.get("official_corporate_actions_reconciled") is True
        and admission.get("pit_liquidity_complete") is True
        and int(admission.get("critical_blocker_count", -1)) == 0
        and int(admission.get("check_count", -1))
        == int(admission.get("passed_check_count", -2))
        and not contract.empty
        and contract.get("passed", pd.Series(dtype=bool)).map(_truthy).all()
    )
    return {
        "active": active,
        "reference": reference,
        "price_union_dir": str(price_union_dir),
        "price_union_root": str(price_union_dir / "price_union"),
        "price_union_rows": union_rows,
        "price_union_complete": price_union_complete,
        "adjusted_dir": str(adjusted_dir),
        "adjusted_complete": adjusted_complete,
        "component_inventory_path": str(component_inventory_path),
        "component_inventory_complete": component_inventory_complete,
        "admission_dir": str(admission_dir),
        "admission_complete": admission_complete,
    }


def _postprocess_lane_state(
    config: JapanExecutionLoopConfig, *, role: str, canonical_run_id: str, tag: str
) -> dict[str, Any]:
    root = config.project_root
    canonical_dir = root / "artifacts/japan_ohlcv_canonical" / canonical_run_id
    depth_dir = root / "artifacts/data_depth" / f"{tag}-DEPTH"
    compaction_dir = root / "artifacts/data_lake_ops" / f"{tag}-COMPACTION"
    equality_dir = root / "artifacts/ohlcv_row_equality" / f"{tag}-ROW-EQUALITY"
    inventory_path = (
        root / "artifacts/japan_g2_hash_inventory" / f"{tag}-HASHES" / f"{role}_hashes.json"
    )
    canonical = _read_json(canonical_dir / "manifest.json")
    depth = _read_csv(depth_dir / "overall.csv")
    compaction = _read_json(compaction_dir / "manifest.json")
    equality = _read_json(equality_dir / "manifest.json")
    inventory_valid = _inventory_file_valid(inventory_path)
    rows = int(canonical.get("canonical_rows", 0) or 0)
    complete = bool(
        canonical.get("status") == "pass"
        and canonical.get("promotion_eligible") is True
        and canonical.get("source_extraction_complete") is True
        and rows > 0
        and len(depth) == 1
        and int(pd.to_numeric(depth.get("rows", pd.Series([-1])), errors="coerce").iloc[0]) == rows
        and compaction.get("status") == "complete"
        and int(compaction.get("source_rows", -1)) == rows
        and int(compaction.get("output_rows", -1)) == rows
        and int(compaction.get("failed_groups", -1)) == 0
        and equality.get("status") == "pass"
        and equality.get("row_equality_passed") is True
        and int(equality.get("source_rows", -1)) == rows
        and int(equality.get("compacted_rows", -1)) == rows
        and not equality.get("failed_years")
        and inventory_valid
    )
    return {
        "role": role,
        "canonical_run_id": canonical_run_id,
        "tag": tag,
        "canonical_dir": str(canonical_dir),
        "canonical_root": str(canonical_dir / "ohlcv"),
        "compacted_root": str(canonical_dir / "ohlcv_compacted"),
        "depth_dir": str(depth_dir),
        "compaction_dir": str(compaction_dir),
        "equality_dir": str(equality_dir),
        "inventory_path": str(inventory_path),
        "rows": rows,
        "complete": complete,
    }


def _g4_partial_evidence_state(config: JapanExecutionLoopConfig) -> dict[str, Any]:
    root = config.project_root
    halt_dir = _under_root(root, config.g4_halt_history_dir)
    halt_validation_dir = _under_root(root, config.g4_halt_validation_dir)
    halt_denominator_dir = _under_root(root, config.g4_halt_denominator_recovery_dir)
    halt_denominator_validation_dir = _under_root(
        root, config.g4_halt_denominator_validation_dir
    )
    halt_report_events_dir = _under_root(root, config.g4_halt_report_events_dir)
    halt_report_events_validation_dir = _under_root(
        root, config.g4_halt_report_events_validation_dir
    )
    pre2013_dir = _under_root(root, config.g4_pre2013_price_limit_dir)
    return_dir = _under_root(root, config.g4_return_normal_dir)
    return_validation_dir = _under_root(root, config.g4_return_normal_validation_dir)
    market_rule_validation_dir = _under_root(root, config.g4_market_rule_validation_dir)
    tick_assignment_validation_dir = _under_root(
        root, config.g4_tick_assignment_validation_dir
    )
    tca_validation_dir = _under_root(root, config.g4_tca_validation_dir)

    halt = _read_json(halt_dir / "manifest.json")
    halt_validation = _read_json(halt_validation_dir / "manifest.json")
    halt_denominator = _read_json(halt_denominator_dir / "manifest.json")
    halt_denominator_validation = _read_json(
        halt_denominator_validation_dir / "manifest.json"
    )
    halt_report_events = _read_json(halt_report_events_dir / "manifest.json")
    halt_report_events_validation = _read_json(
        halt_report_events_validation_dir / "manifest.json"
    )
    pre2013 = _read_json(pre2013_dir / "manifest.json")
    return_normal = _read_json(return_dir / "manifest.json")
    return_validation = _read_json(return_validation_dir / "manifest.json")
    return_missing_plan = _read_csv(return_dir / "missing_price_extraction_plan.csv")
    market_rule_validation = _read_json(market_rule_validation_dir / "manifest.json")
    market_rule_producer_path = Path(str(market_rule_validation.get("producer_manifest", "")))
    market_rule_coverage = _read_csv(market_rule_producer_path.parent / "rule_coverage.csv")
    tick_assignment_validation = _read_json(
        tick_assignment_validation_dir / "manifest.json"
    )
    tca_validation = _read_json(tca_validation_dir / "manifest.json")

    halt_validated_partial = bool(
        halt.get("status") == "partial_candidate"
        and halt.get("scorecard_eligible") is False
        and halt_validation.get("status") == "pass_partial_component"
        and halt_validation.get("scorecard_eligible") is False
        and not halt_validation.get("validation_issues")
    )
    denominator_producer_hashes_valid = _declared_hash_inventory_valid(
        halt_denominator_dir,
        halt_denominator.get("artifact_hashes_sha256", ""),
    )
    denominator_validation_hashes_valid = _declared_hash_inventory_valid(
        halt_denominator_validation_dir,
        halt_denominator_validation.get("artifact_hashes_sha256", ""),
    )
    halt_denominator_validated_partial = bool(
        halt_denominator.get("status") == "partial_recovery_candidate"
        and int(halt_denominator.get("source_denominator_rows", 0) or 0) == 366
        and int(halt_denominator.get("coverage", {}).get("covered", 0) or 0) == 194
        and halt_denominator.get("absence_is_zero") is False
        and halt_denominator.get("scorecard_eligible") is False
        and denominator_producer_hashes_valid
        and halt_denominator_validation.get("status") == "pass_partial_component"
        and int(halt_denominator_validation.get("original_source_rows", 0) or 0) == 366
        and int(halt_denominator_validation.get("fully_reconciled_claims", 0) or 0)
        == 194
        and int(halt_denominator_validation.get("validation_issue_count", -1)) == 0
        and halt_denominator_validation.get("scorecard_eligible") is False
        and denominator_validation_hashes_valid
    )
    halt_report_producer_hashes_valid = _declared_hash_inventory_valid(
        halt_report_events_dir,
        halt_report_events.get("artifact_hashes_sha256", ""),
    )
    halt_report_validation_hashes_valid = _declared_hash_inventory_valid(
        halt_report_events_validation_dir,
        halt_report_events_validation.get("artifact_hashes_sha256", ""),
    )
    halt_report_events_validated = bool(
        halt_report_events.get("status")
        == "bounded_report_event_candidate_pending_independent_validation"
        and int(halt_report_events.get("explicit_report_rows", 0) or 0) == 2711
        and int(halt_report_events.get("covered_month_channel_claims", 0) or 0)
        == 194
        and halt_report_events.get("absence_is_zero") is False
        and halt_report_producer_hashes_valid
        and halt_report_events_validation.get("status")
        == "pass_bounded_report_event_component"
        and int(halt_report_events_validation.get("producer_report_rows", 0) or 0)
        == 2711
        and int(halt_report_events_validation.get("independent_report_rows", 0) or 0)
        == 2711
        and int(halt_report_events_validation.get("exact_field_reconciled_rows", 0) or 0)
        == 2711
        and int(halt_report_events_validation.get("covered_claims_reparsed", 0) or 0)
        == 194
        and int(halt_report_events_validation.get("exact_claim_row_sets", 0) or 0)
        == 194
        and int(halt_report_events_validation.get("validation_issue_count", -1)) == 0
        and halt_report_events_validation.get("absence_is_zero") is False
        and halt_report_events_validation.get("inferred_zero_rejected") is True
        and halt_report_events_validation.get("producer_manifest_sha256")
        == _sha256(halt_report_events_dir / "manifest.json")
        and halt_report_validation_hashes_valid
    )
    halt_report_primary_merged = bool(
        halt_report_events_validated
        and halt_validated_partial
        and halt.get("halt_report_event_packet_admitted") is True
        and int(
            halt.get("halt_report_event_packet", {}).get("source_rows", 0) or 0
        )
        == 2711
        and int(
            halt.get("halt_report_event_packet", {}).get("covered_claims", 0) or 0
        )
        == 194
        and halt_validation.get("producer_run_id") == halt.get("run_id")
        and int(halt.get("coverage", {}).get("complete_month_channels", 0) or 0)
        >= 232
    )
    pre2013_valid_incomplete = bool(
        pre2013.get("status") == "valid_incomplete_public_denominator"
        and pre2013.get("immutable") is True
        and pre2013.get("archive_denominator_proven") is False
        and pre2013.get("completion_claimed") is False
        and pre2013.get("scorecard_eligible") is False
    )
    return_validated_partial = bool(
        return_normal.get("status") == "partial_reconstruction"
        and return_normal.get("immutable") is True
        and return_normal.get("full_component_coverage_claimed") is False
        and return_normal.get("scorecard_eligible") is False
        and return_validation.get("status") == "pass_partial_lane"
        and return_validation.get("scorecard_eligible") is False
        and not return_validation.get("validation_issues")
    )
    halt_counts = halt.get("counts", {}) if isinstance(halt.get("counts"), dict) else {}
    pre2013_counts = (
        pre2013.get("counts", {}) if isinstance(pre2013.get("counts"), dict) else {}
    )
    return_counts = (
        return_normal.get("counts", {})
        if isinstance(return_normal.get("counts"), dict)
        else {}
    )
    return_reasons = (
        return_normal.get("unresolved_reason_counts", {})
        if isinstance(return_normal.get("unresolved_reason_counts"), dict)
        else {}
    )
    halt_coverage = halt.get("coverage", {}) if isinstance(halt.get("coverage"), dict) else {}
    pre2013_coverage = (
        pre2013.get("coverage", {}) if isinstance(pre2013.get("coverage"), dict) else {}
    )
    market_rule_counts = (
        market_rule_validation.get("counts", {})
        if isinstance(market_rule_validation.get("counts"), dict)
        else {}
    )
    tick_assignment_counts = (
        tick_assignment_validation.get("counts", {})
        if isinstance(tick_assignment_validation.get("counts"), dict)
        else {}
    )
    market_rule_validated_partial = bool(
        market_rule_validation.get("status") == "pass_partial_component"
        and market_rule_validation.get("scorecard_eligible") is False
        and not market_rule_validation.get("validation_issues")
    )
    market_rule_coverage_by_component = {
        str(row.subcomponent): (str(row.status), float(row.coverage))
        for row in market_rule_coverage.itertuples(index=False)
        if hasattr(row, "subcomponent") and hasattr(row, "status") and hasattr(row, "coverage")
    }
    tick_rule_continuity_complete = bool(
        market_rule_validated_partial
        and market_rule_coverage_by_component.get("tick_rule_tables_2006_present")
        == ("pass_continuity_candidate", 1.0)
        and market_rule_coverage_by_component.get("pre_2014_tick_rule_table")
        == ("pass_direct_baseline_and_complete_2006_2010_amendment_ledger", 1.0)
    )
    normal_limit_continuity_complete = bool(
        market_rule_validated_partial
        and market_rule_coverage_by_component.get("normal_price_limit_table")
        == (
            "pass_two_regimes_direct_transition_and_complete_2006_2010_amendment_ledger",
            1.0,
        )
    )
    tick_assignment_validated_partial = bool(
        tick_assignment_validation.get("status") == "pass_partial_component"
        and tick_assignment_validation.get("scorecard_eligible") is False
        and not tick_assignment_validation.get("validation_issues")
    )
    tca_complete = bool(
        tca_validation.get("status") == "pass_component"
        and tca_validation.get("component") == "tca_assumptions"
        and tca_validation.get("component_scorecard_eligible") is True
        and tca_validation.get("scorecard_eligible") is False
        and not tca_validation.get("critical_blockers")
        and not tca_validation.get("validation_issues")
    )
    return {
        "halt_history": {
            "artifact_dir": str(halt_dir),
            "validation_dir": str(halt_validation_dir),
            "validated_partial": halt_validated_partial,
            "event_rows": int(halt_counts.get("event_rows", 0) or 0),
            "ric_resolved_rows": int(halt_counts.get("ric_resolved_rows", 0) or 0),
            "ric_unresolved_rows": int(halt_counts.get("ric_unresolved_rows", 0) or 0),
            "pit_mapped_rows": int(halt_counts.get("pit_mapped_rows", 0) or 0),
            "pit_timestamp_rows": int(
                halt_counts.get("historical_pit_timestamp_rows", 0) or 0
            ),
            "pit_timestamp_gap_rows": int(
                halt_counts.get("historical_pit_timestamp_gap_rows", 0) or 0
            ),
            "completion_gap_rows": int(
                halt_counts.get("completion_gap_rows", 0) or 0
            ),
            "month_channels": int(halt_coverage.get("month_channels", 0) or 0),
            "complete_month_channels": int(
                halt_coverage.get("complete_month_channels", 0) or 0
            ),
            "critical_blockers": list(halt.get("critical_blockers", []) or []),
        },
        "halt_denominator_recovery": {
            "artifact_dir": str(halt_denominator_dir),
            "validation_dir": str(halt_denominator_validation_dir),
            "report_events_dir": str(halt_report_events_dir),
            "report_events_validation_dir": str(halt_report_events_validation_dir),
            "validated_partial": halt_denominator_validated_partial,
            "source_rows": int(halt_denominator.get("source_denominator_rows", 0) or 0),
            "covered_claims": int(
                halt_denominator.get("coverage", {}).get("covered", 0) or 0
            ),
            "uncovered_claims": int(
                halt_denominator.get("coverage", {}).get("uncovered", 0) or 0
            ),
            "report_events_validated": halt_report_events_validated,
            "report_events_primary_merged": halt_report_primary_merged,
        },
        "pre2013_price_limits": {
            "artifact_dir": str(pre2013_dir),
            "valid_incomplete": pre2013_valid_incomplete,
            "required_months": int(pre2013_coverage.get("required_months", 0) or 0),
            "unresolved_months": int(
                pre2013_coverage.get("months_with_unresolved_denominator", 0) or 0
            ),
            "migrated_event_rows": int(pre2013_counts.get("migrated_event_rows", 0) or 0),
            "bilingual_matched_rows": int(
                pre2013_counts.get("bilingual_matched_notice_rows", 0) or 0
            ),
            "monthly_route_attempts": int(
                pre2013_counts.get("research_month_route_attempts", 0) or 0
            ),
            "ose_adjacency_rows": int(
                pre2013_counts.get("migrated_ose_scan_rows", 0) or 0
            ),
            "completion_plan_rows": int(
                pre2013_counts.get("completion_plan_rows", 0) or 0
            ),
            "critical_blockers": list(pre2013.get("critical_blockers", []) or []),
        },
        "return_to_normal": {
            "artifact_dir": str(return_dir),
            "validation_dir": str(return_validation_dir),
            "validated_partial": return_validated_partial,
            "activation_cases": int(return_counts.get("activation_cases", 0) or 0),
            "resolved_cases": int(return_counts.get("resolved_cases", 0) or 0),
            "unresolved_cases": int(return_counts.get("unresolved_cases", 0) or 0),
            "missing_ohlcv_cases": int(
                return_reasons.get("missing_ohlcv_without_closing_evidence", 0) or 0
            ),
            "missing_price_plan_rows": int(len(return_missing_plan)),
            "missing_price_plan_rics": int(
                return_missing_plan.get("refinitiv_ric", pd.Series(dtype=str))
                .astype(str)
                .replace("", pd.NA)
                .dropna()
                .nunique()
            ),
            "missing_price_plan_sessions": int(
                pd.to_numeric(
                    return_missing_plan.get("session_count", pd.Series(dtype=float)),
                    errors="coerce",
                )
                .fillna(0)
                .sum()
            ),
            "missing_price_plan_execution_disabled": bool(
                not return_missing_plan.empty
                and return_missing_plan.get(
                    "execution_enabled", pd.Series(dtype=bool)
                ).map(_truthy).eq(False).all()
            ),
        },
        "market_rules": {
            "validation_dir": str(market_rule_validation_dir),
            "validated_partial": market_rule_validated_partial,
            "tick_rule_continuity_complete": tick_rule_continuity_complete,
            "normal_limit_continuity_complete": normal_limit_continuity_complete,
            "tick_regime_windows": int(market_rule_counts.get("tick_regime_windows", 0) or 0),
            "tick_band_rows": int(market_rule_counts.get("tick_band_rows", 0) or 0),
            "price_limit_band_rows": int(
                market_rule_counts.get("price_limit_band_rows", 0) or 0
            ),
            "critical_blockers": list(
                market_rule_validation.get("critical_blockers", []) or []
            ),
        },
        "tick_assignment": {
            "validation_dir": str(tick_assignment_validation_dir),
            "validated_partial": tick_assignment_validated_partial,
            "assignment_events": int(
                tick_assignment_counts.get("assignment_events", 0) or 0
            ),
            "cutoff_master_names": int(
                tick_assignment_counts.get("cutoff_master_names", 0) or 0
            ),
            "cutoff_master_exact_match": bool(
                tick_assignment_counts.get("cutoff_master_exact_match", False)
            ),
            "critical_blockers": list(
                tick_assignment_validation.get("critical_blockers", []) or []
            ),
        },
        "tca": {
            "validation_dir": str(tca_validation_dir),
            "complete": tca_complete,
            "status": str(tca_validation.get("status", "missing")),
            "critical_blockers": list(tca_validation.get("critical_blockers", []) or []),
        },
    }


def _inventory_file_valid(path: Path) -> bool:
    payload = _read_json(path)
    files = payload.get("files", {})
    return bool(
        payload.get("algorithm") == "sha256"
        and payload.get("immutable") is True
        and payload.get("coverage_complete") is True
        and isinstance(files, dict)
        and files
        and all(
            Path(str(item)).is_file() and _sha256(Path(str(item))) == str(digest).lower()
            for item, digest in files.items()
        )
    )


def _declared_hash_inventory_valid(root: Path, expected_inventory_sha256: Any) -> bool:
    inventory_path = root / "artifact_hashes.json"
    inventory = _read_json(inventory_path)
    return bool(
        inventory_path.is_file()
        and str(expected_inventory_sha256)
        and _sha256(inventory_path) == str(expected_inventory_sha256)
        and inventory
        and all(
            (root / str(relative)).is_file()
            and _sha256(root / str(relative)) == str(digest)
            for relative, digest in inventory.items()
        )
    )


def _sha256_checksums_valid(root: Path, checksums_path: Path) -> bool:
    if not checksums_path.is_file():
        return False
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2:
            return False
        expected, relative = parts
        path = root / relative
        if not path.is_file() or _sha256(path) != expected.lower():
            return False
    return True


def _current_enrichment_state(manifest: dict[str, Any]) -> dict[str, Any]:
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    success = int(counts.get("success", 0) or 0)
    source = int(counts.get("source_unique_rics", 0) or 0)
    terminal_errors = int(counts.get("error", 0) or 0) + int(counts.get("failed_empty", 0) or 0)
    complete = (
        manifest.get("status") == "complete_current_observation"
        and source > 0
        and success == source
        and terminal_errors == 0
        and manifest.get("historical_pit_ready") is False
        and manifest.get("current_observation_only") is True
    )
    return {
        "complete": complete,
        "status": manifest.get("status", "missing"),
        "success": success,
        "source_unique_rics": source,
        "terminal_errors": terminal_errors,
        "historical_pit_ready": manifest.get("historical_pit_ready") is True,
    }


def _jquants_listed_info_state(manifest: dict[str, Any]) -> dict[str, Any]:
    status = str(manifest.get("status", "missing"))
    quality = manifest.get("quality_metrics", {}) if isinstance(manifest.get("quality_metrics"), dict) else {}
    planned = int(quality.get("planned_dates", 0) or 0)
    completed = int(quality.get("completed_dates", 0) or 0)
    credential_policy = manifest.get("credential_policy", {})
    source = manifest.get("source", {})
    promotion = manifest.get("promotion_flags", {})
    complete = (
        status == "complete"
        and manifest.get("dataset_id") == "jquants_pro_v2_listed_info"
        and manifest.get("research_only") is True
        and source.get("endpoint") == "https://api.jquants-pro.com/v2/listed/info"
        and source.get("api_version") == "v2"
        and manifest.get("date_plan", {}).get("immutable") is True
        and planned > 0
        and completed == planned
        and int(quality.get("pending_dates", -1)) == 0
        and int(quality.get("failed_dates", -1)) == 0
        and float(quality.get("date_coverage_ratio", 0.0)) == 1.0
        and int(quality.get("normalized_rows", 0)) > 0
        and quality.get("plan_validated_immutable") is True
        and quality.get("raw_integrity_rebuilt") is True
        and credential_policy.get("tokens_persisted") is False
        and credential_policy.get("tokens_logged") is False
        and promotion.get("scorecard_eligible") is False
        and _manifest_sha256_entries_valid(manifest)
    )
    return {
        "complete": complete,
        "status": status,
        "blocked_credentials": status in {"blocked_credentials", "blocked_missing_credentials"},
        "planned_dates": planned,
        "completed_dates": completed,
        "historical_pit_ready": False,
        "scorecard_eligible": False,
    }


def _manifest_sha256_entries_valid(manifest: dict[str, Any]) -> bool:
    artifact_dir = Path(str(manifest.get("artifact_dir", "")))
    entries = manifest.get("sha256", {})
    if not artifact_dir.is_dir() or not isinstance(entries, dict) or not entries:
        return False
    for relative, expected in entries.items():
        path = artifact_dir / str(relative)
        digest = str(expected).lower()
        if len(digest) != 64 or not path.is_file() or _sha256(path) != digest:
            return False
    checksums = artifact_dir / "checksums.sha256"
    declared = str(manifest.get("checksums_file_sha256", "")).lower()
    return checksums.is_file() and len(declared) == 64 and _sha256(checksums) == declared


def _valid_time_state(manifest: dict[str, Any]) -> dict[str, Any]:
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    rows = int(counts.get("valid_time_membership_rows", 0) or 0)
    pit_usable_rows = int(counts.get("pit_usable_membership_rows", 0) or 0)
    complete = (
        manifest.get("status") == "complete_valid_time_baseline_current_observation_only"
        and manifest.get("valid_time_baseline_ready") is True
        and manifest.get("historical_pit_ready") is False
        and rows > 0
        and pit_usable_rows == 0
    )
    return {
        "complete": complete,
        "status": manifest.get("status", "missing"),
        "valid_time_membership_rows": rows,
        "pit_usable_membership_rows": pit_usable_rows,
        "historical_pit_ready": manifest.get("historical_pit_ready") is True,
    }


def _jquants_listed_shares_state(manifest: dict[str, Any]) -> dict[str, Any]:
    status = str(manifest.get("status", "missing"))
    planned = int(manifest.get("plan_rows", 0) or 0)
    completed = int(manifest.get("completed_plan_rows", 0) or 0)
    complete = (
        status == "complete"
        and manifest.get("scope") == "official_jquants_pro_v2_listed_shares_and_shares_change"
        and planned > 0
        and completed == planned
        and int(manifest.get("pending_plan_rows", -1)) == 0
        and int(manifest.get("share_ledger_rows", 0)) > 0
        and int(manifest.get("share_change_ledger_rows", 0)) > 0
        and int(manifest.get("unusable_rows", -1)) == 0
        and manifest.get("credential_values_persisted") is False
        and manifest.get("promotion") is False
        and manifest.get("claims", {}).get("g2_complete") is False
        and _jquants_listed_shares_checksums_valid(manifest)
    )
    return {
        "complete": complete,
        "status": status,
        "blocked_credentials": status in {"blocked_credentials", "blocked_missing_credentials"},
        "planned_rows": planned,
        "completed_rows": completed,
    }


def _g1_materializer_state(manifest: dict[str, Any]) -> dict[str, Any]:
    status = str(manifest.get("status", "missing"))
    blockers = manifest.get("critical_blockers", [])
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    complete = (
        status == "complete"
        and manifest.get("artifact_type") == "japan_g1_pit_materializer_producer"
        and manifest.get("independent_audit") is False
        and manifest.get("scorecard_eligible") is False
        and manifest.get("historical_pit_ready") is True
        and not blockers
        and int(counts.get("listing_windows", counts.get("matched_segments", 0)) or 0) > 0
        and int(counts.get("pit_unusable_windows", counts.get("unresolved_segments", -1))) == 0
        and _materializer_outputs_valid(manifest)
    )
    return {
        "complete": complete,
        "status": status,
        "blocked_upstream": status == "blocked_upstream",
        "critical_blockers": blockers if isinstance(blockers, list) else [str(blockers)],
        "listing_windows": int(counts.get("listing_windows", counts.get("matched_segments", 0)) or 0),
        "unresolved_windows": int(counts.get("unresolved_segments", 0) or 0),
    }


def _materializer_outputs_valid(manifest: dict[str, Any]) -> bool:
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
        return False
    return all(
        Path(str(artifacts.get(alias, ""))).is_file()
        and _sha256(Path(str(artifacts[alias]))) == str(expected[name]).lower()
        for name, alias in aliases.items()
    )


def _g1_validator_state(manifest: dict[str, Any]) -> dict[str, Any]:
    producer = str(manifest.get("producer_run_id", ""))
    validator = str(manifest.get("validator_run_id", ""))
    complete = (
        manifest.get("status") == "pass"
        and manifest.get("artifact_type") == "japan_g1_independent_validation"
        and manifest.get("independent_audit") is True
        and manifest.get("validator_type") == "independent_process"
        and producer
        and validator
        and producer != validator
        and manifest.get("source_checksums_verified") is True
        and manifest.get("validation_status") == "pass"
        and int(manifest.get("critical_blocker_count", -1)) == 0
    )
    return {
        "complete": complete,
        "status": str(manifest.get("status", "missing")),
        "critical_blocker_count": int(manifest.get("critical_blocker_count", -1)),
    }


def _jpx_public_events_state(manifest: dict[str, Any]) -> dict[str, Any]:
    coverage = manifest.get("coverage", {}) if isinstance(manifest.get("coverage"), dict) else {}
    planned = int(coverage.get("planned_pages", 0) or 0)
    completed = int(coverage.get("completed_pages", 0) or 0)
    complete = (
        manifest.get("status") == "complete_limited_coverage"
        and manifest.get("official_jpx_source") is True
        and planned > 0
        and completed == planned
    )
    return {
        "complete": complete,
        "planned_pages": planned,
        "completed_pages": completed,
        "delisting_events": int(coverage.get("delisting_events", 0) or 0),
        "new_listing_events": int(coverage.get("new_listing_events", 0) or 0),
        "historical_pit_ready": manifest.get("historical_pit_ready") is True,
    }


def _jpx_public_crosscheck_state(manifest: dict[str, Any]) -> dict[str, Any]:
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    total = int(counts.get("events", 0) or 0)
    unresolved = int(counts.get("mapping_unmatched_or_ambiguous", 0) or 0)
    date_issues = int(counts.get("date_missing_or_mismatch", 0) or 0)
    passed = manifest.get("status") == "pass" and total > 0 and unresolved == 0 and date_issues == 0
    return {
        "passed": passed,
        "status": str(manifest.get("status", "missing")),
        "events": total,
        "unresolved_mapping_rows": unresolved,
        "date_issue_rows": date_issues,
    }


def _jpx_monthly_quotations_state(manifest: dict[str, Any]) -> dict[str, Any]:
    planned = int(manifest.get("planned_files", 0) or 0)
    completed = int(manifest.get("completed_files", 0) or 0)
    missing_urls = int(manifest.get("missing_source_urls", -1) or 0)
    candidate_complete = bool(
        manifest.get("status") == "complete_candidate_source"
        and manifest.get("official_jpx_source") is True
        and manifest.get("historical_pit_candidate") is True
        and manifest.get("plan_immutable") is True
        and planned > 0
        and completed == planned
        and missing_urls == 0
    )
    return {
        "candidate_complete": candidate_complete,
        "status": str(manifest.get("status", "missing")),
        "planned_files": planned,
        "completed_files": completed,
        "missing_source_urls": missing_urls,
        "normalized_rows": int(manifest.get("normalized_rows", 0) or 0),
    }


def _jpx_monthly_membership_qa_state(manifest: dict[str, Any]) -> dict[str, Any]:
    planned = int(manifest.get("planned_months", 0) or 0)
    completed = int(manifest.get("completed_months", manifest.get("passed_months", 0)) or 0)
    passed = bool(
        manifest.get("status") == "pass_candidate_source"
        and manifest.get("independent_validation") is True
        and planned > 0
        and completed == planned
        and int(manifest.get("anomalies", -1)) == 0
    )
    return {
        "passed_candidate": passed,
        "status": str(manifest.get("status", "missing")),
        "planned_months": planned,
        "completed_months": completed,
        "anomalies": int(manifest.get("anomalies", 0) or 0),
    }


def _jquants_listed_shares_checksums_valid(manifest: dict[str, Any]) -> bool:
    artifact_dir = Path(str(manifest.get("artifact_dir", "")))
    checksums = artifact_dir / "checksums.json"
    declared = str(manifest.get("checksums_sha256", "")).lower()
    if not artifact_dir.is_dir() or not checksums.is_file() or len(declared) != 64:
        return False
    if _sha256(checksums) != declared:
        return False
    payload = _read_json(checksums)
    entries = payload.get("artifacts", {})
    paths = manifest.get("artifacts", {})
    aliases = {
        "plan": "request_plan",
        "checkpoint": "checkpoint",
        "page_manifest": "page_manifest",
        "quality": "quality_metrics",
        "share_ledger": "share_ledger",
        "share_change_ledger": "share_change_ledger",
    }
    if not isinstance(entries, dict) or set(entries) != set(aliases):
        return False
    return all(
        Path(str(paths.get(aliases[name], ""))).is_file()
        and _sha256(Path(str(paths[aliases[name]]))) == str(digest).lower()
        for name, digest in entries.items()
    )


def _execution_queue(
    *,
    config: JapanExecutionLoopConfig,
    paths: JapanDataQualityPaths,
    lanes: dict[str, Any],
    scorecard_accepted: bool,
    universe_state: dict[str, Any],
    vendor_health: dict[str, Any],
    current_enrichment: dict[str, Any],
    valid_time: dict[str, Any],
    jquants_listed_info: dict[str, Any],
    jquants_listed_shares: dict[str, Any],
    g1_materializer: dict[str, Any],
    g1_validator: dict[str, Any],
    jpx_public_events: dict[str, Any],
    jpx_public_crosscheck: dict[str, Any],
    jpx_monthly_quotations: dict[str, Any],
    jpx_monthly_membership_qa: dict[str, Any],
    lock_state: dict[str, Any],
    plan_states: dict[str, dict[str, Any]],
    active_recovery: dict[str, Any],
    g2_vendor: dict[str, Any],
    g2_postprocess: dict[str, Any],
    g4_partial_evidence: dict[str, Any],
    active_symbol_map_path: Path,
    reference_symbol_map_path: Path,
    active_quarantine_path: Path,
    reference_quarantine_path: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(
        quality_gate: str,
        owner: str,
        phase: str,
        status: str,
        depends_on: str,
        command: str,
        acceptance_gate: str,
        notes: str,
        live: bool = False,
        plan: dict[str, Any] | None = None,
    ) -> None:
        rows.append(
            {
                "sequence": len(rows) + 1,
                "quality_gate": quality_gate,
                "owner_agent": owner,
                "phase": phase,
                "status": status,
                "depends_on": depends_on,
                "command": command,
                "acceptance_gate": acceptance_gate,
                "notes": notes,
                "live_extraction": live,
                "bounded_max_plan_rows": (
                    config.active_max_plan_rows if phase == "active_ohlcv" else config.reference_max_plan_rows
                    if phase == "reference_ohlcv"
                    else 0
                ),
                "plan_path": plan.get("plan_path", "") if plan else "",
                "expected_plan_sha256": plan.get("sha256", "") if plan else "",
                "writer_lock_marker": lock_state["marker_path"] if live else "",
                "promotes_alpha": False,
                "promotes_production": False,
                "promotes_ibkr": False,
            }
        )

    if universe_state["eligible"]:
        g1_status = "passed"
        g1_command = ""
        g1_notes = f"Eligible active/reference rows: {universe_state['active_rows']}/{universe_state['reference_rows']}."
    else:
        g1_status = "blocked_requires_historical_pit_repair"
        g1_command = ""
        g1_notes = (
            "Repair G1 end to end: use a JPX-only official listing input; enrich currency and listing windows; "
            "build source-backed historical membership/known-at history; classify every denominator difference; "
            "then independently validate active and reference maps. Current-listing mining alone is insufficient."
        )
    add(
        "G1",
        "jpx_universe_reconciliation_agent",
        "jpx_universe_reconciliation",
        g1_status,
        "",
        g1_command,
        "A listing-level JPX master, historical PIT membership, complete reconciliation, and independently validated active/reference maps must all pass.",
        g1_notes,
    )

    direct_session_test = bool(active_recovery["required"])
    add(
        "G1",
        "vendor_health_guard",
        "vendor_health_7203T",
        "superseded_by_direct_session_test"
        if direct_session_test
        else "passed"
        if vendor_health["valid"]
        else "blocked_retry_probe_required",
        "G1:current_scope",
        "" if direct_session_test else _vendor_health_command(config),
        "Use the one-row staged recovery as the session test when a recovery denominator exists; otherwise require a fresh 7203.T probe.",
        "The redundant quote probe is disabled; the staged payload remains isolated and non-promoting."
        if direct_session_test
        else "; ".join(vendor_health["reasons"]) or "Fresh Japan vendor evidence is present.",
    )

    add(
        "G1",
        "jpx_listing_pit_enrichment_agent",
        "jpx_listing_master_pit_enrichment",
        "passed_current_observation_only"
        if current_enrichment["complete"]
        else "ready"
        if vendor_health["valid"]
        else "queued_after_vendor_health",
        "G1:current_scope,VENDOR_HEALTH",
        " ".join(
            shlex.quote(value)
            for value in (
                "uv",
                "run",
                "--extra",
                "refinitiv",
                "python",
                "examples/run_japan_g1_enrichment.py",
                "--source-dir",
                str(paths.universe_reconciliation_dir),
                "--run-id",
                f"{paths.universe_reconciliation_dir.name}-G1-CURRENT-ENRICHMENT-V2",
                "--max-rics",
                "25",
                "--request-chunk-size",
                "25",
                "--session",
                config.refinitiv_session,
            )
        ),
        "Enrich the listing-level master and historical membership with source-backed currency, listing windows, identifier changes, and known-at timestamps; publish validated active/reference maps.",
        (
            f"Current observation complete at {current_enrichment['success']}/"
            f"{current_enrichment['source_unique_rics']} with zero terminal errors; it remains non-PIT evidence."
            if current_enrichment["complete"]
            else "Use bounded Refinitiv requests and independent denominator reconciliation. Do not promote a current snapshot as historical PIT evidence."
        ),
        live=not current_enrichment["complete"],
    )

    add(
        "G1",
        "jpx_listing_valid_time_agent",
        "jpx_listing_valid_time_baseline",
        "passed_valid_time_only" if valid_time["complete"] else "ready",
        "G1:current_observation",
        ""
        if valid_time["complete"]
        else "uv run python examples/build_japan_listing_valid_time.py",
        "Build listing-level valid-time intervals while keeping unknown source publication times null and PIT-ineligible.",
        (
            f"Valid-time baseline complete for {valid_time['valid_time_membership_rows']} listings; "
            f"PIT-usable rows remain {valid_time['pit_usable_membership_rows']} by design."
            if valid_time["complete"]
            else "This step must not synthesize known_at from effective dates or current observation timestamps."
        ),
    )

    add(
        "G1",
        "jpx_public_archive_agent",
        "jpx_public_listing_event_archive",
        "passed_limited_crosscheck_only" if jpx_public_events["complete"] else "ready",
        "G1:current_observation",
        ""
        if jpx_public_events["complete"]
        else "uv run python examples/run_jpx_public_listing_events.py --run-id JPX-PUBLIC-LISTING-EVENTS-20260712 --max-pages 20",
        "Complete every discoverable official public archive page with raw hashes while explicitly retaining its incomplete historical coverage.",
        f"Pages {jpx_public_events['completed_pages']}/{jpx_public_events['planned_pages']}; delistings={jpx_public_events['delisting_events']}; new listings={jpx_public_events['new_listing_events']}; full PIT={jpx_public_events['historical_pit_ready']}.",
    )
    add(
        "G1",
        "jpx_public_crosscheck_auditor",
        "jpx_public_event_crosscheck",
        "passed_diagnostic_crosscheck"
        if jpx_public_crosscheck["passed"]
        else "diagnostic_discrepancies_nonadmission",
        "G1:jpx_public_listing_event_archive",
        "",
        "Recent public-event cross-checks are retained as diagnostics; only the full monthly G1 denominator and independent validator control admission.",
        f"Events={jpx_public_crosscheck['events']}; unresolved mappings={jpx_public_crosscheck['unresolved_mapping_rows']}; date issues={jpx_public_crosscheck['date_issue_rows']}.",
    )

    add(
        "G1",
        "jpx_monthly_snapshot_agent",
        "jpx_monthly_quotation_pit_source",
        "passed_candidate_source_only"
        if jpx_monthly_quotations["candidate_complete"]
        else "in_progress"
        if jpx_monthly_quotations["completed_files"] > 0
        else "ready",
        "G1:official_public_sources",
        ""
        if jpx_monthly_quotations["candidate_complete"]
        else "uv run python examples/run_jpx_monthly_quotations.py --run-id JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2 --max-files 180 --pacing-seconds 0.08",
        "All 1,383 official monthly quotation files must be hash-verified and normalized before event replay and independent G1 validation.",
        (
            f"Official files={jpx_monthly_quotations['completed_files']}/"
            f"{jpx_monthly_quotations['planned_files']}; normalized_rows="
            f"{jpx_monthly_quotations['normalized_rows']}; missing_urls="
            f"{jpx_monthly_quotations['missing_source_urls']}. This is a candidate source, never direct G1 admission."
        ),
        live=not jpx_monthly_quotations["candidate_complete"],
    )
    add(
        "G1",
        "jpx_monthly_snapshot_auditor",
        "jpx_monthly_membership_continuity_qa",
        "passed_candidate_qa_only"
        if jpx_monthly_membership_qa["passed_candidate"]
        else "in_progress_partial"
        if jpx_monthly_membership_qa["completed_months"] > 0
        else "queued_after_monthly_snapshots",
        "G1:jpx_monthly_quotation_pit_source",
        ""
        if jpx_monthly_membership_qa["passed_candidate"]
        else "uv run python examples/run_japan_combined_monthly_membership_qa.py",
        "Every completed month must pass file-hash, code uniqueness, PIT timestamp, JPY, and cross-month churn checks.",
        (
            f"Months={jpx_monthly_membership_qa['completed_months']}/"
            f"{jpx_monthly_membership_qa['planned_months']}; anomalies="
            f"{jpx_monthly_membership_qa['anomalies']}; partial months are not compared for churn."
        ),
    )

    unresolved_windows = g1_materializer.get("unresolved_windows", 0)
    targeted_complete = g1_materializer["complete"] or unresolved_windows == 0
    targeted_ready = (
        not targeted_complete
        and jpx_monthly_membership_qa["passed_candidate"]
        and vendor_health["valid"]
    )
    add(
        "G1",
        "japan_g1_targeted_identifier_agent",
        "japan_g1_targeted_identifier_discovery",
        "passed"
        if targeted_complete
        else "ready"
        if targeted_ready
        else "blocked_vendor_health"
        if jpx_monthly_membership_qa["passed_candidate"]
        else "queued_after_monthly_membership_qa",
        "G1:jpx_monthly_membership_continuity_qa",
        "uv run python examples/run_japan_g1_targeted_discovery.py --chunk-size 20 --max-codes 100 --sleep-seconds 0.05"
        if targeted_ready
        else "",
        "Resolve every missing historical JPX code with bounded primary-exchange Discovery shards; never force an unmatched RIC.",
        f"Unresolved listing windows={unresolved_windows}; fresh 7203.T vendor health={vendor_health['valid']}.",
        live=targeted_ready,
    )

    if jquants_listed_info["complete"]:
        jquants_status = "passed_raw_history_only"
        jquants_command = ""
        jquants_notes = "Official raw historical listed-info extraction is complete and hash-verified; a separate materializer and independent validator must still produce G1 evidence."
    elif jquants_listed_info["blocked_credentials"]:
        jquants_status = "blocked_by_jquants_credentials"
        jquants_command = ""
        jquants_notes = "Immutable request plan exists, but JQUANTS_ID_TOKEN or JQUANTS_REFRESH_TOKEN is not present in the process environment."
    else:
        jquants_status = "ready_generate_plan_or_bounded_backfill"
        jquants_command = _jquants_listed_info_command(config)
        jquants_notes = "Generate the immutable official JPX date plan first; a credential-free run must stop after publishing a blocked-credentials manifest."
    add(
        "G1",
        "jquants_historical_membership_agent",
        "jquants_historical_listed_info",
        jquants_status,
        "G1:valid_time_baseline",
        jquants_command,
        "Official full-history listed-info evidence must preserve raw responses, endpoint/version lineage, and source-backed availability semantics before historical membership can pass.",
        jquants_notes,
        live=not jquants_listed_info["blocked_credentials"] and not jquants_listed_info["complete"],
    )

    if g1_materializer["complete"]:
        materializer_status = "passed_producer_candidate_only"
        materializer_command = ""
        materializer_notes = f"Producer candidate is complete for {g1_materializer['listing_windows']} windows; independent validation is still mandatory."
    elif jpx_monthly_membership_qa["passed_candidate"]:
        materializer_status = (
            "ready_rerun_after_targeted_discovery"
            if targeted_complete
            else "queued_after_targeted_identifier_discovery"
        )
        materializer_command = _g1_materializer_command(config)
        materializer_notes = (
            f"Official 240-month source is complete; {unresolved_windows} identifier windows remain before producer admission."
        )
    else:
        materializer_status = "queued_after_monthly_membership_qa"
        materializer_command = ""
        materializer_notes = "This producer can emit candidates only after the official combined monthly source passes independent QA."
    add(
        "G1",
        "japan_g1_pit_materializer_agent",
        "japan_g1_pit_materializer",
        materializer_status,
        "G1:japan_g1_targeted_identifier_discovery",
        materializer_command,
        "Materialize hash-verified listing-level PIT candidates with zero ambiguous/unresolved/currency-missing windows and conservative official-schedule known-at timestamps.",
        materializer_notes,
    )

    if g1_validator["complete"]:
        validator_status = "passed"
        validator_command = ""
        validator_notes = "Independent producer/validator separation and all G1 candidate invariants passed."
    else:
        validator_status = "ready" if g1_materializer["complete"] else "queued_after_materializer"
        validator_command = _g1_validator_command(config) if g1_materializer["complete"] else ""
        validator_notes = "Only this separate validator may publish the admitted security master and active/reference maps."
    add(
        "G1",
        "japan_g1_independent_validator",
        "japan_g1_independent_validation",
        validator_status,
        "G1:japan_g1_pit_materializer",
        validator_command,
        "Recompute all input/output hashes and pass map, denominator, window, currency, known-at, and independence checks before publishing admitted G1 artifacts.",
        validator_notes,
    )

    if jquants_listed_shares["complete"]:
        shares_status = "passed_raw_share_history_only"
        shares_command = ""
        shares_notes = "Official listed-share and share-change ledgers are complete and hash-verified; comprehensive dividends and corporate actions remain separate G2 evidence."
    elif jquants_listed_shares["blocked_credentials"]:
        shares_status = "blocked_by_jquants_credentials"
        shares_command = ""
        shares_notes = "The immutable share/factor plan exists, but J-Quants credentials are absent."
    else:
        shares_status = "ready_generate_plan_or_bounded_backfill"
        shares_command = _jquants_listed_shares_command(config)
        shares_notes = "This REST lane does not claim comprehensive dividends, delistings, or G2 completion."
    add(
        "G2",
        "jquants_listed_shares_agent",
        "jquants_listed_shares_history",
        shares_status,
        "G1:jquants_historical_listed_info",
        shares_command,
        "Complete and hash-verify both official listed-share ledgers with UpdatedAt-derived known-at timestamps; keep comprehensive corporate actions separate.",
        shares_notes,
        live=not jquants_listed_shares["blocked_credentials"] and not jquants_listed_shares["complete"],
    )

    active_complete = _plan_complete(
        _read_csv(Path(plan_states["active"]["plan_path"]).parent / "historical_manifest.csv"),
        int(plan_states["active"]["rows"]),
    )
    reference_complete = _plan_complete(
        _read_csv(Path(plan_states["reference"]["plan_path"]).parent / "historical_manifest.csv"),
        int(plan_states["reference"]["rows"]),
    )
    sequential_vendor_ready = bool(vendor_health["valid"] or active_recovery["passed"])

    active_status = _live_status(
        complete=active_complete,
        universe_ok=bool(universe_state["eligible"]),
        vendor_ok=sequential_vendor_ready,
        plan_ok=bool(plan_states["active"]["valid"]),
        lock_exists=bool(lock_state["marker_exists"]),
    )
    if active_recovery["required"]:
        if active_recovery["passed"]:
            recovery_status = "passed"
        elif active_recovery["initial_gate_passed"]:
            recovery_status = "ready_continue_quarantined_recovery"
        elif lock_state["marker_exists"]:
            recovery_status = "blocked_by_writer_lock"
        else:
            recovery_status = "ready_direct_session_test"
        add(
            "G2",
            "jpx_active_ohlcv_recovery_agent",
            "active_ohlcv_quota_recovery",
            recovery_status,
            "G1",
            _active_recovery_command(config, active_symbol_map_path, active_recovery),
            "Every quota-failed active row must produce a clean staged payload and pass a guarded production merge.",
            (
                "The first staged row is the direct session gate; recovery then continues exactly one row at a time. "
                f"Remaining quarantined rows={active_recovery['remaining_rows']}; "
                f"next={active_recovery['next_plan_row_id'] or 'none'}."
            ),
            live=True,
        )
        if not active_recovery["passed"]:
            active_status = "queued_after_quota_recovery"
    add(
        "G2",
        "jpx_active_ohlcv_agent",
        "active_ohlcv",
        active_status,
        "G1,G2:active_ohlcv_quota_recovery" if active_recovery["required"] else "G1,VENDOR_HEALTH",
        _historical_command(
            config,
            config.active_historical_run_id,
            active_symbol_map_path,
            active_quarantine_path,
            plan_states["active"],
            config.active_max_plan_rows,
        ),
        "A bounded active chunk must preserve the plan checksum and produce usable OHLCV or structural evidence.",
        "No alpha, production, or trading promotion is permitted.",
        live=True,
        plan=plan_states["active"],
    )
    active_post = g2_postprocess["active"]
    add(
        "G2",
        "jpx_active_ohlcv_auditor",
        "active_ohlcv_postprocessing",
        "passed"
        if active_post["complete"]
        else "queued_after_active"
        if not active_complete
        else "ready",
        "G2:active_ohlcv",
        _ohlcv_postprocess_command(
            config,
            role="active",
            historical_run_id=config.active_historical_run_id,
            listing_windows_path=paths.active_symbol_map_dir / "symbol_map.csv",
            lane=active_post,
        ),
        "Full active OHLCV must be listing-window canonical, depth-clean, row-equal after compaction, and covered by an immutable hash inventory.",
        "Offline bounded postprocessing; no Refinitiv session is opened.",
    )
    if reference_complete:
        reference_status = "passed"
    elif not active_complete:
        reference_status = "queued_after_active"
    elif not active_post["complete"]:
        reference_status = "queued_after_active_postprocessing"
    else:
        reference_status = _live_status(
            complete=False,
            universe_ok=bool(universe_state["eligible"]),
            vendor_ok=sequential_vendor_ready,
            plan_ok=bool(plan_states["reference"]["valid"]),
            lock_exists=bool(lock_state["marker_exists"]),
        )
    add(
        "G2",
        "jpx_reference_ohlcv_agent",
        "reference_ohlcv",
        reference_status,
        "G2:active_ohlcv_postprocessing",
        _historical_command(
            config,
            config.reference_historical_run_id,
            reference_symbol_map_path,
            reference_quarantine_path,
            plan_states["reference"],
            config.reference_max_plan_rows,
        ),
        "The inactive/delisted reference lane starts only after active OHLCV and uses the same bounded controls.",
        "Reference data is research evidence only.",
        live=True,
        plan=plan_states["reference"],
    )
    reference_post = g2_postprocess["reference"]
    add(
        "G2",
        "jpx_reference_ohlcv_auditor",
        "reference_ohlcv_postprocessing",
        "passed"
        if reference_post["complete"]
        else "queued_after_reference"
        if not reference_complete
        else "ready",
        "G2:reference_ohlcv",
        _ohlcv_postprocess_command(
            config,
            role="reference",
            historical_run_id=config.reference_historical_run_id,
            listing_windows_path=paths.reference_symbol_map_dir / "symbol_map.csv",
            lane=reference_post,
        ),
        "Full inactive/delisted reference OHLCV must independently pass the same canonical, depth, compaction, equality, and hash contracts.",
        "Offline bounded postprocessing; no Refinitiv session is opened.",
    )

    if g2_vendor["corporate_action_complete"]:
        corporate_status = "passed"
    elif not reference_post["complete"]:
        corporate_status = "queued_after_reference_postprocessing"
    elif not g2_vendor["plan_valid"]:
        corporate_status = "blocked_by_plan_checksum"
    elif not sequential_vendor_ready:
        corporate_status = "blocked_by_vendor_health"
    elif lock_state["marker_exists"]:
        corporate_status = "blocked_by_writer_lock"
    else:
        corporate_status = "ready"
    add(
        "G2",
        "jpx_corporate_actions_agent",
        "g2_corporate_actions",
        corporate_status,
        "G2:reference_ohlcv_postprocessing",
        _corporate_action_command(config, g2_vendor),
        "Every immutable dividend and capital-change row must terminate successfully with official PIT event evidence or a successful empty response.",
        "One sequential Refinitiv writer; the 462-row denominator is resumed without regenerating the plan.",
        live=True,
    )

    if g2_vendor["liquidity_field_probe_complete"]:
        field_probe_status = "passed"
    elif not g2_vendor["corporate_action_complete"]:
        field_probe_status = "queued_after_corporate_actions"
    elif not g2_vendor["plan_valid"]:
        field_probe_status = "blocked_by_plan_checksum"
    elif not sequential_vendor_ready:
        field_probe_status = "blocked_by_vendor_health"
    elif lock_state["marker_exists"]:
        field_probe_status = "blocked_by_writer_lock"
    else:
        field_probe_status = "ready"
    add(
        "G2",
        "jpx_liquidity_route_agent",
        "g2_liquidity_field_probe",
        field_probe_status,
        "G2:g2_corporate_actions",
        _liquidity_field_probe_command(config, g2_vendor),
        "Market cap, free float, and shares outstanding must each return usable Japan observations before history is requested.",
        "Nine deterministic field probes run serially against three fixed JPX RICs.",
        live=True,
    )

    if g2_vendor["liquidity_historical_probe_complete"]:
        history_probe_status = "passed"
    elif not g2_vendor["liquidity_field_probe_complete"]:
        history_probe_status = "queued_after_liquidity_field_probe"
    elif not sequential_vendor_ready:
        history_probe_status = "blocked_by_vendor_health"
    elif lock_state["marker_exists"]:
        history_probe_status = "blocked_by_writer_lock"
    else:
        history_probe_status = "ready"
    add(
        "G2",
        "jpx_liquidity_history_probe_agent",
        "g2_liquidity_historical_probe",
        history_probe_status,
        "G2:g2_liquidity_field_probe",
        _liquidity_historical_probe_command(config, g2_vendor),
        "All three accepted liquidity roles must prove historical availability inside the 2006-present request window.",
        "This remains a tiny route test and cannot promote G2.",
        live=True,
    )

    if g2_vendor["liquidity_complete"]:
        liquidity_status = "passed"
    elif not g2_vendor["liquidity_historical_probe_complete"]:
        liquidity_status = "queued_after_liquidity_historical_probe"
    elif not sequential_vendor_ready:
        liquidity_status = "blocked_by_vendor_health"
    elif lock_state["marker_exists"]:
        liquidity_status = "blocked_by_writer_lock"
    else:
        liquidity_status = "ready"
    add(
        "G2",
        "jpx_liquidity_backfill_agent",
        "g2_liquidity_historical_backfill",
        liquidity_status,
        "G2:g2_liquidity_historical_probe",
        _liquidity_historical_backfill_command(config, g2_vendor),
        "The full G1 RIC denominator must have PIT liquidity history, complete terminal rows, no duplicates, and no outside-window observations.",
        "Bounded resumable monthly history; one sequential Refinitiv writer.",
        live=True,
    )

    if g2_postprocess["price_union_complete"]:
        price_union_status = "passed"
    elif not g2_vendor["liquidity_complete"]:
        price_union_status = "queued_after_liquidity"
    elif not (active_post["complete"] and reference_post["complete"]):
        price_union_status = "queued_after_ohlcv_postprocessing"
    else:
        price_union_status = "ready"
    add(
        "G2",
        "jpx_price_union_auditor",
        "g2_active_reference_price_union",
        price_union_status,
        "G2:g2_liquidity_historical_backfill,G2:active_ohlcv_postprocessing,G2:reference_ohlcv_postprocessing",
        _price_union_command(config, g2_postprocess),
        "The active/reference compacted datasets must form one immutable, schema-identical, source-hash-stable union with unique ticker/date keys.",
        "The producer scans in bounded Arrow batches and uses an on-disk uniqueness index.",
    )

    if g2_postprocess["adjusted_complete"]:
        adjusted_status = "passed"
    elif not g2_postprocess["price_union_complete"]:
        adjusted_status = "queued_after_price_union"
    elif not g2_vendor["corporate_action_complete"]:
        adjusted_status = "queued_after_corporate_actions"
    else:
        adjusted_status = "ready"
    add(
        "G2",
        "jpx_adjusted_return_agent",
        "g2_adjusted_return_reconstruction",
        adjusted_status,
        "G2:g2_active_reference_price_union,G2:g2_corporate_actions",
        _adjusted_returns_command(config, g2_vendor, g2_postprocess),
        "Adjusted-return keys must exactly equal the raw active/reference union with 100% event matching and zero quarantine or ambiguous factors.",
        "This is an offline reconstruction; it does not promote alpha or trading.",
    )

    if g2_postprocess["component_inventory_complete"]:
        component_hash_status = "passed"
    elif not (
        g2_postprocess["adjusted_complete"]
        and g2_vendor["liquidity_complete"]
        and g2_vendor["corporate_action_complete"]
    ):
        component_hash_status = "queued_after_components"
    else:
        component_hash_status = "ready"
    add(
        "G2",
        "g2_component_hash_auditor",
        "g2_component_hash_inventory",
        component_hash_status,
        "G2:g2_adjusted_return_reconstruction,G2:g2_liquidity_historical_backfill",
        _component_inventory_command(config, paths, g2_vendor, g2_postprocess),
        "Every G1, corporate-action, liquidity, price-union, and adjusted-return source file must be covered by a verified immutable SHA-256 inventory.",
        "The inventory is write-once and rejects source drift.",
    )

    if g2_postprocess["admission_complete"]:
        strict_admission_status = "passed"
    elif not g2_postprocess["component_inventory_complete"]:
        strict_admission_status = "queued_after_component_hashes"
    else:
        strict_admission_status = "ready"
    add(
        "G2",
        "independent_g2_admission_auditor",
        "g2_strict_aggregate_admission",
        strict_admission_status,
        "G2:g2_component_hash_inventory",
        _strict_g2_admission_command(config, paths, g2_vendor, g2_postprocess),
        "All strict Japan G2 checks must pass with zero critical blockers and stable sources during evaluation.",
        "A failed admission artifact is evidence of remaining work, never partial G2 credit.",
    )

    add(
        "G2",
        "storage_ops_agent",
        "g2_depth_compaction_audit",
        "passed" if g2_postprocess["admission_complete"] else "queued_after_strict_g2_admission",
        "G2:g2_strict_aggregate_admission",
        "",
        "OHLCV, actions, liquidity, depth, and row-equal compaction must all pass G2.",
        "Strict aggregate G2 admission is authoritative; the legacy path-based lane is not used for promotion.",
    )
    g3_plan_manifest = _read_json(_under_root(config.project_root, config.g3_pit_plan_dir) / "manifest.json")
    g3_registry_manifest = _read_json(
        _under_root(config.project_root, config.g3_route_registry_dir) / "manifest.json"
    )
    g3_qa_manifest = _read_json(_under_root(config.project_root, config.g3_fundamentals_qa_dir) / "manifest.json")
    g3_sector_manifest = _read_json(
        _under_root(config.project_root, config.g3_sector_snapshot_dir) / "manifest.json"
    )
    g3_sector_gap_manifest = _read_json(
        _under_root(config.project_root, config.g3_sector_gap_dir) / "manifest.json"
    )
    g3_sector_full_manifest = _read_json(
        _under_root(config.project_root, config.g3_sector_full_dir) / "manifest.json"
    )
    g3_ibes_events_manifest = _read_json(
        _under_root(config.project_root, config.g3_ibes_events_qa_dir) / "manifest.json"
    )
    g3_execution_manifest = _read_json(
        _under_root(config.project_root, config.g3_execution_plan_dir) / "manifest.json"
    )
    g5_preflight_manifest = _read_json(paths.leakage_qa_dir / "manifest.json")
    plan_prepared = bool(
        g3_plan_manifest.get("point_in_time_plan") is True
        and int(g3_plan_manifest.get("selected_names", 0)) == 5_752
        and int(g3_plan_manifest.get("selected_listing_windows", 0)) == 9_706
        and g3_plan_manifest.get("window_denominator_preserved") is True
        and g3_plan_manifest.get("current_survivor") is False
        and set(g3_plan_manifest.get("datasets", []))
        >= {"fundamentals_pit", "ibes_estimates_revisions", "earnings_events_actuals"}
    )
    add(
        "G3",
        "japan_g3_plan_agent",
        "g3_pit_plan_scaffold",
        "passed" if plan_prepared and g3_plan_manifest.get("scorecard_eligible") is True else "blocked_invalid_denominator" if plan_prepared else "ready",
        "G1",
        "" if plan_prepared else "uv run python examples/build_japan_g3_pit_data_plan.py --lseg-connector-status healthy",
        "The full Japan G3 denominator and immutable queue must exist; scorecard admission still requires independent G1.",
        f"rics={g3_plan_manifest.get('selected_names', 0)}; windows={g3_plan_manifest.get('selected_listing_windows', 0)}; scorecard_eligible={g3_plan_manifest.get('scorecard_eligible')}",
    )
    registry_ready = bool(
        g3_registry_manifest.get("status") == "complete_qa_routes"
        and int(g3_registry_manifest.get("required_roles", 0)) == int(g3_registry_manifest.get("accepted_roles", -1))
        and int(g3_registry_manifest.get("unresolved_roles", -1)) == 0
    )
    add(
        "G3",
        "japan_g3_route_validation_agent",
        "g3_route_registry",
        "passed_qa_routes_only" if registry_ready else "ready",
        "VENDOR_HEALTH,G3:g3_pit_plan_scaffold",
        "" if registry_ready else "uv run python examples/build_japan_g3_route_registry.py",
        "All 13 semantic roles require source-backed routes, with period-end forbidden as a release timestamp.",
        f"accepted_roles={g3_registry_manifest.get('accepted_roles', 0)}; unresolved={g3_registry_manifest.get('unresolved_roles', 0)}; full history is still required",
    )
    qa_ready = bool(
        g3_qa_manifest.get("status") == "completed_partial"
        and int(g3_qa_manifest.get("history_rows", 0)) > 0
        and float(g3_qa_manifest.get("known_at_coverage", 0.0)) == 1.0
        and int(g3_qa_manifest.get("pit_timestamp_violations", -1)) == 0
        and g3_qa_manifest.get("scorecard_eligible") is False
    )
    add(
        "G3",
        "japan_g3_fundamentals_agent",
        "g3_parameterized_fundamentals_qa",
        "passed_qa_only" if qa_ready else "queued_after_strict_g2" if not g2_postprocess["admission_complete"] else "ready" if sequential_vendor_ready else "blocked_by_vendor_health",
        "G2:g2_strict_aggregate_admission,VENDOR_HEALTH,G3:g3_route_registry",
        "" if qa_ready else "uv run --extra refinitiv python examples/run_japan_g3_fundamentals_qa.py",
        "A bounded 20-year shard must retain calc dates and report zero PIT violations; it never satisfies full G3.",
        f"history_rows={g3_qa_manifest.get('history_rows', 0)}; known_at={g3_qa_manifest.get('known_at_coverage', 0)}; scorecard_eligible=False",
        live=not qa_ready,
    )
    sector_ready = bool(
        g3_sector_full_manifest.get("status") == "complete_current_routing"
        and int(g3_sector_full_manifest.get("requested_rics", 0)) == 5_752
        and int(g3_sector_full_manifest.get("profile_rics", 0)) == 5_752
        and float(g3_sector_full_manifest.get("coverage", 0.0)) == 1.0
        and g3_sector_full_manifest.get("point_in_time_sector_claimed") is False
        and g3_sector_full_manifest.get("scorecard_eligible") is False
    )
    sector_gap_ready = bool(
        g3_sector_gap_manifest.get("status") == "complete_plan"
        and int(g3_sector_gap_manifest.get("selected_rics", 0)) == 5_752
        and int(g3_sector_gap_manifest.get("existing_profile_rics", 0)) == 3_707
        and int(g3_sector_gap_manifest.get("missing_rics", 0)) == 2_045
        and int(g3_sector_gap_manifest.get("plan_rows", 0)) == 82
        and g3_sector_gap_manifest.get("denominator_reconciles") is True
    )
    add(
        "G3",
        "japan_g3_sector_routing_agent",
        "g3_current_sector_routing",
        "passed_full_denominator_routing_only" if sector_ready else "blocked_missing_gap_plan" if not sector_gap_ready else "queued_after_strict_g2" if not g2_postprocess["admission_complete"] else "ready" if sequential_vendor_ready else "blocked_by_vendor_health",
        "G2:g2_strict_aggregate_admission,VENDOR_HEALTH,G3:g3_pit_plan_scaffold",
        "" if sector_ready else _g3_sector_gap_command(config, g3_sector_gap_manifest),
        "All 5,752 historical RICs need a deterministic field route; current classification remains routing-only, not PIT sector membership.",
        f"baseline={g3_sector_manifest.get('profile_rics', 0)}; missing={g3_sector_gap_manifest.get('missing_rics', 0)}; full={g3_sector_full_manifest.get('profile_rics', 0)}; scorecard_eligible=False",
        live=not sector_ready,
    )
    ibes_events_ready = bool(
        g3_ibes_events_manifest.get("status") == "completed_partial"
        and g3_ibes_events_manifest.get("self_keyed_analyst_fields") is True
        and g3_ibes_events_manifest.get("mixed_target_and_recommendation_query_allowed") is False
        and int(g3_ibes_events_manifest.get("revision_event_rows", 0)) > 0
        and int(g3_ibes_events_manifest.get("earnings_event_rows", 0)) > 0
        and float(g3_ibes_events_manifest.get("known_at_coverage", 0.0)) == 1.0
        and int(g3_ibes_events_manifest.get("pit_timestamp_violations", -1)) == 0
        and g3_ibes_events_manifest.get("scorecard_eligible") is False
    )
    add(
        "G3",
        "japan_g3_ibes_events_agent",
        "g3_self_keyed_ibes_events_qa",
        "passed_qa_only" if ibes_events_ready else "queued_after_strict_g2" if not g2_postprocess["admission_complete"] else "ready" if sequential_vendor_ready else "blocked_by_vendor_health",
        "G2:g2_strict_aggregate_admission,VENDOR_HEALTH,G3:g3_route_registry",
        "" if ibes_events_ready else "uv run --extra refinitiv python examples/run_japan_g3_ibes_events_qa.py",
        "Self-keyed analyst and target families, parameterized actuals, and zero PIT violations are mandatory; mixed row dimensions are forbidden.",
        f"revision_events={g3_ibes_events_manifest.get('revision_event_rows', 0)}; earnings_events={g3_ibes_events_manifest.get('earnings_event_rows', 0)}; rejected_misaligned={g3_ibes_events_manifest.get('rejected_misaligned_event_rows', 0)}",
        live=not ibes_events_ready,
    )
    execution_plan_ready = bool(
        g3_execution_manifest.get("evidence_scope") == "full_execution_plan"
        and int(g3_execution_manifest.get("selected_names", 0)) == 5_752
        and int(g3_execution_manifest.get("ordinary_workspace_plan_rows", 0)) > 0
        and int(g3_execution_manifest.get("analyst_detail_plan_rows", 0)) > 0
        and g3_execution_manifest.get("full_workspace_analyst_detail_auto_start") is False
    )
    add(
        "G3",
        "japan_g3_execution_planner",
        "g3_route_aware_execution_plan",
        "passed" if execution_plan_ready and g3_execution_manifest.get("g1_independent_validation_admitted") else "queued_after_full_sector_routing" if not sector_ready else "ready",
        "G1,G3:g3_route_registry,G3:g3_current_sector_routing,G3:g3_self_keyed_ibes_events_qa",
        "" if execution_plan_ready else _g3_execution_plan_command(config),
        "Route-aware full plans must be immutable; ordinary Workspace lanes and bulk-only analyst detail must remain separate.",
        f"ordinary_plan_rows={g3_execution_manifest.get('ordinary_workspace_plan_rows', 0)}; estimated_field_calls={g3_execution_manifest.get('ordinary_workspace_estimated_field_calls', 0)}; analyst_detail_plan_rows={g3_execution_manifest.get('analyst_detail_plan_rows', 0)}; analyst_auto_start=False",
    )
    halt_state = g4_partial_evidence["halt_history"]
    add(
        "G4",
        "jpx_halt_history_agent",
        "g4_halt_history_evidence",
        "validated_partial_completion_required"
        if halt_state["validated_partial"]
        else "ready_offline_evidence_build",
        "G1",
        ""
        if halt_state["validated_partial"]
        else "uv run python examples/run_japan_g4_halt_history.py",
        "All 480 month-channels require source-backed event or no-event evidence, complete PIT timestamps, exact G1 identity mapping, and independent validation.",
        (
            f"events={halt_state['event_rows']}; complete_month_channels="
            f"{halt_state['complete_month_channels']}/{halt_state['month_channels']}; blockers="
            f"{';'.join(halt_state['critical_blockers']) or 'artifact_missing'}; ric_mapped="
            f"{halt_state['ric_resolved_rows']}/{halt_state['event_rows']}; pit_mapped="
            f"{halt_state['pit_mapped_rows']}; pit_timestamps="
            f"{halt_state['pit_timestamp_rows']}; completion_gaps="
            f"{halt_state['completion_gap_rows']}"
        ),
    )
    halt_denominator_state = g4_partial_evidence["halt_denominator_recovery"]
    add(
        "G4",
        "jpx_halt_report_events_agent",
        "g4_halt_report_denominator_and_events",
        "validated_denominator_waiting_for_event_sets"
        if halt_denominator_state["validated_partial"]
        and not halt_denominator_state["report_events_validated"]
        else "validated_event_sets_merged_partial"
        if halt_denominator_state["report_events_primary_merged"]
        else "validated_event_sets_waiting_for_primary_merge"
        if halt_denominator_state["report_events_validated"]
        else "ready_denominator_recovery",
        "G1,G4:g4_halt_history_evidence",
        "",
        "Count-only report coverage cannot enter the halt component; all covered claims require independently reconciled normalized event sets, source-record hashes, identities, and PIT timestamps.",
        (
            f"covered_claims={halt_denominator_state['covered_claims']}/"
            f"{halt_denominator_state['source_rows']}; uncovered="
            f"{halt_denominator_state['uncovered_claims']}; denominator_validated="
            f"{halt_denominator_state['validated_partial']}; event_sets_validated="
            f"{halt_denominator_state['report_events_validated']}; primary_merged="
            f"{halt_denominator_state['report_events_primary_merged']}"
        ),
    )
    pre2013_state = g4_partial_evidence["pre2013_price_limits"]
    add(
        "G4",
        "jpx_pre2013_price_limit_agent",
        "g4_pre2013_exceptional_price_limit_archive",
        "valid_incomplete_denominator"
        if pre2013_state["valid_incomplete"]
        else "ready_offline_archive_research",
        "G1",
        ""
        if pre2013_state["valid_incomplete"]
        else "uv run python examples/run_japan_pre2013_price_limit_archive.py",
        "Every month from 2006-07 through 2013-08 needs an exact official notice denominator before the exceptional-limit component can pass.",
        (
            f"unresolved_months={pre2013_state['unresolved_months']}/"
            f"{pre2013_state['required_months']}; migrated_events="
            f"{pre2013_state['migrated_event_rows']}; bilingual_matches="
            f"{pre2013_state['bilingual_matched_rows']}; monthly_routes="
            f"{pre2013_state['monthly_route_attempts']}; ose_adjacency="
            f"{pre2013_state['ose_adjacency_rows']}; completion_plan_rows="
            f"{pre2013_state['completion_plan_rows']}; blockers="
            f"{';'.join(pre2013_state['critical_blockers']) or 'artifact_missing'}"
        ),
    )
    return_state = g4_partial_evidence["return_to_normal"]
    add(
        "G4",
        "jpx_return_to_normal_agent",
        "g4_post2020_return_to_normal",
        "validated_partial_waiting_for_prices"
        if return_state["validated_partial"]
        else "queued_after_price_evidence",
        "G1,G2:active_ohlcv,G2:reference_ohlcv",
        ""
        if return_state["validated_partial"]
        else "uv run python scripts/reconstruct_jpx_return_to_normal.py",
        "Every post-2020 activation case requires a positively proven closing transition or an explicitly unresolved state backed by complete admitted OHLCV coverage.",
        (
            f"resolved={return_state['resolved_cases']}/"
            f"{return_state['activation_cases']}; unresolved="
            f"{return_state['unresolved_cases']}; missing_ohlcv="
            f"{return_state['missing_ohlcv_cases']}; exact_plan_rows="
            f"{return_state['missing_price_plan_rows']}; plan_rics="
            f"{return_state['missing_price_plan_rics']}; plan_sessions="
            f"{return_state['missing_price_plan_sessions']}; execution_disabled="
            f"{return_state['missing_price_plan_execution_disabled']}"
        ),
    )
    market_rule_state = g4_partial_evidence["market_rules"]
    add(
        "G4",
        "jpx_market_rule_continuity_agent",
        "g4_tick_and_normal_price_limit_continuity",
        "passed"
        if market_rule_state["tick_rule_continuity_complete"]
        and market_rule_state["normal_limit_continuity_complete"]
        else "validated_partial_completion_required"
        if market_rule_state["validated_partial"]
        else "ready_offline_rule_research",
        "G1",
        ""
        if market_rule_state["validated_partial"]
        else "uv run python examples/run_japan_market_rule_history.py",
        "Official rulebooks and amendment ledgers must prove uninterrupted tick-size and normal daily-price-limit regimes across the full 2006-07 through 2026-06 interval.",
        (
            f"tick_regimes={market_rule_state['tick_regime_windows']}; tick_bands="
            f"{market_rule_state['tick_band_rows']}; price_limit_bands="
            f"{market_rule_state['price_limit_band_rows']}; tick_continuity="
            f"{market_rule_state['tick_rule_continuity_complete']}; normal_limit_continuity="
            f"{market_rule_state['normal_limit_continuity_complete']}; broader_component_blockers="
            f"{';'.join(market_rule_state['critical_blockers']) or 'artifact_missing'}"
        ),
    )
    tick_assignment_state = g4_partial_evidence["tick_assignment"]
    add(
        "G4",
        "jpx_tick_assignment_agent",
        "g4_pit_narrow_tick_assignment",
        "validated_partial_waiting_for_official_cutoff"
        if tick_assignment_state["validated_partial"]
        else "ready_offline_assignment_replay",
        "G1",
        ""
        if tick_assignment_state["validated_partial"]
        else "uv run python examples/run_japan_tick_assignment_history.py",
        "The complete official assignment-event replay must reconcile exactly to the final published month-end TOPIX master at the research cutoff.",
        (
            f"events={tick_assignment_state['assignment_events']}; cutoff_master_names="
            f"{tick_assignment_state['cutoff_master_names']}; exact_match="
            f"{tick_assignment_state['cutoff_master_exact_match']}; blockers="
            f"{';'.join(tick_assignment_state['critical_blockers']) or 'artifact_missing'}"
        ),
    )
    tca_state = g4_partial_evidence["tca"]
    add(
        "G4",
        "jpx_tca_calibration_agent",
        "g4_liquidity_aware_tca_calibration",
        "passed" if tca_state["complete"] else "queued_after_strict_g2_liquidity",
        "G2:g2_strict_aggregate_admission",
        "",
        "TCA must be independently calibrated from admitted full-period Japan liquidity and price data with complete hashes and no fallback policy constants.",
        (
            f"validation_status={tca_state['status']}; blockers="
            f"{';'.join(tca_state['critical_blockers']) or 'awaiting admitted G2 liquidity'}"
        ),
    )
    previous = "G2"
    scorecard_lane_for_gate = {
        "G2": "g2_prices_actions_liquidity",
        "G3": "g3_pit_fundamentals_ibes_events",
        "G4": "g4_jpy_calendar_halts_market_rules_tca",
    }
    for gate, lane_id, owner, phase in (
        ("G3", "g3_pit_fundamentals_ibes_events", "pit_fundamentals_ibes_agent", "g3_pit_fundamentals_ibes_events"),
        ("G4", "g4_jpy_calendar_halts_market_rules_tca", "fx_calendar_tca_agent", "g4_fx_calendar_halts_tca"),
        ("G5", "g5_leakage_bias_immutable_reproducibility", "independent_data_quality_auditor", "g5_leakage_reproducibility"),
    ):
        passed = _lane_passed(lanes, lane_id)
        prior_passed = (
            g2_postprocess["admission_complete"]
            if previous == "G2"
            else _lane_passed(lanes, scorecard_lane_for_gate[previous])
        )
        preflight_ready = bool(
            gate == "G5"
            and g5_preflight_manifest.get("status") == "blocked_upstream"
            and g5_preflight_manifest.get("scope") == "japan_g5_preflight_not_decision_calendar_audit"
            and g5_preflight_manifest.get("scorecard_eligible") is False
            and int(g5_preflight_manifest.get("upstream_components_total", 0)) > 0
        )
        status = "passed" if passed else "ready" if prior_passed else f"queued_after_{previous.lower()}"
        command = ""
        notes = _lane_notes(lanes, lane_id)
        if gate == "G5" and not passed:
            status = "prepared_blocked_upstream" if preflight_ready else status
            command = "" if preflight_ready else "uv run python examples/run_japan_g5_preflight.py"
            if preflight_ready:
                notes = (
                    f"Fail-closed preflight inventoried "
                    f"{g5_preflight_manifest.get('upstream_components_ready', 0)}/"
                    f"{g5_preflight_manifest.get('upstream_components_total', 0)} upstream components; "
                    "the full decision-calendar leakage audit remains unavailable."
                )
        add(
            gate,
            owner,
            phase,
            status,
            previous,
            command,
            f"{gate} must pass its Japan scorecard lane with no critical blockers.",
            notes,
        )
        previous = gate
    add(
        "FINAL",
        "chief_data_pm",
        "scorecard_refresh",
        "blocked_no_promotion" if not scorecard_accepted else "research_evidence_complete_no_promotion",
        "G1,G2,G3,G4,G5",
        f"uv run python examples/run_japan_data_quality_loop.py --run-id {shlex.quote(config.scorecard_run_id)}",
        "Refresh the Japan 5/5 evidence state; this execution queue never promotes alpha, production, or IBKR.",
        f"Current scorecard accepted={scorecard_accepted}.",
    )
    return pd.DataFrame(rows)


def _live_status(*, complete: bool, universe_ok: bool, vendor_ok: bool, plan_ok: bool, lock_exists: bool) -> str:
    if complete:
        return "passed"
    if not universe_ok:
        return "queued_after_g1"
    if not vendor_ok:
        return "blocked_by_vendor_health"
    if not plan_ok:
        return "blocked_by_plan_checksum"
    if lock_exists:
        return "blocked_by_writer_lock"
    return "ready"


def _historical_command(
    config: JapanExecutionLoopConfig,
    run_id: str,
    symbol_map_path: Path,
    quarantine_path: Path,
    plan_state: dict[str, Any],
    max_plan_rows: int,
) -> str:
    plan_path = Path(plan_state["plan_path"])
    lock_path = _under_root(config.project_root, config.writer_lock_marker)
    artifact_root = plan_path.parent.parent
    args = [
        "uv", "run", "--extra", "refinitiv", "python", "examples/run_refinitiv_historical_backfill.py",
        "--run-id", run_id,
        "--artifact-root", str(artifact_root),
        "--lake-root", str(config.project_root / "artifacts/data_lake"),
        "--symbol-map-csv", str(symbol_map_path),
        "--include-countries", "Japan",
        "--start-date", config.start_date,
        "--end-date", config.end_date,
        "--window-years", "5",
        "--batch-size", str(config.batch_size),
        "--max-plan-rows", str(max_plan_rows),
        "--existing-plan-only",
        "--expected-plan-sha256", str(plan_state.get("sha256") or "missing"),
        "--interval", "1d",
        "--refinitiv-session", config.refinitiv_session,
        "--exclude-plan-row-ids-csv", str(quarantine_path),
        "--require-exclude-plan-row-ids-csv",
    ]
    if config.field_catalog_path:
        args.extend(["--field-catalog-csv", str(_under_root(config.project_root, config.field_catalog_path))])
    extraction = " ".join(shlex.quote(value) for value in args)
    checksum = str(plan_state.get("sha256") or "missing")
    return (
        f"lock={shlex.quote(str(lock_path))}; mkdir \"$lock\" || exit 75; "
        "trap 'rmdir \"$lock\"' EXIT INT TERM; "
        f"printf '%s  %s\\n' {shlex.quote(checksum)} {shlex.quote(str(plan_path))} | shasum -a 256 -c - && "
        f"{extraction}"
    )


def _ohlcv_postprocess_command(
    config: JapanExecutionLoopConfig,
    *,
    role: str,
    historical_run_id: str,
    listing_windows_path: Path,
    lane: dict[str, Any],
) -> str:
    root = config.project_root
    raw_root = root / "artifacts/data_lake/prices_long" / historical_run_id / "ohlcv"
    historical_dir = root / "artifacts/historical_backfill" / historical_run_id
    canonical_dir = Path(str(lane["canonical_dir"]))
    commands = [
        [
            "uv", "run", "python", "examples/build_japan_ohlcv_canonical.py",
            "--run-id", str(lane["canonical_run_id"]),
            "--raw-root", str(raw_root),
            "--listing-windows", str(listing_windows_path),
            "--historical-dir", str(historical_dir),
            "--artifact-root", str(root / "artifacts/japan_ohlcv_canonical"),
        ],
        [
            "uv", "run", "python", "examples/run_data_depth_report.py",
            str(lane["canonical_root"]),
            "--run-id", f"{lane['tag']}-DEPTH",
            "--artifact-root", str(root / "artifacts/data_depth"),
            "--extraction-manifest-path", str(canonical_dir / "historical_manifest.csv"),
        ],
        [
            "uv", "run", "python", "examples/run_data_lake_partition_ops.py",
            str(lane["canonical_root"]),
            "--run-id", f"{lane['tag']}-COMPACTION",
            "--artifact-root", str(root / "artifacts/data_lake_ops"),
            "--compact",
            "--output-root", str(lane["compacted_root"]),
        ],
        [
            "uv", "run", "python", "examples/validate_ohlcv_row_equality.py",
            str(lane["canonical_root"]),
            str(lane["compacted_root"]),
            "--run-id", f"{lane['tag']}-ROW-EQUALITY",
            "--artifact-root", str(root / "artifacts/ohlcv_row_equality"),
        ],
        [
            "uv", "run", "python", "examples/build_immutable_hash_inventory.py",
            "--inventory-id", f"{lane['tag']}-HASHES",
            "--output", str(lane["inventory_path"]),
            "--input", str(lane["canonical_dir"]),
            "--input", str(lane["depth_dir"]),
            "--input", str(lane["compaction_dir"]),
            "--input", str(lane["equality_dir"]),
        ],
    ]
    return " && ".join(
        " ".join(shlex.quote(value) for value in command) for command in commands
    )


def _price_union_command(
    config: JapanExecutionLoopConfig, g2_postprocess: dict[str, Any]
) -> str:
    active = g2_postprocess["active"]
    reference = g2_postprocess["reference"]
    args = [
        "uv", "run", "python", "examples/build_japan_g2_price_union.py",
        "--run-id", config.price_union_run_id,
        "--artifact-root", str(config.project_root / "artifacts/japan_g2_price_union"),
        "--active-compacted-root", str(active["compacted_root"]),
        "--active-compaction-manifest", str(Path(active["compaction_dir"]) / "manifest.json"),
        "--active-row-equality-manifest", str(Path(active["equality_dir"]) / "manifest.json"),
        "--reference-compacted-root", str(reference["compacted_root"]),
        "--reference-compaction-manifest", str(Path(reference["compaction_dir"]) / "manifest.json"),
        "--reference-row-equality-manifest", str(Path(reference["equality_dir"]) / "manifest.json"),
        "--key-column", "ticker",
        "--key-column", "date",
    ]
    return " ".join(shlex.quote(value) for value in args)


def _adjusted_returns_command(
    config: JapanExecutionLoopConfig,
    g2_vendor: dict[str, Any],
    g2_postprocess: dict[str, Any],
) -> str:
    args = [
        "uv", "run", "python", "examples/run_adjusted_return_reconstruction.py",
        "--daily-features", str(g2_postprocess["price_union_root"]),
        "--corporate-actions-dir", str(g2_vendor["corporate_action_dir"]),
        "--artifact-root", str(config.project_root / "artifacts/adjusted_return_reconstruction"),
        "--run-id", config.adjusted_returns_run_id,
        "--min-matched-event-share", "1.0",
        "--max-quarantine-share", "0.0",
    ]
    return " ".join(shlex.quote(value) for value in args)


def _component_inventory_command(
    config: JapanExecutionLoopConfig,
    paths: JapanDataQualityPaths,
    g2_vendor: dict[str, Any],
    g2_postprocess: dict[str, Any],
) -> str:
    inputs = [
        paths.security_master_dir,
        _under_root(config.project_root, config.g1_validator_dir),
        paths.active_symbol_map_dir,
        paths.reference_symbol_map_dir,
        Path(str(g2_vendor["corporate_action_dir"])),
        Path(str(g2_postprocess["adjusted_dir"])),
        Path(str(g2_vendor["liquidity_dir"])),
        Path(str(g2_postprocess["price_union_dir"])),
    ]
    args = [
        "uv", "run", "python", "examples/build_immutable_hash_inventory.py",
        "--inventory-id", config.component_inventory_run_id,
        "--output", str(g2_postprocess["component_inventory_path"]),
    ]
    for item in inputs:
        args.extend(["--input", str(item)])
    return " ".join(shlex.quote(value) for value in args)


def _strict_g2_admission_command(
    config: JapanExecutionLoopConfig,
    paths: JapanDataQualityPaths,
    g2_vendor: dict[str, Any],
    g2_postprocess: dict[str, Any],
) -> str:
    active = g2_postprocess["active"]
    reference = g2_postprocess["reference"]
    args = [
        "uv", "run", "python", "examples/build_japan_g2_admission.py",
        "--run-id", config.strict_g2_admission_run_id,
        "--artifact-root", str(config.project_root / "artifacts/japan_g2_admission"),
        "--g1-security-master-dir", str(paths.security_master_dir),
        "--g1-validation-dir", str(_under_root(config.project_root, config.g1_validator_dir)),
        "--active-symbol-map-dir", str(paths.active_symbol_map_dir),
        "--reference-symbol-map-dir", str(paths.reference_symbol_map_dir),
        "--active-historical-dir", str(active["canonical_dir"]),
        "--active-depth-dir", str(active["depth_dir"]),
        "--active-compaction-dir", str(active["compaction_dir"]),
        "--active-hash-inventory", str(active["inventory_path"]),
        "--active-row-equality-dir", str(active["equality_dir"]),
        "--reference-historical-dir", str(reference["canonical_dir"]),
        "--reference-depth-dir", str(reference["depth_dir"]),
        "--reference-compaction-dir", str(reference["compaction_dir"]),
        "--reference-hash-inventory", str(reference["inventory_path"]),
        "--reference-row-equality-dir", str(reference["equality_dir"]),
        "--corporate-actions-dir", str(g2_vendor["corporate_action_dir"]),
        "--adjusted-returns-dir", str(g2_postprocess["adjusted_dir"]),
        "--liquidity-dir", str(g2_vendor["liquidity_dir"]),
        "--component-hash-inventory", str(g2_postprocess["component_inventory_path"]),
    ]
    return " ".join(shlex.quote(value) for value in args)


def _g3_sector_gap_command(
    config: JapanExecutionLoopConfig, gap_manifest: dict[str, Any]
) -> str:
    gap_dir = _under_root(config.project_root, config.g3_sector_gap_dir)
    full_dir = _under_root(config.project_root, config.g3_sector_full_dir)
    args = [
        "uv", "run", "--extra", "refinitiv", "python",
        "examples/run_japan_g3_sector_gap_backfill.py",
        "--gap-plan-dir", str(gap_dir),
        "--expected-plan-sha256", str(
            gap_manifest.get("artifact_sha256", {}).get("sector_gap_plan", "missing")
        ),
        "--baseline-sector-dir", str(_under_root(config.project_root, config.g3_sector_snapshot_dir)),
        "--g1-validation-dir", str(_under_root(config.project_root, config.g1_validator_dir)),
        "--g2-admission-dir", str(
            config.project_root / "artifacts/japan_g2_admission" / config.strict_g2_admission_run_id
        ),
        "--artifact-root", str(full_dir.parent),
        "--run-id", full_dir.name,
        "--max-plan-rows", "1",
        "--session", config.refinitiv_session,
    ]
    return _single_vendor_writer_command(config, args)


def _g3_execution_plan_command(config: JapanExecutionLoopConfig) -> str:
    output_dir = _under_root(config.project_root, config.g3_execution_plan_dir)
    args = [
        "uv", "run", "python", "examples/build_japan_g3_execution_plan.py",
        "--plan-dir", str(_under_root(config.project_root, config.g3_pit_plan_dir)),
        "--sector-snapshot-dir", str(_under_root(config.project_root, config.g3_sector_full_dir)),
        "--route-registry-dir", str(_under_root(config.project_root, config.g3_route_registry_dir)),
        "--g1-validation-dir", str(_under_root(config.project_root, config.g1_validator_dir)),
        "--artifact-root", str(output_dir.parent),
        "--run-id", output_dir.name,
        "--start-date", config.start_date,
        "--end-date", config.end_date,
        "--batch-size", "25",
        "--window-years", "5",
    ]
    return " ".join(shlex.quote(value) for value in args)


def _single_vendor_writer_command(config: JapanExecutionLoopConfig, args: list[str]) -> str:
    lock_path = _under_root(config.project_root, config.writer_lock_marker)
    extraction = " ".join(shlex.quote(value) for value in args)
    return (
        f"lock={shlex.quote(str(lock_path))}; mkdir \"$lock\" || exit 75; "
        "trap 'rmdir \"$lock\"' EXIT INT TERM; "
        f"{extraction}"
    )


def _corporate_action_command(
    config: JapanExecutionLoopConfig, g2_vendor: dict[str, Any]
) -> str:
    args = [
        "uv",
        "run",
        "--extra",
        "refinitiv",
        "python",
        "examples/run_refinitiv_corporate_actions_event_backfill.py",
        "--run-id",
        str(g2_vendor["corporate_action_run_id"]),
        "--artifact-root",
        str(g2_vendor["corporate_action_artifact_root"]),
        "--existing-plan-csv",
        str(g2_vendor["corporate_action_plan_path"]),
        "--expected-plan-sha256",
        str(g2_vendor["corporate_action_plan_sha256"] or "missing"),
        "--selected-universe-csv",
        str(g2_vendor["selected_universe_path"]),
        "--max-plan-rows",
        str(config.corporate_action_max_plan_rows),
        "--sleep-seconds",
        "0.05",
        "--session",
        config.refinitiv_session,
    ]
    return _single_vendor_writer_command(config, args)


def _liquidity_field_probe_command(
    config: JapanExecutionLoopConfig, g2_vendor: dict[str, Any]
) -> str:
    args = [
        "uv",
        "run",
        "--extra",
        "refinitiv",
        "python",
        "examples/run_refinitiv_pit_field_probe.py",
        "--plan-dir",
        str(g2_vendor["plan_dir"]),
        "--run-id",
        config.liquidity_field_probe_run_id,
        "--artifact-root",
        str(config.project_root / "artifacts/refinitiv_pit_field_probe"),
        "--session",
        config.refinitiv_session,
        "--sample-rics",
        ",".join(g2_vendor.get("liquidity_sample_rics", [])),
        "--sample-per-country",
        "3",
        "--tier",
        "u2",
        "--datasets",
        "liquidity_market_cap_free_float",
        "--sleep-seconds",
        "0.05",
        "--progress-every",
        "1",
    ]
    return _single_vendor_writer_command(config, args)


def _liquidity_historical_probe_command(
    config: JapanExecutionLoopConfig, g2_vendor: dict[str, Any]
) -> str:
    args = [
        "uv",
        "run",
        "--extra",
        "refinitiv",
        "python",
        "examples/run_refinitiv_pit_historical_probe.py",
        "--plan-dir",
        str(g2_vendor["plan_dir"]),
        "--field-probe-dir",
        str(g2_vendor["liquidity_field_probe_dir"]),
        "--run-id",
        config.liquidity_historical_probe_run_id,
        "--artifact-root",
        str(config.project_root / "artifacts/refinitiv_pit_historical_probe"),
        "--session",
        config.refinitiv_session,
        "--sample-rics",
        ",".join(g2_vendor.get("liquidity_sample_rics", [])),
        "--sample-per-country",
        "3",
        "--tier",
        "u2",
        "--datasets",
        "liquidity_market_cap_free_float",
        "--start-date",
        config.start_date,
        "--end-date",
        config.end_date,
        "--interval",
        "1M",
        "--sleep-seconds",
        "0.05",
        "--progress-every",
        "1",
    ]
    return _single_vendor_writer_command(config, args)


def _liquidity_historical_backfill_command(
    config: JapanExecutionLoopConfig, g2_vendor: dict[str, Any]
) -> str:
    args = [
        "uv",
        "run",
        "--extra",
        "refinitiv",
        "python",
        "examples/run_refinitiv_pit_historical_backfill.py",
        "--plan-dir",
        str(g2_vendor["plan_dir"]),
        "--historical-probe-dir",
        str(g2_vendor["liquidity_historical_probe_dir"]),
        "--run-id",
        config.liquidity_historical_run_id,
        "--artifact-root",
        str(config.project_root / "artifacts/refinitiv_pit_historical_backfill"),
        "--session",
        config.refinitiv_session,
        "--tier",
        "u2",
        "--start-date",
        config.start_date,
        "--end-date",
        config.end_date,
        "--interval",
        "1M",
        "--batch-size",
        "25",
        "--window-years",
        "5",
        "--max-plan-rows",
        str(config.liquidity_max_plan_rows),
        "--datasets",
        "liquidity_market_cap_free_float",
        "--sleep-seconds",
        "0.05",
    ]
    return _single_vendor_writer_command(config, args)


def _active_recovery_state(root: Path, active_run_id: str) -> dict[str, Any]:
    stage_dir = root / "artifacts/historical_backfill_recovery_stage" / active_run_id
    plan_path = stage_dir / "historical_plan.csv"
    retry_path = stage_dir / "failed_empty_174.csv"
    terminal_path = stage_dir / "historical_manifest.csv"
    marker_path = stage_dir / "guarded_merge_manifest.json"
    retry = _read_csv(retry_path)
    required = bool(plan_path.is_file() and not retry.empty and "plan_row_id" in retry)
    first_plan_row_id = str(retry.iloc[0]["plan_row_id"]) if required else ""
    production_dir = root / "artifacts/historical_backfill" / active_run_id
    production_quarantine = _read_csv(production_dir / "quarantined_plan_rows.csv")
    production_manifest = _read_csv(production_dir / "historical_manifest.csv")
    retry_ids = retry["plan_row_id"].astype(str).tolist() if required else []
    quarantine_ids = (
        set(production_quarantine["plan_row_id"].astype(str))
        if not production_quarantine.empty and "plan_row_id" in production_quarantine
        else set()
    )
    remaining_ids = [plan_row_id for plan_row_id in retry_ids if plan_row_id in quarantine_ids]
    next_plan_row_id = remaining_ids[0] if remaining_ids else ""
    terminal = _read_csv(terminal_path)
    marker = _read_json(marker_path)
    proof_plan_row_id = str(marker.get("plan_row_id") or first_plan_row_id)
    proof = (
        terminal.loc[terminal["plan_row_id"].astype(str).eq(proof_plan_row_id)]
        if not terminal.empty and "plan_row_id" in terminal
        else pd.DataFrame()
    )
    merge_manifest_path = Path(str(marker.get("merge_manifest_path", "")))
    merge_manifest = _read_json(merge_manifest_path)
    merge_valid = bool(
        marker.get("status") == "merged_and_validated"
        and marker.get("plan_row_id") == proof_plan_row_id
        and merge_manifest.get("status") == "merged_and_validated"
        and merge_manifest.get("plan_row_id") == proof_plan_row_id
        and marker.get("merge_manifest_sha256") == _sha256(merge_manifest_path)
    )
    proof_status = str(proof.iloc[0].get("status", "")) if len(proof) == 1 else ""
    proof_rows = _integer(proof.iloc[0].get("long_ohlcv_rows"), -1) if len(proof) == 1 else -1
    proof_terminal = bool(
        (proof_status in {"completed", "completed_partial"} and proof_rows > 0)
        or (
            proof_status in {"completed_prelisting", "completed_postretirement"}
            and proof_rows == 0
            and str(proof.iloc[0].get("structural_reason", ""))
        )
    )
    latest_merge_passed = bool(
        required
        and len(proof) == 1
        and proof_terminal
        and _integer(marker.get("merged_long_ohlcv_rows"), 0)
        == _integer(proof.iloc[0].get("long_ohlcv_rows"), -1)
        and merge_valid
    )
    initial_manifest_path = Path(str(marker.get("initial_gate_manifest_path", "")))
    initial_manifest = _read_json(initial_manifest_path)
    initial_gate_passed = bool(
        marker.get("initial_gate_plan_row_id")
        and initial_manifest_path.is_file()
        and marker.get("initial_gate_manifest_sha256") == _sha256(initial_manifest_path)
        and initial_manifest.get("status") == "merged_and_validated"
        and _integer(initial_manifest.get("merged_long_ohlcv_rows"), 0) > 0
    ) or bool(latest_merge_passed and proof_rows > 0)
    passed = bool(required and initial_gate_passed and latest_merge_passed and not remaining_ids)
    return {
        "required": required,
        "passed": passed,
        "initial_gate_passed": initial_gate_passed,
        "latest_merge_passed": latest_merge_passed,
        "stage_dir": str(stage_dir),
        "plan_path": str(plan_path),
        "plan_sha256": _sha256(plan_path) if plan_path.is_file() else "",
        "retry_path": str(retry_path),
        "retry_rows": len(retry),
        "remaining_rows": len(remaining_ids),
        "next_plan_row_id": next_plan_row_id,
        "first_plan_row_id": first_plan_row_id,
        "terminal_path": str(terminal_path),
        "production_dir": str(production_dir),
        "production_plan_sha256": (
            _sha256(production_dir / "historical_plan.csv")
            if (production_dir / "historical_plan.csv").is_file()
            else ""
        ),
        "production_manifest_rows": len(production_manifest),
        "guarded_merge_marker_path": str(marker_path),
        "guarded_merge_valid": merge_valid,
    }


def _active_recovery_command(
    config: JapanExecutionLoopConfig,
    symbol_map_path: Path,
    recovery: dict[str, Any],
) -> str:
    if not recovery.get("required"):
        return ""
    lock_path = _under_root(config.project_root, config.writer_lock_marker)
    stage_dir = Path(str(recovery["stage_dir"]))
    plan_path = Path(str(recovery["plan_path"]))
    plan_row_id = str(recovery.get("next_plan_row_id") or recovery.get("first_plan_row_id") or "")
    if not plan_row_id:
        return ""
    args = [
        "uv", "run", "--extra", "refinitiv", "python", "examples/run_refinitiv_historical_backfill.py",
        "--run-id", config.active_historical_run_id,
        "--artifact-root", str(stage_dir.parent),
        "--lake-root", str(config.project_root / "artifacts/data_lake_recovery_stage"),
        "--symbol-map-csv", str(symbol_map_path),
        "--include-countries", "Japan",
        "--existing-plan-only",
        "--expected-plan-sha256", str(recovery["plan_sha256"]),
        "--include-plan-row-id", plan_row_id,
        "--max-plan-rows", "1",
        "--interval", "1d",
        "--refinitiv-session", config.refinitiv_session,
    ]
    extraction = " ".join(shlex.quote(value) for value in args)
    merge_run_id = f"JPN-ACTIVE-OHLCV-GUARDED-RECOVERY-MERGE-{plan_row_id.replace('_', '-')}"
    merge_args = [
        "uv", "run", "python", "examples/merge_japan_ohlcv_recovery.py",
        "--run-id", merge_run_id,
        "--plan-row-id", plan_row_id,
        "--stage-artifact-dir", str(stage_dir),
        "--stage-lake-root", str(config.project_root / "artifacts/data_lake_recovery_stage"),
        "--production-artifact-dir", str(recovery["production_dir"]),
        "--production-lake-root", str(config.project_root / "artifacts/data_lake"),
        "--expected-stage-plan-sha256", str(recovery["plan_sha256"]),
        "--expected-production-plan-sha256", str(recovery["production_plan_sha256"]),
    ]
    merge = " ".join(shlex.quote(value) for value in merge_args)
    return (
        f"lock={shlex.quote(str(lock_path))}; mkdir \"$lock\" || exit 75; "
        "trap 'rmdir \"$lock\"' EXIT INT TERM; "
        f"printf '%s  %s\\n' {shlex.quote(str(recovery['plan_sha256']))} {shlex.quote(str(plan_path))} "
        f"| shasum -a 256 -c - && {extraction} && {merge}"
    )


def _vendor_health_command(config: JapanExecutionLoopConfig) -> str:
    evidence_path = _under_root(config.project_root, config.vendor_health_evidence_path)
    return (
        "uv run --extra refinitiv python - <<'PY'\n"
        "import json\nfrom datetime import datetime, timezone\nfrom pathlib import Path\n"
        "from quant_lab.data.vendor_adapters import RefinitivAdapter\n"
        f"path=Path({str(evidence_path)!r})\nsymbol='7203.T'\n"
        f"adapter=RefinitivAdapter(session_name={config.refinitiv_session!r})\n"
        "payload={'status':'failed','symbol':symbol,'rows':0,'observed_at_utc':datetime.now(timezone.utc).isoformat()}\n"
        "try:\n    adapter.open()\n    data=adapter.ld.get_history(universe=[symbol], fields=['TRDPRC_1'], interval='1D')\n"
        "    rows=0 if data is None else len(data.dropna(how='all'))\n"
        "    payload.update(status='healthy' if rows else 'failed', rows=rows)\n"
        "except Exception as exc:\n    payload.update(error_type=type(exc).__name__, error=str(exc)[:1000])\n"
        "finally:\n    try:\n        adapter.close()\n    except Exception:\n        pass\n"
        "path.parent.mkdir(parents=True, exist_ok=True)\ntmp=path.with_suffix('.tmp')\n"
        "tmp.write_text(json.dumps(payload, indent=2), encoding='utf-8')\ntmp.replace(path)\nPY"
    )


def _jquants_listed_info_command(config: JapanExecutionLoopConfig) -> str:
    artifact_root = _under_root(config.project_root, config.jquants_listed_info_dir).parent
    args = (
        "uv",
        "run",
        "python",
        "examples/run_jquants_listed_info_backfill.py",
        "--run-id",
        config.jquants_listed_info_run_id,
        "--artifact-root",
        str(artifact_root),
        "--end-date",
        config.end_date,
        "--max-dates",
        str(config.jquants_max_dates),
    )
    return " ".join(shlex.quote(value) for value in args)


def _jquants_listed_shares_command(config: JapanExecutionLoopConfig) -> str:
    artifact_root = _under_root(config.project_root, config.jquants_listed_shares_dir).parent
    args = (
        "uv",
        "run",
        "--extra",
        "calendar",
        "python",
        "examples/run_jquants_listed_shares_backfill.py",
        "--run-id",
        config.jquants_listed_shares_run_id,
        "--artifact-root",
        str(artifact_root),
        "--start-date",
        "2008-05-07",
        "--end-date",
        config.end_date,
        "--max-dates",
        str(config.jquants_max_dates),
        "--max-codes",
        "1",
    )
    return " ".join(shlex.quote(value) for value in args)


def _g1_materializer_command(config: JapanExecutionLoopConfig) -> str:
    root = config.project_root
    args = (
        "uv", "run", "python", "examples/build_japan_g1_monthly_materializer.py",
        "--run-id", config.g1_materializer_run_id,
        "--monthly-qa-dir", str(_under_root(root, config.jpx_monthly_membership_qa_dir)),
    )
    return " ".join(shlex.quote(value) for value in args)


def _g1_validator_command(config: JapanExecutionLoopConfig) -> str:
    root = config.project_root
    security_master_dir = root / "artifacts/data_lake/pit/security_master_history/JPN-G1-MONTHLY-VALIDATED-SECURITY-MASTER-20260712-V1"
    reconciliation_dir = root / "artifacts/research_universes/JPN-G1-MONTHLY-VALIDATED-RECONCILIATION-20260712-V1"
    active_map_dir = root / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-JPX-MONTHLY-ACTIVE-VALIDATED-20260712-V1"
    reference_map_dir = root / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-JPX-MONTHLY-REFERENCE-VALIDATED-20260712-V1"
    args = (
        "uv", "run", "python", "examples/run_japan_g1_independent_validator.py",
        "--run-id", _under_root(root, config.g1_validator_dir).name,
        "--producer-dir", str(_under_root(root, config.g1_materializer_dir)),
        "--artifact-root", str(_under_root(root, config.g1_validator_dir).parent),
        "--security-master-dir", str(security_master_dir),
        "--reconciliation-dir", str(reconciliation_dir),
        "--active-map-dir", str(active_map_dir),
        "--reference-map-dir", str(reference_map_dir),
    )
    return " ".join(shlex.quote(value) for value in args)


def _agent_roster() -> pd.DataFrame:
    rows = [
        ("chief_data_pm", "Own G1-G5 ordering and keep alpha, production, IBKR, and live trading blocked."),
        ("jpx_universe_reconciliation_agent", "Reconcile official JPX listings to uncapped active/inactive Japan security-master evidence."),
        ("vendor_health_guard", "Require a fresh successful 7203.T probe before any vendor OHLCV command."),
        ("jquants_historical_membership_agent", "Build official JPX historical listed-info evidence with bounded requests, raw lineage, and conservative known-at semantics."),
        ("jquants_listed_shares_agent", "Build official share-count and share-change ledgers without overstating comprehensive corporate-action coverage."),
        ("japan_g1_pit_materializer_agent", "Materialize producer-only listing-level PIT candidates from complete official history."),
        ("japan_g1_independent_validator", "Independently recompute hashes and publish admitted G1 artifacts only on a clean pass."),
        ("jpx_public_archive_agent", "Retain and hash all discoverable official JPX public listing-event archives."),
        ("jpx_public_crosscheck_auditor", "Cross-check recent official events against RIC and listing-date evidence."),
        ("jpx_monthly_snapshot_agent", "Build source-scheduled monthly PIT membership snapshots from official JPX quotation archives."),
        ("jpx_monthly_snapshot_auditor", "Independently verify monthly file hashes, PIT timestamps, uniqueness, and membership continuity."),
        ("japan_g1_targeted_identifier_agent", "Resolve missing historical JPX codes with bounded primary-exchange Discovery shards."),
        ("jpx_listing_valid_time_agent", "Separate listing valid-time intervals from source-backed historical PIT membership."),
        ("jpx_active_ohlcv_agent", "Run bounded active JPX OHLCV chunks under checksum and single-writer guards."),
        ("jpx_active_ohlcv_auditor", "Canonicalize, depth-check, compact, prove row equality, and hash the full active lane."),
        ("jpx_reference_ohlcv_agent", "Run bounded inactive/delisted reference OHLCV only after the active lane."),
        ("jpx_reference_ohlcv_auditor", "Independently apply the full postprocessing contract to reference-only OHLCV."),
        ("jpx_corporate_actions_agent", "Backfill the immutable full-G1 dividend and capital-change denominator sequentially."),
        ("jpx_liquidity_route_agent", "Prove deterministic current liquidity field routes on fixed JPX samples."),
        ("jpx_liquidity_history_probe_agent", "Prove 2006-present history availability before full liquidity extraction."),
        ("jpx_liquidity_backfill_agent", "Build complete PIT market-cap, free-float, and shares history."),
        ("jpx_price_union_auditor", "Build a memory-bounded, key-unique active/reference price union."),
        ("jpx_adjusted_return_agent", "Reconstruct adjusted returns with exact raw-key and event-denominator parity."),
        ("g2_component_hash_auditor", "Freeze and verify the complete G1/G2 component source inventory."),
        ("independent_g2_admission_auditor", "Issue the authoritative strict G2 pass only with zero blockers."),
        ("quarantine_auditor", "Atomically retain and extend failed/empty plan-row quarantine evidence."),
        ("storage_ops_agent", "Require depth and row-equal compaction before G2 can pass."),
        ("pit_fundamentals_ibes_agent", "Own G3 point-in-time fundamentals, IBES, and events evidence."),
        ("jpx_halt_history_agent", "Complete and independently validate every JPX halt/suspension month-channel without inferred no-event coverage."),
        ("jpx_halt_report_events_agent", "Normalize and independently reconcile every event row behind validated monthly halt-report denominator claims before primary merge."),
        ("jpx_pre2013_price_limit_agent", "Recover an exact official pre-2013 exceptional-price-limit notice denominator or retain a fail-closed completion plan."),
        ("jpx_return_to_normal_agent", "Prove every post-2020 exceptional-limit closing transition from admitted events, calendar, identities, and price evidence."),
        ("fx_calendar_tca_agent", "Own G4 JPY/JPX calendar, halt, and research TCA evidence."),
        ("independent_data_quality_auditor", "Own G5 leakage, bias, checksum, and reproducibility evidence."),
    ]
    return pd.DataFrame(rows, columns=["agent_id", "mandate"])


def _nonusable_rows(manifest: pd.DataFrame, now: datetime) -> pd.DataFrame:
    if manifest.empty or "status" not in manifest.columns or "plan_row_id" not in manifest.columns:
        return pd.DataFrame(columns=["plan_row_id", "reason", "detected_at_utc", "retry_policy"])
    status = manifest["status"].astype(str)
    structural = status.isin({"completed_prelisting", "completed_postretirement"})
    failed = status.str.startswith("failed")
    completed = status.isin({"completed", "completed_partial"})
    zero = pd.Series(False, index=manifest.index)
    for column in ("price_columns", "long_ohlcv_rows"):
        if column in manifest.columns:
            zero = zero | pd.to_numeric(manifest[column], errors="coerce").fillna(0).le(0)
    result = manifest.loc[~structural & (failed | (completed & zero))].copy().reset_index(drop=True)
    if result.empty:
        return pd.DataFrame(columns=["plan_row_id", "reason", "detected_at_utc", "retry_policy"])
    result["reason"] = result["status"].astype(str).map(
        lambda value: "failed_empty_ohlcv_payload" if value == "failed_empty" else "historical_plan_row_failed"
        if str(value).startswith("failed")
        else "zero_ohlcv_payload"
    )
    result["detected_at_utc"] = now.isoformat()
    result["retry_policy"] = "route_probe_then_retry_tiny_after_fresh_7203T_vendor_health"
    return result


def _atomic_merge_quarantine(path: Path, detected: pd.DataFrame) -> pd.DataFrame:
    existing = _read_csv(path)
    if existing.empty and detected.empty:
        merged = pd.DataFrame(columns=["plan_row_id", "reason", "detected_at_utc", "retry_policy"])
    else:
        merged = pd.concat([existing, detected], ignore_index=True, sort=False)
        keys = [column for column in ("plan_row_id", "reason") if column in merged.columns]
        if keys:
            merged = merged.drop_duplicates(keys, keep="last")
        merged = merged.sort_values([column for column in ("plan_row_id", "reason") if column in merged.columns])
    _atomic_write_csv(merged.reset_index(drop=True), path)
    return merged.reset_index(drop=True)


def _plan_complete(manifest: pd.DataFrame, planned_rows: int) -> bool:
    if planned_rows <= 0 or manifest.empty or "status" not in manifest.columns:
        return False
    status = manifest["status"].astype(str)
    usable = status.isin({"completed_prelisting", "completed_postretirement"})
    payload = status.eq("completed")
    for column in ("price_columns", "long_ohlcv_rows"):
        if column in manifest.columns:
            payload = payload & pd.to_numeric(manifest[column], errors="coerce").fillna(0).gt(0)
    return int((usable | payload).sum()) >= planned_rows


def _lane_passed(lanes: dict[str, Any], lane_id: str) -> bool:
    lane = lanes.get(lane_id)
    return bool(lane and lane.status == "pass" and not lane.critical_blockers)


def _lane_notes(lanes: dict[str, Any], lane_id: str) -> str:
    lane = lanes.get(lane_id)
    return "lane evidence unavailable" if lane is None else " | ".join(lane.critical_blockers) or lane.evidence


def _summary(
    config: JapanExecutionLoopConfig,
    score: float,
    accepted: bool,
    vendor: dict[str, Any],
    universe: dict[str, Any],
    queue: pd.DataFrame,
) -> str:
    lines = [
        "# Japan Guarded Data Quality Execution Queue",
        "",
        f"Run ID: `{config.run_id}`",
        f"Scorecard: `{score:.3f}/5.000` (accepted=`{accepted}`)",
        f"JPX universe eligible: `{universe['eligible']}`",
        f"Fresh 7203.T vendor evidence: `{vendor['valid']}`",
        "Live extraction executed by this run: `False`",
        "",
        "## Queue",
        "",
    ]
    for row in queue.itertuples(index=False):
        lines.append(f"- P{row.sequence} `{row.quality_gate}` `{row.phase}` [{row.status}]: {row.acceptance_gate}")
    lines.extend(
        [
            "",
            "## Hard Stop",
            "",
            "This queue does not promote Japan alpha, production, IBKR, or live trading. G1-G5 evidence remains data-quality research only.",
        ]
    )
    return "\n".join(lines) + "\n"


def _under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return value if isinstance(value, dict) else {}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _integer(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


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


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
