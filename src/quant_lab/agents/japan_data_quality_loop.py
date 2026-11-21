from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


_PASS_STATUSES = {"complete", "pass", "accepted_backtest"}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_PIT_DATASETS = {
    "fundamentals_pit",
    "ibes_estimates_revisions",
    "earnings_events_actuals",
}
_G4_COMPONENT_ALIASES = {
    "jpy": {"jpy", "jpy_currency", "jpy_fx_rates", "fx_rates"},
    "calendar": {"calendar", "exchange_calendar", "exchange_calendars", "jpx_calendar"},
    "halts": {"halts", "halt_events", "halt_suspension_events"},
    "tick_sizes": {"tick_sizes", "tick_size_bands", "pit_narrow_tick_assignment"},
    "price_limits": {"price_limits", "daily_price_limits", "jpx_price_limits"},
    "lot_sizes": {"lot_sizes", "board_lot_sizes", "trading_unit_lot_sizes"},
    "tca": {"tca", "tca_assumptions", "tca_model"},
}


@dataclass(frozen=True)
class JapanDataQualityPaths:
    security_master_dir: Path
    symbol_map_dir: Path | None = None
    historical_dir: Path | None = None
    depth_dir: Path | None = None
    compaction_dir: Path | None = None
    universe_reconciliation_dir: Path | None = None
    active_symbol_map_dir: Path | None = None
    reference_symbol_map_dir: Path | None = None
    full_reference_symbol_map_dir: Path | None = None
    l2_dir: Path | None = None
    pit_plan_dir: Path | None = None
    corporate_actions_dir: Path | None = None
    adjusted_returns_dir: Path | None = None
    g2_admission_dir: Path | None = None
    liquidity_dir: Path | None = None
    fundamentals_ibes_events_dir: Path | None = None
    g3_admission_dir: Path | None = None
    fx_calendar_tca_dir: Path | None = None
    leakage_qa_dir: Path | None = None
    current_enrichment_dir: Path | None = None
    listing_valid_time_dir: Path | None = None
    jquants_listed_info_dir: Path | None = None
    jquants_listed_shares_dir: Path | None = None
    g1_materializer_dir: Path | None = None
    g1_validation_dir: Path | None = None
    jpx_public_listing_events_dir: Path | None = None
    jpx_public_event_reconciliation_dir: Path | None = None
    jpx_monthly_quotations_dir: Path | None = None
    jpx_monthly_membership_qa_dir: Path | None = None
    jpx_monthly_quotations_extension_dir: Path | None = None
    ose_monthly_quotations_dir: Path | None = None
    ose_monthly_quotations_extension_dir: Path | None = None
    ose_main_monthly_quotations_dir: Path | None = None
    combined_monthly_membership_qa_dir: Path | None = None
    monthly_microstructure_dir: Path | None = None
    monthly_microstructure_validation_dir: Path | None = None
    exchange_calendar_dir: Path | None = None
    exchange_calendar_validation_dir: Path | None = None
    market_rule_history_dir: Path | None = None
    market_rule_history_validation_dir: Path | None = None
    immutable_hashes_path: Path | None = None
    commands_log: Path | None = None


@dataclass(frozen=True)
class JapanQualityLane:
    lane_id: str
    lead_agent: str
    score: int
    status: str
    critical_blockers: tuple[str, ...]
    evidence: str
    next_action: str


@dataclass(frozen=True)
class JapanDataQualityResult:
    run_id: str
    artifact_dir: Path
    overall_score: int
    accepted: bool
    status: str
    lanes: tuple[JapanQualityLane, ...]
    manifest_path: Path
    summary_path: Path
    lane_scorecard_path: Path
    next_queue_path: Path


def default_japan_data_quality_paths(project_root: str | Path = ".") -> JapanDataQualityPaths:
    root = Path(project_root)
    historical_run = "REFINITIV-JPN-HISTORICAL-20Y-SECMASTER-20260709"
    return JapanDataQualityPaths(
        security_master_dir=root
        / "artifacts/data_lake/pit/security_master_history/JPN-G1-MONTHLY-VALIDATED-SECURITY-MASTER-20260712-V1",
        symbol_map_dir=root
        / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-JPN-SECMASTER-MERGED-20260709",
        historical_dir=root / "artifacts/historical_backfill" / historical_run,
        depth_dir=root / "artifacts/data_depth" / f"{historical_run}-DEPTH",
        compaction_dir=root / "artifacts/data_lake_ops" / f"{historical_run}-COMPACT",
        universe_reconciliation_dir=root
        / "artifacts/research_universes/JPN-G1-MONTHLY-VALIDATED-RECONCILIATION-20260712-V1",
        active_symbol_map_dir=root
        / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-JPX-MONTHLY-ACTIVE-VALIDATED-20260712-V1",
        reference_symbol_map_dir=root
        / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-JPX-MONTHLY-REFERENCE-ONLY-VALIDATED-20260712-V3",
        full_reference_symbol_map_dir=root
        / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-JPX-MONTHLY-REFERENCE-VALIDATED-20260712-V1",
        l2_dir=root / "artifacts/l2_deep_extraction/JPN-L2-DEEP-U2-20260711",
        pit_plan_dir=root / "artifacts/pit_data_plan/PIT-DATA-PLAN-JPN-U2-20260712-V3",
        corporate_actions_dir=root
        / "artifacts/refinitiv_corporate_actions_events/REFINITIV-JPN-G2-CORPACT-20060101-20260706-V1",
        adjusted_returns_dir=root
        / "artifacts/adjusted_return_reconstruction/ADJRET-RECON-JPN-U2-20260712-V2",
        g2_admission_dir=root
        / "artifacts/japan_g2_admission/JPN-G2-STRICT-ADMISSION-20260712-V3",
        liquidity_dir=root
        / "artifacts/refinitiv_pit_historical_backfill/REFINITIV-PIT-HISTORICAL-LIQUIDITY-JPN-U2-20260712-V2",
        fundamentals_ibes_events_dir=root
        / "artifacts/refinitiv_pit_historical_backfill/REFINITIV-PIT-HISTORICAL-FUND-IBES-EVENTS-JPN-U2-20260711",
        g3_admission_dir=root
        / "artifacts/japan_g3_admission/JPN-G3-STRICT-ADMISSION-20260712-V1",
        fx_calendar_tca_dir=root / "artifacts/fx_calendar_tca/FX-CALENDAR-TCA-JPN-20260712-V7",
        leakage_qa_dir=root / "artifacts/pit_leakage_qa/PIT-LEAKAGE-QA-JPN-U2-20260711",
        current_enrichment_dir=root
        / "artifacts/japan_g1_enrichment/JPX-UNIVERSE-RECONCILIATION-V2-20260711-G1-CURRENT-ENRICHMENT-V2",
        listing_valid_time_dir=root
        / "artifacts/japan_listing_valid_time/JPX-UNIVERSE-RECONCILIATION-V2-20260711-LISTING-VALID-TIME-BASELINE",
        jquants_listed_info_dir=root
        / "artifacts/jquants_listed_info/JQUANTS-JPN-LISTED-INFO-20080507-20260706-V2",
        jquants_listed_shares_dir=root
        / "artifacts/jquants_listed_shares/JQUANTS-JPN-LISTED-SHARES-20080507-20260706-V1",
        g1_materializer_dir=root
        / "artifacts/japan_g1_monthly_materializer/JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1",
        g1_validation_dir=root
        / "artifacts/japan_g1_validation/JPN-G1-MONTHLY-INDEPENDENT-VALIDATION-20260712-V1",
        jpx_public_listing_events_dir=root
        / "artifacts/jpx_public_listing_events/JPX-PUBLIC-LISTING-EVENTS-20260712",
        jpx_public_event_reconciliation_dir=root
        / "artifacts/jpx_public_event_reconciliation/JPX-PUBLIC-EVENT-REFINITIV-CROSSCHECK-V2-20260712",
        jpx_monthly_quotations_dir=root
        / "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2",
        jpx_monthly_membership_qa_dir=root
        / "artifacts/jpx_monthly_membership_qa/JPN-JPX-MONTHLY-MEMBERSHIP-QA-V2",
        jpx_monthly_quotations_extension_dir=root
        / "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200607-200711-EXT-V1",
        ose_monthly_quotations_dir=root
        / "artifacts/ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200712-201306-V1",
        ose_monthly_quotations_extension_dir=root
        / "artifacts/ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200607-200711-EXT-V1",
        ose_main_monthly_quotations_dir=root
        / "artifacts/ose_main_monthly_quotations/JPN-OSE-MAIN-MONTHLY-QUOTATIONS-200607-201009-V1",
        combined_monthly_membership_qa_dir=root
        / "artifacts/japan_combined_monthly_membership_qa/JPN-COMBINED-MONTHLY-MEMBERSHIP-QA-200607-202606-V1",
        monthly_microstructure_dir=root
        / "artifacts/japan_monthly_microstructure/JPN-MONTHLY-MICROSTRUCTURE-200607-202606-V5",
        monthly_microstructure_validation_dir=root
        / "artifacts/japan_monthly_microstructure_validation/JPN-MONTHLY-MICROSTRUCTURE-INDEPENDENT-VALIDATION-20260712-V1",
        exchange_calendar_dir=root
        / "artifacts/japan_exchange_calendar/JPN-EXCHANGE-CALENDAR-200607-202606-V1",
        exchange_calendar_validation_dir=root
        / "artifacts/japan_exchange_calendar_validation/JPN-EXCHANGE-CALENDAR-INDEPENDENT-VALIDATION-20260712-V1",
        market_rule_history_dir=root
        / "artifacts/japan_market_rule_history/JPN-MARKET-RULE-HISTORY-200607-202606-V1",
        market_rule_history_validation_dir=root
        / "artifacts/japan_market_rule_history_validation/JPN-MARKET-RULE-HISTORY-INDEPENDENT-VALIDATION-20260712-V1",
        immutable_hashes_path=root
        / "artifacts/pit_leakage_qa/PIT-LEAKAGE-QA-JPN-U2-20260711/immutable_hashes.json",
        commands_log=root / "commands.log",
    )


def run_japan_data_quality_loop(
    paths: JapanDataQualityPaths | None = None,
    artifact_root: str | Path = "artifacts/institutional_data_quality",
    run_id: str | None = None,
    project_root: str | Path = ".",
) -> JapanDataQualityResult:
    paths = paths or default_japan_data_quality_paths(project_root)
    run_id = run_id or datetime.now(timezone.utc).strftime("JPN-DATA-QUALITY-LOOP-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    context = _load_context(paths)
    lanes = (
        _g1_security_master(context),
        _g2_prices_actions_liquidity(context),
        _g3_pit_fundamentals_ibes_events(context),
        _g4_fx_calendar_tca(context),
        _g5_leakage_ops(context),
    )
    blockers = [blocker for lane in lanes for blocker in lane.critical_blockers]
    overall_score = sum(lane.score for lane in lanes)
    accepted = overall_score == 5 and all(lane.status == "pass" for lane in lanes) and not blockers
    status = "accepted_backtest_5_of_5" if accepted else "continue_japan_data_quality_loop"

    lane_frame = pd.DataFrame(
        [
            {
                "lane_id": lane.lane_id,
                "lead_agent": lane.lead_agent,
                "score": lane.score,
                "max_score": 1,
                "status": lane.status,
                "critical_blockers": " | ".join(lane.critical_blockers),
                "evidence": lane.evidence,
                "next_action": lane.next_action,
            }
            for lane in lanes
        ]
    )
    queue = _next_queue(lanes)
    lane_path = artifact_dir / "lane_scorecard.csv"
    queue_path = artifact_dir / "agent_next_queue.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"
    quarantine_path = artifact_dir / "historical_quarantine.csv"
    lane_frame.to_csv(lane_path, index=False)
    queue.to_csv(queue_path, index=False)
    _nonusable_historical_rows(context["historical_manifest"]).to_csv(quarantine_path, index=False)
    summary_path.write_text(_summary(run_id, lanes, overall_score, accepted, blockers), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "country": "Japan",
        "country_code": "JPN",
        "exchange_scope": "JPX",
        "overall_score": overall_score,
        "max_score": 5,
        "gate_scoring": "five_binary_fail_closed_gates",
        "accepted": accepted,
        "readiness_scope": "research_backtest_only",
        "research_only": True,
        "backtest_ready": accepted,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "status": status,
        "critical_blockers": blockers,
        "gate_count": len(lanes),
        "passed_gate_count": overall_score,
        "next_queue_rows": int(len(queue)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_sha256": {
            "summary": _sha256(summary_path),
            "lane_scorecard": _sha256(lane_path),
            "agent_next_queue": _sha256(queue_path),
            "historical_quarantine": _sha256(quarantine_path),
        },
        "paths": {key: str(value) if value is not None else None for key, value in vars(paths).items()},
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "lane_scorecard": str(lane_path.resolve()),
            "agent_next_queue": str(queue_path.resolve()),
            "historical_quarantine": str(quarantine_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return JapanDataQualityResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        overall_score=overall_score,
        accepted=accepted,
        status=status,
        lanes=lanes,
        manifest_path=manifest_path,
        summary_path=summary_path,
        lane_scorecard_path=lane_path,
        next_queue_path=queue_path,
    )


def _load_context(paths: JapanDataQualityPaths) -> dict[str, Any]:
    hash_path = paths.immutable_hashes_path or _child(paths.leakage_qa_dir, "immutable_hashes.json")
    return {
        "paths": paths,
        "security": _read_json(_child(paths.security_master_dir, "manifest.json")),
        "universe": _read_json(_child(paths.universe_reconciliation_dir, "manifest.json")),
        "active_map_manifest": _read_json(_child(paths.active_symbol_map_dir, "manifest.json")),
        "active_map": _read_csv(_child(paths.active_symbol_map_dir, "symbol_map.csv")),
        "reference_map_manifest": _read_json(_child(paths.reference_symbol_map_dir, "manifest.json")),
        "reference_map": _read_csv(_child(paths.reference_symbol_map_dir, "symbol_map.csv")),
        "full_reference_map_manifest": _read_json(
            _child(paths.full_reference_symbol_map_dir, "manifest.json")
        ),
        "full_reference_map": _read_csv(
            _child(paths.full_reference_symbol_map_dir, "symbol_map.csv")
        ),
        "g1_validation": _read_json(_child(paths.g1_validation_dir, "manifest.json")),
        "jpx_public_event_crosscheck": _read_json(
            _child(paths.jpx_public_event_reconciliation_dir, "manifest.json")
        ),
        "historical_run": _read_json(_child(paths.historical_dir, "historical_run_manifest.json")),
        "historical_plan": _read_csv(_child(paths.historical_dir, "historical_plan.csv")),
        "historical_manifest": _read_csv(_child(paths.historical_dir, "historical_manifest.csv")),
        "depth": _read_csv(_child(paths.depth_dir, "overall.csv")),
        "compaction": _read_json(_child(paths.compaction_dir, "manifest.json")),
        "pit_plan": _read_json(_child(paths.pit_plan_dir, "manifest.json")),
        "corpact": _read_json(_child(paths.corporate_actions_dir, "manifest.json")),
        "adjusted": _read_json(_child(paths.adjusted_returns_dir, "manifest.json")),
        "g2_admission": _read_json(_child(paths.g2_admission_dir, "manifest.json")),
        "g2_contract": _read_csv(_child(paths.g2_admission_dir, "component_contract.csv")),
        "g2_source_hashes": _read_json(_child(paths.g2_admission_dir, "source_hashes.json")),
        "liquidity": _read_json(_child(paths.liquidity_dir, "manifest.json")),
        "liquidity_quality": _read_csv(_child(paths.liquidity_dir, "historical_quality.csv")),
        "fund": _read_json(_child(paths.fundamentals_ibes_events_dir, "manifest.json")),
        "fund_quality": _read_csv(_child(paths.fundamentals_ibes_events_dir, "historical_quality.csv")),
        "g3_admission": _read_json(_child(paths.g3_admission_dir, "manifest.json")),
        "g3_contract": _read_csv(_child(paths.g3_admission_dir, "component_contract.csv")),
        "g3_source_hashes": _read_json(_child(paths.g3_admission_dir, "source_hashes.json")),
        "fx": _read_json(_child(paths.fx_calendar_tca_dir, "manifest.json")),
        "fx_quality": _read_csv(_child(paths.fx_calendar_tca_dir, "evidence_quality.csv")),
        "leakage": _read_json(_child(paths.leakage_qa_dir, "manifest.json")),
        "hash_path": hash_path,
        "hashes": _read_json(hash_path),
    }


def _g1_security_master(context: dict[str, Any]) -> JapanQualityLane:
    security = context["security"]
    universe = context["universe"]
    blockers: list[str] = []

    if not _eligible_full_artifact(security):
        blockers.append("full listing-level Japan security master is not independently scorecard-eligible")
    if not (
        security.get("security_master_kind") == "listing_level"
        and security.get("listing_level_master") is True
        and security.get("quote_level_master") is False
    ):
        blockers.append("current quote-level security master cannot satisfy G1")
    required_files = (
        "entities.parquet",
        "listing_windows.parquet",
        "identifier_history.parquet",
        "universe_membership_pit.parquet",
    )
    if any(not _child(context["paths"].security_master_dir, name).is_file() for name in required_files):
        blockers.append("Japan PIT security-master tables are incomplete")
    if not all(_provenance_complete(security, item) for item in ("currency", "listing_window", "known_at")):
        blockers.append("currency/listing-window/known-at provenance is incomplete")
    if not _non_overlap_checks_pass(security):
        blockers.append("security-master validity-window non-overlap checks have not passed")

    if not _jpx_universe_reconciliation_ready(universe):
        blockers.append("separate independent JPX universe reconciliation artifact is missing or incomplete")
    if not _g1_independent_validation_ready(context["g1_validation"]):
        blockers.append("separate G1 independent validation artifact is missing or incomplete")
    if not _validated_symbol_map(
        context["active_map_manifest"],
        context["active_map"],
        _child(context["paths"].active_symbol_map_dir, "symbol_map.csv"),
        expected_role="active",
    ):
        blockers.append("validated JPX active symbol map is missing or incomplete")
    if context["paths"].full_reference_symbol_map_dir is None:
        if not _validated_symbol_map(
            context["reference_map_manifest"],
            context["reference_map"],
            _child(context["paths"].reference_symbol_map_dir, "symbol_map.csv"),
            expected_role="reference",
        ):
            blockers.append("validated JPX reference symbol map is missing or incomplete")
    else:
        if not _validated_symbol_map(
            context["full_reference_map_manifest"],
            context["full_reference_map"],
            _child(context["paths"].full_reference_symbol_map_dir, "symbol_map.csv"),
            expected_role="reference",
        ):
            blockers.append("independently validated full JPX reference map is missing or incomplete")
        partition_ok, partition_reason = _validated_reference_partition(context)
        if not partition_ok:
            blockers.append(f"disjoint JPX reference-only partition failed: {partition_reason}")

    return _binary_lane(
        "g1_security_master_universe",
        "independent_security_master_auditor",
        blockers,
        "listing_level="
        f"{security.get('listing_level_master')}; quote_level={security.get('quote_level_master')}; "
        f"universe_reconciled={_jpx_universe_reconciliation_ready(universe)}; "
        f"active_map_rows={len(context['active_map'])}; reference_only_rows={len(context['reference_map'])}; "
        f"full_reference_rows={len(context['full_reference_map'])}; public_event_crosscheck={context['jpx_public_event_crosscheck'].get('status')}",
        "Build and independently reconcile a listing-level JPX universe with complete provenance and validated active/reference maps.",
    )


def _g2_prices_actions_liquidity(context: dict[str, Any]) -> JapanQualityLane:
    blockers: list[str] = []
    admission_ok, admission_reason = _strict_g2_admission_ready(context)
    if not admission_ok:
        blockers.append(f"strict Japan G2 aggregate admission failed: {admission_reason}")
    admission = context["g2_admission"]
    denominators = (
        admission.get("denominators", {})
        if isinstance(admission.get("denominators"), dict)
        else {}
    )

    return _binary_lane(
        "g2_prices_actions_liquidity",
        "independent_prices_actions_liquidity_auditor",
        blockers,
        f"strict_admission={admission_ok}; checks={admission.get('passed_check_count', 0)}/{admission.get('check_count', 0)}; "
        f"g1_rics={denominators.get('g1_rics', 0)}; g1_listing_windows={denominators.get('g1_listing_windows', 0)}; "
        f"raw_union_rows={denominators.get('raw_union_rows', 0)}",
        "Complete every strict G2 component and publish the immutable aggregate admission packet with all checks passing.",
    )


def _g3_pit_fundamentals_ibes_events(context: dict[str, Any]) -> JapanQualityLane:
    pit_plan = context["pit_plan"]
    blockers: list[str] = []

    plan_datasets = set(map(str, pit_plan.get("datasets", [])))
    selected = _as_int(pit_plan.get("selected_names", pit_plan.get("selected_tickers", 0)))
    active_rics = set(context["active_map"].get("refinitiv_ric", pd.Series(dtype=str)).dropna().astype(str))
    reference_rics = set(
        context["reference_map"].get("refinitiv_ric", pd.Series(dtype=str)).dropna().astype(str)
    )
    expected_names = len(active_rics | reference_rics)
    expected_windows = _as_int(context["security"].get("listing_rows"), -1)
    selected_windows = _as_int(pit_plan.get("selected_listing_windows"), -1)
    if not (
        _eligible_full_artifact(pit_plan)
        and expected_names > 0
        and expected_windows > 0
        and selected == expected_names
        and selected_windows == expected_windows
        and _REQUIRED_PIT_DATASETS.issubset(plan_datasets)
        and pit_plan.get("point_in_time_plan") is True
        and pit_plan.get("window_denominator_preserved") is True
        and pit_plan.get("current_survivor") is False
    ):
        blockers.append("full non-survivor Japan PIT acquisition plan does not reconcile to the admitted G1 RIC/window denominator")
    admission_ok, admission_reason = _strict_g3_admission_ready(context)
    if not admission_ok:
        blockers.append(f"strict Japan G3 aggregate admission failed: {admission_reason}")
    admission = context["g3_admission"]
    denominators = (
        admission.get("denominators", {})
        if isinstance(admission.get("denominators"), dict)
        else {}
    )

    return _binary_lane(
        "g3_pit_fundamentals_ibes_events",
        "independent_pit_fundamentals_ibes_auditor",
        blockers,
        f"pit_plan_selected={selected}/{expected_names}; listing_windows={selected_windows}/{expected_windows}; datasets={sorted(plan_datasets)}; "
        f"strict_checks={admission.get('passed_check_count', 0)}/{admission.get('check_count', 0)}; "
        f"history_rows={denominators.get('history_rows', 0)}",
        "Complete full-scope PIT fundamentals, consensus, earnings, and analyst-detail lanes and publish the immutable G3 aggregate admission packet.",
    )


def _g4_fx_calendar_tca(context: dict[str, Any]) -> JapanQualityLane:
    fx = context["fx"]
    quality = context["fx_quality"]
    blockers: list[str] = []
    manifest_components = _component_concepts(fx.get("complete_components", []))
    quality_components = _validated_quality_components(quality)
    assembly_ok, assembly_reason = _strict_g4_assembly_ready(context)

    if not assembly_ok:
        blockers.append(f"strict Japan G4 assembly failed: {assembly_reason}")
    required = set(_G4_COMPONENT_ALIASES)
    if not required.issubset(manifest_components) or not required.issubset(quality_components):
        blockers.append("JPY/calendar/halts/price-limits/lot-sizes/TCA evidence is incomplete")

    return _binary_lane(
        "g4_jpy_calendar_halts_market_rules_tca",
        "independent_jpx_market_rules_tca_auditor",
        blockers,
        f"manifest_components={sorted(manifest_components)}; validated_components={sorted(quality_components)}; "
        f"research_only={fx.get('research_only')}; production_ready={fx.get('production_ready')}; assembly={assembly_ok}",
        "Validate full-history JPY, official JPX calendars/halts/price limits/lot sizes, and calibrated research TCA.",
    )


def _g5_leakage_ops(context: dict[str, Any]) -> JapanQualityLane:
    leakage = context["leakage"]
    commands_log = context["paths"].commands_log
    blockers: list[str] = []

    if not (
        _eligible_full_artifact(leakage)
        and _as_int(leakage.get("total_leakage_violations"), -1) == 0
        and _as_int(leakage.get("future_data_violations"), -1) == 0
        and _as_int(leakage.get("train_test_overlap_rows"), -1) == 0
        and _as_int(leakage.get("survivorship_violations"), -1) == 0
        and _as_int(leakage.get("decision_rows")) > 0
        and _as_int(leakage.get("decision_entities")) > 0
        and bool(leakage.get("features"))
        and _as_float(leakage.get("known_at_coverage")) == 1.0
    ):
        blockers.append("Japan decision-calendar leakage/PIT/bias audit has not passed cleanly")
    row_evidence_ok, row_evidence_reason = _g5_row_evidence_ready(
        leakage, context["paths"].leakage_qa_dir
    )
    if not row_evidence_ok:
        blockers.append(f"Japan G5 row-level evidence failed: {row_evidence_reason}")
    if not commands_log or not commands_log.is_file() or commands_log.stat().st_size == 0:
        blockers.append("non-empty commands.log is missing")
    hashes_ok, hash_reason = _immutable_hash_inventory_ready(context)
    if not hashes_ok:
        blockers.append(f"immutable SHA-256 evidence failed: {hash_reason}")

    return _binary_lane(
        "g5_leakage_bias_immutable_reproducibility",
        "independent_data_quality_auditor",
        blockers,
        f"leakage_status={leakage.get('status')}; violations={leakage.get('total_leakage_violations')}; "
        f"decision_rows={leakage.get('decision_rows', 0)}; row_evidence={row_evidence_ok}; immutable_hashes={hashes_ok}",
        "Run the leakage/bias audit and publish a complete, independently verified immutable SHA-256 inventory.",
    )


def _g5_row_evidence_ready(manifest: dict[str, Any], root: Path | None) -> tuple[bool, str]:
    if root is None:
        return False, "leakage root is missing"
    required = {
        "decision_panel",
        "leakage_violations",
        "survivorship_reconciliation",
        "split_ledger",
        "feature_spec",
        "audit_config",
    }
    artifacts = manifest.get("artifacts", {})
    if not isinstance(artifacts, dict) or not required.issubset(artifacts):
        return False, "required artifact paths are missing"
    paths = {
        name: Path(str(artifacts[name])) if Path(str(artifacts[name])).is_absolute()
        else root / str(artifacts[name])
        for name in required
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return False, f"missing files: {','.join(sorted(missing))}"
    hashes = manifest.get("artifact_sha256", {})
    if not isinstance(hashes, dict) or not required.issubset(hashes):
        return False, "artifact SHA-256 coverage is incomplete"
    for name, path in paths.items():
        if _sha256(path) != str(hashes[name]).lower():
            return False, f"artifact hash mismatch: {name}"
    decision = _read_table(paths["decision_panel"])
    violations = _read_table(paths["leakage_violations"])
    survivorship = _read_table(paths["survivorship_reconciliation"])
    splits = _read_table(paths["split_ledger"])
    feature_spec = _read_json(paths["feature_spec"])
    audit_config = _read_json(paths["audit_config"])
    required_decision_columns = {
        "decision_id",
        "entity",
        "decision_time_utc",
        "listing_start",
        "listing_end",
        "membership_valid_at_decision",
        "max_feature_available_time_utc",
        "label_start_utc",
        "label_end_utc",
    }
    if decision.empty or not required_decision_columns.issubset(decision.columns):
        return False, "decision panel lacks row-level listing, availability, or label evidence"
    decision_ids = decision["decision_id"].fillna("").astype(str).str.strip()
    entities = decision["entity"].fillna("").astype(str).str.strip()
    decision_time = pd.to_datetime(
        decision["decision_time_utc"], errors="coerce", utc=True
    )
    listing_start = pd.to_datetime(decision["listing_start"], errors="coerce", utc=True)
    listing_end = pd.to_datetime(decision["listing_end"], errors="coerce", utc=True)
    available = pd.to_datetime(
        decision["max_feature_available_time_utc"], errors="coerce", utc=True
    )
    label_start = pd.to_datetime(decision["label_start_utc"], errors="coerce", utc=True)
    label_end = pd.to_datetime(decision["label_end_utc"], errors="coerce", utc=True)
    membership = decision["membership_valid_at_decision"].map(_is_true)
    if not (
        len(decision) == _as_int(manifest.get("decision_rows"))
        and decision_ids.ne("").all()
        and not decision_ids.duplicated().any()
        and entities.ne("").all()
        and entities.nunique() == _as_int(manifest.get("decision_entities"))
        and decision_time.notna().all()
        and listing_start.notna().all()
        and listing_end.notna().all()
        and available.notna().all()
        and label_start.notna().all()
        and label_end.notna().all()
        and membership.all()
        and listing_start.le(decision_time).all()
        and decision_time.le(listing_end).all()
        and available.le(decision_time).all()
        and decision_time.le(label_start).all()
        and label_start.le(label_end).all()
    ):
        return False, "decision-panel denominator, PIT membership, availability, or label timing is invalid"
    if not violations.empty:
        return False, "leakage violation table is not empty"
    survivor_entities = survivorship.get("entity", pd.Series(dtype=str)).fillna("").astype(str)
    if (
        survivorship.empty
        or survivor_entities.duplicated().any()
        or set(survivor_entities) != set(entities)
        or not survivorship.get("status", pd.Series(dtype=str)).astype(str).eq("pass").all()
    ):
        return False, "survivorship reconciliation did not pass"
    split_ids = splits.get("decision_id", pd.Series(dtype=str)).astype(str)
    split_labels = set(splits.get("split", pd.Series(dtype=str)).astype(str))
    if not (
        len(splits) == len(decision)
        and not split_ids.duplicated().any()
        and set(split_ids) == set(decision_ids)
        and {"train", "test", "holdout"}.issubset(split_labels)
    ):
        return False, "split ledger is incomplete or overlaps"
    declared_features = feature_spec.get("features", [])
    if list(declared_features) != list(manifest.get("features", [])):
        return False, "feature specification differs from manifest"
    if not (
        _as_int(audit_config.get("purge_days"), -1) >= 0
        and _as_int(audit_config.get("embargo_days"), -1) >= 0
        and bool(audit_config.get("holdout_start"))
        and bool(audit_config.get("decision_schedule"))
    ):
        return False, "audit configuration lacks frozen split controls"
    return True, "pass"


def _binary_lane(
    lane_id: str,
    lead_agent: str,
    blockers: list[str],
    evidence: str,
    next_action: str,
) -> JapanQualityLane:
    passed = not blockers
    return JapanQualityLane(
        lane_id=lane_id,
        lead_agent=lead_agent,
        score=int(passed),
        status="pass" if passed else "fail",
        critical_blockers=tuple(blockers),
        evidence=evidence,
        next_action=next_action,
    )


def _eligible_full_artifact(manifest: dict[str, Any]) -> bool:
    if not manifest or str(manifest.get("status", "")).lower() not in _PASS_STATUSES:
        return False
    if manifest.get("scorecard_eligible") is not True or manifest.get("evidence_scope") != "full":
        return False
    if any(manifest.get(key) is True for key in ("partial", "smoke", "is_smoke_test", "policy_stub", "is_policy_stub")):
        return False
    labels = " ".join(str(manifest.get(key, "")) for key in ("run_id", "artifact_type", "mode", "scope_profile"))
    return not any(marker in labels.lower() for marker in ("smoke", "stub", "partial", "policy-only", "policy_only"))


def _provenance_complete(manifest: dict[str, Any], name: str) -> bool:
    provenance = manifest.get("provenance", {})
    item = provenance.get(name, {}) if isinstance(provenance, dict) else {}
    if isinstance(item, dict) and item:
        return bool(
            item.get("complete") is True
            and _as_float(item.get("coverage")) == 1.0
            and _as_int(item.get("missing_rows"), -1) == 0
            and bool(item.get("source"))
        )
    return bool(
        manifest.get(f"{name}_provenance_complete") is True
        and _as_float(manifest.get(f"{name}_coverage")) == 1.0
        and _as_int(manifest.get(f"{name}_missing_rows"), -1) == 0
    )


def _non_overlap_checks_pass(manifest: dict[str, Any]) -> bool:
    checks = manifest.get("non_overlap_checks", {})
    required = {"listing_windows", "identifier_windows", "currency_windows"}
    if not isinstance(checks, dict) or not required.issubset(checks):
        return False
    return all(
        isinstance(checks[name], dict)
        and checks[name].get("passed") is True
        and _as_int(checks[name].get("overlap_rows"), -1) == 0
        for name in required
    )


def _jpx_universe_reconciliation_ready(manifest: dict[str, Any]) -> bool:
    producer = str(manifest.get("producer_run_id", "")).strip()
    validator = str(manifest.get("validator_run_id", "")).strip()
    return bool(
        _eligible_full_artifact(manifest)
        and manifest.get("artifact_type") == "jpx_universe_reconciliation"
        and manifest.get("independent_audit") is True
        and producer
        and validator
        and producer != validator
        and manifest.get("source_checksums_verified") is True
        and manifest.get("reconciliation_output_sha256_verified") is True
        and manifest.get("official_jpx_source") is True
        and manifest.get("reconciliation_complete") is True
        and _as_int(manifest.get("jpx_source_rows")) > 0
        and _as_float(manifest.get("reconciliation_coverage")) == 1.0
        and _as_int(manifest.get("unmatched_jpx_rows"), -1) == 0
        and _as_int(manifest.get("unmatched_master_rows"), -1) == 0
        and _as_int(manifest.get("duplicate_listing_rows"), -1) == 0
    )


def _g1_independent_validation_ready(manifest: dict[str, Any]) -> bool:
    producer = str(manifest.get("producer_run_id", "")).strip()
    validator = str(manifest.get("validator_run_id", "")).strip()
    return bool(
        _eligible_full_artifact(manifest)
        and manifest.get("artifact_type") == "japan_g1_independent_validation"
        and manifest.get("independent_audit") is True
        and manifest.get("validator_type") == "independent_process"
        and producer
        and validator
        and producer != validator
        and manifest.get("source_checksums_verified") is True
        and manifest.get("validation_status") == "pass"
        and _as_int(manifest.get("critical_blocker_count"), -1) == 0
    )


def _jpx_public_event_crosscheck_ready(manifest: dict[str, Any]) -> bool:
    counts = manifest.get("counts", {}) if isinstance(manifest.get("counts"), dict) else {}
    total = _as_int(counts.get("events"))
    return bool(
        manifest.get("status") == "pass"
        and manifest.get("artifact_type") == "jpx_public_event_refinitiv_cross_validation"
        and manifest.get("evidence_scope") == "recent_official_public_archive_cross_validation"
        and total > 0
        and _as_int(counts.get("mapping_matched")) == total
        and _as_int(counts.get("mapping_unmatched_or_ambiguous"), -1) == 0
        and _as_int(counts.get("date_exact_match")) == total
        and _as_int(counts.get("date_missing_or_mismatch"), -1) == 0
        and _as_float(manifest.get("mapping_coverage")) == 1.0
        and _as_float(manifest.get("date_exact_match_coverage")) == 1.0
    )


def _validated_symbol_map(
    manifest: dict[str, Any], frame: pd.DataFrame | None, map_path: Path, expected_role: str
) -> bool:
    required_columns = {"refinitiv_ric", "listing_start", "listing_end", "currency", "known_at"}
    producer = str(manifest.get("producer_run_id", "")).strip()
    validator = str(manifest.get("validator_run_id", "")).strip()
    expected_digest = str(manifest.get("symbol_map_sha256", "")).lower()
    actual_digest = _sha256(map_path) if map_path.is_file() else ""
    if not (
        _eligible_full_artifact(manifest)
        and manifest.get("map_role") == expected_role
        and manifest.get("validation_status") == "pass"
        and producer
        and validator
        and producer != validator
        and manifest.get("validator_type") == "independent_process"
        and manifest.get("source_checksums_verified") is True
        and _SHA256_RE.fullmatch(expected_digest)
        and expected_digest == actual_digest
        and _as_float(manifest.get("mapping_coverage")) == 1.0
        and _as_int(manifest.get("unresolved_rows"), -1) == 0
        and _as_int(manifest.get("invalid_ric_rows"), -1) == 0
        and _as_int(manifest.get("overlap_conflict_rows"), -1) == 0
        and frame is not None
        and not frame.empty
        and required_columns.issubset(frame.columns)
    ):
        return False
    values = frame[list(required_columns)]
    if values.isna().any().any() or values.astype(str).apply(lambda col: col.str.strip().eq("").any()).any():
        return False
    if not frame["currency"].astype(str).eq("JPY").all():
        return False
    if frame.duplicated(["refinitiv_ric", "listing_start", "listing_end"]).any():
        return False
    starts = pd.to_datetime(frame["listing_start"], errors="coerce", utc=True)
    ends = pd.to_datetime(frame["listing_end"], errors="coerce", utc=True)
    known_at = pd.to_datetime(frame["known_at"], errors="coerce", utc=True)
    if not bool(starts.notna().all() and ends.notna().all() and known_at.notna().all() and starts.le(ends).all()):
        return False
    dated = frame.assign(_start=starts, _end=ends).sort_values(["refinitiv_ric", "_start", "_end"])
    previous_end = dated.groupby("refinitiv_ric", sort=False)["_end"].shift()
    return not bool(dated["_start"].le(previous_end).fillna(False).any())


def _validated_reference_partition(context: dict[str, Any]) -> tuple[bool, str]:
    manifest = context["reference_map_manifest"]
    active = context["active_map"]
    reference_only = context["reference_map"]
    full_reference = context["full_reference_map"]
    paths: JapanDataQualityPaths = context["paths"]
    map_path = _child(paths.reference_symbol_map_dir, "symbol_map.csv")
    required = {"stable_listing_id", "refinitiv_ric", "listing_start", "listing_end", "currency", "known_at"}
    if not (
        _eligible_full_artifact(manifest)
        and manifest.get("artifact_type") == "validated_jpx_reference_only_symbol_map"
        and manifest.get("map_role") == "reference"
        and manifest.get("partition_role") == "reference_only"
        and manifest.get("derived_from_independently_validated_maps") is True
        and manifest.get("source_checksums_verified") is True
        and manifest.get("partition_reconstructs_full_reference") is True
        and _as_int(manifest.get("active_reference_window_overlap_rows"), -1) == 0
        and _as_int(manifest.get("active_reference_interval_overlap_rows"), -1) == 0
        and map_path.is_file()
        and _sha256(map_path) == str(manifest.get("symbol_map_sha256", "")).lower()
        and not active.empty
        and not reference_only.empty
        and not full_reference.empty
        and required.issubset(active.columns)
        and required.issubset(reference_only.columns)
        and required.issubset(full_reference.columns)
    ):
        return False, "partition manifest, map hash, or required columns failed"
    for frame in (active, reference_only, full_reference):
        if frame[list(required)].isna().any().any() or not frame["currency"].astype(str).eq("JPY").all():
            return False, "partition contains null required values or non-JPY rows"

    key_columns = ["stable_listing_id", "refinitiv_ric", "listing_start", "listing_end"]
    active_keys = set(map(tuple, active[key_columns].astype(str).itertuples(index=False, name=None)))
    partition_keys = set(
        map(tuple, reference_only[key_columns].astype(str).itertuples(index=False, name=None))
    )
    full_keys = set(
        map(tuple, full_reference[key_columns].astype(str).itertuples(index=False, name=None))
    )
    if len(active_keys) != len(active) or len(partition_keys) != len(reference_only) or len(full_keys) != len(full_reference):
        return False, "partition key rows are duplicated"
    if active_keys & partition_keys or active_keys | partition_keys != full_keys:
        return False, "active and reference-only keys do not form an exact disjoint full-reference union"
    if not (
        _as_int(manifest.get("active_window_rows"), -1) == len(active)
        and _as_int(manifest.get("reference_only_window_rows"), -1) == len(reference_only)
        and _as_int(manifest.get("full_reference_window_rows"), -1) == len(full_reference)
        and _as_int(manifest.get("partition_union_rows"), -1) == len(full_reference)
    ):
        return False, "partition row denominators do not reconcile"
    input_hashes = manifest.get("input_sha256", {})
    if not isinstance(input_hashes, dict) or len(input_hashes) < 4:
        return False, "partition input hash coverage is incomplete"
    for raw_path, raw_digest in input_hashes.items():
        path = Path(str(raw_path))
        digest = str(raw_digest).lower()
        if not path.is_file() or not _SHA256_RE.fullmatch(digest) or _sha256(path) != digest:
            return False, f"partition input changed: {raw_path}"
    return True, "pass"


def _strict_g2_admission_ready(context: dict[str, Any]) -> tuple[bool, str]:
    paths: JapanDataQualityPaths = context["paths"]
    root = paths.g2_admission_dir
    if root is None:
        return False, "aggregate admission directory is missing"
    manifest = context["g2_admission"]
    contract = context["g2_contract"]
    source_hashes = context["g2_source_hashes"]
    required_contract_columns = {"component", "check_id", "passed"}
    checks = _as_int(manifest.get("check_count"), -1)
    passed = _as_int(manifest.get("passed_check_count"), -2)
    promotion = manifest.get("promotion_policy", {})
    if (
        manifest.get("artifact_type") == "japan_g2_aggregate_admission"
        and manifest.get("admission_passed") is False
    ):
        return (
            False,
            f"not admitted: {passed}/{checks} checks passed; "
            f"critical_blockers={_as_int(manifest.get('critical_blocker_count'), -1)}",
        )
    if not (
        _eligible_full_artifact(manifest)
        and manifest.get("artifact_type") == "japan_g2_aggregate_admission"
        and manifest.get("admission_passed") is True
        and manifest.get("g2_complete") is True
        and manifest.get("component_scorecard_eligible") is True
        and manifest.get("immutable") is True
        and manifest.get("atomic_publish") is True
        and manifest.get("source_checksums_verified") is True
        and manifest.get("window_coverage_complete") is True
        and manifest.get("raw_compacted_parity") is True
        and manifest.get("official_corporate_actions_reconciled") is True
        and manifest.get("pit_liquidity_complete") is True
        and _as_int(manifest.get("critical_blocker_count"), -1) == 0
        and not manifest.get("critical_blockers")
        and checks > 0
        and passed == checks
        and promotion.get("g2_scorecard_credit") is True
        and all(
            promotion.get(key) is False
            for key in ("alpha", "historical_backtest", "production", "ibkr", "live_trading")
        )
    ):
        return False, "aggregate manifest admission predicates failed"
    if (
        contract.empty
        or not required_contract_columns.issubset(contract.columns)
        or len(contract) != checks
        or contract[["component", "check_id"]].isna().any().any()
        or contract.duplicated(["component", "check_id"]).any()
        or not contract["passed"].map(_is_true).all()
    ):
        return False, "component contract is incomplete, duplicated, or contains failed checks"

    expected_windows = _as_int(context["security"].get("listing_rows"), -1)
    expected_rics = len(
        set(context["active_map"].get("refinitiv_ric", pd.Series(dtype=str)).dropna().astype(str))
        | set(context["reference_map"].get("refinitiv_ric", pd.Series(dtype=str)).dropna().astype(str))
    )
    denominators = manifest.get("denominators", {})
    if not isinstance(denominators, dict) or not (
        expected_windows > 0
        and expected_rics > 0
        and _as_int(denominators.get("g1_listing_windows"), -1) == expected_windows
        and _as_int(denominators.get("g1_rics"), -1) == expected_rics
        and _as_int(denominators.get("active_raw_ohlcv_rows"), 0) > 0
        and _as_int(denominators.get("reference_raw_ohlcv_rows"), 0) > 0
        and _as_int(denominators.get("raw_union_rows"), 0)
        == _as_int(denominators.get("active_raw_ohlcv_rows"), -1)
        + _as_int(denominators.get("reference_raw_ohlcv_rows"), -1)
        and _as_int(denominators.get("corporate_action_rows"), 0) > 0
    ):
        return False, "aggregate denominators do not reconcile to admitted G1 and nonempty G2 components"

    artifact_names = {"component_contract", "source_hashes", "summary"}
    artifacts = manifest.get("artifacts", {})
    digests = manifest.get("artifact_sha256", {})
    if not (
        isinstance(artifacts, dict)
        and isinstance(digests, dict)
        and artifact_names.issubset(artifacts)
        and artifact_names.issubset(digests)
    ):
        return False, "aggregate artifact paths or SHA-256 coverage is incomplete"
    for name in artifact_names:
        path = Path(str(artifacts[name]))
        path = path if path.is_absolute() else root / path
        digest = str(digests[name]).lower()
        if not path.is_file() or not _SHA256_RE.fullmatch(digest) or _sha256(path) != digest:
            return False, f"aggregate artifact hash mismatch: {name}"

    inventories = source_hashes.get("inventories", {})
    if not (
        source_hashes.get("algorithm") == "sha256"
        and source_hashes.get("coverage_complete") is True
        and source_hashes.get("immutable_source_verification") is True
        and isinstance(inventories, dict)
        and {"active", "reference", "components"}.issubset(inventories)
    ):
        return False, "aggregate source inventory contract is incomplete"
    for name in ("active", "reference", "components"):
        entry = inventories.get(name, {})
        files = entry.get("files", {}) if isinstance(entry, dict) else {}
        inventory_path = Path(str(entry.get("inventory_path", ""))) if isinstance(entry, dict) else Path()
        inventory_digest = str(entry.get("inventory_sha256", "")).lower() if isinstance(entry, dict) else ""
        if not (
            inventory_path.is_file()
            and _SHA256_RE.fullmatch(inventory_digest)
            and _sha256(inventory_path) == inventory_digest
            and isinstance(files, dict)
            and files
        ):
            return False, f"{name} source inventory is missing, empty, or changed"
    return True, "pass"


def _strict_g3_admission_ready(context: dict[str, Any]) -> tuple[bool, str]:
    paths: JapanDataQualityPaths = context["paths"]
    root = paths.g3_admission_dir
    if root is None:
        return False, "aggregate admission directory is missing"
    manifest = context["g3_admission"]
    contract = context["g3_contract"]
    source_hashes = context["g3_source_hashes"]
    checks = _as_int(manifest.get("check_count"), -1)
    passed = _as_int(manifest.get("passed_check_count"), -2)
    if (
        manifest.get("artifact_type") == "japan_g3_aggregate_admission"
        and manifest.get("admission_passed") is False
    ):
        return (
            False,
            f"not admitted: {passed}/{checks} checks passed; "
            f"critical_blockers={_as_int(manifest.get('critical_blocker_count'), -1)}",
        )
    promotion = manifest.get("promotion_policy", {})
    if not (
        _eligible_full_artifact(manifest)
        and manifest.get("artifact_type") == "japan_g3_aggregate_admission"
        and manifest.get("admission_passed") is True
        and manifest.get("g3_complete") is True
        and manifest.get("component_scorecard_eligible") is True
        and manifest.get("immutable") is True
        and manifest.get("atomic_publish") is True
        and manifest.get("source_checksums_verified") is True
        and manifest.get("producer_validator_independence_verified") is True
        and manifest.get("decision_calendar_eligibility_recomputed") is True
        and _as_int(manifest.get("critical_blocker_count"), -1) == 0
        and not manifest.get("critical_blockers")
        and checks > 0
        and passed == checks
        and set(map(str, manifest.get("datasets", []))) == _REQUIRED_PIT_DATASETS
        and set(map(str, manifest.get("lanes", [])))
        == {"fundamentals", "consensus", "earnings", "analyst_detail"}
        and promotion.get("g3_scorecard_credit") is True
        and all(
            promotion.get(key) is False
            for key in ("alpha", "historical_backtest", "production", "ibkr", "live_trading")
        )
    ):
        return False, "aggregate manifest admission predicates failed"
    if (
        contract.empty
        or not {"component", "check_id", "passed"}.issubset(contract.columns)
        or len(contract) != checks
        or contract[["component", "check_id"]].isna().any().any()
        or contract.duplicated(["component", "check_id"]).any()
        or not contract["passed"].map(_is_true).all()
    ):
        return False, "component contract is incomplete, duplicated, or contains failed checks"

    expected_windows = _as_int(context["security"].get("listing_rows"), -1)
    expected_rics = len(
        set(context["active_map"].get("refinitiv_ric", pd.Series(dtype=str)).dropna().astype(str))
        | set(context["reference_map"].get("refinitiv_ric", pd.Series(dtype=str)).dropna().astype(str))
    )
    denominators = manifest.get("denominators", {})
    if not isinstance(denominators, dict) or not (
        _as_int(denominators.get("g1_rics"), -1) == expected_rics > 0
        and _as_int(denominators.get("g1_listing_windows"), -1) == expected_windows > 0
        and _as_int(denominators.get("execution_plan_rows"), 0) > 0
        and _as_int(denominators.get("terminal_rows"), 0)
        == _as_int(denominators.get("execution_plan_rows"), -1)
        and _as_int(denominators.get("history_rows"), 0) > 0
    ):
        return False, "aggregate G1, plan, terminal, or history denominators do not reconcile"

    artifacts = manifest.get("artifacts", {})
    digests = manifest.get("artifact_sha256", {})
    required_artifacts = {"component_contract", "source_hashes", "summary"}
    if not (
        isinstance(artifacts, dict)
        and isinstance(digests, dict)
        and required_artifacts.issubset(artifacts)
        and required_artifacts.issubset(digests)
    ):
        return False, "aggregate artifact paths or SHA-256 coverage is incomplete"
    for name in required_artifacts:
        path = Path(str(artifacts[name]))
        path = path if path.is_absolute() else root / path
        digest = str(digests[name]).lower()
        if not path.is_file() or not _SHA256_RE.fullmatch(digest) or _sha256(path) != digest:
            return False, f"aggregate artifact hash mismatch: {name}"

    files = source_hashes.get("files", {})
    if not (
        source_hashes.get("algorithm") == "sha256"
        and source_hashes.get("coverage_complete") is True
        and source_hashes.get("immutable_source_verification") is True
        and _as_int(source_hashes.get("missing_source_count"), -1) == 0
        and isinstance(files, dict)
        and files
        and _as_int(source_hashes.get("source_count"), -1) == len(files)
    ):
        return False, "aggregate source inventory contract is incomplete"
    for raw_path, raw_digest in files.items():
        path = Path(str(raw_path))
        digest = str(raw_digest).lower()
        if not path.is_file() or not _SHA256_RE.fullmatch(digest) or _sha256(path) != digest:
            return False, f"aggregate source changed: {raw_path}"
    return True, "pass"


def _terminal_ric_window_coverage(
    run: dict[str, Any], plan: pd.DataFrame, manifest: pd.DataFrame
) -> tuple[bool, str]:
    required_plan = {"plan_row_id", "refinitiv_ric", "window_start", "window_end"}
    if not _eligible_full_artifact(run):
        return False, "historical run is not full-scope and scorecard-eligible"
    if plan.empty or manifest.empty or not required_plan.issubset(plan.columns) or "plan_row_id" not in manifest:
        return False, "plan/manifest or RIC-window keys are missing"
    if plan["plan_row_id"].isna().any() or plan["plan_row_id"].duplicated().any():
        return False, "plan row IDs are null or duplicated"
    if plan.duplicated(["refinitiv_ric", "window_start", "window_end"]).any():
        return False, "RIC/listing windows are duplicated"
    if manifest["plan_row_id"].isna().any() or manifest["plan_row_id"].duplicated().any():
        return False, "terminal manifest row IDs are null or duplicated"
    if set(plan["plan_row_id"].astype(str)) != set(manifest["plan_row_id"].astype(str)):
        return False, "plan and terminal manifest do not reconcile one-to-one"
    if not (
        _as_int(run.get("planned_rows")) == len(plan) == len(manifest)
        and _as_int(run.get("terminal_rows")) == len(plan)
        and _as_int(run.get("failed_rows"), -1) == 0
        and _as_float(run.get("ric_window_coverage")) == 1.0
        and run.get("window_coverage_complete") is True
    ):
        return False, "run-level terminal coverage is not exactly complete"
    status = manifest.get("status", pd.Series("", index=manifest.index)).astype(str)
    completed = status.eq("completed")
    structural = status.isin({"completed_prelisting", "completed_postretirement"})
    if not (completed | structural).all():
        return False, "non-terminal, partial, empty, or failed extraction rows remain"
    prices = pd.to_numeric(manifest.get("price_columns", pd.Series(0, index=manifest.index)), errors="coerce").fillna(0)
    rows = pd.to_numeric(manifest.get("long_ohlcv_rows", pd.Series(0, index=manifest.index)), errors="coerce").fillna(0)
    if not (prices[completed].gt(0).all() and rows[completed].gt(0).all()):
        return False, "completed rows contain zero OHLCV payload"
    structural_reason = manifest.get("structural_reason", pd.Series("", index=manifest.index)).fillna("").astype(str)
    if structural.any() and structural_reason[structural].str.strip().eq("").any():
        return False, "structural terminal rows lack an explicit reason"
    return True, "complete"


def _corporate_actions_ready(manifest: dict[str, Any]) -> bool:
    planned = _as_int(manifest.get("planned_rows"))
    return bool(
        _eligible_full_artifact(manifest)
        and planned > 0
        and _as_int(manifest.get("completed_rows")) == planned
        and _as_int(manifest.get("failed_rows"), -1) == 0
        and _as_int(manifest.get("empty_rows"), -1) == 0
        and _as_int(manifest.get("alpha_usable_rows")) > 0
        and _as_float(manifest.get("known_at_coverage")) == 1.0
        and _as_int(manifest.get("missing_known_at_rows"), -1) == 0
        and manifest.get("official_source") is True
    )


def _adjusted_reconciliation_ready(manifest: dict[str, Any]) -> bool:
    return bool(
        _eligible_full_artifact(manifest)
        and manifest.get("reconciliation_complete") is True
        and manifest.get("production_adjusted_return_ready") is True
        and _as_int(manifest.get("reconciled_rows")) > 0
        and _as_float(manifest.get("matched_event_share")) == 1.0
        and _as_int(manifest.get("unmatched_event_rows"), -1) == 0
        and _as_int(manifest.get("quarantined_rows"), -1) == 0
        and _as_float(manifest.get("quarantine_share"), 1.0) == 0.0
    )


def _pit_history_ready(
    manifest: dict[str, Any], quality: pd.DataFrame, required_datasets: set[str]
) -> bool:
    planned = _as_int(manifest.get("planned_rows"))
    if not (
        _eligible_full_artifact(manifest)
        and planned > 0
        and _as_int(manifest.get("completed_rows")) == planned
        and _as_int(manifest.get("failed_rows"), -1) == 0
        and _as_int(manifest.get("empty_rows"), -1) == 0
        and _as_int(manifest.get("history_rows")) > 0
        and _as_float(manifest.get("known_at_coverage")) == 1.0
        and _as_int(manifest.get("missing_known_at_rows"), -1) == 0
        and _as_int(manifest.get("pit_timestamp_violations"), -1) == 0
    ):
        return False
    required_columns = {
        "dataset_id",
        "failed_rows",
        "history_rows",
        "coverage",
        "known_at_coverage",
        "pit_timestamp_violations",
    }
    if quality.empty or not required_columns.issubset(quality.columns):
        return False
    rows = quality.loc[quality["dataset_id"].astype(str).isin(required_datasets)].copy()
    if set(rows["dataset_id"].astype(str)) != required_datasets or len(rows) != len(required_datasets):
        return False
    numeric = rows[list(required_columns - {"dataset_id"})].apply(pd.to_numeric, errors="coerce")
    return bool(
        numeric.notna().all().all()
        and numeric["failed_rows"].eq(0).all()
        and numeric["history_rows"].gt(0).all()
        and numeric["coverage"].eq(1.0).all()
        and numeric["known_at_coverage"].eq(1.0).all()
        and numeric["pit_timestamp_violations"].eq(0).all()
    )


def _component_concepts(values: Iterable[Any]) -> set[str]:
    names = {str(value) for value in values}
    return {
        concept
        for concept, aliases in _G4_COMPONENT_ALIASES.items()
        if aliases.intersection(names)
    }


def _validated_quality_components(quality: pd.DataFrame) -> set[str]:
    required = {"component", "status", "validated", "coverage", "source_rows"}
    if quality.empty or not required.issubset(quality.columns):
        return set()
    validated = quality["validated"].astype(str).str.lower().isin({"true", "1", "yes"})
    coverage = pd.to_numeric(quality["coverage"], errors="coerce").eq(1.0)
    rows = pd.to_numeric(quality["source_rows"], errors="coerce").gt(0)
    passed = quality["status"].astype(str).eq("pass")
    return _component_concepts(quality.loc[validated & coverage & rows & passed, "component"])


def _strict_g4_assembly_ready(context: dict[str, Any]) -> tuple[bool, str]:
    manifest = context["fx"]
    quality = context["fx_quality"]
    root = context["paths"].fx_calendar_tca_dir
    if root is None:
        return False, "assembly directory is missing"
    promotion = manifest.get("promotion_policy", {})
    component_admission = manifest.get("component_admission", {})
    required_admissions = {
        "currency_identity",
        "exchange_calendar",
        "historical_halts_suspensions",
        "tick_sizes",
        "pit_tick_rules",
        "pit_tick_assignment",
        "pit_lot_sizes",
        "normal_price_limits",
        "exceptional_price_limits",
        "price_limits",
        "liquidity_aware_tca",
    }
    if not (
        manifest.get("artifact_type") == "jpx_g4_full_evidence_assembly"
        and manifest.get("status") == "complete"
        and manifest.get("scorecard_eligible") is True
        and manifest.get("evidence_scope") == "full_hash_verified_g4_2006_07_to_2026_06"
        and manifest.get("research_only") is True
        and manifest.get("production_ready") is False
        and manifest.get("live_execution_ready") is False
        and manifest.get("currency") == "JPY"
        and manifest.get("official_jpx_calendar") is True
        and manifest.get("tca_calibrated") is True
        and not manifest.get("partial_components")
        and not manifest.get("critical_blockers")
        and isinstance(component_admission, dict)
        and required_admissions.issubset(component_admission)
        and all(component_admission[key] is True for key in required_admissions)
        and all(
            promotion.get(key) is False
            for key in ("alpha", "historical_backtest", "production", "ibkr", "live_trading")
        )
    ):
        return False, "full assembly manifest predicates failed"
    required_components = set(_G4_COMPONENT_ALIASES)
    concepts = _validated_quality_components(quality)
    if (
        quality.empty
        or quality["component"].astype(str).duplicated().any()
        or concepts != required_components
    ):
        return False, "quality rows are duplicated or do not exactly cover all required components"
    hashes = manifest.get("sha256", {})
    if not isinstance(hashes, dict) or not hashes or "evidence_quality.csv" not in hashes:
        return False, "assembly SHA-256 inventory is empty or omits evidence quality"
    root = root.resolve()
    for raw_name, raw_digest in hashes.items():
        relative = Path(str(raw_name))
        digest = str(raw_digest).lower()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            return False, f"assembly hash path escapes root: {raw_name}"
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _SHA256_RE.fullmatch(digest)
            or not path.is_file()
            or _sha256(path) != digest
        ):
            return False, f"assembly artifact hash mismatch: {raw_name}"
    return True, "pass"


def _immutable_hash_inventory_ready(context: dict[str, Any]) -> tuple[bool, str]:
    inventory = context["hashes"]
    hash_path: Path = context["hash_path"]
    if not (
        _eligible_full_artifact(inventory)
        and inventory.get("algorithm") == "sha256"
        and inventory.get("immutable") is True
        and inventory.get("coverage_complete") is True
    ):
        return False, "full immutable SHA-256 inventory manifest is missing"
    files = inventory.get("files")
    if not isinstance(files, dict) or not files:
        return False, "hash inventory has no files"

    inventoried: dict[Path, str] = {}
    for raw_path, raw_digest in files.items():
        digest = str(raw_digest).lower()
        if not _SHA256_RE.fullmatch(digest):
            return False, f"invalid SHA-256 for {raw_path}"
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = hash_path.parent / path
        path = path.resolve()
        if not path.is_file():
            return False, f"hashed file is missing: {raw_path}"
        if _sha256(path) != digest:
            return False, f"hash mismatch: {raw_path}"
        inventoried[path] = digest

    required = {path.resolve() for path in _required_hash_inputs(context)}
    missing = sorted(str(path) for path in required - set(inventoried))
    if missing:
        return False, f"hash coverage missing {len(missing)} gate input(s)"
    return True, "pass"


def _strict_admission_hash_inputs(root: Path, *, inventory_style: str) -> list[Path]:
    inputs = [
        root / "manifest.json",
        root / "component_contract.csv",
        root / "source_hashes.json",
        root / "summary.md",
    ]
    payload = _read_json(root / "source_hashes.json")
    if inventory_style == "nested":
        inventories = payload.get("inventories", {})
        if isinstance(inventories, dict):
            for entry in inventories.values():
                if not isinstance(entry, dict):
                    continue
                inventory_path = Path(str(entry.get("inventory_path", "")))
                if str(inventory_path) not in {"", "."}:
                    inputs.append(
                        inventory_path if inventory_path.is_absolute() else root / inventory_path
                    )
                files = entry.get("files", {})
                if isinstance(files, dict):
                    for raw_path in files:
                        path = Path(str(raw_path))
                        inputs.append(path if path.is_absolute() else root / path)
    elif inventory_style == "flat":
        files = payload.get("files", {})
        if isinstance(files, dict):
            for raw_path in files:
                path = Path(str(raw_path))
                inputs.append(path if path.is_absolute() else root / path)
    else:
        raise ValueError(f"unsupported strict admission inventory style: {inventory_style}")
    return inputs


def _required_hash_inputs(context: dict[str, Any]) -> list[Path]:
    paths: JapanDataQualityPaths = context["paths"]
    candidates = [
        _child(paths.security_master_dir, "manifest.json"),
        _child(paths.security_master_dir, "entities.parquet"),
        _child(paths.security_master_dir, "listing_windows.parquet"),
        _child(paths.security_master_dir, "identifier_history.parquet"),
        _child(paths.security_master_dir, "universe_membership_pit.parquet"),
        _child(paths.universe_reconciliation_dir, "manifest.json"),
        _child(paths.active_symbol_map_dir, "manifest.json"),
        _child(paths.active_symbol_map_dir, "symbol_map.csv"),
        _child(paths.reference_symbol_map_dir, "manifest.json"),
        _child(paths.reference_symbol_map_dir, "symbol_map.csv"),
        _child(paths.fx_calendar_tca_dir, "manifest.json"),
        _child(paths.fx_calendar_tca_dir, "evidence_quality.csv"),
        _child(paths.leakage_qa_dir, "manifest.json"),
    ]
    if paths.g2_admission_dir is not None:
        candidates.extend(
            _strict_admission_hash_inputs(paths.g2_admission_dir, inventory_style="nested")
        )
    else:
        candidates.extend(
            [
                _child(paths.historical_dir, "historical_run_manifest.json"),
                _child(paths.historical_dir, "historical_plan.csv"),
                _child(paths.historical_dir, "historical_manifest.csv"),
                _child(paths.depth_dir, "overall.csv"),
                _child(paths.compaction_dir, "manifest.json"),
                _child(paths.corporate_actions_dir, "manifest.json"),
                _child(paths.adjusted_returns_dir, "manifest.json"),
                _child(paths.liquidity_dir, "manifest.json"),
                _child(paths.liquidity_dir, "historical_quality.csv"),
            ]
        )
    if paths.g3_admission_dir is not None:
        candidates.extend(
            _strict_admission_hash_inputs(paths.g3_admission_dir, inventory_style="flat")
        )
    else:
        candidates.extend(
            [
                _child(paths.pit_plan_dir, "manifest.json"),
                _child(paths.fundamentals_ibes_events_dir, "manifest.json"),
                _child(paths.fundamentals_ibes_events_dir, "historical_quality.csv"),
            ]
        )
    if paths.commands_log is not None:
        candidates.append(paths.commands_log)
    if paths.full_reference_symbol_map_dir is not None:
        candidates.extend(
            [
                _child(paths.full_reference_symbol_map_dir, "manifest.json"),
                _child(paths.full_reference_symbol_map_dir, "symbol_map.csv"),
            ]
        )
    leakage_artifacts = context.get("leakage", {}).get("artifacts", {})
    if isinstance(leakage_artifacts, dict):
        for raw_path in leakage_artifacts.values():
            path = Path(str(raw_path))
            candidates.append(path if path.is_absolute() else _child(paths.leakage_qa_dir, str(raw_path)))
    for directory in (
        paths.fx_calendar_tca_dir,
        paths.current_enrichment_dir,
        paths.listing_valid_time_dir,
        paths.jquants_listed_info_dir,
        paths.jquants_listed_shares_dir,
        paths.g1_materializer_dir,
        paths.g1_validation_dir,
        paths.jpx_public_listing_events_dir,
        paths.jpx_public_event_reconciliation_dir,
        paths.jpx_monthly_quotations_dir,
        paths.jpx_monthly_membership_qa_dir,
        paths.jpx_monthly_quotations_extension_dir,
        paths.ose_monthly_quotations_dir,
        paths.ose_monthly_quotations_extension_dir,
        paths.ose_main_monthly_quotations_dir,
        paths.combined_monthly_membership_qa_dir,
        paths.monthly_microstructure_dir,
        paths.monthly_microstructure_validation_dir,
        paths.exchange_calendar_dir,
        paths.exchange_calendar_validation_dir,
        paths.market_rule_history_dir,
        paths.market_rule_history_validation_dir,
    ):
        if directory is None or not directory.is_dir():
            continue
        candidates.extend(
            path
            for path in directory.rglob("*")
            if path.is_file() and not path.name.endswith((".tmp", ".lock"))
        )
    return list(dict.fromkeys(candidates))


def required_japan_hash_inputs(paths: JapanDataQualityPaths) -> list[Path]:
    """Return the exact file inventory required by the Japan G5 scorecard."""

    return _required_hash_inputs({"paths": paths})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _depth_has_critical_errors(depth: pd.DataFrame) -> bool:
    if len(depth) != 1:
        return True
    row = depth.iloc[0]
    labels = " ".join(str(row.get(key, "")) for key in ("run_id", "mode", "scope_profile")).lower()
    return bool(
        not _is_true(row.get("scorecard_eligible"))
        or row.get("evidence_scope") != "full"
        or any(marker in labels for marker in ("smoke", "stub", "partial", "policy"))
        or _as_int(row.get("rows")) <= 0
        or _as_int(row.get("tickers")) <= 0
        or _as_int(row.get("non_positive_close_rows"), -1) != 0
        or _as_int(row.get("duplicate_ohlcv_rows"), -1) != 0
        or _as_int(row.get("outside_listing_window_rows"), -1) != 0
        or _as_float(row.get("volume_missing_pct"), 1.0) != 0.0
    )


def _compaction_passes(manifest: dict[str, Any]) -> bool:
    source = _as_int(manifest.get("source_rows"))
    output = _as_int(manifest.get("output_rows"))
    return bool(
        _eligible_full_artifact(manifest)
        and manifest.get("status") == "complete"
        and source > 0
        and source == output
        and _as_int(manifest.get("failed_groups"), -1) == 0
    )


def _nonusable_historical_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = list(frame.columns) if not frame.empty else ["plan_row_id", "status", "reason"]
    if frame.empty or "status" not in frame:
        return pd.DataFrame(columns=columns)
    status = frame["status"].astype(str)
    completed = status.eq("completed")
    structural = status.isin({"completed_prelisting", "completed_postretirement"})
    prices = pd.to_numeric(frame.get("price_columns", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    rows = pd.to_numeric(frame.get("long_ohlcv_rows", pd.Series(0, index=frame.index)), errors="coerce").fillna(0)
    reason = frame.get("structural_reason", pd.Series("", index=frame.index)).fillna("").astype(str)
    usable = (completed & prices.gt(0) & rows.gt(0)) | (structural & reason.str.strip().ne(""))
    result = frame.loc[~usable].copy()
    if not result.empty:
        result["reason"] = "non_terminal_or_zero_ohlcv_payload"
    return result


def _next_queue(lanes: tuple[JapanQualityLane, ...]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    sequence = 1
    for lane in lanes:
        for blocker in lane.critical_blockers:
            rows.append(
                {
                    "sequence": sequence,
                    "lane_id": lane.lane_id,
                    "owner_agent": lane.lead_agent,
                    "severity": "critical",
                    "blocker": blocker,
                    "next_action": lane.next_action,
                    "promotes_alpha": False,
                }
            )
            sequence += 1
    return pd.DataFrame(
        rows,
        columns=["sequence", "lane_id", "owner_agent", "severity", "blocker", "next_action", "promotes_alpha"],
    )


def _summary(
    run_id: str,
    lanes: tuple[JapanQualityLane, ...],
    overall_score: int,
    accepted: bool,
    blockers: list[str],
) -> str:
    lines = [
        "# Japan Institutional Data Quality Loop",
        "",
        f"Run ID: `{run_id}`",
        f"Status: `{'accepted_backtest_5_of_5' if accepted else 'continue_japan_data_quality_loop'}`",
        f"Accepted: `{accepted}`",
        f"Overall score: `{overall_score}/5`",
        "",
        "Each gate is binary and fail-closed. This score is for leakage-controlled research backtesting only. Production, IBKR, and live execution remain blocked.",
        "",
        "## Gate Scorecard",
        "",
        "| Gate | Auditor | Score | Status | Evidence |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for lane in lanes:
        lines.append(f"| `{lane.lane_id}` | `{lane.lead_agent}` | {lane.score}/1 | {lane.status} | {lane.evidence} |")
    lines.extend(["", "## Critical Blockers", ""])
    lines.extend([f"- {blocker}" for blocker in blockers] or ["- None."])
    lines.extend(
        [
            "",
            "## Admission Rule",
            "",
            "Japan enters leakage-controlled alpha research only when all five independent gates pass. Partial, smoke, current-quote, current-survivor, or policy-stub evidence never passes a gate.",
            "",
        ]
    )
    return "\n".join(lines)


def _child(directory: Path | None, name: str) -> Path:
    return directory / name if directory is not None else Path("/__japan_data_quality_missing__") / name


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError, UnicodeError):
        return pd.DataFrame()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        try:
            return pd.read_parquet(path)
        except (OSError, ValueError):
            return pd.DataFrame()
    return _read_csv(path)
