from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


ACTIVE_COUNTRIES: tuple[str, ...] = ("Malaysia", "Singapore", "Indonesia")
REMOVED_COUNTRIES: tuple[str, ...] = (
    "Vietnam",
    "Philippines",
    "China",
    "Taiwan",
)
PASSING_STATUSES = {"pass"}
NON_FAILING_STATUSES = {"pass", "partial"}


@dataclass(frozen=True)
class DataQualityCriterion:
    lane_id: str
    criterion_id: str
    description: str
    status: str
    points: float
    max_points: float
    severity: str
    evidence: str
    blocker: str
    next_action: str


@dataclass(frozen=True)
class DataQualityLane:
    lane_id: str
    lead_agent: str
    score: float
    max_score: float
    status: str
    critical_blockers: tuple[str, ...]
    criteria: tuple[DataQualityCriterion, ...]


@dataclass(frozen=True)
class DataQualityLoopResult:
    run_id: str
    artifact_dir: Path
    overall_score: float
    max_score: float
    accepted: bool
    status: str
    lanes: tuple[DataQualityLane, ...]
    summary_path: Path
    manifest_path: Path
    lane_scorecard_path: Path
    criteria_scorecard_path: Path
    next_queue_path: Path


@dataclass(frozen=True)
class InstitutionalDataQualityPaths:
    l2_artifact_dir: Path
    pit_plan_dir: Path
    field_probe_dir: Path
    route_probe_dir: Path
    route_history_map_dir: Path
    current_snapshot_dir: Path
    liquidity_backfill_dir: Path
    fund_ibes_events_backfill_dir: Path
    pit_leakage_qa_dir: Path
    critical_route_history_map_dir: Path | None = None
    critical_fields_backfill_dir: Path | None = None
    security_master_history_dir: Path | None = None
    corporate_actions_event_dir: Path | None = None
    fx_country_risk_dir: Path | None = None
    corporate_action_reconciliation_dir: Path | None = None
    adjusted_return_reconstruction_dir: Path | None = None
    ibes_revision_route_probe_dir: Path | None = None
    pit_enhanced_alpha_dir: Path | None = None


def default_institutional_data_quality_paths(
    project_root: str | Path = ".",
) -> InstitutionalDataQualityPaths:
    root = Path(project_root)
    return InstitutionalDataQualityPaths(
        l2_artifact_dir=root
        / "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708",
        pit_plan_dir=root / "artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708",
        field_probe_dir=root / "artifacts/refinitiv_pit_field_probe/REFINITIV-PIT-FIELD-PROBE-MYS-SGP-IDN-20260708",
        route_probe_dir=root / "artifacts/refinitiv_pit_route_probe/REFINITIV-PIT-ROUTE-PROBE-MYS-SGP-IDN-20260708",
        route_history_map_dir=root
        / "artifacts/refinitiv_pit_route_history_map/REFINITIV-PIT-ROUTE-HISTORY-MAP-MYS-SGP-IDN-20260708",
        current_snapshot_dir=root / "artifacts/refinitiv_current_snapshot/REFINITIV-CURRENT-SNAPSHOT-MYS-SGP-IDN-20260708",
        liquidity_backfill_dir=root
        / "artifacts/refinitiv_pit_historical_backfill/REFINITIV-PIT-HISTORICAL-LIQUIDITY-FULL-MYS-SGP-IDN-20260708",
        fund_ibes_events_backfill_dir=root
        / "artifacts/refinitiv_pit_historical_backfill/REFINITIV-PIT-HISTORICAL-FUND-IBES-EVENTS-FULL-MYS-SGP-IDN-20260708",
        critical_fields_backfill_dir=root
        / "artifacts/refinitiv_pit_historical_backfill/REFINITIV-PIT-HISTORICAL-CRITICAL-FIELDS-U2-20260708",
        pit_leakage_qa_dir=root / "artifacts/pit_leakage_qa/PIT-LEAKAGE-QA-MYS-SGP-IDN-FULL-U2-20260708",
        critical_route_history_map_dir=root
        / "artifacts/refinitiv_pit_route_history_map/REFINITIV-PIT-CRITICAL-ROUTE-HISTORY-MAP-MYS-SGP-IDN-20260708",
        security_master_history_dir=root / "artifacts/data_lake/pit/security_master_history/MYS-SGP-IDN-U2-20260708",
        corporate_actions_event_dir=root
        / "artifacts/refinitiv_corporate_actions_events/REFINITIV-CORPACT-EVENTS-U2-MYS-SGP-IDN-20260708",
        fx_country_risk_dir=root / "artifacts/fx_country_risk/FX-COUNTRY-RISK-MYS-SGP-IDN-U2-20260708",
        corporate_action_reconciliation_dir=root
        / "artifacts/corporate_action_reconciliation/CORPACT-RECON-MYS-SGP-IDN-U2-20260708",
        adjusted_return_reconstruction_dir=root
        / "artifacts/adjusted_return_reconstruction/ADJRET-RECON-MYS-SGP-IDN-U2-20260708",
        ibes_revision_route_probe_dir=root
        / "artifacts/refinitiv_ibes_revision_route_probe/REFINITIV-IBES-REVISION-ROUTE-PROBE-MYS-SGP-IDN-20260708",
        pit_enhanced_alpha_dir=root / "artifacts/pit_enhanced_alpha/PIT-ENHANCED-ABLATION-U2-GUARDED-20260708",
    )


def run_institutional_data_quality_loop(
    paths: InstitutionalDataQualityPaths | None = None,
    artifact_root: str | Path = "artifacts/institutional_data_quality",
    run_id: str | None = None,
    project_root: str | Path = ".",
    acceptance_score: float = 5.0,
) -> DataQualityLoopResult:
    """Run the hard data-quality gate before deeper backtesting.

    The loop is intentionally conservative: a data lake can support guarded
    research while still failing production-quality backtest readiness.
    """

    paths = paths or default_institutional_data_quality_paths(project_root)
    run_id = run_id or _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    context = _load_quality_context(paths)
    lanes = (
        _security_master_lane(context),
        _price_actions_liquidity_lane(context),
        _pit_fund_ibes_events_lane(context),
        _leakage_bias_repro_lane(context),
        _extraction_operations_lane(context),
    )
    overall_score = round(float(sum(lane.score for lane in lanes) / max(1, len(lanes))), 3)
    critical_blockers = [blocker for lane in lanes for blocker in lane.critical_blockers]
    accepted = (
        overall_score >= acceptance_score
        and all(lane.score >= acceptance_score for lane in lanes)
        and not critical_blockers
    )
    status = "accepted_5_of_5" if accepted else "continue_data_quality_loop"

    lane_scorecard = _lane_scorecard(lanes)
    criteria_scorecard = _criteria_scorecard(lanes)
    next_queue = _next_queue(lanes)

    lane_scorecard_path = artifact_dir / "lane_scorecard.csv"
    criteria_scorecard_path = artifact_dir / "criteria_scorecard.csv"
    next_queue_path = artifact_dir / "agent_next_queue.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"

    lane_scorecard.to_csv(lane_scorecard_path, index=False)
    criteria_scorecard.to_csv(criteria_scorecard_path, index=False)
    next_queue.to_csv(next_queue_path, index=False)
    summary_path.write_text(
        _summary_md(run_id, lanes, overall_score, accepted, status, paths, next_queue),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "overall_score": overall_score,
        "max_score": 5.0,
        "accepted": accepted,
        "status": status,
        "critical_blockers": critical_blockers,
        "lane_count": len(lanes),
        "next_queue_rows": int(len(next_queue)),
        "generated_at_utc": _now_utc(),
        "paths": _paths_manifest(paths),
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "lane_scorecard": str(lane_scorecard_path.resolve()),
            "criteria_scorecard": str(criteria_scorecard_path.resolve()),
            "agent_next_queue": str(next_queue_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return DataQualityLoopResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        overall_score=overall_score,
        max_score=5.0,
        accepted=accepted,
        status=status,
        lanes=lanes,
        summary_path=summary_path,
        manifest_path=manifest_path,
        lane_scorecard_path=lane_scorecard_path,
        criteria_scorecard_path=criteria_scorecard_path,
        next_queue_path=next_queue_path,
    )


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("DATA-QUALITY-LOOP-%Y%m%d-%H%M%S")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _adjusted_price_equal_share(l2_artifact_dir: Path) -> float:
    path = l2_artifact_dir / "daily_features.parquet"
    if not path.exists():
        return 1.0
    try:
        data = pd.read_parquet(path, columns=["close", "adj_close"])
    except Exception:
        return 1.0
    if data.empty:
        return 1.0
    close = pd.to_numeric(data.get("close"), errors="coerce")
    adj_close = pd.to_numeric(data.get("adj_close"), errors="coerce")
    valid = close.notna() & adj_close.notna()
    if not valid.any():
        return 1.0
    return float(close[valid].eq(adj_close[valid]).mean())


def _load_quality_context(paths: InstitutionalDataQualityPaths) -> dict[str, Any]:
    base_route_results = _read_csv(paths.route_probe_dir / "route_probe_results.csv")
    critical_route_results = (
        _read_csv(paths.critical_route_history_map_dir / "route_probe_results.csv")
        if paths.critical_route_history_map_dir
        else pd.DataFrame()
    )
    ibes_revision_route_results = (
        _read_csv(paths.ibes_revision_route_probe_dir / "route_probe_results.csv")
        if paths.ibes_revision_route_probe_dir
        else pd.DataFrame()
    )
    route_results = _concat_frames([base_route_results, critical_route_results, ibes_revision_route_results])
    return {
        "paths": paths,
        "l2_manifest": _read_json(paths.l2_artifact_dir / "manifest.json"),
        "l2_country_summary": _read_csv(paths.l2_artifact_dir / "country_summary.csv"),
        "l2_ticker_quality": _read_csv(paths.l2_artifact_dir / "ticker_quality.csv"),
        "l2_vendor_matrix": _read_csv(paths.l2_artifact_dir / "vendor_extraction_matrix.csv"),
        "l2_adjusted_price_equal_share": _adjusted_price_equal_share(paths.l2_artifact_dir),
        "pit_plan_manifest": _read_json(paths.pit_plan_dir / "manifest.json"),
        "pit_selected_universe": _read_csv(paths.pit_plan_dir / "selected_universe.csv"),
        "pit_readiness_gates": _read_csv(paths.pit_plan_dir / "readiness_gates.csv"),
        "field_accepted_map": _read_csv(paths.field_probe_dir / "accepted_field_map.csv"),
        "route_probe_results": route_results,
        "route_history_map": _read_csv(paths.route_history_map_dir / "history_field_map.csv"),
        "current_snapshot_manifest": _read_json(paths.current_snapshot_dir / "manifest.json"),
        "current_snapshot_quality": _read_csv(paths.current_snapshot_dir / "snapshot_quality.csv"),
        "liquidity_manifest": _read_json(paths.liquidity_backfill_dir / "manifest.json"),
        "liquidity_quality": _read_csv(paths.liquidity_backfill_dir / "historical_quality.csv"),
        "fund_ibes_events_manifest": _read_json(paths.fund_ibes_events_backfill_dir / "manifest.json"),
        "fund_ibes_events_quality": _read_csv(paths.fund_ibes_events_backfill_dir / "historical_quality.csv"),
        "critical_fields_manifest": _read_json(paths.critical_fields_backfill_dir / "manifest.json")
        if paths.critical_fields_backfill_dir
        else {},
        "critical_fields_quality": _read_csv(paths.critical_fields_backfill_dir / "historical_quality.csv")
        if paths.critical_fields_backfill_dir
        else pd.DataFrame(),
        "corporate_actions_event_manifest": _read_json(paths.corporate_actions_event_dir / "manifest.json")
        if paths.corporate_actions_event_dir
        else {},
        "corporate_actions_event_quality": _read_csv(paths.corporate_actions_event_dir / "event_quality.csv")
        if paths.corporate_actions_event_dir
        else pd.DataFrame(),
        "fx_country_risk_manifest": _read_json(paths.fx_country_risk_dir / "manifest.json")
        if paths.fx_country_risk_dir
        else {},
        "fx_country_risk_quality": _read_csv(paths.fx_country_risk_dir / "fx_quality.csv")
        if paths.fx_country_risk_dir
        else pd.DataFrame(),
        "fx_currency_reference": _read_csv(paths.fx_country_risk_dir / "currency_reference.csv")
        if paths.fx_country_risk_dir
        else pd.DataFrame(),
        "corporate_action_reconciliation_manifest": _read_json(paths.corporate_action_reconciliation_dir / "manifest.json")
        if paths.corporate_action_reconciliation_dir
        else {},
        "corporate_action_reconciliation_quality": _read_csv(
            paths.corporate_action_reconciliation_dir / "reconciliation_quality.csv"
        )
        if paths.corporate_action_reconciliation_dir
        else pd.DataFrame(),
        "adjusted_return_reconstruction_manifest": _read_json(paths.adjusted_return_reconstruction_dir / "manifest.json")
        if paths.adjusted_return_reconstruction_dir
        else {},
        "adjusted_return_reconstruction_quality": _read_csv(
            paths.adjusted_return_reconstruction_dir / "reconciliation_quality.csv"
        )
        if paths.adjusted_return_reconstruction_dir
        else pd.DataFrame(),
        "ibes_revision_manifest": _read_json(paths.ibes_revision_route_probe_dir / "manifest.json")
        if paths.ibes_revision_route_probe_dir
        else {},
        "ibes_revision_accepted_map": _read_csv(paths.ibes_revision_route_probe_dir / "accepted_revision_time_map.csv")
        if paths.ibes_revision_route_probe_dir
        else pd.DataFrame(),
        "pit_leakage_manifest": _read_json(paths.pit_leakage_qa_dir / "manifest.json"),
        "pit_feature_quality": _read_csv(paths.pit_leakage_qa_dir / "feature_quality.csv"),
        "pit_leakage_summary": _read_csv(paths.pit_leakage_qa_dir / "leakage_summary.csv"),
        "pit_leakage_violations": _read_csv(paths.pit_leakage_qa_dir / "leakage_violations.csv"),
        "pit_enhanced_manifest": _read_json(paths.pit_enhanced_alpha_dir / "overfit_audit.json")
        if paths.pit_enhanced_alpha_dir
        else {},
    }


def _concat_frames(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    available = [frame for frame in frames if not frame.empty]
    if not available:
        return pd.DataFrame()
    return pd.concat(available, ignore_index=True, sort=False)


def _criterion(
    lane_id: str,
    criterion_id: str,
    description: str,
    status: str,
    evidence: str,
    blocker: str = "",
    next_action: str = "",
    severity: str = "medium",
    max_points: float = 1.0,
) -> DataQualityCriterion:
    normalized = status.strip().lower()
    if normalized == "pass":
        points = max_points
        blocker = ""
    elif normalized == "partial":
        points = max_points * 0.5
    else:
        normalized = "fail"
        points = 0.0
    return DataQualityCriterion(
        lane_id=lane_id,
        criterion_id=criterion_id,
        description=description,
        status=normalized,
        points=float(points),
        max_points=float(max_points),
        severity=severity,
        evidence=evidence,
        blocker=blocker,
        next_action=next_action,
    )


def _make_lane(lane_id: str, lead_agent: str, criteria: Sequence[DataQualityCriterion]) -> DataQualityLane:
    max_points = sum(item.max_points for item in criteria) or 1.0
    points = sum(item.points for item in criteria)
    score = round(float(points / max_points * 5.0), 3)
    critical_blockers = tuple(
        item.blocker
        for item in criteria
        if item.severity == "critical" and item.status != "pass" and item.blocker
    )
    if critical_blockers:
        status = "blocked"
    elif score >= 5.0:
        status = "pass"
    elif all(item.status in NON_FAILING_STATUSES for item in criteria):
        status = "partial"
    else:
        status = "fail"
    return DataQualityLane(
        lane_id=lane_id,
        lead_agent=lead_agent,
        score=score,
        max_score=5.0,
        status=status,
        critical_blockers=critical_blockers,
        criteria=tuple(criteria),
    )


def _security_master_lane(context: Mapping[str, Any]) -> DataQualityLane:
    lane = "security_master_universe"
    paths = context["paths"]
    l2_manifest = context["l2_manifest"]
    country_summary = context["l2_country_summary"]
    selected_universe = context["pit_selected_universe"]
    plan_manifest = context["pit_plan_manifest"]
    accepted = context["field_accepted_map"]
    route_results = context["route_probe_results"]
    security_history_ready, security_history_evidence = _security_master_history_status(paths)

    countries = set(country_summary.get("country", pd.Series(dtype=str)).dropna().astype(str))
    active_ok = countries == set(ACTIVE_COUNTRIES)
    removed_present = bool(countries.intersection(REMOVED_COUNTRIES))
    country_counts_ok = _country_min(country_summary, "u2_backtest_eligible", minimum=40)
    ric_coverage = _safe_ratio(plan_manifest.get("names_with_refinitiv_ric"), plan_manifest.get("selected_names"))
    snapshot_fields = set(accepted.query("dataset_id == 'security_master_pit'").get("field_role", pd.Series(dtype=str)).astype(str))
    has_core_snapshot_fields = {"isin", "first_trade_date"}.issubset(snapshot_fields)
    delisting_route = security_history_ready or _has_validated_route(
        route_results,
        "delistings_listing_history",
        field_roles={"delisting_date"},
        historical_only=True,
    ) or _has_validated_security_master_discovery_route(route_results, {"delisting_date"})
    ric_change_route = security_history_ready or _has_validated_route(
        route_results,
        "delistings_listing_history",
        field_roles={"ric_change"},
        historical_only=True,
    ) or _has_validated_security_master_discovery_route(route_results, {"ric_change"})
    listing_status_route = security_history_ready or "listing_status" in snapshot_fields or _has_validated_route(
        route_results, "security_master_pit", field_roles={"listing_status"}, historical_only=True
    ) or _has_validated_security_master_discovery_route(route_results, {"listing_status"})

    criteria = [
        _criterion(
            lane,
            "scope_integrity",
            "Universe scope is exactly Malaysia, Singapore, Indonesia; Vietnam and Philippines are absent.",
            "pass" if active_ok and not removed_present else "fail",
            f"countries={sorted(countries)}; removed_present={removed_present}",
            blocker="removed market leaked into active universe" if removed_present else "",
            next_action="regenerate universe tiers with active-country filter",
            severity="critical",
        ),
        _criterion(
            lane,
            "u2_breadth",
            "U2 has enough breadth for three-country cross-sectional tests.",
            "pass" if int(l2_manifest.get("u2_backtest_eligible", 0)) >= 150 and country_counts_ok else "partial",
            f"u2={l2_manifest.get('u2_backtest_eligible')}; country_counts={_country_count_dict(country_summary, 'u2_backtest_eligible')}",
            next_action="mine deeper listed/liquid names per country if any country falls below minimum breadth",
        ),
        _criterion(
            lane,
            "ric_map",
            "Every U2 security has a validated Refinitiv RIC.",
            "pass" if ric_coverage >= 0.999 and not selected_universe.empty else "fail",
            f"ric_coverage={ric_coverage:.3f}; selected={plan_manifest.get('selected_names')}; mapped={plan_manifest.get('names_with_refinitiv_ric')}",
            blocker="unmapped RICs prevent PIT and corporate-action extraction" if ric_coverage < 0.999 else "",
            next_action="repair symbol map and rerun PIT plan",
            severity="critical",
        ),
        _criterion(
            lane,
            "current_reference_fields",
            "Core current-reference identifiers are available for QA, but must not be mistaken for PIT history.",
            "pass" if security_history_ready else "partial" if has_core_snapshot_fields else "fail",
            f"accepted_security_master_roles={sorted(snapshot_fields)}; security_history={security_history_evidence}",
            blocker="current snapshot lacks core identifiers" if not has_core_snapshot_fields else "current snapshot only; not PIT identifier history",
            next_action="promote identifier fields into monthly/event PIT history",
            severity="high",
        ),
        _criterion(
            lane,
            "listing_delisting_history",
            "Listing, delisting, status, and ticker/RIC-change history are usable point-in-time.",
            "pass" if delisting_route and ric_change_route and listing_status_route else "fail",
            f"delisting_route={delisting_route}; ric_change_route={ric_change_route}; listing_status_route={listing_status_route}",
            blocker="survivorship and identifier-history bias not resolved",
            next_action="use Data Item Browser/export route for delistings, listing status, and RIC-change events",
            severity="critical",
        ),
        _criterion(
            lane,
            "historical_denominator_artifact",
            "Physical historical security-master artifacts exist for entities, identifiers, listing windows, and PIT universe membership.",
            "pass" if security_history_ready else "fail",
            security_history_evidence,
            blocker="historical security-master denominator artifact is missing or incomplete",
            next_action="build entities, identifier_history, listing_windows, and universe_membership_pit with country/year QA",
            severity="critical",
        ),
        _criterion(
            lane,
            "non_survivor_denominator",
            "Backtest denominator includes historical dead/delisted names or an explicit validated exclusion policy.",
            "pass" if security_history_ready else "fail",
            security_history_evidence,
            blocker="current-survivor universe cannot support production-quality alpha claims",
            next_action="build PIT security-master snapshots with delisted names and listing windows",
            severity="critical",
        ),
    ]
    return _make_lane(lane, "security_master_agent", criteria)


def _price_actions_liquidity_lane(context: Mapping[str, Any]) -> DataQualityLane:
    lane = "prices_actions_liquidity"
    l2_manifest = context["l2_manifest"]
    ticker_quality = context["l2_ticker_quality"]
    vendor_matrix = context["l2_vendor_matrix"]
    liquidity_manifest = context["liquidity_manifest"]
    liquidity_quality = context["liquidity_quality"]
    critical_quality = context["critical_fields_quality"]
    ca_event_manifest = context["corporate_actions_event_manifest"]
    ca_event_quality = context["corporate_actions_event_quality"]
    route_results = context["route_probe_results"]
    adjusted_equal_share = float(context.get("l2_adjusted_price_equal_share", 1.0))
    fx_ready, fx_evidence = _fx_country_risk_status(
        context["fx_country_risk_manifest"],
        context["fx_country_risk_quality"],
        context["fx_currency_reference"],
    )
    ca_recon_ready, ca_recon_diagnostic, ca_recon_evidence = _corporate_action_reconciliation_status(
        context["corporate_action_reconciliation_manifest"],
        context["corporate_action_reconciliation_quality"],
        context["adjusted_return_reconstruction_manifest"],
        context["adjusted_return_reconstruction_quality"],
    )

    date_min = pd.Timestamp(l2_manifest.get("date_min", pd.NaT))
    date_max = pd.Timestamp(l2_manifest.get("date_max", pd.NaT))
    years = (date_max - date_min).days / 365.25 if pd.notna(date_min) and pd.notna(date_max) else 0.0
    true_zero_pct = float(ticker_quality.get("zero_volume_pct", pd.Series([1.0])).fillna(1.0).mean()) if not ticker_quality.empty else 1.0
    bad_ohlc_rows = int(ticker_quality.get("bad_ohlc_rows", pd.Series([1])).fillna(1).sum()) if not ticker_quality.empty else 1
    currency_coverage = (
        float(ticker_quality["currency"].notna().mean())
        if not ticker_quality.empty and "currency" in ticker_quality.columns
        else 0.0
    )
    liquidity_ok = (
        int(liquidity_manifest.get("failed_rows", 1)) == 0
        and int(liquidity_manifest.get("history_rows", 0)) >= 100_000
        and _quality_min(liquidity_quality, "avg_coverage") >= 0.90
    )
    ca_status = _matrix_availability(vendor_matrix, "corporate_actions")
    ca_event_ready, ca_event_evidence = _corporate_actions_event_status(ca_event_manifest, ca_event_quality)
    official_ca = ("available" in ca_status and "proxy" not in ca_status) or ca_event_ready
    ca_snapshot_only = _has_validated_route(route_results, "corporate_actions") and not official_ca
    adv_direct = _quality_field_role_has_history(liquidity_quality, "adv_value") or _quality_field_role_has_history(
        critical_quality, "turnover_value"
    )
    adv_proxy = _quality_field_role_has_history(critical_quality, "volume_history")
    adv_historical = adv_direct or adv_proxy

    criteria = [
        _criterion(
            lane,
            "twenty_year_ohlcv",
            "Daily OHLCV spans roughly 20 years for the active three-country lake.",
            "pass" if years >= 19.5 and int(l2_manifest.get("raw_rows", 0)) >= 1_000_000 else "partial",
            f"date_min={l2_manifest.get('date_min')}; date_max={l2_manifest.get('date_max')}; years={years:.1f}; raw_rows={l2_manifest.get('raw_rows')}",
            next_action="backfill additional years or missing rows where the calendar is short",
        ),
        _criterion(
            lane,
            "ohlcv_validity",
            "OHLCV has no material bad bars and no true zero-volume pollution.",
            "pass" if bad_ohlc_rows <= 1 and true_zero_pct <= 0.001 else "partial",
            f"bad_ohlc_rows={bad_ohlc_rows}; avg_zero_volume_pct={true_zero_pct:.4f}",
            next_action="quarantine and vendor-check any bad OHLC or suspicious volume rows",
        ),
        _criterion(
            lane,
            "u2_coverage",
            "U2 price/volume coverage is broad enough for backtest-eligible universe construction.",
            "pass" if int(l2_manifest.get("u2_backtest_eligible", 0)) >= 150 else "fail",
            f"u2={l2_manifest.get('u2_backtest_eligible')}; u3={l2_manifest.get('u3_liquid_proxy')}",
            blocker="not enough U2 coverage for robust cross-sectional backtesting"
            if int(l2_manifest.get("u2_backtest_eligible", 0)) < 150
            else "",
            next_action="repair L2 OHLCV backfill and universe tiering",
            severity="critical",
        ),
        _criterion(
            lane,
            "historical_liquidity",
            "Market cap, free float, and shares history are present with high coverage.",
            "pass" if liquidity_ok else "fail",
            f"liquidity_history_rows={liquidity_manifest.get('history_rows')}; failed_rows={liquidity_manifest.get('failed_rows')}; min_avg_coverage={_quality_min(liquidity_quality, 'avg_coverage'):.3f}",
            blocker="capacity and liquidity filters lack robust historical inputs" if not liquidity_ok else "",
            next_action="rerun liquidity historical batch and inspect low-coverage country-field cells",
            severity="high",
        ),
        _criterion(
            lane,
            "turnover_adv",
            "ADV and tradability use lagged turnover/value traded, not current-only fields.",
            "pass" if adv_historical else "fail",
            f"adv_direct={adv_direct}; adv_proxy_from_volume_history={adv_proxy}; turnover_value_backfilled={_quality_field_role_has_history(critical_quality, 'turnover_value')}; volume_history_backfilled={_quality_field_role_has_history(critical_quality, 'volume_history')}; note=TR.AvgDailyValTraded30D was current/static in the first historical route",
            blocker="ADV route is current/proxy-only" if not adv_historical else "",
            next_action="validate historical turnover/value-traded field or lock derived lagged ADV policy",
            severity="high",
        ),
        _criterion(
            lane,
            "official_corporate_actions",
            "Official split/dividend/rights event history is integrated into return construction.",
            "pass" if official_ca else "partial" if ca_snapshot_only else "fail",
            f"vendor_matrix_corporate_actions={ca_status}; event_backfill={ca_event_evidence}; route_probe_snapshot_only={ca_snapshot_only}",
            blocker="official corporate actions are missing or snapshot-only",
            next_action="extract official corporate-action event history through a specialized Refinitiv route/export",
            severity="critical",
        ),
        _criterion(
            lane,
            "adjusted_raw_integrity",
            "Adjusted prices are distinct from raw prices where actions require adjustment, with explicit raw-only labeling otherwise.",
            "pass" if ca_recon_ready else "partial" if ca_recon_diagnostic and official_ca else "fail",
            f"adj_close_equals_close_share={adjusted_equal_share:.3f}; official_corporate_actions={official_ca}; reconciliation={ca_recon_evidence}",
            blocker="production adjusted-return reconstruction is not ready" if not ca_recon_ready else "",
            next_action="build daily_adjusted_returns from official events, ex-dates, FX, and no-double-adjustment QA",
            severity="critical",
        ),
        _criterion(
            lane,
            "currency_fx_metadata",
            "Currency metadata is populated so local and USD return paths are reproducible.",
            "pass" if fx_ready or currency_coverage >= 0.99 else "fail",
            f"ticker_currency_coverage={currency_coverage:.3f}; fx_artifact={fx_evidence}",
            blocker="currency/FX metadata is incomplete",
            next_action="fill price currency and create MYR/SGD/IDR local-plus-USD return paths",
            severity="high",
        ),
        _criterion(
            lane,
            "trading_calendar_halts_tca",
            "Exchange calendars, halt/suspension flags, and country-specific TCA assumptions are model-ready.",
            "pass" if fx_ready and ca_recon_ready and liquidity_ok else "partial",
            f"fx_ready={fx_ready}; adjusted_return_ready={ca_recon_ready}; liquidity_ok={liquidity_ok}; policy=observed trading calendars from daily bars plus conservative country cost grid and halt/suspension proxy flags",
            blocker="trading calendar and halt/TCA controls are not 5/5" if not (fx_ready and ca_recon_ready and liquidity_ok) else "",
            next_action="keep exchange calendar and TCA policy versioned with every simulation",
            severity="high",
        ),
    ]
    return _make_lane(lane, "prices_actions_liquidity_agent", criteria)


def _pit_fund_ibes_events_lane(context: Mapping[str, Any]) -> DataQualityLane:
    lane = "pit_fundamentals_ibes_events"
    hist_manifest = context["fund_ibes_events_manifest"]
    hist_quality = context["fund_ibes_events_quality"]
    critical_quality = context["critical_fields_quality"]
    leakage_manifest = context["pit_leakage_manifest"]
    feature_quality = context["pit_feature_quality"]
    route_history_map = context["route_history_map"]
    route_results = context["route_probe_results"]
    revision_time_artifact_ready, revision_time_evidence = _ibes_revision_timestamp_status(
        context["ibes_revision_manifest"],
        context["ibes_revision_accepted_map"],
    )

    min_feature_coverage = _quality_min(feature_quality, "coverage")
    history_ok = int(hist_manifest.get("failed_rows", 1)) == 0 and int(hist_manifest.get("history_rows", 0)) >= 30_000
    route_fields = set(route_history_map.get("field_role", pd.Series(dtype=str)).dropna().astype(str))
    required_promoted = {"revenue", "book_value", "release_date", "eps_estimate", "revenue_estimate", "earnings_release"}
    actual_eps_route = _has_validated_route(
        route_results, "earnings_events_actuals", field_roles={"actual_eps"}, historical_only=True
    )
    surprise_route = _has_validated_route(
        route_results, "earnings_events_actuals", field_roles={"surprise"}, historical_only=True
    )
    target_price_route = _has_validated_route(
        route_results, "ibes_estimates_revisions", field_roles={"target_price"}, historical_only=True
    )
    revision_time_route = _has_validated_route(
        route_results, "ibes_estimates_revisions", field_roles={"revision_time"}, historical_only=True
    ) or revision_time_artifact_ready
    actual_eps_backfilled = _quality_field_role_has_history(critical_quality, "actual_eps")
    surprise_backfilled = _quality_field_role_has_history(critical_quality, "surprise")
    target_price_backfilled = _quality_field_role_has_history(critical_quality, "target_price")
    event_last = _last_available_by_country(hist_quality, "earnings_events_actuals", "earnings_release")

    criteria = [
        _criterion(
            lane,
            "full_u2_backfill",
            "Full U2 historical fundamentals/IBES/events batch completed without failures.",
            "pass" if history_ok else "fail",
            f"history_rows={hist_manifest.get('history_rows')}; failed_rows={hist_manifest.get('failed_rows')}; completed_rows={hist_manifest.get('completed_rows')}",
            blocker="PIT fundamentals/IBES/events batch is incomplete" if not history_ok else "",
            next_action="rerun failed/empty historical plan rows",
            severity="critical",
        ),
        _criterion(
            lane,
            "promoted_field_roles",
            "Core field roles are promoted into historical maps.",
            "pass" if required_promoted.issubset(route_fields) else "partial",
            f"promoted_roles={sorted(route_fields)}",
            next_action="promote missing field roles through route probe and history map",
        ),
        _criterion(
            lane,
            "leakage_qa",
            "Lagged PIT panel passes formal leakage QA on the backtest decision calendar.",
            "pass" if leakage_manifest.get("status") == "pass" and int(leakage_manifest.get("total_leakage_violations", 1)) == 0 else "fail",
            f"qa_status={leakage_manifest.get('status')}; violations={leakage_manifest.get('total_leakage_violations')}",
            blocker="PIT feature leakage violations found" if leakage_manifest.get("status") != "pass" else "",
            next_action="repair lag/availability rules and rerun leakage QA",
            severity="critical",
        ),
        _criterion(
            lane,
            "feature_coverage",
            "Model-facing PIT feature coverage is above the minimum gated threshold.",
            "pass" if min_feature_coverage >= 0.65 else "partial",
            f"min_feature_coverage={min_feature_coverage:.3f}",
            next_action="expand field routes and fill missing countries/sectors if coverage falls below target",
        ),
        _criterion(
            lane,
            "availability_timestamps",
            "Release and estimate availability timestamps are source-backed rather than safe-lag-only.",
            "pass" if revision_time_artifact_ready and "release_date" in route_fields else "partial",
            f"revision_time_artifact={revision_time_evidence}; promoted_release_date={'release_date' in route_fields}; leakage_status={leakage_manifest.get('status')}; conservative_lags_retained_as_fallback=liquidity=1d,fundamentals=90d,IBES=1d,events=1d",
            blocker="true availability timestamps are not complete" if not revision_time_artifact_ready else "",
            next_action="keep report dates, filing dates, estimate dates, and release timestamps versioned after each refresh",
            severity="high",
        ),
        _criterion(
            lane,
            "actuals_surprises_targets",
            "Actual EPS, EPS surprise, target price, and timestamped revision route are available.",
            "pass"
            if actual_eps_route
            and surprise_route
            and target_price_route
            and revision_time_route
            and actual_eps_backfilled
            and surprise_backfilled
            and target_price_backfilled
            else "fail",
            f"actual_eps_route={actual_eps_route}; surprise_route={surprise_route}; target_price_route={target_price_route}; revision_time_route={revision_time_route}; revision_time_artifact={revision_time_evidence}; actual_eps_backfilled={actual_eps_backfilled}; surprise_backfilled={surprise_backfilled}; target_price_backfilled={target_price_backfilled}",
            blocker="key IBES/event fields remain unresolved",
            next_action="refine Data Item Browser/export route for IBES actuals, surprises, target prices, and estimate timestamps",
            severity="critical",
        ),
        _criterion(
            lane,
            "event_freshness_by_country",
            "Earnings-event history remains fresh across all three active countries.",
            "pass" if history_ok and set(event_last).issuperset(set(ACTIVE_COUNTRIES)) else "partial",
            f"earnings_event_last_available_by_country={event_last}",
            blocker="some event histories may end before latest decision dates" if not set(event_last).issuperset(set(ACTIVE_COUNTRIES)) else "",
            next_action="refresh earnings events and reconcile stale Singapore event window",
            severity="high",
        ),
    ]
    return _make_lane(lane, "pit_fundamentals_ibes_events_agent", criteria)


def _leakage_bias_repro_lane(context: Mapping[str, Any]) -> DataQualityLane:
    lane = "leakage_bias_reproducibility"
    leakage_manifest = context["pit_leakage_manifest"]
    leakage_violations = context["pit_leakage_violations"]
    pit_enhanced_manifest = context["pit_enhanced_manifest"]
    paths = context["paths"]
    security_history_ready, security_history_evidence = _security_master_history_status(paths)
    ca_recon_ready, ca_recon_diagnostic, ca_recon_evidence = _corporate_action_reconciliation_status(
        context["corporate_action_reconciliation_manifest"],
        context["corporate_action_reconciliation_quality"],
        context["adjusted_return_reconstruction_manifest"],
        context["adjusted_return_reconstruction_quality"],
    )

    has_repro_artifacts = (paths.pit_leakage_qa_dir / "manifest.json").exists() and (paths.pit_leakage_qa_dir / "summary.md").exists()
    holdout_frozen = "2024-01-02" in (paths.pit_enhanced_alpha_dir / "run_config.json").read_text(encoding="utf-8") if paths.pit_enhanced_alpha_dir and (paths.pit_enhanced_alpha_dir / "run_config.json").exists() else False
    overfit_diagnostic = pit_enhanced_manifest.get("audit_posture") == "diagnostic_only" or pit_enhanced_manifest.get("warnings")

    criteria = [
        _criterion(
            lane,
            "formal_leakage_report",
            "Formal PIT leakage report exists and has zero violations.",
            "pass" if leakage_manifest.get("status") == "pass" and leakage_violations.empty else "fail",
            f"status={leakage_manifest.get('status')}; violations={leakage_manifest.get('total_leakage_violations')}; violation_rows={len(leakage_violations)}",
            blocker="leakage QA is failing or missing" if leakage_manifest.get("status") != "pass" else "",
            next_action="rerun leakage QA after each data refresh",
            severity="critical",
        ),
        _criterion(
            lane,
            "decision_calendar",
            "Decision calendar is reconstructed from the actual backtest universe.",
            "pass" if int(leakage_manifest.get("decision_entities", 0)) >= 150 else "fail",
            f"decision_entities={leakage_manifest.get('decision_entities')}; decision_rows={leakage_manifest.get('decision_rows')}",
            blocker="decision calendar is not aligned to U2 backtest universe",
            next_action="rebuild calendar from L2 artifact before feature admission",
            severity="critical",
        ),
        _criterion(
            lane,
            "frozen_holdout",
            "Frozen historical holdout is preserved for final candidate validation.",
            "pass" if holdout_frozen else "partial",
            f"holdout_start={'2024-01-02' if holdout_frozen else 'not_detected_in_run_config'}",
            next_action="pin holdout dates in every alpha loop config",
        ),
        _criterion(
            lane,
            "overfit_posture",
            "Overfit audit prevents promotion when trial dependence or deflated Sharpe fails.",
            "pass" if overfit_diagnostic else "partial",
            f"overfit_audit_posture={pit_enhanced_manifest.get('audit_posture', 'missing')}",
            next_action="rerun overfit audit on every candidate ledger",
        ),
        _criterion(
            lane,
            "survivorship_bias",
            "Survivorship bias is neutralized by PIT security master and delisting coverage.",
            "pass" if security_history_ready else "fail",
            security_history_evidence,
            blocker="survivorship bias remains unresolved" if not security_history_ready else "",
            next_action="block production-quality claims until dead names and listing windows are in the denominator",
            severity="critical",
        ),
        _criterion(
            lane,
            "return_integrity_bias",
            "Raw/adjusted return integrity is tied to official corporate actions.",
            "pass" if ca_recon_ready else "partial" if ca_recon_diagnostic else "fail",
            ca_recon_evidence,
            blocker="production adjusted returns are required before production adjusted-return claims" if not ca_recon_ready else "",
            next_action="rebuild total-return series from official split/dividend events and rerun leakage QA",
            severity="critical",
        ),
        _criterion(
            lane,
            "reproducible_artifacts",
            "Scorecards, manifests, and source paths are stored for repeatable QA.",
            "pass" if has_repro_artifacts else "partial",
            f"leakage_manifest_exists={has_repro_artifacts}",
            next_action="keep every loop run manifest-backed and append to RUN_STATE",
        ),
    ]
    return _make_lane(lane, "model_audit_bias_control_agent", criteria)


def _extraction_operations_lane(context: Mapping[str, Any]) -> DataQualityLane:
    lane = "extraction_operations"
    plan_manifest = context["pit_plan_manifest"]
    field_accepted_map = context["field_accepted_map"]
    route_history_map = context["route_history_map"]
    route_results = context["route_probe_results"]
    current_snapshot_manifest = context["current_snapshot_manifest"]
    liquidity_manifest = context["liquidity_manifest"]
    fund_manifest = context["fund_ibes_events_manifest"]
    ca_event_ready, _ca_event_evidence = _corporate_actions_event_status(
        context["corporate_actions_event_manifest"],
        context["corporate_actions_event_quality"],
    )

    plan_ready = int(plan_manifest.get("dataset_count", 0)) >= 8 and int(plan_manifest.get("extraction_batches", 0)) >= 1_000
    batch_safe = all(
        int(manifest.get("failed_rows", 1)) == 0
        for manifest in (liquidity_manifest, fund_manifest)
        if manifest
    )
    local_workspace = current_snapshot_manifest.get("session_name") == "desktop.workspace"
    accepted_rows = len(field_accepted_map)
    history_rows = len(route_history_map)
    security_master_ready, _security_master_evidence = _security_master_history_status(context["paths"])
    ibes_revision_ready, _ibes_revision_evidence = _ibes_revision_timestamp_status(
        context["ibes_revision_manifest"],
        context["ibes_revision_accepted_map"],
    )
    unresolved_critical_routes = _unresolved_critical_route_ids(
        route_results,
        corporate_actions_event_ready=ca_event_ready,
        security_master_history_ready=security_master_ready,
        ibes_revision_timestamp_ready=ibes_revision_ready,
    )
    unresolved_critical = len(unresolved_critical_routes)
    unresolved_text = ", ".join(unresolved_critical_routes) if unresolved_critical_routes else "none"

    criteria = [
        _criterion(
            lane,
            "dataset_plan",
            "Eight-dataset PIT acquisition plan exists with batch queue.",
            "pass" if plan_ready else "fail",
            f"dataset_count={plan_manifest.get('dataset_count')}; extraction_batches={plan_manifest.get('extraction_batches')}",
            blocker="no complete PIT dataset plan" if not plan_ready else "",
            next_action="regenerate PIT data plan",
            severity="critical",
        ),
        _criterion(
            lane,
            "memory_safe_batches",
            "Extraction jobs are chunked and parquet-backed to avoid local memory blow-ups.",
            "pass" if batch_safe else "partial",
            f"liquidity_failed={liquidity_manifest.get('failed_rows')}; fund_ibes_events_failed={fund_manifest.get('failed_rows')}",
            next_action="rerun only failed chunks and keep year/RIC batches bounded",
        ),
        _criterion(
            lane,
            "local_workspace_route",
            "Local LSEG Workspace route is active for read-only extraction.",
            "pass" if local_workspace else "partial",
            f"current_snapshot_session={current_snapshot_manifest.get('session_name')}; snapshot_rows={current_snapshot_manifest.get('snapshot_rows')}",
            next_action="reauthenticate LSEG Workspace if session falls away",
        ),
        _criterion(
            lane,
            "accepted_field_maps",
            "Accepted snapshot and history maps exist for promoted fields.",
            "pass" if accepted_rows >= 8 and history_rows >= 7 else "partial",
            f"accepted_snapshot_fields={accepted_rows}; history_fields={history_rows}",
            next_action="rerun field probes after Data Item Browser refinement",
        ),
        _criterion(
            lane,
            "missing_route_backlog",
            "Unresolved production fields are explicitly visible to subagents.",
            "partial" if unresolved_critical > 0 else "pass",
            f"unresolved_critical_routes={unresolved_text}",
            next_action=f"keep blocker queue explicit and owner-assigned: {unresolved_text}",
        ),
        _criterion(
            lane,
            "critical_route_closure",
            "All critical vendor routes are validated, not merely documented.",
            "fail" if unresolved_critical > 0 else "pass",
            f"unresolved_critical_routes={unresolved_text}",
            blocker="critical Refinitiv routes still unresolved",
            next_action=f"close routes for {unresolved_text}",
            severity="critical",
        ),
        _criterion(
            lane,
            "continuous_loop_artifact",
            "The Codex data-quality loop writes a machine-readable queue for the next agent pass.",
            "pass",
            "this run writes lane_scorecard.csv, criteria_scorecard.csv, agent_next_queue.csv, summary.md, and manifest.json",
            next_action="schedule or rerun this loop after every extraction batch",
        ),
    ]
    return _make_lane(lane, "data_extraction_orchestrator_agent", criteria)


def _has_validated_route(
    route_results: pd.DataFrame,
    dataset_id: str,
    field_roles: set[str] | None = None,
    historical_only: bool = False,
) -> bool:
    if route_results.empty or "dataset_id" not in route_results.columns or "status" not in route_results.columns:
        return False
    data = route_results[route_results["dataset_id"].astype(str).eq(dataset_id)].copy()
    if field_roles and "field_role" in data.columns:
        data = data[data["field_role"].astype(str).isin(field_roles)]
    if historical_only and "route" in data.columns:
        data = data[data["route"].astype(str).eq("get_history")]
    if data.empty:
        return False
    statuses = data["status"].astype(str).str.lower()
    valid_statuses = {"historical_validated", "accepted_from_route_probe"} if historical_only else {
        "historical_validated",
        "snapshot_validated",
        "accepted_from_route_probe",
        "validated",
    }
    return bool(statuses.isin(valid_statuses).any())


def _has_validated_security_master_discovery_route(route_results: pd.DataFrame, field_roles: set[str]) -> bool:
    if route_results.empty or "field_role" not in route_results.columns or "status" not in route_results.columns:
        return False
    data = route_results[route_results["field_role"].astype(str).isin(field_roles)].copy()
    if "dataset_id" in data.columns:
        data = data[
            data["dataset_id"].astype(str).isin({"delistings_listing_history", "security_master_pit"})
        ]
    if "route" in data.columns:
        data = data[
            data["route"].astype(str).isin(
                {
                    "discovery_search",
                    "discovery_search_security_master",
                    "refinitiv_discovery_search",
                    "symbol_conversion",
                }
            )
        ]
    if data.empty:
        return False
    valid_statuses = {
        "historical_validated",
        "accepted_from_route_probe",
        "validated",
        "discovery_validated",
        "complete",
    }
    return bool(data["status"].astype(str).str.lower().isin(valid_statuses).any())


def _field_role_present(frame: pd.DataFrame, field_role: str) -> bool:
    return not frame.empty and "field_role" in frame.columns and frame["field_role"].astype(str).eq(field_role).any()


def _security_master_history_status(paths: InstitutionalDataQualityPaths) -> tuple[bool, str]:
    root = paths.security_master_history_dir
    if root is None:
        return False, "security_master_history_dir=not_configured"
    required = {
        "entities": root / "entities.parquet",
        "identifier_history": root / "identifier_history.parquet",
        "listing_windows": root / "listing_windows.parquet",
        "universe_membership_pit": root / "universe_membership_pit.parquet",
    }
    missing = [name for name, path in required.items() if not path.exists()]
    manifest = _read_json(root / "manifest.json")
    quality = _read_csv(root / "quality/security_master_pit_quality.csv")
    countries = set(quality.get("country", pd.Series(dtype=str)).dropna().astype(str)) if not quality.empty else set()
    has_country_coverage = set(ACTIVE_COUNTRIES).issubset(countries)
    dead_name_rows = _manifest_count(manifest, ("dead_name_rows", "delisted_rows", "historical_delisted_names"))
    if dead_name_rows <= 0 and not quality.empty:
        for column in ("delisted_count", "dead_name_count"):
            if column in quality.columns:
                dead_name_rows = int(pd.to_numeric(quality[column], errors="coerce").fillna(0).sum())
                break
    min_year = None
    if "asof_year" in quality.columns:
        years = pd.to_numeric(quality["asof_year"], errors="coerce").dropna()
        if not years.empty:
            min_year = int(years.min())
    if min_year is None:
        date_min = pd.to_datetime(manifest.get("date_min", pd.NaT), errors="coerce")
        min_year = int(date_min.year) if pd.notna(date_min) else None
    has_pre_2026_history = bool(min_year is not None and min_year < 2026)
    manifest_status = str(manifest.get("status", "")).lower()
    status_ok = manifest_status in {"pass", "ready", "accepted", "complete", "completed"}
    ready = not missing and has_country_coverage and dead_name_rows > 0 and has_pre_2026_history and status_ok
    evidence = (
        f"root={root}; missing={missing}; manifest_status={manifest_status or 'missing'}; "
        f"countries={sorted(countries)}; dead_name_rows={dead_name_rows}; min_year={min_year}"
    )
    return ready, evidence


def _corporate_actions_event_status(manifest: Mapping[str, Any], quality: pd.DataFrame) -> tuple[bool, str]:
    if not manifest or quality.empty:
        return False, "event_backfill=missing"
    status = str(manifest.get("status", "")).lower()
    failed_rows = int(manifest.get("failed_rows", 1) or 0)
    event_rows = int(manifest.get("event_rows", 0) or 0)
    required_roles = {"dividend_event", "split_event"}
    roles = set(quality.get("field_role", pd.Series(dtype=str)).dropna().astype(str))
    countries = set(quality.get("country", pd.Series(dtype=str)).dropna().astype(str))
    no_failed_quality = (
        "failed_rows" in quality.columns
        and int(pd.to_numeric(quality["failed_rows"], errors="coerce").fillna(0).sum()) == 0
    )
    valid_events = (
        "valid_event_date_rows" in quality.columns
        and int(pd.to_numeric(quality["valid_event_date_rows"], errors="coerce").fillna(0).sum()) > 0
    )
    ready = (
        status == "complete"
        and failed_rows == 0
        and event_rows > 0
        and required_roles.issubset(roles)
        and set(ACTIVE_COUNTRIES).issubset(countries)
        and no_failed_quality
        and valid_events
    )
    evidence = (
        f"status={status or 'missing'}; failed_rows={failed_rows}; event_rows={event_rows}; "
        f"roles={sorted(roles)}; countries={sorted(countries)}; valid_events={valid_events}"
    )
    return ready, evidence


def _fx_country_risk_status(
    manifest: Mapping[str, Any],
    fx_quality: pd.DataFrame,
    currency_reference: pd.DataFrame,
) -> tuple[bool, str]:
    if not manifest:
        return False, "fx_country_risk=missing"
    status = str(manifest.get("status", "")).lower()
    currency_coverage = _safe_float(manifest.get("currency_coverage"))
    fx_calendar_coverage = _safe_float(manifest.get("fx_calendar_coverage"))
    usd_return_coverage = _safe_float(manifest.get("usd_return_coverage"))
    countries = set(currency_reference.get("country", pd.Series(dtype=str)).dropna().astype(str))
    fx_tickers = set(fx_quality.get("fx_ticker", pd.Series(dtype=str)).dropna().astype(str))
    ready = (
        status == "pass_currency_fx_returns_partial_tca"
        and currency_coverage >= 0.999
        and fx_calendar_coverage >= 0.90
        and usd_return_coverage >= 0.90
        and set(ACTIVE_COUNTRIES).issubset(countries)
    )
    evidence = (
        f"status={status or 'missing'}; currency_coverage={currency_coverage:.3f}; "
        f"fx_calendar_coverage={fx_calendar_coverage:.3f}; usd_return_coverage={usd_return_coverage:.3f}; "
        f"countries={sorted(countries)}; fx_tickers={sorted(fx_tickers)}"
    )
    return ready, evidence


def _corporate_action_reconciliation_status(
    manifest: Mapping[str, Any],
    quality: pd.DataFrame,
    adjusted_manifest: Mapping[str, Any] | None = None,
    adjusted_quality: pd.DataFrame | None = None,
) -> tuple[bool, bool, str]:
    adjusted_manifest = adjusted_manifest or {}
    adjusted_quality = adjusted_quality if adjusted_quality is not None else pd.DataFrame()
    if adjusted_manifest:
        adjusted_status = str(adjusted_manifest.get("status", "")).lower()
        production_ready = bool(adjusted_manifest.get("production_adjusted_return_ready"))
        valid_event_rows = _safe_int(adjusted_manifest.get("valid_event_rows"))
        matched_share = _safe_float(adjusted_manifest.get("matched_event_share"))
        applied_rows = _safe_int(adjusted_manifest.get("applied_adjustment_rows"))
        quarantine_share = _safe_float(adjusted_manifest.get("quarantine_share"))
        max_quarantine_share = _safe_float(adjusted_manifest.get("max_quarantine_share"))
        countries = (
            set(adjusted_quality.get("country", pd.Series(dtype=str)).dropna().astype(str))
            if not adjusted_quality.empty
            else set()
        )
        families = (
            set(adjusted_quality.get("event_family", pd.Series(dtype=str)).dropna().astype(str))
            if not adjusted_quality.empty
            else set()
        )
        ready = (
            adjusted_status == "pass"
            and production_ready
            and valid_event_rows > 0
            and matched_share >= 0.99
            and applied_rows > 0
            and quarantine_share <= max_quarantine_share
            and set(ACTIVE_COUNTRIES).issubset(countries)
        )
        diagnostic_ready = (
            valid_event_rows > 0
            and matched_share >= 0.95
            and applied_rows > 0
            and set(ACTIVE_COUNTRIES).issubset(countries)
        )
        evidence = (
            f"adjusted_return_reconstruction_status={adjusted_status or 'missing'}; "
            f"production_adjusted_return_ready={production_ready}; valid_event_rows={valid_event_rows}; "
            f"matched_event_share={matched_share:.3f}; applied_adjustment_rows={applied_rows}; "
            f"quarantine_share={quarantine_share:.4f}; max_quarantine_share={max_quarantine_share:.4f}; "
            f"countries={sorted(countries)}; event_families={sorted(families)}"
        )
        return ready, diagnostic_ready, evidence
    if not manifest:
        return False, False, "corporate_action_reconciliation=missing"
    status = str(manifest.get("status", "")).lower()
    valid_event_rows = _safe_int(manifest.get("valid_event_rows"))
    matched_share = _safe_float(manifest.get("matched_event_share_min") or manifest.get("matched_event_share"))
    production_ready = bool(manifest.get("production_adjusted_return_ready"))
    countries = set(quality.get("country", pd.Series(dtype=str)).dropna().astype(str)) if not quality.empty else set()
    families = set(quality.get("event_family", pd.Series(dtype=str)).dropna().astype(str)) if not quality.empty else set()
    diagnostic_ready = (
        status == "diagnostic_reconciliation_only"
        and valid_event_rows > 0
        and matched_share >= 0.95
        and set(ACTIVE_COUNTRIES).issubset(countries)
    )
    ready = (
        status == "pass"
        and production_ready
        and valid_event_rows > 0
        and matched_share >= 0.95
        and set(ACTIVE_COUNTRIES).issubset(countries)
    )
    evidence = (
        f"status={status or 'missing'}; production_adjusted_return_ready={production_ready}; "
        f"valid_event_rows={valid_event_rows}; matched_event_share_min={matched_share:.3f}; "
        f"countries={sorted(countries)}; event_families={sorted(families)}"
    )
    return ready, diagnostic_ready, evidence


def _ibes_revision_timestamp_status(
    manifest: Mapping[str, Any],
    accepted_map: pd.DataFrame,
) -> tuple[bool, str]:
    if not manifest or accepted_map.empty:
        return False, "ibes_revision_route_probe=missing"
    status = str(manifest.get("status", "")).lower()
    accepted_countries = set(str(country) for country in manifest.get("accepted_countries", []))
    if not accepted_countries and "country" in accepted_map.columns:
        accepted_countries = set(accepted_map["country"].dropna().astype(str))
    missing_countries = [str(country) for country in manifest.get("missing_countries", [])]
    non_null_min = _quality_min(accepted_map, "non_null_date_rows")
    unique_min = _quality_min(accepted_map, "unique_date_count")
    fields = sorted(set(accepted_map.get("field", pd.Series(dtype=str)).dropna().astype(str)))
    ready = (
        status == "complete"
        and {"MYS", "SGP", "IDN"}.issubset(accepted_countries)
        and not missing_countries
        and non_null_min > 0
        and unique_min >= 2
    )
    evidence = (
        f"status={status or 'missing'}; accepted_countries={sorted(accepted_countries)}; "
        f"missing_countries={missing_countries}; min_non_null_date_rows={non_null_min:.0f}; "
        f"min_unique_date_count={unique_min:.0f}; fields={fields}"
    )
    return ready, evidence


def _manifest_count(manifest: Mapping[str, Any], keys: Sequence[str]) -> int:
    for key in keys:
        try:
            value = int(manifest.get(key, 0))
        except Exception:
            value = 0
        if value > 0:
            return value
    return 0


def _quality_field_role_has_history(
    frame: pd.DataFrame,
    field_role: str,
    countries: Sequence[str] = ACTIVE_COUNTRIES,
) -> bool:
    if frame.empty or "field_role" not in frame.columns:
        return False
    data = frame[frame["field_role"].astype(str).eq(field_role)].copy()
    if data.empty:
        return False
    if "country" in data.columns:
        present = set(data["country"].dropna().astype(str))
        if not set(countries).issubset(present):
            return False
    if "history_rows" in data.columns and pd.to_numeric(data["history_rows"], errors="coerce").fillna(0).sum() <= 0:
        return False
    if "failed_rows" in data.columns and pd.to_numeric(data["failed_rows"], errors="coerce").fillna(0).sum() > 0:
        return False
    if "completed_rows" in data.columns and pd.to_numeric(data["completed_rows"], errors="coerce").fillna(0).sum() <= 0:
        return False
    return True


def _quality_min(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame.columns:
        return 0.0
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(values.min()) if not values.empty else 0.0


def _safe_ratio(numerator: Any, denominator: Any) -> float:
    try:
        den = float(denominator)
        if den == 0.0:
            return 0.0
        return float(numerator) / den
    except Exception:
        return 0.0


def _safe_float(value: Any) -> float:
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        if pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def _country_min(frame: pd.DataFrame, column: str, minimum: int) -> bool:
    if frame.empty or column not in frame.columns:
        return False
    return bool(pd.to_numeric(frame[column], errors="coerce").fillna(0).ge(minimum).all())


def _country_count_dict(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns or "country" not in frame.columns:
        return {}
    return {
        str(row["country"]): int(row[column])
        for _, row in frame[["country", column]].fillna(0).iterrows()
    }


def _matrix_availability(vendor_matrix: pd.DataFrame, layer: str) -> str:
    if vendor_matrix.empty or "layer" not in vendor_matrix.columns:
        return "missing_matrix"
    row = vendor_matrix[vendor_matrix["layer"].astype(str).eq(layer)]
    if row.empty:
        return "missing_layer"
    col = "current_availability" if "current_availability" in row.columns else "status"
    return str(row.iloc[0].get(col, "unknown"))


def _last_available_by_country(frame: pd.DataFrame, dataset_id: str, field_role: str) -> dict[str, str]:
    if frame.empty:
        return {}
    data = frame.copy()
    for col in ("dataset_id", "field_role", "country", "last_available"):
        if col not in data.columns:
            return {}
    data = data[data["dataset_id"].astype(str).eq(dataset_id) & data["field_role"].astype(str).eq(field_role)]
    return {str(row["country"]): str(row["last_available"]) for _, row in data.iterrows()}


def _unresolved_critical_route_count(route_results: pd.DataFrame) -> int:
    return len(_unresolved_critical_route_ids(route_results))


def _unresolved_critical_route_ids(
    route_results: pd.DataFrame,
    corporate_actions_event_ready: bool = False,
    security_master_history_ready: bool = False,
    ibes_revision_timestamp_ready: bool = False,
) -> list[str]:
    if route_results.empty:
        return ["route_probe_results_missing"]
    critical_pairs = (
        ("corporate_actions", "dividend_event"),
        ("corporate_actions", "split_event"),
        ("delistings_listing_history", "delisting_date"),
        ("delistings_listing_history", "ric_change"),
        ("earnings_events_actuals", "actual_eps"),
        ("earnings_events_actuals", "surprise"),
        ("ibes_estimates_revisions", "target_price"),
        ("ibes_estimates_revisions", "revision_time"),
    )
    unresolved = []
    for dataset_id, field_role in critical_pairs:
        if corporate_actions_event_ready and dataset_id == "corporate_actions":
            continue
        if security_master_history_ready and dataset_id == "delistings_listing_history":
            continue
        if ibes_revision_timestamp_ready and dataset_id == "ibes_estimates_revisions" and field_role == "revision_time":
            continue
        if not _has_validated_route(route_results, dataset_id, field_roles={field_role}, historical_only=True):
            unresolved.append(f"{dataset_id}.{field_role}")
    return unresolved


def _lane_scorecard(lanes: Iterable[DataQualityLane]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lane_id": lane.lane_id,
                "lead_agent": lane.lead_agent,
                "score": lane.score,
                "max_score": lane.max_score,
                "status": lane.status,
                "critical_blockers": " | ".join(lane.critical_blockers),
            }
            for lane in lanes
        ]
    )


def _criteria_scorecard(lanes: Iterable[DataQualityLane]) -> pd.DataFrame:
    return pd.DataFrame([asdict(item) for lane in lanes for item in lane.criteria])


def _next_queue(lanes: Iterable[DataQualityLane]) -> pd.DataFrame:
    rows = []
    priority_rank = {"critical": 1, "high": 2, "medium": 3, "low": 4}
    for lane in lanes:
        for item in lane.criteria:
            if item.status == "pass":
                continue
            rows.append(
                {
                    "priority": priority_rank.get(item.severity, 3),
                    "severity": item.severity,
                    "lane_id": lane.lane_id,
                    "owner_agent": lane.lead_agent,
                    "criterion_id": item.criterion_id,
                    "status": item.status,
                    "blocker": item.blocker,
                    "next_action": item.next_action,
                    "evidence": item.evidence,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                "priority",
                "severity",
                "lane_id",
                "owner_agent",
                "criterion_id",
                "status",
                "blocker",
                "next_action",
                "evidence",
            ]
        )
    return pd.DataFrame(rows).sort_values(["priority", "lane_id", "criterion_id"]).reset_index(drop=True)


def _summary_md(
    run_id: str,
    lanes: Sequence[DataQualityLane],
    overall_score: float,
    accepted: bool,
    status: str,
    paths: InstitutionalDataQualityPaths,
    next_queue: pd.DataFrame,
) -> str:
    lane_table = _lane_scorecard(lanes)
    top_queue = next_queue.head(15)
    blockers = [blocker for lane in lanes for blocker in lane.critical_blockers]
    lines = [
        f"# Institutional Data Quality Loop - {run_id}",
        "",
        "This is the hard gate before deeper backtesting. It is not a strategy result and it is not a production approval.",
        "",
        "## Verdict",
        "",
        f"- Status: `{status}`",
        f"- Accepted 5/5: `{accepted}`",
        f"- Overall score: `{overall_score:.3f} / 5.000`",
        f"- Critical blockers: `{len(blockers)}`",
        "",
        "## Lane Scorecard",
        "",
        lane_table.to_markdown(index=False),
        "",
        "## Critical Blockers",
        "",
    ]
    if blockers:
        lines.extend(f"- {blocker}" for blocker in blockers)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Next Agent Queue",
            "",
            top_queue.to_markdown(index=False) if not top_queue.empty else "No queued actions.",
            "",
            "## Source Artifacts",
            "",
        ]
    )
    for name, path in _paths_manifest(paths).items():
        lines.append(f"- {name}: `{path}`")
    lines.extend(
        [
            "",
            "## Operating Rule",
            "",
            "The loop continues until every lane reaches 5/5 and no critical blocker remains. Guarded research may continue below 5/5, but production-quality backtest claims and trading integration stay blocked.",
        ]
    )
    return "\n".join(lines) + "\n"


def _paths_manifest(paths: InstitutionalDataQualityPaths) -> dict[str, str]:
    return {key: str(value.resolve()) for key, value in asdict(paths).items() if value is not None}
