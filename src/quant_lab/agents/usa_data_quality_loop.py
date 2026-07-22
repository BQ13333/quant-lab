from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class UsaDataQualityPaths:
    research_universe_dir: Path
    security_master_dir: Path
    active_symbol_map_dir: Path
    reference_symbol_map_dir: Path
    active_historical_dir: Path
    active_depth_dir: Path
    active_compaction_dir: Path
    reference_historical_dir: Path
    reference_depth_dir: Path | None = None
    reference_compaction_dir: Path | None = None
    pit_index_constituents_dir: Path | None = None
    corporate_actions_dir: Path | None = None
    adjusted_returns_dir: Path | None = None
    liquidity_backfill_dir: Path | None = None
    fundamentals_ibes_events_dir: Path | None = None
    fx_calendar_tca_dir: Path | None = None
    leakage_qa_dir: Path | None = None
    commands_log: Path | None = None


@dataclass(frozen=True)
class UsaQualityLane:
    lane_id: str
    lead_agent: str
    score: float
    max_score: float
    status: str
    critical_blockers: tuple[str, ...]
    evidence: str
    next_action: str


@dataclass(frozen=True)
class UsaDataQualityResult:
    run_id: str
    artifact_dir: Path
    overall_score: float
    accepted: bool
    status: str
    lanes: tuple[UsaQualityLane, ...]
    manifest_path: Path
    summary_path: Path
    lane_scorecard_path: Path
    next_queue_path: Path


def default_usa_data_quality_paths(project_root: str | Path = ".") -> UsaDataQualityPaths:
    root = Path(project_root)
    return UsaDataQualityPaths(
        research_universe_dir=root / "artifacts/research_universes/USA-LARGE-LIQUID-UNIVERSE-V1-20260709",
        security_master_dir=root
        / "artifacts/data_lake/pit/security_master_history/SECMASTER-DISCOVERY-USA-MERGED-RESIDUAL-20260709",
        active_symbol_map_dir=root
        / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-USA_LARGE_LIQUID_CORE_V1-USA-LARGE-LIQUID-UNIVERSE-V1-20260709",
        reference_symbol_map_dir=root
        / "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-USA_LISTED_REFERENCE_CORE_V1-USA-RESEARCH-UNIVERSE-V1-20260709",
        active_historical_dir=root
        / "artifacts/historical_backfill/REFINITIV-USA-HISTORICAL-20Y-LARGE-LIQUID-CORE-V1-20260709",
        active_depth_dir=root
        / "artifacts/data_depth/REFINITIV-USA-HISTORICAL-20Y-LARGE-LIQUID-CORE-V1-20260709-DEPTH",
        active_compaction_dir=root
        / "artifacts/data_lake_ops/REFINITIV-USA-HISTORICAL-20Y-LARGE-LIQUID-CORE-V1-20260709-COMPACT",
        reference_historical_dir=root
        / "artifacts/historical_backfill/REFINITIV-USA-HISTORICAL-20Y-LISTED-REFERENCE-CORE-V1-20260709",
        reference_depth_dir=root
        / "artifacts/data_depth/REFINITIV-USA-HISTORICAL-20Y-LISTED-REFERENCE-CORE-V1-20260709-DEPTH",
        reference_compaction_dir=root
        / "artifacts/data_lake_ops/REFINITIV-USA-HISTORICAL-20Y-LISTED-REFERENCE-CORE-V1-20260709-COMPACT",
        pit_index_constituents_dir=root
        / "artifacts/index_constituents/USA-PIT-INDEX-CONSTITUENTS-NO-RUSSELL-V1-20260711",
        corporate_actions_dir=root / "artifacts/refinitiv_corporate_actions_events/REFINITIV-CORPACT-EVENTS-USA-20260709",
        adjusted_returns_dir=root / "artifacts/adjusted_return_reconstruction/ADJRET-RECON-USA-20260709",
        liquidity_backfill_dir=root
        / "artifacts/refinitiv_pit_historical_backfill/REFINITIV-PIT-HISTORICAL-LIQUIDITY-USA-U2-FULL-20260711",
        fundamentals_ibes_events_dir=root
        / "artifacts/refinitiv_pit_historical_backfill/REFINITIV-PIT-HISTORICAL-FUND-IBES-EVENTS-USA-20260709",
        fx_calendar_tca_dir=root / "artifacts/fx_calendar_tca/FX-CALENDAR-TCA-USA-20260709",
        leakage_qa_dir=root / "artifacts/pit_leakage_qa/PIT-LEAKAGE-QA-USA-20260709",
        commands_log=root / "commands.log",
    )


def run_usa_data_quality_loop(
    paths: UsaDataQualityPaths | None = None,
    artifact_root: str | Path = "artifacts/institutional_data_quality",
    run_id: str | None = None,
    project_root: str | Path = ".",
    acceptance_score: float = 5.0,
) -> UsaDataQualityResult:
    paths = paths or default_usa_data_quality_paths(project_root)
    run_id = run_id or datetime.now(timezone.utc).strftime("USA-DATA-QUALITY-LOOP-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    context = _load_context(paths)
    lanes = (
        _g1_security_master_universe(context),
        _g2_prices_actions_liquidity(context),
        _g3_pit_index_fundamentals(context),
        _g4_fx_calendar_tca(context),
        _g5_leakage_ops(context),
    )
    overall_score = round(sum(lane.score for lane in lanes) / len(lanes), 3)
    critical_blockers = [blocker for lane in lanes for blocker in lane.critical_blockers]
    accepted = overall_score >= acceptance_score and all(lane.score >= acceptance_score for lane in lanes) and not critical_blockers
    status = "accepted_backtest_5_of_5" if accepted else "continue_usa_data_quality_loop"

    lane_scorecard = pd.DataFrame(
        [
            {
                "lane_id": lane.lane_id,
                "lead_agent": lane.lead_agent,
                "score": lane.score,
                "max_score": lane.max_score,
                "status": lane.status,
                "critical_blockers": " | ".join(lane.critical_blockers),
                "evidence": lane.evidence,
                "next_action": lane.next_action,
            }
            for lane in lanes
        ]
    )
    next_queue = _next_queue(lanes)
    lane_scorecard_path = artifact_dir / "lane_scorecard.csv"
    next_queue_path = artifact_dir / "agent_next_queue.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"
    active_zero_output_path = artifact_dir / "active_zero_output_rows.csv"
    reference_zero_output_path = artifact_dir / "reference_zero_output_rows.csv"
    lane_scorecard.to_csv(lane_scorecard_path, index=False)
    next_queue.to_csv(next_queue_path, index=False)
    _zero_output_completed_frame(context["active_manifest"]).to_csv(
        active_zero_output_path, index=False
    )
    _zero_output_completed_frame(context["reference_manifest"]).to_csv(
        reference_zero_output_path, index=False
    )
    summary_path.write_text(_summary_md(run_id, lanes, overall_score, accepted, status, next_queue), encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "overall_score": overall_score,
        "max_score": 5.0,
        "accepted": accepted,
        "readiness_scope": "research_backtest",
        "backtest_ready": accepted,
        "production_ready": False,
        "live_execution_ready": False,
        "status": status,
        "critical_blockers": critical_blockers,
        "lane_count": len(lanes),
        "next_queue_rows": int(len(next_queue)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "paths": _paths_manifest(paths),
        "artifacts": {
            "summary": str(summary_path.resolve()),
            "lane_scorecard": str(lane_scorecard_path.resolve()),
            "agent_next_queue": str(next_queue_path.resolve()),
            "active_zero_output_rows": str(active_zero_output_path.resolve()),
            "reference_zero_output_rows": str(reference_zero_output_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return UsaDataQualityResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        overall_score=overall_score,
        accepted=accepted,
        status=status,
        lanes=lanes,
        manifest_path=manifest_path,
        summary_path=summary_path,
        lane_scorecard_path=lane_scorecard_path,
        next_queue_path=next_queue_path,
    )


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


def _load_context(paths: UsaDataQualityPaths) -> dict[str, Any]:
    return {
        "paths": paths,
        "research_manifest": _read_json(paths.research_universe_dir / "manifest.json"),
        "active_symbol_map": _read_csv(paths.active_symbol_map_dir / "symbol_map.csv"),
        "reference_symbol_map": _read_csv(paths.reference_symbol_map_dir / "symbol_map.csv"),
        "active_run_manifest": _read_json(paths.active_historical_dir / "historical_run_manifest.json"),
        "active_plan": _read_csv(paths.active_historical_dir / "historical_plan.csv"),
        "active_manifest": _read_csv(paths.active_historical_dir / "historical_manifest.csv"),
        "active_depth": _read_csv(paths.active_depth_dir / "overall.csv"),
        "active_compaction": _read_json(paths.active_compaction_dir / "manifest.json"),
        "reference_run_manifest": _read_json(paths.reference_historical_dir / "historical_run_manifest.json"),
        "reference_plan": _read_csv(paths.reference_historical_dir / "historical_plan.csv"),
        "reference_manifest": _read_csv(paths.reference_historical_dir / "historical_manifest.csv"),
        "reference_depth": _read_csv(paths.reference_depth_dir / "overall.csv") if paths.reference_depth_dir else pd.DataFrame(),
        "reference_compaction": _read_json(paths.reference_compaction_dir / "manifest.json")
        if paths.reference_compaction_dir
        else {},
        "liquidity_manifest": _read_json(paths.liquidity_backfill_dir / "manifest.json")
        if paths.liquidity_backfill_dir
        else {},
        "liquidity_quality": _read_csv(paths.liquidity_backfill_dir / "historical_quality.csv")
        if paths.liquidity_backfill_dir
        else pd.DataFrame(),
    }


def _g1_security_master_universe(context: dict[str, Any]) -> UsaQualityLane:
    manifest = context["research_manifest"]
    active_map = context["active_symbol_map"]
    reference_map = context["reference_symbol_map"]
    blockers: list[str] = []
    if not manifest:
        blockers.append("USA research-universe manifest missing")
    if active_map.empty:
        blockers.append("active IBKR symbol map missing")
    if reference_map.empty:
        blockers.append("listed-reference symbol map missing")
    active_rows = int(manifest.get("active_core_rows", 0))
    reference_rows = int(manifest.get("reference_core_rows", 0))
    security_rows = int(manifest.get("security_master_rows", 0))
    universe_profile = str(manifest.get("universe_profile", ""))
    if security_rows < 300_000:
        blockers.append("full USA security master denominator too small or missing")
    if universe_profile == "usa_large_liquid_operational_core":
        min_large_liquid_rows = int(manifest.get("minimum_expected_large_liquid_rows", 500))
        if active_rows < min_large_liquid_rows:
            blockers.append(f"large-liquid active core has fewer than {min_large_liquid_rows} names")
    elif active_rows < 5_000:
        blockers.append("active core has fewer than 5,000 names")
    if reference_rows <= active_rows:
        blockers.append("listed-reference denominator does not exceed active core")
    if not active_map.empty:
        exchange_codes = set(active_map.get("exchange_code", pd.Series(dtype="object")).dropna().astype(str))
        banned = exchange_codes & {"PNK", "OBB", "OTC"}
        if banned:
            blockers.append(f"active core includes banned OTC exchange codes: {sorted(banned)}")
        if active_map["refinitiv_ric"].isna().any():
            blockers.append("active core has null RICs")
        if universe_profile == "usa_large_liquid_operational_core":
            min_market_cap = float(manifest.get("min_market_cap_usd", 1_000_000_000.0))
            min_adv = float(manifest.get("min_adv_value_usd", 5_000_000.0))
            min_free_float = float(manifest.get("min_free_float_pct", 10.0))
            missing_screen_columns = [
                column
                for column in ("market_cap_usd", "adv_value_usd", "free_float_pct")
                if column not in active_map.columns
            ]
            if missing_screen_columns:
                blockers.append(f"large-liquid core is missing screening columns: {missing_screen_columns}")
            market_cap = pd.to_numeric(active_map.get("market_cap_usd", pd.Series(dtype=float)), errors="coerce")
            adv_value = pd.to_numeric(active_map.get("adv_value_usd", pd.Series(dtype=float)), errors="coerce")
            free_float = pd.to_numeric(active_map.get("free_float_pct", pd.Series(dtype=float)), errors="coerce")
            if market_cap.isna().any() or market_cap.lt(min_market_cap).any():
                blockers.append("large-liquid core contains missing or below-threshold market cap")
            if adv_value.isna().any() or adv_value.lt(min_adv).any():
                blockers.append("large-liquid core contains missing or below-threshold ADV value")
            if min_free_float > 0 and (free_float.isna().any() or free_float.lt(min_free_float).any()):
                blockers.append("large-liquid core contains missing or below-threshold free float")
    status = "pass" if not blockers else "fail"
    score = 5.0 if not blockers else 0.0
    evidence = (
        f"profile={universe_profile or 'legacy_active_core'}; security_master={security_rows}; "
        f"active_core={active_rows}; listed_reference={reference_rows}"
    )
    return UsaQualityLane(
        "g1_security_master_universe",
        "security_master_agent",
        score,
        5.0,
        status,
        tuple(blockers),
        evidence,
        "Keep full security master as denominator; refresh only if source merge changes.",
    )


def _g2_prices_actions_liquidity(context: dict[str, Any]) -> UsaQualityLane:
    manifest = context["research_manifest"]
    active_run = context["active_run_manifest"]
    active_depth = context["active_depth"]
    active_compaction = context["active_compaction"]
    active_manifest = context["active_manifest"]
    reference_manifest = context["reference_manifest"]
    reference_plan = context["reference_plan"]
    reference_run = context["reference_run_manifest"]
    active_core_rows = int(manifest.get("active_core_rows", 0))
    reference_rows = int(manifest.get("reference_core_rows", 0))
    active_planned = int(active_run.get("planned_rows", len(context["active_plan"])))
    active_completed = _usable_completed_rows(active_manifest, int(active_run.get("completed_rows", 0)))
    reference_planned = int(reference_run.get("planned_rows", len(reference_plan)))
    reference_completed = _usable_completed_rows(reference_manifest, int(reference_run.get("completed_rows", 0)))
    active_zero_output = _zero_output_completed_rows(active_manifest)
    reference_zero_output = _zero_output_completed_rows(reference_manifest)
    blockers: list[str] = []
    score = 0.0
    if active_completed > 0:
        score += 1.0
    if _compaction_row_equal(active_compaction):
        score += 0.75
    if not active_depth.empty:
        depth = active_depth.iloc[0]
        if int(depth.get("non_positive_close_rows", 1)) == 0 and float(depth.get("volume_missing_pct", 1.0)) <= 0.001:
            score += 0.5
    if active_planned and active_completed >= active_planned:
        score += 0.75
    else:
        blockers.append(f"active-core OHLCV incomplete: {active_completed}/{active_planned}")
    paths: UsaDataQualityPaths = context["paths"]
    pit_index_manifest = (
        _read_json(paths.pit_index_constituents_dir / "manifest.json")
        if paths.pit_index_constituents_dir
        else {}
    )
    reference_ready = bool(reference_planned and reference_completed >= reference_planned)
    pit_survivorship_ready = _explicit_scorecard_ready(pit_index_manifest)
    if reference_ready or pit_survivorship_ready:
        score += 0.75
    else:
        blockers.append(
            "USA survivorship control incomplete: "
            f"listed-reference OHLCV={reference_completed}/{reference_planned}; "
            f"PIT index eligible={pit_survivorship_ready}"
        )
    if active_zero_output:
        blockers.append(f"active-core OHLCV has completed rows with zero output: {active_zero_output}")
    if reference_zero_output:
        blockers.append(f"listed-reference OHLCV has completed rows with zero output: {reference_zero_output}")
    corporate_actions_manifest = _read_json(paths.corporate_actions_dir / "manifest.json") if paths.corporate_actions_dir else {}
    if _corporate_actions_ready(corporate_actions_manifest):
        score += 0.5
    else:
        blockers.append("official USA corporate actions route/backfill missing")
    adjusted_returns_manifest = _read_json(paths.adjusted_returns_dir / "manifest.json") if paths.adjusted_returns_dir else {}
    if _adjusted_returns_ready(adjusted_returns_manifest):
        score += 0.5
    else:
        blockers.append("USA adjusted/raw price reconciliation missing")
    liquidity_state = _liquidity_evidence_state(context, active_core_rows, reference_rows)
    if liquidity_state["complete"]:
        score += 0.25
    else:
        blockers.append(str(liquidity_state["blocker"]))
    score = round(min(score, 5.0), 3)
    status = "pass" if score >= 5.0 and not blockers else "partial" if score > 0 else "fail"
    evidence = (
        f"active_ohlcv={active_completed}/{active_planned}; reference_ohlcv={reference_completed}/{reference_planned}; "
        f"pit_index_eligible={pit_survivorship_ready}; "
        f"zero_output_rows active/reference={active_zero_output}/{reference_zero_output}; "
        f"active_compaction={active_compaction.get('source_rows', 0)}->{active_compaction.get('output_rows', 0)}; "
        f"liquidity={liquidity_state['evidence']}"
    )
    return UsaQualityLane(
        "g2_prices_actions_liquidity",
        "historical_price_agent",
        score,
        5.0,
        status,
        tuple(blockers),
        evidence,
        (
            "Maintain the accepted no-Russell PIT index survivorship contract."
            if pit_survivorship_ready
            else "Complete survivorship control through licensed PIT index history or the listed-reference OHLCV denominator."
        ),
    )


def _g3_pit_index_fundamentals(context: dict[str, Any]) -> UsaQualityLane:
    paths: UsaDataQualityPaths = context["paths"]
    blockers: list[str] = []
    score = 0.0
    pit_index_manifest = _read_json(paths.pit_index_constituents_dir / "manifest.json") if paths.pit_index_constituents_dir else {}
    if _explicit_scorecard_ready(pit_index_manifest):
        score += 2.0
    else:
        incomplete_indexes = pit_index_manifest.get("incomplete_index_ids", [])
        if isinstance(incomplete_indexes, list) and incomplete_indexes:
            blockers.append(
                "official PIT index constituents incomplete: "
                + ",".join(str(value) for value in incomplete_indexes)
            )
        else:
            blockers.append("official PIT index constituent artifact missing")
    fund_manifest = _read_json(paths.fundamentals_ibes_events_dir / "manifest.json") if paths.fundamentals_ibes_events_dir else {}
    fund_quality = (
        _read_csv(paths.fundamentals_ibes_events_dir / "historical_quality.csv")
        if paths.fundamentals_ibes_events_dir
        else pd.DataFrame()
    )
    if _pit_historical_manifest_ready(fund_manifest):
        score += 2.0
    else:
        blockers.append("USA PIT fundamentals/IBES/events backfill missing")
    if _pit_historical_quality_ready(
        fund_quality,
        required_datasets={"fundamentals_pit", "ibes_estimates_revisions", "earnings_events_actuals"},
    ):
        score += 1.0
    else:
        blockers.append("USA PIT feature coverage/quality evidence missing")
    status = "pass" if score >= 5.0 and not blockers else "partial" if score > 0 else "fail"
    return UsaQualityLane(
        "g3_pit_index_fundamentals_ibes_events",
        "pit_index_fundamentals_agent",
        round(score, 3),
        5.0,
        status,
        tuple(blockers),
        f"pit_index_dir={paths.pit_index_constituents_dir}; "
        f"scope_profile={pit_index_manifest.get('scope_profile', 'missing')}; "
        f"fund_ibes_dir={paths.fundamentals_ibes_events_dir}",
        "Maintain PIT index, fundamentals, IBES, and earnings-event coverage.",
    )


def _g4_fx_calendar_tca(context: dict[str, Any]) -> UsaQualityLane:
    paths: UsaDataQualityPaths = context["paths"]
    blockers: list[str] = []
    score = 0.5  # USD base/local return handling is simpler, but not enough for a pass.
    calendar_manifest = _read_json(paths.fx_calendar_tca_dir / "manifest.json") if paths.fx_calendar_tca_dir else {}
    evidence_quality = (
        _read_csv(paths.fx_calendar_tca_dir / "evidence_quality.csv")
        if paths.fx_calendar_tca_dir
        else pd.DataFrame()
    )
    required_components = {
        "exchange_calendars",
        "halt_suspension_events",
        "tca_assumptions",
    }
    complete_components: list[str] = []
    evidence_gate_ready = False
    if not evidence_quality.empty and "component" in evidence_quality.columns:
        ready = evidence_quality.get("backtest_ready", pd.Series(False, index=evidence_quality.index))
        ready = ready.astype(str).str.lower().isin({"true", "1", "yes"})
        complete_components = evidence_quality.loc[ready, "component"].dropna().astype(str).tolist()
        score = min(5.0, score + 1.5 * len(set(complete_components)))
        evidence_gate_ready = required_components.issubset(set(complete_components))
    validated_backtest_gate = bool(
        calendar_manifest.get("status") == "complete"
        and calendar_manifest.get("backtest_ready") is True
        and calendar_manifest.get("backtest_scorecard_eligible") is True
        and calendar_manifest.get("production_ready") is False
        and calendar_manifest.get("live_execution_ready") is False
        and evidence_gate_ready
    )
    if validated_backtest_gate:
        score = 5.0
    else:
        quality_fields = calendar_manifest.get("quality_fields", {})
        incomplete_components = (
            quality_fields.get(
                "backtest_incomplete_components",
                quality_fields.get("incomplete_components", []),
            )
            if isinstance(quality_fields, dict)
            else []
        )
        if isinstance(incomplete_components, list) and incomplete_components:
            blockers.append(
                "USA calendar/halts/TCA incomplete components: "
                + ",".join(str(value) for value in incomplete_components)
            )
        else:
            blockers.append("USA exchange calendar/halts/suspensions/TCA artifact missing")
    status = "pass" if score >= 5.0 and not blockers else "partial"
    return UsaQualityLane(
        "g4_fx_calendar_tca",
        "calendar_tca_agent",
        round(score, 3),
        5.0,
        status,
        tuple(blockers),
        f"USD market; complete_components={','.join(complete_components) or 'none'}; "
        f"backtest_ready={calendar_manifest.get('backtest_ready', False)}; "
        f"backtest_scorecard_eligible={calendar_manifest.get('backtest_scorecard_eligible', False)}; "
        f"production_ready={calendar_manifest.get('production_ready', False)}; "
        f"live_execution_ready={calendar_manifest.get('live_execution_ready', False)}",
        "Validate research TCA calibration for backtests; require execution fills separately for production/live.",
    )


def _g5_leakage_ops(context: dict[str, Any]) -> UsaQualityLane:
    paths: UsaDataQualityPaths = context["paths"]
    blockers: list[str] = []
    score = 0.0
    if paths.commands_log and paths.commands_log.exists():
        score += 1.0
    else:
        blockers.append("commands log missing")
    if paths.leakage_qa_dir and (paths.leakage_qa_dir / "manifest.json").exists():
        leakage_manifest = _read_json(paths.leakage_qa_dir / "manifest.json")
        leakage_ready = bool(
            leakage_manifest.get("status") == "pass"
            and int(leakage_manifest.get("total_leakage_violations", 1)) == 0
            and int(leakage_manifest.get("decision_rows", 0)) > 0
            and int(leakage_manifest.get("decision_entities", 0)) > 0
            and bool(leakage_manifest.get("features"))
        )
        if leakage_ready:
            score += 3.0
        else:
            blockers.append("USA leakage QA did not pass cleanly")
    else:
        blockers.append("USA leakage QA missing")
    if context["active_manifest"].empty:
        blockers.append("active extraction manifest missing")
    else:
        score += 1.0
    status = "pass" if score >= 5.0 and not blockers else "partial" if score > 0 else "fail"
    return UsaQualityLane(
        "g5_leakage_ops_repro",
        "data_quality_auditor",
        round(score, 3),
        5.0,
        status,
        tuple(blockers),
        f"commands_log={bool(paths.commands_log and paths.commands_log.exists())}; active_manifest_rows={len(context['active_manifest'])}",
        "Run USA leakage QA after PIT features exist; keep command log and scorecards current after every chunk.",
    )


def _compaction_row_equal(manifest: dict[str, Any]) -> bool:
    return bool(
        manifest
        and manifest.get("status") == "complete"
        and int(manifest.get("source_rows", -1)) == int(manifest.get("output_rows", -2))
    )


def _explicit_scorecard_ready(manifest: dict[str, Any]) -> bool:
    return bool(
        manifest
        and manifest.get("status") in {"complete", "pass"}
        and manifest.get("scorecard_eligible") is True
    )


def _corporate_actions_ready(manifest: dict[str, Any]) -> bool:
    if not manifest or manifest.get("status") != "complete":
        return False
    planned = int(manifest.get("planned_rows", 0))
    completed = int(manifest.get("completed_rows", 0))
    failed = int(manifest.get("failed_rows", 0))
    alpha_usable = int(manifest.get("alpha_usable_rows", 0))
    return planned > 0 and completed >= planned and failed == 0 and alpha_usable > 0


def _adjusted_returns_ready(manifest: dict[str, Any]) -> bool:
    return bool(
        manifest
        and manifest.get("status") == "pass"
        and manifest.get("production_adjusted_return_ready") is True
        and int(manifest.get("non_null_total_return_rows", 0)) > 0
    )


def _pit_historical_manifest_ready(manifest: dict[str, Any]) -> bool:
    if not manifest:
        return False
    planned = int(manifest.get("planned_rows", 0))
    completed = int(manifest.get("completed_rows", 0))
    empty = int(manifest.get("empty_rows", 0))
    failed = int(manifest.get("failed_rows", 0))
    history_rows = int(manifest.get("history_rows", 0))
    return planned > 0 and completed + empty >= planned and failed == 0 and history_rows > 0


def _pit_historical_quality_ready(quality: pd.DataFrame, required_datasets: set[str] | None = None) -> bool:
    if quality.empty:
        return False
    if required_datasets:
        if "dataset_id" not in quality.columns:
            return False
        present = set(quality["dataset_id"].dropna().astype(str))
        if not required_datasets.issubset(present):
            return False
    failed = pd.to_numeric(quality.get("failed_rows", pd.Series(1, index=quality.index)), errors="coerce").fillna(1)
    history = pd.to_numeric(quality.get("history_rows", pd.Series(0, index=quality.index)), errors="coerce").fillna(0)
    coverage = pd.to_numeric(
        quality.get("avg_coverage", quality.get("min_coverage", pd.Series(0, index=quality.index))),
        errors="coerce",
    ).fillna(0)
    return bool(failed.eq(0).all() and history.gt(0).all() and coverage.gt(0).all())


def _zero_output_completed_rows(manifest: pd.DataFrame) -> int:
    return int(len(_zero_output_completed_frame(manifest)))


def _zero_output_completed_frame(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        return manifest.copy()
    if "status" not in manifest.columns:
        return manifest.iloc[0:0].copy()
    completed = manifest["status"].astype(str).isin(["completed", "completed_partial"])
    zero_output = pd.Series(False, index=manifest.index)
    if "price_columns" in manifest.columns:
        zero_output = zero_output | pd.to_numeric(manifest["price_columns"], errors="coerce").fillna(0).le(0)
    if "long_ohlcv_rows" in manifest.columns:
        zero_output = zero_output | pd.to_numeric(manifest["long_ohlcv_rows"], errors="coerce").fillna(0).le(0)
    return manifest.loc[completed & zero_output].copy().reset_index(drop=True)


def _usable_completed_rows(manifest: pd.DataFrame, fallback: int = 0) -> int:
    if manifest.empty:
        return fallback
    if "status" not in manifest.columns:
        return fallback
    status = manifest["status"].astype(str)
    completed = status.isin(["completed", "completed_partial"])
    structural = status.isin(["completed_prelisting", "completed_postretirement"])
    if "price_columns" not in manifest.columns and "long_ohlcv_rows" not in manifest.columns:
        return int((completed | structural).sum())
    zero_output = pd.Series(False, index=manifest.index)
    if "price_columns" in manifest.columns:
        zero_output = zero_output | pd.to_numeric(manifest["price_columns"], errors="coerce").fillna(0).le(0)
    if "long_ohlcv_rows" in manifest.columns:
        zero_output = zero_output | pd.to_numeric(manifest["long_ohlcv_rows"], errors="coerce").fillna(0).le(0)
    return int(((completed & ~zero_output) | structural).sum())


def _liquidity_evidence_state(context: dict[str, Any], active_core_rows: int, reference_rows: int) -> dict[str, object]:
    del active_core_rows, reference_rows
    manifest = context.get("liquidity_manifest", {})
    quality = context.get("liquidity_quality", pd.DataFrame())
    if manifest:
        planned = int(manifest.get("planned_rows", 0))
        completed = int(manifest.get("completed_rows", 0))
        failed = int(manifest.get("failed_rows", 0))
        history_rows = int(manifest.get("history_rows", 0))
        min_coverage = float(quality["min_coverage"].min()) if not quality.empty and "min_coverage" in quality else 0.0
        complete = planned > 0 and completed >= planned and failed == 0 and history_rows > 0 and min_coverage >= 0.8
        return {
            "complete": complete,
            "blocker": "USA historical liquidity route proven but full liquidity/capacity backfill incomplete"
            if history_rows > 0
            else "USA historical ADV/turnover/capacity evidence missing",
            "evidence": f"{completed}/{planned} rows; failed={failed}; history_rows={history_rows}; min_coverage={min_coverage:.3f}",
        }
    depth = context["active_depth"]
    if depth.empty:
        return {
            "complete": False,
            "blocker": "USA historical ADV/turnover/capacity evidence missing",
            "evidence": "no liquidity PIT artifact and no active depth",
        }
    tickers = int(depth.iloc[0].get("tickers", 0))
    return {
        "complete": False,
        "blocker": "USA historical ADV/turnover/capacity evidence missing",
        "evidence": f"active_depth_tickers={tickers}; PIT liquidity artifact missing",
    }


def _next_queue(lanes: tuple[UsaQualityLane, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    priority = 1
    for lane in lanes:
        for blocker in lane.critical_blockers:
            rows.append(
                {
                    "priority": priority,
                    "lane_id": lane.lane_id,
                    "lead_agent": lane.lead_agent,
                    "blocker": blocker,
                    "next_action": lane.next_action,
                }
            )
            priority += 1
    return pd.DataFrame(
        rows,
        columns=("priority", "lane_id", "lead_agent", "blocker", "next_action"),
    )


def _summary_md(
    run_id: str,
    lanes: tuple[UsaQualityLane, ...],
    overall_score: float,
    accepted: bool,
    status: str,
    next_queue: pd.DataFrame,
) -> str:
    lines = [
        "# USA Data Quality Loop",
        "",
        f"Run ID: `{run_id}`",
        f"Status: `{status}`",
        f"Accepted: `{accepted}`",
        f"Overall score: `{overall_score:.3f}/5.000`",
        "",
        "## Lane Scorecard",
        "",
        "| Lane | Agent | Score | Status | Evidence |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for lane in lanes:
        lines.append(f"| `{lane.lane_id}` | `{lane.lead_agent}` | {lane.score:.3f}/5 | {lane.status} | {lane.evidence} |")
    lines.extend(["", "## Next Queue", ""])
    if next_queue.empty:
        lines.append("No blockers remain.")
    else:
        for _, row in next_queue.iterrows():
            lines.append(f"- P{row['priority']} `{row['lane_id']}`: {row['blocker']} -> {row['next_action']}")
    lines.extend(["", "## Admission Rule", ""])
    if accepted:
        lines.append(
            "USA may enter leakage-controlled alpha research and backtesting. Production execution and IBKR/live trading remain blocked until execution-fill TCA evidence passes."
        )
    else:
        lines.append(
            "USA cannot enter alpha optimization or backtesting until all five research lanes are 5/5 with zero critical blockers. Production execution and IBKR/live trading remain separately blocked."
        )
    return "\n".join(lines) + "\n"


def _paths_manifest(paths: UsaDataQualityPaths) -> dict[str, str | None]:
    return {key: str(value) if value is not None else None for key, value in paths.__dict__.items()}
