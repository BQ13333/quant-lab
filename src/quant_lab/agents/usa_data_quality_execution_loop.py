from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.agents.usa_data_quality_loop import UsaDataQualityPaths, default_usa_data_quality_paths, run_usa_data_quality_loop


ACTIVE_RUN_ID = "REFINITIV-USA-HISTORICAL-20Y-IBKR-ACTIVE-LIQUID-CORE-V1-20260709"
LISTED_REFERENCE_RUN_ID = "REFINITIV-USA-HISTORICAL-20Y-LISTED-REFERENCE-CORE-V1-20260709"
LARGE_LIQUID_UNIVERSE_RUN_ID = "USA-LARGE-LIQUID-UNIVERSE-V1-20260709"
LARGE_LIQUID_SNAPSHOT_RUN_ID = "REFINITIV-USA-LARGE-LIQUID-SNAPSHOT-20260709"
LARGE_LIQUID_RUN_ID = "REFINITIV-USA-HISTORICAL-20Y-LARGE-LIQUID-CORE-V1-20260709"
ACTIVE_SYMBOL_MAP = (
    "artifacts/refinitiv_symbol_map/"
    "REFINITIV-SYMBOL-MAP-USA_IBKR_ACTIVE_LIQUID_CORE_V1-USA-RESEARCH-UNIVERSE-V1-20260709/symbol_map.csv"
)
LARGE_LIQUID_SYMBOL_MAP = (
    "artifacts/refinitiv_symbol_map/"
    f"REFINITIV-SYMBOL-MAP-USA_LARGE_LIQUID_CORE_V1-{LARGE_LIQUID_UNIVERSE_RUN_ID}/symbol_map.csv"
)
LARGE_LIQUID_SNAPSHOT_CSV = f"artifacts/refinitiv_current_snapshot/{LARGE_LIQUID_SNAPSHOT_RUN_ID}/current_snapshot.csv"
LARGE_LIQUID_QUARANTINE_CSV = f"artifacts/historical_backfill/{LARGE_LIQUID_RUN_ID}/quarantined_plan_rows.csv"
LISTED_REFERENCE_SYMBOL_MAP = (
    "artifacts/refinitiv_symbol_map/"
    "REFINITIV-SYMBOL-MAP-USA_LISTED_REFERENCE_CORE_V1-USA-RESEARCH-UNIVERSE-V1-20260709/symbol_map.csv"
)
FIELD_CATALOG = "artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv"


@dataclass(frozen=True)
class UsaExecutionLoopConfig:
    run_id: str
    artifact_root: Path = Path("artifacts/usa_data_quality_execution_loop")
    project_root: Path = Path(".")
    scorecard_run_id: str = "USA-DATA-QUALITY-LOOP-20260709"
    start_date: str = "2006-01-01"
    end_date: str = "2026-07-06"
    batch_size: int = 20
    active_max_plan_rows: int = 5
    reference_max_plan_rows: int = 5
    refinitiv_session: str = "desktop.workspace"
    vendor_health_status: str = "unknown"
    min_market_cap_usd: float = 1_000_000_000.0
    min_adv_value_usd: float = 5_000_000.0
    min_free_float_pct: float = 10.0
    paths: UsaDataQualityPaths | None = None


@dataclass(frozen=True)
class UsaExecutionLoopArtifacts:
    run_id: str
    artifact_dir: Path
    agent_roster_path: Path
    execution_queue_path: Path
    zero_output_quarantine_path: Path
    historical_quarantine_path: Path
    large_zero_output_quarantine_path: Path
    large_historical_quarantine_path: Path
    summary_path: Path
    manifest_path: Path
    execution_queue: pd.DataFrame
    zero_output_quarantine: pd.DataFrame
    large_zero_output_quarantine: pd.DataFrame


def latest_usa_execution_loop_run_id() -> str:
    return datetime.now(timezone.utc).strftime("USA-DATA-QUALITY-EXECUTION-LOOP-%Y%m%d-%H%M%S")


def write_usa_execution_loop_artifacts(config: UsaExecutionLoopConfig) -> UsaExecutionLoopArtifacts:
    project_root = config.project_root
    paths = config.paths or default_usa_data_quality_paths(project_root)
    artifact_dir = project_root / config.artifact_root / config.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    scorecard = run_usa_data_quality_loop(
        paths=paths,
        artifact_root=project_root / "artifacts/institutional_data_quality",
        run_id=config.scorecard_run_id,
        project_root=project_root,
    )
    active_manifest = _read_csv(paths.active_historical_dir / "historical_manifest.csv")
    active_plan = _read_csv(paths.active_historical_dir / "historical_plan.csv")
    reference_manifest = _read_csv(paths.reference_historical_dir / "historical_manifest.csv")
    reference_plan = _read_csv(paths.reference_historical_dir / "historical_plan.csv")
    large_historical_dir = project_root / "artifacts/historical_backfill" / LARGE_LIQUID_RUN_ID
    large_manifest = _read_csv(large_historical_dir / "historical_manifest.csv")
    large_plan = _read_csv(large_historical_dir / "historical_plan.csv")
    large_symbol_map_exists = (project_root / LARGE_LIQUID_SYMBOL_MAP).exists()
    large_snapshot_exists = (project_root / LARGE_LIQUID_SNAPSHOT_CSV).exists()
    large_request_exists = (
        project_root / "artifacts/research_universes" / LARGE_LIQUID_UNIVERSE_RUN_ID / "usa_large_liquid_snapshot_request.csv"
    ).exists()
    zero_output = _zero_output_rows(active_manifest)
    large_zero_output = _nonusable_rows(large_manifest)

    zero_output_path = artifact_dir / "active_zero_output_quarantine.csv"
    historical_quarantine_path = paths.active_historical_dir / "quarantined_plan_rows.csv"
    large_zero_output_path = artifact_dir / "large_liquid_zero_output_quarantine.csv"
    large_historical_quarantine_path = project_root / LARGE_LIQUID_QUARANTINE_CSV
    zero_output.to_csv(zero_output_path, index=False)
    historical_quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    zero_output.to_csv(historical_quarantine_path, index=False)
    large_zero_output.to_csv(large_zero_output_path, index=False)
    large_historical_quarantine_path.parent.mkdir(parents=True, exist_ok=True)
    large_zero_output.to_csv(large_historical_quarantine_path, index=False)
    agent_roster = _agent_roster()
    execution_queue = _execution_queue(
        config=config,
        scorecard_accepted=scorecard.accepted,
        zero_output_path=zero_output_path,
        zero_output_count=int(len(zero_output)),
        active_manifest=active_manifest,
        active_plan=active_plan,
        reference_manifest=reference_manifest,
        reference_plan=reference_plan,
        large_manifest=large_manifest,
        large_plan=large_plan,
        large_symbol_map_exists=large_symbol_map_exists,
        large_snapshot_exists=large_snapshot_exists,
        large_request_exists=large_request_exists,
    )

    agent_roster_path = artifact_dir / "agent_roster.csv"
    execution_queue_path = artifact_dir / "execution_queue.csv"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"
    agent_roster.to_csv(agent_roster_path, index=False)
    execution_queue.to_csv(execution_queue_path, index=False)
    summary_path.write_text(
        _summary_md(config, scorecard.accepted, zero_output, large_zero_output, execution_queue),
        encoding="utf-8",
    )
    manifest = {
        "run_id": config.run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "scorecard_run_id": config.scorecard_run_id,
        "scorecard_accepted": bool(scorecard.accepted),
        "scorecard_overall_score": scorecard.overall_score,
        "vendor_health_status": config.vendor_health_status,
        "large_liquid_universe_run_id": LARGE_LIQUID_UNIVERSE_RUN_ID,
        "large_liquid_snapshot_run_id": LARGE_LIQUID_SNAPSHOT_RUN_ID,
        "large_liquid_historical_run_id": LARGE_LIQUID_RUN_ID,
        "large_liquid_symbol_map_exists": bool(large_symbol_map_exists),
        "large_liquid_snapshot_exists": bool(large_snapshot_exists),
        "active_zero_output_rows": int(len(zero_output)),
        "large_liquid_zero_output_rows": int(len(large_zero_output)),
        "queue_rows": int(len(execution_queue)),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {
            "agent_roster": str(agent_roster_path.resolve()),
            "execution_queue": str(execution_queue_path.resolve()),
            "active_zero_output_quarantine": str(zero_output_path.resolve()),
            "historical_quarantine": str(historical_quarantine_path.resolve()),
            "large_liquid_zero_output_quarantine": str(large_zero_output_path.resolve()),
            "large_liquid_historical_quarantine": str(large_historical_quarantine_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return UsaExecutionLoopArtifacts(
        run_id=config.run_id,
        artifact_dir=artifact_dir,
        agent_roster_path=agent_roster_path,
        execution_queue_path=execution_queue_path,
        zero_output_quarantine_path=zero_output_path,
        historical_quarantine_path=historical_quarantine_path,
        large_zero_output_quarantine_path=large_zero_output_path,
        large_historical_quarantine_path=large_historical_quarantine_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        execution_queue=execution_queue,
        zero_output_quarantine=zero_output,
        large_zero_output_quarantine=large_zero_output,
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _agent_roster() -> pd.DataFrame:
    rows = [
        (
            "chief_data_pm",
            "Own the 5/5 admission decision and block alpha/backtests/live trading while any critical blocker remains.",
        ),
        (
            "vendor_throttle_guard",
            "Run tiny health checks only; stop live extraction on Refinitiv 429 or repeated provider errors.",
        ),
        (
            "symbol_route_validator",
            "Probe ASQ/NAQ RIC variants and field route issues on quarantined rows before larger retries.",
        ),
        (
            "usa_large_liquid_universe_agent",
            "Convert the broad active USA candidate pool into a >$1bn market-cap and liquidity-gated operational core.",
        ),
        (
            "historical_price_agent",
            "Resume active-core and listed-reference OHLCV in bounded chunks using quarantine controls.",
        ),
        (
            "storage_ops_agent",
            "Run depth, partition health, and row-equal compaction after every successful chunk.",
        ),
        (
            "data_quality_auditor",
            "Rerun USA 5/5 scorecard, update queue artifacts, and keep zero-output rows visible.",
        ),
    ]
    return pd.DataFrame(rows, columns=["agent_id", "mandate"])


def _zero_output_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty or "status" not in manifest.columns:
        return pd.DataFrame(columns=["plan_row_id"])
    status = manifest["status"].astype(str)
    structural = status.isin(["completed_prelisting", "completed_postretirement"])
    completed = status.isin(["completed", "completed_partial"])
    zero = pd.Series(False, index=manifest.index)
    if "price_columns" in manifest.columns:
        zero = zero | pd.to_numeric(manifest["price_columns"], errors="coerce").fillna(0).le(0)
    if "long_ohlcv_rows" in manifest.columns:
        zero = zero | pd.to_numeric(manifest["long_ohlcv_rows"], errors="coerce").fillna(0).le(0)
    cols = [
        "plan_row_id",
        "country",
        "exchange",
        "batch_no",
        "window_no",
        "start_date",
        "end_date",
        "requested_tickers",
        "tickers",
        "local_symbols",
        "status",
        "price_rows",
        "price_columns",
        "failures",
        "long_ohlcv_rows",
        "coverage_path",
    ]
    available_cols = [col for col in cols if col in manifest.columns]
    quarantined = manifest.loc[completed & zero, available_cols].copy().reset_index(drop=True)
    if quarantined.empty:
        return quarantined
    quarantined["reason"] = "zero_ohlcv_payload"
    quarantined["evidence_path"] = quarantined["coverage_path"] if "coverage_path" in quarantined.columns else ""
    quarantined["detected_at_utc"] = datetime.now(timezone.utc).isoformat()
    quarantined["retry_policy"] = "retry_tiny_after_vendor_cooldown_and_ric_route_probe"
    return quarantined


def _nonusable_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty or "status" not in manifest.columns:
        return pd.DataFrame(columns=["plan_row_id"])
    completed = manifest["status"].astype(str).isin(["completed", "completed_partial"])
    failed_empty = manifest["status"].astype(str).eq("failed_empty")
    zero = pd.Series(False, index=manifest.index)
    if "price_columns" in manifest.columns:
        zero = zero | pd.to_numeric(manifest["price_columns"], errors="coerce").fillna(0).le(0)
    if "long_ohlcv_rows" in manifest.columns:
        zero = zero | pd.to_numeric(manifest["long_ohlcv_rows"], errors="coerce").fillna(0).le(0)
    cols = [
        "plan_row_id",
        "country",
        "exchange",
        "batch_no",
        "window_no",
        "start_date",
        "end_date",
        "requested_tickers",
        "tickers",
        "local_symbols",
        "status",
        "price_rows",
        "price_columns",
        "failures",
        "long_ohlcv_rows",
        "coverage_path",
        "error_type",
        "error",
    ]
    available_cols = [col for col in cols if col in manifest.columns]
    quarantined = manifest.loc[(completed & zero) | failed_empty, available_cols].copy().reset_index(drop=True)
    if quarantined.empty:
        return quarantined
    quarantined["reason"] = quarantined["status"].map({"failed_empty": "failed_empty_ohlcv_payload"}).fillna(
        "zero_ohlcv_payload"
    )
    quarantined["evidence_path"] = quarantined["coverage_path"] if "coverage_path" in quarantined.columns else ""
    quarantined["detected_at_utc"] = datetime.now(timezone.utc).isoformat()
    quarantined["retry_policy"] = "route_probe_then_retry_tiny_after_vendor_cooldown"
    return quarantined


def _usable_completed_rows(manifest: pd.DataFrame) -> int:
    if manifest.empty or "status" not in manifest.columns:
        return 0
    status = manifest["status"].astype(str)
    structural = status.eq("completed_prelisting")
    completed = status.isin(["completed", "completed_partial"])
    if "price_columns" in manifest.columns:
        completed = completed & pd.to_numeric(manifest["price_columns"], errors="coerce").fillna(0).gt(0)
    if "long_ohlcv_rows" in manifest.columns:
        completed = completed & pd.to_numeric(manifest["long_ohlcv_rows"], errors="coerce").fillna(0).gt(0)
    return int((completed | structural).sum())


def _execution_queue(
    config: UsaExecutionLoopConfig,
    scorecard_accepted: bool,
    zero_output_path: Path,
    zero_output_count: int,
    active_manifest: pd.DataFrame,
    active_plan: pd.DataFrame,
    reference_manifest: pd.DataFrame,
    reference_plan: pd.DataFrame,
    large_manifest: pd.DataFrame,
    large_plan: pd.DataFrame,
    large_symbol_map_exists: bool,
    large_snapshot_exists: bool,
    large_request_exists: bool,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    vendor_ok = config.vendor_health_status == "healthy"
    active_usable = _usable_completed_rows(active_manifest)
    reference_usable = _usable_completed_rows(reference_manifest)
    large_usable = _usable_completed_rows(large_manifest)
    active_planned = int(len(active_plan))
    reference_planned = int(len(reference_plan))
    large_planned = int(len(large_plan))

    def add(
        owner_agent: str,
        phase: str,
        status: str,
        command: str,
        acceptance_gate: str,
        notes: str,
        promotes_alpha: bool = False,
    ) -> None:
        rows.append(
            {
                "sequence": len(rows) + 1,
                "owner_agent": owner_agent,
                "phase": phase,
                "status": status,
                "command": command,
                "acceptance_gate": acceptance_gate,
                "notes": notes,
                "promotes_alpha": bool(promotes_alpha),
            }
        )

    if not vendor_ok:
        add(
            "vendor_throttle_guard",
            "vendor_health",
            "blocked",
            "uv run --extra refinitiv python - <<'PY'\nfrom quant_lab.data.vendor_adapters import RefinitivAdapter\nadapter=RefinitivAdapter(session_name='desktop.workspace')\ntry:\n    adapter.open(); print(adapter.health_check())\nfinally:\n    adapter.close()\nPY",
            "Refinitiv health check must pass before live extraction resumes.",
            f"Vendor status is {config.vendor_health_status}; do not run live OHLCV jobs during throttle.",
        )
        if not large_request_exists:
            add(
                "usa_large_liquid_universe_agent",
                "large_liquid_snapshot_request",
                "ready",
                _large_liquid_request_command(config),
                "Snapshot request and field map must be staged before the next healthy vendor window.",
                "Non-live artifact build only; safe to run while Refinitiv is throttled.",
            )
        add(
            "usa_large_liquid_universe_agent",
            "large_liquid_snapshot",
            "blocked_by_vendor_health",
            _large_liquid_snapshot_command(config),
            "Refinitiv health check must pass before requesting market-cap/ADV fields.",
            "This is the next live USA task after the hourly health check clears.",
        )
    else:
        if not large_request_exists:
            add(
                "usa_large_liquid_universe_agent",
                "large_liquid_snapshot_request",
                "ready",
                _large_liquid_request_command(config),
                "Snapshot request and field map must exist before live market-cap/ADV extraction.",
                "Builds request artifacts from the existing active USA candidate map.",
            )
        if not large_snapshot_exists:
            add(
                "usa_large_liquid_universe_agent",
                "large_liquid_snapshot",
                "ready",
                _large_liquid_snapshot_command(config),
                "Current Refinitiv market-cap/ADV/free-float snapshot must have acceptable coverage.",
                "Run in bounded chunks; stop on renewed throttle.",
            )
            add(
                "usa_large_liquid_universe_agent",
                "large_liquid_universe_build",
                "queued_after_snapshot",
                _large_liquid_build_command(config),
                "Large-liquid symbol map must be built from the completed snapshot before OHLCV spend.",
                f"Thresholds: market cap >= {config.min_market_cap_usd:,.0f}, ADV >= {config.min_adv_value_usd:,.0f}.",
            )
        elif not large_symbol_map_exists:
            add(
                "usa_large_liquid_universe_agent",
                "large_liquid_universe_build",
                "ready",
                _large_liquid_build_command(config),
                "Large-liquid symbol map must be created before historical OHLCV spend.",
                "Snapshot exists; build the filtered universe now.",
            )
        elif not large_planned or large_usable < large_planned:
            add(
                "historical_price_agent",
                "large_liquid_ohlcv",
                "ready",
                _large_liquid_backfill_command(config),
                "Large-liquid OHLCV bounded chunk must finish without zero-output rows.",
                f"Large-liquid usable rows {large_usable}/{large_planned or 'unplanned'}; old active core is paused to save quota.",
            )
        else:
            add(
                "symbol_route_validator",
                "legacy_quarantine_probe",
                "queued_after_large_liquid",
                _ric_route_probe_command(config),
                "Quarantined legacy active-core rows must be explained only after the large-liquid lane is complete.",
                f"Legacy active rows {active_usable}/{active_planned}; reference rows {reference_usable}/{reference_planned}.",
            )

    add(
        "storage_ops_agent",
        "post_chunk_depth",
        "queued_after_live_chunk",
        _depth_command(LARGE_LIQUID_RUN_ID),
        "Depth report must update rows/tickers and quality rates.",
        "Run after any successful large-liquid OHLCV chunk.",
    )
    add(
        "storage_ops_agent",
        "post_chunk_compaction",
        "queued_after_depth",
        _compaction_command(LARGE_LIQUID_RUN_ID),
        "Compaction must be row-equal.",
        "Run after depth, before scorecard refresh.",
    )
    add(
        "data_quality_auditor",
        "scorecard_refresh",
        "ready",
        f"uv run python examples/run_usa_data_quality_loop.py --run-id {config.scorecard_run_id}",
        "USA remains blocked unless scorecard is 5/5 with zero critical blockers.",
        f"Current scorecard accepted={scorecard_accepted}; never promote alpha while false.",
    )
    return pd.DataFrame(rows)


def _active_backfill_command(config: UsaExecutionLoopConfig, zero_output_path: Path) -> str:
    return (
        "uv run --extra refinitiv python examples/run_refinitiv_historical_backfill.py "
        f"--run-id {ACTIVE_RUN_ID} "
        f"--symbol-map-csv {ACTIVE_SYMBOL_MAP} "
        f"--field-catalog-csv {FIELD_CATALOG} "
        '--include-countries "United States" '
        f"--start-date {config.start_date} --end-date {config.end_date} "
        f"--window-years 5 --batch-size {config.batch_size} "
        f"--max-plan-rows {config.active_max_plan_rows} --interval 1d "
        f"--refinitiv-session {config.refinitiv_session} "
        f"--exclude-plan-row-ids-csv {zero_output_path}"
    )


def _reference_backfill_command(config: UsaExecutionLoopConfig) -> str:
    return (
        "uv run --extra refinitiv python examples/run_refinitiv_historical_backfill.py "
        f"--run-id {LISTED_REFERENCE_RUN_ID} "
        f"--symbol-map-csv {LISTED_REFERENCE_SYMBOL_MAP} "
        f"--field-catalog-csv {FIELD_CATALOG} "
        '--include-countries "United States" '
        f"--start-date {config.start_date} --end-date {config.end_date} "
        f"--window-years 5 --batch-size {config.batch_size} "
        f"--max-plan-rows {config.reference_max_plan_rows} --interval 1d "
        f"--refinitiv-session {config.refinitiv_session}"
    )


def _large_liquid_request_command(config: UsaExecutionLoopConfig) -> str:
    return (
        "uv run python examples/build_usa_large_liquid_universe.py "
        f"--run-id {LARGE_LIQUID_UNIVERSE_RUN_ID} "
        f"--min-market-cap-usd {config.min_market_cap_usd:.0f} "
        f"--min-adv-value-usd {config.min_adv_value_usd:.0f} "
        f"--min-free-float-pct {config.min_free_float_pct:.4f}"
    )


def _large_liquid_snapshot_command(config: UsaExecutionLoopConfig) -> str:
    return (
        "uv run --extra refinitiv python examples/run_usa_large_liquid_snapshot.py "
        f"--run-id {LARGE_LIQUID_SNAPSHOT_RUN_ID} "
        f"--candidate-symbol-map {ACTIVE_SYMBOL_MAP} "
        f"--session {config.refinitiv_session} "
        "--chunk-size 25 --sleep-seconds 0.25"
    )


def _large_liquid_build_command(config: UsaExecutionLoopConfig) -> str:
    return (
        "uv run python examples/build_usa_large_liquid_universe.py "
        f"--run-id {LARGE_LIQUID_UNIVERSE_RUN_ID} "
        f"--liquidity-snapshot-csv {LARGE_LIQUID_SNAPSHOT_CSV} "
        f"--min-market-cap-usd {config.min_market_cap_usd:.0f} "
        f"--min-adv-value-usd {config.min_adv_value_usd:.0f} "
        f"--min-free-float-pct {config.min_free_float_pct:.4f}"
    )


def _large_liquid_backfill_command(config: UsaExecutionLoopConfig) -> str:
    return (
        "uv run --extra refinitiv python examples/run_refinitiv_historical_backfill.py "
        f"--run-id {LARGE_LIQUID_RUN_ID} "
        f"--symbol-map-csv {LARGE_LIQUID_SYMBOL_MAP} "
        f"--field-catalog-csv {FIELD_CATALOG} "
        '--include-countries "United States" '
        f"--start-date {config.start_date} --end-date {config.end_date} "
        f"--window-years 5 --batch-size {config.batch_size} "
        f"--max-plan-rows {config.active_max_plan_rows} --interval 1d "
        f"--refinitiv-session {config.refinitiv_session} "
        f"--exclude-plan-row-ids-csv {LARGE_LIQUID_QUARANTINE_CSV}"
    )


def _ric_route_probe_command(config: UsaExecutionLoopConfig) -> str:
    del config
    return (
        "uv run --extra refinitiv python - <<'PY'\n"
        "from quant_lab.data.vendor_adapters import RefinitivAdapter\n"
        "symbols=['ABEO.O','ABEO.OQ','ABAT.O','ABAT.OQ','AAPL.O','AAPL.OQ']\n"
        "adapter=RefinitivAdapter(session_name='desktop.workspace')\n"
        "try:\n"
        "    adapter.open()\n"
        "    for sym in symbols:\n"
        "        data=adapter.ld.get_history(universe=[sym], fields=['TRDPRC_1'], start='2024-01-01', end='2026-07-06', interval='1D')\n"
        "        print(sym, 0 if data is None else len(data.dropna(how='all')))\n"
        "finally:\n"
        "    adapter.close()\n"
        "PY"
    )


def _depth_command(run_id: str) -> str:
    return (
        "uv run python examples/run_data_depth_report.py "
        f"artifacts/data_lake/prices_long/{run_id}/ohlcv "
        f"--run-id {run_id}-DEPTH "
        f"--extraction-manifest-path artifacts/historical_backfill/{run_id}/historical_manifest.csv"
    )


def _compaction_command(run_id: str) -> str:
    return (
        "uv run python examples/run_data_lake_partition_ops.py "
        f"artifacts/data_lake/prices_long/{run_id}/ohlcv "
        f"--run-id {run_id}-COMPACT --compact --overwrite"
    )


def _summary_md(
    config: UsaExecutionLoopConfig,
    scorecard_accepted: bool,
    zero_output: pd.DataFrame,
    large_zero_output: pd.DataFrame,
    execution_queue: pd.DataFrame,
) -> str:
    lines = [
        "# USA Data Quality Execution Loop",
        "",
        f"Run ID: `{config.run_id}`",
        f"Vendor health status: `{config.vendor_health_status}`",
        f"Scorecard accepted: `{scorecard_accepted}`",
        f"Active zero-output quarantine rows: `{len(zero_output)}`",
        f"Large-liquid non-usable quarantine rows: `{len(large_zero_output)}`",
        "",
        "## Agent Queue",
        "",
    ]
    for _, row in execution_queue.iterrows():
        lines.append(
            f"- P{row['sequence']} `{row['owner_agent']}` `{row['phase']}` "
            f"[{row['status']}]: {row['acceptance_gate']}"
        )
    lines.extend(
        [
            "",
            "## Hard Stop",
            "",
            "Do not promote USA alpha/backtests/portfolio construction/live trading until the USA scorecard is 5/5 with zero critical blockers.",
        ]
    )
    return "\n".join(lines) + "\n"
