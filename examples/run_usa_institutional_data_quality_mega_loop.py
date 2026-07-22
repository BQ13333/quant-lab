from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.vendor_adapters import RefinitivAdapter


SCORECARD_RUN_ID = "USA-DATA-QUALITY-LOOP-20260709"
EXECUTION_LOOP_RUN_ID = "USA-DATA-QUALITY-EXECUTION-LOOP-20260709"
ACTIVE_RUN_ID = "REFINITIV-USA-HISTORICAL-20Y-LARGE-LIQUID-CORE-V1-20260709"
REFERENCE_RUN_ID = "REFINITIV-USA-HISTORICAL-20Y-LISTED-REFERENCE-CORE-V1-20260709"
L2_RUN_ID = "USA-L2-DEEP-LARGE-LIQUID-20260710"
PIT_PLAN_RUN_ID = "PIT-DATA-PLAN-USA-LARGE-LIQUID-20260710"
CORPACT_RUN_ID = "REFINITIV-CORPACT-EVENTS-USA-20260709"
ADJRET_RUN_ID = "ADJRET-RECON-USA-20260709"
LIQUIDITY_RUN_ID = "REFINITIV-PIT-HISTORICAL-LIQUIDITY-USA-U2-FULL-20260711"
FUND_IBES_EVENTS_RUN_ID = "REFINITIV-PIT-HISTORICAL-FUND-IBES-EVENTS-USA-20260709"
PIT_LEAKAGE_RUN_ID = "PIT-LEAKAGE-QA-USA-20260709"
FUND_ROUTE_PROBE_RUN_ID = "REFINITIV-PIT-ROUTE-PROBE-USA-FUND-IBES-EVENTS-20260710"
FUND_HISTORY_MAP_RUN_ID = "REFINITIV-PIT-ROUTE-HISTORY-MAP-USA-FUND-IBES-EVENTS-20260710"

FIELD_CATALOG = "artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv"
ACTIVE_SYMBOL_MAP = (
    "artifacts/refinitiv_symbol_map/"
    "REFINITIV-SYMBOL-MAP-USA_LARGE_LIQUID_CORE_V1-USA-LARGE-LIQUID-UNIVERSE-V1-20260709/symbol_map.csv"
)
REFERENCE_SYMBOL_MAP = (
    "artifacts/refinitiv_symbol_map/"
    "REFINITIV-SYMBOL-MAP-USA_LISTED_REFERENCE_CORE_V1-USA-RESEARCH-UNIVERSE-V1-20260709/symbol_map.csv"
)
LIQUIDITY_HISTORY_MAP_DIR = "artifacts/refinitiv_pit_route_history_map/REFINITIV-PIT-ROUTE-HISTORY-MAP-USA-LIQUIDITY-FULL-20260711"
LOOP_ROOT = "artifacts/usa_institutional_data_quality_mega_loop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a guarded USA institutional data-quality loop across OHLCV, L2 staging, PIT plan, "
            "corporate actions, adjusted returns, liquidity, fundamentals/IBES/events, and leakage QA."
        )
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-cycles", type=int, default=999)
    parser.add_argument("--active-max-plan-rows", type=int, default=25)
    parser.add_argument("--quarantine-retry-max-plan-rows", type=int, default=5)
    parser.add_argument("--reference-max-plan-rows", type=int, default=10)
    parser.add_argument("--pit-max-plan-rows", type=int, default=4)
    parser.add_argument("--corpact-max-plan-rows", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--pit-batch-size", type=int, default=25)
    parser.add_argument("--sleep-between-cycles", type=float, default=10.0)
    parser.add_argument("--sleep-on-throttle", type=float, default=900.0)
    parser.add_argument("--run-tests-every", type=int, default=3)
    parser.add_argument("--refinitiv-session", default="desktop.workspace")
    parser.add_argument("--health-ric", default="IBM.N")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--pit-start-date", default="2021-01-01")
    parser.add_argument("--stop-after-score-accepted", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loop_run_id = args.run_id or datetime.now(timezone.utc).strftime("USA-INSTITUTIONAL-DQ-MEGA-LOOP-%Y%m%d-%H%M%S")
    artifact_dir = Path(LOOP_ROOT) / loop_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    events_path = artifact_dir / "events.jsonl"
    print(f"loop_run_id={loop_run_id}")
    print(f"artifact_dir={artifact_dir.resolve()}")

    for cycle in range(1, args.max_cycles + 1):
        score_before = _run_scorecard(events_path, cycle, "scorecard_before")
        state_before = _lake_state()
        _event(events_path, cycle, "cycle_start", {"scorecard": score_before, "state": state_before})
        print(f"\ncycle={cycle} state_before={state_before}")
        print(f"score_before={_compact_score(score_before)}")

        if score_before.get("accepted"):
            print("usa_scorecard_accepted=true")
            break

        phase = _choose_phase()
        print(f"selected_phase={phase}")
        _event(events_path, cycle, "selected_phase", {"phase": phase})

        live_phase = phase in {
            "active_ohlcv",
            "active_quarantine_retry",
            "reference_ohlcv",
            "corporate_actions",
            "liquidity_pit_backfill",
            "fund_route_probe",
            "fund_ibes_events_backfill",
        }
        if live_phase:
            health_ok, health_error = _health_check(args.refinitiv_session, args.health_ric)
            _event(events_path, cycle, "health_check", {"ok": health_ok, "error": health_error})
            print(f"health_ok={health_ok}")
            if not health_ok:
                _run_execution_loop("throttled")
                print(f"Refinitiv unhealthy/throttled: {health_error}. Sleeping {args.sleep_on_throttle:.0f}s.")
                time.sleep(args.sleep_on_throttle)
                continue
            _run_execution_loop("healthy")

        if phase == "active_ohlcv":
            _run_active_ohlcv(args, events_path, cycle)
        elif phase == "active_quarantine_retry":
            _run_active_quarantine_retry(args, events_path, cycle)
        elif phase == "build_l2":
            _run_l2_staging(args, events_path, cycle)
        elif phase == "build_pit_plan":
            _run_pit_plan(args, events_path, cycle)
        elif phase == "reference_ohlcv":
            _run_reference_ohlcv(args, events_path, cycle)
        elif phase == "corporate_actions":
            _run_corporate_actions(args, events_path, cycle)
        elif phase == "adjusted_returns":
            _run_adjusted_returns(events_path, cycle)
        elif phase == "liquidity_pit_backfill":
            _run_liquidity_pit(args, events_path, cycle)
        elif phase == "fund_route_probe":
            _run_fund_route_probe(args, events_path, cycle)
        elif phase == "fund_ibes_events_backfill":
            _run_fund_ibes_events(args, events_path, cycle)
        elif phase == "leakage_qa":
            _run_leakage_qa(args, events_path, cycle)
        elif phase == "pit_index_constituents_blocked":
            _event(
                events_path,
                cycle,
                "pit_index_constituents_blocked",
                {
                    "reason": (
                        "No accepted official PIT Russell/S&P/Nasdaq/Dow constituent route exists yet. "
                        "Loop will not create a fake pass artifact."
                    )
                },
            )
            print("official_pit_index_constituents_missing=true")
            time.sleep(args.sleep_between_cycles)
        else:
            print("no_actionable_phase=true")
            time.sleep(args.sleep_between_cycles)

        score_after = _run_scorecard(events_path, cycle, "scorecard_after")
        state_after = _lake_state()
        if args.run_tests_every > 0 and cycle % args.run_tests_every == 0:
            _run_tests(events_path, cycle)
        _run_execution_loop("healthy" if live_phase else "unknown")
        _event(events_path, cycle, "cycle_complete", {"scorecard": score_after, "state": state_after})
        print(f"cycle={cycle} state_after={state_after}")
        print(f"score_after={_compact_score(score_after)}")

        if args.stop_after_score_accepted and score_after.get("accepted"):
            print("usa_scorecard_accepted=true")
            break
        time.sleep(args.sleep_between_cycles)


def _choose_phase() -> str:
    active = _historical_state(ACTIVE_RUN_ID)
    if active["planned_rows"] == 0 or active["usable_rows"] < active["effective_planned_rows"]:
        return "active_ohlcv"
    if active["quarantined_rows"] > 0 and active["usable_rows"] < active["planned_rows"]:
        return "active_quarantine_retry"
    if not _l2_dir().joinpath("manifest.json").exists():
        return "build_l2"
    if not _pit_plan_dir().joinpath("manifest.json").exists():
        return "build_pit_plan"
    if not _corporate_actions_manifest_complete(_corpact_dir() / "manifest.json"):
        return "corporate_actions"
    if not _adjret_dir().joinpath("manifest.json").exists():
        return "adjusted_returns"
    if not _manifest_complete(_liquidity_dir() / "manifest.json"):
        return "liquidity_pit_backfill"
    fund_map = _fund_history_map_dir() / "history_field_map.csv"
    if not fund_map.exists() or _read_csv(fund_map).empty:
        return "fund_route_probe"
    if not _manifest_complete(_fund_dir() / "manifest.json"):
        return "fund_ibes_events_backfill"
    if not _leakage_dir().joinpath("manifest.json").exists():
        return "leakage_qa"
    if not _manifest_scorecard_eligible(_pit_index_dir() / "manifest.json"):
        return "pit_index_constituents_blocked"
    return "scorecard_only"


def _run_active_ohlcv(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run_historical_backfill(
        run_id=ACTIVE_RUN_ID,
        symbol_map=ACTIVE_SYMBOL_MAP,
        max_plan_rows=args.active_max_plan_rows,
        batch_size=args.batch_size,
        start_date=args.start_date,
        end_date=args.end_date,
        session=args.refinitiv_session,
        events_path=events_path,
        cycle=cycle,
        phase="active_ohlcv",
    )
    _run_depth_and_compaction(ACTIVE_RUN_ID, events_path, cycle, "active")


def _run_active_quarantine_retry(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run_historical_backfill(
        run_id=ACTIVE_RUN_ID,
        symbol_map=ACTIVE_SYMBOL_MAP,
        max_plan_rows=args.quarantine_retry_max_plan_rows,
        batch_size=args.batch_size,
        start_date=args.start_date,
        end_date=args.end_date,
        session=args.refinitiv_session,
        events_path=events_path,
        cycle=cycle,
        phase="active_quarantine_retry",
        retry_quarantine_only=True,
    )
    _run_depth_and_compaction(ACTIVE_RUN_ID, events_path, cycle, "active_quarantine_retry")


def _run_reference_ohlcv(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run_historical_backfill(
        run_id=REFERENCE_RUN_ID,
        symbol_map=REFERENCE_SYMBOL_MAP,
        max_plan_rows=args.reference_max_plan_rows,
        batch_size=args.batch_size,
        start_date=args.start_date,
        end_date=args.end_date,
        session=args.refinitiv_session,
        events_path=events_path,
        cycle=cycle,
        phase="reference_ohlcv",
    )
    _run_depth_and_compaction(REFERENCE_RUN_ID, events_path, cycle, "reference")


def _run_historical_backfill(
    *,
    run_id: str,
    symbol_map: str,
    max_plan_rows: int,
    batch_size: int,
    start_date: str,
    end_date: str,
    session: str,
    events_path: Path,
    cycle: int,
    phase: str,
    retry_quarantine_only: bool = False,
) -> None:
    quarantine = f"artifacts/historical_backfill/{run_id}/quarantined_plan_rows.csv"
    command = [
        "uv",
        "run",
        "--extra",
        "refinitiv",
        "python",
        "examples/run_refinitiv_historical_backfill.py",
        "--run-id",
        run_id,
        "--symbol-map-csv",
        symbol_map,
        "--field-catalog-csv",
        FIELD_CATALOG,
        "--include-countries",
        "United States",
        "--start-date",
        start_date,
        "--end-date",
        end_date,
        "--window-years",
        "5",
        "--batch-size",
        str(batch_size),
        "--max-plan-rows",
        str(max_plan_rows),
        "--interval",
        "1d",
        "--refinitiv-session",
        session,
    ]
    if retry_quarantine_only:
        command.extend(["--include-plan-row-ids-csv", quarantine])
    else:
        command.extend(["--exclude-plan-row-ids-csv", quarantine])
    _run(command, events_path, cycle, phase)


def _run_depth_and_compaction(run_id: str, events_path: Path, cycle: int, prefix: str) -> None:
    manifest = f"artifacts/historical_backfill/{run_id}/historical_manifest.csv"
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_data_depth_report.py",
            f"artifacts/data_lake/prices_long/{run_id}/ohlcv",
            "--run-id",
            f"{run_id}-DEPTH",
            "--extraction-manifest-path",
            manifest,
        ],
        events_path,
        cycle,
        f"{prefix}_depth_report",
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_data_lake_partition_ops.py",
            f"artifacts/data_lake/prices_long/{run_id}/ohlcv",
            "--run-id",
            f"{run_id}-COMPACT",
            "--compact",
            "--overwrite",
        ],
        events_path,
        cycle,
        f"{prefix}_compaction",
    )


def _run_l2_staging(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_l2_deep_extraction.py",
            f"artifacts/data_lake/prices_long/{ACTIVE_RUN_ID}/ohlcv_compacted",
            "--run-id",
            L2_RUN_ID,
            "--include-countries",
            "United States",
            "--min-rows",
            "2500",
            "--max-missing-ratio",
            "0.25",
            "--top-liquid-per-country",
            "500",
            "--symbol-map-path",
            ACTIVE_SYMBOL_MAP,
            "--field-catalog-path",
            FIELD_CATALOG,
            "--historical-manifest-path",
            f"artifacts/historical_backfill/{ACTIVE_RUN_ID}/historical_manifest.csv",
        ],
        events_path,
        cycle,
        "build_l2",
    )


def _run_pit_plan(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_pit_data_acquisition_plan.py",
            "--l2-artifact-dir",
            str(_l2_dir()),
            "--run-id",
            PIT_PLAN_RUN_ID,
            "--symbol-map-path",
            ACTIVE_SYMBOL_MAP,
            "--active-countries",
            "United States",
            "--tiers",
            "u2,u3",
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--batch-size",
            str(args.pit_batch_size),
            "--lseg-connector-status",
            args.refinitiv_session,
        ],
        events_path,
        cycle,
        "build_pit_plan",
    )


def _run_corporate_actions(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "--extra",
            "refinitiv",
            "python",
            "examples/run_refinitiv_corporate_actions_event_backfill.py",
            "--plan-dir",
            str(_pit_plan_dir()),
            "--run-id",
            CORPACT_RUN_ID,
            "--tier",
            "u2",
            "--start-date",
            args.start_date,
            "--end-date",
            args.end_date,
            "--batch-size",
            str(args.pit_batch_size),
            "--max-plan-rows",
            str(args.corpact_max_plan_rows),
            "--session",
            args.refinitiv_session,
            "--skip-failed-on-resume",
        ],
        events_path,
        cycle,
        "corporate_actions",
    )


def _run_adjusted_returns(events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_adjusted_return_reconstruction.py",
            "--daily-features",
            str(_l2_dir() / "daily_features.parquet"),
            "--corporate-actions-dir",
            str(_corpact_dir()),
            "--run-id",
            ADJRET_RUN_ID,
            "--min-matched-event-share",
            "0.95",
            "--max-quarantine-share",
            "0.02",
        ],
        events_path,
        cycle,
        "adjusted_returns",
    )


def _run_liquidity_pit(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "--extra",
            "refinitiv",
            "python",
            "examples/run_refinitiv_pit_historical_backfill.py",
            "--plan-dir",
            str(_pit_plan_dir()),
            "--historical-probe-dir",
            LIQUIDITY_HISTORY_MAP_DIR,
            "--run-id",
            LIQUIDITY_RUN_ID,
            "--tier",
            "u2",
            "--start-date",
            args.pit_start_date,
            "--end-date",
            args.end_date,
            "--interval",
            "1M",
            "--batch-size",
            str(args.pit_batch_size),
            "--window-years",
            "5",
            "--max-plan-rows",
            str(args.pit_max_plan_rows),
            "--datasets",
            "liquidity_market_cap_free_float",
            "--session",
            args.refinitiv_session,
            "--skip-failed-on-resume",
        ],
        events_path,
        cycle,
        "liquidity_pit_backfill",
    )


def _run_fund_route_probe(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "--extra",
            "refinitiv",
            "python",
            "examples/run_refinitiv_pit_route_probe.py",
            "--plan-dir",
            str(_pit_plan_dir()),
            "--run-id",
            FUND_ROUTE_PROBE_RUN_ID,
            "--tier",
            "u3",
            "--sample-per-country",
            "2",
            "--start-date",
            args.pit_start_date,
            "--end-date",
            args.end_date,
            "--interval",
            "1M",
            "--datasets",
            "fundamentals_pit,ibes_estimates_revisions,earnings_events_actuals",
            "--routes",
            "get_data,get_history",
            "--session",
            args.refinitiv_session,
        ],
        events_path,
        cycle,
        "fund_route_probe",
    )
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_refinitiv_pit_promote_route_history_map.py",
            "--route-probe-dir",
            f"artifacts/refinitiv_pit_route_probe/{FUND_ROUTE_PROBE_RUN_ID}",
            "--run-id",
            FUND_HISTORY_MAP_RUN_ID,
            "--datasets",
            "fundamentals_pit,ibes_estimates_revisions,earnings_events_actuals",
            "--min-inside-rows",
            "1",
            "--min-coverage",
            "0.0",
        ],
        events_path,
        cycle,
        "fund_history_map",
    )


def _run_fund_ibes_events(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "--extra",
            "refinitiv",
            "python",
            "examples/run_refinitiv_pit_historical_backfill.py",
            "--plan-dir",
            str(_pit_plan_dir()),
            "--historical-probe-dir",
            str(_fund_history_map_dir()),
            "--run-id",
            FUND_IBES_EVENTS_RUN_ID,
            "--tier",
            "u2",
            "--start-date",
            args.pit_start_date,
            "--end-date",
            args.end_date,
            "--interval",
            "1M",
            "--batch-size",
            str(args.pit_batch_size),
            "--window-years",
            "5",
            "--max-plan-rows",
            str(args.pit_max_plan_rows),
            "--datasets",
            "fundamentals_pit,ibes_estimates_revisions,earnings_events_actuals",
            "--session",
            args.refinitiv_session,
            "--skip-failed-on-resume",
        ],
        events_path,
        cycle,
        "fund_ibes_events_backfill",
    )


def _run_leakage_qa(args: argparse.Namespace, events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_pit_leakage_qa_report.py",
            "--history-roots",
            f"{_liquidity_dir()},{_fund_dir()}",
            "--l2-artifact-dir",
            str(_l2_dir()),
            "--run-id",
            PIT_LEAKAGE_RUN_ID,
            "--universe-tier",
            "u2",
            "--include-countries",
            "United States",
            "--price-col",
            "adj_close",
            "--start-date",
            args.pit_start_date,
            "--end-date",
            args.end_date,
        ],
        events_path,
        cycle,
        "leakage_qa",
    )


def _run_scorecard(events_path: Path, cycle: int, phase: str) -> dict[str, Any]:
    _run(
        ["uv", "run", "python", "examples/run_usa_data_quality_loop.py", "--run-id", SCORECARD_RUN_ID],
        events_path,
        cycle,
        phase,
    )
    return _read_json(Path("artifacts/institutional_data_quality") / SCORECARD_RUN_ID / "manifest.json")


def _run_execution_loop(vendor_health_status: str) -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "examples/run_usa_data_quality_execution_loop.py",
            "--run-id",
            EXECUTION_LOOP_RUN_ID,
            "--vendor-health-status",
            vendor_health_status,
            "--min-market-cap-usd",
            "1000000000",
            "--min-adv-value-usd",
            "5000000",
            "--min-free-float-pct",
            "10",
        ],
        check=True,
    )


def _run_tests(events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "pytest",
            "tests/test_historical_backfill.py",
            "tests/test_usa_research_universe.py",
            "tests/test_usa_data_quality_execution_loop.py",
            "tests/test_usa_data_quality_loop.py",
            "tests/test_corporate_actions_pit.py",
            "tests/test_adjusted_returns.py",
            "tests/test_refinitiv_pit_extraction.py",
            "tests/test_pit_leakage_qa.py",
        ],
        events_path,
        cycle,
        "focused_tests",
    )


def _health_check(session_name: str, health_ric: str) -> tuple[bool, str]:
    adapter = RefinitivAdapter(session_name=session_name, health_ric=health_ric)
    try:
        adapter.open()
        ok = bool(adapter.health_check())
        return ok, "" if ok else getattr(adapter, "last_health_error", "")
    except Exception as exc:  # pragma: no cover - live vendor path
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            adapter.close()
        except Exception:
            pass


def _run(command: list[str], events_path: Path, cycle: int, phase: str) -> None:
    started = datetime.now(timezone.utc).isoformat()
    _event(events_path, cycle, f"{phase}_start", {"command": command, "started_at": started})
    print(f"phase={phase} command={' '.join(command)}")
    result = subprocess.run(command, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    payload = {"command": command, "returncode": result.returncode, "started_at": started, "finished_at": finished}
    _event(events_path, cycle, f"{phase}_complete", payload)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def _lake_state() -> dict[str, Any]:
    return {
        "active": _historical_state(ACTIVE_RUN_ID),
        "reference": _historical_state(REFERENCE_RUN_ID),
        "l2_exists": (_l2_dir() / "manifest.json").exists(),
        "pit_plan_exists": (_pit_plan_dir() / "manifest.json").exists(),
        "corpact": _manifest_state(_corpact_dir() / "manifest.json"),
        "adjusted_returns_exists": (_adjret_dir() / "manifest.json").exists(),
        "liquidity": _manifest_state(_liquidity_dir() / "manifest.json"),
        "fund_ibes_events": _manifest_state(_fund_dir() / "manifest.json"),
        "fund_history_map_rows": int(len(_read_csv(_fund_history_map_dir() / "history_field_map.csv"))),
        "leakage_exists": (_leakage_dir() / "manifest.json").exists(),
        "pit_index_exists": (_pit_index_dir() / "manifest.json").exists(),
    }


def _historical_state(run_id: str) -> dict[str, Any]:
    root = Path("artifacts/historical_backfill") / run_id
    run_manifest = _read_json(root / "historical_run_manifest.json")
    plan = _read_csv(root / "historical_plan.csv")
    historical = _read_csv(root / "historical_manifest.csv")
    quarantine = _read_csv(root / "quarantined_plan_rows.csv")
    planned_rows = int(run_manifest.get("planned_rows", len(plan)))
    quarantined_rows = _quarantined_rows(quarantine)
    effective_planned_rows = max(planned_rows - quarantined_rows, 0)
    if historical.empty:
        return {
            "planned_rows": planned_rows,
            "quarantined_rows": quarantined_rows,
            "effective_planned_rows": effective_planned_rows,
            "executed_rows": 0,
            "usable_rows": 0,
            "zero_output_rows": 0,
            "failed_empty_rows": 0,
            "long_ohlcv_rows": 0,
        }
    status = historical["status"].astype(str)
    structural = status.isin(["completed_prelisting", "completed_postretirement"])
    completed = status.isin(["completed", "completed_partial"])
    price_columns = pd.to_numeric(historical.get("price_columns", pd.Series(0, index=historical.index)), errors="coerce").fillna(0)
    long_rows = pd.to_numeric(historical.get("long_ohlcv_rows", pd.Series(0, index=historical.index)), errors="coerce").fillna(0)
    usable = (completed & price_columns.gt(0) & long_rows.gt(0)) | structural
    zero = completed & (price_columns.le(0) | long_rows.le(0))
    failed_empty = historical["status"].astype(str).eq("failed_empty")
    return {
        "planned_rows": planned_rows,
        "quarantined_rows": quarantined_rows,
        "effective_planned_rows": effective_planned_rows,
        "executed_rows": int(len(historical)),
        "usable_rows": int(usable.sum()),
        "zero_output_rows": int(zero.sum()),
        "failed_empty_rows": int(failed_empty.sum()),
        "long_ohlcv_rows": int(long_rows.sum()),
        "status_counts": historical["status"].value_counts(dropna=False).to_dict(),
    }


def _quarantined_rows(quarantine: pd.DataFrame) -> int:
    if quarantine.empty or "plan_row_id" not in quarantine.columns:
        return 0
    values = quarantine["plan_row_id"].dropna().astype(str)
    return int(values.loc[values.ne("")].nunique())


def _manifest_state(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    if not manifest:
        return {"exists": False}
    return {
        "exists": True,
        "planned_rows": manifest.get("planned_rows"),
        "completed_rows": manifest.get("completed_rows"),
        "failed_rows": manifest.get("failed_rows"),
        "history_rows": manifest.get("history_rows"),
        "status": manifest.get("status"),
    }


def _manifest_complete(path: Path) -> bool:
    manifest = _read_json(path)
    if not manifest:
        return False
    if "planned_rows" in manifest and "completed_rows" in manifest:
        planned = int(manifest.get("planned_rows") or 0)
        completed = int(manifest.get("completed_rows") or 0)
        failed = int(manifest.get("failed_rows") or 0)
        history_rows = int(manifest.get("history_rows") or 0)
        return planned > 0 and completed >= planned and failed == 0 and history_rows > 0
    return bool(manifest.get("status") == "pass")


def _corporate_actions_manifest_complete(path: Path) -> bool:
    manifest = _read_json(path)
    if not manifest or manifest.get("status") != "complete":
        return False
    planned = int(manifest.get("planned_rows") or 0)
    completed = int(manifest.get("completed_rows") or 0)
    failed = int(manifest.get("failed_rows") or 0)
    alpha_usable = int(manifest.get("alpha_usable_rows") or 0)
    return planned > 0 and completed >= planned and failed == 0 and alpha_usable > 0


def _manifest_scorecard_eligible(path: Path) -> bool:
    manifest = _read_json(path)
    return bool(
        manifest
        and manifest.get("status") in {"complete", "pass"}
        and manifest.get("scorecard_eligible") is True
    )


def _compact_score(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "overall_score": score.get("overall_score"),
        "accepted": score.get("accepted"),
        "critical_blockers": score.get("critical_blockers", [])[:5],
        "critical_blocker_count": len(score.get("critical_blockers", [])),
    }


def _l2_dir() -> Path:
    return Path("artifacts/l2_deep_extraction") / L2_RUN_ID


def _pit_plan_dir() -> Path:
    return Path("artifacts/pit_data_plan") / PIT_PLAN_RUN_ID


def _corpact_dir() -> Path:
    return Path("artifacts/refinitiv_corporate_actions_events") / CORPACT_RUN_ID


def _adjret_dir() -> Path:
    return Path("artifacts/adjusted_return_reconstruction") / ADJRET_RUN_ID


def _liquidity_dir() -> Path:
    return Path("artifacts/refinitiv_pit_historical_backfill") / LIQUIDITY_RUN_ID


def _fund_dir() -> Path:
    return Path("artifacts/refinitiv_pit_historical_backfill") / FUND_IBES_EVENTS_RUN_ID


def _fund_history_map_dir() -> Path:
    return Path("artifacts/refinitiv_pit_route_history_map") / FUND_HISTORY_MAP_RUN_ID


def _leakage_dir() -> Path:
    return Path("artifacts/pit_leakage_qa") / PIT_LEAKAGE_RUN_ID


def _pit_index_dir() -> Path:
    return Path(
        "artifacts/index_constituents/USA-PIT-INDEX-CONSTITUENTS-NO-RUSSELL-V1-20260711"
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


def _event(path: Path, cycle: int, event: str, payload: dict[str, Any]) -> None:
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "cycle": cycle,
        "event": event,
        "payload": payload,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    main()
