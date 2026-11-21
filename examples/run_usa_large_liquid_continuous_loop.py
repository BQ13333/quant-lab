from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.vendor_adapters import RefinitivAdapter


RUN_ID = "REFINITIV-USA-HISTORICAL-20Y-LARGE-LIQUID-CORE-V1-20260709"
EXECUTION_LOOP_RUN_ID = "USA-DATA-QUALITY-EXECUTION-LOOP-20260709"
SCORECARD_RUN_ID = "USA-DATA-QUALITY-LOOP-20260709"
SYMBOL_MAP = (
    "artifacts/refinitiv_symbol_map/"
    "REFINITIV-SYMBOL-MAP-USA_LARGE_LIQUID_CORE_V1-USA-LARGE-LIQUID-UNIVERSE-V1-20260709/symbol_map.csv"
)
FIELD_CATALOG = "artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv"
QUARANTINE = f"artifacts/historical_backfill/{RUN_ID}/quarantined_plan_rows.csv"
MANIFEST = f"artifacts/historical_backfill/{RUN_ID}/historical_manifest.csv"
RUN_MANIFEST = f"artifacts/historical_backfill/{RUN_ID}/historical_run_manifest.json"
LOOP_ROOT = "artifacts/usa_large_liquid_continuous_loop"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuously run guarded USA large-liquid OHLCV extraction cycles.")
    parser.add_argument("--max-plan-rows-per-cycle", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-cycles", type=int, default=999)
    parser.add_argument("--sleep-between-cycles", type=float, default=10.0)
    parser.add_argument("--sleep-on-throttle", type=float, default=900.0)
    parser.add_argument("--run-tests-every", type=int, default=1)
    parser.add_argument("--refinitiv-session", default="desktop.workspace")
    parser.add_argument("--health-ric", default="IBM.N")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--stop-after-score-accepted", action="store_true")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loop_run_id = args.run_id or datetime.now(timezone.utc).strftime("USA-LARGE-LIQUID-CONTINUOUS-%Y%m%d-%H%M%S")
    artifact_dir = Path(LOOP_ROOT) / loop_run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    events_path = artifact_dir / "events.jsonl"
    print(f"loop_run_id={loop_run_id}")
    print(f"artifact_dir={artifact_dir.resolve()}")

    for cycle in range(1, args.max_cycles + 1):
        before = _manifest_state()
        _event(events_path, cycle, "cycle_start", before)
        print(f"\ncycle={cycle} before={before}")

        health_ok, health_error = _health_check(args.refinitiv_session, args.health_ric)
        _event(events_path, cycle, "health_check", {"ok": health_ok, "error": health_error})
        print(f"health_ok={health_ok}")
        if not health_ok:
            _run_execution_loop("throttled")
            _event(events_path, cycle, "throttled_sleep", {"seconds": args.sleep_on_throttle, "error": health_error})
            print(f"Refinitiv unhealthy/throttled: {health_error}. Sleeping {args.sleep_on_throttle:.0f}s.")
            time.sleep(args.sleep_on_throttle)
            continue

        _run_execution_loop("healthy")
        _run(
            [
                "uv",
                "run",
                "--extra",
                "refinitiv",
                "python",
                "examples/run_refinitiv_historical_backfill.py",
                "--run-id",
                RUN_ID,
                "--symbol-map-csv",
                SYMBOL_MAP,
                "--field-catalog-csv",
                FIELD_CATALOG,
                "--include-countries",
                "United States",
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--window-years",
                "5",
                "--batch-size",
                str(args.batch_size),
                "--max-plan-rows",
                str(args.max_plan_rows_per_cycle),
                "--interval",
                "1d",
                "--refinitiv-session",
                args.refinitiv_session,
                "--exclude-plan-row-ids-csv",
                QUARANTINE,
            ],
            events_path,
            cycle,
            "historical_backfill",
        )
        _run_depth(events_path, cycle)
        _run_compaction(events_path, cycle)
        score = _run_scorecard(events_path, cycle)
        if args.run_tests_every > 0 and cycle % args.run_tests_every == 0:
            _run_tests(events_path, cycle)
        _run_execution_loop("healthy")

        after = _manifest_state()
        _event(events_path, cycle, "cycle_complete", {"before": before, "after": after, "scorecard": score})
        print(f"cycle={cycle} after={after} scorecard={score}")

        if after.get("usable_rows", 0) >= after.get("effective_planned_rows", after.get("planned_rows", 10**9)):
            print("large_liquid_ohlcv_complete=true")
            break
        if args.stop_after_score_accepted and score.get("accepted"):
            print("usa_scorecard_accepted=true")
            break
        time.sleep(args.sleep_between_cycles)


def _health_check(session_name: str, health_ric: str) -> tuple[bool, str]:
    adapter = RefinitivAdapter(session_name=session_name, health_ric=health_ric)
    try:
        adapter.open()
        ok = bool(adapter.health_check())
        return ok, "" if ok else getattr(adapter, "last_health_error", "")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            adapter.close()
        except Exception:
            pass


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


def _run_depth(events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_data_depth_report.py",
            f"artifacts/data_lake/prices_long/{RUN_ID}/ohlcv",
            "--run-id",
            f"{RUN_ID}-DEPTH",
            "--extraction-manifest-path",
            MANIFEST,
        ],
        events_path,
        cycle,
        "depth_report",
    )


def _run_compaction(events_path: Path, cycle: int) -> None:
    _run(
        [
            "uv",
            "run",
            "python",
            "examples/run_data_lake_partition_ops.py",
            f"artifacts/data_lake/prices_long/{RUN_ID}/ohlcv",
            "--run-id",
            f"{RUN_ID}-COMPACT",
            "--compact",
            "--overwrite",
        ],
        events_path,
        cycle,
        "compaction",
    )


def _run_scorecard(events_path: Path, cycle: int) -> dict[str, object]:
    _run(
        ["uv", "run", "python", "examples/run_usa_data_quality_loop.py", "--run-id", SCORECARD_RUN_ID],
        events_path,
        cycle,
        "scorecard",
    )
    path = Path("artifacts/institutional_data_quality") / SCORECARD_RUN_ID / "manifest.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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
        ],
        events_path,
        cycle,
        "focused_tests",
    )


def _run(command: list[str], events_path: Path, cycle: int, phase: str) -> None:
    started = datetime.now(timezone.utc).isoformat()
    _event(events_path, cycle, f"{phase}_start", {"command": command, "started_at": started})
    print(f"phase={phase} command={' '.join(command)}")
    result = subprocess.run(command, check=False)
    finished = datetime.now(timezone.utc).isoformat()
    _event(
        events_path,
        cycle,
        f"{phase}_complete",
        {"command": command, "returncode": result.returncode, "started_at": started, "finished_at": finished},
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, command)


def _manifest_state() -> dict[str, object]:
    run_manifest = _read_json(Path(RUN_MANIFEST))
    historical = _read_csv(Path(MANIFEST))
    quarantined = _read_csv(Path(QUARANTINE))
    excluded_rows = _quarantined_rows(quarantined)
    if historical.empty:
        planned_rows = int(run_manifest.get("planned_rows", 0))
        return {
            "planned_rows": planned_rows,
            "excluded_rows": excluded_rows,
            "effective_planned_rows": max(planned_rows - excluded_rows, 0),
            "executed_rows": int(run_manifest.get("executed_rows", 0)),
            "usable_rows": 0,
            "zero_output_rows": 0,
            "long_ohlcv_rows": 0,
            "failures": 0,
        }
    status = historical["status"].astype(str)
    structural = status.isin(["completed_prelisting", "completed_postretirement"])
    completed = status.isin(["completed", "completed_partial"])
    price_columns = pd.to_numeric(historical.get("price_columns", pd.Series(0, index=historical.index)), errors="coerce").fillna(0)
    long_rows = pd.to_numeric(historical.get("long_ohlcv_rows", pd.Series(0, index=historical.index)), errors="coerce").fillna(0)
    usable = (completed & price_columns.gt(0) & long_rows.gt(0)) | structural
    zero = completed & (price_columns.le(0) | long_rows.le(0))
    failures = pd.to_numeric(historical.get("failures", pd.Series(0, index=historical.index)), errors="coerce").fillna(0)
    planned_rows = int(run_manifest.get("planned_rows", len(_read_csv(Path(f"artifacts/historical_backfill/{RUN_ID}/historical_plan.csv")))))
    return {
        "planned_rows": planned_rows,
        "excluded_rows": excluded_rows,
        "effective_planned_rows": max(planned_rows - excluded_rows, 0),
        "executed_rows": int(len(historical)),
        "usable_rows": int(usable.sum()),
        "zero_output_rows": int(zero.sum()),
        "long_ohlcv_rows": int(long_rows.sum()),
        "failures": int(failures.sum()),
        "status_counts": historical["status"].value_counts(dropna=False).to_dict(),
    }


def _quarantined_rows(quarantine: pd.DataFrame) -> int:
    if quarantine.empty or "plan_row_id" not in quarantine.columns:
        return 0
    return int(quarantine["plan_row_id"].dropna().astype(str).loc[lambda values: values.ne("")].nunique())


def _read_json(path: Path) -> dict[str, object]:
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


def _event(path: Path, cycle: int, event: str, payload: dict[str, object]) -> None:
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
