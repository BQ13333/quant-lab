from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_pit_extraction import (
    build_refinitiv_pit_historical_plan,
    execute_refinitiv_pit_historical_plan,
    load_pit_plan_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def _filter_field_roles(history_field_map: pd.DataFrame, field_roles: list[str] | None) -> pd.DataFrame:
    if not field_roles:
        return history_field_map
    if "field_role" not in history_field_map.columns:
        raise ValueError("history field map missing `field_role`")
    wanted = {role.strip() for role in field_roles if role.strip()}
    return history_field_map[history_field_map["field_role"].astype(str).isin(wanted)].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batched Refinitiv PIT historical extraction.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708")
    parser.add_argument("--historical-probe-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_pit_historical_backfill")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tier", default="u2", choices=["u2", "u3"])
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--interval", default="1M")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--window-years", type=int, default=5)
    parser.add_argument("--max-plan-rows", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--skip-failed-on-resume",
        action="store_true",
        help="When resuming, do not retry rows already marked failed in historical_manifest.csv.",
    )
    parser.add_argument(
        "--datasets",
        default="liquidity_market_cap_free_float",
        help="Comma-separated dataset ids from history_field_map.csv.",
    )
    parser.add_argument(
        "--field-roles",
        default=None,
        help="Optional comma-separated field roles from history_field_map.csv.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=60.0,
        help="Maximum wall-clock seconds for one synchronous Refinitiv history request; use 0 to disable.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-PIT-HISTORICAL-BACKFILL-%Y%m%d-%H%M%S")
    datasets = _parse_csv_arg(args.datasets)
    field_roles = _parse_csv_arg(args.field_roles)
    plan_artifacts = load_pit_plan_artifacts(Path(args.plan_dir))
    history_map_path = Path(args.historical_probe_dir) / "history_field_map.csv"
    if not history_map_path.exists():
        raise FileNotFoundError(f"history field map not found: {history_map_path}")
    history_field_map = pd.read_csv(history_map_path, low_memory=False)
    history_field_map = _filter_field_roles(history_field_map, field_roles)

    extraction_plan = build_refinitiv_pit_historical_plan(
        selected_universe=plan_artifacts.selected_universe,
        history_field_map=history_field_map,
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        tier=args.tier,
        datasets=datasets,
        batch_size=args.batch_size,
        window_years=args.window_years,
        interval=args.interval,
    )

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifacts = execute_refinitiv_pit_historical_plan(
            adapter.ld,
            plan=extraction_plan,
            selected_universe=plan_artifacts.selected_universe,
            artifact_root=Path(args.artifact_root),
            run_id=run_id,
            max_plan_rows=args.max_plan_rows or None,
            resume=not args.no_resume,
            skip_failed_on_resume=args.skip_failed_on_resume,
            sleep_seconds=args.sleep_seconds,
            session_name=args.session,
            request_timeout_seconds=args.request_timeout_seconds or None,
        )
    finally:
        adapter.close()

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Planned rows: {len(artifacts.plan)}")
    print(f"Executed manifest rows: {len(artifacts.manifest)}")
    print(f"Quality: {artifacts.quality_path}")
    print(f"Errors: {len(artifacts.errors)}")
    if not artifacts.quality.empty:
        print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
