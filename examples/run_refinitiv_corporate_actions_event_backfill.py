from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.corporate_actions_pit import (
    build_corporate_action_event_plan,
    execute_corporate_action_event_plan,
    stage_corporate_action_event_plan,
)
from quant_lab.data.refinitiv_pit_extraction import load_pit_plan_artifacts
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Refinitiv corporate-action event backfill.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708")
    parser.add_argument("--existing-plan-csv", default=None)
    parser.add_argument("--expected-plan-sha256", default=None)
    parser.add_argument("--selected-universe-csv", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_corporate_actions_events")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tier", default="u2", choices=["u2", "u3"])
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-plan-rows", type=int, default=0)
    parser.add_argument("--event-families", default=None, help="Optional comma-separated event families.")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-failed-on-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    args = parser.parse_args()

    existing_plan_path = Path(args.existing_plan_csv) if args.existing_plan_csv else None
    if existing_plan_path is not None:
        if not existing_plan_path.is_file() or not args.expected_plan_sha256 or not args.selected_universe_csv:
            raise ValueError(
                "--existing-plan-csv requires an existing file, --expected-plan-sha256, and --selected-universe-csv"
            )
        actual = hashlib.sha256(existing_plan_path.read_bytes()).hexdigest()
        if actual != args.expected_plan_sha256.lower():
            raise ValueError(f"immutable corporate-action plan checksum mismatch: {actual}")
        extraction_plan = pd.read_csv(existing_plan_path, low_memory=False)
        selected_universe = pd.read_csv(args.selected_universe_csv, low_memory=False)
        plan_run_ids = extraction_plan.get("run_id", pd.Series(dtype=str)).dropna().astype(str).unique()
        if len(plan_run_ids) != 1:
            raise ValueError("existing corporate-action plan must contain exactly one run_id")
        run_id = args.run_id or str(plan_run_ids[0])
        if str(plan_run_ids[0]) != run_id:
            raise ValueError("--run-id does not match existing corporate-action plan")
    else:
        run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-CORPACT-EVENTS-%Y%m%d-%H%M%S")
        plan_artifacts = load_pit_plan_artifacts(Path(args.plan_dir))
        selected_universe = plan_artifacts.selected_universe
        extraction_plan = build_corporate_action_event_plan(
            selected_universe=selected_universe,
            artifact_root=Path(args.artifact_root),
            run_id=run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            tier=args.tier,
            batch_size=args.batch_size,
            event_families=_parse_csv_arg(args.event_families),
        )
    artifact_dir = Path(args.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stage_corporate_action_event_plan(extraction_plan, artifact_dir / "event_plan.csv")

    if args.dry_run:
        print(f"Run ID: {run_id}")
        print(f"Artifact dir: {artifact_dir}")
        print(f"Planned rows: {len(extraction_plan)}")
        print(extraction_plan.head(10).to_string(index=False))
        return

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifacts = execute_corporate_action_event_plan(
            adapter.ld,
            plan=extraction_plan,
            selected_universe=selected_universe,
            artifact_root=Path(args.artifact_root),
            run_id=run_id,
            max_plan_rows=args.max_plan_rows or None,
            resume=not args.no_resume,
            skip_failed_on_resume=args.skip_failed_on_resume,
            sleep_seconds=args.sleep_seconds,
            session_name=args.session,
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
