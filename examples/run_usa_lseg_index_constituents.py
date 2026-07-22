from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from quant_lab.data.usa_index_constituents_lseg import (
    build_lseg_usa_index_constituent_plan,
    execute_lseg_usa_index_constituents,
)
from quant_lab.data.usa_index_constituents_pit import (
    USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
    USA_INDEX_SCOPE_PROFILE_FULL,
    resolve_usa_index_scope_profile,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract USA index constituents from a current LSEG chain and reverse Joiner/Leaver history."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/lseg_usa_index_constituents")
    parser.add_argument("--start-date", default="2006-01-03")
    parser.add_argument("--end-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--chunk-years", type=int, default=1)
    parser.add_argument("--max-requests", type=int, default=10)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument(
        "--scope-profile",
        type=lambda value: resolve_usa_index_scope_profile(value).name,
        choices=(USA_INDEX_SCOPE_PROFILE_FULL, USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL),
        default=USA_INDEX_SCOPE_PROFILE_FULL,
        help="Index scope contract; backtest_no_russell requires exactly S&P 500, Nasdaq-100, and Dow.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("LSEG-USA-INDEX-PIT-%Y%m%d-%H%M%S")
    if args.dry_run:
        plan = build_lseg_usa_index_constituent_plan(
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_years=args.chunk_years,
            scope_profile=args.scope_profile,
        )
        artifact_dir = Path(args.artifact_root) / run_id
        artifact_dir.mkdir(parents=True, exist_ok=True)
        plan.to_csv(artifact_dir / "request_plan.csv", index=False)
        print(f"Run ID: {run_id}")
        print(f"Artifact dir: {artifact_dir.resolve()}")
        print(f"Planned requests: {len(plan)}")
        print(f"Scope profile: {args.scope_profile}")
        print(f"Blocked Russell requests: {(~plan['live_request_allowed']).sum()}")
        return

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifacts = execute_lseg_usa_index_constituents(
            client=adapter.ld,
            artifact_root=Path(args.artifact_root),
            run_id=run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            chunk_years=args.chunk_years,
            max_requests=args.max_requests,
            resume=not args.no_resume,
            sleep_seconds=args.sleep_seconds,
            session_name=args.session,
            scope_profile=args.scope_profile,
        )
    finally:
        adapter.close()

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir.resolve()}")
    print(f"Completed requests: {manifest['completed_requests']}/{manifest['planned_requests']}")
    print(f"PIT rows: {manifest['pit_rows']}")
    print(f"Intervals: {manifest['constituent_intervals']}")
    print(f"Scope profile: {manifest['scope_profile']}")
    print(f"Status: {manifest['status']}")
    print(f"Missing Russell coverage: {','.join(manifest['missing_russell_coverage'])}")
    print(f"Manifest: {artifacts.manifest_path.resolve()}")


if __name__ == "__main__":
    main()
