from __future__ import annotations

import argparse
from datetime import date, datetime, timezone
from pathlib import Path

from quant_lab.data.jquants_listed_shares import HISTORY_START, run_jquants_listed_shares_backfill


def _parse_codes(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a bounded, resumable J-Quants Pro v2 listed-shares backfill."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/jquants_listed_shares")
    parser.add_argument("--start-date", default=HISTORY_START.isoformat())
    parser.add_argument("--end-date", default=date.today().isoformat())
    parser.add_argument(
        "--codes",
        default=None,
        help="Optional comma-separated 4/5-character security codes for a narrower immutable plan.",
    )
    parser.add_argument(
        "--max-dates",
        required=True,
        type=int,
        help="Maximum distinct pending updated_at dates in this invocation.",
    )
    parser.add_argument(
        "--max-codes",
        required=True,
        type=int,
        help="Maximum distinct pending codes in this invocation; use 1 for an all-code plan.",
    )
    parser.add_argument("--max-pages-per-plan-row", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--pacing-seconds", type=float, default=1.0)
    parser.add_argument("--base-retry-seconds", type=float, default=2.0)
    parser.add_argument("--max-retry-seconds", type=float, default=60.0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime(
        "JQUANTS-LISTED-SHARES-%Y%m%d-%H%M%S"
    )
    artifacts = run_jquants_listed_shares_backfill(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        codes=_parse_codes(args.codes),
        max_dates=args.max_dates,
        max_codes=args.max_codes,
        max_pages_per_plan_row=args.max_pages_per_plan_row,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        pacing_seconds=args.pacing_seconds,
        base_retry_seconds=args.base_retry_seconds,
        max_retry_seconds=args.max_retry_seconds,
        resume=not args.no_resume,
    )

    print(f"Run ID: {artifacts.run_id}")
    print(f"Status: {artifacts.manifest['status']}")
    print(
        "Completed plan rows: "
        f"{artifacts.manifest['completed_plan_rows']}/{artifacts.manifest['plan_rows']}"
    )
    print(f"Share ledger rows: {len(artifacts.share_ledger)}")
    print(f"Share-change ledger rows: {len(artifacts.share_change_ledger)}")
    print(f"Promotion: {artifacts.manifest['promotion']}")
    print(f"Manifest: {artifacts.manifest_path.resolve()}")


if __name__ == "__main__":
    main()
