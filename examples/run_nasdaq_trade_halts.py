from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.nasdaq_trade_halts import extract_nasdaq_trade_halts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract bounded historical trade halts from the official Nasdaq Trader RSS route."
    )
    parser.add_argument("--start-date", required=True, help="Inclusive YYYY-MM-DD start date.")
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD end date.")
    parser.add_argument(
        "--max-dates",
        required=True,
        type=int,
        help="Maximum pending business dates to request in this invocation.",
    )
    parser.add_argument("--artifact-root", default="artifacts/nasdaq_trade_halts")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded concurrent requests; one manifest writer remains in this process.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=60.0,
        help="Delay between requests and retries; Nasdaq publishes a one-minute refresh guideline.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Require a new empty artifact directory instead of resuming a checkpoint.",
    )
    args = parser.parse_args()

    artifacts = extract_nasdaq_trade_halts(
        artifact_root=Path(args.artifact_root),
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        max_dates=args.max_dates,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        sleep_seconds=args.sleep_seconds,
        workers=args.workers,
        resume=not args.no_resume,
    )

    print(f"Run ID: {artifacts.run_id}")
    print(f"Status: {artifacts.manifest['status']}")
    print(f"Covered dates: {artifacts.manifest['covered_dates']}/{artifacts.manifest['planned_business_dates']}")
    print(f"Event rows: {len(artifacts.halt_events)}")
    print(f"Artifact dir: {artifacts.artifact_dir.resolve()}")
    print(f"Manifest: {artifacts.manifest_path.resolve()}")


if __name__ == "__main__":
    main()
