from __future__ import annotations

import argparse
import json

import pandas as pd

from quant_lab.data.japan_g4_tse_wayback_lifecycle import (
    run_japan_g4_tse_wayback_lifecycle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Acquire bounded original TSE lifecycle PDFs through Wayback CDX captures."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--start-month", required=True)
    parser.add_argument("--end-month", required=True)
    parser.add_argument("--max-months", type=int, default=12)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g4_tse_wayback_lifecycle"
    )
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    months = [
        str(period)
        for period in pd.period_range(args.start_month, args.end_month, freq="M")
    ]
    result = run_japan_g4_tse_wayback_lifecycle(
        run_id=args.run_id,
        months=months,
        artifact_root=args.artifact_root,
        max_months=args.max_months,
        sleep_seconds=args.sleep_seconds,
        retries=args.retries,
        timeout_seconds=args.timeout_seconds,
        resume=not args.no_resume,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
