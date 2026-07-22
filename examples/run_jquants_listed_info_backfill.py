from __future__ import annotations

import argparse
import json

from quant_lab.data.jquants_listed_info import extract_jquants_listed_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a bounded, resumable research backfill of official J-Quants Pro "
            "v2 listed-issue information. Credentials are read only from "
            "JQUANTS_ID_TOKEN or JQUANTS_REFRESH_TOKEN."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--end-date", required=True, help="Inclusive YYYY-MM-DD end date")
    parser.add_argument("--max-dates", type=int, default=1)
    parser.add_argument(
        "--artifact-root", default="artifacts/jquants_listed_info"
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    parser.add_argument("--max-backoff-seconds", type=float, default=60.0)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--rate-limit-per-minute", type=float, default=5.0)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = extract_jquants_listed_info(
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        end_date=args.end_date,
        max_dates=args.max_dates,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        max_backoff_seconds=args.max_backoff_seconds,
        sleep_seconds=args.sleep_seconds,
        rate_limit_per_minute=args.rate_limit_per_minute,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "run_id": artifacts.run_id,
                "status": artifacts.manifest["status"],
                "attempted_dates_this_invocation": artifacts.manifest[
                    "request_policy"
                ]["attempted_dates_this_invocation"],
                "completed_dates": artifacts.quality_metrics["completed_dates"],
                "pending_dates": artifacts.quality_metrics["pending_dates"],
                "normalized_rows": artifacts.quality_metrics["normalized_rows"],
                "manifest_path": str(artifacts.manifest_path.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
