from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_price_limit_event_history import write_japan_price_limit_event_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Build resumable official JPX exceptional daily-price-limit history.")
    parser.add_argument("--run-id", default="JPN-PRICE-LIMIT-EVENTS-201309-202606-V1")
    parser.add_argument("--artifact-root", default="artifacts/japan_price_limit_event_history")
    parser.add_argument(
        "--archive-manifest",
        default="artifacts/japan_tick_assignment_history/JPN-TICK-ASSIGNMENT-NOTICES-201401-202606-V3/archive_source_manifest.csv",
    )
    parser.add_argument("--cutoff-date", default="2026-06-30")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    args = parser.parse_args()
    result = write_japan_price_limit_event_history(
        run_id=args.run_id,
        archive_manifest_path=args.archive_manifest,
        artifact_root=args.artifact_root,
        cutoff_date=args.cutoff_date,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
