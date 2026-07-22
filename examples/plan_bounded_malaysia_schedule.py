from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

from quant_lab.research.institutional_workflow import FiniteCampaignSchedule


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit a finite, non-overlapping Malaysia campaign plan.")
    parser.add_argument("--start-at", required=True, help="Timezone-aware ISO-8601 start timestamp.")
    parser.add_argument("--campaign-count", type=int, default=4)
    parser.add_argument("--interval-hours", type=int, default=6)
    parser.add_argument("--max-trials", type=int, default=3)
    args = parser.parse_args()
    schedule = FiniteCampaignSchedule(
        start_at=datetime.fromisoformat(args.start_at),
        campaign_count=args.campaign_count,
        interval_hours=args.interval_hours,
        max_trials_per_campaign=args.max_trials,
    )
    sys.stdout.write(json.dumps(schedule.planned_runs(), indent=2) + "\n")


if __name__ == "__main__":
    main()
