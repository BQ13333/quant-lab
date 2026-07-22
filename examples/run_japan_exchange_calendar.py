from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_exchange_calendar import write_japan_exchange_calendar_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a Japan exchange-calendar candidate from accepted official JPX monthly evidence.")
    parser.add_argument("--run-id", default="JPN-EXCHANGE-CALENDAR-200607-202606-V1")
    parser.add_argument("--artifact-root", default="artifacts/japan_exchange_calendar")
    args = parser.parse_args()
    result = write_japan_exchange_calendar_candidate(
        run_id=args.run_id,
        source_dirs=[
            "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200607-200711-EXT-V1/raw",
            "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2/raw",
        ],
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
