from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_exchange_calendar_validator import write_japan_exchange_calendar_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate the Japan exchange calendar candidate.")
    parser.add_argument("--run-id", default="JPN-EXCHANGE-CALENDAR-INDEPENDENT-VALIDATION-20260712-V1")
    parser.add_argument("--producer-dir", default="artifacts/japan_exchange_calendar/JPN-EXCHANGE-CALENDAR-200607-202606-V1")
    parser.add_argument("--artifact-root", default="artifacts/japan_exchange_calendar_validation")
    args = parser.parse_args()
    result = write_japan_exchange_calendar_validation(
        run_id=args.run_id, producer_dir=args.producer_dir, artifact_root=args.artifact_root
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
