from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_price_limit_event_history_validation import (
    write_japan_price_limit_event_history_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate official JPX exceptional price-limit history.")
    parser.add_argument("--run-id", default="JPN-PRICE-LIMIT-EVENTS-INDEPENDENT-VALIDATION-20260712-V1")
    parser.add_argument(
        "--producer-dir",
        default="artifacts/japan_price_limit_event_history/JPN-PRICE-LIMIT-EVENTS-201309-202606-V1",
    )
    parser.add_argument("--artifact-root", default="artifacts/japan_price_limit_event_history_validation")
    args = parser.parse_args()
    result = write_japan_price_limit_event_history_validation(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
