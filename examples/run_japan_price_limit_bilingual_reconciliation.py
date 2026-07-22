from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_price_limit_bilingual_reconciliation import (
    write_japan_price_limit_bilingual_reconciliation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile exceptional price-limit mismatches against official Japanese JPX notices."
    )
    parser.add_argument(
        "--run-id",
        default="JPN-PRICE-LIMIT-BILINGUAL-RECONCILIATION-201309-202606-V1",
    )
    parser.add_argument(
        "--producer-dir",
        default=(
            "artifacts/japan_price_limit_event_history/"
            "JPN-PRICE-LIMIT-EVENTS-201309-202606-V1"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/japan_price_limit_bilingual_reconciliation",
    )
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    args = parser.parse_args()
    result = write_japan_price_limit_bilingual_reconciliation(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
