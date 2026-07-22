from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.data.japan_price_limit_bilingual_validation import (
    RECONCILIATION_RUN_ID,
    SOURCE_RUN_ID,
    write_japan_price_limit_bilingual_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate the immutable JPX bilingual price-limit reconciliation."
    )
    parser.add_argument(
        "--run-id",
        default="JPN-PRICE-LIMIT-BILINGUAL-VALIDATION-201309-202606-V1",
    )
    parser.add_argument(
        "--producer-dir",
        type=Path,
        default=Path("artifacts/japan_price_limit_event_history") / SOURCE_RUN_ID,
    )
    parser.add_argument(
        "--reconciliation-dir",
        type=Path,
        default=Path("artifacts/japan_price_limit_bilingual_reconciliation")
        / RECONCILIATION_RUN_ID,
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/japan_price_limit_bilingual_validation"),
    )
    args = parser.parse_args()
    result = write_japan_price_limit_bilingual_validation(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        reconciliation_dir=args.reconciliation_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
