from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.data.japan_price_limit_completion_plan import (
    write_japan_price_limit_completion_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the immutable planning-only artifact for the two retained JPX price-limit blockers."
    )
    parser.add_argument(
        "--run-id",
        default="JPN-PRICE-LIMIT-COMPLETION-PLAN-20260712-V1",
    )
    parser.add_argument(
        "--producer-dir",
        type=Path,
        default=Path("artifacts/japan_price_limit_event_history")
        / "JPN-PRICE-LIMIT-EVENTS-201309-202606-V1",
    )
    parser.add_argument(
        "--reconciliation-dir",
        type=Path,
        default=Path("artifacts/japan_price_limit_bilingual_reconciliation")
        / "JPN-PRICE-LIMIT-BILINGUAL-RECONCILIATION-201309-202606-V1",
    )
    parser.add_argument(
        "--validation-dir",
        type=Path,
        default=Path("artifacts/japan_price_limit_bilingual_validation")
        / "JPN-PRICE-LIMIT-BILINGUAL-VALIDATION-201309-202606-V1",
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/japan_price_limit_completion_plan"),
    )
    parser.add_argument("--return-batch-size", type=int, default=25)
    parser.add_argument("--max-return-sessions", type=int, default=20)
    parser.add_argument("--max-return-lookahead-days", type=int, default=35)
    args = parser.parse_args()

    result = write_japan_price_limit_completion_plan(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        reconciliation_dir=args.reconciliation_dir,
        validation_dir=args.validation_dir,
        artifact_root=args.artifact_root,
        return_batch_size=args.return_batch_size,
        max_return_sessions=args.max_return_sessions,
        max_return_lookahead_days=args.max_return_lookahead_days,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
