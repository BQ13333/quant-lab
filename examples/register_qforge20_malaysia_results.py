from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.research.qforge20_workflow import (
    build_qforge20_malaysia_index,
    persist_qforge20_index,
)
from quant_lab.research.trial_ledger import TrialLedger


DEFAULT_VAULT = Path("artifacts/bounded_malaysia_loop/_malaysia_lifetime_vault")


def main() -> None:
    parser = argparse.ArgumentParser(description="Register the closed Malaysia cohort under QForge-20.")
    parser.add_argument("--source-vault", type=Path, default=DEFAULT_VAULT)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/index.json"),
    )
    parser.add_argument(
        "--v4-review-status",
        choices=("pending_rotated_key", "review_in_progress", "review_complete"),
        default="pending_rotated_key",
    )
    parser.add_argument("--reviewed-strategy-id", action="append", default=[])
    args = parser.parse_args()
    cohort_dir = args.source_vault / "cohorts"
    ledger = TrialLedger(args.source_vault / "trial_ledger.sqlite", args.source_vault / "evidence")
    index = build_qforge20_malaysia_index(
        cohort_manifest_path=cohort_dir / "MALAYSIA-ATOMIC-COHORT-V1.json",
        cohort_report_path=cohort_dir / "MALAYSIA-ATOMIC-COHORT-V1-report.json",
        lifetime_counts=ledger.lifetime_counts(campaign="malaysia"),
        v4_review_status=args.v4_review_status,
        v4_reviewed_strategy_ids=tuple(args.reviewed_strategy_id),
    )
    target = persist_qforge20_index(index, args.output)
    print(
        json.dumps(
            {
                "workflow_name": index["workflow_name"],
                "workflow_version": index["workflow_version"],
                "strategy_count": index["strategy_count"],
                "v4_review_status": index["v4_review_status"],
                "output": str(target.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
