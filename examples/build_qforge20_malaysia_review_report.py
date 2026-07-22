from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.research.qforge20_workflow import (
    build_qforge20_v4_review_report,
    persist_qforge20_review_report,
)


QFORGE_ROOT = Path("artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1")
KNOWN_SCHEMA_SMOKE_COST_USD = 0.00121066


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and summarize the completed QForge-20 Malaysia V4 review ledger."
    )
    parser.add_argument("--qforge-root", type=Path, default=QFORGE_ROOT)
    args = parser.parse_args()
    report = build_qforge20_v4_review_report(
        index_path=args.qforge_root / "index.json",
        ledger_path=args.qforge_root / "review_ledger.sqlite",
        evidence_root=args.qforge_root / "review_evidence",
        known_unledgered_call_count=1,
        known_unledgered_cost_usd=KNOWN_SCHEMA_SMOKE_COST_USD,
    )
    target = persist_qforge20_review_report(
        report,
        args.qforge_root / "v4_review_report.json",
    )
    print(
        json.dumps(
            {
                "workflow_name": report["workflow_name"],
                "completion": report["completion"],
                "cost_accounting": report["cost_accounting"],
                "promotion_status": report["promotion_status"],
                "report_path": str(target),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
