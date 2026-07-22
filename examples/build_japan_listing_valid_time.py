from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_listing_valid_time import (
    DEFAULT_G1_ENRICHMENT_DIR,
    DEFAULT_JAPAN_LISTING_VALID_TIME_ARTIFACT_ROOT,
    DEFAULT_JPX_RECONCILIATION_DIR,
    write_japan_listing_valid_time_artifacts,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the immutable Japan listing valid-time baseline."
    )
    parser.add_argument("--reconciliation-dir", default=str(DEFAULT_JPX_RECONCILIATION_DIR))
    parser.add_argument("--enrichment-dir", default=str(DEFAULT_G1_ENRICHMENT_DIR))
    parser.add_argument(
        "--artifact-root", default=str(DEFAULT_JAPAN_LISTING_VALID_TIME_ARTIFACT_ROOT)
    )
    parser.add_argument("--run-id")
    parser.add_argument("--as-of-date")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    artifacts = write_japan_listing_valid_time_artifacts(
        reconciliation_dir=args.reconciliation_dir,
        enrichment_dir=args.enrichment_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        as_of_date=args.as_of_date,
    )
    print(
        json.dumps(
            {
                "run_id": artifacts.run_id,
                "artifact_dir": str(artifacts.artifact_dir.resolve()),
                "status": artifacts.manifest["status"],
                "as_of_date": artifacts.manifest["as_of_date"],
                "counts": artifacts.manifest["counts"],
                "historical_backtest_eligible": False,
                "alpha_promotion_allowed": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
