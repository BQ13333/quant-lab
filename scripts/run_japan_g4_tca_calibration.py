from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_tca_calibration import (
    calibrate_japan_g4_tca,
    validate_japan_g4_tca_calibration,
)
from quant_lab.data.usa_tca_calibration import FeeInput


def main() -> None:
    parser = argparse.ArgumentParser(description="Produce or validate Japan G4 TCA evidence.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    produce = subparsers.add_parser("produce")
    produce.add_argument("--daily-features", required=True)
    produce.add_argument("--eligible-universe-tiers", required=True)
    produce.add_argument("--admitted-market-data-manifest", required=True)
    produce.add_argument("--start-date", required=True)
    produce.add_argument("--end-date", required=True)
    produce.add_argument("--run-id", required=True)
    produce.add_argument("--artifact-root", default="artifacts/japan_g4_tca_calibration")
    _fee_arguments(produce, "commission")
    _fee_arguments(produce, "exchange-fee")
    _fee_arguments(produce, "regulatory-sell-fee")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--producer-dir", required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument(
        "--artifact-root", default="artifacts/japan_g4_tca_calibration_validation"
    )
    args = parser.parse_args()
    if args.command == "produce":
        result = calibrate_japan_g4_tca(
            args.daily_features,
            args.eligible_universe_tiers,
            args.admitted_market_data_manifest,
            start_date=args.start_date,
            end_date=args.end_date,
            commission=_fee(args, "commission"),
            exchange_fee=_fee(args, "exchange_fee"),
            regulatory_sell_fee=_fee(args, "regulatory_sell_fee"),
            artifact_root=args.artifact_root,
            run_id=args.run_id,
        )
    else:
        result = validate_japan_g4_tca_calibration(
            run_id=args.run_id,
            producer_dir=args.producer_dir,
            artifact_root=args.artifact_root,
        )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


def _fee_arguments(parser: argparse.ArgumentParser, name: str) -> None:
    parser.add_argument(f"--{name}-bps", required=True, type=float)
    parser.add_argument(f"--{name}-source-name", required=True)
    parser.add_argument(f"--{name}-source-quality", required=True)


def _fee(args: argparse.Namespace, name: str) -> FeeInput:
    return FeeInput(
        bps=getattr(args, f"{name}_bps"),
        source_name=getattr(args, f"{name}_source_name"),
        source_quality=getattr(args, f"{name}_source_quality"),
    )


if __name__ == "__main__":
    main()
