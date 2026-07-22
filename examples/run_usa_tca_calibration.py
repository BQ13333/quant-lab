from __future__ import annotations

import argparse
import json

from quant_lab.data.usa_tca_calibration import FeeInput, calibrate_usa_tca


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate research-only USA TCA scalars from U2 daily market data."
    )
    parser.add_argument("--daily-features", required=True)
    parser.add_argument("--eligible-universe-tiers", required=True)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--artifact-root", default="artifacts/usa_tca_calibration")
    parser.add_argument("--run-id")
    parser.add_argument("--reference-participation-rate", type=float, default=0.10)
    parser.add_argument("--max-participation-rate", type=float, default=0.10)
    parser.add_argument("--commission-bps", type=float, required=True)
    parser.add_argument("--commission-source-name", required=True)
    parser.add_argument("--commission-source-quality", required=True)
    parser.add_argument("--exchange-fee-bps", type=float, required=True)
    parser.add_argument("--exchange-fee-source-name", required=True)
    parser.add_argument("--exchange-fee-source-quality", required=True)
    parser.add_argument("--regulatory-sell-fee-bps", type=float, required=True)
    parser.add_argument("--regulatory-sell-fee-source-name", required=True)
    parser.add_argument("--regulatory-sell-fee-source-quality", required=True)
    parser.add_argument("--market-data-source-name", default="usa_l2_daily_features_u2")
    parser.add_argument("--market-data-source-quality", default="derived_ohlcv_market_data")
    parser.add_argument("--batch-size", type=int, default=65_536)
    parser.add_argument("--max-calibration-samples", type=int, default=250_000)
    parser.add_argument("--random-seed", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    artifacts = calibrate_usa_tca(
        daily_features_path=args.daily_features,
        eligible_universe_tiers_path=args.eligible_universe_tiers,
        start_date=args.start_date,
        end_date=args.end_date,
        commission=FeeInput(
            args.commission_bps,
            args.commission_source_name,
            args.commission_source_quality,
        ),
        exchange_fee=FeeInput(
            args.exchange_fee_bps,
            args.exchange_fee_source_name,
            args.exchange_fee_source_quality,
        ),
        regulatory_sell_fee=FeeInput(
            args.regulatory_sell_fee_bps,
            args.regulatory_sell_fee_source_name,
            args.regulatory_sell_fee_source_quality,
        ),
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        reference_participation_rate=args.reference_participation_rate,
        max_participation_rate=args.max_participation_rate,
        market_data_source_name=args.market_data_source_name,
        market_data_source_quality=args.market_data_source_quality,
        batch_size=args.batch_size,
        max_calibration_samples=args.max_calibration_samples,
        random_seed=args.random_seed,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "artifact_dir": str(artifacts.artifact_dir.resolve()),
                "status": artifacts.manifest["status"],
                "live_trading_ready": artifacts.manifest["live_trading_ready"],
                "ibkr_ready": artifacts.manifest["ibkr_ready"],
                "assumptions_json": str(artifacts.assumptions_json_path.resolve()),
                "manifest": str(artifacts.manifest_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
