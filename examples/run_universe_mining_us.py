from __future__ import annotations

import argparse

from quant_lab.data.universe_mining import (
    download_mined_universe_prices,
    save_mined_us_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine current US listings from Nasdaq Trader.")
    parser.add_argument("--include-etfs", action="store_true", help="Include ETF rows.")
    parser.add_argument("--include-non-stock-like", action="store_true", help="Include warrants, units, ETNs, preferreds, and other non-stock-like rows.")
    parser.add_argument("--include-test-issues", action="store_true", help="Include test issues.")
    parser.add_argument("--download-prices", action="store_true", help="Download Yahoo price history for mined tickers.")
    parser.add_argument("--max-tickers", type=int, default=200, help="Price-download cap. Ignored unless --download-prices is set.")
    parser.add_argument("--period", default="5y", help="Yahoo Finance history period.")
    args = parser.parse_args()

    result = save_mined_us_universe(
        artifact_root="artifacts/universes",
        include_etfs=args.include_etfs,
        stock_like_only=not args.include_non_stock_like,
        include_test_issues=args.include_test_issues,
    )

    print(f"Run ID: {result.run_id}")
    print(f"Artifacts: {result.artifact_dir}")
    print(f"Listings: {len(result.listings)}")
    print(result.listings[["ticker", "name", "exchange", "security_type", "asset_type", "yahoo_ticker"]].head(20).to_string(index=False))

    if args.download_prices:
        prices = download_mined_universe_prices(
            result.listings,
            output_path=result.artifact_dir / "prices.parquet",
            period=args.period,
            max_tickers=args.max_tickers,
        )
        print()
        print(f"Downloaded price columns: {prices.prices.shape[1]}")
        print(f"Downloaded price rows: {prices.prices.shape[0]}")
        print(f"Failures: {len(prices.failures)}")


if __name__ == "__main__":
    main()
