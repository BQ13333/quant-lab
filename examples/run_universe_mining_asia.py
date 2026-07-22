from __future__ import annotations

import argparse

from quant_lab.data.universe_mining import (
    download_mined_universe_prices,
    save_mined_asia_universe,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine current Asia listings from public exchange sources.")
    parser.add_argument(
        "--sources",
        default="twse,tpex,krx,hkex,jpx,sse,szse",
        help="Comma-separated sources: twse,tpex,krx,hkex,jpx,sse,szse.",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="Log failed exchange sources and save the rest.")
    parser.add_argument("--download-prices", action="store_true", help="Download Yahoo price history for mined tickers.")
    parser.add_argument("--max-tickers", type=int, default=250, help="Price-download cap.")
    parser.add_argument("--period", default="1y", help="Yahoo Finance history period.")
    args = parser.parse_args()

    sources = tuple(source.strip() for source in args.sources.split(",") if source.strip())
    result = save_mined_asia_universe(
        artifact_root="artifacts/universes",
        sources=sources,
        continue_on_error=args.continue_on_error,
    )

    print(f"Run ID: {result.run_id}")
    print(f"Artifacts: {result.artifact_dir}")
    print(f"Listings: {len(result.listings)}")
    if not result.listings.empty:
        print()
        print("--- By Country ---")
        print(result.listings["country"].value_counts().to_string())
        print()
        print("--- By Exchange ---")
        print(result.listings["exchange"].value_counts().head(20).to_string())
        print()
        print(result.listings[["ticker", "yahoo_ticker", "name", "exchange", "country", "currency"]].head(25).to_string(index=False))

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
