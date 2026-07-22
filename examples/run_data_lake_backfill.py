from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_lab.data.data_lake import (
    backfill_prices_in_batches,
    default_data_lake_root,
    write_universe_partitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write mined universe and optional price batches to the data lake.")
    parser.add_argument("listings_csv", help="Path to a mined listings.csv file.")
    parser.add_argument("--universe-id", default=None, help="Universe id. Defaults to listings parent folder name.")
    parser.add_argument("--root", default=None, help="Data lake root. Defaults to QUANT_LAB_DATA_ROOT, Google Drive mount, or artifacts/data_lake.")
    parser.add_argument("--download-prices", action="store_true", help="Download prices in bounded batches.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing partitioned listings directory.")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-tickers", type=int, default=250)
    parser.add_argument("--period", default="1y")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--provider", default="yahoo")
    parser.add_argument("--adjustment", default="auto_adjusted")
    parser.add_argument("--data-tier", default="bronze")
    parser.add_argument("--write-long-ohlcv", action="store_true", help="Also write normalized long OHLCV partitions.")
    args = parser.parse_args()

    listings_path = Path(args.listings_csv)
    listings = pd.read_csv(listings_path)
    universe_id = args.universe_id or listings_path.parent.name
    root = Path(args.root) if args.root else default_data_lake_root(Path.cwd())
    root.mkdir(parents=True, exist_ok=True)

    partition_path = write_universe_partitions(
        listings,
        root=root,
        universe_id=universe_id,
        overwrite=args.overwrite,
    )
    print(f"Data lake root: {root}")
    print(f"Partitioned listings: {partition_path}")
    print(f"Listings rows: {len(listings)}")

    if args.download_prices:
        result = backfill_prices_in_batches(
            listings=listings,
            root=root,
            batch_size=args.batch_size,
            max_tickers=args.max_tickers,
            period=args.period,
            interval=args.interval,
            provider=args.provider,
            adjustment=args.adjustment,
            data_tier=args.data_tier,
            write_long_ohlcv=args.write_long_ohlcv,
        )
        print(f"Price batch root: {result.root}")
        print(f"Manifest: {result.manifest_path}")
        if result.long_ohlcv_root:
            print(f"Long OHLCV root: {result.long_ohlcv_root}")
        print(result.manifest.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
