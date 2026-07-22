from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_ohlcv_canonicalizer import canonicalize_japan_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an immutable listing-window canonical Japan OHLCV candidate.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--listing-windows", required=True)
    parser.add_argument("--historical-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts/japan_ohlcv_canonical")
    args = parser.parse_args()
    result = canonicalize_japan_ohlcv(
        raw_root=args.raw_root,
        listing_windows_path=args.listing_windows,
        historical_dir=args.historical_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
    )
    print(json.dumps(result.manifest, indent=2))


if __name__ == "__main__":
    main()
