from __future__ import annotations

import argparse

from quant_lab.data.asean_l2 import ASEAN_L2_COUNTRIES
from quant_lab.data.l2_deep import build_l2_deep_extraction


def _parse_countries(value: str) -> tuple[str, ...]:
    countries = tuple(item.strip() for item in value.split(",") if item.strip())
    return countries or ASEAN_L2_COUNTRIES


def main() -> None:
    parser = argparse.ArgumentParser(description="Build L2 deep OHLCV, liquidity, quality, and extraction-readiness artifacts.")
    parser.add_argument("long_ohlcv_root", help="Path to prices_long/<run_id>/ohlcv.")
    parser.add_argument("--artifact-root", default="artifacts/l2_deep_extraction")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--include-countries", default=",".join(ASEAN_L2_COUNTRIES))
    parser.add_argument("--min-rows", type=int, default=2500)
    parser.add_argument("--max-missing-ratio", type=float, default=0.25)
    parser.add_argument("--top-liquid-per-country", type=int, default=50)
    parser.add_argument("--symbol-map-path", default=None)
    parser.add_argument("--field-catalog-path", default=None)
    parser.add_argument("--historical-manifest-path", default=None)
    args = parser.parse_args()

    result = build_l2_deep_extraction(
        long_ohlcv_root=args.long_ohlcv_root,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        include_countries=_parse_countries(args.include_countries),
        min_rows=args.min_rows,
        max_missing_ratio=args.max_missing_ratio,
        top_liquid_per_country=args.top_liquid_per_country,
        symbol_map_path=args.symbol_map_path,
        field_catalog_path=args.field_catalog_path,
        historical_manifest_path=args.historical_manifest_path,
    )

    print("=" * 80)
    print("L2 DEEP EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Run ID:              {result.run_id}")
    print(f"Artifact dir:        {result.artifact_dir}")
    print(f"Manifest:            {result.manifest_path}")
    print(f"Security master:     {result.security_master_path}")
    print(f"Universe tiers:      {result.universe_tiers_path}")
    print(f"Daily features:      {result.daily_features_path}")
    print(f"Vendor matrix:       {result.vendor_matrix_path}")
    print(f"Summary:             {result.markdown_path}")
    print(result.country_summary.to_string(index=False))


if __name__ == "__main__":
    main()
