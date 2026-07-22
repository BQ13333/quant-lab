from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.asean_l2 import write_asean_l2_plan_artifacts
from quant_lab.data.universe_mining import save_mined_asean_universe


def _read_csv_or_empty(path: str | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    return pd.read_csv(Path(path).expanduser(), low_memory=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ASEAN L2 data-depth plan and Refinitiv symbol-map input.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/asean_l2_data_plan")
    parser.add_argument("--listings", default=None, help="Optional ASEAN current listings CSV.")
    parser.add_argument("--symbol-map-csv", default=None, help="Optional validated Refinitiv symbol_map.csv.")
    parser.add_argument("--target-per-country", type=int, default=100)
    parser.add_argument("--sources", default="sgx,pse,hnx,upcom", help="Public ASEAN sources to mine if --listings is omitted.")
    parser.add_argument("--mine-public", dest="mine_public", action="store_true", default=True)
    parser.add_argument("--no-mine-public", dest="mine_public", action="store_false")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("ASEAN-L2-DATA-PLAN-%Y%m%d-%H%M%S")
    if args.listings:
        listings = pd.read_csv(Path(args.listings).expanduser(), low_memory=False)
    elif args.mine_public:
        sources = tuple(item.strip() for item in args.sources.split(",") if item.strip())
        mined = save_mined_asean_universe(run_id=f"{run_id}-PUBLIC-LISTINGS", sources=sources)
        listings = mined.listings
    else:
        listings = pd.DataFrame()

    symbol_map = _read_csv_or_empty(args.symbol_map_csv)
    artifact = write_asean_l2_plan_artifacts(
        public_listings=listings,
        symbol_map=symbol_map,
        artifact_root=args.artifact_root,
        run_id=run_id,
        target_per_country=args.target_per_country,
    )

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifact.artifact_dir}")
    print(f"Public listing rows: {len(listings)}")
    print(f"Ranked public candidate rows: {len(artifact.ranked_public_candidates)}")
    print(f"L2 candidate rows: {len(artifact.l2_candidate_symbols)}")
    print(f"Refinitiv symbol-map input rows: {len(artifact.refinitiv_symbol_map_input)}")
    print(f"Country targets: {artifact.country_targets_path}")
    print(artifact.country_targets[["country", "public_l2_candidate_rows", "validated_refinitiv_price_symbols", "readiness"]].to_string(index=False))
    print(f"Plan: {artifact.markdown_path}")


if __name__ == "__main__":
    main()
