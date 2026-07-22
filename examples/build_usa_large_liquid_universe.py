from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.usa_research_universe import (
    USA_LARGE_LIQUID_DEFAULT_MIN_ADV_VALUE_USD,
    USA_LARGE_LIQUID_DEFAULT_MIN_FREE_FLOAT_PCT,
    USA_LARGE_LIQUID_DEFAULT_MIN_MARKET_CAP_USD,
    write_usa_large_liquid_universe_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the USA >$1bn market-cap and liquidity-gated research universe. "
            "Without --liquidity-snapshot-csv this writes the bounded Refinitiv snapshot request only."
        )
    )
    parser.add_argument(
        "--security-master",
        default=(
            "artifacts/data_lake/pit/security_master_history/"
            "SECMASTER-DISCOVERY-USA-MERGED-RESIDUAL-20260709/security_master.csv"
        ),
    )
    parser.add_argument("--liquidity-snapshot-csv", default=None)
    parser.add_argument("--run-id", default="USA-LARGE-LIQUID-UNIVERSE-V1-20260709")
    parser.add_argument("--artifact-root", default="artifacts/research_universes")
    parser.add_argument("--symbol-map-artifact-root", default="artifacts/refinitiv_symbol_map")
    parser.add_argument("--min-market-cap-usd", type=float, default=USA_LARGE_LIQUID_DEFAULT_MIN_MARKET_CAP_USD)
    parser.add_argument("--min-adv-value-usd", type=float, default=USA_LARGE_LIQUID_DEFAULT_MIN_ADV_VALUE_USD)
    parser.add_argument("--min-free-float-pct", type=float, default=USA_LARGE_LIQUID_DEFAULT_MIN_FREE_FLOAT_PCT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = write_usa_large_liquid_universe_artifacts(
        security_master_path=args.security_master,
        run_id=args.run_id,
        liquidity_snapshot_path=args.liquidity_snapshot_csv,
        artifact_root=Path(args.artifact_root),
        symbol_map_artifact_root=Path(args.symbol_map_artifact_root),
        min_market_cap_usd=args.min_market_cap_usd,
        min_adv_value_usd=args.min_adv_value_usd,
        min_free_float_pct=args.min_free_float_pct,
    )
    print(f"run_id={artifacts.run_id}")
    print(f"status={artifacts.manifest['status']}")
    print(f"artifact_dir={artifacts.artifact_dir.resolve()}")
    print(f"snapshot_request={artifacts.snapshot_request_path.resolve()}")
    print(f"snapshot_field_map={artifacts.snapshot_field_map_path.resolve()}")
    if artifacts.large_liquid_symbol_map_path:
        print(f"large_liquid_symbol_map={artifacts.large_liquid_symbol_map_path.resolve()}")
    print(f"manifest={artifacts.manifest_path.resolve()}")


if __name__ == "__main__":
    main()
