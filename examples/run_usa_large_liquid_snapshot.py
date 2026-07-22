from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_pit_extraction import extract_refinitiv_current_snapshot
from quant_lab.data.usa_research_universe import snapshot_request_from_symbol_map, usa_large_liquid_snapshot_field_map
from quant_lab.data.vendor_adapters import RefinitivAdapter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract the bounded Refinitiv current snapshot for the USA large-liquid screen.")
    parser.add_argument(
        "--candidate-symbol-map",
        default=(
            "artifacts/refinitiv_symbol_map/"
            "REFINITIV-SYMBOL-MAP-USA_IBKR_ACTIVE_LIQUID_CORE_V1-USA-RESEARCH-UNIVERSE-V1-20260709/symbol_map.csv"
        ),
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_current_snapshot")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--chunk-size", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=0.25)
    parser.add_argument("--max-rics", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-USA-LARGE-LIQUID-SNAPSHOT-%Y%m%d-%H%M%S")
    symbol_map = pd.read_csv(args.candidate_symbol_map, low_memory=False)
    selected_universe = snapshot_request_from_symbol_map(symbol_map)
    if args.max_rics:
        selected_universe = selected_universe.head(args.max_rics).copy()
    field_map = usa_large_liquid_snapshot_field_map()
    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifacts = extract_refinitiv_current_snapshot(
            adapter.ld,
            selected_universe=selected_universe,
            accepted_field_map=field_map,
            artifact_root=Path(args.artifact_root),
            run_id=run_id,
            tier="u2",
            datasets=["liquidity_market_cap_free_float"],
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
            session_name=args.session,
        )
    finally:
        adapter.close()

    print(f"run_id={run_id}")
    print(f"artifact_dir={artifacts.artifact_dir.resolve()}")
    print(f"requested_rics={len(selected_universe)}")
    print(f"snapshot_rows={len(artifacts.snapshot)}")
    print(f"snapshot_csv={artifacts.snapshot_csv_path.resolve()}")
    print(f"quality={artifacts.quality_path.resolve()}")
    if not artifacts.quality.empty:
        print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
