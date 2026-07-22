from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from quant_lab.agents.data_lake_expansion_loop import (
    DataLakeExpansionConfig,
    latest_data_lake_expansion_run_id,
    write_data_lake_expansion_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a bounded agent queue for expanding the quant data lake.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/data_lake_expansion")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--smoke-max-plan-rows", type=int, default=1)
    parser.add_argument("--symbol-map-sample-per-market", type=int, default=3)
    parser.add_argument("--symbol-map-max-symbols", type=int, default=150)
    parser.add_argument("--refinitiv-session", default="desktop.workspace")
    parser.add_argument("--field-catalog-csv", default=None)
    parser.add_argument("--baseline-symbol-map-csv", default=None)
    parser.add_argument("--exclude-us", action="store_true", help="Leave US in discovery backlog for a later run.")
    parser.add_argument("--include-parked-thailand", action="store_true", help="Add Thailand as optional, not default admitted.")
    args = parser.parse_args()

    run_id = args.run_id or latest_data_lake_expansion_run_id()
    config_kwargs = {
        "run_id": run_id,
        "artifact_root": Path(args.artifact_root),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "include_us": not args.exclude_us,
        "include_parked_thailand": args.include_parked_thailand,
        "batch_size": args.batch_size,
        "smoke_max_plan_rows": args.smoke_max_plan_rows,
        "symbol_map_sample_per_market": args.symbol_map_sample_per_market,
        "symbol_map_max_symbols": args.symbol_map_max_symbols,
        "refinitiv_session": args.refinitiv_session,
    }
    if args.field_catalog_csv:
        config_kwargs["existing_field_catalog_csv"] = args.field_catalog_csv
    if args.baseline_symbol_map_csv:
        config_kwargs["baseline_symbol_map_csv"] = args.baseline_symbol_map_csv
    config = DataLakeExpansionConfig(**config_kwargs)
    artifacts = write_data_lake_expansion_artifacts(config)

    print(f"Run ID: {artifacts.run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Market targets: {artifacts.market_targets_path}")
    print(f"Agent queue: {artifacts.extraction_queue_path}")
    print(f"Commands: {artifacts.commands_path}")
    print(f"Summary: {artifacts.summary_path}")
    print(f"Manifest: {artifacts.manifest_path}")
    print()
    print("--- Market Targets ---")
    print(
        artifacts.market_targets[
            ["country", "exchange", "region", "stage", "status", "target_security_count"]
        ].to_string(index=False)
    )
    print()
    print("--- Next Commands ---")
    display = artifacts.extraction_queue[
        ["sequence", "agent_id", "phase", "command_type", "acceptance_gate", "promotes_alpha"]
    ]
    print(display.to_string(index=False))
    print()
    print(f"Generated at UTC: {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
