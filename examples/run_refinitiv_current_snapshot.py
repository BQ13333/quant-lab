from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_pit_extraction import (
    extract_refinitiv_current_snapshot,
    load_pit_plan_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a current Refinitiv snapshot from an accepted PIT field map.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708")
    parser.add_argument("--field-probe-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_current_snapshot")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tier", default="u2", choices=["u2", "u3"])
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument(
        "--datasets",
        default="security_master_pit,liquidity_market_cap_free_float",
        help="Comma-separated dataset ids accepted by the field probe.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-CURRENT-SNAPSHOT-%Y%m%d-%H%M%S")
    datasets = _parse_csv_arg(args.datasets)
    plan = load_pit_plan_artifacts(Path(args.plan_dir))
    accepted_path = Path(args.field_probe_dir) / "accepted_field_map.csv"
    if not accepted_path.exists():
        raise FileNotFoundError(f"accepted field map not found: {accepted_path}")
    accepted_field_map = pd.read_csv(accepted_path, low_memory=False)

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifacts = extract_refinitiv_current_snapshot(
            adapter.ld,
            selected_universe=plan.selected_universe,
            accepted_field_map=accepted_field_map,
            artifact_root=Path(args.artifact_root),
            run_id=run_id,
            tier=args.tier,
            datasets=datasets,
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
            session_name=args.session,
        )
    finally:
        adapter.close()

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Snapshot rows: {len(artifacts.snapshot)}")
    print(f"Fields: {len(accepted_field_map)} accepted rows before dataset filter")
    print(f"Quality: {artifacts.quality_path}")
    print(f"Chunk errors: {len(artifacts.chunk_errors)}")
    print(f"Snapshot CSV: {artifacts.snapshot_csv_path}")
    if not artifacts.quality.empty:
        print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
