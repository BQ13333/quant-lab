from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.japan_g3_sector_snapshot import (
    finalize_sector_snapshot_artifact,
    sector_field_map,
)
from quant_lab.data.refinitiv_pit_extraction import extract_refinitiv_current_snapshot, load_pit_plan_artifacts
from quant_lab.data.vendor_adapters import RefinitivAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the current Japan sector-routing snapshot.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-JPN-U2-20260711")
    parser.add_argument("--run-id", default="REFINITIV-JPN-G3-SECTOR-ROUTING-20260712")
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_current_snapshot")
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--sleep-seconds", type=float, default=0.4)
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    plan = load_pit_plan_artifacts(Path(args.plan_dir))
    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifacts = extract_refinitiv_current_snapshot(
            adapter.ld,
            selected_universe=plan.selected_universe,
            accepted_field_map=sector_field_map(),
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            tier="u2",
            datasets=["fundamentals_sector_routing"],
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
            session_name=args.session,
        )
    finally:
        adapter.close()
    finalize_sector_snapshot_artifact(
        artifacts.artifact_dir,
        snapshot=artifacts.snapshot,
        requested_rics=int(plan.selected_universe["selected_for_u2"].fillna(False).astype(bool).sum()),
        run_id=args.run_id,
    )
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Snapshot rows: {len(artifacts.snapshot)}")
    print(f"Chunk errors: {len(artifacts.chunk_errors)}")


if __name__ == "__main__":
    main()
