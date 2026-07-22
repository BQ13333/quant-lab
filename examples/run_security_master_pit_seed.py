from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_lab.data.security_master_pit import write_security_master_pit_seed_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a current-survivor security-master PIT scaffold.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708")
    parser.add_argument(
        "--current-snapshot-dir",
        default="artifacts/refinitiv_current_snapshot/REFINITIV-CURRENT-SNAPSHOT-MYS-SGP-IDN-20260708",
    )
    parser.add_argument(
        "--l2-artifact-dir",
        default="artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708",
    )
    parser.add_argument("--artifact-root", default="artifacts/data_lake/pit/security_master_history")
    parser.add_argument("--run-id", default="MYS-SGP-IDN-U2-20260708")
    args = parser.parse_args()

    selected_universe = pd.read_csv(Path(args.plan_dir) / "selected_universe.csv", low_memory=False)
    current_snapshot_path = Path(args.current_snapshot_dir) / "current_snapshot.csv"
    current_snapshot = pd.read_csv(current_snapshot_path, low_memory=False) if current_snapshot_path.exists() else None
    survivor_path = Path(args.l2_artifact_dir) / "security_master_current_survivor.csv"
    survivor = pd.read_csv(survivor_path, low_memory=False) if survivor_path.exists() else None

    artifacts = write_security_master_pit_seed_artifacts(
        selected_universe=selected_universe,
        current_snapshot=current_snapshot,
        current_survivor_master=survivor,
        artifact_root=Path(args.artifact_root),
        run_id=args.run_id,
    )

    print(f"Run ID: {artifacts.run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Entities: {len(artifacts.entities)}")
    print(f"Identifier rows: {len(artifacts.identifier_history)}")
    print(f"Listing windows: {len(artifacts.listing_windows)}")
    print(f"Membership rows: {len(artifacts.universe_membership)}")
    print(f"Status: current_survivor_seed_only")


if __name__ == "__main__":
    main()
