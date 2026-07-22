from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_pit_extraction import recover_refinitiv_pit_historical_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover a Refinitiv PIT manifest from parquet checkpoints.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_pit_historical_backfill")
    parser.add_argument("--plan-path", default=None)
    parser.add_argument("--session", default="desktop.workspace")
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_root) / args.run_id
    plan_path = Path(args.plan_path) if args.plan_path else artifact_dir / "historical_plan.csv"
    plan = pd.read_csv(plan_path, low_memory=False)
    artifacts = recover_refinitiv_pit_historical_manifest(
        plan=plan,
        artifact_root=Path(args.artifact_root),
        run_id=args.run_id,
        session_name=args.session,
    )

    print(f"Run ID: {artifacts.run_id}")
    print(f"Recovered manifest rows: {len(artifacts.manifest)}")
    print(f"Recovered history rows: {int(pd.to_numeric(artifacts.manifest.get('history_rows'), errors='coerce').fillna(0).sum())}")
    print(f"Manifest: {artifacts.manifest_path}")


if __name__ == "__main__":
    main()
