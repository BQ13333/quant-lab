from __future__ import annotations

import argparse

from quant_lab.data.japan_pit_data_plan import build_japan_g3_pit_data_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Japan G3 PIT acquisition plan.")
    parser.add_argument(
        "--universe-path",
        default=(
            "artifacts/research_universes/JPX-UNIVERSE-RECONCILIATION-V2-20260711/"
            "japan_jpx_active_operating_core_v1.csv"
        ),
    )
    parser.add_argument(
        "--g1-validation-dir",
        default="artifacts/japan_g1_validation/JPN-G1-MONTHLY-INDEPENDENT-VALIDATION-20260712-V1",
    )
    parser.add_argument("--artifact-root", default="artifacts/pit_data_plan")
    parser.add_argument("--run-id", default="PIT-DATA-PLAN-JPN-U2-20260711")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--lseg-connector-status", default="not_checked")
    args = parser.parse_args()

    result = build_japan_g3_pit_data_plan(
        universe_path=args.universe_path,
        g1_validation_dir=args.g1_validation_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        lseg_connector_status=args.lseg_connector_status,
    )
    print(f"Run ID: {result.run_id}")
    print(f"Artifact dir: {result.artifact_dir}")
    print(f"Selected names: {result.selected_names}")
    print(f"Extraction batches: {result.extraction_batches}")
    print(f"Scorecard eligible: {result.scorecard_eligible}")


if __name__ == "__main__":
    main()
