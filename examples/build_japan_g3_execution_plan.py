from __future__ import annotations

import argparse

from quant_lab.data.japan_g3_execution_plan import write_japan_g3_execution_plan_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the route-aware Japan G3 execution plan.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-JPN-U2-20260711")
    parser.add_argument(
        "--sector-snapshot-dir",
        default="artifacts/refinitiv_current_snapshot/REFINITIV-JPN-G3-SECTOR-ROUTING-20260712",
    )
    parser.add_argument(
        "--route-registry-dir",
        default="artifacts/japan_g3_route_registry/JPN-G3-ROUTE-REGISTRY-20260712",
    )
    parser.add_argument(
        "--g1-validation-dir",
        default="artifacts/japan_g1_validation/JPN-G1-MONTHLY-INDEPENDENT-VALIDATION-20260712-V1",
    )
    parser.add_argument("--artifact-root", default="artifacts/japan_g3_execution_plan")
    parser.add_argument("--run-id", default="JPN-G3-EXECUTION-PLAN-20260712")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--window-years", type=int, default=5)
    args = parser.parse_args()
    result = write_japan_g3_execution_plan_artifacts(
        plan_dir=args.plan_dir,
        sector_snapshot_dir=args.sector_snapshot_dir,
        route_registry_dir=args.route_registry_dir,
        g1_validation_dir=args.g1_validation_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        window_years=args.window_years,
    )
    print(f"Artifact dir: {result.artifact_dir}")
    print(f"Fundamentals plan rows: {len(result.fundamentals_plan)}")
    print(f"Consensus plan rows: {len(result.consensus_plan)}")
    print(f"Earnings plan rows: {len(result.earnings_plan)}")
    print(f"Analyst-detail plan rows: {len(result.analyst_detail_plan)}")


if __name__ == "__main__":
    main()
