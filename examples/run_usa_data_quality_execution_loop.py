from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.agents.usa_data_quality_execution_loop import (
    UsaExecutionLoopConfig,
    latest_usa_execution_loop_run_id,
    write_usa_execution_loop_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the executable USA data-quality continuation queue.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/usa_data_quality_execution_loop")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--scorecard-run-id", default="USA-DATA-QUALITY-LOOP-20260709")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--active-max-plan-rows", type=int, default=5)
    parser.add_argument("--reference-max-plan-rows", type=int, default=5)
    parser.add_argument("--refinitiv-session", default="desktop.workspace")
    parser.add_argument("--min-market-cap-usd", type=float, default=1_000_000_000.0)
    parser.add_argument("--min-adv-value-usd", type=float, default=5_000_000.0)
    parser.add_argument("--min-free-float-pct", type=float, default=10.0)
    parser.add_argument(
        "--vendor-health-status",
        choices=["unknown", "healthy", "throttled", "failed"],
        default="unknown",
        help="Use healthy only after a tiny Refinitiv health check succeeds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or latest_usa_execution_loop_run_id()
    config = UsaExecutionLoopConfig(
        run_id=run_id,
        artifact_root=Path(args.artifact_root),
        project_root=Path(args.project_root),
        scorecard_run_id=args.scorecard_run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        active_max_plan_rows=args.active_max_plan_rows,
        reference_max_plan_rows=args.reference_max_plan_rows,
        refinitiv_session=args.refinitiv_session,
        vendor_health_status=args.vendor_health_status,
        min_market_cap_usd=args.min_market_cap_usd,
        min_adv_value_usd=args.min_adv_value_usd,
        min_free_float_pct=args.min_free_float_pct,
    )
    artifacts = write_usa_execution_loop_artifacts(config)
    print(f"run_id={artifacts.run_id}")
    print(f"artifact_dir={artifacts.artifact_dir.resolve()}")
    print(f"execution_queue={artifacts.execution_queue_path.resolve()}")
    print(f"zero_output_quarantine={artifacts.zero_output_quarantine_path.resolve()}")
    print(f"historical_quarantine={artifacts.historical_quarantine_path.resolve()}")
    print(f"summary={artifacts.summary_path.resolve()}")
    print()
    print(artifacts.execution_queue[["sequence", "owner_agent", "phase", "status", "promotes_alpha"]].to_string(index=False))


if __name__ == "__main__":
    main()
