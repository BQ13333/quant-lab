from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.agents.japan_data_quality_execution_loop import (
    JapanExecutionLoopConfig,
    latest_japan_execution_loop_run_id,
    write_japan_execution_loop_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the guarded Japan/JPX data-quality execution queue without running live extraction."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/japan_data_quality_execution_loop")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--scorecard-run-id", default="JPN-DATA-QUALITY-LOOP-20260711")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--active-max-plan-rows", type=int, default=5)
    parser.add_argument("--reference-max-plan-rows", type=int, default=5)
    parser.add_argument(
        "--active-historical-run-id",
        default="REFINITIV-JPN-HISTORICAL-20Y-JPX-ACTIVE-G1V2",
    )
    parser.add_argument(
        "--reference-historical-run-id",
        default="REFINITIV-JPN-HISTORICAL-20Y-JPX-REFERENCE-ONLY-G1V4",
    )
    parser.add_argument(
        "--g2-vendor-plan-dir",
        default="artifacts/japan_g2_vendor_plan/JPN-G2-VENDOR-PLAN-20260712-V2",
    )
    parser.add_argument("--corporate-action-max-plan-rows", type=int, default=5)
    parser.add_argument("--liquidity-max-plan-rows", type=int, default=5)
    parser.add_argument("--refinitiv-session", default="desktop.workspace")
    parser.add_argument("--vendor-health-evidence", default="artifacts/vendor_health/japan_refinitiv_health.json")
    parser.add_argument("--vendor-health-max-age-minutes", type=int, default=60)
    parser.add_argument(
        "--g1-current-enrichment-dir",
        default=(
            "artifacts/japan_g1_enrichment/"
            "JPX-UNIVERSE-RECONCILIATION-V2-20260711-G1-CURRENT-ENRICHMENT-V2"
        ),
    )
    parser.add_argument(
        "--listing-valid-time-dir",
        default=(
            "artifacts/japan_listing_valid_time/"
            "JPX-UNIVERSE-RECONCILIATION-V2-20260711-LISTING-VALID-TIME-BASELINE"
        ),
    )
    parser.add_argument(
        "--jquants-listed-info-dir",
        default="artifacts/jquants_listed_info/JQUANTS-JPN-LISTED-INFO-20080507-20260706-V2",
    )
    parser.add_argument(
        "--jquants-listed-info-run-id",
        default="JQUANTS-JPN-LISTED-INFO-20080507-20260706-V2",
    )
    parser.add_argument("--jquants-max-dates", type=int, default=5)
    parser.add_argument(
        "--jquants-listed-shares-dir",
        default=(
            "artifacts/jquants_listed_shares/"
            "JQUANTS-JPN-LISTED-SHARES-20080507-20260706-V1"
        ),
    )
    parser.add_argument(
        "--jquants-listed-shares-run-id",
        default="JQUANTS-JPN-LISTED-SHARES-20080507-20260706-V1",
    )
    parser.add_argument(
        "--g1-materializer-dir",
        default="artifacts/japan_g1_monthly_materializer/JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1",
    )
    parser.add_argument("--g1-materializer-run-id", default="JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1")
    parser.add_argument(
        "--g1-validator-dir",
        default="artifacts/japan_g1_validation/JPN-G1-MONTHLY-INDEPENDENT-VALIDATION-20260712-V1",
    )
    parser.add_argument(
        "--jpx-public-events-dir",
        default="artifacts/jpx_public_listing_events/JPX-PUBLIC-LISTING-EVENTS-20260712",
    )
    parser.add_argument(
        "--jpx-public-crosscheck-dir",
        default="artifacts/jpx_public_event_reconciliation/JPX-PUBLIC-EVENT-REFINITIV-CROSSCHECK-V2-20260712",
    )
    parser.add_argument(
        "--jpx-monthly-quotations-dir",
        default="artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2",
    )
    parser.add_argument(
        "--jpx-monthly-membership-qa-dir",
        default="artifacts/japan_combined_monthly_membership_qa/JPN-COMBINED-MONTHLY-MEMBERSHIP-QA-200607-202606-V1",
    )
    parser.add_argument("--writer-lock-marker", default="artifacts/historical_backfill/.japan_ohlcv_writer.lock")
    parser.add_argument("--field-catalog", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = JapanExecutionLoopConfig(
        run_id=args.run_id or latest_japan_execution_loop_run_id(),
        artifact_root=Path(args.artifact_root),
        project_root=Path(args.project_root),
        scorecard_run_id=args.scorecard_run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        active_max_plan_rows=args.active_max_plan_rows,
        reference_max_plan_rows=args.reference_max_plan_rows,
        active_historical_run_id=args.active_historical_run_id,
        reference_historical_run_id=args.reference_historical_run_id,
        g2_vendor_plan_dir=Path(args.g2_vendor_plan_dir),
        corporate_action_max_plan_rows=args.corporate_action_max_plan_rows,
        liquidity_max_plan_rows=args.liquidity_max_plan_rows,
        refinitiv_session=args.refinitiv_session,
        vendor_health_evidence_path=Path(args.vendor_health_evidence),
        vendor_health_max_age_minutes=args.vendor_health_max_age_minutes,
        g1_current_enrichment_dir=Path(args.g1_current_enrichment_dir),
        listing_valid_time_dir=Path(args.listing_valid_time_dir),
        jquants_listed_info_dir=Path(args.jquants_listed_info_dir),
        jquants_listed_info_run_id=args.jquants_listed_info_run_id,
        jquants_max_dates=args.jquants_max_dates,
        jquants_listed_shares_dir=Path(args.jquants_listed_shares_dir),
        jquants_listed_shares_run_id=args.jquants_listed_shares_run_id,
        g1_materializer_dir=Path(args.g1_materializer_dir),
        g1_materializer_run_id=args.g1_materializer_run_id,
        g1_validator_dir=Path(args.g1_validator_dir),
        jpx_public_events_dir=Path(args.jpx_public_events_dir),
        jpx_public_crosscheck_dir=Path(args.jpx_public_crosscheck_dir),
        jpx_monthly_quotations_dir=Path(args.jpx_monthly_quotations_dir),
        jpx_monthly_membership_qa_dir=Path(args.jpx_monthly_membership_qa_dir),
        writer_lock_marker=Path(args.writer_lock_marker),
        field_catalog_path=Path(args.field_catalog) if args.field_catalog else None,
    )
    artifacts = write_japan_execution_loop_artifacts(config)
    print(f"run_id={artifacts.run_id}")
    print(f"artifact_dir={artifacts.artifact_dir.resolve()}")
    print(f"execution_queue={artifacts.execution_queue_path.resolve()}")
    print(f"execution_state={artifacts.execution_state_path.resolve()}")
    print(f"historical_quarantine={artifacts.historical_quarantine_path.resolve()}")
    print("live_extraction_executed=False")
    print()
    print(
        artifacts.execution_queue[
            ["sequence", "quality_gate", "owner_agent", "phase", "status", "live_extraction"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
