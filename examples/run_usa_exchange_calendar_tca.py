from __future__ import annotations

import argparse
import json

from quant_lab.data.usa_exchange_calendar_tca import (
    TCAAssumptions,
    write_usa_exchange_calendar_tca_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline USA exchange-calendar and TCA evidence artifacts.")
    parser.add_argument("--artifact-root", default="artifacts/fx_calendar_tca")
    parser.add_argument("--run-id", default="FX-CALENDAR-TCA-USA-20260709")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-09")
    parser.add_argument(
        "--calendar-backend",
        choices=("auto", "exchange_calendars", "pandas_market_calendars", "builtin"),
        default="auto",
    )
    parser.add_argument("--halt-suspension-input")
    parser.add_argument(
        "--calibration-manifest",
        help="Research TCA calibration manifest; its checksummed assumptions are loaded automatically.",
    )
    parser.add_argument("--halt-source-name", default="not_provided")
    parser.add_argument("--halt-source-quality", default="missing_official_source")
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--exchange-fee-bps", type=float, default=0.30)
    parser.add_argument("--regulatory-sell-fee-bps", type=float, default=0.0)
    parser.add_argument("--half-spread-bps", type=float, default=2.50)
    parser.add_argument("--impact-bps", type=float, default=5.0)
    parser.add_argument("--reference-participation-rate", type=float, default=0.10)
    parser.add_argument("--max-participation-rate", type=float, default=0.10)
    parser.add_argument("--tca-source-quality", default="model_assumption_unvalidated")
    parser.add_argument("--tca-source-name", default="user_configured_research_assumptions")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    assumptions = None
    if args.calibration_manifest is None:
        assumptions = TCAAssumptions(
            commission_bps=args.commission_bps,
            exchange_fee_bps=args.exchange_fee_bps,
            regulatory_sell_fee_bps=args.regulatory_sell_fee_bps,
            half_spread_bps=args.half_spread_bps,
            impact_bps_at_reference_participation=args.impact_bps,
            reference_participation_rate=args.reference_participation_rate,
            max_participation_rate=args.max_participation_rate,
            source_quality=args.tca_source_quality,
            source_name=args.tca_source_name,
        )
    artifacts = write_usa_exchange_calendar_tca_artifacts(
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        calendar_backend=args.calendar_backend,
        tca_assumptions=assumptions,
        calibration_manifest_input=args.calibration_manifest,
        halt_suspension_input=args.halt_suspension_input,
        halt_source_quality=args.halt_source_quality,
        halt_source_name=args.halt_source_name,
        overwrite=args.overwrite,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "artifact_dir": str(artifacts.artifact_dir.resolve()),
                "status": manifest["status"],
                "backtest_ready": manifest["backtest_ready"],
                "backtest_scorecard_eligible": manifest["backtest_scorecard_eligible"],
                "production_ready": manifest["production_ready"],
                "live_execution_ready": manifest["live_execution_ready"],
                "incomplete_components": manifest["quality_fields"]["incomplete_components"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
