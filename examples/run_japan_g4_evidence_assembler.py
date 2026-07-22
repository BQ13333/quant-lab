from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_evidence_assembler import assemble_japan_g4_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble independently admitted Japan G4 evidence.")
    parser.add_argument("--run-id", default="FX-CALENDAR-TCA-JPN-20260712-V6")
    parser.add_argument(
        "--base-rules-dir",
        default="artifacts/fx_calendar_tca/FX-CALENDAR-TCA-JPN-20260711",
    )
    parser.add_argument(
        "--calendar-validation-dir",
        default=(
            "artifacts/japan_exchange_calendar_validation/"
            "JPN-EXCHANGE-CALENDAR-INDEPENDENT-VALIDATION-20260712-V1"
        ),
    )
    parser.add_argument(
        "--rule-history-validation-dir",
        default=(
            "artifacts/japan_market_rule_history_validation/"
            "JPN-MARKET-RULE-HISTORY-INDEPENDENT-VALIDATION-20260712-V1"
        ),
    )
    parser.add_argument(
        "--lot-size-validation-dir",
        default=(
            "artifacts/japan_monthly_microstructure_validation/"
            "JPN-MONTHLY-MICROSTRUCTURE-INDEPENDENT-VALIDATION-20260712-V1"
        ),
    )
    parser.add_argument(
        "--tick-assignment-validation-dir",
        default=(
            "artifacts/japan_tick_assignment_history_validation/"
            "JPN-TICK-ASSIGNMENT-INDEPENDENT-VALIDATION-20260712-V3"
        ),
    )
    parser.add_argument("--halt-validation-dir")
    parser.add_argument("--currency-validation-dir")
    parser.add_argument("--tick-rule-validation-dir")
    parser.add_argument("--pit-tick-assignment-validation-dir")
    parser.add_argument("--normal-price-limit-validation-dir")
    parser.add_argument("--exceptional-price-limit-validation-dir")
    parser.add_argument("--tca-validation-dir")
    parser.add_argument("--artifact-root", default="artifacts/fx_calendar_tca")
    args = parser.parse_args()
    result = assemble_japan_g4_evidence(
        run_id=args.run_id,
        base_rules_dir=args.base_rules_dir,
        lot_size_validation_dir=args.lot_size_validation_dir,
        calendar_validation_dir=args.calendar_validation_dir,
        rule_history_validation_dir=args.rule_history_validation_dir,
        tick_assignment_validation_dir=args.tick_assignment_validation_dir,
        halt_validation_dir=args.halt_validation_dir,
        currency_validation_dir=args.currency_validation_dir,
        tick_rule_validation_dir=args.tick_rule_validation_dir,
        pit_tick_assignment_validation_dir=args.pit_tick_assignment_validation_dir,
        normal_price_limit_validation_dir=args.normal_price_limit_validation_dir,
        exceptional_price_limit_validation_dir=args.exceptional_price_limit_validation_dir,
        tca_validation_dir=args.tca_validation_dir,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.manifest["status"],
                "complete_components": result.manifest["complete_components"],
                "critical_blockers": result.manifest["critical_blockers"],
                "manifest": str(result.manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
