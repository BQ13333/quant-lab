from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_denominator_recovery import (
    run_japan_g4_halt_denominator_recovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded, fail-closed Japan G4 halt denominator recovery.")
    parser.add_argument("--run-id", default="JPN-G4-HALT-DENOMINATOR-RECOVERY-20260712-V2")
    parser.add_argument("--unresolved-gaps", default="artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V2/unresolved_gaps.csv")
    parser.add_argument("--artifact-root", default="artifacts/japan_g4_halt_denominator_recovery")
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--max-pdfs", type=int, default=1000)
    parser.add_argument(
        "--oracle-validation-dir",
        default=(
            "artifacts/japan_g4_halt_denominator_recovery_validation/"
            "JPN-G4-HALT-DENOMINATOR-INDEPENDENT-VALIDATION-20260712-V1"
        ),
    )
    args = parser.parse_args()
    result = run_japan_g4_halt_denominator_recovery(
        run_id=args.run_id,
        unresolved_gaps_path=args.unresolved_gaps,
        artifact_root=args.artifact_root,
        end_year=args.end_year,
        max_pdfs=args.max_pdfs,
        oracle_validation_dir=args.oracle_validation_dir,
        require_oracle=True,
        write_once=True,
    )
    print(json.dumps({"artifact_dir": str(result.artifact_dir.resolve()), "status": result.manifest["status"], "coverage": result.manifest["coverage"], "scorecard_eligible": result.manifest["scorecard_eligible"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
