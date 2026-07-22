from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_denominator_recovery_validator import (
    validate_japan_g4_halt_denominator_recovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate Japan G4 halt denominator recovery.")
    parser.add_argument("--run-id", default="JPN-G4-HALT-DENOMINATOR-INDEPENDENT-VALIDATION-20260712-V2")
    parser.add_argument(
        "--producer-dir",
        default="artifacts/japan_g4_halt_denominator_recovery/JPN-G4-HALT-DENOMINATOR-RECOVERY-20260712-V2",
    )
    parser.add_argument("--artifact-root", default="artifacts/japan_g4_halt_denominator_recovery_validation")
    args = parser.parse_args()
    result = validate_japan_g4_halt_denominator_recovery(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({
        "artifact_dir": str(result.artifact_dir.resolve()),
        "status": result.manifest["status"],
        "covered_claims": result.manifest["covered_claims"],
        "explicit_section_claims_proven": result.manifest["explicit_section_claims_proven"],
        "fully_reconciled_claims": result.manifest["fully_reconciled_claims"],
        "validation_issue_count": result.manifest["validation_issue_count"],
        "scorecard_eligible": result.manifest["scorecard_eligible"],
        "promotion_recommended": result.manifest["promotion_recommended"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
