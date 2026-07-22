from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_mapping_recovery_v8_validator import (
    validate_japan_g4_halt_mapping_recovery_v8,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate sealed Japan G4 V8 mapping recovery."
    )
    parser.add_argument(
        "--run-id",
        default=(
            "JPN-G4-HALT-MAPPING-RECOVERY-INDEPENDENT-"
            "VALIDATION-20260712-V4"
        ),
    )
    parser.add_argument(
        "--producer-dir",
        default=(
            "artifacts/japan_g4_halt_mapping_recovery/"
            "JPN-G4-HALT-MAPPING-RECOVERY-20260712-V3"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/japan_g4_halt_mapping_recovery_validation",
    )
    args = parser.parse_args()
    result = validate_japan_g4_halt_mapping_recovery_v8(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "artifact_dir": str(result.artifact_dir),
                "status": result.manifest["status"],
                "accepted_rows": result.manifest["accepted_rows"],
                "accepted_rows_validated": result.manifest[
                    "accepted_rows_validated"
                ],
                "validation_issue_count": result.manifest[
                    "validation_issue_count"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
