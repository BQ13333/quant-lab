from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_history_validator import (
    validate_japan_g4_halt_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate Japan G4 halt/suspension evidence."
    )
    parser.add_argument("--run-id", default="JPN-G4-HALTS-INDEPENDENT-VALIDATION-V1")
    parser.add_argument("--producer-dir", required=True)
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g4_halt_history_validation"
    )
    args = parser.parse_args()
    result = validate_japan_g4_halt_history(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.manifest["status"],
                "validation_issues": result.manifest["validation_issues"],
                "critical_blockers": result.manifest["critical_blockers"],
                "manifest": str(result.manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
