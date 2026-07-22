from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_report_normalization_repair_validator import (
    validate_halt_report_normalization_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate the bounded Japan G4 halt normalization repair.")
    parser.add_argument("--run-id", default="JPN-G4-HALT-REPORT-NORMALIZATION-REPAIR-INDEPENDENT-VALIDATION-20260712-V1")
    parser.add_argument("--producer-dir", default="artifacts/japan_g4_halt_report_normalization_repair/JPN-G4-HALT-REPORT-NORMALIZATION-REPAIR-20260712-V1")
    parser.add_argument("--artifact-root", default="artifacts/japan_g4_halt_report_normalization_repair_validation")
    args = parser.parse_args()
    result = validate_halt_report_normalization_repair(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"artifact_dir": str(result.artifact_dir), **result.manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
