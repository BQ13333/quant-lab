from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_normalized_events import run_japan_g4_halt_normalized_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize validated Japan G4 exact-month lifecycle events.")
    parser.add_argument("--run-id", default="JPN-G4-HALT-NORMALIZED-EVENTS-20260712-V3-R1")
    parser.add_argument("--denominator-v2-dir", default="artifacts/japan_g4_halt_denominator_recovery/JPN-G4-HALT-DENOMINATOR-RECOVERY-20260712-V2")
    parser.add_argument("--validation-v2-dir", default="artifacts/japan_g4_halt_denominator_recovery_validation/JPN-G4-HALT-DENOMINATOR-INDEPENDENT-VALIDATION-20260712-V2")
    parser.add_argument("--artifact-root", default="artifacts/japan_g4_halt_normalized_events")
    args = parser.parse_args()
    result = run_japan_g4_halt_normalized_events(
        run_id=args.run_id,
        denominator_v2_dir=args.denominator_v2_dir,
        validation_v2_dir=args.validation_v2_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"artifact_dir": str(result.artifact_dir.resolve()), **result.manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
