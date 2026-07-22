from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_timestamp_recovery_v8 import (
    produce_japan_g4_halt_timestamp_recovery_v8,
)
from quant_lab.data.japan_g4_halt_timestamp_recovery_v8_validator import (
    validate_japan_g4_halt_timestamp_recovery_v8,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover and independently validate the exact 10-event Japan G4 V8 "
            "publication-date denominator from nine official JPX pages."
        )
    )
    parser.add_argument(
        "--run-id",
        default="JPN-G4-HALT-OFFICIAL-PUBLICATION-TIMESTAMPS-V8-20260712-V1",
    )
    parser.add_argument(
        "--validation-run-id",
        default=(
            "JPN-G4-HALT-OFFICIAL-PUBLICATION-TIMESTAMPS-V8-"
            "INDEPENDENT-VALIDATION-20260712-V1"
        ),
    )
    parser.add_argument(
        "--events",
        default=(
            "artifacts/japan_g4_halt_history/"
            "JPN-G4-HALTS-200607-202606-V8/halt_suspension_events.parquet"
        ),
    )
    parser.add_argument(
        "--artifact-root",
        default="artifacts/japan_g4_halt_timestamp_recovery",
    )
    parser.add_argument(
        "--validation-artifact-root",
        default="artifacts/japan_g4_halt_timestamp_recovery_validation",
    )
    args = parser.parse_args()

    producer = produce_japan_g4_halt_timestamp_recovery_v8(
        run_id=args.run_id,
        events_path=args.events,
        artifact_root=args.artifact_root,
    )
    validation_manifest_path, validation = (
        validate_japan_g4_halt_timestamp_recovery_v8(
            producer_dir=producer.artifact_dir,
            validation_run_id=args.validation_run_id,
            artifact_root=args.validation_artifact_root,
        )
    )
    print(
        json.dumps(
            {
                "producer_counts": producer.manifest["counts"],
                "producer_manifest": str(producer.manifest_path),
                "producer_status": producer.manifest["status"],
                "validation_counts": validation["counts"],
                "validation_manifest": str(validation_manifest_path),
                "validation_status": validation["status"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
