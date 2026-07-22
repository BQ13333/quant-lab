from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_timestamp_recovery import (
    produce_japan_g4_halt_timestamp_recovery,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recover Japan G4 halt PIT timestamps from official JPX decisions."
    )
    parser.add_argument(
        "--run-id", default="JPN-G4-HALT-TIMESTAMP-RECOVERY-20260712-V1"
    )
    parser.add_argument(
        "--unresolved-gaps",
        default=(
            "artifacts/japan_g4_halt_history/"
            "JPN-G4-HALTS-200607-202606-V2/unresolved_gaps.csv"
        ),
    )
    parser.add_argument(
        "--events",
        default=(
            "artifacts/japan_g4_halt_history/"
            "JPN-G4-HALTS-200607-202606-V2/halt_suspension_events.parquet"
        ),
    )
    parser.add_argument(
        "--archive-source-manifest",
        default=(
            "artifacts/japan_tick_assignment_history/"
            "JPN-TICK-ASSIGNMENT-NOTICES-201401-202606-V3/"
            "archive_source_manifest.csv"
        ),
    )
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g4_halt_timestamp_recovery"
    )
    args = parser.parse_args()
    result = produce_japan_g4_halt_timestamp_recovery(
        run_id=args.run_id,
        unresolved_gaps_path=args.unresolved_gaps,
        events_path=args.events,
        archive_source_manifest_path=args.archive_source_manifest,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.manifest["status"],
                "counts": result.manifest["counts"],
                "promotion_eligible": result.manifest["promotion_eligible"],
                "validation_status": result.validation_manifest["status"],
                "manifest": str(result.manifest_path),
                "validation_manifest": str(result.validation_manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
