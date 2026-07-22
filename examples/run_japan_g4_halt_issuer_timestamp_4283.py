from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_issuer_timestamp_4283 import (
    produce_japan_g4_halt_issuer_timestamp_4283,
)
from quant_lab.data.japan_g4_halt_issuer_timestamp_4283_validator import (
    validate_japan_g4_halt_issuer_timestamp_4283,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline fail-closed issuer publication timestamp audit for JPX code 4283."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-run-id", required=True)
    parser.add_argument(
        "--events-path",
        default="artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V16/halt_suspension_events.parquet",
    )
    parser.add_argument(
        "--event-source-path",
        default=(
            "artifacts/japan_g4_tse_wayback_lifecycle/"
            "JPN-G4-TSE-WAYBACK-LIFECYCLE-2015-01-12-COPIED-20260712-V1/"
            "raw/pdfs/2015-07.pdf"
        ),
    )
    parser.add_argument("--issuer-source-path")
    parser.add_argument("--issuer-source-url", default="")
    parser.add_argument("--issuer-source-sha256", default="")
    parser.add_argument("--publication-date", default="")
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g4_halt_issuer_timestamp_4283"
    )
    parser.add_argument(
        "--validation-artifact-root",
        default="artifacts/japan_g4_halt_issuer_timestamp_4283_validation",
    )
    args = parser.parse_args()

    producer = produce_japan_g4_halt_issuer_timestamp_4283(
        run_id=args.run_id,
        events_path=args.events_path,
        event_source_path=args.event_source_path,
        issuer_source_path=args.issuer_source_path,
        issuer_source_url=args.issuer_source_url,
        issuer_source_sha256=args.issuer_source_sha256,
        publication_date=args.publication_date,
        artifact_root=args.artifact_root,
    )
    validation = validate_japan_g4_halt_issuer_timestamp_4283(
        run_id=args.validation_run_id,
        producer_manifest_path=producer.manifest_path,
        artifact_root=args.validation_artifact_root,
    )
    print(json.dumps({
        "producer_manifest": str(producer.manifest_path),
        "producer_status": producer.manifest["status"],
        "accepted_events": producer.manifest["counts"]["accepted_events"],
        "critical_blockers": producer.manifest["counts"]["critical_blockers"],
        "validation_manifest": str(validation.manifest_path),
        "validation_status": validation.manifest["status"],
        "validation_issues": validation.manifest["counts"]["validation_issues"],
        "promotion_eligible": validation.manifest["promotion_eligible"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
