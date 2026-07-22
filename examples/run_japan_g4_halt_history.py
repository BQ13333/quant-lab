from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g4_halt_history import (
    discover_local_jpx_source_catalog,
    produce_japan_g4_halt_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce checkpointed, denominator-complete Japan G4 halt evidence."
    )
    parser.add_argument("--run-id", default="JPN-G4-HALTS-200607-202606-V16")
    parser.add_argument("--source-catalog")
    parser.add_argument(
        "--halt-rules-dir",
        action="append",
        default=[],
        help="JPX public-rules artifact containing raw_manifest.csv; repeatable.",
    )
    parser.add_argument("--listing-events-dir")
    parser.add_argument(
        "--archive-root",
        action="append",
        default=[],
        help="Caller-declared official local archive root to hash in full; repeatable.",
    )
    parser.add_argument("--symbol-map")
    parser.add_argument("--timestamp-recovery-dir")
    parser.add_argument("--official-timestamp-recovery-dir")
    parser.add_argument("--official-timestamp-recovery-validation-dir")
    parser.add_argument("--official-timestamp-v14-recovery-dir")
    parser.add_argument("--official-timestamp-v14-recovery-validation-dir")
    parser.add_argument("--halt-report-events-dir")
    parser.add_argument("--halt-report-events-validation-dir")
    parser.add_argument("--halt-normalization-repair-dir")
    parser.add_argument("--halt-normalization-repair-validation-dir")
    parser.add_argument("--lifecycle-taxonomy-repair-dir")
    parser.add_argument("--lifecycle-taxonomy-repair-validation-dir")
    parser.add_argument(
        "--tse-wayback-lifecycle-dir",
        action="append",
        default=[],
        help="Sealed TSE Wayback lifecycle producer packet; repeatable and paired by order.",
    )
    parser.add_argument(
        "--tse-wayback-lifecycle-validation-dir",
        action="append",
        default=[],
        help="Sealed institutional validation packet; repeatable and paired by order.",
    )
    parser.add_argument("--halt-mapping-recovery-dir", action="append", default=[])
    parser.add_argument(
        "--halt-mapping-recovery-validation-dir", action="append", default=[]
    )
    parser.add_argument(
        "--g1-producer-manifest",
        default=(
            "artifacts/japan_g1_monthly_materializer/"
            "JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1/manifest.json"
        ),
    )
    parser.add_argument(
        "--g1-validation-manifest",
        default=(
            "artifacts/japan_g1_validation/"
            "JPN-G1-MONTHLY-INDEPENDENT-VALIDATION-20260712-V1/manifest.json"
        ),
    )
    parser.add_argument("--start-date", default="2006-07-01")
    parser.add_argument("--end-date", default="2026-06-30")
    parser.add_argument("--max-sources", type=int, default=1000)
    parser.add_argument("--artifact-root", default="artifacts/japan_g4_halt_history")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    lifecycle_producers = args.tse_wayback_lifecycle_dir
    lifecycle_validations = args.tse_wayback_lifecycle_validation_dir
    if not lifecycle_producers and not lifecycle_validations:
        lifecycle_producers = [
            "artifacts/japan_g4_tse_wayback_lifecycle/"
            "JPN-G4-TSE-WAYBACK-LIFECYCLE-2012-PILOT-20260712-V2",
            "artifacts/japan_g4_tse_wayback_lifecycle/"
            "JPN-G4-TSE-WAYBACK-LIFECYCLE-2013-PILOT-20260712-V2",
        ]
        lifecycle_validations = [
            "artifacts/japan_g4_tse_wayback_lifecycle_validation/"
            "JPN-G4-TSE-WAYBACK-LIFECYCLE-2012-HARDENED-"
            "INDEPENDENT-VALIDATION-20260712-V4",
            "artifacts/japan_g4_tse_wayback_lifecycle_validation/"
            "JPN-G4-TSE-WAYBACK-LIFECYCLE-2013-PILOT-"
            "INDEPENDENT-VALIDATION-20260712-V3",
        ]
    if len(lifecycle_producers) != len(lifecycle_validations):
        parser.error(
            "--tse-wayback-lifecycle-dir and "
            "--tse-wayback-lifecycle-validation-dir counts must match"
        )
    if len(args.halt_mapping_recovery_dir) != len(
        args.halt_mapping_recovery_validation_dir
    ):
        parser.error(
            "--halt-mapping-recovery-dir and "
            "--halt-mapping-recovery-validation-dir counts must match"
        )

    if args.source_catalog:
        catalog: pd.DataFrame | Path = Path(args.source_catalog)
    else:
        if not args.halt_rules_dir and not args.listing_events_dir:
            parser.error(
                "provide --source-catalog or at least one --halt-rules-dir/--listing-events-dir"
            )
        catalog = discover_local_jpx_source_catalog(
            halt_rules_dirs=args.halt_rules_dir,
            listing_events_dir=args.listing_events_dir,
        )
    result = produce_japan_g4_halt_history(
        run_id=args.run_id,
        source_catalog=catalog,
        symbol_map_path=args.symbol_map,
        g1_producer_manifest_path=args.g1_producer_manifest,
        g1_validation_manifest_path=args.g1_validation_manifest,
        timestamp_recovery_dir=args.timestamp_recovery_dir,
        official_timestamp_recovery_dir=args.official_timestamp_recovery_dir,
        official_timestamp_recovery_validation_dir=(
            args.official_timestamp_recovery_validation_dir
        ),
        official_timestamp_v14_recovery_dir=(
            args.official_timestamp_v14_recovery_dir
        ),
        official_timestamp_v14_recovery_validation_dir=(
            args.official_timestamp_v14_recovery_validation_dir
        ),
        halt_report_events_dir=args.halt_report_events_dir,
        halt_report_events_validation_dir=args.halt_report_events_validation_dir,
        halt_normalization_repair_dir=args.halt_normalization_repair_dir,
        halt_normalization_repair_validation_dir=(
            args.halt_normalization_repair_validation_dir
        ),
        lifecycle_taxonomy_repair_dir=args.lifecycle_taxonomy_repair_dir,
        lifecycle_taxonomy_repair_validation_dir=(
            args.lifecycle_taxonomy_repair_validation_dir
        ),
        tse_wayback_lifecycle_pairs=list(
            zip(lifecycle_producers, lifecycle_validations, strict=True)
        ),
        halt_mapping_recovery_pairs=list(
            zip(
                args.halt_mapping_recovery_dir,
                args.halt_mapping_recovery_validation_dir,
                strict=True,
            )
        ),
        archive_roots=args.archive_root,
        artifact_root=args.artifact_root,
        start_date=args.start_date,
        end_date=args.end_date,
        max_sources=args.max_sources,
        resume=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "status": result.manifest["status"],
                "counts": result.manifest["counts"],
                "coverage": result.manifest["coverage"],
                "critical_blockers": result.manifest["critical_blockers"],
                "manifest": str(result.manifest_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
