from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_mapping_recovery_v8 import (
    recover_japan_g4_halt_mappings_v8,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sealed Japan G4 V8 halt RIC mapping recovery."
    )
    parser.add_argument(
        "--run-id", default="JPN-G4-HALT-MAPPING-RECOVERY-20260712-V3"
    )
    parser.add_argument(
        "--v8-dir",
        default=(
            "artifacts/japan_g4_halt_history/"
            "JPN-G4-HALTS-200607-202606-V8"
        ),
    )
    parser.add_argument(
        "--quotation-dir",
        default=(
            "artifacts/jpx_monthly_quotations/"
            "JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2"
        ),
    )
    parser.add_argument(
        "--prior-producer-dir",
        default=(
            "artifacts/japan_g4_halt_mapping_recovery/"
            "JPN-G4-HALT-MAPPING-RECOVERY-20260712-V1"
        ),
    )
    parser.add_argument(
        "--prior-validation-dir",
        default=(
            "artifacts/japan_g4_halt_mapping_recovery_validation/"
            "JPN-G4-HALT-MAPPING-RECOVERY-INDEPENDENT-VALIDATION-20260712-V1"
        ),
    )
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g4_halt_mapping_recovery"
    )
    args = parser.parse_args()
    result = recover_japan_g4_halt_mappings_v8(
        run_id=args.run_id,
        v8_dir=args.v8_dir,
        quotation_dir=args.quotation_dir,
        prior_producer_dir=args.prior_producer_dir,
        prior_validation_dir=args.prior_validation_dir,
        artifact_root=args.artifact_root,
    )
    print(
        json.dumps(
            {
                "artifact_dir": str(result.artifact_dir),
                "status": result.manifest["status"],
                "denominator_rows": result.manifest["denominator_rows"],
                "accepted_rows": result.manifest["accepted_rows"],
                "rejected_rows": result.manifest["rejected_rows"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
