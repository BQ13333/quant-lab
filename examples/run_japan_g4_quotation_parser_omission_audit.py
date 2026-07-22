from __future__ import annotations

import argparse

from quant_lab.data.japan_g4_quotation_parser_omission_audit import (
    produce_japan_g4_quotation_parser_omission_audit,
)
from quant_lab.data.japan_g4_quotation_parser_omission_audit_validator import (
    validate_japan_g4_quotation_parser_omission_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-run-id", required=True)
    parser.add_argument(
        "--mapping-v3-dir",
        default="artifacts/japan_g4_halt_mapping_recovery/JPN-G4-HALT-MAPPING-RECOVERY-20260712-V3",
    )
    parser.add_argument(
        "--halt-v16-dir",
        default="artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V16",
    )
    args = parser.parse_args()
    producer = produce_japan_g4_quotation_parser_omission_audit(
        run_id=args.run_id,
        mapping_v3_dir=args.mapping_v3_dir,
        halt_v16_dir=args.halt_v16_dir,
    )
    validation_path, validation = validate_japan_g4_quotation_parser_omission_audit(
        producer_dir=producer.artifact_dir,
        run_id=args.validation_run_id,
    )
    print(f"producer={producer.manifest_path}")
    print(f"producer_status={producer.manifest['status']}")
    print(f"validation={validation_path}")
    print(f"validation_status={validation['status']}")


if __name__ == "__main__":
    main()
