from __future__ import annotations

import argparse

from quant_lab.data.japan_g4_zero_equity_publication_timestamp_audit import (
    produce_japan_g4_zero_equity_publication_timestamp_audit,
)
from quant_lab.data.japan_g4_zero_equity_publication_timestamp_audit_validator import (
    validate_japan_g4_zero_equity_publication_timestamp_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit retained V16 bytes for the 2014-11 zero-equity publication timestamp."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-run-id", required=True)
    parser.add_argument("--v16-root", required=True)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/japan_g4_zero_equity_publication_timestamp_audit",
    )
    parser.add_argument(
        "--validation-artifact-root",
        default="artifacts/japan_g4_zero_equity_publication_timestamp_validation",
    )
    args = parser.parse_args()
    result = produce_japan_g4_zero_equity_publication_timestamp_audit(
        run_id=args.run_id,
        v16_root=args.v16_root,
        artifact_root=args.artifact_root,
    )
    _, validation = validate_japan_g4_zero_equity_publication_timestamp_audit(
        producer_dir=result.artifact_dir,
        validation_run_id=args.validation_run_id,
        artifact_root=args.validation_artifact_root,
    )
    print(result.manifest_path)
    print(validation["status"])


if __name__ == "__main__":
    main()
