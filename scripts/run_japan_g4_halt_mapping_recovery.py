from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_mapping_recovery import (
    recover_japan_g4_halt_mappings,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the bounded Japan G4 halt mapping-recovery packet."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--v5-dir", required=True)
    parser.add_argument("--quotation-dir", required=True)
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g4_halt_mapping_recovery"
    )
    args = parser.parse_args()
    result = recover_japan_g4_halt_mappings(
        run_id=args.run_id,
        v5_dir=args.v5_dir,
        quotation_dir=args.quotation_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
