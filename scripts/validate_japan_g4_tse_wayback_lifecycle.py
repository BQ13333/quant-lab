from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_tse_wayback_lifecycle_validator import (
    validate_japan_g4_tse_wayback_lifecycle,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate a bounded TSE Wayback lifecycle acquisition."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producer-dir", required=True)
    parser.add_argument(
        "--artifact-root",
        default="artifacts/japan_g4_tse_wayback_lifecycle_validation",
    )
    args = parser.parse_args()
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
