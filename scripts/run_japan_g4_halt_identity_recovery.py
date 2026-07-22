from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_identity_recovery import (
    recover_japan_g4_halt_identities,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Produce the bounded Japan G4 halt identity recovery plan."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--unresolved-gaps", required=True)
    parser.add_argument("--transition-validation-dir", required=True)
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g4_halt_identity_recovery"
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    result = recover_japan_g4_halt_identities(
        run_id=args.run_id,
        unresolved_gaps_path=args.unresolved_gaps,
        transition_validation_dir=args.transition_validation_dir,
        artifact_root=args.artifact_root,
        timeout=args.timeout,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
