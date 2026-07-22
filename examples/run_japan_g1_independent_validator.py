from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g1_independent_validator import write_japan_g1_independent_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate and publish admitted Japan G1 artifacts.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--producer-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts/japan_g1_validation")
    parser.add_argument("--security-master-dir", required=True)
    parser.add_argument("--reconciliation-dir", required=True)
    parser.add_argument("--active-map-dir", required=True)
    parser.add_argument("--reference-map-dir", required=True)
    args = parser.parse_args()
    result = write_japan_g1_independent_validation(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
        security_master_dir=args.security_master_dir,
        reconciliation_dir=args.reconciliation_dir,
        active_map_dir=args.active_map_dir,
        reference_map_dir=args.reference_map_dir,
    )
    print(json.dumps({
        "run_id": result.run_id,
        "status": result.manifest["status"],
        "critical_blocker_count": result.manifest["critical_blocker_count"],
        "scorecard_eligible": result.manifest["scorecard_eligible"],
        "manifest": str(result.manifest_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
