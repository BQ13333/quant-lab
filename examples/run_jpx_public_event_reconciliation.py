from __future__ import annotations

import argparse
import json

from quant_lab.data.jpx_public_event_reconciliation import write_jpx_public_event_reconciliation


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-check recent official JPX public listing events against the reference ledger.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--public-events-dir", required=True)
    parser.add_argument("--reference-map-csv", required=True)
    parser.add_argument("--security-master-csv", default=None)
    parser.add_argument("--artifact-root", default="artifacts/jpx_public_event_reconciliation")
    args = parser.parse_args()
    result = write_jpx_public_event_reconciliation(
        run_id=args.run_id,
        public_events_dir=args.public_events_dir,
        reference_map_csv=args.reference_map_csv,
        security_master_csv=args.security_master_csv,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"run_id": result.run_id, "status": result.manifest["status"], "counts": result.manifest["counts"], "manifest": str(result.manifest_path.resolve())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
