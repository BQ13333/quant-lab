from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g1_pit_materializer import write_japan_g1_pit_materializer


def main() -> None:
    parser = argparse.ArgumentParser(description="Build producer-only Japan G1 PIT candidate artifacts.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jquants-dir", required=True)
    parser.add_argument("--enrichment-dir", required=True)
    parser.add_argument("--valid-time-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts/japan_g1_pit_materializer")
    args = parser.parse_args()
    result = write_japan_g1_pit_materializer(
        run_id=args.run_id,
        jquants_dir=args.jquants_dir,
        enrichment_dir=args.enrichment_dir,
        valid_time_dir=args.valid_time_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps({
        "run_id": result.run_id,
        "status": result.manifest["status"],
        "historical_pit_ready": result.manifest.get("historical_pit_ready", False),
        "scorecard_eligible": False,
        "critical_blockers": result.manifest.get("critical_blockers", []),
        "manifest": str(result.manifest_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
