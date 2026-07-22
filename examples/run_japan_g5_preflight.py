from __future__ import annotations

import argparse

from quant_lab.agents.japan_data_quality_loop import default_japan_data_quality_paths
from quant_lab.data.japan_g5_preflight import write_japan_g5_preflight_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the fail-closed Japan G5 preflight and hash inventory.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--run-id", default="PIT-LEAKAGE-QA-JPN-U2-20260711")
    parser.add_argument(
        "--qa-manifests",
        default=(
            "artifacts/refinitiv_japan_g3_fundamentals/REFINITIV-JPN-G3-FUNDAMENTALS-QA-20260712/manifest.json,"
            "artifacts/refinitiv_japan_g3_ibes_events/REFINITIV-JPN-G3-IBES-EVENTS-QA-20260712/manifest.json"
        ),
    )
    args = parser.parse_args()
    qa_manifests = [path.strip() for path in args.qa_manifests.split(",") if path.strip()]
    result = write_japan_g5_preflight_artifacts(
        paths=default_japan_data_quality_paths(args.project_root),
        run_id=args.run_id,
        qa_manifest_paths=qa_manifests,
    )
    print(f"Artifact dir: {result.artifact_dir}")
    print(f"Upstream ready: {int(result.upstream_readiness['ready'].sum())}/{len(result.upstream_readiness)}")
    print(f"Timestamp QA rows: {len(result.timestamp_audit)}")
    print(f"Hash coverage: {int(result.hash_coverage['exists'].sum())}/{len(result.hash_coverage)}")
    print(f"Immutable inventory: {result.immutable_hashes_path}")


if __name__ == "__main__":
    main()
