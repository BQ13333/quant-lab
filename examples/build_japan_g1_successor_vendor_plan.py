from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g1_successor_vendor_plan import (
    write_japan_g1_successor_vendor_plan,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the bounded Refinitiv lookup plan for official JPX successor edges."
    )
    parser.add_argument(
        "--run-id", default="JPN-G1-SUCCESSOR-VENDOR-PLAN-20260712-V1"
    )
    parser.add_argument(
        "--transition-evidence-dir",
        default=(
            "artifacts/japan_g1_corporate_transition_evidence/"
            "JPN-G1-CORPORATE-TRANSITION-EVIDENCE-20260712-V1"
        ),
    )
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g1_successor_vendor_plan"
    )
    parser.add_argument("--expected-lookup-rows", type=int, default=12)
    args = parser.parse_args()
    result = write_japan_g1_successor_vendor_plan(
        run_id=args.run_id,
        transition_evidence_dir=args.transition_evidence_dir,
        artifact_root=args.artifact_root,
        expected_lookup_rows=args.expected_lookup_rows,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
