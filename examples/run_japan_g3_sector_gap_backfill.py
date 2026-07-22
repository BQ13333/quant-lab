from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from quant_lab.data.japan_g3_sector_gap_executor import execute_japan_g3_sector_gap_plan
from quant_lab.data.vendor_adapters import RefinitivAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the bounded resumable Japan G3 sector-gap backfill.")
    parser.add_argument("--gap-plan-dir", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--baseline-sector-dir", required=True)
    parser.add_argument("--g1-validation-dir", required=True)
    parser.add_argument("--g2-admission-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_current_snapshot")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-plan-rows", type=int, default=1)
    parser.add_argument("--session", default="desktop.workspace")
    args = parser.parse_args()
    plan = Path(args.gap_plan_dir) / "sector_gap_plan.csv"
    if hashlib.sha256(plan.read_bytes()).hexdigest() != args.expected_plan_sha256:
        raise ValueError("sector-gap plan checksum mismatch")
    adapter = RefinitivAdapter(session_name=args.session)
    adapter.open()
    try:
        result = execute_japan_g3_sector_gap_plan(
            adapter.ld,
            gap_plan_dir=args.gap_plan_dir,
            expected_plan_sha256=args.expected_plan_sha256,
            baseline_sector_dir=args.baseline_sector_dir,
            g1_validation_dir=args.g1_validation_dir,
            g2_admission_dir=args.g2_admission_dir,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            max_plan_rows=args.max_plan_rows,
        )
    finally:
        adapter.close()
    print(f"artifact_dir={result.artifact_dir}")
    print(f"profile_rics={result.manifest['profile_rics']}")
    print(f"status={result.manifest['status']}")


if __name__ == "__main__":
    main()
