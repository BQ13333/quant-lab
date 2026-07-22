from __future__ import annotations

import argparse

from quant_lab.data.japan_g3_sector_gap_plan import build_japan_g3_sector_gap_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the immutable full-denominator Japan G3 sector-routing gap plan.")
    parser.add_argument("--pit-plan-dir", required=True)
    parser.add_argument("--existing-sector-dir", required=True)
    parser.add_argument("--artifact-root", default="artifacts/japan_g3_sector_gap_plan")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--batch-size", type=int, default=25)
    args = parser.parse_args()
    result = build_japan_g3_sector_gap_plan(
        pit_plan_dir=args.pit_plan_dir,
        existing_sector_dir=args.existing_sector_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        batch_size=args.batch_size,
    )
    print(f"artifact_dir={result.artifact_dir}")
    print(f"missing_rics={len(result.missing_rics)}")
    print(f"plan_rows={len(result.plan)}")


if __name__ == "__main__":
    main()
