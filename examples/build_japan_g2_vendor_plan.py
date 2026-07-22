from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g2_vendor_plan import build_japan_g2_vendor_plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Build immutable Japan G2 corporate-action and liquidity plans.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--active-symbol-map", required=True)
    parser.add_argument("--reference-symbol-map", required=True)
    parser.add_argument("--artifact-root", default="artifacts/japan_g2_vendor_plan")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument(
        "--corporate-action-artifact-root",
        default="artifacts/refinitiv_corporate_actions_events",
    )
    parser.add_argument(
        "--corporate-action-run-id",
        default="REFINITIV-JPN-G2-CORPACT-20060101-20260706-V1",
    )
    args = parser.parse_args()
    result = build_japan_g2_vendor_plan(
        active_symbol_map_path=args.active_symbol_map,
        reference_symbol_map_path=args.reference_symbol_map,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        corporate_action_artifact_root=args.corporate_action_artifact_root,
        corporate_action_run_id=args.corporate_action_run_id,
    )
    print(json.dumps(result.manifest, indent=2))


if __name__ == "__main__":
    main()
