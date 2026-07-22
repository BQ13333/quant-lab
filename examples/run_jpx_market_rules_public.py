from __future__ import annotations

import argparse
import json

from quant_lab.data.jpx_market_rules_public import run_jpx_market_rules_public


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official current JPX market-rule evidence without overstating historical completeness.")
    parser.add_argument("--run-id", default="FX-CALENDAR-TCA-JPN-20260711")
    parser.add_argument("--artifact-root", default="artifacts/fx_calendar_tca")
    args = parser.parse_args()
    result = run_jpx_market_rules_public(run_id=args.run_id, artifact_root=args.artifact_root)
    print(json.dumps({"run_id": result.run_id, "status": result.manifest["status"], "counts": result.manifest["counts"], "critical_blockers": result.manifest["critical_blockers"], "manifest": str(result.manifest_path.resolve())}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
