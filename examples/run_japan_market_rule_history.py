from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_market_rule_history import write_japan_market_rule_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Build source-backed Japan tick and normal price-limit rule continuity.")
    parser.add_argument("--run-id", default="JPN-MARKET-RULE-HISTORY-200607-202606-V2")
    parser.add_argument("--artifact-root", default="artifacts/japan_market_rule_history")
    args = parser.parse_args()
    result = write_japan_market_rule_history(run_id=args.run_id, artifact_root=args.artifact_root)
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
