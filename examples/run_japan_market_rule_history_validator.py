from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_market_rule_history_validator import write_japan_market_rule_history_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate Japan tick and normal price-limit rule continuity.")
    parser.add_argument("--run-id", default="JPN-MARKET-RULE-HISTORY-INDEPENDENT-VALIDATION-20260712-V2-R1")
    parser.add_argument("--producer-dir", default="artifacts/japan_market_rule_history/JPN-MARKET-RULE-HISTORY-200607-202606-V2")
    parser.add_argument("--artifact-root", default="artifacts/japan_market_rule_history_validation")
    args = parser.parse_args()
    result = write_japan_market_rule_history_validation(run_id=args.run_id, producer_dir=args.producer_dir, artifact_root=args.artifact_root)
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
