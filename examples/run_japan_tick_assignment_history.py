from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_tick_assignment_history import write_japan_tick_assignment_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Build official JPX PIT narrow-tick assignment notice history.")
    parser.add_argument("--run-id", default="JPN-TICK-ASSIGNMENT-NOTICES-201401-202606-V3")
    parser.add_argument("--artifact-root", default="artifacts/japan_tick_assignment_history")
    parser.add_argument(
        "--checkpoint-path",
        default="artifacts/japan_market_rule_history/JPN-MARKET-RULE-HISTORY-200607-202606-V1/assignment_snapshots.csv",
    )
    parser.add_argument("--cutoff-date", default="2026-06-30")
    args = parser.parse_args()
    result = write_japan_tick_assignment_history(
        run_id=args.run_id,
        checkpoint_path=args.checkpoint_path,
        artifact_root=args.artifact_root,
        cutoff_date=args.cutoff_date,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
