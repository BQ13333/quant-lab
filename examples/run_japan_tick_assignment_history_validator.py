from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_tick_assignment_history_validator import write_japan_tick_assignment_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate official Japan tick-assignment history.")
    parser.add_argument("--run-id", default="JPN-TICK-ASSIGNMENT-INDEPENDENT-VALIDATION-20260712-V3")
    parser.add_argument(
        "--producer-dir",
        default="artifacts/japan_tick_assignment_history/JPN-TICK-ASSIGNMENT-NOTICES-201401-202606-V3",
    )
    parser.add_argument("--artifact-root", default="artifacts/japan_tick_assignment_history_validation")
    args = parser.parse_args()
    result = write_japan_tick_assignment_validation(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
