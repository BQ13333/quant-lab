from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g4_halt_report_normalization_repair import (
    run_halt_report_normalization_repair,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the bounded Japan G4 halt normalization repair packet.")
    parser.add_argument("--run-id", default="JPN-G4-HALT-REPORT-NORMALIZATION-REPAIR-20260712-V1")
    parser.add_argument("--source-2012-02", default="artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V8/inputs/tse_wayback_lifecycle/packets/packet-0001/producer/raw/pdfs/2012-02.pdf")
    parser.add_argument("--source-2013-09", default="artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V8/inputs/tse_wayback_lifecycle/packets/packet-0002/producer/raw/pdfs/2013-09.pdf")
    parser.add_argument("--source-2020-10", default="artifacts/japan_g4_halt_report_events/JPN-G4-HALT-REPORT-EVENTS-20260712-V2/raw/lifecycle/2020-10.pdf")
    parser.add_argument("--artifact-root", default="artifacts/japan_g4_halt_report_normalization_repair")
    args = parser.parse_args()
    result = run_halt_report_normalization_repair(
        run_id=args.run_id,
        source_paths={
            "2012-02": args.source_2012_02,
            "2013-09": args.source_2013_09,
            "2020-10": args.source_2020_10,
        },
        artifact_root=args.artifact_root,
    )
    print(json.dumps({"artifact_dir": str(result.artifact_dir), **result.manifest}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
