from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.japan_g3_admission import (
    JapanG3AdmissionInputs,
    JapanG3TerminalRun,
    write_japan_g3_admission,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable Japan G3 aggregate admission packet from local artifacts only."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/japan_g3_admission"))
    parser.add_argument("--g1-validation-dir", type=Path, required=True)
    parser.add_argument("--g2-admission-dir", type=Path, required=True)
    parser.add_argument("--pit-plan-dir", type=Path, required=True)
    parser.add_argument("--execution-plan-dir", type=Path, required=True)
    parser.add_argument("--sector-routing-path", type=Path, required=True)
    parser.add_argument("--decision-calendar-path", type=Path, required=True)
    for lane in ("fundamentals", "consensus", "earnings", "analyst-detail"):
        parser.add_argument(f"--{lane}-producer-dir", type=Path, required=True)
        parser.add_argument(f"--{lane}-validator-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    terminal_runs = tuple(
        JapanG3TerminalRun(
            lane=lane,
            producer_dir=getattr(args, f"{lane.replace('_', '_')}_producer_dir"),
            validator_dir=getattr(args, f"{lane.replace('_', '_')}_validator_dir"),
        )
        for lane in ("fundamentals", "consensus", "earnings", "analyst_detail")
    )
    result = write_japan_g3_admission(
        JapanG3AdmissionInputs(
            g1_validation_dir=args.g1_validation_dir,
            g2_admission_dir=args.g2_admission_dir,
            pit_plan_dir=args.pit_plan_dir,
            execution_plan_dir=args.execution_plan_dir,
            sector_routing_path=args.sector_routing_path,
            decision_calendar_path=args.decision_calendar_path,
            terminal_runs=terminal_runs,
        ),
        artifact_root=args.artifact_root,
        run_id=args.run_id,
    )
    print(f"artifact_dir={result.artifact_dir}")
    print(f"admission_passed={str(result.admitted).lower()}")
    print(f"scorecard_eligible={str(result.manifest['scorecard_eligible']).lower()}")
    return 0 if result.admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
