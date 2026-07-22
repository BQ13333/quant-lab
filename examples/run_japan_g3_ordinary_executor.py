from __future__ import annotations

import argparse

from quant_lab.data.japan_g3_ordinary_executor import (
    execute_japan_g3_ordinary_plan,
    validate_japan_g3_ordinary_execution_admission,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a bounded immutable Japan G3 ordinary plan.")
    parser.add_argument("--plan-path", required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument(
        "--g1-validation-dir",
        default="artifacts/japan_g1_validation/JPN-G1-MONTHLY-INDEPENDENT-VALIDATION-20260712-V1",
    )
    parser.add_argument(
        "--g2-admission-dir",
        default="artifacts/japan_g2_admission/JPN-G2-STRICT-ADMISSION-20260712-V3",
    )
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_japan_g3_ordinary")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-plan-rows", type=int, default=1)
    parser.add_argument("--sleep-seconds", type=float, default=0.4)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    validate_japan_g3_ordinary_execution_admission(
        args.plan_path,
        args.expected_plan_sha256,
        args.g1_validation_dir,
        args.g2_admission_dir,
        args.max_plan_rows,
    )
    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        result = execute_japan_g3_ordinary_plan(
            adapter.ld,
            plan_path=args.plan_path,
            expected_plan_sha256=args.expected_plan_sha256,
            g1_validation_dir=args.g1_validation_dir,
            g2_admission_dir=args.g2_admission_dir,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            max_plan_rows=args.max_plan_rows,
            sleep_seconds=args.sleep_seconds,
            resume=not args.no_resume,
        )
    finally:
        adapter.close()
    print(f"Artifact dir: {result.artifact_dir}")
    print(result.quality.to_string(index=False))


if __name__ == "__main__":
    main()
