from __future__ import annotations

import argparse

from quant_lab.agents.japan_data_quality_loop import run_japan_data_quality_loop


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the Japan/JPX institutional research data lake.")
    parser.add_argument("--run-id", default="JPN-DATA-QUALITY-LOOP-20260711")
    args = parser.parse_args()
    result = run_japan_data_quality_loop(run_id=args.run_id)
    print(f"Run ID: {result.run_id}")
    print(f"Artifact dir: {result.artifact_dir.resolve()}")
    print(f"Overall score: {result.overall_score:.3f}/5.000")
    print(f"Accepted: {result.accepted}")
    print(f"Status: {result.status}")


if __name__ == "__main__":
    main()
