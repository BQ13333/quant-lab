from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.agents.usa_data_quality_loop import default_usa_data_quality_paths, run_usa_data_quality_loop


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the USA institutional 5/5 data-quality gate.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--artifact-root", default="artifacts/institutional_data_quality")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--acceptance-score", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root)
    paths = default_usa_data_quality_paths(project_root)
    result = run_usa_data_quality_loop(
        paths=paths,
        artifact_root=project_root / args.artifact_root,
        run_id=args.run_id,
        project_root=project_root,
        acceptance_score=args.acceptance_score,
    )
    print(f"run_id={result.run_id}")
    print(f"status={result.status}")
    print(f"accepted={result.accepted}")
    print(f"overall_score={result.overall_score:.3f}/5.000")
    print(f"summary={result.summary_path.resolve()}")
    print(f"next_queue={result.next_queue_path.resolve()}")


if __name__ == "__main__":
    main()
