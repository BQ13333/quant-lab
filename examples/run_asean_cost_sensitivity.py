from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.research.asean import run_asean_cost_sensitivity


def latest_artifact(root: Path) -> Path:
    candidates = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith("ASEAN-"))
    if not candidates:
        raise FileNotFoundError(f"no ASEAN artifacts found under {root}")
    return candidates[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=None)
    args = parser.parse_args()

    artifact_dir = args.artifact_dir or latest_artifact(Path("artifacts/asean"))
    result = run_asean_cost_sensitivity(artifact_dir)

    print("Artifact:", result.artifact_dir.resolve())
    print("\nBest by cost:")
    print(result.best_by_cost.to_string(index=False))


if __name__ == "__main__":
    main()
