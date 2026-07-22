from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.japan_g1_reconciliation_qa import (
    DEFAULT_JAPAN_G1_QA_ARTIFACT_ROOT,
    DEFAULT_JAPAN_G1_SOURCE_DIR,
    write_japan_g1_reconciliation_qa_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable Japan G1 denominator-difference QA artifact."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_JAPAN_G1_SOURCE_DIR)
    parser.add_argument(
        "--artifact-root", type=Path, default=DEFAULT_JAPAN_G1_QA_ARTIFACT_ROOT
    )
    parser.add_argument("--run-id")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    artifacts = write_japan_g1_reconciliation_qa_artifacts(
        source_dir=args.source_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
    )
    counts = artifacts.manifest["counts"]
    print(f"Artifact: {artifacts.artifact_dir}")
    print(f"Classified rows: {counts['classified_rows']}")
    print(f"Public unmatched: {counts['public_in_scope_unmatched_rows']}")
    print(f"Active-master-only: {counts['active_master_only_rows']}")
    print(f"Truly unresolved: {counts['truly_unresolved_rows']}")
    print("Historical PIT claimed: False")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
