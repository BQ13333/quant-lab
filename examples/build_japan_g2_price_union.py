from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.japan_g2_price_union import build_japan_g2_price_union


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an immutable, memory-bounded Japan G2 active/reference price union."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/japan_g2_price_union"))
    parser.add_argument("--active-compacted-root", type=Path, required=True)
    parser.add_argument("--active-compaction-manifest", type=Path, required=True)
    parser.add_argument("--active-row-equality-manifest", type=Path, required=True)
    parser.add_argument("--reference-compacted-root", type=Path, required=True)
    parser.add_argument("--reference-compaction-manifest", type=Path, required=True)
    parser.add_argument("--reference-row-equality-manifest", type=Path, required=True)
    parser.add_argument(
        "--key-column",
        action="append",
        dest="key_columns",
        help="Logical uniqueness key column; repeat for composite keys (default: infer ticker/ric and date).",
    )
    parser.add_argument("--batch-size", type=int, default=8_192)
    parser.add_argument("--copy-chunk-size", type=int, default=1024 * 1024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_japan_g2_price_union(
        active_compacted_root=args.active_compacted_root,
        active_compaction_manifest=args.active_compaction_manifest,
        active_row_equality_manifest=args.active_row_equality_manifest,
        reference_compacted_root=args.reference_compacted_root,
        reference_compaction_manifest=args.reference_compaction_manifest,
        reference_row_equality_manifest=args.reference_row_equality_manifest,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        key_columns=args.key_columns,
        batch_size=args.batch_size,
        copy_chunk_size=args.copy_chunk_size,
    )
    print(f"artifact_dir={result.artifact_dir}")
    print(f"dataset_root={result.dataset_root}")
    print(f"output_rows={result.manifest['output_rows']}")
    print("promotion_allowed=false")


if __name__ == "__main__":
    main()
