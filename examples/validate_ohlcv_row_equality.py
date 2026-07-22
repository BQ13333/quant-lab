from __future__ import annotations

import argparse
import json

from quant_lab.data.ohlcv_row_equality import validate_ohlcv_row_equality


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exact OHLCV source/compaction row equality by year.")
    parser.add_argument("source_root")
    parser.add_argument("compacted_root")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default="artifacts/ohlcv_row_equality")
    args = parser.parse_args()
    result = validate_ohlcv_row_equality(
        source_root=args.source_root,
        compacted_root=args.compacted_root,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
    )
    print(json.dumps(result.manifest, indent=2))


if __name__ == "__main__":
    main()
