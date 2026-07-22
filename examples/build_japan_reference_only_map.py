from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.japan_reference_only_map import build_japan_reference_only_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Partition admitted JPX maps into disjoint active and reference-only windows.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--active-map-dir", type=Path, required=True)
    parser.add_argument("--full-reference-map-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/refinitiv_symbol_map"))
    args = parser.parse_args()
    result = build_japan_reference_only_map(
        active_map_dir=args.active_map_dir,
        full_reference_map_dir=args.full_reference_map_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
    )
    print(f"artifact_dir={result.artifact_dir}")
    print(f"reference_only_windows={len(result.symbol_map)}")
    print(f"reference_only_rics={result.symbol_map['refinitiv_ric'].nunique()}")


if __name__ == "__main__":
    main()
