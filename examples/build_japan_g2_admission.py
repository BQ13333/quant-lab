from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.japan_g2_admission import (
    JapanG2AdmissionInputs,
    JapanG2OHLCVRun,
    write_japan_g2_admission,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the immutable Japan G2 aggregate admission sidecar from local artifacts only."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts/japan_g2_admission"))
    parser.add_argument("--g1-security-master-dir", type=Path, required=True)
    parser.add_argument("--g1-validation-dir", type=Path, required=True)
    parser.add_argument("--active-symbol-map-dir", type=Path, required=True)
    parser.add_argument("--reference-symbol-map-dir", type=Path, required=True)
    parser.add_argument("--active-historical-dir", type=Path, required=True)
    parser.add_argument("--active-depth-dir", type=Path, required=True)
    parser.add_argument("--active-compaction-dir", type=Path, required=True)
    parser.add_argument("--active-hash-inventory", type=Path, required=True)
    parser.add_argument("--active-row-equality-dir", type=Path, required=True)
    parser.add_argument("--reference-historical-dir", type=Path, required=True)
    parser.add_argument("--reference-depth-dir", type=Path, required=True)
    parser.add_argument("--reference-compaction-dir", type=Path, required=True)
    parser.add_argument("--reference-hash-inventory", type=Path, required=True)
    parser.add_argument("--reference-row-equality-dir", type=Path, required=True)
    parser.add_argument("--corporate-actions-dir", type=Path, required=True)
    parser.add_argument("--adjusted-returns-dir", type=Path, required=True)
    parser.add_argument("--liquidity-dir", type=Path, required=True)
    parser.add_argument("--component-hash-inventory", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = JapanG2AdmissionInputs(
        g1_security_master_dir=args.g1_security_master_dir,
        g1_validation_dir=args.g1_validation_dir,
        active_symbol_map_dir=args.active_symbol_map_dir,
        reference_symbol_map_dir=args.reference_symbol_map_dir,
        active_ohlcv=JapanG2OHLCVRun(
            historical_dir=args.active_historical_dir,
            depth_dir=args.active_depth_dir,
            compaction_dir=args.active_compaction_dir,
            hash_inventory_path=args.active_hash_inventory,
            row_equality_dir=args.active_row_equality_dir,
        ),
        reference_ohlcv=JapanG2OHLCVRun(
            historical_dir=args.reference_historical_dir,
            depth_dir=args.reference_depth_dir,
            compaction_dir=args.reference_compaction_dir,
            hash_inventory_path=args.reference_hash_inventory,
            row_equality_dir=args.reference_row_equality_dir,
        ),
        corporate_actions_dir=args.corporate_actions_dir,
        adjusted_returns_dir=args.adjusted_returns_dir,
        liquidity_dir=args.liquidity_dir,
        component_hash_inventory_path=args.component_hash_inventory,
    )
    result = write_japan_g2_admission(
        inputs=inputs,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
    )
    print(f"artifact_dir={result.artifact_dir}")
    print(f"admission_passed={str(result.admitted).lower()}")
    print(f"scorecard_eligible={str(result.manifest['scorecard_eligible']).lower()}")


if __name__ == "__main__":
    main()
