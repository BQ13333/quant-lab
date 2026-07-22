from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.immutable_hash_inventory import write_immutable_hash_inventory


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a write-once SHA-256 inventory for data-quality admission.")
    parser.add_argument("--inventory-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    args = parser.parse_args()

    result = write_immutable_hash_inventory(
        input_paths=args.input,
        output_path=args.output,
        inventory_id=args.inventory_id,
    )
    print(f"inventory={result.inventory_path}")
    print(f"files={result.file_count}")


if __name__ == "__main__":
    main()
