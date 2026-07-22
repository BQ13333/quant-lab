from __future__ import annotations

import argparse

from quant_lab.data.japan_ohlcv_recovery_merge import merge_japan_ohlcv_recovery


def main() -> None:
    parser = argparse.ArgumentParser(description="Guardedly merge a positive staged Japan OHLCV recovery row.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--plan-row-id", required=True)
    parser.add_argument("--stage-artifact-dir", required=True)
    parser.add_argument("--stage-lake-root", required=True)
    parser.add_argument("--production-artifact-dir", required=True)
    parser.add_argument("--production-lake-root", required=True)
    parser.add_argument("--expected-stage-plan-sha256", required=True)
    parser.add_argument("--expected-production-plan-sha256", required=True)
    args = parser.parse_args()
    result = merge_japan_ohlcv_recovery(**vars(args))
    print(f"manifest={result.manifest_path}")
    print(f"status={result.manifest['status']}")
    print(f"merged_long_ohlcv_rows={result.manifest['merged_long_ohlcv_rows']}")


if __name__ == "__main__":
    main()
