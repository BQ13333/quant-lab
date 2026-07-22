from __future__ import annotations

import argparse

from quant_lab.data.japan_monthly_microstructure_validator import (
    write_japan_monthly_microstructure_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate the Japan monthly lot-size producer.")
    parser.add_argument("--run-id", default="JPN-MONTHLY-MICROSTRUCTURE-INDEPENDENT-VALIDATION-20260712-V1")
    parser.add_argument(
        "--producer-dir",
        default="artifacts/japan_monthly_microstructure/JPN-MONTHLY-MICROSTRUCTURE-200607-202606-V5",
    )
    args = parser.parse_args()
    result = write_japan_monthly_microstructure_validation(
        run_id=args.run_id,
        producer_dir=args.producer_dir,
    )
    print(f"artifact_dir={result.artifact_dir.resolve()}")
    print(f"status={result.manifest['status']}")
    print(f"critical_blockers={result.manifest['critical_blockers']}")
    print(f"counts={result.manifest['counts']}")


if __name__ == "__main__":
    main()
