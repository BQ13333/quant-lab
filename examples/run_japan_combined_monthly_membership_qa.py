from __future__ import annotations

import argparse

from quant_lab.data.japan_combined_monthly_membership_qa import (
    run_japan_combined_monthly_membership_qa,
)


DEFAULT_SOURCES = [
    "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200607-200711-EXT-V1",
    "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2",
    "artifacts/ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200607-200711-EXT-V1",
    "artifacts/ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200712-201306-V1",
    "artifacts/ose_main_monthly_quotations/JPN-OSE-MAIN-MONTHLY-QUOTATIONS-200607-201009-V1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently audit the 20-year TSE+OSE monthly membership union.")
    parser.add_argument("--run-id", default="JPN-COMBINED-MONTHLY-MEMBERSHIP-QA-200607-202606-V1")
    parser.add_argument("--source-dir", action="append", dest="source_dirs")
    parser.add_argument("--max-monthly-churn-rate", type=float, default=0.15)
    parser.add_argument("--min-monthly-codes", type=int, default=1000)
    args = parser.parse_args()
    result = run_japan_combined_monthly_membership_qa(
        source_dirs=args.source_dirs or DEFAULT_SOURCES,
        run_id=args.run_id,
        max_monthly_churn_rate=args.max_monthly_churn_rate,
        min_monthly_codes=args.min_monthly_codes,
    )
    print(f"artifact_dir={result.artifact_dir.resolve()}")
    print(f"status={result.manifest['status']}")
    print(f"passed_months={result.manifest['passed_months']}/{result.manifest['planned_months']}")
    print(f"membership_rows={result.manifest['membership_rows']}")
    print(f"anomalies={result.manifest['anomalies']}")


if __name__ == "__main__":
    main()
