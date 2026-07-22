from __future__ import annotations

import argparse

from quant_lab.data.japan_monthly_microstructure import write_japan_monthly_microstructure


DEFAULT_SOURCES = [
    "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200607-200711-EXT-V1",
    "artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2",
    "artifacts/ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200607-200711-EXT-V1",
    "artifacts/ose_monthly_quotations/JPN-OSE-MONTHLY-QUOTATIONS-200712-201306-V1",
    "artifacts/ose_main_monthly_quotations/JPN-OSE-MAIN-MONTHLY-QUOTATIONS-200607-201009-V1",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract resumable Japan lot-size history from accepted official monthly PDFs.")
    parser.add_argument("--run-id", default="JPN-MONTHLY-MICROSTRUCTURE-200607-202606-V5")
    parser.add_argument("--source-dir", action="append", default=None)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--seed-run-dir", default=None)
    parser.add_argument(
        "--combined-membership-dir",
        default="artifacts/japan_combined_monthly_membership_qa/JPN-COMBINED-MONTHLY-MEMBERSHIP-QA-200607-202606-V1",
    )
    args = parser.parse_args()
    result = write_japan_monthly_microstructure(
        run_id=args.run_id,
        source_dirs=args.source_dir or DEFAULT_SOURCES,
        combined_membership_dir=args.combined_membership_dir,
        max_files=args.max_files,
        seed_run_dir=args.seed_run_dir,
    )
    print(f"artifact_dir={result.artifact_dir.resolve()}")
    print(f"status={result.manifest['status']}")
    print(f"coverage={result.manifest['coverage']:.8f}")
    print(f"counts={result.manifest['counts']}")
    print(f"critical_blockers={result.manifest['critical_blockers']}")


if __name__ == "__main__":
    main()
