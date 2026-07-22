from __future__ import annotations

import argparse

from quant_lab.data.ose_main_monthly_quotations import run_ose_main_monthly_quotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded official pre-integration OSE main-market monthly extraction.")
    parser.add_argument("--run-id", default="JPN-OSE-MAIN-MONTHLY-QUOTATIONS-200607-201009-V1")
    parser.add_argument("--start-month", default="2006-07")
    parser.add_argument("--end-month", default="2010-09")
    parser.add_argument("--max-files", type=int, default=45)
    parser.add_argument("--pacing-seconds", type=float, default=0.05)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    result = run_ose_main_monthly_quotations(
        run_id=args.run_id,
        start_month=args.start_month,
        end_month=args.end_month,
        max_files=args.max_files,
        pacing_seconds=args.pacing_seconds,
        retries=args.retries,
    )
    print(f"artifact_dir={result.artifact_dir.resolve()}")
    print(f"status={result.manifest['status']}")
    print(f"completed_files={result.manifest['completed_files']}/{result.manifest['planned_files']}")
    print(f"normalized_rows={result.manifest['normalized_rows']}")


if __name__ == "__main__":
    main()
