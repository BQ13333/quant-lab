from __future__ import annotations

import argparse

from quant_lab.data.jpx_monthly_quotations import run_jpx_monthly_quotations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded official JPX monthly quotation PIT extraction.")
    parser.add_argument("--run-id", default="JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2")
    parser.add_argument("--artifact-root", default="artifacts/jpx_monthly_quotations")
    parser.add_argument("--start-month", default="2007-12")
    parser.add_argument("--end-month", default="2026-06")
    parser.add_argument("--max-files", type=int, default=9)
    parser.add_argument("--pacing-seconds", type=float, default=0.2)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    result = run_jpx_monthly_quotations(
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        start_month=args.start_month,
        end_month=args.end_month,
        max_files=args.max_files,
        pacing_seconds=args.pacing_seconds,
        retries=args.retries,
    )
    print(f"artifact_dir={result.artifact_dir.resolve()}")
    print(f"status={result.manifest['status']}")
    print(f"completed_files={result.manifest['completed_files']}/{result.manifest['planned_files']}")
    print(f"missing_source_urls={result.manifest['missing_source_urls']}")
    print(f"normalized_rows={result.manifest['normalized_rows']}")


if __name__ == "__main__":
    main()
