from __future__ import annotations

import argparse

from quant_lab.data.jpx_monthly_membership_qa import run_jpx_monthly_membership_qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Run independent continuity QA on JPX monthly PIT membership snapshots.")
    parser.add_argument(
        "--source-dir",
        default="artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2",
    )
    parser.add_argument("--run-id", default="JPN-JPX-MONTHLY-MEMBERSHIP-QA-V2")
    parser.add_argument("--artifact-root", default="artifacts/jpx_monthly_membership_qa")
    parser.add_argument("--max-monthly-churn-rate", type=float, default=0.15)
    args = parser.parse_args()
    result = run_jpx_monthly_membership_qa(
        source_dir=args.source_dir,
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        max_monthly_churn_rate=args.max_monthly_churn_rate,
    )
    print(f"artifact_dir={result.artifact_dir.resolve()}")
    print(f"status={result.manifest['status']}")
    print(f"completed_files={result.manifest['completed_files']}/{result.manifest['planned_files']}")
    print(f"completed_months={result.manifest['completed_months']}/{result.manifest['planned_months']}")
    print(f"anomalies={result.manifest['anomalies']}")


if __name__ == "__main__":
    main()
