from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.fx_country_risk import write_fx_country_risk_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build currency, FX, and country-risk/TCA policy artifacts.")
    parser.add_argument(
        "--daily-features",
        default="artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708/daily_features.parquet",
    )
    parser.add_argument("--artifact-root", default="artifacts/fx_country_risk")
    parser.add_argument("--run-id", default="FX-COUNTRY-RISK-MYS-SGP-IDN-U2-20260708")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    args = parser.parse_args()

    artifacts = write_fx_country_risk_artifacts(
        daily_features_path=Path(args.daily_features),
        artifact_root=Path(args.artifact_root),
        run_id=args.run_id,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"Run ID: {artifacts.run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Currency rows: {len(artifacts.currency_reference)}")
    print(f"FX rows: {len(artifacts.fx_rates)}")
    print(f"FX quality: {artifacts.fx_quality_path}")
    print(artifacts.fx_quality.to_string(index=False))


if __name__ == "__main__":
    main()
