from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.corporate_action_reconciliation import write_corporate_action_reconciliation_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Align official corporate-action events to price dates.")
    parser.add_argument(
        "--daily-features",
        default="artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708/daily_features.parquet",
    )
    parser.add_argument(
        "--corporate-actions-dir",
        default="artifacts/refinitiv_corporate_actions_events/REFINITIV-CORPACT-EVENTS-U2-MYS-SGP-IDN-20260708",
    )
    parser.add_argument("--artifact-root", default="artifacts/corporate_action_reconciliation")
    parser.add_argument("--run-id", default="CORPACT-RECON-MYS-SGP-IDN-U2-20260708")
    parser.add_argument("--max-match-lag-days", type=int, default=7)
    args = parser.parse_args()

    artifacts = write_corporate_action_reconciliation_artifacts(
        daily_features_path=Path(args.daily_features),
        corporate_action_event_dir=Path(args.corporate_actions_dir),
        artifact_root=Path(args.artifact_root),
        run_id=args.run_id,
        max_match_lag_days=args.max_match_lag_days,
    )

    print(f"Run ID: {artifacts.run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Event matches: {len(artifacts.event_matches)}")
    print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
