from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.adjusted_returns import write_adjusted_return_reconstruction_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild daily adjusted returns from official corporate-action events and daily prices."
    )
    parser.add_argument(
        "--daily-features",
        default="artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708/daily_features.parquet",
    )
    parser.add_argument(
        "--corporate-actions-dir",
        default="artifacts/refinitiv_corporate_actions_events/REFINITIV-CORPACT-EVENTS-U2-MYS-SGP-IDN-20260708",
    )
    parser.add_argument("--artifact-root", default="artifacts/adjusted_return_reconstruction")
    parser.add_argument("--run-id", default="ADJRET-RECON-MYS-SGP-IDN-U2-20260708")
    parser.add_argument("--fx-rates", default=None, help="Optional parquet with date/from_currency/to_currency/rate rows.")
    parser.add_argument("--max-match-lag-days", type=int, default=7)
    parser.add_argument("--factor-reflection-tolerance", type=float, default=0.15)
    parser.add_argument("--min-matched-event-share", type=float, default=0.99)
    parser.add_argument("--max-quarantine-share", type=float, default=0.005)
    args = parser.parse_args()

    artifacts = write_adjusted_return_reconstruction_artifacts(
        daily_features_path=Path(args.daily_features),
        corporate_action_event_dir=Path(args.corporate_actions_dir),
        artifact_root=Path(args.artifact_root),
        run_id=args.run_id,
        fx_rates_path=Path(args.fx_rates) if args.fx_rates else None,
        max_match_lag_days=args.max_match_lag_days,
        factor_reflection_tolerance=args.factor_reflection_tolerance,
        min_matched_event_share=args.min_matched_event_share,
        max_quarantine_share=args.max_quarantine_share,
    )

    print(f"Run ID: {artifacts.run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Normalized events: {len(artifacts.normalized_events)}")
    print(f"Reconciliation rows: {len(artifacts.reconciliation)}")
    print(f"Daily adjusted return rows: {len(artifacts.daily_adjusted_returns)}")
    print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
