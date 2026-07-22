from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.pit_features import read_pit_history_partitions
from quant_lab.data.pit_leakage_qa import (
    load_l2_backtest_decision_calendar,
    write_pit_leakage_qa_artifacts,
)


DEFAULT_L2_ARTIFACT = (
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/"
    "05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)


def _parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


def _parse_csv_arg(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_lag_arg(value: str | None) -> dict[str, int] | None:
    if value is None or not value.strip():
        return None
    result: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        key, raw_days = item.split("=", 1)
        result[key.strip()] = int(raw_days.strip())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PIT feature leakage QA against the L2 backtest decision calendar.")
    parser.add_argument("--history-roots", required=True, help="Comma-separated backfill artifact roots or history directories.")
    parser.add_argument("--l2-artifact-dir", default=DEFAULT_L2_ARTIFACT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/pit_leakage_qa")
    parser.add_argument("--universe-tier", default="u2", choices=("u2", "u3", "all"))
    parser.add_argument("--include-countries", default="Malaysia,Singapore,Indonesia")
    parser.add_argument("--price-col", default="adj_close")
    parser.add_argument("--min-rows", type=int, default=None)
    parser.add_argument("--max-missing-ratio", type=float, default=None)
    parser.add_argument("--ffill-limit", type=int, default=5)
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--lag-days", default=None, help="Optional dataset=days overrides, comma-separated.")
    parser.add_argument("--max-staleness-days", default=None, help="Optional dataset=days stale-data caps, comma-separated.")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("PIT-LEAKAGE-QA-%Y%m%d-%H%M%S")
    frames = []
    for root in _parse_paths(args.history_roots):
        frame = read_pit_history_partitions(root)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("no PIT history rows found in supplied roots")
    history = pd.concat(frames, ignore_index=True, sort=False)

    decisions = load_l2_backtest_decision_calendar(
        args.l2_artifact_dir,
        universe_tier=args.universe_tier,
        include_countries=_parse_csv_arg(args.include_countries),
        price_col=args.price_col,
        min_rows=args.min_rows,
        max_missing_ratio=args.max_missing_ratio,
        ffill_limit=args.ffill_limit,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    artifacts = write_pit_leakage_qa_artifacts(
        history=history,
        decisions=decisions,
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        lag_days_by_dataset=_parse_lag_arg(args.lag_days),
        max_staleness_days_by_dataset=_parse_lag_arg(args.max_staleness_days),
    )

    total_violations = (
        int(artifacts.leakage_summary["leakage_violations"].sum())
        if not artifacts.leakage_summary.empty
        else 0
    )
    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Decision rows: {len(artifacts.decision_panel)}")
    print(f"Features checked: {len(artifacts.feature_quality)}")
    print(f"Leakage violations: {total_violations}")
    print(f"Summary: {artifacts.summary_path}")
    if not artifacts.leakage_summary.empty:
        cols = [
            "feature",
            "feature_coverage",
            "leakage_violations",
            "min_available_lag_days",
            "median_available_lag_days",
            "max_available_lag_days",
            "status",
        ]
        print(artifacts.leakage_summary[[col for col in cols if col in artifacts.leakage_summary.columns]].to_string(index=False))


if __name__ == "__main__":
    main()
