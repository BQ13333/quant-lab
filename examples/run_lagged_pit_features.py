from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.pit_features import read_pit_history_partitions, write_lagged_pit_feature_artifacts


def _parse_paths(value: str) -> list[Path]:
    return [Path(part.strip()) for part in value.split(",") if part.strip()]


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
    parser = argparse.ArgumentParser(description="Build leakage-safe lagged PIT feature panels from history artifacts.")
    parser.add_argument("--history-roots", required=True, help="Comma-separated backfill artifact roots or history directories.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/pit_lagged_features")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--decision-dates", default=None, help="Optional comma-separated decision dates.")
    parser.add_argument("--lag-days", default=None, help="Optional dataset=days overrides, comma-separated.")
    parser.add_argument("--max-staleness-days", default=None, help="Optional dataset=days stale-data caps, comma-separated.")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("PIT-LAGGED-FEATURES-%Y%m%d-%H%M%S")
    frames = []
    for root in _parse_paths(args.history_roots):
        frame = read_pit_history_partitions(root)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        raise ValueError("no PIT history rows found in supplied roots")
    history = pd.concat(frames, ignore_index=True, sort=False)
    decision_dates = [item.strip() for item in args.decision_dates.split(",") if item.strip()] if args.decision_dates else None

    artifacts = write_lagged_pit_feature_artifacts(
        history,
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        decision_dates=decision_dates,
        start_date=args.start_date,
        end_date=args.end_date,
        lag_days_by_dataset=_parse_lag_arg(args.lag_days),
        max_staleness_days_by_dataset=_parse_lag_arg(args.max_staleness_days),
    )

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Feature rows: {len(artifacts.features)}")
    print(f"Feature columns: {len(artifacts.features.columns)}")
    print(f"Feature parquet: {artifacts.features_parquet_path}")
    if not artifacts.quality.empty:
        print(artifacts.quality.to_string(index=False))


if __name__ == "__main__":
    main()
