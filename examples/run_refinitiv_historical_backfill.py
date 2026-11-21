from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.historical_backfill import (
    default_historical_lake_root,
    execute_refinitiv_historical_plan,
    parse_country_filter,
    refinitiv_universe_from_symbol_map,
    write_refinitiv_historical_plan,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


DEFAULT_ASIA_COUNTRIES = (
    "Malaysia,Singapore,Indonesia,"
    "Hong Kong,South Korea,Japan"
)


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("REFINITIV-HISTORICAL-%Y%m%d-%H%M%S")


def _load_plan_row_ids(path: str | None, *, label: str, required: bool = False) -> set[str]:
    if not path:
        if required:
            raise ValueError(f"{label} plan-row CSV is required")
        return set()
    source = Path(path).expanduser()
    if not source.exists():
        if required:
            raise FileNotFoundError(f"{label} plan-row CSV does not exist: {path}")
        return set()
    frame = pd.read_csv(source, low_memory=False)
    if "plan_row_id" not in frame.columns:
        raise ValueError(f"{label} plan-row CSV is missing plan_row_id: {path}")
    plan_row_ids = {str(value) for value in frame["plan_row_id"].dropna().astype(str) if str(value)}
    return plan_row_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a resumable Refinitiv historical OHLCV data-lake backfill.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--symbol-map-csv", required=True)
    parser.add_argument("--field-catalog-csv", default=None)
    parser.add_argument("--artifact-root", default="artifacts/historical_backfill")
    parser.add_argument("--lake-root", default=None)
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--window-years", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--max-plan-rows", type=int, default=None)
    parser.add_argument("--include-countries", default=DEFAULT_ASIA_COUNTRIES)
    parser.add_argument("--exclude-countries", default=None)
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--refinitiv-session", default="desktop.workspace")
    parser.add_argument("--refinitiv-config", default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--existing-plan-only",
        action="store_true",
        help="Execute the immutable historical_plan.csv already stored under the run directory.",
    )
    parser.add_argument(
        "--expected-plan-sha256",
        default=None,
        help="Required SHA-256 of historical_plan.csv when --existing-plan-only is used.",
    )
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--exclude-plan-row-ids-csv",
        default=None,
        help="CSV with plan_row_id column to quarantine while continuing later rows.",
    )
    parser.add_argument(
        "--require-exclude-plan-row-ids-csv",
        action="store_true",
        help="Fail closed if the quarantine/exclusion CSV is absent.",
    )
    parser.add_argument(
        "--include-plan-row-ids-csv",
        default=None,
        help="CSV with plan_row_id column to retry only those plan rows.",
    )
    parser.add_argument(
        "--include-plan-row-id",
        action="append",
        default=None,
        help="Retry one exact plan row; repeat to supply more than one row.",
    )
    args = parser.parse_args()

    run_id = args.run_id or _latest_run_id()
    lake_root = Path(args.lake_root) if args.lake_root else default_historical_lake_root(Path.cwd())
    include_countries = parse_country_filter(args.include_countries)
    exclude_countries = parse_country_filter(args.exclude_countries)
    artifact_dir = Path(args.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.existing_plan_only:
        if not args.expected_plan_sha256:
            raise ValueError("--expected-plan-sha256 is required with --existing-plan-only")
        plan_path = artifact_dir / "historical_plan.csv"
        if not plan_path.is_file():
            raise FileNotFoundError(f"immutable historical plan is missing: {plan_path}")
        actual_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        if actual_sha256 != args.expected_plan_sha256.lower():
            raise ValueError(
                f"immutable historical plan checksum mismatch: expected {args.expected_plan_sha256}, "
                f"found {actual_sha256}"
            )
        plan = pd.read_csv(plan_path, low_memory=False)
    else:
        plan_path, plan = write_refinitiv_historical_plan(
            symbol_map_csv=args.symbol_map_csv,
            root=lake_root,
            run_id=run_id,
            start_date=args.start_date,
            end_date=args.end_date,
            artifact_root=args.artifact_root,
            include_countries=include_countries,
            exclude_countries=exclude_countries,
            max_tickers=args.max_tickers,
            batch_size=args.batch_size,
            window_years=args.window_years,
            interval=args.interval,
        )
    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifact_dir}")
    print(f"Lake root: {lake_root}")
    print(f"Plan rows: {len(plan)}")
    print(f"Plan: {plan_path}")
    if not plan.empty:
        print(plan.groupby(["country", "exchange"], dropna=False).size().sort_values(ascending=False).head(30).to_string())

    if args.plan_only:
        return

    universe = refinitiv_universe_from_symbol_map(
        args.symbol_map_csv,
        include_countries=include_countries,
        exclude_countries=exclude_countries,
        max_tickers=args.max_tickers,
    )
    field_catalog = pd.read_csv(args.field_catalog_csv, low_memory=False) if args.field_catalog_csv else pd.DataFrame()
    exclude_plan_row_ids = _load_plan_row_ids(
        args.exclude_plan_row_ids_csv,
        label="exclude",
        required=args.require_exclude_plan_row_ids_csv,
    )
    include_plan_row_ids = _load_plan_row_ids(
        args.include_plan_row_ids_csv,
        label="include",
        required=bool(args.include_plan_row_ids_csv),
    )
    include_plan_row_ids.update(str(value) for value in (args.include_plan_row_id or []) if str(value))
    if exclude_plan_row_ids:
        print(f"Excluded/quarantined plan rows: {len(exclude_plan_row_ids)}")
    if include_plan_row_ids:
        print(f"Include-only plan rows: {len(include_plan_row_ids)}")
    adapter = RefinitivAdapter(session_name=args.refinitiv_session, config_name=args.refinitiv_config)
    result = execute_refinitiv_historical_plan(
        plan=plan,
        root=lake_root,
        run_id=run_id,
        downloader=adapter.download_price_history_range,
        universe=universe,
        artifact_dir=artifact_dir,
        field_catalog=field_catalog,
        max_plan_rows=args.max_plan_rows,
        resume=not args.no_resume,
        exclude_plan_row_ids=exclude_plan_row_ids,
        include_plan_row_ids=include_plan_row_ids,
    )

    print(f"Manifest: {result.manifest_path}")
    print(f"Long OHLCV root: {result.long_ohlcv_root}")
    if not result.manifest.empty:
        print(result.manifest["status"].value_counts(dropna=False).to_string())
        cols = ["country", "exchange", "batch_no", "window_no", "start_date", "end_date", "status", "price_columns", "failures", "long_ohlcv_rows"]
        print(result.manifest[[col for col in cols if col in result.manifest.columns]].tail(30).to_string(index=False))


if __name__ == "__main__":
    main()
