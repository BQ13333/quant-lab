from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from quant_lab.data.ibes_revisions_pit import (
    REQUESTED_COUNTRIES,
    default_ibes_revision_parameter_grid,
    probe_ibes_revision_timestamp_routes,
    write_ibes_revision_route_probe_artifacts,
)
from quant_lab.data.refinitiv_pit_extraction import load_pit_plan_artifacts
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_csv_arg(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe IBES revision timestamp routes for PIT estimate revisions.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_ibes_revision_route_probe")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tier", default="u3", choices=["u2", "u3"])
    parser.add_argument("--countries", default=",".join(REQUESTED_COUNTRIES))
    parser.add_argument("--sample-per-country", type=int, default=1)
    parser.add_argument("--sample-rics", default=None)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument(
        "--fields",
        default=None,
        help="Optional comma-separated timestamp fields; defaults to the full discovery grid.",
    )
    parser.add_argument(
        "--periods",
        default="FY1,FY2",
        help="Comma-separated IBES forecast periods used in the daily parameter grid.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-IBES-REVISION-ROUTE-PROBE-%Y%m%d-%H%M%S")
    countries = _parse_csv_arg(args.countries) or list(REQUESTED_COUNTRIES)
    fields = _parse_csv_arg(args.fields) or None
    periods = _parse_csv_arg(args.periods)
    if not periods:
        raise ValueError("at least one IBES period is required")
    parameter_grid = [
        {**params, "Period": period}
        for params in default_ibes_revision_parameter_grid(args.start_date, args.end_date)[:1]
        for period in periods
    ]
    plan = load_pit_plan_artifacts(Path(args.plan_dir))
    selected_universe = plan.selected_universe
    sample_rics = _parse_csv_arg(args.sample_rics)
    if sample_rics:
        wanted = set(sample_rics)
        selected_universe = selected_universe.loc[
            selected_universe["refinitiv_ric"].astype(str).isin(wanted)
        ].copy()
        missing = wanted - set(selected_universe["refinitiv_ric"].astype(str))
        if missing:
            raise ValueError(f"sample RICs not present in selected universe: {sorted(missing)}")

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        results = probe_ibes_revision_timestamp_routes(
            adapter.ld,
            selected_universe=selected_universe,
            start_date=args.start_date,
            end_date=args.end_date,
            countries=countries,
            fields=fields,
            parameter_grid=parameter_grid,
            sample_per_country=args.sample_per_country,
            tier=args.tier,
            sleep_seconds=args.sleep_seconds,
            progress_every=args.progress_every,
        )
    finally:
        adapter.close()

    artifacts = write_ibes_revision_route_probe_artifacts(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        route_probe_results=results,
        countries=countries,
        session_name=args.session,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Probe rows: {len(artifacts.route_results)}")
    print(f"Accepted routes: {len(artifacts.accepted_revision_time_map)}")
    print(f"Results: {artifacts.route_results_path}")
    print(f"Accepted map: {artifacts.accepted_map_path}")
    print(f"Manifest: {artifacts.manifest_path}")
    if not artifacts.accepted_revision_time_map.empty:
        print(
            artifacts.accepted_revision_time_map[
                ["country", "field", "parameter_set", "non_null_date_rows", "date_min", "date_max", "unique_date_count"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
