from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from quant_lab.data.refinitiv_pit_extraction import (
    critical_refinitiv_route_probe_queue,
    load_pit_plan_artifacts,
    probe_refinitiv_pit_route_candidates,
    write_refinitiv_pit_route_probe_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe expanded critical Refinitiv routes blocking 5/5 data quality."
    )
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_pit_critical_route_probe")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--tier", default="u3", choices=["u2", "u3"])
    parser.add_argument("--sample-per-country", type=int, default=1)
    parser.add_argument("--sample-rics", default=None)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--interval", default="1M")
    parser.add_argument("--datasets", default=None, help="Comma-separated dataset ids to probe.")
    parser.add_argument("--field-roles", default=None, help="Comma-separated field roles to probe.")
    parser.add_argument("--routes", default="get_data,get_history")
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    if args.max_candidates is not None and args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive when provided")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-PIT-CRITICAL-ROUTE-PROBE-%Y%m%d-%H%M%S")
    datasets = _parse_csv_arg(args.datasets)
    field_roles = _parse_csv_arg(args.field_roles)
    routes = _parse_csv_arg(args.routes) or ["get_data", "get_history"]

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
    queue = critical_refinitiv_route_probe_queue(datasets=datasets, field_roles=field_roles)
    if args.max_candidates is not None:
        queue = queue.head(args.max_candidates).copy()
    if queue.empty:
        raise ValueError("critical route queue is empty after filters")

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        results = probe_refinitiv_pit_route_candidates(
            adapter.ld,
            selected_universe=selected_universe,
            field_probe_queue=queue,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            sample_per_country=args.sample_per_country,
            tier=args.tier,
            datasets=datasets,
            routes=routes,
            sleep_seconds=args.sleep_seconds,
            progress_every=args.progress_every,
        )
    finally:
        adapter.close()

    artifacts = write_refinitiv_pit_route_probe_artifacts(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        results=results,
        datasets=datasets,
        session_name=args.session,
    )
    queue_path = artifacts.artifact_dir / "critical_route_probe_queue.csv"
    queue.to_csv(queue_path, index=False)

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Queue rows: {len(queue)}")
    print(f"Probe rows: {len(artifacts.results)}")
    print(f"Queue: {queue_path}")
    print(f"Results: {artifacts.results_path}")
    if not artifacts.results.empty:
        display = artifacts.results[
            [
                "dataset_id",
                "field_role",
                "candidate_field",
                "route",
                "status",
                "non_null_values",
                "inside_requested_rows",
                "coverage",
            ]
        ]
        print(display.to_string(index=False))


if __name__ == "__main__":
    main()
