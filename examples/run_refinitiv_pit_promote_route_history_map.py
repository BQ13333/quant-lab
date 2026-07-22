from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_pit_extraction import write_refinitiv_pit_route_history_map_artifacts


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote usable Refinitiv route-probe rows into a history field map.")
    parser.add_argument("--route-probe-dir", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_pit_route_history_map")
    parser.add_argument(
        "--datasets",
        default="fundamentals_pit,ibes_estimates_revisions,earnings_events_actuals",
        help="Comma-separated dataset ids to promote.",
    )
    parser.add_argument("--min-inside-rows", type=int, default=1)
    parser.add_argument("--min-coverage", type=float, default=0.0)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-PIT-ROUTE-HISTORY-MAP-%Y%m%d-%H%M%S")
    route_results_path = Path(args.route_probe_dir) / "route_probe_results.csv"
    if not route_results_path.exists():
        raise FileNotFoundError(f"route probe results not found: {route_results_path}")
    datasets = _parse_csv_arg(args.datasets)
    route_probe_results = pd.read_csv(route_results_path, low_memory=False)

    artifacts = write_refinitiv_pit_route_history_map_artifacts(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        route_probe_results=route_probe_results,
        datasets=datasets,
        min_inside_rows=args.min_inside_rows,
        min_coverage=args.min_coverage,
    )

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Route rows: {len(artifacts.route_results)}")
    print(f"Promoted fields: {len(artifacts.history_field_map)}")
    print(f"History field map: {artifacts.history_field_map_path}")
    if not artifacts.history_field_map.empty:
        print(
            artifacts.history_field_map[
                ["dataset_id", "field_role", "candidate_field", "coverage", "history_status"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
