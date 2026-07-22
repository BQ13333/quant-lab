from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from quant_lab.data.usa_index_constituents_pit import (
    read_pit_constituent_rows,
    write_usa_index_constituent_plan_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an offline USA PIT index-constituent schema and Refinitiv probe/request plan."
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/index_constituent_plans")
    parser.add_argument("--start-date", default="2000-01-01")
    parser.add_argument("--end-date", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument(
        "--field-catalog",
        default=None,
        help="Optional existing Refinitiv field_catalog.csv used for annotations only.",
    )
    parser.add_argument(
        "--pit-rows",
        default=None,
        help="Optional source-backed CSV/Parquet rows to validate; omitted means no constituent rows are written.",
    )
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("USA-PIT-INDEX-CONSTITUENT-PLAN-%Y%m%d-%H%M%S")
    pit_rows = read_pit_constituent_rows(args.pit_rows) if args.pit_rows else None
    artifacts = write_usa_index_constituent_plan_artifacts(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        start_date=args.start_date,
        end_date=args.end_date,
        field_catalog_path=args.field_catalog,
        pit_rows=pit_rows,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Index targets: {len(artifacts.index_catalog)}")
    print(f"Candidate field probes: {len(artifacts.field_probe_plan)}")
    print(f"Logical requests: {len(artifacts.request_plan)}")
    print(f"Supplied PIT rows: {artifacts.validation.supplied_rows}")
    print(f"PIT-usable rows: {artifacts.validation.usable_rows}")
    print(f"Manifest status: {manifest['status']}")
    print(f"Scorecard eligible: {manifest['scorecard_eligible']}")
    print(f"Manifest: {artifacts.manifest_path}")
    if manifest["status"] != "complete":
        print("Guardrail: planning artifacts and candidate fields are not constituent history.")


if __name__ == "__main__":
    main()
