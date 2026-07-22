from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from quant_lab.data.refinitiv_pit_extraction import (
    load_pit_plan_artifacts,
    probe_refinitiv_pit_fields,
    write_refinitiv_pit_field_probe_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_csv_arg(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe PIT Refinitiv fields from a PIT data plan.")
    parser.add_argument("--plan-dir", default="artifacts/pit_data_plan/PIT-DATA-PLAN-MYS-SGP-IDN-20260708")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_pit_field_probe")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--sample-per-country", type=int, default=3)
    parser.add_argument(
        "--sample-rics",
        default=None,
        help="Optional comma-separated allowlist used to make a probe exactly reproducible.",
    )
    parser.add_argument("--tier", default="u3", choices=["u2", "u3"])
    parser.add_argument(
        "--datasets",
        default="security_master_pit,liquidity_market_cap_free_float",
        help="Comma-separated dataset ids from field_probe_queue.csv.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-PIT-FIELD-PROBE-%Y%m%d-%H%M%S")
    datasets = _parse_csv_arg(args.datasets)
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
        results = probe_refinitiv_pit_fields(
            adapter.ld,
            selected_universe=selected_universe,
            field_probe_queue=plan.field_probe_queue,
            sample_per_country=args.sample_per_country,
            tier=args.tier,
            datasets=datasets,
            sleep_seconds=args.sleep_seconds,
            progress_every=args.progress_every,
        )
    finally:
        adapter.close()

    artifacts = write_refinitiv_pit_field_probe_artifacts(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        selected_universe=selected_universe,
        field_probe_queue=plan.field_probe_queue,
        results=results,
        sample_per_country=args.sample_per_country,
        tier=args.tier,
        datasets=datasets,
        session_name=args.session,
    )

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Probe rows: {len(artifacts.results)}")
    print(f"Accepted fields: {len(artifacts.accepted_field_map)}")
    print(f"Accepted field map: {artifacts.accepted_field_map_path}")
    if not artifacts.accepted_field_map.empty:
        display_cols = ["dataset_id", "field_role", "candidate_field", "coverage", "acceptance_status"]
        print(artifacts.accepted_field_map[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
