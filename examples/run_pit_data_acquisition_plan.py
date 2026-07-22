from __future__ import annotations

import argparse

from quant_lab.data.pit_data_plan import build_pit_data_acquisition_plan


DEFAULT_L2_ARTIFACT = (
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/"
    "05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the missing PIT data acquisition plan for institutional quant backtests."
    )
    parser.add_argument("--l2-artifact-dir", default=DEFAULT_L2_ARTIFACT)
    parser.add_argument("--artifact-root", default="artifacts/pit_data_plan")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--symbol-map-path", default=None)
    parser.add_argument("--active-countries", default="Malaysia,Singapore,Indonesia")
    parser.add_argument("--tiers", default="u3,u2")
    parser.add_argument("--start-date", default="2006-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--lseg-connector-status", default="not_checked")
    args = parser.parse_args()

    result = build_pit_data_acquisition_plan(
        l2_artifact_dir=args.l2_artifact_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        symbol_map_path=args.symbol_map_path,
        active_countries=tuple(country.strip() for country in args.active_countries.split(",") if country.strip()),
        tiers=tuple(tier.strip() for tier in args.tiers.split(",") if tier.strip()),
        start_date=args.start_date,
        end_date=args.end_date,
        batch_size=args.batch_size,
        lseg_connector_status=args.lseg_connector_status,
    )

    print("=" * 80)
    print("PIT DATA ACQUISITION PLAN COMPLETE")
    print("=" * 80)
    print(f"Run ID:              {result.run_id}")
    print(f"Artifacts:           {result.artifact_dir}")
    print(f"Selected names:      {len(result.selected_universe)}")
    print(f"U2 names:            {int(result.selected_universe['selected_for_u2'].sum())}")
    print(f"U3 names:            {int(result.selected_universe['selected_for_u3'].sum())}")
    print(f"Dataset groups:      {len(result.dataset_requirements)}")
    print(f"Field-probe rows:    {len(result.field_probe_queue)}")
    print(f"Extraction batches:  {len(result.extraction_batches)}")
    print(f"Summary:             {result.summary_path}")


if __name__ == "__main__":
    main()
