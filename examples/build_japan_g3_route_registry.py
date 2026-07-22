from __future__ import annotations

import argparse

from quant_lab.data.japan_g3_route_registry import write_japan_g3_route_registry_artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Japan G3 accepted route registry.")
    parser.add_argument(
        "--historical-probe-dir",
        default="artifacts/refinitiv_pit_historical_probe/REFINITIV-PIT-HISTORICAL-PROBE-JPN-G3-20260712",
    )
    parser.add_argument(
        "--critical-probe-dir",
        default="artifacts/refinitiv_pit_critical_route_probe/REFINITIV-PIT-CRITICAL-ROUTE-PROBE-JPN-G3-20260712",
    )
    parser.add_argument(
        "--revision-probe-dir",
        default="artifacts/refinitiv_ibes_revision_route_probe/REFINITIV-IBES-REVISION-ROUTE-PROBE-JPN-20260712",
    )
    parser.add_argument(
        "--fundamentals-qa-dir",
        default="artifacts/refinitiv_japan_g3_fundamentals/REFINITIV-JPN-G3-FUNDAMENTALS-QA-20260712",
    )
    parser.add_argument("--artifact-root", default="artifacts/japan_g3_route_registry")
    parser.add_argument(
        "--sector-snapshot-dir",
        default="artifacts/refinitiv_current_snapshot/REFINITIV-JPN-G3-SECTOR-ROUTING-20260712",
    )
    parser.add_argument("--run-id", default="JPN-G3-ROUTE-REGISTRY-20260712")
    parser.add_argument(
        "--ibes-events-qa-dir",
        default="artifacts/refinitiv_japan_g3_ibes_events/REFINITIV-JPN-G3-IBES-EVENTS-QA-20260712",
    )
    args = parser.parse_args()
    result = write_japan_g3_route_registry_artifacts(
        historical_probe_dir=args.historical_probe_dir,
        critical_probe_dir=args.critical_probe_dir,
        revision_probe_dir=args.revision_probe_dir,
        fundamentals_qa_dir=args.fundamentals_qa_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        sector_snapshot_dir=args.sector_snapshot_dir,
        ibes_events_qa_dir=args.ibes_events_qa_dir,
    )
    print(f"Artifact dir: {result.artifact_dir}")
    print(f"Accepted roles: {len(result.registry)}")
    print(f"Unresolved roles: {len(result.unresolved)}")


if __name__ == "__main__":
    main()
