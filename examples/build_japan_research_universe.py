from __future__ import annotations

import argparse
from datetime import datetime, timezone

from quant_lab.data.japan_research_universe import (
    write_japan_research_universe_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Japan research universe from the merged security master and "
            "a current public JPX listing file."
        )
    )
    parser.add_argument("--security-master", required=True)
    parser.add_argument("--jpx-listings", required=True)
    parser.add_argument(
        "--run-id",
        default=datetime.now(timezone.utc).strftime(
            "JPN-JPX-RESEARCH-UNIVERSE-%Y%m%d-%H%M%S"
        ),
    )
    parser.add_argument(
        "--artifact-root", default="artifacts/research_universes"
    )
    args = parser.parse_args()

    artifacts = write_japan_research_universe_artifacts(
        security_master_path=args.security_master,
        jpx_listings_path=args.jpx_listings,
        run_id=args.run_id,
        artifact_root=args.artifact_root,
    )
    print(f"Manifest: {artifacts.manifest_path}")
    print(
        "Active / reference rows: "
        f"{artifacts.manifest['counts']['active_operating_core_rows']} / "
        f"{artifacts.manifest['counts']['historical_listed_reference_rows']}"
    )
    print(f"Scorecard eligible: {artifacts.manifest['scorecard_eligible']}")
    print("Historical PIT ready: False")


if __name__ == "__main__":
    main()
