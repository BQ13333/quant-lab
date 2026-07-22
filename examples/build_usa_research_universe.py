from __future__ import annotations

import argparse
from datetime import datetime, timezone

from quant_lab.data.usa_research_universe import write_usa_research_universe_artifacts


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("USA-RESEARCH-UNIVERSE-%Y%m%d-%H%M%S")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build USA IBKR active liquid-core and listed-reference symbol maps "
            "from the Refinitiv Discovery security master."
        )
    )
    parser.add_argument("--security-master", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/research_universes")
    parser.add_argument("--symbol-map-artifact-root", default="artifacts/refinitiv_symbol_map")
    args = parser.parse_args()

    run_id = args.run_id or _latest_run_id()
    artifacts = write_usa_research_universe_artifacts(
        security_master_path=args.security_master,
        run_id=run_id,
        artifact_root=args.artifact_root,
        symbol_map_artifact_root=args.symbol_map_artifact_root,
    )
    manifest = artifacts.manifest

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifacts.artifact_dir}")
    print(f"Manifest: {artifacts.manifest_path}")
    print(f"Active IBKR liquid-core symbol map: {artifacts.active_symbol_map_path}")
    print(f"Listed reference symbol map: {artifacts.reference_symbol_map_path}")
    print(f"Security-master rows: {manifest['security_master_rows']}")
    print(f"Base USA common-share rows: {manifest['base_usa_common_share_rows']}")
    print(f"Active core rows: {manifest['active_core_rows']}")
    print(f"Listed reference rows: {manifest['reference_core_rows']}")
    print("Active core exchanges:")
    for exchange, count in sorted(manifest["active_core_exchange_counts"].items()):
        print(f"  {exchange}: {count}")


if __name__ == "__main__":
    main()
