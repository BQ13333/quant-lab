from __future__ import annotations

import argparse
import warnings
from datetime import datetime, timezone

from quant_lab.data.refinitiv_discovery import (
    ASEAN_REFINITIV_COUNTRY_CODES,
    write_refinitiv_asean_l2_discovery_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def _parse_countries(value: str) -> tuple[str, ...]:
    if not value:
        return tuple(ASEAN_REFINITIV_COUNTRY_CODES)
    allowed = {country.casefold(): country for country in ASEAN_REFINITIV_COUNTRY_CODES}
    countries = []
    for item in value.split(","):
        key = item.strip().casefold()
        if not key:
            continue
        if key not in allowed:
            raise SystemExit(f"Unsupported country for ASEAN Refinitiv discovery: {item}")
        countries.append(allowed[key])
    return tuple(dict.fromkeys(countries))


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"lseg\.data\..*")
    parser = argparse.ArgumentParser(description="Build a Refinitiv Discovery ASEAN L2 liquid equity universe.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_discovery")
    parser.add_argument("--countries", default=",".join(ASEAN_REFINITIV_COUNTRY_CODES))
    parser.add_argument("--max-per-country", type=int, default=100)
    parser.add_argument("--max-screen-results-per-country", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-ASEAN-L2-DISCOVERY-%Y%m%d-%H%M%S")
    countries = _parse_countries(args.countries)
    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        artifact = write_refinitiv_asean_l2_discovery_artifacts(
            ld=adapter.ld,
            artifact_root=args.artifact_root,
            run_id=run_id,
            countries=countries,
            max_per_country=args.max_per_country,
            max_screen_results_per_country=args.max_screen_results_per_country,
            chunk_size=args.chunk_size,
            sleep_seconds=args.sleep_seconds,
        )
    finally:
        adapter.close()

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifact.artifact_dir}")
    print(f"Raw screener rows: {len(artifact.raw_universe)}")
    print(f"Ranked universe rows: {len(artifact.ranked_universe)}")
    print(f"Symbol-map input rows: {len(artifact.symbol_map_input)}")
    print(artifact.country_summary.to_string(index=False))
    print(f"Symbol-map input: {artifact.symbol_map_input_path}")


if __name__ == "__main__":
    main()
