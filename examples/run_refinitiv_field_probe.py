from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_field_probe import (
    probe_refinitiv_fields,
    sample_refinitiv_universe,
    write_refinitiv_field_probe_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe Refinitiv/LSEG historical price fields by market.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_field_probe")
    parser.add_argument("--listings", default=None, help="Optional combined listings CSV with refinitiv_ric.")
    parser.add_argument("--sample-per-market", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=0, help="Optional cap after default liquid samples are included.")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--interval", default="1D")
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=50)
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-FIELD-PROBE-%Y%m%d-%H%M%S")
    listings = pd.read_csv(args.listings, low_memory=False) if args.listings else None
    samples = sample_refinitiv_universe(
        listings=listings,
        sample_per_market=args.sample_per_market,
        max_samples=args.max_samples or None,
    )

    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
    adapter.open()
    try:
        results = probe_refinitiv_fields(
            adapter.ld,
            samples=samples,
            interval=args.interval,
            count=args.count,
            sleep_seconds=args.sleep_seconds,
            progress_every=args.progress_every,
        )
    finally:
        adapter.close()

    artifact = write_refinitiv_field_probe_artifacts(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        samples=samples,
        results=results,
        session_name=args.session,
        interval=args.interval,
        count=args.count,
    )

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifact.artifact_dir}")
    print(f"Samples: {len(samples)}")
    print(f"Probe rows: {len(results)}")
    print(f"Recommended map: {artifact.recommended_path}")
    print(f"Field catalog: {artifact.field_catalog_path}")
    print(f"RIC validation: {artifact.validation_path}")
    if not artifact.recommended.empty:
        display_cols = ["country", "exchange", "role", "field", "successful_rics", "avg_coverage", "status"]
        print(artifact.recommended[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
