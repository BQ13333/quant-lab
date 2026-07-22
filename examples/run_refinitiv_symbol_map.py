from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_symbol_map import (
    build_refinitiv_symbol_candidates,
    sample_symbol_map_listings,
    validate_refinitiv_symbol_candidates,
    write_refinitiv_symbol_map_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and live-validate a Refinitiv RIC symbol map.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--artifact-root", default="artifacts/refinitiv_symbol_map")
    parser.add_argument("--listings", default=None, help="Optional combined listings CSV.")
    parser.add_argument("--sample-per-market", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=150)
    parser.add_argument("--no-curated-seeds", action="store_true")
    parser.add_argument("--session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    parser.add_argument("--interval", default="1D")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--no-live-validation", action="store_true")
    args = parser.parse_args()

    run_id = args.run_id or datetime.now(timezone.utc).strftime("REFINITIV-SYMBOL-MAP-%Y%m%d-%H%M%S")
    listings = pd.read_csv(args.listings, low_memory=False) if args.listings else pd.DataFrame()
    sampled = sample_symbol_map_listings(
        listings=listings,
        sample_per_market=args.sample_per_market,
        max_symbols=args.max_symbols or None,
        include_curated_seeds=not args.no_curated_seeds,
    )
    candidates = build_refinitiv_symbol_candidates(sampled)

    if args.no_live_validation:
        validations = candidates.assign(
            rows=0,
            non_null_values=0,
            coverage=0.0,
            first_available=pd.NaT,
            last_available=pd.NaT,
            last_value=pd.NA,
            returned_columns="",
            validated_close_field=pd.NA,
            error="not live validated",
            interval=args.interval,
            count=args.count,
            status="not_validated",
        )
    else:
        adapter = RefinitivAdapter(session_name=args.session, config_name=args.config)
        adapter.open()
        try:
            validations = validate_refinitiv_symbol_candidates(
                adapter.ld,
                candidates=candidates,
                interval=args.interval,
                count=args.count,
                sleep_seconds=args.sleep_seconds,
                progress_every=args.progress_every,
            )
        finally:
            adapter.close()

    artifact = write_refinitiv_symbol_map_artifacts(
        artifact_root=Path(args.artifact_root),
        run_id=run_id,
        candidates=candidates,
        validations=validations,
        session_name=args.session,
        interval=args.interval,
        count=args.count,
    )

    print(f"Run ID: {run_id}")
    print(f"Artifact dir: {artifact.artifact_dir}")
    print(f"Input symbols: {len(sampled)}")
    print(f"Candidate rows: {len(candidates)}")
    print(f"Validation rows: {len(validations)}")
    print(f"Symbol map: {artifact.symbol_map_path}")
    if not artifact.symbol_map.empty:
        print(artifact.symbol_map["refinitiv_symbol_status"].value_counts(dropna=False).to_string())
        display_cols = [
            "country",
            "exchange",
            "ticker",
            "refinitiv_ric",
            "refinitiv_symbol_status",
            "candidate_method",
            "validated_close_field",
        ]
        print(artifact.symbol_map[display_cols].head(50).to_string(index=False))


if __name__ == "__main__":
    main()
