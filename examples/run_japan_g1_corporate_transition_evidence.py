from __future__ import annotations

import argparse
import json

from quant_lab.data.japan_g1_corporate_transition_evidence import (
    write_japan_g1_corporate_transition_evidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build resumable official JPX evidence for the 66 G1 corporate-transition edges."
    )
    parser.add_argument(
        "--run-id", default="JPN-G1-CORPORATE-TRANSITION-EVIDENCE-20260712-V1"
    )
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g1_corporate_transition_evidence"
    )
    parser.add_argument(
        "--quote-chain-candidates",
        default=(
            "artifacts/japan_g1_quote_chain_materializer/"
            "JPN-G1-QUOTE-CHAIN-MATERIALIZER-20260712-V1/quote_chain_candidates.csv"
        ),
    )
    parser.add_argument(
        "--archive-manifest",
        default=(
            "artifacts/japan_tick_assignment_history/"
            "JPN-TICK-ASSIGNMENT-NOTICES-201401-202606-V3/archive_source_manifest.csv"
        ),
    )
    parser.add_argument(
        "--delisting-events",
        default=(
            "artifacts/jpx_public_listing_events/"
            "JPX-PUBLIC-LISTING-EVENTS-20260712/delisting_events.csv"
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Maximum new requests in this invocation; zero completes all remaining sources.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.02)
    args = parser.parse_args()
    result = write_japan_g1_corporate_transition_evidence(
        run_id=args.run_id,
        quote_chain_candidates_path=args.quote_chain_candidates,
        archive_manifest_path=args.archive_manifest,
        delisting_events_path=args.delisting_events,
        artifact_root=args.artifact_root,
        max_pages=args.max_pages,
        sleep_seconds=args.sleep_seconds,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
