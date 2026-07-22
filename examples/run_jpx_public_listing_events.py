from __future__ import annotations

import argparse
import json

from quant_lab.data.jpx_public_listing_events import run_jpx_public_listing_events


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract official JPX public listing-event archives for independent cross-validation.")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--artifact-root", default="artifacts/jpx_public_listing_events")
    parser.add_argument("--max-pages", type=int, default=20)
    parser.add_argument("--pacing-seconds", type=float, default=0.25)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    result = run_jpx_public_listing_events(
        run_id=args.run_id,
        artifact_root=args.artifact_root,
        max_pages=args.max_pages,
        pacing_seconds=args.pacing_seconds,
        retries=args.retries,
    )
    print(json.dumps({
        "run_id": result.run_id,
        "status": result.manifest["status"],
        "coverage": result.manifest["coverage"],
        "historical_pit_ready": result.manifest["historical_pit_ready"],
        "manifest": str(result.manifest_path.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
