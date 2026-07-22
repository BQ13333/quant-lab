from __future__ import annotations

import argparse
import json

from quant_lab.data.jpx_archival_fallbacks import build_jpx_archival_fallbacks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build validated archival routes for removed official TSE PDFs.")
    parser.add_argument(
        "--source-dir",
        default="artifacts/jpx_monthly_quotations/JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2",
    )
    parser.add_argument("--max-routes", type=int, default=50)
    parser.add_argument("--pacing-seconds", type=float, default=0.1)
    args = parser.parse_args()
    result = build_jpx_archival_fallbacks(
        source_dir=args.source_dir,
        max_routes=args.max_routes,
        pacing_seconds=args.pacing_seconds,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
