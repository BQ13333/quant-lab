from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from quant_lab.data.japan_g1_enrichment import (
    DEFAULT_JAPAN_G1_ARTIFACT_ROOT,
    DEFAULT_JAPAN_G1_SOURCE_DIR,
    write_japan_g1_enrichment_artifacts,
)
from quant_lab.data.vendor_adapters import RefinitivAdapter


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resume bounded current-observation Refinitiv enrichment for the reconciled JPX G1 universes."
    )
    parser.add_argument("--source-dir", default=str(DEFAULT_JAPAN_G1_SOURCE_DIR))
    parser.add_argument("--artifact-root", default=str(DEFAULT_JAPAN_G1_ARTIFACT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--max-rics", type=int, default=100)
    parser.add_argument("--request-chunk-size", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--session", "--refinitiv-session", dest="session", default="desktop.workspace")
    parser.add_argument("--config", default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    adapter = RefinitivAdapter(session_name=args.session, config_name=args.config, health_ric="7203.T")
    adapter.open()
    try:
        artifacts = write_japan_g1_enrichment_artifacts(
            adapter.ld,
            source_dir=Path(args.source_dir),
            artifact_root=Path(args.artifact_root),
            run_id=args.run_id,
            max_rics=args.max_rics,
            request_chunk_size=args.request_chunk_size,
            sleep_seconds=args.sleep_seconds,
            session_name=args.session,
        )
    finally:
        adapter.close()

    print(f"run_id={artifacts.run_id}")
    print(f"artifact_dir={artifacts.artifact_dir.resolve()}")
    print(f"status={artifacts.manifest['status']}")
    print(f"attempted_rics={artifacts.manifest['bounds']['attempted_rics']}")
    print(f"remaining_without_success={artifacts.manifest['counts']['remaining_without_success']}")
    print(f"manifest={artifacts.manifest_path.resolve()}")
    print("historical_pit_claimed=False")
    print("alpha_promotion_allowed=False")
    print("production_promotion_allowed=False")
    if str(artifacts.manifest["status"]).startswith("failed_closed"):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
