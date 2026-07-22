from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.data.bloomberg_malaysia_bulk import build_bloomberg_malaysia_bulk_lake


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, quarantine, normalize, and audit the Malaysia Bloomberg bulk source."
    )
    parser.add_argument(
        "--source-config",
        type=Path,
        default=Path("configs/bloomberg_malaysia_bulk_source_v1.json"),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/data_lake/vendor/bloomberg_malaysia_bulk"),
    )
    parser.add_argument(
        "--run-id",
        default="BLOOMBERG-MYS-BULK-2005-202604-V1",
    )
    parser.add_argument(
        "--existing-security-master-root",
        type=Path,
        default=Path("artifacts/data_lake/pit/security_master_history/MYS-SGP-IDN-U2-20260708"),
    )
    parser.add_argument("--max-download-workers", type=int, default=2)
    parser.add_argument("--chunk-rows", type=int, default=100_000)
    parser.add_argument("--rebuild-generated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_bloomberg_malaysia_bulk_lake(
        args.source_config,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        existing_security_master_root=args.existing_security_master_root,
        max_download_workers=args.max_download_workers,
        chunk_rows=args.chunk_rows,
        force_rebuild_generated=args.rebuild_generated,
        progress=lambda message: print(message, flush=True),
    )
    admission = json.loads(result.admission_report_path.read_text(encoding="utf-8"))
    print(
        json.dumps(
            {
                "run_id": result.run_id,
                "artifact_dir": str(result.artifact_dir.resolve()),
                "status": result.manifest["status"],
                "admission_status": admission["status"],
                "counts": admission["counts"],
                "institutional_pit_gates": admission["institutional_pit_gates"],
                "strategy_trials_reserved": result.manifest["strategy_trials_reserved"],
                "dry_run_started": result.manifest["dry_run_started"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
