from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.data.japan_g1_corporate_transition_validator import (
    write_japan_g1_corporate_transition_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Independently validate official JPX corporate transitions and vendor aliases."
    )
    parser.add_argument(
        "--run-id", default="JPN-G1-CORPORATE-TRANSITION-INDEPENDENT-VALIDATION-20260712-V1"
    )
    parser.add_argument(
        "--transition-evidence-dir",
        default=(
            "artifacts/japan_g1_corporate_transition_evidence/"
            "JPN-G1-CORPORATE-TRANSITION-EVIDENCE-20260712-V1"
        ),
    )
    parser.add_argument(
        "--direct-validation-dir",
        default=(
            "artifacts/japan_g1_quote_chain_validation/"
            "JPN-G1-QUOTE-CHAIN-INDEPENDENT-VALIDATION-20260712-V1"
        ),
    )
    parser.add_argument(
        "--quote-chain-control-dir",
        default=(
            "artifacts/japan_g1_quote_chain_discovery/"
            "JPN-G1-QUOTE-CHAIN-DISCOVERY-20260712-V1"
        ),
    )
    parser.add_argument(
        "--successor-vendor-dir",
        default=(
            "artifacts/data_lake/pit/security_master_history/"
            "SECMASTER-DISCOVERY-JPN-G1-TRANSITION-SUCCESSORS-20260712-V1-SHARD-01"
        ),
    )
    parser.add_argument(
        "--artifact-root", default="artifacts/japan_g1_corporate_transition_validation"
    )
    args = parser.parse_args()
    control = Path(args.quote_chain_control_dir)
    shard_plan = __import__("pandas").read_csv(control / "shard_plan.csv", dtype=str).fillna("")
    shard_dirs = list(shard_plan["artifact_dir"])
    result = write_japan_g1_corporate_transition_validation(
        run_id=args.run_id,
        transition_evidence_dir=args.transition_evidence_dir,
        direct_validation_dir=args.direct_validation_dir,
        quote_chain_shard_dirs=shard_dirs,
        successor_vendor_dir=args.successor_vendor_dir,
        artifact_root=args.artifact_root,
    )
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
