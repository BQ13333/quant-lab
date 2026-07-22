from __future__ import annotations

import argparse
from pathlib import Path

from quant_lab.data.japan_g1_monthly_materializer import (
    write_japan_g1_monthly_materializer,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build producer-only Japan G1 candidates from official monthly membership.")
    parser.add_argument("--run-id", default="JPN-G1-MONTHLY-PIT-MATERIALIZER-20260712-V1")
    parser.add_argument(
        "--monthly-qa-dir",
        default="artifacts/japan_combined_monthly_membership_qa/JPN-COMBINED-MONTHLY-MEMBERSHIP-QA-200607-202606-V1",
    )
    parser.add_argument(
        "--security-master-dir",
        action="append",
        dest="security_master_dirs",
        default=None,
    )
    args = parser.parse_args()
    if args.security_master_dirs:
        masters = args.security_master_dirs
    else:
        root = Path("artifacts/data_lake/pit/security_master_history")
        masters = [str(root / "SECMASTER-DISCOVERY-JPN-MERGED-20260709")]
        masters.extend(
            str(path)
            for path in sorted(
                root.glob("SECMASTER-DISCOVERY-JPN-G1-TARGETED-MISSING-*-SHARD-*")
            )
            if (path / "manifest.json").is_file()
        )
        validated_transition = Path(
            "artifacts/japan_g1_corporate_transition_validation/"
            "JPN-G1-CORPORATE-TRANSITION-INDEPENDENT-VALIDATION-20260712-V1"
        )
        validated_quote_chain = Path(
            "artifacts/japan_g1_quote_chain_validation/"
            "JPN-G1-QUOTE-CHAIN-INDEPENDENT-VALIDATION-20260712-V1"
        )
        if (validated_transition / "manifest.json").is_file():
            masters.append(str(validated_transition))
        elif (validated_quote_chain / "manifest.json").is_file():
            masters.append(str(validated_quote_chain))
    result = write_japan_g1_monthly_materializer(
        run_id=args.run_id,
        monthly_qa_dir=args.monthly_qa_dir,
        security_master_dirs=masters,
    )
    print(f"artifact_dir={result.artifact_dir.resolve()}")
    print(f"status={result.manifest['status']}")
    print(f"historical_pit_ready={result.manifest['historical_pit_ready']}")
    print(f"critical_blockers={result.manifest['critical_blockers']}")
    if "counts" in result.manifest:
        print(f"counts={result.manifest['counts']}")


if __name__ == "__main__":
    main()
