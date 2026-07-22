from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.data.japan_g1_quote_chain_validator import (
    write_japan_g1_quote_chain_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Independently validate direct Japan G1 quote-chain routes.")
    parser.add_argument("--run-id", default="JPN-G1-QUOTE-CHAIN-INDEPENDENT-VALIDATION-20260712-V1")
    parser.add_argument("--materializer-dir", default="artifacts/japan_g1_quote_chain_materializer/JPN-G1-QUOTE-CHAIN-MATERIALIZER-20260712-V1")
    parser.add_argument("--identity-plan-dir", default="artifacts/japan_g1_quote_chain_discovery/JPN-G1-QUOTE-CHAIN-DISCOVERY-20260712-V1")
    parser.add_argument("--shard-dir", action="append", dest="shard_dirs")
    args = parser.parse_args()
    shards = args.shard_dirs or [str(path) for path in sorted(Path("artifacts/data_lake/pit/security_master_history").glob("JPN-G1-QUOTE-CHAIN-DISCOVERY-20260712-V1-SHARD-*"))]
    result = write_japan_g1_quote_chain_validation(run_id=args.run_id, materializer_dir=args.materializer_dir, identity_plan_dir=args.identity_plan_dir, shard_dirs=shards)
    print(json.dumps(result.manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
