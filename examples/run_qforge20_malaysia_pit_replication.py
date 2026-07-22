from __future__ import annotations

import argparse
import json
from pathlib import Path

from quant_lab.research.malaysia_pit_replication import (
    audit_pit_replication_inputs,
    load_frozen_protocol,
    persist_json,
)


DEFAULT_PROTOCOL = Path("configs/qforge20_malaysia_pit_replication_frozen_v1.json")
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/qforge20/QFORGE20-MALAYSIA-RESEARCH-V1/pit_replication_14_frozen_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed control gate for the frozen fourteen-strategy Malaysia PIT replication."
        )
    )
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    protocol = load_frozen_protocol(args.protocol)
    report = audit_pit_replication_inputs(protocol, repo_root=repo_root)
    report_path = persist_json(report, args.output_root / "data_admissibility_report.json")
    stop_receipt = {
        "schema_version": 1,
        "workflow_name": "QForge-20",
        "protocol_id": report["protocol_id"],
        "protocol_sha256": report["protocol_sha256"],
        "status": (
            "ready_for_atomic_fourteen_strategy_reservation"
            if report["admissible"]
            else "stopped_at_data_admissibility_gate"
        ),
        "failed_gate_ids": report["failed_gate_ids"],
        "strategy_trials_reserved": False,
        "strategy_computations_started": False,
        "paired_performance_evaluation_completed": False,
        "parameter_optimization_performed": False,
        "dry_run_started": False,
        "next_action": (
            "reserve the exact fourteen identities atomically and run the frozen paired replication once"
            if report["admissible"]
            else "obtain historical Malaysia U2 membership plus signal inputs and adjusted returns for inactive and delisted names"
        ),
    }
    stop_path = persist_json(stop_receipt, args.output_root / "control_gate_receipt.json")
    print(
        json.dumps(
            {
                "protocol_sha256": report["protocol_sha256"],
                "admissible": report["admissible"],
                "failed_gate_ids": report["failed_gate_ids"],
                "strategy_trials_reserved": False,
                "dry_run_started": False,
                "data_admissibility_report": str(report_path),
                "control_gate_receipt": str(stop_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
