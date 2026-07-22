from __future__ import annotations

import tempfile

from quant_lab.agents.workflow import build_research_workflow, default_research_state


def main() -> None:
    graph = build_research_workflow()
    with tempfile.TemporaryDirectory(prefix="quant-lab-workflow-example-") as temporary_vault:
        result = graph.invoke(
            default_research_state(
                hypothesis="Cross-sectional momentum with volatility targeting can survive walk-forward costs.",
                universe="Synthetic daily equity panel",
                run_id="SYNTHETIC-WORKFLOW-EXAMPLE",
                vault_root=temporary_vault,
                run_kind="synthetic",
            )
        )

        print("Workflow status:", result["workflow_status"])
        print("Validation status:", result["validation_status"])
        print("Promotion status:", result["promotion_status"])
        print("Approval required:", result["approval_required"])
        print("\nAudit log:")
        for item in result["audit_log"]:
            print(f"- {item}")


if __name__ == "__main__":
    main()
