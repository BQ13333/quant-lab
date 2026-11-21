from __future__ import annotations

import pytest

from quant_lab.agents.workflow import (
    QuantResearchState,
    build_research_workflow,
    data_steward,
    default_research_state,
    feature_engineer,
    human_approval,
    modeler,
    portfolio_optimizer,
    research_director,
)
from quant_lab.research.control_plane import ExperimentRunStore, REQUIRED_VALIDATION_GATES


def make_test_state(tmp_path, hypothesis="h", universe="u", run_id="TEST-RUN-001"):
    return default_research_state(
        hypothesis,
        universe,
        run_id=run_id,
        vault_root=str(tmp_path),
        run_kind="test",
    )


def apply_nodes(state, *nodes):
    for node in nodes:
        state.update(node(state))
    return state


def test_langgraph_research_workflow_runs(tmp_path):
    pytest.importorskip("langgraph")

    from quant_lab.agents.workflow import build_research_workflow, default_research_state

    graph = build_research_workflow()
    result = graph.invoke(
        default_research_state(
            "test hypothesis",
            "test universe",
            run_id="FULL-GRAPH-TEST",
            vault_root=str(tmp_path),
            run_kind="test",
        )
    )

    assert result["workflow_stage"] == "reporting_complete"
    assert result["workflow_status"] == "complete"
    assert result["validation_status"] == "not_run"
    assert result["promotion_status"] == "ineligible"
    assert result["approval_required"] is False
    assert tuple(result["validation_gates"]) == REQUIRED_VALIDATION_GATES


def test_nine_agent_workflow_compiles_and_runs(tmp_path):
    pytest.importorskip("langgraph")

    from quant_lab.agents.workflow import build_research_workflow, default_research_state

    graph = build_research_workflow()
    state = default_research_state(
        "momentum factor hypothesis",
        "ASEAN equities",
        run_id="test_run_01",
        vault_root=str(tmp_path),
        run_kind="test",
    )
    result = graph.invoke(state)

    assert result["workflow_stage"] == "reporting_complete"
    assert result["workflow_status"] == "complete"
    assert result["validation_status"] == "not_run"

    assert result["features"] == [
        "momentum_21_63_126",
        "volatility_regime",
        "cross_sectional_rank",
        "liquidity_proxy",
        "seasonal_decomposition",
    ]
    assert result["allocators"] == [
        "equal_weight",
        "inverse_volatility",
        "risk_parity_hrp",
        "max_sharpe_shrinkage",
        "minimum_variance",
    ]
    assert result["optimization_result"] == {}

    assert len(result["audit_log"]) == 8

    assert len(result["obsidian_notes"]) == 8
    for path in result["obsidian_notes"]:
        assert path.startswith(str(tmp_path))
        assert path.endswith(".md")

    assert result["approval_required"] is False
    manifest = ExperimentRunStore(tmp_path, "test_run_01").load()
    assert manifest.workflow_status.value == "complete"
    assert manifest.run_id == result["run_id"]
    assert all(f'run_id: "{result["run_id"]}"' in open(path).read() for path in result["obsidian_notes"])


def test_feature_engineer_node(tmp_path):
    state = make_test_state(tmp_path, run_id="FEATURE-TEST")
    apply_nodes(state, research_director, data_steward)
    result = feature_engineer(state)
    assert result["workflow_stage"] == "model_scope"
    assert "features" in result
    assert len(result["features"]) == 5
    assert "audit_log" in result


def test_portfolio_optimizer_node(tmp_path):
    state = make_test_state(tmp_path, run_id="PORTFOLIO-TEST")
    apply_nodes(state, research_director, data_steward, feature_engineer, modeler)
    result = portfolio_optimizer(state)
    assert result["workflow_stage"] == "validation_scope"
    assert "allocators" in result
    assert len(result["allocators"]) == 5
    assert "audit_log" in result


def test_human_approval_node_stays_closed_before_deterministic_validation(tmp_path):
    state = make_test_state(tmp_path, run_id="APPROVAL-TEST")
    state["candidate_models"] = ["model_a"]
    state["allocators"] = ["alloc_a"]
    state["validation_gates"] = ["gate_a"]
    state["risk_checks"] = ["check_a"]
    state["report_artifacts"] = ["artifact_a"]
    with pytest.raises(RuntimeError, match="gate is closed"):
        human_approval(state)


def test_vault_root_is_mandatory():
    with pytest.raises(ValueError, match="vault_root is required"):
        default_research_state("h", "u", run_id="TEST-NO-VAULT", vault_root="", run_kind="test")


def test_research_manifest_metadata_is_mandatory(tmp_path):
    with pytest.raises(ValueError, match="identity fields"):
        default_research_state(
            "real hypothesis",
            "Malaysia equities",
            run_id="MYS-RESEARCH-001",
            vault_root=str(tmp_path),
            run_kind="research",
        )


def test_fallback_workflow_without_langgraph(tmp_path):
    import importlib
    import sys

    mod_name = "quant_lab.agents.workflow"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    orig = sys.modules.get("langgraph")
    if "langgraph" in sys.modules:
        del sys.modules["langgraph"]
    sys.modules["langgraph"] = None

    try:
        from quant_lab.agents.workflow import build_research_workflow, default_research_state

        graph = build_research_workflow()
        state = default_research_state(
            "fallback hypothesis",
            "fallback universe",
            run_id="fb_01",
            vault_root=str(tmp_path),
            run_kind="test",
        )
        result = graph.invoke(state)
        assert result["workflow_stage"] == "reporting_complete"
        assert result["workflow_status"] == "complete"
        assert len(result["obsidian_notes"]) == 8
    finally:
        if orig is not None:
            sys.modules["langgraph"] = orig
        elif "langgraph" in sys.modules:
            del sys.modules["langgraph"]
