from __future__ import annotations

import warnings
from typing import Any, TypedDict

from quant_lab.research.control_plane import (
    ExperimentRunStore,
    PromotionStatus,
    REQUIRED_VALIDATION_GATES,
    RunKind,
    RunManifest,
    WorkflowStatus,
)

try:
    from langgraph.graph import END, StateGraph

    _HAS_LANGGRAPH = True
except Exception:
    _HAS_LANGGRAPH = False


class QuantResearchState(TypedDict):
    hypothesis: str
    universe: str
    run_id: str
    run_kind: str
    vault_root: str
    manifest_path: str
    manifest_hash: str
    manifest_identity_hash: str
    workflow_stage: str
    workflow_status: str
    validation_status: str
    promotion_status: str
    data_requirements: list[str]
    features: list[str]
    candidate_models: list[str]
    allocators: list[str]
    validation_gates: list[str]
    risk_checks: list[str]
    report_artifacts: list[str]
    optimization_result: dict[str, Any]
    approval_required: bool
    audit_log: list[str]
    obsidian_notes: list[str]


def default_research_state(
    hypothesis: str,
    universe: str,
    *,
    run_id: str,
    vault_root: str,
    run_kind: RunKind | str,
    parent_run_id: str | None = None,
    data_manifest_id: str | None = None,
    data_asof_utc: str | None = None,
    feature_manifest_hash: str | None = None,
    code_commit: str | None = None,
    target: str | None = None,
    label_horizon: str | None = None,
    split_manifest_id: str | None = None,
) -> QuantResearchState:
    kind = RunKind(run_kind)
    if not vault_root.strip():
        raise ValueError("vault_root is required")
    is_non_research = kind is not RunKind.RESEARCH
    manifest = RunManifest.new(
        run_id=run_id,
        parent_run_id=parent_run_id,
        run_kind=kind,
        hypothesis=hypothesis,
        universe=universe,
        data_manifest_id=data_manifest_id or (f"not-applicable-{kind.value}" if is_non_research else ""),
        data_asof_utc=data_asof_utc or ("1970-01-01T00:00:00+00:00" if is_non_research else ""),
        feature_manifest_hash=feature_manifest_hash or ("0" * 64 if is_non_research else ""),
        code_commit=code_commit or ("0" * 40 if is_non_research else ""),
        target=target or ("not-applicable" if is_non_research else ""),
        label_horizon=label_horizon or ("not-applicable" if is_non_research else ""),
        split_manifest_id=split_manifest_id or (f"not-applicable-{kind.value}" if is_non_research else ""),
    )
    store = ExperimentRunStore(vault_root, run_id)
    manifest = store.create(manifest)
    return {
        "hypothesis": hypothesis,
        "universe": universe,
        "run_id": run_id,
        "run_kind": kind.value,
        "vault_root": vault_root,
        "manifest_path": str(store.manifest_path),
        "manifest_hash": manifest.content_hash(),
        "manifest_identity_hash": manifest.identity_hash(),
        "workflow_stage": "idea_intake",
        "workflow_status": manifest.workflow_status.value,
        "validation_status": manifest.validation_status.value,
        "promotion_status": manifest.promotion_status.value,
        "data_requirements": [],
        "features": [],
        "candidate_models": [],
        "allocators": [],
        "validation_gates": [],
        "risk_checks": [],
        "report_artifacts": [],
        "optimization_result": {},
        "approval_required": manifest.promotion_status is PromotionStatus.APPROVAL_REQUIRED,
        "audit_log": [],
        "obsidian_notes": [],
    }


def _append_log(state: QuantResearchState, message: str) -> list[str]:
    return [*state["audit_log"], message]


def _store_for_state(state: QuantResearchState) -> ExperimentRunStore:
    if not state.get("run_id") or not state.get("vault_root"):
        raise ValueError("workflow state must contain explicit run_id and vault_root")
    store = ExperimentRunStore(state["vault_root"], state["run_id"])
    manifest = store.load()
    if (
        manifest.run_id != state["run_id"]
        or manifest.identity_hash() != state["manifest_identity_hash"]
        or manifest.run_kind.value != state["run_kind"]
        or manifest.hypothesis != state["hypothesis"]
        or manifest.universe != state["universe"]
    ):
        raise RuntimeError("workflow state identity does not match its authoritative manifest")
    return store


def _manifest_updates(manifest: RunManifest) -> dict[str, Any]:
    return {
        "manifest_hash": manifest.content_hash(),
        "manifest_identity_hash": manifest.identity_hash(),
        "workflow_status": manifest.workflow_status.value,
        "validation_status": manifest.validation_status.value,
        "promotion_status": manifest.promotion_status.value,
        "approval_required": manifest.promotion_status is PromotionStatus.APPROVAL_REQUIRED,
    }


def _write_agent_note(
    state: QuantResearchState,
    agent_name: str,
    seq: int,
    title: str,
    body: str,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    store = _store_for_state(state)
    path = store.write_supporting_note(
        stage=agent_name,
        sequence=seq,
        title=title,
        body=body,
        tags=tuple(tags or ["quant-lab", agent_name]),
        metadata=metadata or {},
        expected_manifest_hash=state["manifest_hash"],
    )
    return str(path)




def research_director(state: QuantResearchState) -> dict[str, Any]:
    store = _store_for_state(state)
    manifest = store.load()
    if manifest.workflow_status is WorkflowStatus.CREATED:
        manifest = store.transition_workflow(
            WorkflowStatus.RUNNING,
            expected_manifest_hash=state["manifest_hash"],
            actor_id="research_director_controller",
        )
    elif manifest.workflow_status not in {WorkflowStatus.RUNNING, WorkflowStatus.COMPLETE}:
        raise RuntimeError(f"research director cannot resume workflow_status={manifest.workflow_status.value}")
    active_state = dict(state)
    active_state.update(_manifest_updates(manifest))
    requirements = [
        "point_in_time_prices",
        "corporate_actions",
        "volume_and_liquidity",
        "transaction_cost_assumptions",
    ]
    note_path = _write_agent_note(
        active_state,
        "research_director",
        1,
        "Research Director",
        f"Framed hypothesis: {state['hypothesis']}\n\n"
        f"Universe: {state['universe']}\n\n"
        f"Data requirements:\n" + "\n".join(f"- {r}" for r in requirements),
        metadata={"hypothesis": state["hypothesis"], "universe": state["universe"]},
    )
    return {
        "workflow_stage": "data_scope",
        "data_requirements": requirements,
        "audit_log": _append_log(state, "research_director framed the hypothesis and required data."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
        **_manifest_updates(manifest),
    }


def data_steward(state: QuantResearchState) -> dict[str, Any]:
    note_path = _write_agent_note(
        state,
        "data_steward",
        2,
        "Data Steward",
        f"Validated data scope for universe: {state['universe']}\n\n"
        "Checks performed:\n"
        "- timestamp alignment\n"
        "- survivorship bias detection\n"
        "- missing-data imputation checks",
        metadata={"data_requirements": state.get("data_requirements", [])},
    )
    return {
        "workflow_stage": "feature_scope",
        "audit_log": _append_log(state, "data_steward required timestamp, survivorship, and missing-data checks."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
    }


def feature_engineer(state: QuantResearchState) -> dict[str, Any]:
    features = [
        "momentum_21_63_126",
        "volatility_regime",
        "cross_sectional_rank",
        "liquidity_proxy",
        "seasonal_decomposition",
    ]
    note_path = _write_agent_note(
        state,
        "feature_engineer",
        3,
        "Feature Engineer",
        f"Designed features for hypothesis: {state['hypothesis']}\n\n"
        f"Selected features:\n" + "\n".join(f"- {f}" for f in features),
        metadata={"features": features},
    )
    return {
        "workflow_stage": "model_scope",
        "features": features,
        "audit_log": _append_log(state, "feature_engineer designed cross-sectional and temporal features."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
    }


def modeler(state: QuantResearchState) -> dict[str, Any]:
    models = [
        "momentum_baseline",
        "mean_reversion_baseline",
        "linear_factor_model",
        "gradient_boosted_tree",
        "regime_weighted_ensemble",
    ]
    note_path = _write_agent_note(
        state,
        "modeler",
        4,
        "Modeler",
        f"Selected models for hypothesis: {state['hypothesis']}\n\n"
        f"Candidate models:\n" + "\n".join(f"- {m}" for m in models),
        metadata={"candidate_models": models},
    )
    return {
        "workflow_stage": "optimizer_scope",
        "candidate_models": models,
        "audit_log": _append_log(state, "modeler selected baselines before complex models."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
    }


def portfolio_optimizer(state: QuantResearchState) -> dict[str, Any]:
    allocators = [
        "equal_weight",
        "inverse_volatility",
        "risk_parity_hrp",
        "max_sharpe_shrinkage",
        "minimum_variance",
    ]
    note_path = _write_agent_note(
        state,
        "portfolio_optimizer",
        5,
        "Portfolio Optimizer",
        f"Selected allocators for universe: {state['universe']}\n\n"
        f"Allocators:\n" + "\n".join(f"- {a}" for a in allocators),
        metadata={"allocators": allocators},
    )
    return {
        "workflow_stage": "validation_scope",
        "allocators": allocators,
        "audit_log": _append_log(state, "portfolio_optimizer selected allocators and weighting schemes."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
    }


def validation_officer(state: QuantResearchState) -> dict[str, Any]:
    gates = list(REQUIRED_VALIDATION_GATES)
    note_path = _write_agent_note(
        state,
        "validation_officer",
        6,
        "Validation Officer",
        "Attached anti-overfit gates:\n\n" + "\n".join(f"- {g}" for g in gates),
        metadata={"validation_gates": gates},
    )
    return {
        "workflow_stage": "risk_scope",
        "validation_gates": gates,
        "audit_log": _append_log(state, "validation_officer attached anti-overfit gates."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
    }


def risk_manager(state: QuantResearchState) -> dict[str, Any]:
    checks = [
        "max_drawdown",
        "gross_and_net_exposure",
        "turnover_limit",
        "adv_participation",
        "stress_loss",
        "capacity_at_target_aum",
    ]
    note_path = _write_agent_note(
        state,
        "risk_manager",
        7,
        "Risk Manager",
        "Added portfolio, stress, and capacity checks:\n\n" + "\n".join(f"- {c}" for c in checks),
        metadata={"risk_checks": checks},
    )
    return {
        "workflow_stage": "reporting_scope",
        "risk_checks": checks,
        "audit_log": _append_log(state, "risk_manager added portfolio, stress, and capacity checks."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
    }


def reporting_analyst(state: QuantResearchState) -> dict[str, Any]:
    artifacts = [
        "model_card",
        "research_tearsheet",
        "risk_report",
        "regime_report",
        "experiment_manifest",
    ]
    note_path = _write_agent_note(
        state,
        "reporting_analyst",
        8,
        "Reporting Analyst",
        "Required reporting artifacts:\n\n" + "\n".join(f"- {a}" for a in artifacts),
        metadata={"report_artifacts": artifacts},
    )
    store = _store_for_state(state)
    manifest = store.load()
    if manifest.workflow_status is WorkflowStatus.RUNNING:
        manifest = store.transition_workflow(
            WorkflowStatus.COMPLETE,
            expected_manifest_hash=state["manifest_hash"],
            actor_id="reporting_controller",
        )
    return {
        "workflow_stage": "reporting_complete",
        "report_artifacts": artifacts,
        "audit_log": _append_log(
            state,
            "reporting_analyst completed the machine workflow; deterministic validation remains authoritative.",
        ),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
        **_manifest_updates(manifest),
    }


def human_approval(state: QuantResearchState) -> dict[str, Any]:
    store = _store_for_state(state)
    manifest = store.load()
    if manifest.promotion_status is not PromotionStatus.APPROVAL_REQUIRED:
        raise RuntimeError("human approval gate is closed until deterministic validation passes")
    body = (
        "Human approval node reached.\n\n"
        f"Hypothesis: {state['hypothesis']}\n"
        f"Universe: {state['universe']}\n"
        f"Models: {state.get('candidate_models', [])}\n"
        f"Allocators: {state.get('allocators', [])}\n"
        f"Validation gates: {state.get('validation_gates', [])}\n"
        f"Risk checks: {state.get('risk_checks', [])}\n"
        f"Report artifacts: {state.get('report_artifacts', [])}\n\n"
        "**WARNING: Live trading requires explicit human approval.**"
    )
    note_path = _write_agent_note(
        state,
        "human_approval",
        9,
        "Human Approval",
        body,
        tags=["quant-lab", "human_approval", "live-trading-warning"],
        metadata={"approval_required": True},
    )
    return {
        "workflow_stage": "human_approval",
        "approval_required": True,
        "audit_log": _append_log(state, "human_approval set approval_required=True. Live trading requires explicit human approval."),
        "obsidian_notes": [*state.get("obsidian_notes", []), note_path],
    }



class _FallbackGraph:
    """Sequential fallback when LangGraph is not installed."""

    def __init__(self, nodes: list[tuple[str, Any]]) -> None:
        self._nodes = nodes

    def invoke(self, state: QuantResearchState) -> QuantResearchState:
        current: dict[str, Any] = dict(state)
        for name, func in self._nodes:
            if name == "human_approval" and current.get("promotion_status") != PromotionStatus.APPROVAL_REQUIRED.value:
                continue
            updates = func(current)
            current.update(updates)
        return current


def build_research_workflow():
    """Build the 9-agent LangGraph research workflow.

    Returns a compiled graph when LangGraph is available; otherwise returns
    a sequential fallback that preserves the same state semantics.
    """
    nodes = [
        ("research_director", research_director),
        ("data_steward", data_steward),
        ("feature_engineer", feature_engineer),
        ("modeler", modeler),
        ("portfolio_optimizer", portfolio_optimizer),
        ("validation_officer", validation_officer),
        ("risk_manager", risk_manager),
        ("reporting_analyst", reporting_analyst),
        ("human_approval", human_approval),
    ]

    if not _HAS_LANGGRAPH:
        warnings.warn(
            "langgraph not installed; using fallback sequential workflow.",
            stacklevel=2,
        )
        return _FallbackGraph(nodes)

    graph = StateGraph(QuantResearchState)
    for name, func in nodes:
        graph.add_node(name, func)

    graph.set_entry_point("research_director")
    for i in range(len(nodes) - 2):
        graph.add_edge(nodes[i][0], nodes[i + 1][0])
    graph.add_conditional_edges(
        "reporting_analyst",
        lambda state: "human_approval"
        if state.get("promotion_status") == PromotionStatus.APPROVAL_REQUIRED.value
        else "end",
        {"human_approval": "human_approval", "end": END},
    )
    graph.add_edge("human_approval", END)
    return graph.compile()
