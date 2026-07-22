from __future__ import annotations

import base64
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

import quant_lab.research.control_plane as control_plane
import quant_lab.research._control_plane_store as control_plane_store

from quant_lab.agents.workflow import build_research_workflow, default_research_state
from quant_lab.research.control_plane import (
    ApprovalDecision,
    ApprovalError,
    ApprovalTrustStore,
    AuthorizedHumanSigner,
    ConcurrentUpdateError,
    ExperimentRunStore,
    GateStatus,
    HumanApprovalRecord,
    HumanSupersessionRecord,
    IllegalTransitionError,
    IntegrityError,
    PromotionStatus,
    REQUIRED_VALIDATION_GATES,
    RunKind,
    RunManifest,
    ValidationArtifact,
    ValidationRunnerTrustStore,
    ValidationStatus,
    WorkflowStatus,
)


DATA_ASOF = "2026-07-19T00:00:00+00:00"
FEATURE_HASH = "a" * 64
CODE_COMMIT = "b" * 40
CANDIDATE_ID = "MYS-CANDIDATE-001"
CANDIDATE_HASH = "1" * 64
DATA_MANIFEST_HASH = "c" * 64
SPLIT_MANIFEST_HASH = "d" * 64
CALCULATOR_SOURCE_HASH = "f" * 64
RUNNER_ID = "PY-VALIDATION-RUNNER-001"
RUNNER_KEY_ID = "PY-VALIDATION-KEY-001"


def calculator_id_for(gate_id):
    return f"CALCULATOR-{REQUIRED_VALIDATION_GATES.index(gate_id) + 1:02d}"


def validation_policy_payload():
    thresholds = {
        "point_in_time_data_audit": {"minimum_row_count": 1_000},
        "purged_embargoed_walk_forward_splits": {
            "minimum_fold_count": 5,
            "minimum_purge_observations": 1,
            "minimum_embargo_observations": 1,
        },
        "cpcv_pbo": {"minimum_path_count": 16, "maximum_pbo": 0.50},
        "dsr_psr": {"minimum_probability": 0.80},
        "multiple_testing_fdr": {
            "alpha": 0.05,
            "allowed_methods": ["benjamini_hochberg", "benjamini_yekutieli"],
        },
        "cost_sensitivity": {
            "minimum_stress_multiplier": 2.0,
            "minimum_stressed_net_sharpe": 0.0,
        },
        "drawdown_turnover": {
            "maximum_drawdown": -0.20,
            "maximum_annual_turnover": 3.0,
        },
        "liquidity_capacity": {
            "maximum_adv_participation": 0.05,
            "minimum_capacity_aum": 10_000_000.0,
        },
        "regime_stability": {
            "minimum_regime_count": 3,
            "minimum_observations_per_regime": 63,
        },
        "sealed_holdout": {
            "minimum_observation_count": 63,
            "required_release_count": 1,
        },
        "data_feature_split_lineage": {},
        "code_reproducibility": {"minimum_rerun_count": 2},
    }
    return {
        "policy_id": "MYS-INSTITUTIONAL-VALIDATION-V1",
        "gates": {
            gate_id: {
                "allowed_calculators": [
                    {
                        "calculator_id": calculator_id_for(gate_id),
                        "calculator_version": "1-0-0",
                        "calculator_source_sha256": CALCULATOR_SOURCE_HASH,
                    }
                ],
                "thresholds": thresholds[gate_id],
            }
            for gate_id in REQUIRED_VALIDATION_GATES
        },
    }


@pytest.fixture
def runner_authority(tmp_path, monkeypatch):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    trust_payload = {
        "schema_version": 1,
        "validation_policy": validation_policy_payload(),
        "runners": [
            {
                "runner_id": RUNNER_ID,
                "key_id": RUNNER_KEY_ID,
                "role": "deterministic_validation_runner",
                "public_key_base64": base64.b64encode(public_bytes).decode("ascii"),
                "valid_from": "2026-01-01T00:00:00+00:00",
                "valid_until": "2027-01-01T00:00:00+00:00",
                "revoked_at": None,
            }
        ],
    }
    trust_path = tmp_path.parent / f"{tmp_path.name}-validation-runners.json"
    trust_path.write_text(json.dumps(trust_payload, sort_keys=True))
    trust_hash = hashlib.sha256(trust_path.read_bytes()).hexdigest()
    trust_path.chmod(0o400)
    monkeypatch.setattr(
        ValidationRunnerTrustStore,
        "_read_admin_anchor",
        classmethod(lambda cls: trust_hash),
    )
    return private_key, ValidationRunnerTrustStore.from_file(trust_path)


def passing_metrics(gate_id, *, feature_hash=FEATURE_HASH, code_commit=CODE_COMMIT):
    return {
        "point_in_time_data_audit": {
            "row_count": 10_000,
            "future_leakage_count": 0,
            "survivorship_violation_count": 0,
            "timestamp_lineage_complete": True,
        },
        "purged_embargoed_walk_forward_splits": {
            "fold_count": 6,
            "purge_observations": 21,
            "embargo_observations": 5,
            "overlap_violation_count": 0,
        },
        "cpcv_pbo": {"path_count": 20, "pbo": 0.20, "pbo_limit": 0.50},
        "dsr_psr": {
            "deflated_sharpe_probability": 0.95,
            "probabilistic_sharpe_probability": 0.97,
            "minimum_probability": 0.80,
        },
        "multiple_testing_fdr": {
            "hypotheses_tested": 20,
            "method": "benjamini_hochberg",
            "adjusted_p_value": 0.02,
            "alpha": 0.05,
        },
        "cost_sensitivity": {
            "base_cost_bps": 10.0,
            "stressed_cost_bps": 20.0,
            "stressed_net_sharpe": 0.6,
            "minimum_net_sharpe": 0.0,
        },
        "drawdown_turnover": {
            "max_drawdown": -0.15,
            "max_drawdown_limit": -0.20,
            "annual_turnover": 2.0,
            "turnover_limit": 3.0,
        },
        "liquidity_capacity": {
            "adv_participation": 0.03,
            "adv_participation_limit": 0.05,
            "capacity_aum": 20_000_000.0,
            "target_aum": 10_000_000.0,
        },
        "regime_stability": {
            "regime_count": 4,
            "minimum_observations_per_regime": 63,
            "unstable_regime_count": 0,
        },
        "sealed_holdout": {
            "sealed": True,
            "release_count": 1,
            "observation_count": 126,
            "optimization_access_count": 0,
            "passed": True,
            "holdout_manifest_sha256": "6" * 64,
            "release_ledger_sha256": "7" * 64,
        },
        "data_feature_split_lineage": {
            "data_manifest_sha256": DATA_MANIFEST_HASH,
            "feature_manifest_sha256": feature_hash,
            "split_manifest_sha256": SPLIT_MANIFEST_HASH,
        },
        "code_reproducibility": {
            "code_commit": code_commit,
            "environment_lock_sha256": "e" * 64,
            "random_seed": 17,
            "rerun_count": 2,
            "reproduced": True,
        },
    }[gate_id]


def research_state(tmp_path, run_id="MYS-RESEARCH-001"):
    return default_research_state(
        "Lagged medium-horizon momentum persists after Malaysia trading costs.",
        "Malaysia point-in-time ordinary equities",
        run_id=run_id,
        vault_root=str(tmp_path),
        run_kind="research",
        data_manifest_id="LSEG-MYS-PIT-20260719",
        data_asof_utc=DATA_ASOF,
        feature_manifest_hash=FEATURE_HASH,
        code_commit=CODE_COMMIT,
        target="next_21d_excess_return",
        label_horizon="21_trading_days",
        split_manifest_id="MYS-PURGED-CV-001",
    )


def completed_run(tmp_path, run_id="MYS-RESEARCH-001", runner_authority=None):
    state = research_state(tmp_path, run_id)
    if runner_authority is not None:
        ExperimentRunStore(tmp_path, run_id).register_validation_policy(
            runner_authority[1],
            expected_manifest_hash=state["manifest_hash"],
        )
    result = build_research_workflow().invoke(state)
    assert result["workflow_status"] == "complete"
    return ExperimentRunStore(tmp_path, run_id), result


def signed_validation_artifact(artifact, private_key):
    signature = base64.b64encode(private_key.sign(artifact.signing_bytes())).decode("ascii")
    return replace(artifact, runner_signature=signature)


def write_gate_artifacts(
    store,
    runner_authority,
    *,
    status=GateStatus.PASS,
    candidate_id=CANDIDATE_ID,
):
    private_key, runner_trust_store = runner_authority
    manifest = store.load()
    paths = {}
    for index, gate_id in enumerate(REQUIRED_VALIDATION_GATES, start=1):
        artifact = ValidationArtifact(
            gate_id=gate_id,
            run_id=manifest.run_id,
            candidate_id=candidate_id,
            candidate_hash=CANDIDATE_HASH,
            data_manifest_id=manifest.data_manifest_id,
            data_asof_utc=manifest.data_asof_utc,
            feature_manifest_hash=manifest.feature_manifest_hash,
            code_commit=manifest.code_commit,
            split_manifest_id=manifest.split_manifest_id,
            status=status,
            producer_execution_id=f"PY-VALIDATOR-{index:02d}",
            calculator_id=calculator_id_for(gate_id),
            calculator_version="1-0-0",
            calculator_source_sha256=CALCULATOR_SOURCE_HASH,
            input_hashes={
                "candidate": CANDIDATE_HASH,
                "data_manifest": DATA_MANIFEST_HASH,
                "feature_manifest": manifest.feature_manifest_hash,
                "split_manifest": SPLIT_MANIFEST_HASH,
                "calculator_source": CALCULATOR_SOURCE_HASH,
            },
            runner_id=RUNNER_ID,
            runner_key_id=RUNNER_KEY_ID,
            runner_signature=base64.b64encode(b"0" * 64).decode("ascii"),
            metrics=passing_metrics(
                gate_id,
                feature_hash=manifest.feature_manifest_hash,
                code_commit=manifest.code_commit,
            ),
            generated_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).replace(
                microsecond=0
            ).isoformat(),
        )
        artifact = signed_validation_artifact(artifact, private_key)
        paths[gate_id] = store.write_validation_artifact(
            artifact,
            runner_trust_store=runner_trust_store,
        )
    return paths


def replace_gate_artifact(store, paths, gate_id, status, runner_authority, *, metrics=None):
    private_key, runner_trust_store = runner_authority
    manifest = store.load()
    artifact = ValidationArtifact(
        gate_id=gate_id,
        run_id=manifest.run_id,
        candidate_id=CANDIDATE_ID,
        candidate_hash=CANDIDATE_HASH,
        data_manifest_id=manifest.data_manifest_id,
        data_asof_utc=manifest.data_asof_utc,
        feature_manifest_hash=manifest.feature_manifest_hash,
        code_commit=manifest.code_commit,
        split_manifest_id=manifest.split_manifest_id,
        status=status,
        producer_execution_id=f"PY-RECHECK-{gate_id[:12].upper()}",
        calculator_id=calculator_id_for(gate_id),
        calculator_version="1-0-0",
        calculator_source_sha256=CALCULATOR_SOURCE_HASH,
        input_hashes={
            "candidate": CANDIDATE_HASH,
            "data_manifest": DATA_MANIFEST_HASH,
            "feature_manifest": manifest.feature_manifest_hash,
            "split_manifest": SPLIT_MANIFEST_HASH,
            "calculator_source": CALCULATOR_SOURCE_HASH,
        },
        runner_id=RUNNER_ID,
        runner_key_id=RUNNER_KEY_ID,
        runner_signature=base64.b64encode(b"0" * 64).decode("ascii"),
        metrics=metrics
        or passing_metrics(
            gate_id,
            feature_hash=manifest.feature_manifest_hash,
            code_commit=manifest.code_commit,
        ),
        generated_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).replace(
            microsecond=0
        ).isoformat(),
    )
    artifact = signed_validation_artifact(artifact, private_key)
    paths[gate_id] = store.write_validation_artifact(
        artifact,
        runner_trust_store=runner_trust_store,
    )


def approve_ready_run(tmp_path, runner_authority, run_id="MYS-RESEARCH-001"):
    store, state = completed_run(tmp_path, run_id, runner_authority)
    paths = write_gate_artifacts(store, runner_authority)
    manifest, bundle = store.admit_validation(
        candidate_id=CANDIDATE_ID,
        artifact_paths=paths,
        runner_trust_store=runner_authority[1],
        expected_manifest_hash=state["manifest_hash"],
    )
    assert manifest.validation_status is ValidationStatus.PASS
    assert manifest.promotion_status is PromotionStatus.APPROVAL_REQUIRED
    return store, manifest, bundle


def signed_record(
    manifest,
    bundle,
    *,
    trust_path,
    monkeypatch,
    decision=ApprovalDecision.APPROVED,
    approval_id="APR-001",
    return_private=False,
):
    private_key = Ed25519PrivateKey.generate()
    public_bytes = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    signer = AuthorizedHumanSigner(
        approved_by="Research Approver",
        key_id="BEN-HUMAN-KEY-001",
        role="authorized_human_approver",
        public_key_base64=base64.b64encode(public_bytes).decode("ascii"),
        valid_from="2026-01-01T00:00:00+00:00",
        valid_until="2027-01-01T00:00:00+00:00",
    )
    unsigned = HumanApprovalRecord(
        approval_id=approval_id,
        run_id=manifest.run_id,
        manifest_identity_hash=manifest.identity_hash(),
        decision=decision,
        approved_by="Research Approver",
        approved_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).replace(microsecond=0).isoformat(),
        candidate_id=bundle.candidate_id,
        evidence_bundle_hash=bundle.content_hash(),
        limitations_acknowledged=("Historical results do not guarantee future returns.",),
        signer_key_id=signer.key_id,
        signature=base64.b64encode(b"0" * 64).decode("ascii"),
    )
    signature = base64.b64encode(private_key.sign(unsigned.signing_bytes())).decode("ascii")
    trust_payload = {
        "schema_version": 1,
        "signers": [
            {
                "approved_by": signer.approved_by,
                "key_id": signer.key_id,
                "role": signer.role,
                "public_key_base64": signer.public_key_base64,
                "valid_from": signer.valid_from,
                "valid_until": signer.valid_until,
                "revoked_at": signer.revoked_at,
            }
        ],
    }
    trust_path.write_text(json.dumps(trust_payload, sort_keys=True))
    trust_hash = hashlib.sha256(trust_path.read_bytes()).hexdigest()
    trust_path.chmod(0o400)
    monkeypatch.setattr(
        ApprovalTrustStore,
        "_read_admin_anchor",
        classmethod(lambda cls: trust_hash),
    )
    result = (replace(unsigned, signature=signature), ApprovalTrustStore.from_file(trust_path))
    return (*result, private_key) if return_private else result


def test_manifest_has_exact_authoritative_fields_and_rejects_unknown_fields(tmp_path):
    state = research_state(tmp_path)
    payload = json.loads(open(state["manifest_path"]).read())
    assert set(payload) == {
        "run_id",
        "parent_run_id",
        "run_kind",
        "promotion_eligible",
        "hypothesis",
        "universe",
        "data_manifest_id",
        "data_asof_utc",
        "feature_manifest_hash",
        "code_commit",
        "target",
        "label_horizon",
        "split_manifest_id",
        "workflow_status",
        "validation_status",
        "promotion_status",
        "created_at",
        "updated_at",
    }
    with pytest.raises(ValueError, match="fields mismatch"):
        RunManifest.from_dict({**payload, "agent_comment": "override"})
    with pytest.raises(TypeError, match="textual fields"):
        RunManifest.from_dict({**payload, "hypothesis": 123})


def test_store_rejects_preapproved_manifest_without_transitions_or_evidence(tmp_path):
    initial = RunManifest.new(
        run_id="MYS-FORGED-001",
        parent_run_id=None,
        run_kind="research",
        hypothesis="A genuine-looking but unvalidated Malaysia hypothesis.",
        universe="Malaysia ordinary equities",
        data_manifest_id="LSEG-MYS-PIT-001",
        data_asof_utc=DATA_ASOF,
        feature_manifest_hash=FEATURE_HASH,
        code_commit=CODE_COMMIT,
        target="return",
        label_horizon="21d",
        split_manifest_id="SPLIT-001",
    )
    forged = replace(
        initial,
        workflow_status=WorkflowStatus.COMPLETE,
        validation_status=ValidationStatus.PASS,
        promotion_status=PromotionStatus.APPROVED,
    )
    with pytest.raises(IllegalTransitionError, match="must start"):
        ExperimentRunStore(tmp_path, forged.run_id).create(forged)


def test_parent_run_must_exist_and_match_run_kind(tmp_path):
    with pytest.raises(FileNotFoundError):
        default_research_state(
            "Child hypothesis with explicit but missing lineage parent.",
            "Malaysia equities",
            run_id="MYS-CHILD-001",
            parent_run_id="MYS-MISSING-001",
            vault_root=str(tmp_path),
            run_kind="research",
            data_manifest_id="DATA-001",
            data_asof_utc=DATA_ASOF,
            feature_manifest_hash=FEATURE_HASH,
            code_commit=CODE_COMMIT,
            target="return",
            label_horizon="21d",
            split_manifest_id="SPLIT-001",
        )


@pytest.mark.parametrize("run_id", ("../escaped", "a/b", "/absolute", ".."))
def test_run_id_cannot_escape_the_vault(tmp_path, run_id):
    with pytest.raises(ValueError, match="path-safe"):
        ExperimentRunStore(tmp_path, run_id)


@pytest.mark.parametrize("kind", (RunKind.TEST, RunKind.SYNTHETIC, RunKind.SMOKE))
def test_non_research_runs_cannot_become_promotable_even_with_passing_artifacts(
    tmp_path, kind, runner_authority
):
    state = default_research_state(
        "synthetic hypothesis",
        "synthetic universe",
        run_id=f"{kind.value.upper()}-RUN-001",
        vault_root=str(tmp_path),
        run_kind=kind,
    )
    ExperimentRunStore(tmp_path, state["run_id"]).register_validation_policy(
        runner_authority[1],
        expected_manifest_hash=state["manifest_hash"],
    )
    result = build_research_workflow().invoke(state)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    paths = write_gate_artifacts(store, runner_authority)
    manifest, bundle = store.admit_validation(
        candidate_id=CANDIDATE_ID,
        artifact_paths=paths,
        runner_trust_store=runner_authority[1],
        expected_manifest_hash=result["manifest_hash"],
    )
    assert bundle.overall_status() is ValidationStatus.PASS
    assert manifest.promotion_eligible is False
    assert manifest.promotion_status is PromotionStatus.INELIGIBLE


def test_placeholder_and_ambiguous_research_ids_fail_closed(tmp_path):
    for index, (run_id, hypothesis, universe) in enumerate(
        (
            ("MYS-RESEARCH-H", "h", "Malaysia equities"),
            ("MYS-RESEARCH-U", "real hypothesis", "u"),
            ("deadbeef", "real hypothesis", "Malaysia equities"),
            ("MYS-TEMP-001", "real hypothesis", "Malaysia equities"),
            ("MYS-RESEARCH-H", "real hypothesis", "Malaysia equities"),
            ("MYS-RESEARCH-U", "real hypothesis", "Malaysia equities"),
        )
    ):
        manifest = RunManifest.new(
            run_id=run_id,
            parent_run_id=None,
            run_kind="research",
            hypothesis=hypothesis,
            universe=universe,
            data_manifest_id=f"DATA-{index}",
            data_asof_utc=DATA_ASOF,
            feature_manifest_hash=FEATURE_HASH,
            code_commit=CODE_COMMIT,
            target="return",
            label_horizon="21d",
            split_manifest_id=f"SPLIT-{index}",
        )
        assert manifest.promotion_eligible is False


def test_workflow_state_machine_rejects_illegal_jump_and_stale_writer(tmp_path):
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    with pytest.raises(IllegalTransitionError, match="created -> complete"):
        store.transition_workflow(
            WorkflowStatus.COMPLETE,
            expected_manifest_hash=state["manifest_hash"],
            actor_id="bad-controller",
        )
    running = store.transition_workflow(
        WorkflowStatus.RUNNING,
        expected_manifest_hash=state["manifest_hash"],
        actor_id="research-director",
    )
    with pytest.raises(ConcurrentUpdateError, match="stale manifest"):
        store.transition_workflow(
            WorkflowStatus.FAILED,
            expected_manifest_hash=state["manifest_hash"],
            actor_id="stale-controller",
        )
    assert running.workflow_status is WorkflowStatus.RUNNING


def test_same_note_is_idempotent_under_concurrency(tmp_path):
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    running = store.transition_workflow(
        WorkflowStatus.RUNNING,
        expected_manifest_hash=state["manifest_hash"],
        actor_id="research-director",
    )

    def write_once(_):
        return store.write_supporting_note(
            stage="research_director",
            sequence=1,
            title="Research Director",
            body="Deterministic body",
            tags=("quant-lab",),
            metadata={"root_run_id": state["run_id"]},
            expected_manifest_hash=running.content_hash(),
        )

    with ThreadPoolExecutor(max_workers=16) as pool:
        paths = list(pool.map(write_once, range(64)))
    assert len(set(paths)) == 1
    event_files = list((store.run_root / ".control" / "events").glob("*.json"))
    stage_events = [
        json.loads(path.read_text())
        for path in event_files
        if json.loads(path.read_text())["event_type"] == "workflow_stage_completed"
    ]
    assert len(stage_events) == 1


def test_stage_order_and_authoritative_note_metadata_cannot_be_bypassed(tmp_path):
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    running = store.transition_workflow(
        WorkflowStatus.RUNNING,
        expected_manifest_hash=state["manifest_hash"],
        actor_id="research-director",
    )
    with pytest.raises(IllegalTransitionError, match="expected research_director"):
        store.write_supporting_note(
            stage="feature_engineer",
            sequence=3,
            title="Feature Engineer",
            body="Out of order",
            tags=("quant-lab",),
            metadata={},
            expected_manifest_hash=running.content_hash(),
        )
    with pytest.raises(ValueError, match="authoritative fields"):
        store.write_supporting_note(
            stage="research_director",
            sequence=1,
            title="Research Director",
            body="Attempted identity override",
            tags=("quant-lab",),
            metadata={"run_id": "OTHER-RUN-001"},
            expected_manifest_hash=running.content_hash(),
        )


def test_manifest_recovers_after_event_commit_before_current_projection(tmp_path):
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    created_content = store.manifest_path.read_bytes()
    running = store.transition_workflow(
        WorkflowStatus.RUNNING,
        expected_manifest_hash=state["manifest_hash"],
        actor_id="research-director",
    )
    store.manifest_path.write_bytes(created_content)
    recovered = store.load()
    assert recovered.content_hash() == running.content_hash()
    assert recovered.workflow_status is WorkflowStatus.RUNNING


def test_uncommitted_manifest_version_is_quarantined_before_retry(tmp_path):
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    current = store.load()
    orphan = replace(
        current,
        workflow_status=WorkflowStatus.RUNNING,
        updated_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).replace(
            microsecond=0
        ).isoformat(),
    )
    orphan_path = (
        store.run_root
        / ".control"
        / "manifests"
        / f"000002-{orphan.content_hash()}.json"
    )
    orphan_path.write_bytes(orphan.canonical_bytes())
    running = store.transition_workflow(
        WorkflowStatus.RUNNING,
        expected_manifest_hash=state["manifest_hash"],
        actor_id="research-director",
    )
    assert running.workflow_status is WorkflowStatus.RUNNING
    assert not orphan_path.exists()
    assert (
        store.run_root / ".control" / "orphans" / "manifests" / orphan_path.name
    ).is_file()
    assert store.verify_integrity()["status"] == "pass"


def test_exclusive_atomic_write_never_exposes_partial_final_file(tmp_path, monkeypatch):
    target = tmp_path / "immutable.json"

    def fail_before_publish(source, destination):
        raise RuntimeError("simulated interruption before atomic publication")

    monkeypatch.setattr(control_plane_store.os, "link", fail_before_publish)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        control_plane._atomic_write(target, b"complete-content", exclusive=True)
    assert not target.exists()
    assert not list(tmp_path.glob(".*.tmp"))


def test_workflow_is_idempotent_under_concurrent_full_invocations(tmp_path):
    state = research_state(tmp_path)
    workflow = build_research_workflow()
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(lambda _: workflow.invoke(state), range(32)))
    assert {result["workflow_status"] for result in results} == {"complete"}
    store = ExperimentRunStore(tmp_path, state["run_id"])
    assert store.verify_integrity()["status"] == "pass"


def test_semantically_empty_promotion_event_is_rejected_on_replay(tmp_path):
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    with store._locked():
        store._append_event_locked(
            event_type="promotion_transition",
            before_hash=state["manifest_hash"],
            after_hash=state["manifest_hash"],
            actor_type="controller",
            actor_id="fabricated-controller",
            payload={},
        )
    with pytest.raises(IntegrityError, match="payload fields mismatch"):
        store.verify_integrity()


def test_validation_requires_exact_set_and_rejects_tampering(tmp_path, runner_authority):
    store, state = completed_run(tmp_path, runner_authority=runner_authority)
    paths = write_gate_artifacts(store, runner_authority)
    missing = dict(paths)
    missing.pop(REQUIRED_VALIDATION_GATES[-1])
    with pytest.raises(ValueError, match="exact deterministic gate set"):
        store.admit_validation(
            candidate_id=CANDIDATE_ID,
            artifact_paths=missing,
            runner_trust_store=runner_authority[1],
            expected_manifest_hash=state["manifest_hash"],
        )
    tampered_path = paths[REQUIRED_VALIDATION_GATES[0]]
    payload = json.loads(tampered_path.read_text())
    payload["run_id"] = "OTHER-RUN-001"
    tampered_path.write_text(json.dumps(payload))
    with pytest.raises(IntegrityError, match="bound to another manifest"):
        store.admit_validation(
            candidate_id=CANDIDATE_ID,
            artifact_paths=paths,
            runner_trust_store=runner_authority[1],
            expected_manifest_hash=state["manifest_hash"],
        )


def test_validation_policy_cannot_be_registered_after_workflow_starts(
    tmp_path, runner_authority
):
    store, state = completed_run(tmp_path)
    with pytest.raises(IllegalTransitionError, match="still created"):
        store.register_validation_policy(
            runner_authority[1],
            expected_manifest_hash=state["manifest_hash"],
        )


def test_registered_validation_policy_cannot_be_swapped_for_relaxed_thresholds(
    tmp_path, monkeypatch, runner_authority
):
    private_key, original_store = runner_authority
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    store.register_validation_policy(
        original_store,
        expected_manifest_hash=state["manifest_hash"],
    )
    relaxed_payload = json.loads(original_store.source_path.read_text())
    relaxed_payload["validation_policy"]["gates"]["cpcv_pbo"]["thresholds"][
        "maximum_pbo"
    ] = 0.99
    relaxed_path = tmp_path.parent / f"{tmp_path.name}-relaxed-policy.json"
    relaxed_path.write_text(json.dumps(relaxed_payload, sort_keys=True))
    relaxed_hash = hashlib.sha256(relaxed_path.read_bytes()).hexdigest()
    relaxed_path.chmod(0o400)
    monkeypatch.setattr(
        ValidationRunnerTrustStore,
        "_read_admin_anchor",
        classmethod(lambda cls: frozenset({original_store.source_hash, relaxed_hash})),
    )
    relaxed_store = ValidationRunnerTrustStore.from_file(relaxed_path)
    result = build_research_workflow().invoke(state)
    assert result["workflow_status"] == "complete"
    with pytest.raises(IntegrityError, match="differs from the preregistered policy"):
        write_gate_artifacts(store, (private_key, relaxed_store))


def test_generic_pass_flags_are_not_real_gate_evidence(tmp_path, runner_authority):
    store, _ = completed_run(tmp_path, runner_authority=runner_authority)
    manifest = store.load()
    with pytest.raises(ValueError, match="metrics fields mismatch"):
        store.write_validation_artifact(
            ValidationArtifact(
                gate_id="point_in_time_data_audit",
                run_id=manifest.run_id,
                candidate_id=CANDIDATE_ID,
                candidate_hash=CANDIDATE_HASH,
                data_manifest_id=manifest.data_manifest_id,
                data_asof_utc=manifest.data_asof_utc,
                feature_manifest_hash=manifest.feature_manifest_hash,
                code_commit=manifest.code_commit,
                split_manifest_id=manifest.split_manifest_id,
                status=GateStatus.PASS,
                producer_execution_id="PY-GENERIC-001",
                calculator_id="CALCULATOR-GENERIC-001",
                calculator_version="1-0-0",
                calculator_source_sha256=CALCULATOR_SOURCE_HASH,
                input_hashes={
                    "candidate": CANDIDATE_HASH,
                    "data_manifest": DATA_MANIFEST_HASH,
                    "feature_manifest": manifest.feature_manifest_hash,
                    "split_manifest": SPLIT_MANIFEST_HASH,
                    "calculator_source": CALCULATOR_SOURCE_HASH,
                },
                runner_id=RUNNER_ID,
                runner_key_id=RUNNER_KEY_ID,
                runner_signature=base64.b64encode(b"0" * 64).decode("ascii"),
                metrics={"passed": True},
                generated_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).replace(
                    microsecond=0
                ).isoformat(),
            ),
            runner_trust_store=runner_authority[1],
        )


def test_plausible_metrics_without_valid_runner_attestation_are_rejected(tmp_path, runner_authority):
    store, _ = completed_run(tmp_path, runner_authority=runner_authority)
    paths = write_gate_artifacts(store, runner_authority)
    payload = json.loads(paths["point_in_time_data_audit"].read_text())
    artifact = ValidationArtifact.from_dict(payload)
    forged = replace(
        artifact,
        metrics={**artifact.metrics, "row_count": artifact.metrics["row_count"] + 1},
    )
    with pytest.raises(IntegrityError, match="runner attestation"):
        store.write_validation_artifact(
            forged,
            runner_trust_store=runner_authority[1],
        )


def test_inconclusive_pit_evidence_parks_for_data(tmp_path, runner_authority):
    store, state = completed_run(tmp_path, runner_authority=runner_authority)
    paths = write_gate_artifacts(store, runner_authority)
    replace_gate_artifact(
        store,
        paths,
        "point_in_time_data_audit",
        GateStatus.INCONCLUSIVE,
        runner_authority,
    )
    manifest, _ = store.admit_validation(
        candidate_id=CANDIDATE_ID,
        artifact_paths=paths,
        runner_trust_store=runner_authority[1],
        expected_manifest_hash=state["manifest_hash"],
    )
    assert manifest.validation_status is ValidationStatus.INCONCLUSIVE
    assert manifest.promotion_status is PromotionStatus.PARKED_FOR_DATA
    with pytest.raises(IntegrityError, match="runner trust store"):
        store.verify_integrity()
    assert store.verify_integrity(runner_trust_store=runner_authority[1])[
        "evidence_bundle_hash"
    ]


def test_passing_evidence_only_requests_approval_and_signed_human_decision_approves(
    tmp_path, monkeypatch, runner_authority
):
    store, manifest, bundle = approve_ready_run(tmp_path, runner_authority)
    assert manifest.promotion_status is PromotionStatus.APPROVAL_REQUIRED
    record, trust_store = signed_record(
        manifest,
        bundle,
        trust_path=tmp_path.parent / f"{tmp_path.name}-approvers.json",
        monkeypatch=monkeypatch,
    )
    approved = store.apply_human_decision(
        record,
        trust_store=trust_store,
        runner_trust_store=runner_authority[1],
        expected_manifest_hash=manifest.content_hash(),
    )
    assert approved.promotion_status is PromotionStatus.APPROVED
    assert store.verify_integrity(
        trust_store=trust_store,
        runner_trust_store={
            runner_authority[1].source_hash: runner_authority[1]
        },
    )["status"] == "pass"


def test_signed_human_supersession_requires_an_explicit_research_child(
    tmp_path, monkeypatch, runner_authority
):
    store, manifest, bundle = approve_ready_run(tmp_path, runner_authority)
    approval, trust_store, private_key = signed_record(
        manifest,
        bundle,
        trust_path=tmp_path.parent / f"{tmp_path.name}-approvers.json",
        monkeypatch=monkeypatch,
        return_private=True,
    )
    approved = store.apply_human_decision(
        approval,
        trust_store=trust_store,
        runner_trust_store=runner_authority[1],
        expected_manifest_hash=manifest.content_hash(),
    )
    child_state = default_research_state(
        "A materially revised Malaysia strategy replaces its approved parent.",
        "Malaysia point-in-time ordinary equities",
        run_id="MYS-RESEARCH-CHILD-001",
        parent_run_id=approved.run_id,
        vault_root=str(tmp_path),
        run_kind="research",
        data_manifest_id="LSEG-MYS-PIT-20260719",
        data_asof_utc=DATA_ASOF,
        feature_manifest_hash="8" * 64,
        code_commit="9" * 40,
        target="next_21d_excess_return",
        label_horizon="21_trading_days",
        split_manifest_id="MYS-PURGED-CV-CHILD-001",
    )
    rotated_payload = json.loads(trust_store.source_path.read_text())
    rotated_payload["signers"][0]["valid_until"] = "2028-01-01T00:00:00+00:00"
    rotated_path = tmp_path.parent / f"{tmp_path.name}-approvers-rotated.json"
    rotated_path.write_text(json.dumps(rotated_payload, sort_keys=True))
    rotated_hash = hashlib.sha256(rotated_path.read_bytes()).hexdigest()
    rotated_path.chmod(0o400)
    monkeypatch.setattr(
        ApprovalTrustStore,
        "_read_admin_anchor",
        classmethod(lambda cls: frozenset({trust_store.source_hash, rotated_hash})),
    )
    rotated_trust_store = ApprovalTrustStore.from_file(rotated_path)
    unsigned = HumanSupersessionRecord(
        supersession_id="SUP-001",
        run_id=approved.run_id,
        replacement_run_id=child_state["run_id"],
        manifest_identity_hash=approved.identity_hash(),
        evidence_bundle_hash=bundle.content_hash(),
        approved_by=approval.approved_by,
        approved_at=(datetime.now(timezone.utc) + timedelta(seconds=3)).replace(
            microsecond=0
        ).isoformat(),
        limitations_acknowledged=("The replacement must complete its own independent validation.",),
        signer_key_id=approval.signer_key_id,
        signature=base64.b64encode(b"0" * 64).decode("ascii"),
    )
    record = replace(
        unsigned,
        signature=base64.b64encode(private_key.sign(unsigned.signing_bytes())).decode("ascii"),
    )
    superseded = store.supersede(
        record,
        trust_store=rotated_trust_store,
        expected_manifest_hash=approved.content_hash(),
    )
    assert superseded.promotion_status is PromotionStatus.SUPERSEDED
    assert store.verify_integrity(
        trust_store={
            trust_store.source_hash: trust_store,
            rotated_trust_store.source_hash: rotated_trust_store,
        },
        runner_trust_store=runner_authority[1],
    )["status"] == "pass"


def test_supersession_rejects_a_replacement_outside_the_parent_lineage(
    tmp_path, monkeypatch, runner_authority
):
    store, manifest, bundle = approve_ready_run(tmp_path, runner_authority)
    approval, trust_store, private_key = signed_record(
        manifest,
        bundle,
        trust_path=tmp_path.parent / f"{tmp_path.name}-approvers.json",
        monkeypatch=monkeypatch,
        return_private=True,
    )
    approved = store.apply_human_decision(
        approval,
        trust_store=trust_store,
        runner_trust_store=runner_authority[1],
        expected_manifest_hash=manifest.content_hash(),
    )
    unrelated = research_state(tmp_path, "MYS-UNRELATED-001")
    unsigned = HumanSupersessionRecord(
        supersession_id="SUP-WRONG-LINEAGE-001",
        run_id=approved.run_id,
        replacement_run_id=unrelated["run_id"],
        manifest_identity_hash=approved.identity_hash(),
        evidence_bundle_hash=bundle.content_hash(),
        approved_by=approval.approved_by,
        approved_at=(datetime.now(timezone.utc) + timedelta(seconds=3)).replace(
            microsecond=0
        ).isoformat(),
        limitations_acknowledged=("This replacement lineage is intentionally invalid.",),
        signer_key_id=approval.signer_key_id,
        signature=base64.b64encode(b"0" * 64).decode("ascii"),
    )
    record = replace(
        unsigned,
        signature=base64.b64encode(private_key.sign(unsigned.signing_bytes())).decode("ascii"),
    )
    with pytest.raises(ApprovalError, match="explicit research child"):
        store.supersede(
            record,
            trust_store=trust_store,
            expected_manifest_hash=approved.content_hash(),
        )


def test_trust_store_inside_agent_vault_cannot_authorize(
    tmp_path, monkeypatch, runner_authority
):
    store, manifest, bundle = approve_ready_run(tmp_path, runner_authority)
    record, trust_store = signed_record(
        manifest,
        bundle,
        trust_path=store.run_root / "agent-created-approvers.json",
        monkeypatch=monkeypatch,
    )
    with pytest.raises(ApprovalError, match="outside"):
        store.apply_human_decision(
            record,
            trust_store=trust_store,
            runner_trust_store=runner_authority[1],
            expected_manifest_hash=manifest.content_hash(),
        )


def test_human_approval_is_disabled_without_root_owned_anchor(tmp_path):
    trust_path = tmp_path / "approvers.json"
    trust_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "signers": [
                    {
                        "approved_by": "Research Approver",
                        "key_id": "BEN-HUMAN-KEY-001",
                        "role": "authorized_human_approver",
                        "public_key_base64": base64.b64encode(b"1" * 32).decode("ascii"),
                        "valid_from": "2026-01-01T00:00:00+00:00",
                        "valid_until": "2027-01-01T00:00:00+00:00",
                        "revoked_at": None,
                    }
                ],
            }
        )
    )
    trust_path.chmod(0o400)

    class UnanchoredTrustStore(ApprovalTrustStore):
        ADMIN_ANCHOR_PATH = tmp_path / "missing-root-anchor.sha256"

    with pytest.raises(ApprovalError, match="remains disabled"):
        UnanchoredTrustStore.from_file(trust_path)


def test_mutated_or_machine_approval_cannot_approve(tmp_path, monkeypatch, runner_authority):
    store, manifest, bundle = approve_ready_run(tmp_path, runner_authority)
    record, trust_store = signed_record(
        manifest,
        bundle,
        trust_path=tmp_path.parent / f"{tmp_path.name}-approvers.json",
        monkeypatch=monkeypatch,
    )
    mutated = replace(record, decision=ApprovalDecision.REJECTED)
    with pytest.raises(ApprovalError, match="invalid human approval signature"):
        store.apply_human_decision(
            mutated,
            trust_store=trust_store,
            runner_trust_store=runner_authority[1],
            expected_manifest_hash=manifest.content_hash(),
        )
    with pytest.raises(ValueError, match="machine identities"):
        replace(record, approved_by="agent-validator").validate()


def test_approval_is_bound_to_exact_run_and_bundle(tmp_path, monkeypatch, runner_authority):
    store, manifest, bundle = approve_ready_run(
        tmp_path / "first", runner_authority, "MYS-RESEARCH-001"
    )
    record, trust_store = signed_record(
        manifest,
        bundle,
        trust_path=tmp_path / "approvers.json",
        monkeypatch=monkeypatch,
    )
    other_store, other_manifest, _ = approve_ready_run(
        tmp_path / "second", runner_authority, "MYS-RESEARCH-002"
    )
    with pytest.raises(ApprovalError, match="another run"):
        other_store.apply_human_decision(
            record,
            trust_store=trust_store,
            runner_trust_store=runner_authority[1],
            expected_manifest_hash=other_manifest.content_hash(),
        )
    assert store.load().promotion_status is PromotionStatus.APPROVAL_REQUIRED


def test_failed_gate_rejects_instead_of_requesting_approval(tmp_path, runner_authority):
    store, state = completed_run(tmp_path, runner_authority=runner_authority)
    paths = write_gate_artifacts(store, runner_authority, status=GateStatus.PASS)
    failed_gate = REQUIRED_VALIDATION_GATES[4]
    failed = ValidationArtifact(
        gate_id=failed_gate,
        run_id=state["run_id"],
        candidate_id=CANDIDATE_ID,
        candidate_hash=CANDIDATE_HASH,
        data_manifest_id="LSEG-MYS-PIT-20260719",
        data_asof_utc=DATA_ASOF,
        feature_manifest_hash=FEATURE_HASH,
        code_commit=CODE_COMMIT,
        split_manifest_id="MYS-PURGED-CV-001",
        status=GateStatus.FAIL,
        producer_execution_id="PY-VALIDATOR-FAIL",
        calculator_id=calculator_id_for(failed_gate),
        calculator_version="1-0-0",
        calculator_source_sha256=CALCULATOR_SOURCE_HASH,
        input_hashes={
            "candidate": CANDIDATE_HASH,
            "data_manifest": DATA_MANIFEST_HASH,
            "feature_manifest": FEATURE_HASH,
            "split_manifest": SPLIT_MANIFEST_HASH,
            "calculator_source": CALCULATOR_SOURCE_HASH,
        },
        runner_id=RUNNER_ID,
        runner_key_id=RUNNER_KEY_ID,
        runner_signature=base64.b64encode(b"0" * 64).decode("ascii"),
        metrics={**passing_metrics(failed_gate), "adjusted_p_value": 0.9},
        generated_at=(datetime.now(timezone.utc) + timedelta(seconds=1)).replace(
            microsecond=0
        ).isoformat(),
    )
    failed = signed_validation_artifact(failed, runner_authority[0])
    paths[failed_gate] = store.write_validation_artifact(
        failed,
        runner_trust_store=runner_authority[1],
    )
    manifest, _ = store.admit_validation(
        candidate_id=CANDIDATE_ID,
        artifact_paths=paths,
        runner_trust_store=runner_authority[1],
        expected_manifest_hash=state["manifest_hash"],
    )
    assert manifest.validation_status is ValidationStatus.FAIL
    assert manifest.promotion_status is PromotionStatus.REJECTED
    assert store.verify_integrity(runner_trust_store=runner_authority[1])[
        "status"
    ] == "pass"

    # A coherent rewrite of the unsigned local projection/event must not be
    # able to contradict the statuses derived from signed gate artifacts.
    event_path = sorted((store.run_root / ".control" / "events").glob("*.json"))[-1]
    event_payload = json.loads(event_path.read_text())
    assert event_payload["event_type"] == "validation_admitted"
    old_after_hash = event_payload["after_manifest_hash"]
    history_path = next(
        (store.run_root / ".control" / "manifests").glob(f"*-{old_after_hash}.json")
    )
    rewritten_manifest = json.loads(history_path.read_text())
    rewritten_manifest["validation_status"] = ValidationStatus.PASS.value
    rewritten_manifest["promotion_status"] = PromotionStatus.APPROVAL_REQUIRED.value
    manifest_content = json.dumps(
        rewritten_manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    rewritten_hash = hashlib.sha256(manifest_content).hexdigest()
    rewritten_history_path = history_path.with_name(
        f"{history_path.name.split('-', 1)[0]}-{rewritten_hash}.json"
    )
    history_path.unlink()
    rewritten_history_path.write_bytes(manifest_content)
    store.manifest_path.write_bytes(manifest_content)
    event_payload["after_manifest_hash"] = rewritten_hash
    event_payload["payload"]["validation_status"] = ValidationStatus.PASS.value
    event_payload["payload"]["promotion_status"] = PromotionStatus.APPROVAL_REQUIRED.value
    event_content = json.dumps(
        event_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    rewritten_event_path = event_path.with_name(
        f"{event_path.name.split('-', 1)[0]}-{hashlib.sha256(event_content).hexdigest()}.json"
    )
    event_path.unlink()
    rewritten_event_path.write_bytes(event_content)
    with pytest.raises(IntegrityError, match="does not match the signed evidence bundle"):
        store.verify_integrity(runner_trust_store=runner_authority[1])


def test_missing_lseg_data_has_explicit_parked_state(tmp_path):
    state = research_state(tmp_path)
    store = ExperimentRunStore(tmp_path, state["run_id"])
    parked = store.park_for_data(
        reason_code="LSEG_ENTITLEMENT_UNAVAILABLE",
        expected_manifest_hash=state["manifest_hash"],
    )
    assert parked.validation_status is ValidationStatus.INCONCLUSIVE
    assert parked.promotion_status is PromotionStatus.PARKED_FOR_DATA
