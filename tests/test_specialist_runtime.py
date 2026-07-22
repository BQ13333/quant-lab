from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

import pytest

from quant_lab.agents.phd_math_specialists import load_specialist_panel
from quant_lab.agents.specialist_runtime import (
    SELECTION_BIAS_ROLE,
    MalaysiaReviewDossier,
    SpecialistReviewRuntime,
)
from quant_lab.research.contracts import ModelCallRecord, SpecialistReview, SpecialistReviewDraft
from quant_lab.research.providers import (
    LocalModelResult,
    LocalModelRouter,
    MalformedModelOutput,
    ModelTimedOut,
)
from quant_lab.research.trial_ledger import TrialLedger


HASH = "a" * 64


def dossier(**overrides: Any) -> MalaysiaReviewDossier:
    values = {
        "candidate_id": "MYS-C-REVIEW-001",
        "hypothesis_hash": "b" * 64,
        "candidate_hash": "c" * 64,
        "problem": "Audit a Malaysia residual-momentum strategy for dependence, selection bias, and robust portfolio construction.",
        "tags": ("time_series", "selection_bias", "robust_portfolio"),
        "evidence_hashes": {"research_panel": "d" * 64},
        "research_context": {"horizon": "5d", "cost_bps_one_way": 20, "universe": "malaysia_u3_primary"},
    }
    values.update(overrides)
    return MalaysiaReviewDossier(**values)


class ScriptedReviewProvider:
    """Deterministic local fake whose response is derived only from the blinded task."""

    def __init__(
        self,
        *,
        veto_role: str | None = None,
        malformed_role: str | None = None,
        conditional_role: str | None = None,
    ) -> None:
        self.veto_role = veto_role
        self.malformed_role = malformed_role
        self.conditional_role = conditional_role
        self.requests = []

    def generate(self, request, *, model: str):
        self.requests.append((request, model))
        role = request.payload["specialist_id"]
        if role == self.malformed_role:
            raise MalformedModelOutput("scripted invalid JSON")
        veto = role == self.veto_role
        review = SpecialistReviewDraft(
            verdict="fail" if veto else "conditional" if role == self.conditional_role else "pass",
            claims=("The research claim has a falsifiable purged OOS test.",),
            assumptions=("All signals are lagged one Bursa session.",),
            counterexamples=("A shuffled-label CSCV replay reproduces the selected rank.",) if veto else (),
            falsification_tests=("Run CSCV with the full lifetime trial denominator.",),
            veto=veto,
            confidence=0.8,
        )
        now = datetime.now(timezone.utc).isoformat()
        digest = lambda value: hashlib.sha256(value.encode()).hexdigest()
        record = ModelCallRecord(
            call_id=request.request_id,
            provider="ollama",
            model=model,
            endpoint="http://127.0.0.1:11434",
            started_at=now,
            completed_at=now,
            status="fake",
            prompt_hash=digest(request.system_prompt + request.prompt_payload_json()),
            schema_hash=digest("specialist_review_v1"),
            config_hash=digest(json.dumps(dict(request.config), sort_keys=True)),
            response_hash=digest(review.canonical_json()),
            input_tokens=0,
            output_tokens=0,
            stop_reason="deterministic_fake",
            redactions=("prompt_body", "response_body", "candidate_identity"),
        )
        return LocalModelResult(review, record, request.fingerprint(model))


def reserve(ledger: TrialLedger) -> str:
    return ledger.reserve_trial(
        idempotency_key="malaysia-specialist-review-001",
        campaign="malaysia",
        family="specialist_review",
        hypothesis_id="MYS-H-001",
        candidate_id="MYS-C-REVIEW-001",
        code_hash=HASH,
        data_hash=HASH,
        config_hash=HASH,
        prompt_hash=HASH,
        model_hash=HASH,
    ).trial_id


def test_progression_panel_always_has_selection_bias_and_two_distinct_relevant_roles():
    runtime = SpecialistReviewRuntime(LocalModelRouter(ScriptedReviewProvider()))
    selected = runtime.assign(dossier())
    assert selected[0].id == SELECTION_BIAS_ROLE
    assert len(selected) == len({item.id for item in selected}) == 3
    result = runtime.review_candidate(dossier())
    assert result.progression_allowed
    assert result.selected_specialists == tuple(item.id for item in selected)


def test_all_twenty_roles_execute_the_same_typed_schema_through_local_fake():
    panel = load_specialist_panel()
    provider = ScriptedReviewProvider()
    runtime = SpecialistReviewRuntime(LocalModelRouter(provider), panel=panel)
    result = runtime.review_roles(dossier(), panel.specialists)
    assert len(result.attempts) == 20
    assert {item.specialist_id for item in result.attempts} == {item.id for item in panel.specialists}
    assert all(isinstance(item.review, SpecialistReview) for item in result.attempts)
    assert all(item.model_call.provider == "ollama" for item in result.attempts)


def test_blinded_requests_have_no_holdout_author_provider_or_broker_fields():
    provider = ScriptedReviewProvider()
    SpecialistReviewRuntime(LocalModelRouter(provider)).review_candidate(dossier())
    for request, _ in provider.requests:
        body = json.dumps(request.payload, sort_keys=True).lower()
        assert all(
            blocked not in key.lower()
            for key in request.payload
            for blocked in ("holdout", "author", "provider", "broker")
        )
        assert "holdout_score" not in body and "holdout_return" not in body
        assert "candidate_author" not in body and "model_provider" not in body
        assert "broker_token" not in body and "broker_capability" not in body
        assert request.payload["market"] == "Malaysia"
        assert request.payload["exchange"] == "Bursa Malaysia"
        assert "blind_candidate_id" in request.payload
        assert request.payload["specialist_mandate"]
        assert request.payload["specialist_lane"]
        assert request.payload["routing_tags"]
        assert request.payload["proof_obligations"]
        assert request.payload["kill_criteria"]
        assert request.payload["corpus_anchors"]
        assert dossier().candidate_id not in body
        assert request.expected_contract is SpecialistReviewDraft
        assert all(
            "plain strings" in request.payload["field_contract"][field]
            for field in ("claims", "assumptions", "counterexamples", "falsification_tests")
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"market": "Japan", "exchange": "JPX"},
        {"problem": "Compare Malaysia with Japan"},
        {"research_context": {"terminal_holdout_sharpe": 3.0}},
        {"research_context": {"candidate_author": "team-a"}},
        {"research_context": {"broker_token": "secret"}},
    ],
)
def test_non_malaysia_or_unblinded_dossier_fails_closed(overrides):
    with pytest.raises(ValueError):
        dossier(**overrides).validate()


def test_concrete_counterexample_veto_defeats_two_passing_reviews():
    base = SpecialistReviewRuntime(LocalModelRouter(ScriptedReviewProvider()))
    veto_role = base.assign(dossier())[1].id
    result = SpecialistReviewRuntime(
        LocalModelRouter(ScriptedReviewProvider(veto_role=veto_role))
    ).review_candidate(dossier())
    assert [item.review.verdict for item in result.attempts].count("pass") == 2
    assert result.vetoed
    assert not result.progression_allowed
    assert "concrete_counterexample_veto" in result.reasons
    assert "verdict_disagreement" in result.conflicts


def test_conditional_review_allows_backtest_but_never_full_progression():
    base = SpecialistReviewRuntime(LocalModelRouter(ScriptedReviewProvider()))
    conditional_role = base.assign(dossier())[1].id
    result = SpecialistReviewRuntime(
        LocalModelRouter(ScriptedReviewProvider(conditional_role=conditional_role))
    ).review_candidate(dossier())
    assert result.backtest_allowed
    assert not result.progression_allowed
    assert "conditional_review" in result.reasons


def test_malformed_review_is_a_recorded_failure_and_never_approves(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    trial_id = reserve(ledger)
    baseline = SpecialistReviewRuntime(LocalModelRouter(ScriptedReviewProvider()))
    malformed = baseline.assign(dossier())[2].id
    runtime = SpecialistReviewRuntime(
        LocalModelRouter(ScriptedReviewProvider(malformed_role=malformed)), ledger=ledger
    )
    result = runtime.review_candidate(dossier(), trial_id=trial_id)
    assert not result.progression_allowed
    assert "failed_review_attempt" in result.reasons
    manifest = ledger.replay_manifest(trial_id)
    review_events = [item for item in manifest["events"] if item["event_type"] == "specialist_review"]
    assert len(review_events) == 3
    assert sum(item["status"] == "failure" for item in review_events) == 1
    assert manifest["events"][-1]["event_type"] == "panel_disposition"


def test_review_lineage_is_persisted_without_prompt_or_response_bodies(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    trial_id = reserve(ledger)
    result = SpecialistReviewRuntime(
        LocalModelRouter(ScriptedReviewProvider()), ledger=ledger
    ).review_candidate(dossier(), trial_id=trial_id)
    manifest = ledger.replay_manifest(trial_id)
    review_events = [item for item in manifest["events"] if item["event_type"] == "specialist_review"]
    assert len(review_events) == 3 and result.progression_allowed
    for event in review_events:
        bundle = ledger.verify_evidence(event["evidence_sha256"]).path.read_text()
        assert "model_call" in bundle and "prompt_hash" in bundle
        assert "prompt_body" in bundle and "response_body" in bundle
        assert dossier().problem not in bundle


def test_controller_not_model_owns_review_identity_evidence_and_provenance():
    provider = ScriptedReviewProvider()
    base = dossier()
    result = SpecialistReviewRuntime(LocalModelRouter(provider)).review_candidate(base)
    for attempt in result.attempts:
        assert attempt.review is not None
        assert attempt.review.review_id.startswith("MYS-REVIEW-")
        assert attempt.review.task_id == attempt.task_id
        assert attempt.review.specialist_id == attempt.specialist_id
        assert attempt.review.evidence == ()
        assert attempt.review.provenance_hashes == {"candidate": base.candidate_hash}


def test_assignment_and_disposition_hash_are_replay_deterministic():
    first = SpecialistReviewRuntime(LocalModelRouter(ScriptedReviewProvider())).review_candidate(dossier())
    second = SpecialistReviewRuntime(LocalModelRouter(ScriptedReviewProvider())).review_candidate(dossier())
    assert first.selected_specialists == second.selected_specialists
    assert first.progression_allowed == second.progression_allowed
    assert first.disposition_hash == second.disposition_hash


def test_partial_panel_resume_replays_recorded_role_without_duplicate_call(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    trial_id = reserve(ledger)
    panel = load_specialist_panel()
    first_provider = ScriptedReviewProvider()
    SpecialistReviewRuntime(LocalModelRouter(first_provider), ledger=ledger).review_roles(
        dossier(), [panel.specialists[0]], trial_id=trial_id
    )
    assert len(first_provider.requests) == 1

    resumed_provider = ScriptedReviewProvider()
    resumed = SpecialistReviewRuntime(LocalModelRouter(resumed_provider), ledger=ledger).review_roles(
        dossier(), [panel.specialists[0]], trial_id=trial_id
    )
    assert resumed_provider.requests == []
    assert resumed.attempts[0].review is not None
    assert resumed.attempts[0].model_call.status == "replay"
    assert resumed.attempts[0].model_call.cost_usd == 0.0
    manifest = ledger.replay_manifest(trial_id)
    assert sum(item["event_type"] == "specialist_review" for item in manifest["events"]) == 1


def test_qforge_infrastructure_failure_is_retried_while_successful_roles_replay(tmp_path):
    class TimedOutProvider:
        def generate(self, request, *, model: str):
            raise ModelTimedOut("temporary cloud timeout")

    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    trial_id = reserve(ledger)
    panel = load_specialist_panel()
    role = panel.specialists[0]
    first = SpecialistReviewRuntime(
        LocalModelRouter(TimedOutProvider()),
        ledger=ledger,
        retry_infrastructure_failures=True,
    ).review_roles(dossier(), [role], trial_id=trial_id)
    assert first.attempts[0].failure_code == "MODEL_TIMED_OUT"
    provider = ScriptedReviewProvider()
    resumed = SpecialistReviewRuntime(
        LocalModelRouter(provider),
        ledger=ledger,
        retry_infrastructure_failures=True,
    ).review_roles(dossier(), [role], trial_id=trial_id)
    assert len(provider.requests) == 1
    assert resumed.attempts[0].review is not None
    events = [
        item
        for item in ledger.replay_manifest(trial_id)["events"]
        if item["event_type"] == "specialist_review"
    ]
    assert [item["status"] for item in events] == ["failure", "pass"]


def test_malformed_provider_receipt_is_persisted_without_response_body(tmp_path):
    class ReceiptedMalformedProvider:
        def generate(self, request, *, model: str):
            now = datetime.now(timezone.utc).isoformat()
            record = ModelCallRecord(
                call_id=request.request_id,
                provider="deepseek",
                model="deepseek-v4-pro",
                endpoint="https://api.deepseek.com",
                started_at=now,
                completed_at=now,
                status="malformed_output",
                prompt_hash="1" * 64,
                schema_hash="2" * 64,
                config_hash="3" * 64,
                response_hash="4" * 64,
                input_tokens=100,
                output_tokens=200,
                cost_usd=0.002,
                stop_reason="length",
                redactions=("api_key", "prompt_body", "response_body", "reasoning_content"),
            )
            raise MalformedModelOutput("invalid contract", call_record=record)

    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    trial_id = reserve(ledger)
    role = load_specialist_panel().specialists[0]
    result = SpecialistReviewRuntime(
        LocalModelRouter(ReceiptedMalformedProvider()), ledger=ledger
    ).review_roles(dossier(), [role], trial_id=trial_id)
    assert result.attempts[0].model_call.cost_usd == 0.002
    event = [
        item
        for item in ledger.replay_manifest(trial_id)["events"]
        if item["event_type"] == "specialist_review"
    ][0]
    assert event["failure_class"] == "MALFORMED_MODEL_OUTPUT"
    receipt = ledger.verify_evidence(event["evidence_sha256"]).path.read_text()
    assert "malformed_output" in receipt and "response_body" in receipt
    assert "invalid contract" not in receipt
