from __future__ import annotations

import json

import pytest

from quant_lab.research.contracts import HypothesisSpec
from quant_lab.research.providers import FakeLocalProvider, LocalModelRouter, ModelCapability
from quant_lab.research.research_extractor import LOCAL_RESEARCH_SYSTEM_PROMPT, ResearchExtractor
from quant_lab.research.trial_ledger import TrialLedger


HASH = "a" * 64


def hypothesis_json() -> str:
    return HypothesisSpec(
        hypothesis_id="MYS-H-EXTRACT-001",
        title="Malaysia low-volatility reversal",
        claim="Lagged low-volatility reversal predicts five-day Bursa Malaysia returns after costs.",
        universe="malaysia_u3_primary",
        features=("return_5d", "volatility_21d"),
        horizon="5d",
        units={"return_5d": "decimal_return", "volatility_21d": "annualized_decimal"},
        information_cutoff="decision_date_minus_1_bursa_session",
        data_availability=("Bursa OHLCV known after close", "signals lagged one session"),
        assumptions=("Malaysia universe is survivorship controlled",),
        falsifiers=("net purged OOS IR is non-positive",),
        provenance_hashes={"data": HASH, "code": "b" * 64, "config": "c" * 64},
    ).canonical_json()


def reserve(ledger: TrialLedger) -> str:
    return ledger.reserve_trial(
        idempotency_key="malaysia-extractor-001",
        campaign="malaysia",
        family="local_research_extraction",
        hypothesis_id="MYS-H-EXTRACT-001",
        candidate_id="MYS-C-EXTRACT-001",
        code_hash=HASH,
        data_hash=HASH,
        config_hash=HASH,
        prompt_hash=HASH,
        model_hash=HASH,
    ).trial_id


def test_extractor_routes_local_contract_and_records_only_redacted_receipt(tmp_path):
    raw_private_marker = "PRIVATE-MALAYSIA-ROW-DO-NOT-LOG"
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    trial_id = reserve(ledger)
    provider = FakeLocalProvider({ModelCapability.HYPOTHESIS_EXTRACTION.value: hypothesis_json()})
    extractor = ResearchExtractor(LocalModelRouter(provider), ledger)
    receipt = extractor.extract(
        request_id="CALL-MYS-EXTRACT-001",
        capability=ModelCapability.HYPOTHESIS_EXTRACTION,
        expected_contract=HypothesisSpec,
        payload={"market": "Malaysia", "exchange": "Bursa Malaysia", "private_marker": raw_private_marker},
        trial_id=trial_id,
        max_tokens=512,
        timeout_seconds=30,
    )
    assert receipt.result.contract.universe == "malaysia_u3_primary"
    assert receipt.ledger_event_id is not None
    manifest = ledger.replay_manifest(trial_id)
    event = manifest["events"][-1]
    assert event["event_type"] == "model_call"
    assert event["status"] == "fake"
    event_text = json.dumps(event)
    assert raw_private_marker not in event_text
    evidence = ledger.verify_evidence(event["evidence_sha256"])
    evidence_text = evidence.path.read_text()
    assert raw_private_marker not in evidence_text
    assert "Malaysia low-volatility reversal" not in evidence_text
    assert "prompt_hash" in evidence_text and "response_hash" in evidence_text


def test_ledger_extraction_requires_reserved_trial_identity(tmp_path):
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    provider = FakeLocalProvider({ModelCapability.HYPOTHESIS_EXTRACTION.value: hypothesis_json()})
    extractor = ResearchExtractor(LocalModelRouter(provider), ledger)
    with pytest.raises(ValueError, match="trial_id"):
        extractor.extract(
            request_id="CALL-MYS-NO-TRIAL",
            capability=ModelCapability.HYPOTHESIS_EXTRACTION,
            expected_contract=HypothesisSpec,
            payload={"market": "Malaysia"},
        )


def test_strong_system_prompt_declares_the_nonnegotiable_research_boundaries():
    prompt = LOCAL_RESEARCH_SYSTEM_PROMPT.lower()
    for phrase in (
        "local malaysia",
        "exactly one json object",
        "terminal holdout evidence is forbidden",
        "never recommend or attempt live orders",
        "raw sharpe is not proof",
        "selection bias",
        "counterexamples",
        "rather than inventing",
    ):
        assert phrase in prompt
