from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest

from quant_lab.contracts import SignalFrame
from quant_lab.research.contracts import (
    AgentTask,
    AlphaSignal,
    CandidateSpec,
    EvidenceReference,
    HypothesisSpec,
    ModelCallRecord,
    ResearchContract,
    SpecialistReview,
    alpha_signals_from_frame,
)


HASH = "a" * 64
PROVENANCE = {"data": HASH, "code": "b" * 64, "config": "c" * 64}


def hypothesis() -> HypothesisSpec:
    return HypothesisSpec(
        hypothesis_id="MYS-H-001",
        title="Lagged residual momentum",
        claim="Lagged residual momentum predicts five-day Malaysia returns after costs.",
        universe="malaysia_u3_primary",
        features=("residual_momentum_63d", "volatility_21d"),
        horizon="5d",
        units={"residual_momentum_63d": "decimal_return", "volatility_21d": "annualized_decimal"},
        information_cutoff="decision_date_minus_1_session",
        data_availability=("OHLCV known after Bursa close", "features lagged one session"),
        assumptions=("corporate actions reconstructed point in time", "U3 is fixed as primary"),
        falsifiers=("non-positive median purged OOS IR after 20 bps one-way cost",),
        provenance_hashes=PROVENANCE,
    )


def test_every_contract_round_trips_without_hash_drift():
    evidence = EvidenceReference("artifact://x", HASH, "backtest", "2026-07-06")
    contracts = [
        hypothesis(),
        AlphaSignal(asset="1155.KL", as_of="2026-07-06", value=0.25, horizon="5d", units="score", model_name="model", provenance_hashes=PROVENANCE),
        CandidateSpec(candidate_id="C1", hypothesis_id="MYS-H-001", family="momentum", implementation_ref="module:Model", parameters={"lookback": 63}, evidence=(evidence,), provenance_hashes=PROVENANCE),
        AgentTask(task_id="T1", role="reviewer", objective="Find counterexamples", permitted_inputs=("research_dossier",), prohibited_inputs=("forward_holdout",), output_contract="specialist_review", privacy_class="local_only", max_tokens=2048, timeout_seconds=120, provenance_hashes=PROVENANCE),
        SpecialistReview(review_id="R1", task_id="T1", specialist_id="selection_bias_validation_specialist", verdict="conditional", claims=("claim",), assumptions=("dependent trials",), counterexamples=("null can reproduce rank",), falsification_tests=("run CSCV",), veto=True, confidence=0.8, evidence=(evidence,), provenance_hashes=PROVENANCE),
        ModelCallRecord(call_id="CALL1", provider="ollama", model="deepseek-r1:70b", endpoint="http://127.0.0.1:11434", started_at="start", completed_at="end", status="ok", prompt_hash=HASH, schema_hash=HASH, config_hash=HASH, response_hash=HASH, input_tokens=10, output_tokens=20, stop_reason="stop", redactions=("raw_data",)),
    ]
    for item in contracts:
        encoded = item.canonical_json()
        decoded = ResearchContract.from_json(encoded)
        assert decoded == item
        assert decoded.content_hash() == item.content_hash()


@pytest.mark.parametrize(
    "mutator",
    [
        lambda item: replace(item, schema_version="2.0"),
        lambda item: replace(item, provenance_hashes={}),
        lambda item: replace(item, falsifiers=()),
        lambda item: replace(item, data_availability=()),
    ],
)
def test_hypothesis_rejects_unknown_version_missing_provenance_or_pit_falsifiers(mutator):
    with pytest.raises(ValueError):
        mutator(hypothesis()).canonical_json()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.01, 1.01])
def test_signal_rejects_nonfinite_and_out_of_range_values(value):
    with pytest.raises(ValueError):
        AlphaSignal(asset="A", as_of="date", value=value, horizon="1d", units="score", model_name="model", provenance_hashes=PROVENANCE).canonical_json()


def test_rejects_unknown_fields_and_external_provider_endpoint():
    payload = hypothesis().to_dict()
    payload["surprise"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        ResearchContract.from_dict(payload)
    with pytest.raises(ValueError, match="approved routes"):
        replace(
            ModelCallRecord(call_id="C", provider="openrouter", model="m", endpoint="https://openrouter.ai", started_at="s", completed_at="e", status="ok", prompt_hash=HASH, schema_hash=HASH, config_hash=HASH, response_hash=HASH, input_tokens=0, output_tokens=0, stop_reason="stop"),
        ).canonical_json()


def test_existing_signal_frame_adapter_preserves_legacy_object():
    index = pd.to_datetime(["2026-07-06"])
    frame = SignalFrame(pd.DataFrame({"1155.KL": [0.25], "1023.KL": [-0.5]}, index=index), "legacy", "5d", {"family": "test"})
    adapted = alpha_signals_from_frame(frame, as_of=index[0], provenance_hashes=PROVENANCE)
    assert [item.value for item in adapted] == [0.25, -0.5]
    assert frame.model_name == "legacy"


def test_canonical_json_is_key_order_independent():
    left = CandidateSpec(candidate_id="C", hypothesis_id="H", family="family", implementation_ref="ref", parameters={"b": 2, "a": 1}, evidence=(), provenance_hashes=PROVENANCE)
    right = CandidateSpec(candidate_id="C", hypothesis_id="H", family="family", implementation_ref="ref", parameters={"a": 1, "b": 2}, evidence=(), provenance_hashes=dict(reversed(list(PROVENANCE.items()))))
    assert left.canonical_json() == right.canonical_json()
    assert left.content_hash() == right.content_hash()
