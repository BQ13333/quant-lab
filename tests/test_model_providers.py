from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from quant_lab.research.contracts import HypothesisSpec
from quant_lab.research.providers import (
    BudgetExceeded,
    LocalModelRequest,
    LocalModelRouter,
    LocalRoutingPolicy,
    MalformedModelOutput,
    ModelCapability,
    ModelRateLimited,
    ModelTimedOut,
    ModelUnavailable,
    OllamaProvider,
    PrivacyViolation,
    ReplayProvider,
    RetryExhausted,
)
from quant_lab.research.providers.replay import FakeLocalProvider


HASHES = {"data": "a" * 64, "code": "b" * 64, "config": "c" * 64}


def hypothesis_json() -> str:
    return HypothesisSpec(
        hypothesis_id="MYS-H-LOCAL-001",
        title="Malaysia residual momentum",
        claim="Lagged residual momentum predicts five-day Bursa Malaysia returns after costs.",
        universe="malaysia_u3_primary",
        features=("residual_momentum_63d", "volatility_21d"),
        horizon="5d",
        units={
            "residual_momentum_63d": "decimal_return",
            "volatility_21d": "annualized_decimal",
        },
        information_cutoff="decision_date_minus_1_bursa_session",
        data_availability=("Bursa OHLCV available after close", "all features lagged one session"),
        assumptions=("corporate actions are point-in-time", "universe is Malaysia only"),
        falsifiers=("purged net OOS IR is non-positive at 20 bps one-way cost",),
        provenance_hashes=HASHES,
    ).canonical_json()


def request(**overrides: Any) -> LocalModelRequest:
    values: dict[str, Any] = {
        "request_id": "CALL-MYS-001",
        "capability": ModelCapability.HYPOTHESIS_EXTRACTION,
        "expected_contract": HypothesisSpec,
        "system_prompt": "Return exact local-only Malaysia research JSON.",
        "payload": {"market": "Malaysia", "exchange": "Bursa Malaysia"},
        "max_tokens": 512,
        "max_prompt_chars": 4096,
        "timeout_seconds": 30,
        "max_attempts": 1,
        "temperature": 0.0,
        "seed": 7,
    }
    values.update(overrides)
    return LocalModelRequest(**values)


class CaptureTransport:
    def __init__(self, response: Mapping[str, Any] | None = None, error: Exception | None = None) -> None:
        self.response = response or {
            "message": {"content": hypothesis_json()},
            "prompt_eval_count": 19,
            "eval_count": 41,
            "done_reason": "stop",
        }
        self.error = error
        self.calls: list[tuple[str, Mapping[str, Any], int]] = []

    def post_json(self, url: str, payload: Mapping[str, Any], *, timeout: int) -> Mapping[str, Any]:
        self.calls.append((url, payload, timeout))
        if self.error is not None:
            raise self.error
        return self.response


def test_routing_policy_assigns_all_roles_to_exact_local_models():
    provider = FakeLocalProvider({capability.value: hypothesis_json() for capability in ModelCapability})
    router = LocalModelRouter(provider)
    expected = {
        ModelCapability.HYPOTHESIS_EXTRACTION: "qwen3.5:35b-mlx",
        ModelCapability.ROUTINE_REVIEW: "qwen3.5:35b-mlx",
        ModelCapability.ADVERSARIAL_MATH: "deepseek-r1:70b",
        ModelCapability.CODE_IMPLEMENTATION: "qwen3-coder:30b",
    }
    assert router.policy.models == expected
    for capability, model in expected.items():
        result = router.route(request(capability=capability, request_id=f"CALL-{capability.value}"))
        assert result.call_record.model == model
        assert isinstance(result.contract, HypothesisSpec)


def test_ollama_success_is_structured_bounded_and_redacted():
    transport = CaptureTransport()
    result = OllamaProvider(transport=transport).generate(request(), model="qwen3.5:35b-mlx")
    assert result.contract.universe == "malaysia_u3_primary"
    assert len(transport.calls) == 1
    url, payload, timeout = transport.calls[0]
    assert url == "http://127.0.0.1:11434/api/chat"
    assert timeout == 30
    assert payload["stream"] is False
    assert payload["think"] is False
    assert payload["format"]["additionalProperties"] is False
    assert "exactly these top-level keys and no others" in payload["messages"][0]["content"]
    assert payload["options"] == {
        "temperature": 0.0,
        "seed": 7,
        "num_predict": 512,
        "num_ctx": 8192,
    }
    assert result.call_record.input_tokens == 19
    assert result.call_record.output_tokens == 41
    receipt = result.call_record.canonical_json()
    assert "Return exact local-only" not in receipt
    assert "Bursa Malaysia" not in receipt
    assert hypothesis_json() not in receipt
    assert all(len(value) == 64 for value in (
        result.call_record.prompt_hash,
        result.call_record.schema_hash,
        result.call_record.config_hash,
        result.call_record.response_hash,
    ))


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://127.0.0.1:11434",
        "http://ollama.example:11434",
        "http://localhost.evil:11434",
        "http://127.0.0.1:9999",
        "http://user:pass@127.0.0.1:11434",
        "http://127.0.0.1:11434?redirect=external",
        "http://127.0.0.1:11434/#external",
    ],
)
def test_external_or_ambiguous_endpoints_fail_closed(endpoint: str):
    with pytest.raises(PrivacyViolation):
        LocalRoutingPolicy(endpoint=endpoint).validate()
    with pytest.raises(PrivacyViolation):
        OllamaProvider(endpoint=endpoint, transport=CaptureTransport())


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens": 8193},
        {"timeout_seconds": 901},
        {"max_attempts": 4},
        {"temperature": 1.1},
        {"max_prompt_chars": 5},
    ],
)
def test_request_budgets_fail_before_transport(overrides: Mapping[str, Any]):
    transport = CaptureTransport()
    with pytest.raises(BudgetExceeded):
        OllamaProvider(transport=transport).generate(request(**overrides), model="qwen3.5:35b-mlx")
    assert transport.calls == []


def test_malformed_model_content_has_typed_failure():
    transport = CaptureTransport(response={"message": {"content": "{not-json"}})
    with pytest.raises(MalformedModelOutput):
        OllamaProvider(transport=transport).generate(request(), model="qwen3.5:35b-mlx")


@pytest.mark.parametrize("error", [ModelTimedOut("timeout"), ModelRateLimited("limited")])
def test_single_attempt_transient_failure_preserves_typed_error(error: Exception):
    with pytest.raises(type(error)):
        OllamaProvider(transport=CaptureTransport(error=error)).generate(
            request(max_attempts=1), model="qwen3.5:35b-mlx"
        )


def test_retry_is_bounded_and_exhaustion_is_typed():
    transport = CaptureTransport(error=ModelTimedOut("timeout"))
    with pytest.raises(RetryExhausted):
        OllamaProvider(transport=transport).generate(
            request(max_attempts=2), model="qwen3.5:35b-mlx"
        )
    assert len(transport.calls) == 2


def test_unavailable_model_is_not_retried():
    transport = CaptureTransport(error=ModelUnavailable("missing"))
    with pytest.raises(ModelUnavailable):
        OllamaProvider(transport=transport).generate(request(), model="qwen3.5:35b-mlx")
    assert len(transport.calls) == 1


def test_offline_replay_is_content_addressed_and_deterministic():
    model = "qwen3.5:35b-mlx"
    item = request()
    fingerprint = item.fingerprint(model)
    provider = ReplayProvider({fingerprint: hypothesis_json()})
    first = provider.generate(item, model=model)
    second = provider.generate(item, model=model)
    assert first.request_fingerprint == second.request_fingerprint == fingerprint
    assert first.contract.content_hash() == second.contract.content_hash()
    assert first.call_record.status == "replay"


def test_local_routing_config_has_no_hosted_fallback_or_provider_key():
    config = json.loads(Path("configs/local_model_routing.json").read_text())
    assert config["endpoint"] == "http://127.0.0.1:11434"
    assert config["privacy_class"] == "local_only"
    assert config["hosted_fallback"] is False
    assert set(config["models"].values()) == {
        "qwen3.5:35b-mlx",
        "deepseek-r1:70b",
        "qwen3-coder:30b",
    }
    combined = "\n".join(
        path.read_text().lower()
        for path in [
            Path("src/quant_lab/research/providers/ollama.py"),
            Path("src/quant_lab/research/providers/router.py"),
            Path("src/quant_lab/research/research_extractor.py"),
        ]
    )
    for forbidden in ("openrouter", "api_key", "os.environ", "quant_lab.execution", "ibkr"):
        assert forbidden not in combined
