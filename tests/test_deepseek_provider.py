from __future__ import annotations

from typing import Any, Mapping

import pytest

from quant_lab.research.contracts import HypothesisSpec
from quant_lab.research.providers import (
    BudgetExceeded,
    CloudDossierSanitizer,
    CloudModelRequest,
    DEEPSEEK_MODEL,
    DeepSeekProvider,
    DeepSeekReplayProvider,
    LocalModelRequest,
    MalformedModelOutput,
    ModelAuthenticationFailed,
    ModelCapability,
    ModelRateLimited,
    PrivacyViolation,
    QForge20ModelRouter,
    load_deepseek_api_key,
)
from quant_lab.research.providers.replay import FakeLocalProvider


HASHES = {"data": "a" * 64, "code": "b" * 64, "config": "c" * 64}


def hypothesis_json() -> str:
    return HypothesisSpec(
        hypothesis_id="MYS-H-QFORGE-001",
        title="QForge residual momentum",
        claim="Lagged residual momentum predicts Malaysia returns after costs.",
        universe="malaysia_u3_primary",
        features=("residual_momentum_126d",),
        horizon="5d",
        units={"residual_momentum_126d": "decimal_return"},
        information_cutoff="decision_date_minus_1_bursa_session",
        data_availability=("lagged Bursa OHLCV",),
        assumptions=("point-in-time universe",),
        falsifiers=("non-positive net research IR",),
        provenance_hashes=HASHES,
    ).canonical_json()


def cloud_request(**overrides: Any) -> CloudModelRequest:
    values: dict[str, Any] = {
        "request_id": "QFORGE-CALL-001",
        "capability": ModelCapability.ADVERSARIAL_MATH,
        "expected_contract": HypothesisSpec,
        "system_prompt": "Return the exact QForge-20 research contract as JSON.",
        "payload": {"market": "Malaysia", "candidate_hash": "d" * 64},
        "max_tokens": 512,
        "max_prompt_chars": 4096,
        "timeout_seconds": 30,
        "max_attempts": 1,
        "temperature": 0.0,
        "reasoning_effort": "high",
        "max_cost_usd": 0.01,
    }
    values.update(overrides)
    return CloudModelRequest(**values)


class CaptureDeepSeekTransport:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str], int]] = []

    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout: int,
    ) -> Mapping[str, Any]:
        self.calls.append((url, payload, headers, timeout))
        if self.error is not None:
            raise self.error
        return {
            "choices": [
                {
                    "message": {
                        "content": hypothesis_json(),
                        "reasoning_content": "must never enter the receipt",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "prompt_cache_hit_tokens": 40,
                "prompt_cache_miss_tokens": 60,
            },
        }


def test_deepseek_v4_pro_call_is_structured_costed_and_redacted():
    transport = CaptureDeepSeekTransport()
    provider = DeepSeekProvider(transport=transport, key_loader=lambda: "unit-test-key")
    result = provider.generate(cloud_request(), model=DEEPSEEK_MODEL)
    assert result.contract.hypothesis_id == "MYS-H-QFORGE-001"
    assert len(transport.calls) == 1
    url, payload, headers, timeout = transport.calls[0]
    assert url == "https://api.deepseek.com/chat/completions"
    assert headers["Authorization"] == "Bearer unit-test-key"
    assert timeout == 30
    assert payload["model"] == "deepseek-v4-pro"
    assert payload["thinking"] == {"type": "enabled"}
    assert payload["reasoning_effort"] == "high"
    assert payload["response_format"] == {"type": "json_object"}
    assert result.call_record.provider == "deepseek"
    assert result.call_record.input_tokens == 100
    assert result.call_record.output_tokens == 50
    assert result.call_record.cost_usd == pytest.approx(
        (40 * 0.003625 + 60 * 0.435 + 50 * 0.87) / 1_000_000
    )
    receipt = result.call_record.canonical_json()
    assert "unit-test-key" not in receipt
    assert "must never enter" not in receipt
    assert "reasoning_content" in result.call_record.redactions


def test_deepseek_key_is_required_but_never_part_of_request_fingerprint():
    transport = CaptureDeepSeekTransport()
    provider = DeepSeekProvider(transport=transport, key_loader=lambda: None)
    with pytest.raises(ModelAuthenticationFailed, match="not provisioned"):
        provider.generate(cloud_request(), model=DEEPSEEK_MODEL)
    assert transport.calls == []


def test_environment_key_loader_returns_secret_without_persisting_it(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "environment-unit-test-key")
    assert load_deepseek_api_key() == "environment-unit-test-key"


@pytest.mark.parametrize(
    "payload",
    [
        {"raw_data": [1, 2, 3]},
        {"terminal_holdout": 0.02},
        {"nested": {"broker_account": "x"}},
        {"notes": "credential sk-1234567890abcdefghijkl"},
    ],
)
def test_cloud_sanitizer_rejects_restricted_fields_and_secret_patterns(payload):
    with pytest.raises(PrivacyViolation):
        CloudDossierSanitizer().sanitize(
            system_prompt="QForge policy",
            payload=payload,
            max_chars=4096,
        )


def test_deepseek_route_is_exact_and_cost_fails_before_transport():
    transport = CaptureDeepSeekTransport()
    provider = DeepSeekProvider(transport=transport, key_loader=lambda: "unit-test-key")
    with pytest.raises(PrivacyViolation, match="v4-pro"):
        provider.generate(cloud_request(), model="deepseek-v4-flash")
    with pytest.raises(BudgetExceeded, match="preflight"):
        provider.generate(cloud_request(max_cost_usd=0.000001), model=DEEPSEEK_MODEL)
    assert transport.calls == []


def test_deepseek_transient_failure_remains_typed():
    transport = CaptureDeepSeekTransport(error=ModelRateLimited("limited"))
    provider = DeepSeekProvider(transport=transport, key_loader=lambda: "unit-test-key")
    with pytest.raises(ModelRateLimited):
        provider.generate(cloud_request(), model=DEEPSEEK_MODEL)


def test_qforge_router_keeps_planning_local_and_routes_reviews_to_v4_pro():
    local = FakeLocalProvider(
        {capability.value: hypothesis_json() for capability in ModelCapability}
    )
    transport = CaptureDeepSeekTransport()
    deepseek = DeepSeekProvider(transport=transport, key_loader=lambda: "unit-test-key")
    router = QForge20ModelRouter(local_provider=local, deepseek_provider=deepseek)
    local_request = LocalModelRequest(
        request_id="LOCAL-PLAN",
        capability=ModelCapability.HYPOTHESIS_EXTRACTION,
        expected_contract=HypothesisSpec,
        system_prompt="Plan locally.",
        payload={"market": "Malaysia"},
        max_tokens=512,
        max_prompt_chars=4096,
        timeout_seconds=30,
        max_attempts=1,
    )
    local_result = router.route(local_request)
    assert local_result.call_record.provider == "ollama"
    assert transport.calls == []
    cloud_source = LocalModelRequest(
        request_id="CLOUD-REVIEW",
        capability=ModelCapability.ROUTINE_REVIEW,
        expected_contract=HypothesisSpec,
        system_prompt="Review the sanitized Malaysia dossier.",
        payload={"market": "Malaysia", "candidate_hash": "d" * 64},
        max_tokens=512,
        max_prompt_chars=4096,
        timeout_seconds=30,
        max_attempts=1,
    )
    cloud_result = router.route(cloud_source)
    assert cloud_result.call_record.provider == "deepseek"
    assert cloud_result.call_record.model == "deepseek-v4-pro"
    assert len(transport.calls) == 1


def test_deepseek_replay_is_content_addressed_and_never_needs_a_key():
    item = cloud_request()
    fingerprint = item.fingerprint(DEEPSEEK_MODEL)
    provider = DeepSeekReplayProvider({fingerprint: hypothesis_json()})
    first = provider.generate(item, model=DEEPSEEK_MODEL)
    second = provider.generate(item, model=DEEPSEEK_MODEL)
    assert first.request_fingerprint == second.request_fingerprint == fingerprint
    assert first.contract.content_hash() == second.contract.content_hash()
    assert first.call_record.provider == "deepseek"
    assert first.call_record.status == "replay"
    assert first.call_record.cost_usd == 0.0


def test_deepseek_accepts_one_json_markdown_fence_but_still_validates_contract():
    transport = CaptureDeepSeekTransport()
    original = transport.post_json

    def fenced(*args, **kwargs):
        body = dict(original(*args, **kwargs))
        choice = dict(body["choices"][0])
        message = dict(choice["message"])
        message["content"] = f"```json\n{message['content']}\n```"
        choice["message"] = message
        body["choices"] = [choice]
        return body

    transport.post_json = fenced  # type: ignore[method-assign]
    result = DeepSeekProvider(transport=transport, key_loader=lambda: "unit-test-key").generate(
        cloud_request(), model=DEEPSEEK_MODEL
    )
    assert result.contract.hypothesis_id == "MYS-H-QFORGE-001"


def test_malformed_deepseek_output_carries_a_redacted_usage_receipt():
    transport = CaptureDeepSeekTransport()
    original = transport.post_json

    def malformed(*args, **kwargs):
        body = dict(original(*args, **kwargs))
        choice = dict(body["choices"][0])
        message = dict(choice["message"])
        message["content"] = "not-json-and-must-never-be-stored"
        choice["message"] = message
        body["choices"] = [choice]
        return body

    transport.post_json = malformed  # type: ignore[method-assign]
    with pytest.raises(MalformedModelOutput) as caught:
        DeepSeekProvider(transport=transport, key_loader=lambda: "unit-test-key").generate(
            cloud_request(), model=DEEPSEEK_MODEL
        )
    receipt = caught.value.call_record
    assert receipt is not None
    assert receipt.status == "malformed_output"
    assert receipt.input_tokens == 100 and receipt.output_tokens == 50
    serialized = receipt.canonical_json()
    assert "not-json" not in serialized
    assert "must never enter" not in serialized
