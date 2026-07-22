from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol
from urllib.parse import urlparse

import requests

from quant_lab.research.contracts import ModelCallRecord
from quant_lab.research.providers.base import (
    BudgetExceeded,
    CloudModelRequest,
    LocalModelResult,
    MalformedModelOutput,
    ModelAuthenticationFailed,
    ModelQuotaExceeded,
    ModelRateLimited,
    ModelTimedOut,
    ModelUnavailable,
    PrivacyViolation,
    RetryExhausted,
)
from quant_lab.research.providers.ollama import contract_json_schema


DEEPSEEK_ENDPOINT = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"
INPUT_CACHE_HIT_USD_PER_MILLION = 0.003625
INPUT_CACHE_MISS_USD_PER_MILLION = 0.435
OUTPUT_USD_PER_MILLION = 0.87
KEYCHAIN_SERVICE = "qforge20-deepseek"


class DeepSeekJSONTransport(Protocol):
    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout: int,
    ) -> Mapping[str, Any]:
        ...


class RequestsDeepSeekTransport:
    def post_json(
        self,
        url: str,
        payload: Mapping[str, Any],
        *,
        headers: Mapping[str, str],
        timeout: int,
    ) -> Mapping[str, Any]:
        try:
            response = requests.post(url, json=payload, headers=dict(headers), timeout=timeout)
        except requests.Timeout as exc:
            raise ModelTimedOut("DeepSeek request timed out") from exc
        except requests.ConnectionError as exc:
            raise ModelUnavailable("DeepSeek API is unavailable") from exc
        if response.status_code in {401, 403}:
            raise ModelAuthenticationFailed("DeepSeek authentication failed")
        if response.status_code in {402, 422}:
            raise ModelQuotaExceeded("DeepSeek quota or balance is unavailable")
        if response.status_code == 429:
            raise ModelRateLimited("DeepSeek rate limited the request")
        if not response.ok:
            raise ModelUnavailable(f"DeepSeek returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise MalformedModelOutput("DeepSeek returned invalid JSON") from exc
        if not isinstance(body, Mapping):
            raise MalformedModelOutput("DeepSeek response must be an object")
        return body


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _unwrap_single_json_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    opener = lines[0].strip().lower()
    if opener not in {"```", "```json"}:
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _usage_cost(body: Mapping[str, Any]) -> tuple[int, int, float]:
    usage = body.get("usage") if isinstance(body.get("usage"), Mapping) else {}
    prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    cache_hit = min(prompt_tokens, int(usage.get("prompt_cache_hit_tokens", 0) or 0))
    cache_miss = int(usage.get("prompt_cache_miss_tokens", prompt_tokens - cache_hit) or 0)
    cache_miss = max(0, min(prompt_tokens, cache_miss))
    if cache_hit + cache_miss < prompt_tokens:
        cache_miss += prompt_tokens - cache_hit - cache_miss
    cost = (
        cache_hit * INPUT_CACHE_HIT_USD_PER_MILLION
        + cache_miss * INPUT_CACHE_MISS_USD_PER_MILLION
        + output_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000.0
    return prompt_tokens, output_tokens, cost


def _call_record(
    *,
    request: CloudModelRequest,
    schema: Mapping[str, Any],
    effective_system_prompt: str,
    prompt_payload: str,
    started_at: str,
    body: Mapping[str, Any],
    first: Mapping[str, Any] | None,
    status: str,
    response_material: str,
) -> ModelCallRecord:
    prompt_tokens, output_tokens, cost = _usage_cost(body)
    completed_at = datetime.now(timezone.utc).isoformat()
    config_text = json.dumps(
        {
            "capability": request.capability.value,
            "max_tokens": request.max_tokens,
            "timeout_seconds": request.timeout_seconds,
            "max_attempts": request.max_attempts,
            "temperature": request.temperature,
            "privacy_class": request.privacy_class,
            "reasoning_effort": request.reasoning_effort,
            "max_cost_usd": request.max_cost_usd,
            "config": dict(request.config),
        },
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    record = ModelCallRecord(
        call_id=request.request_id,
        provider="deepseek",
        model=DEEPSEEK_MODEL,
        endpoint=DEEPSEEK_ENDPOINT,
        started_at=started_at,
        completed_at=completed_at,
        status=status,
        prompt_hash=_hash(effective_system_prompt + prompt_payload),
        schema_hash=_hash(json.dumps(schema, sort_keys=True, separators=(",", ":"))),
        config_hash=_hash(config_text),
        response_hash=_hash(response_material),
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        cost_usd=cost,
        stop_reason=str(first.get("finish_reason", "unknown")) if first else "unknown",
        redactions=(
            "api_key",
            "prompt_body",
            "response_body",
            "reasoning_content",
            "environment_secrets",
        ),
    )
    record.validate()
    return record


def load_deepseek_api_key() -> str | None:
    """Read the key from process environment or the operator's macOS Keychain."""

    environment = os.getenv("DEEPSEEK_API_KEY")
    if environment and environment.strip():
        return environment.strip()
    account = os.getenv("USER")
    if sys.platform != "darwin" or not account:
        return None
    result = subprocess.run(
        [
            "security",
            "find-generic-password",
            "-a",
            account,
            "-s",
            KEYCHAIN_SERVICE,
            "-w",
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


class DeepSeekProvider:
    """Direct, bounded DeepSeek V4-Pro provider for sanitized QForge-20 work."""

    def __init__(
        self,
        *,
        endpoint: str = DEEPSEEK_ENDPOINT,
        transport: DeepSeekJSONTransport | None = None,
        key_loader: Callable[[], str | None] | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if (
            endpoint.rstrip("/") != DEEPSEEK_ENDPOINT
            or parsed.scheme != "https"
            or parsed.hostname != "api.deepseek.com"
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise PrivacyViolation("DeepSeek provider requires the exact approved HTTPS endpoint")
        self.endpoint = endpoint.rstrip("/")
        self.transport = transport or RequestsDeepSeekTransport()
        self.key_loader = key_loader or load_deepseek_api_key

    def generate(self, request: CloudModelRequest, *, model: str) -> LocalModelResult:
        request.validate()
        if model != DEEPSEEK_MODEL:
            raise PrivacyViolation("QForge-20 permits only deepseek-v4-pro on the cloud route")
        worst_case_cost = (
            request.max_prompt_chars * INPUT_CACHE_MISS_USD_PER_MILLION
            + request.max_tokens * OUTPUT_USD_PER_MILLION
        ) / 1_000_000.0
        if worst_case_cost > request.max_cost_usd:
            raise BudgetExceeded("DeepSeek request exceeds its preflight cost ceiling")
        key = self.key_loader()
        if not isinstance(key, str) or not key.strip():
            raise ModelAuthenticationFailed("DEEPSEEK_API_KEY is not provisioned")
        schema = contract_json_schema(request.expected_contract)
        prompt_payload = request.prompt_payload_json()
        schema_guard = (
            "\nReturn one JSON object and no prose. The object must match this exact JSON Schema: "
            + json.dumps(schema, sort_keys=True, separators=(",", ":"))
        )
        effective_system_prompt = request.system_prompt + schema_guard
        api_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": prompt_payload},
            ],
            "stream": False,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled"},
            "reasoning_effort": request.reasoning_effort,
        }
        headers = {
            "Authorization": f"Bearer {key.strip()}",
            "Content-Type": "application/json",
        }
        fingerprint = request.fingerprint(model)
        started_at = datetime.now(timezone.utc).isoformat()
        last_error: Exception | None = None
        body: Mapping[str, Any] | None = None
        for _ in range(request.max_attempts):
            try:
                body = self.transport.post_json(
                    f"{self.endpoint}/chat/completions",
                    api_payload,
                    headers=headers,
                    timeout=request.timeout_seconds,
                )
                break
            except (ModelTimedOut, ModelRateLimited) as exc:
                last_error = exc
        if body is None:
            if request.max_attempts == 1 and last_error is not None:
                raise last_error
            raise RetryExhausted(
                f"DeepSeek exhausted {request.max_attempts} bounded attempts"
            ) from last_error
        choices = body.get("choices")
        first = choices[0] if isinstance(choices, list) and choices else None
        message = first.get("message") if isinstance(first, Mapping) else None
        content = message.get("content") if isinstance(message, Mapping) else None
        safe_diagnostic = json.dumps(
            {
                "content_chars": len(content) if isinstance(content, str) else 0,
                "finish_reason": first.get("finish_reason") if isinstance(first, Mapping) else None,
                "reasoning_chars": (
                    len(message.get("reasoning_content"))
                    if isinstance(message, Mapping)
                    and isinstance(message.get("reasoning_content"), str)
                    else 0
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        if _usage_cost(body)[2] > request.max_cost_usd:
            raise BudgetExceeded(
                "DeepSeek response exceeded its per-call cost ceiling",
                call_record=_call_record(
                    request=request,
                    schema=schema,
                    effective_system_prompt=effective_system_prompt,
                    prompt_payload=prompt_payload,
                    started_at=started_at,
                    body=body,
                    first=first if isinstance(first, Mapping) else None,
                    status="budget_exceeded",
                    response_material=content if isinstance(content, str) else safe_diagnostic,
                ),
            )
        if not isinstance(content, str) or not content.strip():
            reasoning = message.get("reasoning_content") if isinstance(message, Mapping) else None
            reasoning_chars = len(reasoning) if isinstance(reasoning, str) else 0
            finish_reason = first.get("finish_reason") if isinstance(first, Mapping) else None
            raise MalformedModelOutput(
                "DeepSeek returned no structured final content; "
                f"finish_reason={finish_reason}; reasoning_chars={reasoning_chars}",
                call_record=_call_record(
                    request=request,
                    schema=schema,
                    effective_system_prompt=effective_system_prompt,
                    prompt_payload=prompt_payload,
                    started_at=started_at,
                    body=body,
                    first=first if isinstance(first, Mapping) else None,
                    status="malformed_output",
                    response_material=safe_diagnostic,
                ),
            )
        normalized_content = _unwrap_single_json_fence(content)
        try:
            decoded = json.loads(normalized_content)
        except json.JSONDecodeError as exc:
            raise MalformedModelOutput(
                "DeepSeek final content was not one JSON object",
                call_record=_call_record(
                    request=request,
                    schema=schema,
                    effective_system_prompt=effective_system_prompt,
                    prompt_payload=prompt_payload,
                    started_at=started_at,
                    body=body,
                    first=first if isinstance(first, Mapping) else None,
                    status="malformed_output",
                    response_material=normalized_content,
                ),
            ) from exc
        if not isinstance(decoded, Mapping):
            raise MalformedModelOutput(
                "DeepSeek final content was not a JSON object",
                call_record=_call_record(
                    request=request,
                    schema=schema,
                    effective_system_prompt=effective_system_prompt,
                    prompt_payload=prompt_payload,
                    started_at=started_at,
                    body=body,
                    first=first if isinstance(first, Mapping) else None,
                    status="malformed_output",
                    response_material=normalized_content,
                ),
            )
        expected_keys = set(schema["properties"])
        observed_keys = {str(key) for key in decoded}
        missing = sorted(expected_keys - observed_keys)
        unknown = sorted(observed_keys - expected_keys)
        try:
            contract = request.expected_contract.from_dict(decoded)
        except (TypeError, ValueError) as exc:
            raise MalformedModelOutput(
                "DeepSeek content failed the expected contract; "
                f"missing_keys={missing}; unknown_keys={unknown}; validation={type(exc).__name__}",
                call_record=_call_record(
                    request=request,
                    schema=schema,
                    effective_system_prompt=effective_system_prompt,
                    prompt_payload=prompt_payload,
                    started_at=started_at,
                    body=body,
                    first=first if isinstance(first, Mapping) else None,
                    status="malformed_output",
                    response_material=normalized_content,
                ),
            ) from exc
        record = _call_record(
            request=request,
            schema=schema,
            effective_system_prompt=effective_system_prompt,
            prompt_payload=prompt_payload,
            started_at=started_at,
            body=body,
            first=first if isinstance(first, Mapping) else None,
            status="success",
            response_material=normalized_content,
        )
        return LocalModelResult(contract, record, fingerprint)


class DeepSeekReplayProvider:
    """Content-addressed V4-Pro replay that never reads a key or opens a network."""

    def __init__(self, records: Mapping[str, str]) -> None:
        self.records = dict(records)

    def generate(self, request: CloudModelRequest, *, model: str) -> LocalModelResult:
        request.validate()
        if model != DEEPSEEK_MODEL:
            raise PrivacyViolation("QForge-20 replay permits only deepseek-v4-pro")
        fingerprint = request.fingerprint(model)
        content = self.records.get(fingerprint)
        if content is None:
            raise ModelUnavailable(f"no DeepSeek replay record for {fingerprint}")
        try:
            contract = request.expected_contract.from_json(content)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MalformedModelOutput("DeepSeek replay failed the expected contract") from exc
        now = datetime.now(timezone.utc).isoformat()
        schema = contract_json_schema(request.expected_contract)
        record = ModelCallRecord(
            call_id=request.request_id,
            provider="deepseek",
            model=model,
            endpoint=DEEPSEEK_ENDPOINT,
            started_at=now,
            completed_at=now,
            status="replay",
            prompt_hash=_hash(request.system_prompt + request.prompt_payload_json()),
            schema_hash=_hash(json.dumps(schema, sort_keys=True, separators=(",", ":"))),
            config_hash=_hash(
                json.dumps(
                    {
                        "capability": request.capability.value,
                        "privacy_class": request.privacy_class,
                        "reasoning_effort": request.reasoning_effort,
                        "config": dict(request.config),
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                )
            ),
            response_hash=_hash(content),
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.0,
            stop_reason="deterministic_replay",
            redactions=("api_key", "prompt_body", "response_body", "reasoning_content"),
        )
        record.validate()
        return LocalModelResult(contract, record, fingerprint)
