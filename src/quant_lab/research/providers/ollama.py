from __future__ import annotations

import hashlib
import json
from dataclasses import fields
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol, get_args, get_origin, get_type_hints
from urllib.parse import urlparse

import requests

from quant_lab.research.contracts import ModelCallRecord, ResearchContract
from quant_lab.research.providers.base import (
    LocalModelRequest,
    LocalModelResult,
    MalformedModelOutput,
    ModelRateLimited,
    ModelTimedOut,
    ModelUnavailable,
    PrivacyViolation,
    RetryExhausted,
)


class JSONTransport(Protocol):
    def post_json(self, url: str, payload: Mapping[str, Any], *, timeout: int) -> Mapping[str, Any]:
        ...


class RequestsTransport:
    def post_json(self, url: str, payload: Mapping[str, Any], *, timeout: int) -> Mapping[str, Any]:
        try:
            response = requests.post(url, json=payload, timeout=timeout)
        except requests.Timeout as exc:
            raise ModelTimedOut("local Ollama request timed out") from exc
        except requests.ConnectionError as exc:
            raise ModelUnavailable("local Ollama is unavailable") from exc
        if response.status_code == 429:
            raise ModelRateLimited("local Ollama rate limited the request")
        if response.status_code == 404:
            raise ModelUnavailable("requested local Ollama model is unavailable")
        if not response.ok:
            raise ModelUnavailable(f"local Ollama returned HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise MalformedModelOutput("local Ollama returned invalid JSON") from exc
        if not isinstance(body, Mapping):
            raise MalformedModelOutput("local Ollama response must be an object")
        return body


def _json_type(annotation: Any) -> dict[str, Any]:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is tuple:
        return {"type": "array", "items": _json_type(args[0] if args else Any)}
    if origin in (dict, Mapping) or "Mapping" in str(origin):
        return {"type": "object"}
    if origin is not None and type(None) in args:
        candidates = [_json_type(item) for item in args if item is not type(None)]
        return {"anyOf": [*candidates, {"type": "null"}]}
    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}
    if isinstance(annotation, type) and hasattr(annotation, "__dataclass_fields__"):
        return contract_json_schema(annotation, include_contract_type=False)
    return {}


def contract_json_schema(
    contract: type[Any], *, include_contract_type: bool = True
) -> dict[str, Any]:
    hints = get_type_hints(contract)
    properties = {item.name: _json_type(hints[item.name]) for item in fields(contract)}
    required = [item.name for item in fields(contract)]
    if include_contract_type:
        properties = {
            "contract_type": {"type": "string", "const": contract.contract_type},
            **properties,
        }
        required = ["contract_type", *required]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class OllamaProvider:
    """Structured local Ollama client; no credentials or hosted fallback exist."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:11434",
        transport: JSONTransport | None = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise PrivacyViolation("Ollama provider requires a loopback HTTP endpoint")
        if (
            parsed.port not in {None, 11434}
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise PrivacyViolation("Ollama provider endpoint is not approved")
        self.endpoint = endpoint.rstrip("/")
        self.transport = transport or RequestsTransport()

    def generate(self, request: LocalModelRequest, *, model: str) -> LocalModelResult:
        request.validate()
        schema = contract_json_schema(request.expected_contract)
        prompt_payload = request.prompt_payload_json()
        fingerprint = request.fingerprint(model)
        allowed_keys = ", ".join(schema["properties"])
        schema_guard = (
            "\nThe response object must contain exactly these top-level keys and no others: "
            f"{allowed_keys}. Do not add notes, status, risks, assumptions, or explanatory fields "
            "unless one of those exact keys explicitly requests them. "
            f"Set contract_type to exactly {request.expected_contract.contract_type!r} and "
            "schema_version to exactly '1.0'."
        )
        effective_system_prompt = request.system_prompt + schema_guard
        payload = {
            "model": model,
            "stream": False,
            "think": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": effective_system_prompt},
                {"role": "user", "content": prompt_payload},
            ],
            "options": {
                "temperature": request.temperature,
                "seed": request.seed,
                "num_predict": request.max_tokens,
                "num_ctx": 8192,
            },
        }
        started_at = datetime.now(timezone.utc).isoformat()
        last_error: Exception | None = None
        body: Mapping[str, Any] | None = None
        for _ in range(request.max_attempts):
            try:
                body = self.transport.post_json(
                    f"{self.endpoint}/api/chat", payload, timeout=request.timeout_seconds
                )
                break
            except (ModelTimedOut, ModelRateLimited) as exc:
                last_error = exc
        if body is None:
            if request.max_attempts == 1 and last_error is not None:
                raise last_error
            raise RetryExhausted(
                f"local Ollama exhausted {request.max_attempts} bounded attempts"
            ) from last_error
        message = body.get("message")
        content = message.get("content") if isinstance(message, Mapping) else None
        if not isinstance(content, str) or not content.strip():
            raise MalformedModelOutput("local Ollama returned no structured content")
        try:
            contract = request.expected_contract.from_json(content)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise MalformedModelOutput("local Ollama content failed the expected contract") from exc
        completed_at = datetime.now(timezone.utc).isoformat()
        schema_text = json.dumps(schema, sort_keys=True, separators=(",", ":"))
        config_text = json.dumps(
            {
                "capability": request.capability.value,
                "max_tokens": request.max_tokens,
                "timeout_seconds": request.timeout_seconds,
                "max_attempts": request.max_attempts,
                "temperature": request.temperature,
                "seed": request.seed,
                "config": dict(request.config),
            },
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        record = ModelCallRecord(
            call_id=request.request_id,
            provider="ollama",
            model=model,
            endpoint=self.endpoint,
            started_at=started_at,
            completed_at=completed_at,
            status="success",
            prompt_hash=_hash(effective_system_prompt + prompt_payload),
            schema_hash=_hash(schema_text),
            config_hash=_hash(config_text),
            response_hash=_hash(content),
            input_tokens=int(body.get("prompt_eval_count", 0) or 0),
            output_tokens=int(body.get("eval_count", 0) or 0),
            stop_reason=str(body.get("done_reason", "stop")),
            redactions=("prompt_body", "response_body", "environment_secrets"),
        )
        record.validate()
        return LocalModelResult(contract, record, fingerprint)
