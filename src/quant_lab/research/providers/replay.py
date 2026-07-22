from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Mapping

from quant_lab.research.contracts import ModelCallRecord, ResearchContract
from quant_lab.research.providers.base import (
    LocalModelRequest,
    LocalModelResult,
    MalformedModelOutput,
    ModelUnavailable,
)


def _record(request: LocalModelRequest, model: str, content: str, *, status: str) -> ModelCallRecord:
    now = datetime.now(timezone.utc).isoformat()
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    return ModelCallRecord(
        call_id=request.request_id,
        provider="ollama",
        model=model,
        endpoint="http://127.0.0.1:11434",
        started_at=now,
        completed_at=now,
        status=status,
        prompt_hash=digest(request.system_prompt + request.prompt_payload_json()),
        schema_hash=digest(request.expected_contract.contract_type),
        config_hash=digest(str(sorted(request.config.items()))),
        response_hash=digest(content),
        input_tokens=0,
        output_tokens=0,
        stop_reason="deterministic_replay",
        redactions=("prompt_body", "response_body"),
    )


class FakeLocalProvider:
    """Deterministic provider used by tests; never opens a network connection."""

    def __init__(self, responses: Mapping[str, str]) -> None:
        self.responses = dict(responses)

    def generate(self, request: LocalModelRequest, *, model: str) -> LocalModelResult:
        request.validate()
        content = self.responses.get(request.capability.value)
        if content is None:
            raise ModelUnavailable(f"no fake response for {request.capability.value}")
        try:
            contract = request.expected_contract.from_json(content)
        except Exception as exc:
            raise MalformedModelOutput("fake response failed the expected contract") from exc
        record = _record(request, model, content, status="fake")
        return LocalModelResult(contract, record, request.fingerprint(model))


class ReplayProvider:
    """Content-addressed offline replay keyed by request fingerprint."""

    def __init__(self, records: Mapping[str, str]) -> None:
        self.records = dict(records)

    def generate(self, request: LocalModelRequest, *, model: str) -> LocalModelResult:
        request.validate()
        fingerprint = request.fingerprint(model)
        content = self.records.get(fingerprint)
        if content is None:
            raise ModelUnavailable(f"no replay record for {fingerprint}")
        try:
            contract: ResearchContract = request.expected_contract.from_json(content)
        except Exception as exc:
            raise MalformedModelOutput("replay record failed the expected contract") from exc
        return LocalModelResult(contract, _record(request, model, content, status="replay"), fingerprint)
