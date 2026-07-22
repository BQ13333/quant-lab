from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

from quant_lab.research.contracts import ModelCallRecord, ResearchContract


class ModelCapability(str, Enum):
    HYPOTHESIS_EXTRACTION = "hypothesis_extraction"
    ROUTINE_REVIEW = "routine_review"
    ADVERSARIAL_MATH = "adversarial_math"
    CODE_IMPLEMENTATION = "code_implementation"


class LocalProviderError(RuntimeError):
    code = "LOCAL_PROVIDER_ERROR"

    def __init__(
        self,
        message: str,
        *,
        call_record: ModelCallRecord | None = None,
    ) -> None:
        super().__init__(message)
        self.call_record = call_record


class PrivacyViolation(LocalProviderError):
    code = "PRIVACY_VIOLATION"


class BudgetExceeded(LocalProviderError):
    code = "BUDGET_EXCEEDED"


class MalformedModelOutput(LocalProviderError):
    code = "MALFORMED_MODEL_OUTPUT"


class ModelTimedOut(LocalProviderError):
    code = "MODEL_TIMED_OUT"


class ModelRateLimited(LocalProviderError):
    code = "MODEL_RATE_LIMITED"


class ModelUnavailable(LocalProviderError):
    code = "MODEL_UNAVAILABLE"


class RetryExhausted(LocalProviderError):
    code = "RETRY_EXHAUSTED"


class ModelAuthenticationFailed(LocalProviderError):
    code = "MODEL_AUTHENTICATION_FAILED"


class ModelQuotaExceeded(LocalProviderError):
    code = "MODEL_QUOTA_EXCEEDED"


@dataclass(frozen=True)
class LocalModelRequest:
    request_id: str
    capability: ModelCapability
    expected_contract: type[ResearchContract]
    system_prompt: str
    payload: Mapping[str, Any]
    max_tokens: int = 2048
    max_prompt_chars: int = 32_000
    timeout_seconds: int = 180
    max_attempts: int = 2
    temperature: float = 0.0
    seed: int = 42
    privacy_class: str = "local_only"
    config: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id.strip() or not self.system_prompt.strip():
            raise ValueError("request identity and system prompt are required")
        if self.privacy_class != "local_only":
            raise PrivacyViolation("only local_only requests are permitted")
        if self.max_tokens <= 0 or self.max_tokens > 8192:
            raise BudgetExceeded("max_tokens must be within 1..8192")
        if self.max_prompt_chars <= 0 or self.max_prompt_chars > 128_000:
            raise BudgetExceeded("max_prompt_chars must be within 1..128000")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 900:
            raise BudgetExceeded("timeout_seconds must be within 1..900")
        if self.max_attempts <= 0 or self.max_attempts > 3:
            raise BudgetExceeded("max_attempts must be within 1..3")
        if not 0.0 <= self.temperature <= 1.0:
            raise BudgetExceeded("temperature must be within [0, 1]")
        serialized = self.prompt_payload_json()
        if len(self.system_prompt) + len(serialized) > self.max_prompt_chars:
            raise BudgetExceeded("prompt exceeds max_prompt_chars")

    def prompt_payload_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def fingerprint(self, model: str) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8"))
        digest.update(self.capability.value.encode("utf-8"))
        digest.update(self.system_prompt.encode("utf-8"))
        digest.update(self.prompt_payload_json().encode("utf-8"))
        digest.update(self.expected_contract.contract_type.encode("utf-8"))
        digest.update(json.dumps(dict(self.config), sort_keys=True, default=str).encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True)
class CloudModelRequest:
    """A separately typed request that may cross the QForge-20 cloud boundary."""

    request_id: str
    capability: ModelCapability
    expected_contract: type[ResearchContract]
    system_prompt: str
    payload: Mapping[str, Any]
    max_tokens: int = 2048
    max_prompt_chars: int = 32_000
    timeout_seconds: int = 180
    max_attempts: int = 2
    temperature: float = 0.0
    privacy_class: str = "cloud_sanitized"
    reasoning_effort: str = "high"
    max_cost_usd: float = 0.10
    config: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.request_id.strip() or not self.system_prompt.strip():
            raise ValueError("request identity and system prompt are required")
        if self.privacy_class != "cloud_sanitized":
            raise PrivacyViolation("DeepSeek requests require cloud_sanitized classification")
        if self.max_tokens <= 0 or self.max_tokens > 8192:
            raise BudgetExceeded("max_tokens must be within 1..8192")
        if self.max_prompt_chars <= 0 or self.max_prompt_chars > 128_000:
            raise BudgetExceeded("max_prompt_chars must be within 1..128000")
        if self.timeout_seconds <= 0 or self.timeout_seconds > 900:
            raise BudgetExceeded("timeout_seconds must be within 1..900")
        if self.max_attempts <= 0 or self.max_attempts > 3:
            raise BudgetExceeded("max_attempts must be within 1..3")
        if not 0.0 <= self.temperature <= 1.0:
            raise BudgetExceeded("temperature must be within [0, 1]")
        if self.reasoning_effort not in {"high", "max"}:
            raise BudgetExceeded("QForge-20 reasoning_effort must be high or max")
        if not 0.0 < self.max_cost_usd <= 5.0:
            raise BudgetExceeded("per-call cloud cost ceiling must be within (0, 5]")
        serialized = self.prompt_payload_json()
        if len(self.system_prompt) + len(serialized) > self.max_prompt_chars:
            raise BudgetExceeded("prompt exceeds max_prompt_chars")

    def prompt_payload_json(self) -> str:
        return json.dumps(self.payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    def fingerprint(self, model: str) -> str:
        digest = hashlib.sha256()
        digest.update(model.encode("utf-8"))
        digest.update(self.capability.value.encode("utf-8"))
        digest.update(self.system_prompt.encode("utf-8"))
        digest.update(self.prompt_payload_json().encode("utf-8"))
        digest.update(self.expected_contract.contract_type.encode("utf-8"))
        digest.update(self.privacy_class.encode("utf-8"))
        digest.update(self.reasoning_effort.encode("utf-8"))
        digest.update(json.dumps(dict(self.config), sort_keys=True, default=str).encode("utf-8"))
        return digest.hexdigest()


@dataclass(frozen=True)
class LocalModelResult:
    contract: ResearchContract
    call_record: ModelCallRecord
    request_fingerprint: str


class ModelProvider(Protocol):
    def generate(self, request: LocalModelRequest, *, model: str) -> LocalModelResult:
        ...
