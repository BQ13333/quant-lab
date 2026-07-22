from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import urlparse

from quant_lab.research.providers.base import (
    LocalModelRequest,
    LocalModelResult,
    ModelCapability,
    ModelProvider,
    PrivacyViolation,
)


@dataclass(frozen=True)
class LocalRoutingPolicy:
    endpoint: str = "http://127.0.0.1:11434"
    models: dict[ModelCapability, str] = field(
        default_factory=lambda: {
            ModelCapability.HYPOTHESIS_EXTRACTION: "qwen3.5:35b-mlx",
            ModelCapability.ROUTINE_REVIEW: "qwen3.5:35b-mlx",
            ModelCapability.ADVERSARIAL_MATH: "deepseek-r1:70b",
            ModelCapability.CODE_IMPLEMENTATION: "qwen3-coder:30b",
        }
    )

    def validate(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise PrivacyViolation("Ollama endpoint must use HTTP on a loopback host")
        if parsed.port not in {None, 11434} or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise PrivacyViolation("Ollama endpoint contains a forbidden authority, port, or suffix")
        missing = set(ModelCapability) - set(self.models)
        if missing:
            raise ValueError(f"routing policy is missing capabilities: {sorted(item.value for item in missing)}")
        approved = {"qwen3.5:35b-mlx", "deepseek-r1:70b", "qwen3-coder:30b"}
        if set(self.models.values()) - approved:
            raise PrivacyViolation("routing policy contains an unapproved model")


class LocalModelRouter:
    """Capability router with a hard loopback and approved-model allowlist."""

    def __init__(self, provider: ModelProvider, policy: LocalRoutingPolicy | None = None) -> None:
        self.provider = provider
        self.policy = policy or LocalRoutingPolicy()
        self.policy.validate()

    def route(self, request: LocalModelRequest) -> LocalModelResult:
        request.validate()
        model = self.policy.models[request.capability]
        return self.provider.generate(request, model=model)
