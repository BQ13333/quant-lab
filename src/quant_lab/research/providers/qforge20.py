from __future__ import annotations

from dataclasses import dataclass, field

from quant_lab.research.providers.base import CloudModelRequest, LocalModelRequest, LocalModelResult, ModelCapability
from quant_lab.research.providers.cloud_sanitizer import CloudDossierSanitizer
from quant_lab.research.providers.deepseek import DEEPSEEK_MODEL, DeepSeekProvider
from quant_lab.research.providers.ollama import OllamaProvider
from quant_lab.research.providers.router import LocalModelRouter, LocalRoutingPolicy


@dataclass(frozen=True)
class QForge20RoutingPolicy:
    workflow_name: str = "QForge-20"
    local_policy: LocalRoutingPolicy = field(default_factory=LocalRoutingPolicy)
    cloud_capabilities: frozenset[ModelCapability] = frozenset(
        {ModelCapability.ROUTINE_REVIEW, ModelCapability.ADVERSARIAL_MATH}
    )
    per_call_cost_usd: float = 0.10
    reasoning_effort: str = "high"

    @property
    def models(self) -> dict[ModelCapability, str]:
        return {
            capability: DEEPSEEK_MODEL if capability in self.cloud_capabilities else model
            for capability, model in self.local_policy.models.items()
        }

    def validate(self) -> None:
        self.local_policy.validate()
        if self.workflow_name != "QForge-20":
            raise ValueError("QForge-20 workflow identity cannot be renamed")
        if not self.cloud_capabilities or not self.cloud_capabilities.issubset(set(ModelCapability)):
            raise ValueError("QForge-20 cloud capability set is invalid")
        if self.reasoning_effort not in {"high", "max"}:
            raise ValueError("QForge-20 reasoning effort must be high or max")
        if not 0.0 < self.per_call_cost_usd <= 5.0:
            raise ValueError("QForge-20 per-call cost must be within (0, 5]")


class QForge20ModelRouter:
    """Hybrid router: local Qwen coordination and sanitized V4-Pro research."""

    def __init__(
        self,
        *,
        local_provider: OllamaProvider | None = None,
        deepseek_provider: DeepSeekProvider | None = None,
        policy: QForge20RoutingPolicy | None = None,
        sanitizer: CloudDossierSanitizer | None = None,
    ) -> None:
        self.policy = policy or QForge20RoutingPolicy()
        self.policy.validate()
        self.local_router = LocalModelRouter(
            local_provider or OllamaProvider(),
            policy=self.policy.local_policy,
        )
        self.deepseek_provider = deepseek_provider or DeepSeekProvider()
        self.sanitizer = sanitizer or CloudDossierSanitizer()

    def route(self, request: LocalModelRequest) -> LocalModelResult:
        request.validate()
        if request.capability not in self.policy.cloud_capabilities:
            return self.local_router.route(request)
        sanitized = self.sanitizer.sanitize(
            system_prompt=request.system_prompt,
            payload=request.payload,
            max_chars=request.max_prompt_chars,
        )
        cloud_request = CloudModelRequest(
            request_id=request.request_id,
            capability=request.capability,
            expected_contract=request.expected_contract,
            system_prompt=sanitized.system_prompt,
            payload=sanitized.payload,
            max_tokens=max(4096, request.max_tokens),
            max_prompt_chars=request.max_prompt_chars,
            timeout_seconds=request.timeout_seconds,
            max_attempts=request.max_attempts,
            temperature=request.temperature,
            privacy_class="cloud_sanitized",
            reasoning_effort=self.policy.reasoning_effort,
            max_cost_usd=self.policy.per_call_cost_usd,
            config={
                **dict(request.config),
                "workflow_name": self.policy.workflow_name,
                "cloud_boundary": "sanitized_direct_deepseek",
            },
        )
        return self.deepseek_provider.generate(cloud_request, model=DEEPSEEK_MODEL)
