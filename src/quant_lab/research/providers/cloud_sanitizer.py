from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from quant_lab.research.providers.base import PrivacyViolation


_FORBIDDEN_KEY_PARTS = (
    "account",
    "api_key",
    "authorization",
    "broker",
    "credential",
    "holdout",
    "live_order",
    "position",
    "raw_data",
    "secret",
    "terminal",
    "target_weights",
    "reference_prices",
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class SanitizedCloudDossier:
    system_prompt: str
    payload: Mapping[str, Any]


class CloudDossierSanitizer:
    """Fail-closed sanitizer for the QForge-20 direct DeepSeek boundary."""

    def sanitize(
        self,
        *,
        system_prompt: str,
        payload: Mapping[str, Any],
        max_chars: int,
    ) -> SanitizedCloudDossier:
        if not system_prompt.strip():
            raise PrivacyViolation("cloud system prompt is empty")
        safe = self._walk(payload, path="payload")
        serialized = json.dumps(safe, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        combined = system_prompt + serialized
        if len(combined) > max_chars:
            raise PrivacyViolation("sanitized cloud dossier exceeds its character ceiling")
        self._reject_secret_text(combined)
        return SanitizedCloudDossier(system_prompt=system_prompt, payload=safe)

    def _walk(self, value: Any, *, path: str) -> Any:
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for key, item in value.items():
                name = str(key)
                normalized = name.lower()
                if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                    raise PrivacyViolation(f"restricted cloud dossier field: {path}.{name}")
                result[name] = self._walk(item, path=f"{path}.{name}")
            return result
        if isinstance(value, (tuple, list)):
            return [self._walk(item, path=path) for item in value]
        if isinstance(value, str):
            if len(value) > 2048:
                raise PrivacyViolation(f"cloud dossier text is too long at {path}")
            self._reject_secret_text(value)
            return value
        if isinstance(value, (int, float, bool)) or value is None:
            return value
        raise PrivacyViolation(f"unsupported cloud dossier value at {path}: {type(value).__name__}")

    @staticmethod
    def _reject_secret_text(value: str) -> None:
        if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
            raise PrivacyViolation("credential-like material is forbidden in a cloud dossier")
