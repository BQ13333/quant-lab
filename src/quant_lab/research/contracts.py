from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import types
from collections.abc import Mapping as ABCMapping
from dataclasses import MISSING, dataclass, field
from datetime import datetime
from typing import Any, ClassVar, Mapping, TypeVar, Union, get_args, get_origin, get_type_hints

import pandas as pd

from quant_lab.contracts import SignalFrame


SCHEMA_VERSION = "1.0"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CONTRACTS: dict[str, type["ResearchContract"]] = {}
T = TypeVar("T", bound="ResearchContract")


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {item.name: _canonical(getattr(value, item.name)) for item in dataclasses.fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _canonical(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("contract numbers must be finite")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported contract value: {type(value).__name__}")


def _validate_hashes(values: Mapping[str, str], *, field_name: str) -> None:
    if not values:
        raise ValueError(f"{field_name} must not be empty")
    for name, value in values.items():
        if not name.strip() or not _SHA256.fullmatch(value):
            raise ValueError(f"{field_name} must contain named lowercase SHA-256 hashes")


def _coerce(annotation: Any, value: Any) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for candidate in args:
            if candidate is type(None):
                continue
            try:
                return _coerce(candidate, value)
            except (TypeError, ValueError):
                continue
        raise TypeError(f"value does not match {annotation}")
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError("tuple field must be an array")
        item_type = args[0] if args else Any
        return tuple(_coerce(item_type, item) for item in value)
    if origin in (dict, ABCMapping):
        if not isinstance(value, Mapping):
            raise TypeError("mapping field must be an object")
        key_type, item_type = args or (str, Any)
        return {_coerce(key_type, key): _coerce(item_type, item) for key, item in value.items()}
    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if not isinstance(value, Mapping):
            raise TypeError("nested contract field must be an object")
        return _from_mapping(annotation, value)
    if annotation is Any:
        _canonical(value)
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("float field must be numeric")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("contract numbers must be finite")
        return result
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("integer field must be an integer")
        return value
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError("boolean field must be boolean")
        return value
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError("string field must be text")
        return value
    return value


def _from_mapping(cls: type[T], payload: Mapping[str, Any]) -> T:
    allowed = {item.name for item in dataclasses.fields(cls)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown fields for {cls.__name__}: {sorted(unknown)}")
    hints = get_type_hints(cls)
    values: dict[str, Any] = {}
    for item in dataclasses.fields(cls):
        if item.name in payload:
            try:
                values[item.name] = _coerce(hints[item.name], payload[item.name])
            except (TypeError, ValueError) as exc:
                raise type(exc)(f"invalid field {item.name}: {exc}") from exc
        elif item.default is MISSING and item.default_factory is MISSING:
            raise ValueError(f"missing required field: {item.name}")
    result = cls(**values)
    result.validate()
    return result


@dataclass(frozen=True, kw_only=True)
class ResearchContract:
    """Strict, provider-neutral base for all local research-loop messages."""

    contract_type: ClassVar[str] = "contract"
    schema_version: str = SCHEMA_VERSION

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        kind = getattr(cls, "contract_type", "contract")
        if kind != "contract":
            _CONTRACTS[kind] = cls

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema_version: {self.schema_version}")
        _canonical(self)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"contract_type": self.contract_type, **_canonical(self)}

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    def content_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls: type[T], payload: Mapping[str, Any]) -> T:
        body = dict(payload)
        kind = body.pop("contract_type", cls.contract_type)
        if cls is ResearchContract:
            target = _CONTRACTS.get(str(kind))
            if target is None:
                raise ValueError(f"unknown contract_type: {kind}")
            return _from_mapping(target, body)
        if kind != cls.contract_type:
            raise ValueError(f"expected contract_type {cls.contract_type}, got {kind}")
        return _from_mapping(cls, body)

    @classmethod
    def from_json(cls: type[T], value: str) -> T:
        payload = json.loads(value)
        if not isinstance(payload, Mapping):
            raise TypeError("contract JSON must contain an object")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class EvidenceReference:
    uri: str
    sha256: str
    kind: str
    as_of: str

    def validate(self) -> None:
        if not self.uri.strip() or not self.kind.strip() or not self.as_of.strip():
            raise ValueError("evidence uri, kind, and as_of are required")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("evidence sha256 must be a lowercase SHA-256 hash")


@dataclass(frozen=True)
class HypothesisSpec(ResearchContract):
    contract_type: ClassVar[str] = "hypothesis"
    hypothesis_id: str = ""
    title: str = ""
    claim: str = ""
    universe: str = ""
    features: tuple[str, ...] = ()
    horizon: str = ""
    units: Mapping[str, str] = field(default_factory=dict)
    information_cutoff: str = ""
    data_availability: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    falsifiers: tuple[str, ...] = ()
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)
    parent_hypothesis_id: str | None = None

    def validate(self) -> None:
        super().validate()
        required = (self.hypothesis_id, self.title, self.claim, self.universe, self.horizon, self.information_cutoff)
        if any(not value.strip() for value in required):
            raise ValueError("hypothesis identifiers, claim, universe, horizon, and information_cutoff are required")
        if not self.features or not self.data_availability or not self.assumptions or not self.falsifiers:
            raise ValueError("hypothesis features, PIT availability, assumptions, and falsifiers are required")
        if set(self.features) - set(self.units):
            raise ValueError("every feature must declare units")
        _validate_hashes(self.provenance_hashes, field_name="provenance_hashes")


@dataclass(frozen=True)
class AlphaSignal(ResearchContract):
    contract_type: ClassVar[str] = "alpha_signal"
    asset: str = ""
    as_of: str = ""
    value: float = 0.0
    horizon: str = ""
    units: str = "score"
    model_name: str = ""
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if not all(value.strip() for value in (self.asset, self.as_of, self.horizon, self.units, self.model_name)):
            raise ValueError("signal asset, as_of, horizon, units, and model_name are required")
        if not math.isfinite(self.value) or not -1.0 <= self.value <= 1.0:
            raise ValueError("signal value must be finite and within [-1, 1]")
        _validate_hashes(self.provenance_hashes, field_name="provenance_hashes")


@dataclass(frozen=True)
class CandidateSpec(ResearchContract):
    contract_type: ClassVar[str] = "candidate"
    candidate_id: str = ""
    hypothesis_id: str = ""
    family: str = ""
    implementation_ref: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    evidence: tuple[EvidenceReference, ...] = ()
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)
    parent_candidate_id: str | None = None

    def validate(self) -> None:
        super().validate()
        if not all(value.strip() for value in (self.candidate_id, self.hypothesis_id, self.family, self.implementation_ref)):
            raise ValueError("candidate identifiers, family, and implementation_ref are required")
        for item in self.evidence:
            item.validate()
        _validate_hashes(self.provenance_hashes, field_name="provenance_hashes")


@dataclass(frozen=True)
class AgentTask(ResearchContract):
    contract_type: ClassVar[str] = "agent_task"
    task_id: str = ""
    role: str = ""
    objective: str = ""
    permitted_inputs: tuple[str, ...] = ()
    prohibited_inputs: tuple[str, ...] = ()
    output_contract: str = ""
    privacy_class: str = "local_only"
    max_tokens: int = 0
    timeout_seconds: int = 0
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if not all(value.strip() for value in (self.task_id, self.role, self.objective, self.output_contract)):
            raise ValueError("task identity, role, objective, and output_contract are required")
        if self.privacy_class != "local_only":
            raise ValueError("this campaign permits local_only tasks")
        if self.max_tokens <= 0 or self.timeout_seconds <= 0:
            raise ValueError("task budgets must be positive")
        if set(self.permitted_inputs) & set(self.prohibited_inputs):
            raise ValueError("task inputs cannot be both permitted and prohibited")
        _validate_hashes(self.provenance_hashes, field_name="provenance_hashes")


@dataclass(frozen=True)
class SpecialistReviewDraft(ResearchContract):
    """Model-authored judgment only; the controller owns identity and provenance."""

    contract_type: ClassVar[str] = "specialist_review_draft"
    verdict: str = "abstain"
    claims: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()
    falsification_tests: tuple[str, ...] = ()
    veto: bool = False
    confidence: float = 0.0

    def validate(self) -> None:
        super().validate()
        if self.verdict not in {"pass", "conditional", "fail", "abstain"}:
            raise ValueError("unsupported specialist verdict")
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("review confidence must be finite and within [0, 1]")
        if not self.assumptions or not self.falsification_tests:
            raise ValueError("review assumptions and falsification_tests are required")
        if self.veto and not self.counterexamples:
            raise ValueError("a veto must identify at least one concrete counterexample")
        if self.veto and self.verdict not in {"fail", "conditional"}:
            raise ValueError("a veto cannot carry a passing or abstaining verdict")


@dataclass(frozen=True)
class SpecialistReview(ResearchContract):
    contract_type: ClassVar[str] = "specialist_review"
    review_id: str = ""
    task_id: str = ""
    specialist_id: str = ""
    verdict: str = "abstain"
    claims: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()
    counterexamples: tuple[str, ...] = ()
    falsification_tests: tuple[str, ...] = ()
    veto: bool = False
    confidence: float = 0.0
    evidence: tuple[EvidenceReference, ...] = ()
    provenance_hashes: Mapping[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        super().validate()
        if not all(value.strip() for value in (self.review_id, self.task_id, self.specialist_id)):
            raise ValueError("review, task, and specialist identifiers are required")
        if self.verdict not in {"pass", "conditional", "fail", "abstain"}:
            raise ValueError("unsupported specialist verdict")
        if not 0.0 <= self.confidence <= 1.0 or not math.isfinite(self.confidence):
            raise ValueError("review confidence must be finite and within [0, 1]")
        if not self.assumptions or not self.falsification_tests:
            raise ValueError("review assumptions and falsification_tests are required")
        if self.veto and not self.counterexamples:
            raise ValueError("a veto must identify at least one concrete counterexample")
        if self.veto and self.verdict not in {"fail", "conditional"}:
            raise ValueError("a veto cannot carry a passing or abstaining verdict")
        for item in self.evidence:
            item.validate()
        _validate_hashes(self.provenance_hashes, field_name="provenance_hashes")


@dataclass(frozen=True)
class ModelCallRecord(ResearchContract):
    contract_type: ClassVar[str] = "model_call"
    call_id: str = ""
    provider: str = "ollama"
    model: str = ""
    endpoint: str = "http://127.0.0.1:11434"
    started_at: str = ""
    completed_at: str = ""
    status: str = ""
    prompt_hash: str = ""
    schema_hash: str = ""
    config_hash: str = ""
    response_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: str = ""
    redactions: tuple[str, ...] = ()

    def validate(self) -> None:
        super().validate()
        approved = {
            ("ollama", "http://127.0.0.1:11434"),
            ("ollama", "http://localhost:11434"),
            ("deepseek", "https://api.deepseek.com"),
        }
        if (self.provider, self.endpoint) not in approved:
            raise ValueError("model call provider and endpoint are outside the approved routes")
        if not all(value.strip() for value in (self.call_id, self.model, self.started_at, self.completed_at, self.status, self.stop_reason)):
            raise ValueError("model call identity, timing, status, and stop_reason are required")
        _validate_hashes(
            {
                "prompt": self.prompt_hash,
                "schema": self.schema_hash,
                "config": self.config_hash,
                "response": self.response_hash,
            },
            field_name="model call hashes",
        )
        if self.input_tokens < 0 or self.output_tokens < 0 or self.cost_usd < 0.0 or not math.isfinite(self.cost_usd):
            raise ValueError("token counts and model-call cost must be finite and non-negative")


def alpha_signals_from_frame(
    signal: SignalFrame,
    *,
    as_of: str | pd.Timestamp,
    provenance_hashes: Mapping[str, str],
    units: str = "score",
) -> tuple[AlphaSignal, ...]:
    """Adapt one existing `SignalFrame` row without changing legacy model APIs."""

    timestamp = pd.Timestamp(as_of)
    if timestamp not in signal.scores.index:
        raise ValueError("as_of is not present in the signal frame")
    row = signal.scores.loc[timestamp]
    results = tuple(
        AlphaSignal(
            asset=str(asset),
            as_of=timestamp.isoformat(),
            value=float(value),
            horizon=signal.horizon,
            units=units,
            model_name=signal.model_name,
            provenance_hashes=dict(provenance_hashes),
            metadata=dict(signal.metadata),
        )
        for asset, value in row.items()
    )
    for result in results:
        result.validate()
    return results
