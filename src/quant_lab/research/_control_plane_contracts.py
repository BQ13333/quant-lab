from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping


SCHEMA_VERSION = 1
RUN_MANIFEST_NAME = "run_manifest.json"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{7,64}$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")
_EPHEMERAL_ID = re.compile(
    r"(?:[0-9a-f]{8}|[0-9a-f]{32}|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})",
    re.IGNORECASE,
)
_PLACEHOLDER_TERMS = {"h", "u", "test", "test hypothesis", "test universe", "tmp", "temp"}
_NON_RESEARCH_TOKENS = re.compile(
    r"(?:^|[._-])(h|u|test|tmp|temp|synthetic|smoke|demo|example)(?:$|[._-])",
    re.I,
)


class ControlPlaneError(RuntimeError):
    pass


class ConcurrentUpdateError(ControlPlaneError):
    pass


class IntegrityError(ControlPlaneError):
    pass


class IllegalTransitionError(ControlPlaneError):
    pass


class ApprovalError(ControlPlaneError):
    pass


class RunKind(str, Enum):
    RESEARCH = "research"
    TEST = "test"
    SYNTHETIC = "synthetic"
    SMOKE = "smoke"


class WorkflowStatus(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class ValidationStatus(str, Enum):
    NOT_RUN = "not_run"
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


class PromotionStatus(str, Enum):
    INELIGIBLE = "ineligible"
    PARKED_FOR_DATA = "parked_for_data"
    REJECTED = "rejected"
    APPROVAL_REQUIRED = "approval_required"
    APPROVED = "approved"
    SUPERSEDED = "superseded"


class ApprovalDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"
    REWORK = "rework"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"


REQUIRED_VALIDATION_GATES = (
    "point_in_time_data_audit",
    "purged_embargoed_walk_forward_splits",
    "cpcv_pbo",
    "dsr_psr",
    "multiple_testing_fdr",
    "cost_sensitivity",
    "drawdown_turnover",
    "liquidity_capacity",
    "regime_stability",
    "sealed_holdout",
    "data_feature_split_lineage",
    "code_reproducibility",
)

WORKFLOW_STAGES = (
    "research_director",
    "data_steward",
    "feature_engineer",
    "modeler",
    "portfolio_optimizer",
    "validation_officer",
    "risk_manager",
    "reporting_analyst",
    "human_approval",
)

_GATE_METRIC_FIELDS = {
    "point_in_time_data_audit": {
        "row_count",
        "future_leakage_count",
        "survivorship_violation_count",
        "timestamp_lineage_complete",
    },
    "purged_embargoed_walk_forward_splits": {
        "fold_count",
        "purge_observations",
        "embargo_observations",
        "overlap_violation_count",
    },
    "cpcv_pbo": {"path_count", "pbo", "pbo_limit"},
    "dsr_psr": {
        "deflated_sharpe_probability",
        "probabilistic_sharpe_probability",
        "minimum_probability",
    },
    "multiple_testing_fdr": {"hypotheses_tested", "method", "adjusted_p_value", "alpha"},
    "cost_sensitivity": {
        "base_cost_bps",
        "stressed_cost_bps",
        "stressed_net_sharpe",
        "minimum_net_sharpe",
    },
    "drawdown_turnover": {"max_drawdown", "max_drawdown_limit", "annual_turnover", "turnover_limit"},
    "liquidity_capacity": {"adv_participation", "adv_participation_limit", "capacity_aum", "target_aum"},
    "regime_stability": {"regime_count", "minimum_observations_per_regime", "unstable_regime_count"},
    "sealed_holdout": {
        "sealed",
        "release_count",
        "observation_count",
        "optimization_access_count",
        "passed",
        "holdout_manifest_sha256",
        "release_ledger_sha256",
    },
    "data_feature_split_lineage": {
        "data_manifest_sha256",
        "feature_manifest_sha256",
        "split_manifest_sha256",
    },
    "code_reproducibility": {
        "code_commit",
        "environment_lock_sha256",
        "random_seed",
        "rerun_count",
        "reproduced",
    },
}

_POLICY_THRESHOLD_FIELDS = {
    "point_in_time_data_audit": {"minimum_row_count"},
    "purged_embargoed_walk_forward_splits": {
        "minimum_fold_count",
        "minimum_purge_observations",
        "minimum_embargo_observations",
    },
    "cpcv_pbo": {"minimum_path_count", "maximum_pbo"},
    "dsr_psr": {"minimum_probability"},
    "multiple_testing_fdr": {"alpha", "allowed_methods"},
    "cost_sensitivity": {"minimum_stress_multiplier", "minimum_stressed_net_sharpe"},
    "drawdown_turnover": {"maximum_drawdown", "maximum_annual_turnover"},
    "liquidity_capacity": {"maximum_adv_participation", "minimum_capacity_aum"},
    "regime_stability": {"minimum_regime_count", "minimum_observations_per_regime"},
    "sealed_holdout": {"minimum_observation_count", "required_release_count"},
    "data_feature_split_lineage": set(),
    "code_reproducibility": {"minimum_rerun_count"},
}
_VALIDATION_POLICY_FIELDS = {"policy_id", "gates"}
_POLICY_GATE_FIELDS = {"allowed_calculators", "thresholds"}
_POLICY_CALCULATOR_FIELDS = {
    "calculator_id",
    "calculator_version",
    "calculator_source_sha256",
}

_MANIFEST_FIELDS = {
    "run_id",
    "parent_run_id",
    "run_kind",
    "promotion_eligible",
    "hypothesis",
    "universe",
    "data_manifest_id",
    "data_asof_utc",
    "feature_manifest_hash",
    "code_commit",
    "target",
    "label_horizon",
    "split_manifest_id",
    "workflow_status",
    "validation_status",
    "promotion_status",
    "created_at",
    "updated_at",
}


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _hash_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_utc(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field_name} must be explicitly UTC")
    return parsed


def _strict_fields(payload: Mapping[str, Any], expected: set[str], kind: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{kind} fields mismatch; missing={missing}, extra={extra}")


def _enum(enum_type: type[Enum], value: Any, field_name: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {field_name}: {value!r}") from exc


def _require_sha256(field_name: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")


def _anchor_hashes(value: str | frozenset[str]) -> frozenset[str]:
    values = frozenset(value.splitlines()) if isinstance(value, str) else value
    values = frozenset(item.strip() for item in values if item.strip())
    if not values:
        raise IntegrityError("root-owned trust registry must contain at least one policy hash")
    for item in values:
        _require_sha256("root-owned trust registry entry", item)
    return values


def _validate_identifier(field_name: str, value: str | None) -> None:
    if value is not None and not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{field_name} must be a path-safe identifier")


def _metric_int(metrics: Mapping[str, Any], name: str, *, minimum: int = 0) -> int:
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"validation metric {name} must be an integer >= {minimum}")
    return value


def _metric_number(metrics: Mapping[str, Any], name: str) -> float:
    value = metrics[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"validation metric {name} must be a finite number")
    return float(value)


def _metric_bool(metrics: Mapping[str, Any], name: str) -> bool:
    value = metrics[name]
    if not isinstance(value, bool):
        raise ValueError(f"validation metric {name} must be boolean")
    return value


def _validate_gate_metrics(artifact: "ValidationArtifact") -> None:
    metrics = artifact.metrics
    expected = _GATE_METRIC_FIELDS[artifact.gate_id]
    _strict_fields(metrics, expected, f"{artifact.gate_id} metrics")
    gate_passed: bool
    if artifact.gate_id == "point_in_time_data_audit":
        gate_passed = (
            _metric_int(metrics, "row_count", minimum=1) > 0
            and _metric_int(metrics, "future_leakage_count") == 0
            and _metric_int(metrics, "survivorship_violation_count") == 0
            and _metric_bool(metrics, "timestamp_lineage_complete")
        )
    elif artifact.gate_id == "purged_embargoed_walk_forward_splits":
        gate_passed = (
            _metric_int(metrics, "fold_count", minimum=2) >= 2
            and _metric_int(metrics, "purge_observations", minimum=1) >= 1
            and _metric_int(metrics, "embargo_observations", minimum=1) >= 1
            and _metric_int(metrics, "overlap_violation_count") == 0
        )
    elif artifact.gate_id == "cpcv_pbo":
        path_count = _metric_int(metrics, "path_count", minimum=2)
        pbo = _metric_number(metrics, "pbo")
        limit = _metric_number(metrics, "pbo_limit")
        if not 0.0 <= pbo <= 1.0 or not 0.0 <= limit <= 1.0:
            raise ValueError("PBO metrics must lie within [0, 1]")
        gate_passed = path_count >= 2 and pbo <= limit
    elif artifact.gate_id == "dsr_psr":
        dsr = _metric_number(metrics, "deflated_sharpe_probability")
        psr = _metric_number(metrics, "probabilistic_sharpe_probability")
        minimum = _metric_number(metrics, "minimum_probability")
        if any(not 0.0 <= value <= 1.0 for value in (dsr, psr, minimum)):
            raise ValueError("Sharpe probability metrics must lie within [0, 1]")
        gate_passed = dsr >= minimum and psr >= minimum
    elif artifact.gate_id == "multiple_testing_fdr":
        _metric_int(metrics, "hypotheses_tested", minimum=1)
        if metrics["method"] not in {"benjamini_hochberg", "benjamini_yekutieli"}:
            raise ValueError("FDR method must be a supported deterministic correction")
        adjusted = _metric_number(metrics, "adjusted_p_value")
        alpha = _metric_number(metrics, "alpha")
        if not 0.0 <= adjusted <= 1.0 or not 0.0 < alpha < 1.0:
            raise ValueError("FDR probability metrics are outside their valid ranges")
        gate_passed = adjusted <= alpha
    elif artifact.gate_id == "cost_sensitivity":
        base = _metric_number(metrics, "base_cost_bps")
        stressed = _metric_number(metrics, "stressed_cost_bps")
        stressed_sharpe = _metric_number(metrics, "stressed_net_sharpe")
        minimum_sharpe = _metric_number(metrics, "minimum_net_sharpe")
        if base < 0.0 or stressed < 0.0:
            raise ValueError("cost assumptions cannot be negative")
        gate_passed = stressed >= 2.0 * base and stressed_sharpe >= minimum_sharpe
    elif artifact.gate_id == "drawdown_turnover":
        drawdown = _metric_number(metrics, "max_drawdown")
        drawdown_limit = _metric_number(metrics, "max_drawdown_limit")
        turnover = _metric_number(metrics, "annual_turnover")
        turnover_limit = _metric_number(metrics, "turnover_limit")
        if not -1.0 <= drawdown <= 0.0 or not -1.0 <= drawdown_limit <= 0.0:
            raise ValueError("drawdown metrics must lie within [-1, 0]")
        if turnover < 0.0 or turnover_limit < 0.0:
            raise ValueError("turnover metrics cannot be negative")
        gate_passed = drawdown >= drawdown_limit and turnover <= turnover_limit
    elif artifact.gate_id == "liquidity_capacity":
        participation = _metric_number(metrics, "adv_participation")
        participation_limit = _metric_number(metrics, "adv_participation_limit")
        capacity = _metric_number(metrics, "capacity_aum")
        target = _metric_number(metrics, "target_aum")
        if min(participation, participation_limit, capacity, target) < 0.0:
            raise ValueError("liquidity and capacity metrics cannot be negative")
        gate_passed = participation <= participation_limit and capacity >= target
    elif artifact.gate_id == "regime_stability":
        gate_passed = (
            _metric_int(metrics, "regime_count", minimum=2) >= 2
            and _metric_int(metrics, "minimum_observations_per_regime", minimum=1) >= 1
            and _metric_int(metrics, "unstable_regime_count") == 0
        )
    elif artifact.gate_id == "sealed_holdout":
        _require_sha256("holdout_manifest_sha256", str(metrics["holdout_manifest_sha256"]))
        _require_sha256("release_ledger_sha256", str(metrics["release_ledger_sha256"]))
        gate_passed = (
            _metric_bool(metrics, "sealed")
            and _metric_int(metrics, "release_count") == 1
            and _metric_int(metrics, "observation_count", minimum=63) >= 63
            and _metric_int(metrics, "optimization_access_count") == 0
            and _metric_bool(metrics, "passed")
        )
    elif artifact.gate_id == "data_feature_split_lineage":
        for name in expected:
            _require_sha256(name, str(metrics[name]))
        gate_passed = (
            str(metrics["data_manifest_sha256"]) == artifact.input_hashes["data_manifest"]
            and str(metrics["feature_manifest_sha256"]) == artifact.input_hashes["feature_manifest"]
            and str(metrics["split_manifest_sha256"]) == artifact.input_hashes["split_manifest"]
        )
    else:
        if str(metrics["code_commit"]) != artifact.code_commit:
            raise ValueError("reproducibility code_commit does not match the artifact binding")
        _require_sha256("environment_lock_sha256", str(metrics["environment_lock_sha256"]))
        _metric_int(metrics, "random_seed")
        reruns = _metric_int(metrics, "rerun_count", minimum=2)
        gate_passed = reruns >= 2 and _metric_bool(metrics, "reproduced")
    if artifact.status is GateStatus.PASS and not gate_passed:
        raise ValueError(f"{artifact.gate_id} cannot declare pass when its deterministic metrics fail")
    if artifact.status is GateStatus.FAIL and gate_passed:
        raise ValueError(f"{artifact.gate_id} cannot declare fail when its deterministic metrics pass")


def _looks_non_promotable(run_id: str, hypothesis: str, universe: str) -> bool:
    fields = (run_id.strip(), hypothesis.strip(), universe.strip())
    lowered = tuple(value.lower() for value in fields)
    return (
        any(value in _PLACEHOLDER_TERMS for value in lowered)
        or any(_NON_RESEARCH_TOKENS.search(value) for value in fields)
        or bool(_EPHEMERAL_ID.fullmatch(run_id))
        or "synthetic" in hypothesis.lower()
        or "synthetic" in universe.lower()
    )


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    parent_run_id: str | None
    run_kind: RunKind
    promotion_eligible: bool
    hypothesis: str
    universe: str
    data_manifest_id: str
    data_asof_utc: str
    feature_manifest_hash: str
    code_commit: str
    target: str
    label_horizon: str
    split_manifest_id: str
    workflow_status: WorkflowStatus
    validation_status: ValidationStatus
    promotion_status: PromotionStatus
    created_at: str
    updated_at: str

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        parent_run_id: str | None,
        run_kind: RunKind | str,
        hypothesis: str,
        universe: str,
        data_manifest_id: str,
        data_asof_utc: str,
        feature_manifest_hash: str,
        code_commit: str,
        target: str,
        label_horizon: str,
        split_manifest_id: str,
        created_at: str | None = None,
    ) -> "RunManifest":
        kind = _enum(RunKind, run_kind, "run_kind")
        timestamp = created_at or _utc_now()
        promotion_eligible = kind is RunKind.RESEARCH and not _looks_non_promotable(
            run_id, hypothesis, universe
        )
        manifest = cls(
            run_id=run_id,
            parent_run_id=parent_run_id,
            run_kind=kind,
            promotion_eligible=promotion_eligible,
            hypothesis=hypothesis,
            universe=universe,
            data_manifest_id=data_manifest_id,
            data_asof_utc=data_asof_utc,
            feature_manifest_hash=feature_manifest_hash,
            code_commit=code_commit,
            target=target,
            label_horizon=label_horizon,
            split_manifest_id=split_manifest_id,
            workflow_status=WorkflowStatus.CREATED,
            validation_status=ValidationStatus.NOT_RUN,
            promotion_status=PromotionStatus.INELIGIBLE,
            created_at=timestamp,
            updated_at=timestamp,
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if not isinstance(self.run_kind, RunKind):
            raise TypeError("run_kind must be a RunKind")
        if not isinstance(self.workflow_status, WorkflowStatus):
            raise TypeError("workflow_status must be a WorkflowStatus")
        if not isinstance(self.validation_status, ValidationStatus):
            raise TypeError("validation_status must be a ValidationStatus")
        if not isinstance(self.promotion_status, PromotionStatus):
            raise TypeError("promotion_status must be a PromotionStatus")
        string_fields = {
            "run_id": self.run_id,
            "hypothesis": self.hypothesis,
            "universe": self.universe,
            "data_manifest_id": self.data_manifest_id,
            "data_asof_utc": self.data_asof_utc,
            "feature_manifest_hash": self.feature_manifest_hash,
            "code_commit": self.code_commit,
            "target": self.target,
            "label_horizon": self.label_horizon,
            "split_manifest_id": self.split_manifest_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if any(not isinstance(value, str) for value in string_fields.values()):
            raise TypeError("run manifest textual fields must be strings")
        if self.parent_run_id is not None and not isinstance(self.parent_run_id, str):
            raise TypeError("parent_run_id must be null or a string")
        if not isinstance(self.promotion_eligible, bool):
            raise TypeError("promotion_eligible must be boolean")
        _validate_identifier("run_id", self.run_id)
        _validate_identifier("parent_run_id", self.parent_run_id)
        if self.parent_run_id == self.run_id:
            raise ValueError("parent_run_id cannot equal run_id")
        if not all(
            value.strip()
            for value in (
                self.hypothesis,
                self.universe,
                self.data_manifest_id,
                self.target,
                self.label_horizon,
                self.split_manifest_id,
            )
        ):
            raise ValueError("manifest research identity fields must not be empty")
        data_asof = _parse_utc(self.data_asof_utc, "data_asof_utc")
        created = _parse_utc(self.created_at, "created_at")
        updated = _parse_utc(self.updated_at, "updated_at")
        if updated < created:
            raise ValueError("updated_at cannot precede created_at")
        if data_asof > created:
            raise ValueError("data_asof_utc cannot follow run creation")
        _require_sha256("feature_manifest_hash", self.feature_manifest_hash)
        if not _GIT_COMMIT.fullmatch(self.code_commit):
            raise ValueError("code_commit must be a lowercase Git commit hash")
        expected_eligible = self.run_kind is RunKind.RESEARCH and not _looks_non_promotable(
            self.run_id, self.hypothesis, self.universe
        )
        if self.promotion_eligible is not expected_eligible:
            raise ValueError("promotion_eligible must be derived from run identity and run_kind")
        if not self.promotion_eligible and self.promotion_status not in {
            PromotionStatus.INELIGIBLE,
            PromotionStatus.REJECTED,
            PromotionStatus.SUPERSEDED,
        }:
            raise ValueError("non-promotable runs cannot enter a promotable state")
        if self.promotion_status is PromotionStatus.APPROVED and self.validation_status is not ValidationStatus.PASS:
            raise ValueError("approved runs must have passing validation")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "run_kind": self.run_kind.value,
            "promotion_eligible": self.promotion_eligible,
            "hypothesis": self.hypothesis,
            "universe": self.universe,
            "data_manifest_id": self.data_manifest_id,
            "data_asof_utc": self.data_asof_utc,
            "feature_manifest_hash": self.feature_manifest_hash,
            "code_commit": self.code_commit,
            "target": self.target,
            "label_horizon": self.label_horizon,
            "split_manifest_id": self.split_manifest_id,
            "workflow_status": self.workflow_status.value,
            "validation_status": self.validation_status.value,
            "promotion_status": self.promotion_status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RunManifest":
        _strict_fields(payload, _MANIFEST_FIELDS, "run manifest")
        if not isinstance(payload["promotion_eligible"], bool):
            raise TypeError("promotion_eligible must be boolean")
        string_fields = _MANIFEST_FIELDS - {"parent_run_id", "promotion_eligible"}
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise TypeError("run manifest textual fields must be strings")
        if payload["parent_run_id"] is not None and not isinstance(payload["parent_run_id"], str):
            raise TypeError("parent_run_id must be null or a string")
        manifest = cls(
            run_id=payload["run_id"],
            parent_run_id=payload["parent_run_id"],
            run_kind=_enum(RunKind, payload["run_kind"], "run_kind"),
            promotion_eligible=payload["promotion_eligible"],
            hypothesis=payload["hypothesis"],
            universe=payload["universe"],
            data_manifest_id=payload["data_manifest_id"],
            data_asof_utc=payload["data_asof_utc"],
            feature_manifest_hash=payload["feature_manifest_hash"],
            code_commit=payload["code_commit"],
            target=payload["target"],
            label_horizon=payload["label_horizon"],
            split_manifest_id=payload["split_manifest_id"],
            workflow_status=_enum(WorkflowStatus, payload["workflow_status"], "workflow_status"),
            validation_status=_enum(ValidationStatus, payload["validation_status"], "validation_status"),
            promotion_status=_enum(PromotionStatus, payload["promotion_status"], "promotion_status"),
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
        )
        manifest.validate()
        return manifest

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def content_hash(self) -> str:
        return _hash_bytes(self.canonical_bytes())

    def configuration_hash(self) -> str:
        payload = self.to_dict()
        for field_name in (
            "workflow_status",
            "validation_status",
            "promotion_status",
            "created_at",
            "updated_at",
        ):
            payload.pop(field_name)
        return _hash_bytes(_canonical_bytes(payload))

    def identity_hash(self) -> str:
        payload = self.to_dict()
        for field_name in ("workflow_status", "validation_status", "promotion_status", "updated_at"):
            payload.pop(field_name)
        return _hash_bytes(_canonical_bytes(payload))


_ARTIFACT_FIELDS = {
    "schema_version",
    "gate_id",
    "run_id",
    "candidate_id",
    "candidate_hash",
    "data_manifest_id",
    "data_asof_utc",
    "feature_manifest_hash",
    "code_commit",
    "split_manifest_id",
    "status",
    "producer",
    "producer_execution_id",
    "calculator_id",
    "calculator_version",
    "calculator_source_sha256",
    "input_hashes",
    "runner_id",
    "runner_key_id",
    "runner_signature_algorithm",
    "runner_signature",
    "metrics",
    "generated_at",
}
_VALIDATION_INPUT_HASHES = {
    "candidate",
    "data_manifest",
    "feature_manifest",
    "split_manifest",
    "calculator_source",
}


@dataclass(frozen=True)
class ValidationArtifact:
    gate_id: str
    run_id: str
    candidate_id: str
    candidate_hash: str
    data_manifest_id: str
    data_asof_utc: str
    feature_manifest_hash: str
    code_commit: str
    split_manifest_id: str
    status: GateStatus
    producer_execution_id: str
    calculator_id: str
    calculator_version: str
    calculator_source_sha256: str
    input_hashes: Mapping[str, str]
    runner_id: str
    runner_key_id: str
    runner_signature: str
    metrics: Mapping[str, Any]
    generated_at: str
    producer: str = "deterministic_python"
    runner_signature_algorithm: str = "Ed25519"
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported validation artifact schema_version")
        if not isinstance(self.status, GateStatus):
            raise TypeError("validation artifact status must be a GateStatus")
        string_values = (
            self.gate_id,
            self.run_id,
            self.candidate_id,
            self.candidate_hash,
            self.data_manifest_id,
            self.data_asof_utc,
            self.feature_manifest_hash,
            self.code_commit,
            self.split_manifest_id,
            self.producer,
            self.producer_execution_id,
            self.calculator_id,
            self.calculator_version,
            self.calculator_source_sha256,
            self.runner_id,
            self.runner_key_id,
            self.runner_signature,
            self.runner_signature_algorithm,
            self.generated_at,
        )
        if any(not isinstance(value, str) for value in string_values):
            raise TypeError("validation artifact textual fields must be strings")
        if self.gate_id not in REQUIRED_VALIDATION_GATES:
            raise ValueError(f"unknown validation gate: {self.gate_id}")
        _validate_identifier("run_id", self.run_id)
        _validate_identifier("candidate_id", self.candidate_id)
        _validate_identifier("producer_execution_id", self.producer_execution_id)
        _validate_identifier("calculator_id", self.calculator_id)
        _validate_identifier("calculator_version", self.calculator_version)
        _validate_identifier("runner_id", self.runner_id)
        _validate_identifier("runner_key_id", self.runner_key_id)
        if self.producer != "deterministic_python":
            raise ValueError("validation artifacts must be produced by deterministic_python")
        if not self.data_manifest_id.strip() or not self.split_manifest_id.strip():
            raise ValueError("validation artifact data and split identities are required")
        _parse_utc(self.data_asof_utc, "data_asof_utc")
        _parse_utc(self.generated_at, "generated_at")
        _require_sha256("feature_manifest_hash", self.feature_manifest_hash)
        _require_sha256("candidate_hash", self.candidate_hash)
        _require_sha256("calculator_source_sha256", self.calculator_source_sha256)
        if not _GIT_COMMIT.fullmatch(self.code_commit):
            raise ValueError("code_commit must be a lowercase Git commit hash")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be an object")
        if not isinstance(self.input_hashes, Mapping) or set(self.input_hashes) != _VALIDATION_INPUT_HASHES:
            raise ValueError("input_hashes must contain the exact validation input set")
        for name, value in self.input_hashes.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise TypeError("input_hashes must map strings to strings")
            _require_sha256(f"input_hashes.{name}", value)
        if self.input_hashes["candidate"] != self.candidate_hash:
            raise ValueError("candidate input hash does not match candidate_hash")
        if self.input_hashes["feature_manifest"] != self.feature_manifest_hash:
            raise ValueError("feature input hash does not match feature_manifest_hash")
        if self.input_hashes["calculator_source"] != self.calculator_source_sha256:
            raise ValueError("calculator source input hash does not match calculator_source_sha256")
        if self.runner_signature_algorithm != "Ed25519":
            raise ValueError("only detached Ed25519 runner signatures are supported")
        try:
            runner_signature = base64.b64decode(self.runner_signature, validate=True)
        except ValueError as exc:
            raise ValueError("runner_signature must be valid base64") from exc
        if len(runner_signature) != 64:
            raise ValueError("Ed25519 runner signature must contain 64 bytes")
        _canonical_bytes(dict(self.metrics))
        _validate_gate_metrics(self)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "gate_id": self.gate_id,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "data_manifest_id": self.data_manifest_id,
            "data_asof_utc": self.data_asof_utc,
            "feature_manifest_hash": self.feature_manifest_hash,
            "code_commit": self.code_commit,
            "split_manifest_id": self.split_manifest_id,
            "status": self.status.value,
            "producer": self.producer,
            "producer_execution_id": self.producer_execution_id,
            "calculator_id": self.calculator_id,
            "calculator_version": self.calculator_version,
            "calculator_source_sha256": self.calculator_source_sha256,
            "input_hashes": dict(self.input_hashes),
            "runner_id": self.runner_id,
            "runner_key_id": self.runner_key_id,
            "runner_signature_algorithm": self.runner_signature_algorithm,
            "metrics": dict(self.metrics),
            "generated_at": self.generated_at,
        }

    def signing_bytes(self) -> bytes:
        return _canonical_bytes(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self.unsigned_dict(), "runner_signature": self.runner_signature}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationArtifact":
        _strict_fields(payload, _ARTIFACT_FIELDS, "validation artifact")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported validation artifact schema_version")
        if not isinstance(payload["metrics"], Mapping):
            raise TypeError("metrics must be an object")
        if not isinstance(payload["input_hashes"], Mapping):
            raise TypeError("validation artifact input_hashes must be an object")
        string_fields = _ARTIFACT_FIELDS - {"schema_version", "metrics", "input_hashes"}
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise TypeError("validation artifact textual fields must be strings")
        artifact = cls(
            schema_version=SCHEMA_VERSION,
            gate_id=payload["gate_id"],
            run_id=payload["run_id"],
            candidate_id=payload["candidate_id"],
            candidate_hash=payload["candidate_hash"],
            data_manifest_id=payload["data_manifest_id"],
            data_asof_utc=payload["data_asof_utc"],
            feature_manifest_hash=payload["feature_manifest_hash"],
            code_commit=payload["code_commit"],
            split_manifest_id=payload["split_manifest_id"],
            status=_enum(GateStatus, payload["status"], "status"),
            producer=payload["producer"],
            producer_execution_id=payload["producer_execution_id"],
            calculator_id=payload["calculator_id"],
            calculator_version=payload["calculator_version"],
            calculator_source_sha256=payload["calculator_source_sha256"],
            input_hashes=dict(payload["input_hashes"]),
            runner_id=payload["runner_id"],
            runner_key_id=payload["runner_key_id"],
            runner_signature_algorithm=payload["runner_signature_algorithm"],
            runner_signature=payload["runner_signature"],
            metrics=dict(payload["metrics"]),
            generated_at=payload["generated_at"],
        )
        artifact.validate()
        return artifact


_ARTIFACT_REF_FIELDS = {"artifact_uri", "sha256", "size_bytes", "status", "producer_execution_id"}
_BUNDLE_FIELDS = {
    "schema_version",
    "run_id",
    "candidate_id",
    "candidate_hash",
    "manifest_identity_hash",
    "runner_trust_policy_hash",
    "artifacts",
    "created_at",
}


@dataclass(frozen=True)
class EvidenceArtifactReference:
    artifact_uri: str
    sha256: str
    size_bytes: int
    status: GateStatus
    producer_execution_id: str

    def to_dict(self) -> dict[str, Any]:
        if not isinstance(self.status, GateStatus):
            raise TypeError("evidence reference status must be a GateStatus")
        if not all(
            isinstance(value, str)
            for value in (self.artifact_uri, self.sha256, self.producer_execution_id)
        ):
            raise TypeError("evidence reference textual fields must be strings")
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("evidence reference size_bytes must be an integer")
        _require_sha256("artifact sha256", self.sha256)
        if self.size_bytes <= 0:
            raise ValueError("artifact size_bytes must be positive")
        _validate_identifier("producer_execution_id", self.producer_execution_id)
        return {
            "artifact_uri": self.artifact_uri,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "status": self.status.value,
            "producer_execution_id": self.producer_execution_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "EvidenceArtifactReference":
        _strict_fields(payload, _ARTIFACT_REF_FIELDS, "evidence artifact reference")
        if not all(
            isinstance(payload[name], str)
            for name in ("artifact_uri", "sha256", "status", "producer_execution_id")
        ):
            raise TypeError("evidence reference textual fields must be strings")
        if isinstance(payload["size_bytes"], bool) or not isinstance(payload["size_bytes"], int):
            raise TypeError("evidence reference size_bytes must be an integer")
        reference = cls(
            artifact_uri=payload["artifact_uri"],
            sha256=payload["sha256"],
            size_bytes=payload["size_bytes"],
            status=_enum(GateStatus, payload["status"], "status"),
            producer_execution_id=payload["producer_execution_id"],
        )
        _require_sha256("artifact sha256", reference.sha256)
        if reference.size_bytes <= 0:
            raise ValueError("artifact size_bytes must be positive")
        _validate_identifier("producer_execution_id", reference.producer_execution_id)
        return reference


@dataclass(frozen=True)
class ValidationEvidenceBundle:
    run_id: str
    candidate_id: str
    candidate_hash: str
    manifest_identity_hash: str
    runner_trust_policy_hash: str
    artifacts: Mapping[str, EvidenceArtifactReference]
    created_at: str
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported evidence bundle schema_version")
        if not all(
            isinstance(value, str)
            for value in (
                self.run_id,
                self.candidate_id,
                self.candidate_hash,
                self.manifest_identity_hash,
                self.runner_trust_policy_hash,
                self.created_at,
            )
        ):
            raise TypeError("evidence bundle textual fields must be strings")
        _validate_identifier("run_id", self.run_id)
        _validate_identifier("candidate_id", self.candidate_id)
        _require_sha256("candidate_hash", self.candidate_hash)
        _require_sha256("manifest_identity_hash", self.manifest_identity_hash)
        _require_sha256("runner_trust_policy_hash", self.runner_trust_policy_hash)
        _parse_utc(self.created_at, "created_at")
        expected = set(REQUIRED_VALIDATION_GATES)
        actual = set(self.artifacts)
        if actual != expected:
            raise ValueError(
                f"evidence bundle gate set mismatch; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        uris = [item.artifact_uri for item in self.artifacts.values()]
        if len(uris) != len(set(uris)):
            raise ValueError("evidence bundle artifact paths must be unique")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "candidate_id": self.candidate_id,
            "candidate_hash": self.candidate_hash,
            "manifest_identity_hash": self.manifest_identity_hash,
            "runner_trust_policy_hash": self.runner_trust_policy_hash,
            "artifacts": {key: self.artifacts[key].to_dict() for key in sorted(self.artifacts)},
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationEvidenceBundle":
        _strict_fields(payload, _BUNDLE_FIELDS, "evidence bundle")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported evidence bundle schema_version")
        if not isinstance(payload["artifacts"], Mapping):
            raise TypeError("evidence bundle artifacts must be an object")
        if type(payload["schema_version"]) is not int:
            raise TypeError("evidence bundle schema_version must be an integer")
        if not all(
            isinstance(payload[name], str)
            for name in (
                "run_id",
                "candidate_id",
                "candidate_hash",
                "manifest_identity_hash",
                "runner_trust_policy_hash",
                "created_at",
            )
        ):
            raise TypeError("evidence bundle textual fields must be strings")
        bundle = cls(
            schema_version=SCHEMA_VERSION,
            run_id=payload["run_id"],
            candidate_id=payload["candidate_id"],
            candidate_hash=payload["candidate_hash"],
            manifest_identity_hash=payload["manifest_identity_hash"],
            runner_trust_policy_hash=payload["runner_trust_policy_hash"],
            artifacts={
                str(key): EvidenceArtifactReference.from_dict(value)
                for key, value in payload["artifacts"].items()
                if isinstance(value, Mapping)
            },
            created_at=payload["created_at"],
        )
        if len(bundle.artifacts) != len(payload["artifacts"]):
            raise TypeError("every evidence artifact reference must be an object")
        bundle.validate()
        return bundle

    def canonical_bytes(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def content_hash(self) -> str:
        return _hash_bytes(self.canonical_bytes())

    def overall_status(self) -> ValidationStatus:
        statuses = {item.status for item in self.artifacts.values()}
        if GateStatus.FAIL in statuses:
            return ValidationStatus.FAIL
        if statuses == {GateStatus.PASS}:
            return ValidationStatus.PASS
        return ValidationStatus.INCONCLUSIVE


_APPROVAL_FIELDS = {
    "schema_version",
    "approval_id",
    "run_id",
    "manifest_identity_hash",
    "decision",
    "approved_by",
    "approved_at",
    "candidate_id",
    "evidence_bundle_hash",
    "limitations_acknowledged",
    "signer_key_id",
    "signature_algorithm",
    "signature",
}
_SUPERSESSION_FIELDS = {
    "schema_version",
    "supersession_id",
    "run_id",
    "replacement_run_id",
    "manifest_identity_hash",
    "evidence_bundle_hash",
    "approved_by",
    "approved_at",
    "limitations_acknowledged",
    "signer_key_id",
    "signature_algorithm",
    "signature",
}


@dataclass(frozen=True)
class HumanApprovalRecord:
    approval_id: str
    run_id: str
    manifest_identity_hash: str
    decision: ApprovalDecision
    approved_by: str
    approved_at: str
    candidate_id: str
    evidence_bundle_hash: str
    limitations_acknowledged: tuple[str, ...]
    signer_key_id: str
    signature: str
    signature_algorithm: str = "Ed25519"
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported human approval schema_version")
        if not isinstance(self.decision, ApprovalDecision):
            raise TypeError("human approval decision must be an ApprovalDecision")
        string_values = (
            self.approval_id,
            self.run_id,
            self.manifest_identity_hash,
            self.approved_by,
            self.approved_at,
            self.candidate_id,
            self.evidence_bundle_hash,
            self.signer_key_id,
            self.signature,
            self.signature_algorithm,
        )
        if any(not isinstance(value, str) for value in string_values):
            raise TypeError("human approval textual fields must be strings")
        if not isinstance(self.limitations_acknowledged, tuple) or not all(
            isinstance(value, str) for value in self.limitations_acknowledged
        ):
            raise TypeError("limitations_acknowledged must be a tuple of strings")
        for field_name, value in (
            ("approval_id", self.approval_id),
            ("run_id", self.run_id),
            ("candidate_id", self.candidate_id),
            ("signer_key_id", self.signer_key_id),
        ):
            _validate_identifier(field_name, value)
        if not self.approved_by.strip():
            raise ValueError("approved_by is required")
        if self.approved_by.lower().startswith(("agent", "model", "service")):
            raise ValueError("machine identities cannot approve research")
        _parse_utc(self.approved_at, "approved_at")
        _require_sha256("manifest_identity_hash", self.manifest_identity_hash)
        _require_sha256("evidence_bundle_hash", self.evidence_bundle_hash)
        if not self.limitations_acknowledged or any(not item.strip() for item in self.limitations_acknowledged):
            raise ValueError("limitations_acknowledged must identify at least one acknowledged limitation")
        if self.signature_algorithm != "Ed25519":
            raise ValueError("only detached Ed25519 approval signatures are supported")
        try:
            signature = base64.b64decode(self.signature, validate=True)
        except ValueError as exc:
            raise ValueError("signature must be valid base64") from exc
        if len(signature) != 64:
            raise ValueError("Ed25519 signature must contain 64 bytes")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "approval_id": self.approval_id,
            "run_id": self.run_id,
            "manifest_identity_hash": self.manifest_identity_hash,
            "decision": self.decision.value,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "candidate_id": self.candidate_id,
            "evidence_bundle_hash": self.evidence_bundle_hash,
            "limitations_acknowledged": list(self.limitations_acknowledged),
            "signer_key_id": self.signer_key_id,
            "signature_algorithm": self.signature_algorithm,
        }

    def signing_bytes(self) -> bytes:
        return _canonical_bytes(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self.unsigned_dict(), "signature": self.signature}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanApprovalRecord":
        _strict_fields(payload, _APPROVAL_FIELDS, "human approval record")
        limitations = payload["limitations_acknowledged"]
        if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
            raise TypeError("limitations_acknowledged must be an array of strings")
        if type(payload["schema_version"]) is not int:
            raise TypeError("human approval schema_version must be an integer")
        string_fields = _APPROVAL_FIELDS - {"schema_version", "limitations_acknowledged"}
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise TypeError("human approval textual fields must be strings")
        record = cls(
            schema_version=int(payload["schema_version"]),
            approval_id=payload["approval_id"],
            run_id=payload["run_id"],
            manifest_identity_hash=payload["manifest_identity_hash"],
            decision=_enum(ApprovalDecision, payload["decision"], "decision"),
            approved_by=payload["approved_by"],
            approved_at=payload["approved_at"],
            candidate_id=payload["candidate_id"],
            evidence_bundle_hash=payload["evidence_bundle_hash"],
            limitations_acknowledged=tuple(limitations),
            signer_key_id=payload["signer_key_id"],
            signature_algorithm=payload["signature_algorithm"],
            signature=payload["signature"],
        )
        record.validate()
        return record
@dataclass(frozen=True)
class HumanSupersessionRecord:
    supersession_id: str
    run_id: str
    replacement_run_id: str
    manifest_identity_hash: str
    evidence_bundle_hash: str
    approved_by: str
    approved_at: str
    limitations_acknowledged: tuple[str, ...]
    signer_key_id: str
    signature: str
    signature_algorithm: str = "Ed25519"
    schema_version: int = SCHEMA_VERSION

    def validate(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != SCHEMA_VERSION:
            raise ValueError("unsupported human supersession schema_version")
        values = (
            self.supersession_id,
            self.run_id,
            self.replacement_run_id,
            self.manifest_identity_hash,
            self.evidence_bundle_hash,
            self.approved_by,
            self.approved_at,
            self.signer_key_id,
            self.signature,
            self.signature_algorithm,
        )
        if any(not isinstance(value, str) for value in values):
            raise TypeError("human supersession textual fields must be strings")
        for field_name, value in (
            ("supersession_id", self.supersession_id),
            ("run_id", self.run_id),
            ("replacement_run_id", self.replacement_run_id),
            ("signer_key_id", self.signer_key_id),
        ):
            _validate_identifier(field_name, value)
        if self.run_id == self.replacement_run_id:
            raise ValueError("replacement_run_id must differ from run_id")
        if not self.approved_by.strip() or self.approved_by.lower().startswith(("agent", "model", "service")):
            raise ValueError("supersession requires a named human identity")
        _parse_utc(self.approved_at, "approved_at")
        _require_sha256("manifest_identity_hash", self.manifest_identity_hash)
        _require_sha256("evidence_bundle_hash", self.evidence_bundle_hash)
        if not isinstance(self.limitations_acknowledged, tuple) or not self.limitations_acknowledged:
            raise TypeError("limitations_acknowledged must be a nonempty tuple")
        if any(not isinstance(item, str) or not item.strip() for item in self.limitations_acknowledged):
            raise ValueError("supersession limitations must be nonempty strings")
        if self.signature_algorithm != "Ed25519":
            raise ValueError("only detached Ed25519 supersession signatures are supported")
        try:
            signature = base64.b64decode(self.signature, validate=True)
        except ValueError as exc:
            raise ValueError("supersession signature must be valid base64") from exc
        if len(signature) != 64:
            raise ValueError("Ed25519 supersession signature must contain 64 bytes")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "supersession_id": self.supersession_id,
            "run_id": self.run_id,
            "replacement_run_id": self.replacement_run_id,
            "manifest_identity_hash": self.manifest_identity_hash,
            "evidence_bundle_hash": self.evidence_bundle_hash,
            "approved_by": self.approved_by,
            "approved_at": self.approved_at,
            "limitations_acknowledged": list(self.limitations_acknowledged),
            "signer_key_id": self.signer_key_id,
            "signature_algorithm": self.signature_algorithm,
        }

    def signing_bytes(self) -> bytes:
        return _canonical_bytes(self.unsigned_dict())

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self.unsigned_dict(), "signature": self.signature}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HumanSupersessionRecord":
        _strict_fields(payload, _SUPERSESSION_FIELDS, "human supersession record")
        limitations = payload["limitations_acknowledged"]
        if not isinstance(limitations, list) or not all(isinstance(item, str) for item in limitations):
            raise TypeError("supersession limitations must be an array of strings")
        string_fields = _SUPERSESSION_FIELDS - {"schema_version", "limitations_acknowledged"}
        if type(payload["schema_version"]) is not int or any(
            not isinstance(payload[name], str) for name in string_fields
        ):
            raise TypeError("invalid human supersession field types")
        record = cls(
            supersession_id=payload["supersession_id"],
            run_id=payload["run_id"],
            replacement_run_id=payload["replacement_run_id"],
            manifest_identity_hash=payload["manifest_identity_hash"],
            evidence_bundle_hash=payload["evidence_bundle_hash"],
            approved_by=payload["approved_by"],
            approved_at=payload["approved_at"],
            limitations_acknowledged=tuple(limitations),
            signer_key_id=payload["signer_key_id"],
            signature=payload["signature"],
            signature_algorithm=payload["signature_algorithm"],
            schema_version=payload["schema_version"],
        )
        record.validate()
        return record
