from __future__ import annotations

import base64
import json
import math
import os
import stat
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping

from quant_lab.research._control_plane_contracts import (
    SCHEMA_VERSION,
    ApprovalError,
    HumanApprovalRecord,
    HumanSupersessionRecord,
    IntegrityError,
    REQUIRED_VALIDATION_GATES,
    ValidationArtifact,
    _POLICY_CALCULATOR_FIELDS,
    _POLICY_GATE_FIELDS,
    _POLICY_THRESHOLD_FIELDS,
    _VALIDATION_POLICY_FIELDS,
    _anchor_hashes,
    _canonical_bytes,
    _hash_bytes,
    _parse_utc,
    _require_sha256,
    _strict_fields,
    _validate_identifier,
)


_SIGNER_FIELDS = {
    "approved_by",
    "key_id",
    "role",
    "public_key_base64",
    "valid_from",
    "valid_until",
    "revoked_at",
}
_TRUST_FIELDS = {"schema_version", "signers"}
_RUNNER_SIGNER_FIELDS = {
    "runner_id",
    "key_id",
    "role",
    "public_key_base64",
    "valid_from",
    "valid_until",
    "revoked_at",
}
_RUNNER_TRUST_FIELDS = {"schema_version", "runners", "validation_policy"}


@dataclass(frozen=True)
class AuthorizedHumanSigner:
    approved_by: str
    key_id: str
    role: str
    public_key_base64: str
    valid_from: str
    valid_until: str
    revoked_at: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizedHumanSigner":
        _strict_fields(payload, _SIGNER_FIELDS, "authorized signer")
        string_fields = _SIGNER_FIELDS - {"revoked_at"}
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise TypeError("authorized signer textual fields must be strings")
        if payload["revoked_at"] is not None and not isinstance(payload["revoked_at"], str):
            raise TypeError("authorized signer revoked_at must be null or a string")
        signer = cls(
            approved_by=payload["approved_by"],
            key_id=payload["key_id"],
            role=payload["role"],
            public_key_base64=payload["public_key_base64"],
            valid_from=payload["valid_from"],
            valid_until=payload["valid_until"],
            revoked_at=payload["revoked_at"],
        )
        signer.validate()
        return signer

    def validate(self) -> None:
        values = (
            self.approved_by,
            self.key_id,
            self.role,
            self.public_key_base64,
            self.valid_from,
            self.valid_until,
        )
        if any(not isinstance(value, str) for value in values):
            raise TypeError("authorized signer textual fields must be strings")
        if self.revoked_at is not None and not isinstance(self.revoked_at, str):
            raise TypeError("authorized signer revoked_at must be null or a string")
        _validate_identifier("key_id", self.key_id)
        if self.role != "authorized_human_approver":
            raise ValueError("signer role must be authorized_human_approver")
        if not self.approved_by.strip() or self.approved_by.lower().startswith(("agent", "model", "service")):
            raise ValueError("authorized signer must be a named human identity")
        start = _parse_utc(self.valid_from, "valid_from")
        end = _parse_utc(self.valid_until, "valid_until")
        if end <= start:
            raise ValueError("signer valid_until must follow valid_from")
        if self.revoked_at is not None:
            _parse_utc(self.revoked_at, "revoked_at")
        try:
            key = base64.b64decode(self.public_key_base64, validate=True)
        except ValueError as exc:
            raise ValueError("public_key_base64 must be valid base64") from exc
        if len(key) != 32:
            raise ValueError("Ed25519 public key must contain 32 bytes")


@dataclass(frozen=True)
class AuthorizedValidationRunner:
    runner_id: str
    key_id: str
    role: str
    public_key_base64: str
    valid_from: str
    valid_until: str
    revoked_at: str | None = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizedValidationRunner":
        _strict_fields(payload, _RUNNER_SIGNER_FIELDS, "authorized validation runner")
        string_fields = _RUNNER_SIGNER_FIELDS - {"revoked_at"}
        if any(not isinstance(payload[name], str) for name in string_fields):
            raise TypeError("authorized validation runner textual fields must be strings")
        if payload["revoked_at"] is not None and not isinstance(payload["revoked_at"], str):
            raise TypeError("validation runner revoked_at must be null or a string")
        runner = cls(**payload)
        runner.validate()
        return runner

    def validate(self) -> None:
        _validate_identifier("runner_id", self.runner_id)
        _validate_identifier("key_id", self.key_id)
        if self.role != "deterministic_validation_runner":
            raise ValueError("validation runner role must be deterministic_validation_runner")
        start = _parse_utc(self.valid_from, "valid_from")
        end = _parse_utc(self.valid_until, "valid_until")
        if end <= start:
            raise ValueError("validation runner valid_until must follow valid_from")
        if self.revoked_at is not None:
            _parse_utc(self.revoked_at, "revoked_at")
        try:
            key = base64.b64decode(self.public_key_base64, validate=True)
        except ValueError as exc:
            raise ValueError("validation runner public key must be valid base64") from exc
        if len(key) != 32:
            raise ValueError("validation runner Ed25519 public key must contain 32 bytes")


class ValidationRunnerTrustStore:
    """Root-anchored allowlist for external deterministic validation runners."""

    ADMIN_ANCHOR_PATH = Path(
        "/Library/Application Support/QuantLab/validation-runner-trust.sha256"
    )

    @classmethod
    def _read_admin_anchor(cls) -> frozenset[str]:
        anchor = cls.ADMIN_ANCHOR_PATH
        if anchor.is_symlink() or not anchor.is_file():
            raise IntegrityError(
                "root-owned validation-runner trust anchor is not provisioned; validation remains disabled"
            )
        details = anchor.stat()
        if details.st_uid != 0 or stat.S_IMODE(details.st_mode) & 0o022:
            raise IntegrityError(
                "validation-runner trust anchor must be root-owned and not group/world writable"
            )
        return _anchor_hashes(anchor.read_text(encoding="ascii"))

    def __init__(self, path: str | Path) -> None:
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise ValueError("validation-runner trust store cannot be a symlink")
        source = requested.resolve()
        if not source.is_file():
            raise ValueError("validation-runner trust store must be a regular file")
        details = source.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o222:
            raise PermissionError(
                "validation-runner trust store must be current-user-owned and read-only"
            )
        raw = source.read_bytes()
        actual_hash = _hash_bytes(raw)
        if actual_hash not in _anchor_hashes(self._read_admin_anchor()):
            raise IntegrityError(
                "validation-runner trust store does not match the root-owned trust anchor"
            )
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise TypeError("validation-runner trust store must contain an object")
        _strict_fields(payload, _RUNNER_TRUST_FIELDS, "validation-runner trust store")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("invalid validation-runner trust-store schema_version")
        if not isinstance(payload["runners"], list):
            raise TypeError("validation-runner trust store runners must be an array")
        if not isinstance(payload["validation_policy"], Mapping):
            raise TypeError("validation-runner trust store validation_policy must be an object")
        runners = [
            AuthorizedValidationRunner.from_dict(item)
            for item in payload["runners"]
            if isinstance(item, Mapping)
        ]
        if len(runners) != len(payload["runners"]):
            raise TypeError("every validation runner must be an object")
        if not runners or len({item.key_id for item in runners}) != len(runners):
            raise ValueError("validation-runner keys must be nonempty and unique")
        self.source_path = source
        self.source_hash = actual_hash
        self.source_bytes = raw
        self._runners = {item.key_id: item for item in runners}
        self._validation_policy = self._parse_validation_policy(payload["validation_policy"])

    @staticmethod
    def _parse_validation_policy(payload: Mapping[str, Any]) -> dict[str, Any]:
        _strict_fields(payload, _VALIDATION_POLICY_FIELDS, "validation policy")
        if not isinstance(payload["policy_id"], str):
            raise TypeError("validation policy policy_id must be a string")
        _validate_identifier("validation policy policy_id", payload["policy_id"])
        if not isinstance(payload["gates"], Mapping):
            raise TypeError("validation policy gates must be an object")
        if set(payload["gates"]) != set(REQUIRED_VALIDATION_GATES):
            raise ValueError("validation policy must define the exact deterministic gate set")
        parsed: dict[str, Any] = {"policy_id": payload["policy_id"], "gates": {}}
        for gate_id in REQUIRED_VALIDATION_GATES:
            gate = payload["gates"][gate_id]
            if not isinstance(gate, Mapping):
                raise TypeError(f"validation policy gate must be an object: {gate_id}")
            _strict_fields(gate, _POLICY_GATE_FIELDS, f"validation policy gate {gate_id}")
            calculators = gate["allowed_calculators"]
            thresholds = gate["thresholds"]
            if not isinstance(calculators, list) or not calculators:
                raise ValueError(f"validation policy calculators must be a nonempty array: {gate_id}")
            parsed_calculators: list[dict[str, str]] = []
            for calculator in calculators:
                if not isinstance(calculator, Mapping):
                    raise TypeError(f"validation policy calculator must be an object: {gate_id}")
                _strict_fields(
                    calculator,
                    _POLICY_CALCULATOR_FIELDS,
                    f"validation policy calculator {gate_id}",
                )
                if any(not isinstance(calculator[name], str) for name in _POLICY_CALCULATOR_FIELDS):
                    raise TypeError(f"validation policy calculator fields must be strings: {gate_id}")
                _validate_identifier("calculator_id", calculator["calculator_id"])
                _validate_identifier("calculator_version", calculator["calculator_version"])
                _require_sha256(
                    "calculator_source_sha256", calculator["calculator_source_sha256"]
                )
                parsed_calculators.append(dict(calculator))
            if len(
                {
                    (
                        item["calculator_id"],
                        item["calculator_version"],
                        item["calculator_source_sha256"],
                    )
                    for item in parsed_calculators
                }
            ) != len(parsed_calculators):
                raise ValueError(f"validation policy calculator allowlist has duplicates: {gate_id}")
            if not isinstance(thresholds, Mapping):
                raise TypeError(f"validation policy thresholds must be an object: {gate_id}")
            _strict_fields(
                thresholds,
                _POLICY_THRESHOLD_FIELDS[gate_id],
                f"validation policy thresholds {gate_id}",
            )
            _canonical_bytes(dict(thresholds))
            parsed["gates"][gate_id] = {
                "allowed_calculators": parsed_calculators,
                "thresholds": dict(thresholds),
            }
        return parsed

    @classmethod
    def from_file(cls, path: str | Path) -> "ValidationRunnerTrustStore":
        return cls(path)

    def assert_external_and_unchanged(self, vault_root: Path) -> None:
        try:
            self.source_path.relative_to(vault_root.resolve())
        except ValueError:
            pass
        else:
            raise IntegrityError(
                "validation-runner trust store must live outside the agent-controlled vault"
            )
        if self.source_path.is_symlink() or not self.source_path.is_file():
            raise IntegrityError("validation-runner trust store disappeared or became a symlink")
        details = self.source_path.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o222:
            raise IntegrityError("validation-runner trust store ownership or mode changed")
        if _hash_bytes(self.source_path.read_bytes()) != self.source_hash:
            raise IntegrityError("validation-runner trust store changed after loading")
        if self.source_hash not in _anchor_hashes(self._read_admin_anchor()):
            raise IntegrityError("validation-runner policy is absent from the root-owned registry")

    def verify(self, artifact: ValidationArtifact) -> None:
        artifact.validate()
        runner = self._runners.get(artifact.runner_key_id)
        if runner is None or runner.runner_id != artifact.runner_id:
            raise IntegrityError("validation artifact runner is not authorized")
        generated_at = _parse_utc(artifact.generated_at, "generated_at")
        if not (
            _parse_utc(runner.valid_from, "valid_from")
            <= generated_at
            <= _parse_utc(runner.valid_until, "valid_until")
        ):
            raise IntegrityError("validation artifact falls outside the runner key validity window")
        if runner.revoked_at is not None and generated_at >= _parse_utc(runner.revoked_at, "revoked_at"):
            raise IntegrityError("validation runner key was revoked at artifact generation time")
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise IntegrityError("cryptography is required to verify validation-runner attestations") from exc
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(runner.public_key_base64)
        )
        try:
            public_key.verify(
                base64.b64decode(artifact.runner_signature),
                artifact.signing_bytes(),
            )
        except InvalidSignature as exc:
            raise IntegrityError("invalid validation-runner attestation") from exc
        self._verify_validation_policy(artifact)

    def _verify_validation_policy(self, artifact: ValidationArtifact) -> None:
        gate_policy = self._validation_policy["gates"][artifact.gate_id]
        identity = (
            artifact.calculator_id,
            artifact.calculator_version,
            artifact.calculator_source_sha256,
        )
        allowed = {
            (
                item["calculator_id"],
                item["calculator_version"],
                item["calculator_source_sha256"],
            )
            for item in gate_policy["allowed_calculators"]
        }
        if identity not in allowed:
            raise IntegrityError(
                f"validation calculator is not preregistered for gate: {artifact.gate_id}"
            )
        metrics = artifact.metrics
        policy = gate_policy["thresholds"]

        def number(name: str) -> float:
            value = policy[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise IntegrityError(f"validation policy threshold must be finite: {name}")
            return float(value)

        gate = artifact.gate_id
        compliant = True
        if gate == "point_in_time_data_audit":
            compliant = metrics["row_count"] >= number("minimum_row_count")
        elif gate == "purged_embargoed_walk_forward_splits":
            compliant = (
                metrics["fold_count"] >= number("minimum_fold_count")
                and metrics["purge_observations"] >= number("minimum_purge_observations")
                and metrics["embargo_observations"] >= number("minimum_embargo_observations")
            )
        elif gate == "cpcv_pbo":
            compliant = (
                metrics["path_count"] >= number("minimum_path_count")
                and metrics["pbo_limit"] == number("maximum_pbo")
            )
        elif gate == "dsr_psr":
            compliant = metrics["minimum_probability"] == number("minimum_probability")
        elif gate == "multiple_testing_fdr":
            methods = policy["allowed_methods"]
            if not isinstance(methods, list) or not methods or not all(isinstance(item, str) for item in methods):
                raise IntegrityError("validation policy allowed_methods must be a nonempty string array")
            compliant = metrics["alpha"] == number("alpha") and metrics["method"] in methods
        elif gate == "cost_sensitivity":
            compliant = (
                metrics["stressed_cost_bps"]
                >= number("minimum_stress_multiplier") * metrics["base_cost_bps"]
                and metrics["minimum_net_sharpe"] == number("minimum_stressed_net_sharpe")
            )
        elif gate == "drawdown_turnover":
            compliant = (
                metrics["max_drawdown_limit"] == number("maximum_drawdown")
                and metrics["turnover_limit"] == number("maximum_annual_turnover")
            )
        elif gate == "liquidity_capacity":
            compliant = (
                metrics["adv_participation_limit"] == number("maximum_adv_participation")
                and metrics["capacity_aum"] >= number("minimum_capacity_aum")
            )
        elif gate == "regime_stability":
            compliant = (
                metrics["regime_count"] >= number("minimum_regime_count")
                and metrics["minimum_observations_per_regime"]
                >= number("minimum_observations_per_regime")
            )
        elif gate == "sealed_holdout":
            compliant = (
                metrics["observation_count"] >= number("minimum_observation_count")
                and metrics["release_count"] == number("required_release_count")
            )
        elif gate == "code_reproducibility":
            compliant = metrics["rerun_count"] >= number("minimum_rerun_count")
        if not compliant:
            raise IntegrityError(
                f"validation artifact violates preregistered policy: {artifact.gate_id}"
            )


class ApprovalTrustStore:
    """Read-only signer allowlist loaded from outside the agent-controlled run tree."""

    ADMIN_ANCHOR_PATH = Path("/Library/Application Support/QuantLab/approval-trust.sha256")

    @classmethod
    def _read_admin_anchor(cls) -> frozenset[str]:
        anchor = cls.ADMIN_ANCHOR_PATH
        if anchor.is_symlink() or not anchor.is_file():
            raise ApprovalError(
                "root-owned approval trust anchor is not provisioned; human approval remains disabled"
            )
        details = anchor.stat()
        if details.st_uid != 0 or stat.S_IMODE(details.st_mode) & 0o022:
            raise ApprovalError("approval trust anchor must be root-owned and not group/world writable")
        try:
            return _anchor_hashes(anchor.read_text(encoding="ascii"))
        except IntegrityError as exc:
            raise ApprovalError(str(exc)) from exc

    def __init__(self, path: str | Path) -> None:
        requested_source = Path(path).expanduser()
        if requested_source.is_symlink():
            raise ValueError("approval trust store must be a regular non-symlink file")
        source = requested_source.resolve()
        if not source.is_file():
            raise ValueError("approval trust store must be a regular non-symlink file")
        details = source.stat()
        if details.st_uid != os.getuid():
            raise PermissionError("approval trust store must be owned by the current human account")
        mode = stat.S_IMODE(details.st_mode)
        if mode & 0o222:
            raise PermissionError("approval trust store must be read-only before it can authorize decisions")
        raw = source.read_bytes()
        actual_hash = _hash_bytes(raw)
        if actual_hash not in _anchor_hashes(self._read_admin_anchor()):
            raise ApprovalError("approval trust store does not match the root-owned trust anchor")
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise TypeError("approval trust store must contain an object")
        _strict_fields(payload, _TRUST_FIELDS, "approval trust store")
        if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
            raise ValueError("invalid approval trust store schema_version")
        if not isinstance(payload["signers"], list):
            raise TypeError("approval trust store signers must be an array")
        signers = [
            AuthorizedHumanSigner.from_dict(item)
            for item in payload["signers"]
            if isinstance(item, Mapping)
        ]
        if len(signers) != len(payload["signers"]):
            raise TypeError("every approval signer must be an object")
        if len({item.key_id for item in signers}) != len(signers):
            raise ValueError("approval signer key IDs must be unique")
        self.source_path = source
        self.source_hash = actual_hash
        self._signers = {item.key_id: item for item in signers}
        if not self._signers:
            raise ValueError("approval trust store must contain at least one signer")

    @classmethod
    def from_file(cls, path: str | Path) -> "ApprovalTrustStore":
        return cls(path)

    def assert_external_and_unchanged(self, vault_root: Path) -> None:
        try:
            self.source_path.relative_to(vault_root.resolve())
        except ValueError:
            pass
        else:
            raise ApprovalError("approval trust store must live outside the agent-controlled vault")
        self._assert_unchanged()

    def _assert_unchanged(self) -> None:
        if self.source_path.is_symlink() or not self.source_path.is_file():
            raise ApprovalError("approval trust store disappeared or became a symlink")
        details = self.source_path.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o222:
            raise ApprovalError("approval trust store ownership or read-only mode changed")
        if _hash_bytes(self.source_path.read_bytes()) != self.source_hash:
            raise ApprovalError("approval trust store changed after it was loaded")
        if self.source_hash not in _anchor_hashes(self._read_admin_anchor()):
            raise ApprovalError("approval policy is absent from the root-owned registry")

    def verify(self, record: HumanApprovalRecord) -> None:
        self._assert_unchanged()
        record.validate()
        signer = self._signers.get(record.signer_key_id)
        if signer is None:
            raise ApprovalError("approval signer key is not authorized")
        if signer.approved_by != record.approved_by:
            raise ApprovalError("approval identity does not match the authorized key")
        approved_at = _parse_utc(record.approved_at, "approved_at")
        if approved_at > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ApprovalError("approval timestamp is unreasonably far in the future")
        if not (_parse_utc(signer.valid_from, "valid_from") <= approved_at <= _parse_utc(signer.valid_until, "valid_until")):
            raise ApprovalError("approval signature falls outside the key validity window")
        if signer.revoked_at is not None and approved_at >= _parse_utc(signer.revoked_at, "revoked_at"):
            raise ApprovalError("approval key was revoked at the decision time")
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise ApprovalError("cryptography is required to verify Ed25519 approvals") from exc
        public_key = Ed25519PublicKey.from_public_bytes(base64.b64decode(signer.public_key_base64))
        try:
            public_key.verify(base64.b64decode(record.signature), record.signing_bytes())
        except InvalidSignature as exc:
            raise ApprovalError("invalid human approval signature") from exc

    def verify_supersession(self, record: HumanSupersessionRecord) -> None:
        self._assert_unchanged()
        record.validate()
        signer = self._signers.get(record.signer_key_id)
        if signer is None or signer.approved_by != record.approved_by:
            raise ApprovalError("supersession signer is not authorized for this human identity")
        approved_at = _parse_utc(record.approved_at, "approved_at")
        if approved_at > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise ApprovalError("supersession timestamp is unreasonably far in the future")
        if not (
            _parse_utc(signer.valid_from, "valid_from")
            <= approved_at
            <= _parse_utc(signer.valid_until, "valid_until")
        ):
            raise ApprovalError("supersession falls outside the key validity window")
        if signer.revoked_at is not None and approved_at >= _parse_utc(signer.revoked_at, "revoked_at"):
            raise ApprovalError("supersession key was revoked at the decision time")
        try:
            from cryptography.exceptions import InvalidSignature
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        except ImportError as exc:
            raise ApprovalError("cryptography is required to verify Ed25519 supersessions") from exc
        public_key = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(signer.public_key_base64)
        )
        try:
            public_key.verify(
                base64.b64decode(record.signature),
                record.signing_bytes(),
            )
        except InvalidSignature as exc:
            raise ApprovalError("invalid human supersession signature") from exc
