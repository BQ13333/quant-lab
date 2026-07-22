"""Root-anchored, file-backed admission of LSEG data for governed reruns.

This module is deliberately offline.  It resolves immutable local artifacts,
recomputes their hashes, verifies Ed25519 attestations, validates every admitted
partition, and returns a sealed typed object.  No provider callback is reachable
through an in-memory mapping or an unverified hexadecimal placeholder.
"""

from __future__ import annotations

import base64
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import stat
import subprocess
import tempfile
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from quant_lab.research._control_plane_contracts import (
    IntegrityError,
    _anchor_hashes,
    _canonical_bytes,
    _parse_utc,
    _require_sha256,
    _strict_fields,
    _validate_identifier,
)


SCHEMA_VERSION = 1
_SAFE_RELATIVE = re.compile(r"^[^\x00]+$")
_GIT_COMMIT = re.compile(r"^[0-9a-f]{40}(?:[0-9a-f]{24})?$")
_MIC = re.compile(r"^[A-Z0-9]{4}$")
_SIGNER_ROLES = {
    "authorized_entitlement_verifier",
    "authorized_data_manifest_signer",
    "authorized_execution_contract_signer",
    "authorized_data_release_approver",
}
_REF_FIELDS = {"path", "byte_size", "sha256"}
_SIGNER_FIELDS = {
    "identity",
    "key_id",
    "role",
    "public_key_base64",
    "valid_from",
    "valid_until",
    "revoked_at",
}
_TRUST_FIELDS = {"schema_version", "signers"}
_ENTITLEMENT_FIELDS = {
    "schema_version", "attestation_id", "provider", "customer_legal_entity",
    "entitled_account_id", "requester_pays_account_id", "aws_region",
    "dataset_product_id", "delivery_method", "trial_id", "permitted_use",
    "permitted_markets", "redistribution_allowed", "data_location_constraints",
    "valid_from", "valid_until", "checked_at", "source_evidence_ref",
    "plan_sha256", "legacy_anchor_run_id", "reserved_root_run_id", "attestor_id",
    "signer_key_id", "trust_policy_sha256", "signature_algorithm", "signature",
}
_DATA_MANIFEST_FIELDS = {
    "schema_version", "manifest_id", "provider", "dataset_product_id",
    "entitled_account_id", "requester_pays_account_id", "aws_region",
    "entitlement_attestation_sha256", "plan_sha256", "legacy_anchor_run_id",
    "reserved_root_run_id", "generated_at", "data_asof_utc", "requested_start",
    "requested_end", "markets", "fields", "storage_format", "schema_sha256",
    "inventory_ref", "profiling_ref", "calendar_ref", "partition_root", "content_root_sha256",
    "partition_count", "row_count", "total_bytes", "point_in_time",
    "includes_inactive_and_delisted", "corporate_actions_complete",
    "tradability_and_liquidity_complete", "timestamp_lineage_complete",
    "revision_vintages_retained", "sealed_out_of_sample_period",
    "coverage_metrics", "calendar_policy_sha256", "timezone_policy_sha256",
    "corporate_action_policy_sha256", "license_classification",
    "admission_runner_id", "signer_key_id", "trust_policy_sha256",
    "signature_algorithm", "signature",
}
_EXECUTION_FIELDS = {
    "schema_version", "contract_id", "plan_sha256", "legacy_anchor_run_id",
    "reserved_root_run_id", "source_bundle_sha256", "code_commit", "clean_tree",
    "code_snapshot_ref", "validation_policy_ref", "research_config_ref",
    "feature_spec_sha256", "split_policy_sha256",
    "validation_policy_sha256", "research_config_sha256", "required_markets",
    "required_fields", "required_start", "required_end", "required_coverage",
    "created_at", "signer_id", "signer_key_id", "trust_policy_sha256",
    "signature_algorithm", "signature",
}
_RELEASE_FIELDS = {
    "schema_version", "release_id", "decision", "admission_subject_sha256",
    "plan_sha256", "entitlement_attestation_sha256", "admitted_data_manifest_sha256",
    "execution_contract_sha256", "legacy_anchor_run_id", "reserved_root_run_id",
    "approved_by", "approved_at", "expires_at", "limitations_acknowledged",
    "signer_key_id", "trust_policy_sha256", "signature_algorithm", "signature",
}
_BUNDLE_FIELDS = {
    "schema_version", "admission_id", "plan_ref", "entitlement_ref",
    "data_manifest_ref", "execution_contract_ref", "human_release_ref", "created_at",
}
_INVENTORY_FIELDS = {
    "relative_path", "byte_size", "sha256", "row_count", "market",
    "min_effective_at", "max_effective_at", "min_published_at", "max_published_at",
    "min_ingested_at", "max_ingested_at", "schema_sha256",
}
_PROFILE_FIELDS = {
    "schema_version", "manifest_id", "markets", "fields", "coverage_metrics",
    "flags", "row_count", "partition_count", "generated_at", "runner_id",
}
_CODE_SNAPSHOT_FIELDS = {
    "schema_version", "code_commit", "git_tree", "source_bundle_sha256", "sources",
}
_CALENDAR_FIELDS = {"schema_version", "markets"}
_ADMISSION_ARCHIVE_FIELDS = {
    "schema_version", "verified_admission", "bundle_base64", "entitlement_base64",
    "data_manifest_base64", "execution_contract_base64", "human_release_base64",
}


class DataAdmissionError(IntegrityError):
    """Raised before execution when an offline admission invariant fails."""


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _semantic_hash(value: Mapping[str, Any] | Sequence[Any]) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    )


def frozen_validation_policy_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow_stages": plan["workflow_stages"],
        "validation_gates": plan["deterministic_validation_gates"],
        "approval_policy": plan["approval_policy"],
    }


def frozen_research_config_payload(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "research_contract": plan["research_contract"],
        "data_contract": plan["data_contract"],
        "required_frozen_controls": plan["required_frozen_controls"],
        "activation_policy": plan["activation_policy"],
    }


def _require_string(payload: Mapping[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise DataAdmissionError(f"{field} must be a nonempty string")
    return value


def _require_exact_string_list(payload: Mapping[str, Any], field: str) -> tuple[str, ...]:
    value = payload.get(field)
    if not isinstance(value, list) or not value or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise DataAdmissionError(f"{field} must be a nonempty string array")
    if len(value) != len(set(value)):
        raise DataAdmissionError(f"{field} contains duplicate values")
    return tuple(value)


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataAdmissionError(f"{field} must be an integer >= {minimum}")
    return value


def _require_finite(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataAdmissionError(f"{field} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise DataAdmissionError(f"{field} must be finite")
    return parsed


def _strict_utc(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise DataAdmissionError(f"{field} must be an ISO-8601 UTC timestamp")
    try:
        return _parse_utc(value, field)
    except (TypeError, ValueError) as exc:
        raise DataAdmissionError(str(exc)) from exc


def _strict_date(value: Any, field: str) -> str:
    if not isinstance(value, str):
        raise DataAdmissionError(f"{field} must be YYYY-MM-DD")
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise DataAdmissionError(f"{field} must be YYYY-MM-DD") from exc
    if parsed.strftime("%Y-%m-%d") != value:
        raise DataAdmissionError(f"{field} must be canonical YYYY-MM-DD")
    return value


def _safe_relative(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value or not _SAFE_RELATIVE.fullmatch(value):
        raise DataAdmissionError(f"{field} must be a safe relative path")
    path = Path(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise DataAdmissionError(f"{field} must remain below its configured root")
    return path


@dataclass(frozen=True)
class ArtifactRef:
    path: str
    byte_size: int
    sha256: str

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], field: str) -> "ArtifactRef":
        try:
            _strict_fields(payload, _REF_FIELDS, field)
            path = _safe_relative(payload["path"], f"{field}.path").as_posix()
            size = _require_int(payload["byte_size"], f"{field}.byte_size")
            if not isinstance(payload["sha256"], str):
                raise DataAdmissionError(f"{field}.sha256 must be a string")
            _require_sha256(f"{field}.sha256", payload["sha256"])
        except (TypeError, ValueError, IntegrityError) as exc:
            if isinstance(exc, DataAdmissionError):
                raise
            raise DataAdmissionError(str(exc)) from exc
        return cls(path, size, payload["sha256"])

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "byte_size": self.byte_size, "sha256": self.sha256}


def _assert_directory_custody(root: Path, path: Path) -> None:
    root = root.resolve()
    try:
        path.resolve().relative_to(root)
    except ValueError as exc:
        raise DataAdmissionError("artifact escaped the administrator-controlled admission root") from exc
    cursor = path.parent
    while True:
        if cursor.is_symlink() or not cursor.is_dir():
            raise DataAdmissionError("admission artifact parent must be a regular directory")
        details = cursor.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o022:
            raise DataAdmissionError("admission directories must be current-user-owned and not group/world writable")
        if cursor.resolve() == root:
            return
        if cursor == cursor.parent:
            raise DataAdmissionError("admission artifact parent escaped its root")
        cursor = cursor.parent


@contextmanager
def open_immutable_ref(
    root: Path, reference: ArtifactRef, field: str
) -> Iterator[BinaryIO]:
    """Open and hash-check a read-only artifact without loading it into memory."""
    root = root.expanduser().resolve()
    if root.is_symlink() or not root.is_dir():
        raise DataAdmissionError("admission root must be a regular non-symlink directory")
    _assert_directory_custody(root, root / reference.path)
    path = root / reference.path
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DataAdmissionError(f"cannot open immutable artifact {field}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_uid != os.getuid():
            raise DataAdmissionError(f"{field} must be a current-user-owned regular file")
        if stat.S_IMODE(before.st_mode) & 0o222 or before.st_nlink != 1:
            raise DataAdmissionError(f"{field} must be read-only and have exactly one hard link")
        digest = hashlib.sha256()
        byte_size = 0
        while block := os.read(descriptor, 1024 * 1024):
            digest.update(block)
            byte_size += len(block)
        after = os.fstat(descriptor)
        identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        if identity != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise DataAdmissionError(f"{field} changed while it was being verified")
        if byte_size != reference.byte_size or digest.hexdigest() != reference.sha256:
            raise DataAdmissionError(f"{field} size or SHA-256 does not match its signed reference")
        os.lseek(descriptor, 0, os.SEEK_SET)
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            yield handle
        final = os.fstat(descriptor)
        if identity != (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns):
            raise DataAdmissionError(f"{field} changed while it was open")
    finally:
        os.close(descriptor)


def read_immutable_ref(root: Path, reference: ArtifactRef, field: str) -> bytes:
    with open_immutable_ref(root, reference, field) as handle:
        return handle.read()


def canonical_arrow_schema_sha256(schema: pa.Schema) -> str:
    """Return the stable hash used by signed manifests and inventory rows."""
    return _sha256_bytes(schema.remove_metadata().serialize().to_pybytes())


@dataclass(frozen=True)
class AuthorizedDataSigner:
    identity: str
    key_id: str
    role: str
    public_key_base64: str
    valid_from: str
    valid_until: str
    revoked_at: str | None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuthorizedDataSigner":
        _strict_fields(payload, _SIGNER_FIELDS, "authorized data signer")
        signer = cls(**payload)
        signer.validate()
        return signer

    def validate(self) -> None:
        if not self.identity.strip() or self.identity.lower().startswith(("agent", "model", "service")):
            raise DataAdmissionError("data-admission signer must be a named human or registered verifier")
        _validate_identifier("key_id", self.key_id)
        if self.role not in _SIGNER_ROLES:
            raise DataAdmissionError("data-admission signer role is not authorized")
        start = _strict_utc(self.valid_from, "valid_from")
        end = _strict_utc(self.valid_until, "valid_until")
        if end <= start:
            raise DataAdmissionError("signer validity window is empty")
        if self.revoked_at is not None:
            _strict_utc(self.revoked_at, "revoked_at")
        try:
            key = base64.b64decode(self.public_key_base64, validate=True)
        except Exception as exc:
            raise DataAdmissionError("signer public key must be valid base64") from exc
        if len(key) != 32:
            raise DataAdmissionError("signer public key must contain 32 Ed25519 bytes")


class _RootAnchoredTrustStore:
    ADMIN_ANCHOR_PATH: Path
    ALLOWED_ROLES: frozenset[str]

    @classmethod
    def _read_admin_anchor(cls) -> frozenset[str]:
        anchor = cls.ADMIN_ANCHOR_PATH
        if anchor.is_symlink() or not anchor.is_file():
            raise DataAdmissionError("root-owned data-admission trust anchor is not provisioned")
        details = anchor.stat()
        if details.st_uid != 0 or stat.S_IMODE(details.st_mode) & 0o022:
            raise DataAdmissionError("data-admission trust anchor must be root-owned and protected")
        return _anchor_hashes(anchor.read_text(encoding="ascii"))

    def __init__(self, path: str | Path) -> None:
        requested = Path(path).expanduser()
        if requested.is_symlink():
            raise DataAdmissionError("trust store cannot be a symlink")
        source = requested.resolve()
        if not source.is_file():
            raise DataAdmissionError("trust store must be a regular file")
        details = source.stat()
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o222:
            raise DataAdmissionError("trust store must be current-user-owned and read-only")
        raw = source.read_bytes()
        digest = _sha256_bytes(raw)
        if digest not in _anchor_hashes(self._read_admin_anchor()):
            raise DataAdmissionError("trust store is absent from the root-owned trust registry")
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise DataAdmissionError("trust store must contain an object")
        _strict_fields(payload, _TRUST_FIELDS, "data-admission trust store")
        if payload["schema_version"] != SCHEMA_VERSION or type(payload["schema_version"]) is not int:
            raise DataAdmissionError("unsupported trust-store schema")
        if not isinstance(payload["signers"], list) or not payload["signers"]:
            raise DataAdmissionError("trust store must contain signers")
        signers = [AuthorizedDataSigner.from_dict(item) for item in payload["signers"]]
        if len({item.key_id for item in signers}) != len(signers):
            raise DataAdmissionError("trust-store key IDs must be unique")
        if any(item.role not in self.ALLOWED_ROLES for item in signers):
            raise DataAdmissionError("trust store contains a signer for the wrong authority domain")
        self.source_path = source
        self.source_hash = digest
        self.source_bytes = raw
        self._signers = {item.key_id: item for item in signers}

    def assert_external_and_unchanged(self, protected_root: Path) -> None:
        try:
            self.source_path.relative_to(protected_root.resolve())
        except ValueError:
            pass
        else:
            raise DataAdmissionError("trust store must live outside the agent-controlled vault")
        details = self.source_path.stat()
        if self.source_path.is_symlink() or not self.source_path.is_file():
            raise DataAdmissionError("trust store disappeared or became a symlink")
        if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o222:
            raise DataAdmissionError("trust-store custody changed")
        if _sha256_bytes(self.source_path.read_bytes()) != self.source_hash:
            raise DataAdmissionError("trust store changed after loading")
        if self.source_hash not in _anchor_hashes(self._read_admin_anchor()):
            raise DataAdmissionError("trust store is no longer root-anchored")

    def verify(
        self,
        payload: Mapping[str, Any],
        *,
        required_role: str,
        identity_field: str,
        signed_at_field: str,
    ) -> None:
        if payload.get("trust_policy_sha256") != self.source_hash:
            raise DataAdmissionError("signed artifact trust-policy binding is wrong")
        if payload.get("signature_algorithm") != "Ed25519":
            raise DataAdmissionError("only Ed25519 signatures are accepted")
        key_id = payload.get("signer_key_id")
        signer = self._signers.get(key_id)
        if signer is None or signer.role != required_role:
            raise DataAdmissionError("signer key or authority role is not authorized")
        if payload.get(identity_field) != signer.identity:
            raise DataAdmissionError("signed identity does not match its authorized key")
        signed_at = _strict_utc(payload.get(signed_at_field), signed_at_field)
        if signed_at > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise DataAdmissionError("signed timestamp is unreasonably far in the future")
        if not (_strict_utc(signer.valid_from, "valid_from") <= signed_at <= _strict_utc(signer.valid_until, "valid_until")):
            raise DataAdmissionError("signature falls outside the key validity window")
        if signer.revoked_at is not None and signed_at >= _strict_utc(signer.revoked_at, "revoked_at"):
            raise DataAdmissionError("signer key was revoked")
        signature = payload.get("signature")
        try:
            signature_bytes = base64.b64decode(signature, validate=True)
        except Exception as exc:
            raise DataAdmissionError("signature must be valid base64") from exc
        if len(signature_bytes) != 64:
            raise DataAdmissionError("Ed25519 signature must contain 64 bytes")
        unsigned = dict(payload)
        unsigned.pop("signature")
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

            Ed25519PublicKey.from_public_bytes(
                base64.b64decode(signer.public_key_base64, validate=True)
            ).verify(signature_bytes, _canonical_bytes(unsigned))
        except Exception as exc:
            raise DataAdmissionError("signed artifact signature verification failed") from exc


class DataAdmissionTrustStore(_RootAnchoredTrustStore):
    ADMIN_ANCHOR_PATH = Path(
        "/Library/Application Support/QuantLab/data-admission-trust.sha256"
    )
    ALLOWED_ROLES = frozenset(
        {
            "authorized_entitlement_verifier",
            "authorized_data_manifest_signer",
            "authorized_execution_contract_signer",
        }
    )


class DataReleaseTrustStore(_RootAnchoredTrustStore):
    ADMIN_ANCHOR_PATH = Path(
        "/Library/Application Support/QuantLab/data-release-trust.sha256"
    )
    ALLOWED_ROLES = frozenset({"authorized_data_release_approver"})


@dataclass(frozen=True)
class VerifiedDataAdmission:
    admission_id: str
    admission_bundle_sha256: str
    admission_subject_sha256: str
    plan_sha256: str
    legacy_anchor_run_id: str
    root_run_id: str
    dataset_product_id: str
    data_manifest_id: str
    data_asof_utc: str
    feature_manifest_hash: str
    split_manifest_id: str
    split_manifest_hash: str
    validation_policy_hash: str
    code_commit: str
    entitlement_attestation_sha256: str
    data_manifest_sha256: str
    execution_contract_sha256: str
    human_release_sha256: str
    admission_trust_policy_sha256: str
    release_trust_policy_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def archive_verified_data_admission(
    *,
    admission_root: Path,
    bundle_path: Path,
    verified: VerifiedDataAdmission,
) -> dict[str, Any]:
    """Seal the original signed components for later offline authority replay."""
    bundle, bundle_bytes = _load_bundle(admission_root.resolve(), bundle_path)
    refs = {
        name: ArtifactRef.from_dict(bundle[name], name)
        for name in (
            "entitlement_ref", "data_manifest_ref", "execution_contract_ref",
            "human_release_ref",
        )
    }
    raw = {
        name: read_immutable_ref(admission_root, reference, name)
        for name, reference in refs.items()
    }
    if (
        _sha256_bytes(bundle_bytes) != verified.admission_bundle_sha256
        or refs["entitlement_ref"].sha256 != verified.entitlement_attestation_sha256
        or refs["data_manifest_ref"].sha256 != verified.data_manifest_sha256
        or refs["execution_contract_ref"].sha256 != verified.execution_contract_sha256
        or refs["human_release_ref"].sha256 != verified.human_release_sha256
    ):
        raise DataAdmissionError("archived admission differs from the verified receipt")
    return {
        "schema_version": SCHEMA_VERSION,
        "verified_admission": verified.to_dict(),
        "bundle_base64": base64.b64encode(bundle_bytes).decode("ascii"),
        "entitlement_base64": base64.b64encode(raw["entitlement_ref"]).decode("ascii"),
        "data_manifest_base64": base64.b64encode(raw["data_manifest_ref"]).decode("ascii"),
        "execution_contract_base64": base64.b64encode(raw["execution_contract_ref"]).decode("ascii"),
        "human_release_base64": base64.b64encode(raw["human_release_ref"]).decode("ascii"),
    }


def verify_archived_data_admission(
    archive: Mapping[str, Any],
    *,
    admission_trust_store: DataAdmissionTrustStore,
    release_trust_store: DataReleaseTrustStore,
    vault_root: Path,
) -> VerifiedDataAdmission:
    """Cryptographically replay an archived signed admission without source data."""
    _strict_fields(archive, _ADMISSION_ARCHIVE_FIELDS, "archived data admission")
    if archive["schema_version"] != SCHEMA_VERSION:
        raise DataAdmissionError("unsupported archived data-admission schema")
    receipt_payload = archive["verified_admission"]
    if not isinstance(receipt_payload, Mapping) or set(receipt_payload) != set(
        VerifiedDataAdmission.__dataclass_fields__
    ):
        raise DataAdmissionError("archived verified-admission receipt has an invalid schema")
    receipt = VerifiedDataAdmission(**receipt_payload)

    def decode(field: str) -> bytes:
        try:
            return base64.b64decode(archive[field], validate=True)
        except (TypeError, ValueError) as exc:
            raise DataAdmissionError(f"{field} is not valid base64") from exc

    bundle_bytes = decode("bundle_base64")
    entitlement_bytes = decode("entitlement_base64")
    manifest_bytes = decode("data_manifest_base64")
    execution_bytes = decode("execution_contract_base64")
    release_bytes = decode("human_release_base64")
    if _sha256_bytes(bundle_bytes) != receipt.admission_bundle_sha256:
        raise DataAdmissionError("archived admission bundle hash differs from its receipt")
    bundle = _parse_json_object(bundle_bytes, "archived admission bundle")
    _strict_fields(bundle, _BUNDLE_FIELDS, "archived admission bundle")
    if ArtifactRef.from_dict(bundle["plan_ref"], "plan_ref").sha256 != receipt.plan_sha256:
        raise DataAdmissionError("archived bundle plan reference differs from receipt")
    raw_by_ref = {
        "entitlement_ref": (entitlement_bytes, receipt.entitlement_attestation_sha256),
        "data_manifest_ref": (manifest_bytes, receipt.data_manifest_sha256),
        "execution_contract_ref": (execution_bytes, receipt.execution_contract_sha256),
        "human_release_ref": (release_bytes, receipt.human_release_sha256),
    }
    for name, (content, expected_hash) in raw_by_ref.items():
        reference = ArtifactRef.from_dict(bundle[name], name)
        if reference.sha256 != expected_hash or _sha256_bytes(content) != expected_hash:
            raise DataAdmissionError(f"archived {name} hash differs from its signed bundle")
    entitlement = _parse_json_object(entitlement_bytes, "archived entitlement")
    manifest = _parse_json_object(manifest_bytes, "archived data manifest")
    execution = _parse_json_object(execution_bytes, "archived execution contract")
    release = _parse_json_object(release_bytes, "archived human release")
    _strict_fields(entitlement, _ENTITLEMENT_FIELDS, "archived entitlement")
    _strict_fields(manifest, _DATA_MANIFEST_FIELDS, "archived data manifest")
    _strict_fields(execution, _EXECUTION_FIELDS, "archived execution contract")
    _strict_fields(release, _RELEASE_FIELDS, "archived human release")
    admission_trust_store.assert_external_and_unchanged(vault_root)
    release_trust_store.assert_external_and_unchanged(vault_root)
    if (
        admission_trust_store.source_hash != receipt.admission_trust_policy_sha256
        or release_trust_store.source_hash != receipt.release_trust_policy_sha256
    ):
        raise DataAdmissionError("archived admission uses a different historical trust policy")
    admission_trust_store.verify(
        entitlement, required_role="authorized_entitlement_verifier",
        identity_field="attestor_id", signed_at_field="checked_at",
    )
    admission_trust_store.verify(
        manifest, required_role="authorized_data_manifest_signer",
        identity_field="admission_runner_id", signed_at_field="generated_at",
    )
    admission_trust_store.verify(
        execution, required_role="authorized_execution_contract_signer",
        identity_field="signer_id", signed_at_field="created_at",
    )
    release_trust_store.verify(
        release, required_role="authorized_data_release_approver",
        identity_field="approved_by", signed_at_field="approved_at",
    )
    component_bindings = {
        "plan_sha256": receipt.plan_sha256,
        "legacy_anchor_run_id": receipt.legacy_anchor_run_id,
        "reserved_root_run_id": receipt.root_run_id,
    }
    for component in (entitlement, manifest, execution, release):
        if any(component.get(field) != value for field, value in component_bindings.items()):
            raise DataAdmissionError("archived admission component binding differs from receipt")
    if (
        manifest.get("manifest_id") != receipt.data_manifest_id
        or manifest.get("data_asof_utc") != receipt.data_asof_utc
        or execution.get("feature_spec_sha256") != receipt.feature_manifest_hash
        or execution.get("split_policy_sha256") != receipt.split_manifest_hash
        or execution.get("validation_policy_sha256") != receipt.validation_policy_hash
        or execution.get("code_commit") != receipt.code_commit
    ):
        raise DataAdmissionError("archived execution/data identity differs from receipt")
    subject = {
        "plan_sha256": receipt.plan_sha256,
        "entitlement_attestation_sha256": receipt.entitlement_attestation_sha256,
        "admitted_data_manifest_sha256": receipt.data_manifest_sha256,
        "execution_contract_sha256": receipt.execution_contract_sha256,
        "legacy_anchor_run_id": receipt.legacy_anchor_run_id,
        "reserved_root_run_id": receipt.root_run_id,
    }
    if (
        _semantic_hash(subject) != receipt.admission_subject_sha256
        or release.get("admission_subject_sha256") != receipt.admission_subject_sha256
        or release.get("decision") != "authorize_rerun"
    ):
        raise DataAdmissionError("archived human release is not bound to the admission subject")
    return receipt


def _parse_json_object(content: bytes, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataAdmissionError(f"{kind} must contain UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise DataAdmissionError(f"{kind} must contain an object")
    # Canonical JSON serialization also rejects NaN and Infinity recursively.
    try:
        _canonical_bytes(dict(payload))
    except (TypeError, ValueError) as exc:
        raise DataAdmissionError(f"{kind} contains a noncanonical value") from exc
    return dict(payload)


def _load_bundle(root: Path, bundle_path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        relative = bundle_path.expanduser().resolve().relative_to(root.expanduser().resolve())
    except ValueError as exc:
        raise DataAdmissionError("admission bundle must live below the admission root") from exc
    details = bundle_path.stat()
    reference = ArtifactRef(relative.as_posix(), details.st_size, _sha256_bytes(bundle_path.read_bytes()))
    content = read_immutable_ref(root, reference, "data-admission bundle")
    payload = _parse_json_object(content, "data-admission bundle")
    _strict_fields(payload, _BUNDLE_FIELDS, "data-admission bundle")
    if payload["schema_version"] != SCHEMA_VERSION or type(payload["schema_version"]) is not int:
        raise DataAdmissionError("unsupported data-admission bundle schema")
    _validate_identifier("admission_id", _require_string(payload, "admission_id"))
    _strict_utc(payload["created_at"], "bundle.created_at")
    return payload, content


def _verify_inventory(
    root: Path,
    manifest: Mapping[str, Any],
    inventory_content: bytes,
    calendar_content: bytes,
    minimum_lookback_trading_days: int,
    minimum_coverage: Mapping[str, float],
) -> tuple[int, int, int, str, dict[str, float]]:
    try:
        raw_rows = json.loads(inventory_content)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataAdmissionError("partition inventory must be a JSON array") from exc
    if not isinstance(raw_rows, list) or not raw_rows:
        raise DataAdmissionError("partition inventory must be a nonempty JSON array")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    metric_numerators = {
        "stable_identifier_fraction": 0,
        "membership_event_completeness": 0,
        "corporate_action_completeness": 0,
        "required_timestamp_fraction": 0,
        "price_observation_fraction": 0,
    }
    physical_row_count = 0
    calendar = _parse_json_object(calendar_content, "exchange-session calendar")
    _strict_fields(calendar, _CALENDAR_FIELDS, "exchange-session calendar")
    if calendar["schema_version"] != SCHEMA_VERSION or not isinstance(
        calendar["markets"], Mapping
    ):
        raise DataAdmissionError("unsupported exchange-session calendar")
    requested_start = _strict_date(manifest["requested_start"], "requested_start")
    requested_end = _strict_date(manifest["requested_end"], "requested_end")
    market_calendars: dict[str, tuple[str, ...]] = {}
    for market in _require_exact_string_list(manifest, "markets"):
        sessions = calendar["markets"].get(market)
        if not isinstance(sessions, list) or not sessions:
            raise DataAdmissionError(f"exchange-session calendar is missing {market}")
        parsed_sessions = tuple(_strict_date(value, f"calendar.{market}") for value in sessions)
        if tuple(sorted(set(parsed_sessions))) != parsed_sessions:
            raise DataAdmissionError(f"exchange-session calendar is not sorted and unique: {market}")
        start_date = datetime.strptime(requested_start, "%Y-%m-%d").date()
        end_date = datetime.strptime(requested_end, "%Y-%m-%d").date()
        first_session = datetime.strptime(parsed_sessions[0], "%Y-%m-%d").date()
        last_session = datetime.strptime(parsed_sessions[-1], "%Y-%m-%d").date()
        if not (
            start_date <= first_session <= start_date + timedelta(days=7)
            and end_date - timedelta(days=7) <= last_session <= end_date
        ):
            raise DataAdmissionError(f"exchange-session calendar does not span the window: {market}")
        if any(datetime.strptime(value, "%Y-%m-%d").weekday() >= 5 for value in parsed_sessions):
            raise DataAdmissionError(f"exchange-session calendar contains a weekend: {market}")
        months = Counter(value[:7] for value in parsed_sessions)
        month_cursor = start_date.replace(day=1)
        final_month = end_date.replace(day=1)
        sparse_month = False
        while month_cursor <= final_month:
            next_month = (
                month_cursor.replace(year=month_cursor.year + 1, month=1)
                if month_cursor.month == 12
                else month_cursor.replace(month=month_cursor.month + 1)
            )
            bounded_start = max(start_date, month_cursor)
            bounded_end = min(end_date, next_month - timedelta(days=1))
            possible_weekdays = sum(
                (bounded_start + timedelta(days=offset)).weekday() < 5
                for offset in range((bounded_end - bounded_start).days + 1)
            )
            if months.get(month_cursor.strftime("%Y-%m"), 0) < min(15, possible_weekdays):
                sparse_month = True
                break
            month_cursor = next_month
        if len(parsed_sessions) < minimum_lookback_trading_days or sparse_month:
            raise DataAdmissionError(f"exchange-session calendar is too sparse: {market}")
        market_calendars[market] = parsed_sessions
    if set(calendar["markets"]) != set(market_calendars):
        raise DataAdmissionError("exchange-session calendar market set differs from manifest")
    temporary_directory = tempfile.TemporaryDirectory(prefix="quant-lab-coverage-")
    temporary_path = Path(temporary_directory.name) / "coverage.sqlite3"
    coverage_db = sqlite3.connect(temporary_path)
    coverage_db.execute("PRAGMA journal_mode=OFF")
    coverage_db.execute("PRAGMA synchronous=OFF")
    coverage_db.execute(
        """CREATE TABLE observations(
               market TEXT NOT NULL, instrument TEXT NOT NULL, session_date TEXT NOT NULL,
               revision_id TEXT NOT NULL, published_at TEXT NOT NULL, ingested_at TEXT NOT NULL,
               listing_date TEXT NOT NULL, delisting_date TEXT, price_complete INTEGER NOT NULL,
               PRIMARY KEY(market,instrument,session_date,revision_id)) WITHOUT ROWID"""
    )
    partition_root = _safe_relative(manifest["partition_root"], "partition_root")
    for raw in raw_rows:
        if not isinstance(raw, Mapping):
            raise DataAdmissionError("every partition inventory row must be an object")
        _strict_fields(raw, _INVENTORY_FIELDS, "partition inventory row")
        row = dict(raw)
        rel = _safe_relative(row["relative_path"], "inventory.relative_path").as_posix()
        if rel in seen:
            raise DataAdmissionError("partition inventory contains duplicate paths")
        seen.add(rel)
        size = _require_int(row["byte_size"], "inventory.byte_size", minimum=1)
        count = _require_int(row["row_count"], "inventory.row_count", minimum=1)
        _require_sha256("inventory.sha256", row["sha256"])
        _require_sha256("inventory.schema_sha256", row["schema_sha256"])
        if row["schema_sha256"] != manifest["schema_sha256"]:
            raise DataAdmissionError("partition schema hash differs from the admitted schema")
        if not isinstance(row["market"], str) or not _MIC.fullmatch(row["market"]):
            raise DataAdmissionError("partition market must be a MIC")
        parsed_times = {}
        for field in (
            "min_effective_at", "max_effective_at", "min_published_at",
            "max_published_at", "min_ingested_at", "max_ingested_at",
        ):
            parsed_times[field] = _strict_utc(row[field], f"inventory.{field}")
        for prefix in ("effective", "published", "ingested"):
            if parsed_times[f"min_{prefix}_at"] > parsed_times[f"max_{prefix}_at"]:
                raise DataAdmissionError(f"partition {prefix} time range is reversed")
        partition_ref = ArtifactRef((partition_root / rel).as_posix(), size, row["sha256"])
        try:
            with open_immutable_ref(root, partition_ref, f"partition {rel}") as handle:
                parquet_file = pq.ParquetFile(handle)
                schema = parquet_file.schema_arrow.remove_metadata()
                physical_schema_hash = canonical_arrow_schema_sha256(schema)
                if physical_schema_hash != row["schema_sha256"]:
                    raise DataAdmissionError("physical Parquet schema differs from signed inventory")
                if parquet_file.metadata.num_rows != count:
                    raise DataAdmissionError("physical Parquet row count differs from signed inventory")
                required_fields = set(_require_exact_string_list(manifest, "fields"))
                if not required_fields.issubset(schema.names):
                    raise DataAdmissionError("physical Parquet is missing an admitted field")
                scan_columns = [
                    "stable_instrument_id", "primary_exchange_mic",
                    "listing_effective_at", "delisting_effective_at",
                    "security_status_effective_at",
                    "cash_dividends", "stock_splits",
                    "rights_and_other_corporate_actions", "unadjusted_close",
                    "total_return_or_adjustment_factors", "trading_volume",
                    "revision_or_vintage_id",
                    "effective_timestamp_utc", "publication_timestamp_utc",
                    "ingestion_timestamp_utc",
                ]
                if not set(scan_columns).issubset(schema.names):
                    raise DataAdmissionError("physical Parquet is missing PIT lineage columns")
                observed_markets: set[str] = set()
                observed_min: dict[str, datetime | None] = {
                    "effective": None, "published": None, "ingested": None,
                }
                observed_max: dict[str, datetime | None] = dict(observed_min)
                for batch in parquet_file.iter_batches(batch_size=65_536, columns=scan_columns):
                    values = batch.to_pydict()
                    physical_row_count += batch.num_rows
                    observed_markets.update(str(value) for value in values["primary_exchange_mic"])
                    for index in range(batch.num_rows):
                        stable = values["stable_instrument_id"][index]
                        mic = values["primary_exchange_mic"][index]
                        listing = values["listing_effective_at"][index]
                        delisting = values["delisting_effective_at"][index]
                        status_at = values["security_status_effective_at"][index]
                        effective = values["effective_timestamp_utc"][index]
                        published = values["publication_timestamp_utc"][index]
                        ingested = values["ingestion_timestamp_utc"][index]
                        revision = values["revision_or_vintage_id"][index]
                        if stable is not None and str(stable).strip():
                            metric_numerators["stable_identifier_fraction"] += 1
                        if all(value is not None for value in (stable, mic, listing, status_at)):
                            metric_numerators["membership_event_completeness"] += 1
                        if all(
                            values[column][index] is not None
                            for column in (
                                "cash_dividends", "stock_splits",
                                "rights_and_other_corporate_actions",
                            )
                        ):
                            metric_numerators["corporate_action_completeness"] += 1
                        if all(value is not None for value in (effective, published, ingested)):
                            metric_numerators["required_timestamp_fraction"] += 1
                        price_complete = all(
                            values[column][index] is not None
                            for column in (
                                "unadjusted_close", "total_return_or_adjustment_factors",
                                "trading_volume",
                            )
                        )
                        if price_complete:
                            metric_numerators["price_observation_fraction"] += 1
                        if effective is not None and all(
                            value is not None and str(value).strip()
                            for value in (stable, mic, listing, revision, published, ingested)
                        ):
                            effective_utc = (
                                effective.astimezone(timezone.utc)
                                if isinstance(effective, datetime) and effective.tzinfo is not None
                                else _strict_utc(effective, "effective_timestamp_utc")
                            )
                            published_utc = (
                                published.astimezone(timezone.utc)
                                if isinstance(published, datetime) and published.tzinfo is not None
                                else _strict_utc(published, "publication_timestamp_utc")
                            )
                            ingested_utc = (
                                ingested.astimezone(timezone.utc)
                                if isinstance(ingested, datetime) and ingested.tzinfo is not None
                                else _strict_utc(ingested, "ingestion_timestamp_utc")
                            )
                            if ingested_utc < published_utc:
                                raise DataAdmissionError(
                                    "ingestion timestamp predates publication timestamp"
                                )
                            session_date = effective_utc.date().isoformat()
                            if session_date not in market_calendars.get(str(mic), ()):
                                raise DataAdmissionError(
                                    "price observation is outside its signed exchange calendar"
                                )
                            if not isinstance(listing, datetime) or listing.tzinfo is None:
                                raise DataAdmissionError("listing_effective_at must be timezone-aware")
                            listing_date = listing.astimezone(timezone.utc).date().isoformat()
                            delisting_date = None
                            if delisting is not None:
                                if not isinstance(delisting, datetime) or delisting.tzinfo is None:
                                    raise DataAdmissionError(
                                        "delisting_effective_at must be timezone-aware"
                                    )
                                delisting_date = delisting.astimezone(timezone.utc).date().isoformat()
                            try:
                                coverage_db.execute(
                                    "INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)",
                                    (
                                        str(mic), str(stable), session_date, str(revision),
                                        published_utc.isoformat(), ingested_utc.isoformat(),
                                        listing_date, delisting_date, int(price_complete),
                                    ),
                                )
                            except sqlite3.IntegrityError as exc:
                                raise DataAdmissionError(
                                    "duplicate instrument-session-vintage observation in physical Parquet"
                                ) from exc
                    for prefix, column in (
                        ("effective", "effective_timestamp_utc"),
                        ("published", "publication_timestamp_utc"),
                        ("ingested", "ingestion_timestamp_utc"),
                    ):
                        for value in values[column]:
                            if isinstance(value, datetime):
                                if value.tzinfo is None:
                                    raise DataAdmissionError(f"{column} must be timezone-aware")
                                parsed = value.astimezone(timezone.utc)
                            else:
                                parsed = _strict_utc(value, column)
                            if observed_min[prefix] is None or parsed < observed_min[prefix]:
                                observed_min[prefix] = parsed
                            if observed_max[prefix] is None or parsed > observed_max[prefix]:
                                observed_max[prefix] = parsed
                if observed_markets != {row["market"]}:
                    raise DataAdmissionError("physical Parquet MIC differs from signed inventory")
                for prefix in ("effective", "published", "ingested"):
                    if observed_min[prefix] is None or observed_max[prefix] is None:
                        raise DataAdmissionError(f"physical Parquet has no {prefix} timestamps")
                    if (
                        observed_min[prefix] != parsed_times[f"min_{prefix}_at"]
                        or observed_max[prefix] != parsed_times[f"max_{prefix}_at"]
                    ):
                        raise DataAdmissionError(
                            f"physical Parquet {prefix} range differs from signed inventory"
                        )
        except DataAdmissionError:
            raise
        except Exception as exc:
            raise DataAdmissionError(f"partition {rel} is not valid Parquet: {exc}") from exc
        rows.append(row)
    canonical_rows = sorted(rows, key=lambda item: item["relative_path"])
    if rows != canonical_rows:
        raise DataAdmissionError("partition inventory must be sorted by relative_path")
    for market in _require_exact_string_list(manifest, "markets"):
        market_rows = [row for row in rows if row["market"] == market]
        if not market_rows:
            raise DataAdmissionError(f"partition inventory does not cover market {market}")
        market_start = min(
            _strict_utc(row["min_effective_at"], "min_effective_at").date().isoformat()
            for row in market_rows
        )
        market_end = max(
            _strict_utc(row["max_effective_at"], "max_effective_at").date().isoformat()
            for row in market_rows
        )
        if (
            market_start != market_calendars[market][0]
            or market_end != market_calendars[market][-1]
        ):
            raise DataAdmissionError(f"physical partitions do not span requested window for {market}")
    if physical_row_count < 1:
        raise DataAdmissionError("physical partitions contain no rows")
    expected_instrument_sessions = 0
    for market, instrument, listing_date, delisting_date in coverage_db.execute(
        """SELECT market,instrument,MIN(listing_date),MAX(delisting_date)
           FROM observations GROUP BY market,instrument"""
    ):
        life_start = max(requested_start, listing_date)
        life_end = min(requested_end, delisting_date or requested_end)
        expected_instrument_sessions += sum(
            life_start <= session <= life_end for session in market_calendars[market]
        )
    complete_instrument_sessions = coverage_db.execute(
        """SELECT COUNT(*) FROM (
               SELECT market,instrument,session_date
               FROM observations
               GROUP BY market,instrument,session_date
               HAVING MAX(price_complete)=1
           )"""
    ).fetchone()[0]
    coverage_db.close()
    temporary_directory.cleanup()
    if expected_instrument_sessions < 1:
        raise DataAdmissionError("exchange calendar and instrument lives have no expected sessions")
    physical_coverage = {
        name: numerator / physical_row_count
        for name, numerator in metric_numerators.items()
    }
    physical_coverage["price_observation_fraction"] = (
        complete_instrument_sessions / expected_instrument_sessions
    )
    if physical_coverage["price_observation_fraction"] > 1.0:
        raise DataAdmissionError("physical prices exceed the calendar/instrument denominator")
    for name, threshold in minimum_coverage.items():
        if physical_coverage.get(name, -1.0) < float(threshold):
            raise DataAdmissionError(f"physical Parquet coverage is below threshold: {name}")
    content_root = _semantic_hash(canonical_rows)
    return (
        len(rows), sum(row["row_count"] for row in rows),
        sum(row["byte_size"] for row in rows), content_root, physical_coverage,
    )


def verify_data_admission(
    *,
    plan_path: Path,
    bundle_path: Path,
    admission_root: Path,
    admission_trust_store: DataAdmissionTrustStore,
    release_trust_store: DataReleaseTrustStore,
    workspace_root: Path,
) -> VerifiedDataAdmission:
    """Verify a complete immutable admission bundle and return sealed bindings."""
    workspace_root = workspace_root.expanduser().resolve()
    admission_root = admission_root.expanduser().resolve()
    admission_trust_store.assert_external_and_unchanged(workspace_root / "quant_lab" / "vault")
    release_trust_store.assert_external_and_unchanged(workspace_root / "quant_lab" / "vault")
    if plan_path.is_symlink() or not plan_path.is_file():
        raise DataAdmissionError("rerun plan must be a regular non-symlink file")
    plan_bytes = plan_path.read_bytes()
    plan = _parse_json_object(plan_bytes, "rerun plan")
    plan_hash = _sha256_bytes(plan_bytes)
    plan_without_hash = dict(plan)
    claimed_semantic = plan_without_hash.pop("package_semantic_sha256", None)
    if not isinstance(claimed_semantic, str) or _semantic_hash(plan_without_hash) != claimed_semantic:
        raise DataAdmissionError("rerun plan semantic hash mismatch")
    if plan.get("package_status") != "ready_waiting_for_lseg" or plan.get("execution_authorized") is not False:
        raise DataAdmissionError("rerun plan is not in the fail-closed waiting state")

    bundle, bundle_bytes = _load_bundle(admission_root, bundle_path)
    refs = {
        name: ArtifactRef.from_dict(bundle[name], name)
        for name in (
            "plan_ref", "entitlement_ref", "data_manifest_ref",
            "execution_contract_ref", "human_release_ref",
        )
    }
    copied_plan = read_immutable_ref(admission_root, refs["plan_ref"], "plan_ref")
    if copied_plan != plan_bytes:
        raise DataAdmissionError("admission plan copy differs from the authoritative rerun plan")
    entitlement = _parse_json_object(
        read_immutable_ref(admission_root, refs["entitlement_ref"], "entitlement_ref"),
        "LSEG entitlement attestation",
    )
    data_manifest = _parse_json_object(
        read_immutable_ref(admission_root, refs["data_manifest_ref"], "data_manifest_ref"),
        "admitted data manifest",
    )
    execution = _parse_json_object(
        read_immutable_ref(admission_root, refs["execution_contract_ref"], "execution_contract_ref"),
        "frozen rerun execution contract",
    )
    release = _parse_json_object(
        read_immutable_ref(admission_root, refs["human_release_ref"], "human_release_ref"),
        "human data-availability release",
    )
    _strict_fields(entitlement, _ENTITLEMENT_FIELDS, "LSEG entitlement attestation")
    _strict_fields(data_manifest, _DATA_MANIFEST_FIELDS, "admitted data manifest")
    _strict_fields(execution, _EXECUTION_FIELDS, "frozen rerun execution contract")
    _strict_fields(release, _RELEASE_FIELDS, "human data-availability release")
    for payload, kind in (
        (entitlement, "entitlement"), (data_manifest, "data manifest"),
        (execution, "execution contract"), (release, "human release"),
    ):
        if payload["schema_version"] != SCHEMA_VERSION or type(payload["schema_version"]) is not int:
            raise DataAdmissionError(f"unsupported {kind} schema")

    admission_trust_store.verify(
        entitlement, required_role="authorized_entitlement_verifier",
        identity_field="attestor_id", signed_at_field="checked_at",
    )
    admission_trust_store.verify(
        data_manifest, required_role="authorized_data_manifest_signer",
        identity_field="admission_runner_id", signed_at_field="generated_at",
    )
    admission_trust_store.verify(
        execution, required_role="authorized_execution_contract_signer",
        identity_field="signer_id", signed_at_field="created_at",
    )
    release_trust_store.verify(
        release, required_role="authorized_data_release_approver",
        identity_field="approved_by", signed_at_field="approved_at",
    )

    if entitlement["provider"] != "LSEG" or entitlement["delivery_method"] != "s3_direct":
        raise DataAdmissionError("entitlement must attest LSEG S3 Direct delivery")
    if entitlement["permitted_use"] != "research_trial" or entitlement["redistribution_allowed"] is not False:
        raise DataAdmissionError("entitlement use or redistribution terms do not match this trial")
    permitted_markets = set(_require_exact_string_list(entitlement, "permitted_markets"))
    if any(not _MIC.fullmatch(item) for item in permitted_markets):
        raise DataAdmissionError("entitlement market identifiers must be MICs")
    valid_from = _strict_utc(entitlement["valid_from"], "entitlement.valid_from")
    valid_until = _strict_utc(entitlement["valid_until"], "entitlement.valid_until")
    checked_at = _strict_utc(entitlement["checked_at"], "entitlement.checked_at")
    if not valid_from <= checked_at <= valid_until:
        raise DataAdmissionError("entitlement attestation falls outside its validity window")
    if not valid_from <= datetime.now(timezone.utc) <= valid_until:
        raise DataAdmissionError("LSEG entitlement is not active at activation time")
    if _strict_utc(plan["created_at"], "plan.created_at") > checked_at:
        raise DataAdmissionError("entitlement check predates the frozen rerun plan")
    evidence_ref = ArtifactRef.from_dict(entitlement["source_evidence_ref"], "source_evidence_ref")
    read_immutable_ref(admission_root, evidence_ref, "entitlement source evidence")

    legacy_anchor = plan["legacy_anchor_run_id"]
    root_run_id = plan["next_run_id"]
    bindings = ("plan_sha256", "legacy_anchor_run_id", "reserved_root_run_id")
    for component in (entitlement, data_manifest, execution, release):
        expected = (plan_hash, legacy_anchor, root_run_id)
        actual = tuple(component[field] for field in bindings)
        if actual != expected:
            raise DataAdmissionError("signed admission components are cross-bound to another plan or run")
    for component in (data_manifest,):
        for field in ("dataset_product_id", "entitled_account_id", "requester_pays_account_id", "aws_region"):
            if component[field] != entitlement[field]:
                raise DataAdmissionError(f"data manifest entitlement binding differs: {field}")
    if data_manifest["entitlement_attestation_sha256"] != refs["entitlement_ref"].sha256:
        raise DataAdmissionError("data manifest is not bound to the verified entitlement attestation")

    contract = plan["data_contract"]
    required_markets = tuple(contract["markets"])
    admitted_markets = _require_exact_string_list(data_manifest, "markets")
    if tuple(sorted(admitted_markets)) != tuple(sorted(required_markets)):
        raise DataAdmissionError("admitted markets must exactly equal the frozen requested markets")
    if not set(admitted_markets).issubset(permitted_markets):
        raise DataAdmissionError("admitted market is not entitled")
    fields = set(_require_exact_string_list(data_manifest, "fields"))
    if not set(contract["required_fields"]).issubset(fields):
        raise DataAdmissionError("admitted data is missing a frozen required field")
    start = _strict_date(data_manifest["requested_start"], "requested_start")
    end = _strict_date(data_manifest["requested_end"], "requested_end")
    if start > contract["replication_window"]["start"] or end < contract["replication_window"]["end"]:
        raise DataAdmissionError("admitted data does not cover the frozen replication window")
    if data_manifest["provider"] != "LSEG" or data_manifest["aws_region"] != contract["expected_aws_region"]:
        raise DataAdmissionError("admitted provider or AWS region differs from the plan")
    flags = (
        "point_in_time", "includes_inactive_and_delisted", "corporate_actions_complete",
        "tradability_and_liquidity_complete", "timestamp_lineage_complete",
        "revision_vintages_retained", "sealed_out_of_sample_period",
    )
    if any(data_manifest[field] is not True for field in flags):
        raise DataAdmissionError("admitted data is missing a mandatory PIT/governance guarantee")
    coverage = data_manifest["coverage_metrics"]
    if not isinstance(coverage, Mapping) or set(coverage) != set(contract["coverage_thresholds"]):
        raise DataAdmissionError("coverage metrics must contain the exact frozen threshold set")
    for name, threshold in contract["coverage_thresholds"].items():
        if _require_finite(coverage[name], f"coverage.{name}") < float(threshold):
            raise DataAdmissionError(f"coverage threshold failed: {name}")
    for field in (
        "schema_sha256", "content_root_sha256", "calendar_policy_sha256",
        "timezone_policy_sha256", "corporate_action_policy_sha256",
    ):
        _require_sha256(field, data_manifest[field])
    inventory_ref = ArtifactRef.from_dict(data_manifest["inventory_ref"], "inventory_ref")
    profiling_ref = ArtifactRef.from_dict(data_manifest["profiling_ref"], "profiling_ref")
    calendar_ref = ArtifactRef.from_dict(data_manifest["calendar_ref"], "calendar_ref")
    inventory_content = read_immutable_ref(admission_root, inventory_ref, "partition inventory")
    profile = _parse_json_object(
        read_immutable_ref(admission_root, profiling_ref, "signed profiling evidence"),
        "signed profiling evidence",
    )
    calendar_content = read_immutable_ref(
        admission_root, calendar_ref, "exchange-session calendar"
    )
    if calendar_ref.sha256 != data_manifest["calendar_policy_sha256"]:
        raise DataAdmissionError("exchange-session calendar hash differs from manifest policy")
    _strict_fields(profile, _PROFILE_FIELDS, "signed profiling evidence")
    if profile["schema_version"] != SCHEMA_VERSION or type(profile["schema_version"]) is not int:
        raise DataAdmissionError("unsupported profiling-evidence schema")
    part_count, row_count, total_bytes, content_root, physical_coverage = _verify_inventory(
        admission_root, data_manifest, inventory_content, calendar_content,
        int(plan["research_contract"]["minimum_lookback_trading_days"]),
        contract["coverage_thresholds"],
    )
    if (
        part_count != _require_int(data_manifest["partition_count"], "partition_count", minimum=1)
        or row_count != _require_int(data_manifest["row_count"], "row_count", minimum=1)
        or total_bytes != _require_int(data_manifest["total_bytes"], "total_bytes", minimum=1)
        or content_root != data_manifest["content_root_sha256"]
    ):
        raise DataAdmissionError("partition inventory totals or content root do not match the manifest")
    for name, measured in physical_coverage.items():
        if not math.isclose(
            measured, _require_finite(coverage[name], f"coverage.{name}"),
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise DataAdmissionError(
                f"signed coverage metric differs from physical Parquet measurement: {name}"
            )
    inventory_markets = {
        item["market"] for item in json.loads(inventory_content)
    }
    if inventory_markets != set(admitted_markets):
        raise DataAdmissionError("partition inventory must cover every and only admitted market")
    expected_profile = {
        "manifest_id": data_manifest["manifest_id"],
        "markets": data_manifest["markets"],
        "fields": data_manifest["fields"],
        "coverage_metrics": data_manifest["coverage_metrics"],
        "flags": {field: data_manifest[field] for field in flags},
        "row_count": data_manifest["row_count"],
        "partition_count": data_manifest["partition_count"],
        "generated_at": data_manifest["generated_at"],
        "runner_id": data_manifest["admission_runner_id"],
    }
    if any(profile[field] != value for field, value in expected_profile.items()):
        raise DataAdmissionError("profiling evidence differs from the signed data manifest")

    generated_at = _strict_utc(data_manifest["generated_at"], "generated_at")
    data_asof = _strict_utc(data_manifest["data_asof_utc"], "data_asof_utc")
    if data_asof > generated_at or not valid_from <= generated_at <= valid_until:
        raise DataAdmissionError("data manifest timestamp is inconsistent with entitlement/as-of time")
    if refs["entitlement_ref"].sha256 != data_manifest["entitlement_attestation_sha256"]:
        raise DataAdmissionError("entitlement hash binding failed")

    if not _GIT_COMMIT.fullmatch(_require_string(execution, "code_commit")):
        raise DataAdmissionError("execution contract code_commit must be a full lowercase Git hash")
    if execution["clean_tree"] is not True:
        raise DataAdmissionError("execution contract requires a clean code snapshot")
    source_rows = [
        *plan["frozen_source_contracts"]["feature_sources"],
        *plan["frozen_source_contracts"]["split_and_validation_sources"],
    ]
    current_sources = []
    for row in source_rows:
        source = workspace_root / row["path"]
        if source.is_symlink() or not source.is_file() or _sha256_bytes(source.read_bytes()) != row["sha256"]:
            raise DataAdmissionError(f"frozen rerun source drifted: {row['path']}")
        current_sources.append(row)
    if execution["source_bundle_sha256"] != _semantic_hash(current_sources):
        raise DataAdmissionError("execution source bundle is not the frozen plan source bundle")
    code_ref = ArtifactRef.from_dict(execution["code_snapshot_ref"], "code_snapshot_ref")
    code_snapshot = _parse_json_object(
        read_immutable_ref(admission_root, code_ref, "code snapshot"), "code snapshot"
    )
    _strict_fields(code_snapshot, _CODE_SNAPSHOT_FIELDS, "code snapshot")
    if code_snapshot["schema_version"] != SCHEMA_VERSION:
        raise DataAdmissionError("unsupported code snapshot schema")
    if (
        code_snapshot["sources"] != current_sources
        or code_snapshot["source_bundle_sha256"] != execution["source_bundle_sha256"]
        or code_snapshot["code_commit"] != execution["code_commit"]
    ):
        raise DataAdmissionError("code snapshot differs from the frozen execution sources")
    repository = workspace_root / "quant_lab"

    def git_output(*arguments: str) -> str:
        try:
            completed = subprocess.run(
                ["git", "-C", str(repository), *arguments],
                check=True, capture_output=True, text=True,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            raise DataAdmissionError(f"cannot verify frozen Git snapshot: {exc}") from exc
        return completed.stdout.strip()

    current_commit = git_output("rev-parse", "HEAD")
    current_tree = git_output("rev-parse", "HEAD^{tree}")
    if current_commit != execution["code_commit"] or current_tree != code_snapshot["git_tree"]:
        raise DataAdmissionError("execution contract is not bound to the checked-out Git tree")
    # The signed tree identifies the entire repository, so the cleanliness claim
    # must cover the entire checkout too.  Limiting this check to the explicitly
    # frozen source rows would let an altered transitive import execute outside
    # the signed source bundle while ``clean_tree`` still appeared true.
    git_output("diff", "--exit-code", "HEAD")
    if git_output("ls-files", "--others", "--exclude-standard"):
        raise DataAdmissionError("repository contains untracked files outside the bound Git snapshot")

    validation_ref = ArtifactRef.from_dict(
        execution["validation_policy_ref"], "validation_policy_ref"
    )
    research_ref = ArtifactRef.from_dict(
        execution["research_config_ref"], "research_config_ref"
    )
    expected_validation = frozen_validation_policy_payload(plan)
    expected_research = frozen_research_config_payload(plan)
    if read_immutable_ref(admission_root, validation_ref, "validation policy") != _canonical_bytes(
        expected_validation
    ):
        raise DataAdmissionError("validation policy artifact differs from the frozen plan")
    if read_immutable_ref(admission_root, research_ref, "research config") != _canonical_bytes(
        expected_research
    ):
        raise DataAdmissionError("research config artifact differs from the frozen plan")
    expected_execution = {
        "feature_spec_sha256": plan["frozen_source_contracts"]["feature_spec_hash"],
        "split_policy_sha256": plan["frozen_source_contracts"]["split_and_validation_policy_hash"],
        "validation_policy_sha256": plan["frozen_source_contracts"]["validation_policy_sha256"],
        "research_config_sha256": plan["frozen_source_contracts"]["research_config_sha256"],
        "required_markets": list(required_markets),
        "required_fields": list(contract["required_fields"]),
        "required_start": contract["replication_window"]["start"],
        "required_end": contract["replication_window"]["end"],
        "required_coverage": contract["coverage_thresholds"],
    }
    for field, expected in expected_execution.items():
        if execution[field] != expected:
            raise DataAdmissionError(f"execution contract differs from frozen plan: {field}")
    for field in (
        "source_bundle_sha256", "feature_spec_sha256", "split_policy_sha256",
        "validation_policy_sha256", "research_config_sha256",
    ):
        _require_sha256(field, execution[field])
    if (
        validation_ref.sha256 != execution["validation_policy_sha256"]
        or research_ref.sha256 != execution["research_config_sha256"]
    ):
        raise DataAdmissionError("execution artifact references do not match their frozen hashes")

    if release["decision"] != "authorize_rerun":
        raise DataAdmissionError("human data release did not authorize this rerun")
    limitations = _require_exact_string_list(release, "limitations_acknowledged")
    if not limitations:
        raise DataAdmissionError("human release must acknowledge limitations")
    approved_at = _strict_utc(release["approved_at"], "approved_at")
    expires_at = _strict_utc(release["expires_at"], "expires_at")
    execution_created = _strict_utc(execution["created_at"], "execution.created_at")
    bundle_created = _strict_utc(bundle["created_at"], "bundle.created_at")
    if not checked_at <= generated_at <= execution_created <= approved_at <= bundle_created <= expires_at:
        raise DataAdmissionError("signed admission timestamps are out of order")
    if expires_at < datetime.now(timezone.utc):
        raise DataAdmissionError("human data-availability release has expired")
    component_hashes = {
        "plan_sha256": plan_hash,
        "entitlement_attestation_sha256": refs["entitlement_ref"].sha256,
        "admitted_data_manifest_sha256": refs["data_manifest_ref"].sha256,
        "execution_contract_sha256": refs["execution_contract_ref"].sha256,
        "legacy_anchor_run_id": legacy_anchor,
        "reserved_root_run_id": root_run_id,
    }
    subject_hash = _semantic_hash(component_hashes)
    release_bindings = {
        **component_hashes,
        "admission_subject_sha256": subject_hash,
    }
    if any(release[field] != value for field, value in release_bindings.items()):
        raise DataAdmissionError("human release is not bound to the verified admission subject")

    return VerifiedDataAdmission(
        admission_id=bundle["admission_id"],
        admission_bundle_sha256=_sha256_bytes(bundle_bytes),
        admission_subject_sha256=subject_hash,
        plan_sha256=plan_hash,
        legacy_anchor_run_id=legacy_anchor,
        root_run_id=root_run_id,
        dataset_product_id=data_manifest["dataset_product_id"],
        data_manifest_id=data_manifest["manifest_id"],
        data_asof_utc=data_manifest["data_asof_utc"],
        feature_manifest_hash=execution["feature_spec_sha256"],
        split_manifest_id=f"split-{execution['split_policy_sha256'][:20]}",
        split_manifest_hash=execution["split_policy_sha256"],
        validation_policy_hash=execution["validation_policy_sha256"],
        code_commit=execution["code_commit"],
        entitlement_attestation_sha256=refs["entitlement_ref"].sha256,
        data_manifest_sha256=refs["data_manifest_ref"].sha256,
        execution_contract_sha256=refs["execution_contract_ref"].sha256,
        human_release_sha256=refs["human_release_ref"].sha256,
        admission_trust_policy_sha256=admission_trust_store.source_hash,
        release_trust_policy_sha256=release_trust_store.source_hash,
    )


__all__ = [
    "ArtifactRef",
    "AuthorizedDataSigner",
    "DataAdmissionError",
    "DataAdmissionTrustStore",
    "DataReleaseTrustStore",
    "VerifiedDataAdmission",
    "archive_verified_data_admission",
    "canonical_arrow_schema_sha256",
    "frozen_research_config_payload",
    "frozen_validation_policy_payload",
    "open_immutable_ref",
    "read_immutable_ref",
    "verify_archived_data_admission",
    "verify_data_admission",
]
