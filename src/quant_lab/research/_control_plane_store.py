from __future__ import annotations

import fcntl
import json
import os
import tempfile
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from quant_lab.research._control_plane_contracts import (
    SCHEMA_VERSION,
    RUN_MANIFEST_NAME,
    ApprovalDecision,
    ApprovalError,
    ConcurrentUpdateError,
    EvidenceArtifactReference,
    GateStatus,
    HumanApprovalRecord,
    HumanSupersessionRecord,
    IllegalTransitionError,
    IntegrityError,
    PromotionStatus,
    REQUIRED_VALIDATION_GATES,
    RunKind,
    RunManifest,
    ValidationArtifact,
    ValidationEvidenceBundle,
    ValidationStatus,
    WORKFLOW_STAGES,
    WorkflowStatus,
    _canonical_bytes,
    _enum,
    _hash_bytes,
    _hash_file,
    _parse_utc,
    _require_sha256,
    _strict_fields,
    _utc_now,
    _validate_identifier,
)
from quant_lab.research._control_plane_trust import (
    ApprovalTrustStore,
    ValidationRunnerTrustStore,
)


_EVENT_FIELDS = {
    "schema_version",
    "sequence",
    "run_id",
    "event_type",
    "previous_event_hash",
    "before_manifest_hash",
    "after_manifest_hash",
    "actor_type",
    "actor_id",
    "payload",
    "created_at",
}
_EVENT_TYPES = {
    "run_created",
    "data_admission_verified",
    "workflow_transition",
    "workflow_stage_completed",
    "validation_policy_registered",
    "validation_artifact_attached",
    "validation_admitted",
    "promotion_transition",
    "human_decision_verified",
}
_ACTOR_TYPES = {"controller", "agent", "deterministic_python", "authorized_human"}
_EVENT_ACTOR_TYPES = {
    "run_created": {"controller"},
    "data_admission_verified": {"controller"},
    "workflow_transition": {"controller"},
    "workflow_stage_completed": {"agent"},
    "validation_policy_registered": {"controller"},
    "validation_artifact_attached": {"deterministic_python"},
    "validation_admitted": {"deterministic_python"},
    "human_decision_verified": {"authorized_human"},
    # A deterministic controller can park a run for missing data; only a
    # separately authorized human can supersede a previously approved run.
    "promotion_transition": {"controller", "authorized_human"},
}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, content: bytes, *, exclusive: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary_path = Path(temporary)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            # Publish a fully fsynced inode with no-clobber semantics. A crash
            # before link(2) leaves no partial final file for a retry to find.
            os.link(temporary_path, path)
            _fsync_directory(path.parent)
        finally:
            temporary_path.unlink(missing_ok=True)
        return
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)


class ExperimentRunStore:
    """Locked, append-audited control plane for one immutable root run ID."""

    def __init__(self, vault_root: str | Path, run_id: str) -> None:
        _validate_identifier("run_id", run_id)
        self.vault_root = Path(vault_root).expanduser().resolve()
        self.experiments_root = self.vault_root / "30_experiments"
        if self.experiments_root.is_symlink():
            raise ValueError("experiments root cannot be a symlink")
        if self.experiments_root.exists() and not self.experiments_root.is_dir():
            raise ValueError("experiments root must be a directory")
        self.run_root = self.experiments_root / run_id
        self.run_id = run_id
        if self.run_root.parent != self.experiments_root:
            raise ValueError("run_id escaped the experiments root")
        if self.run_root.exists() and self.run_root.is_symlink():
            raise ValueError("run root cannot be a symlink")
        self.manifest_path = self.run_root / RUN_MANIFEST_NAME
        self._history_root = self.run_root / ".control" / "manifests"
        self._events_root = self.run_root / ".control" / "events"
        self._lock_path = self.run_root / ".control" / "writer.lock"

    @contextmanager
    def _locked(self) -> Iterator[None]:
        if self.experiments_root.is_symlink():
            raise ValueError("experiments root cannot be a symlink")
        self.run_root.mkdir(parents=True, exist_ok=True)
        if self.run_root.is_symlink():
            raise ValueError("run root cannot be a symlink")
        control_root = self._lock_path.parent
        if control_root.exists() and control_root.is_symlink():
            raise ValueError("control directory cannot be a symlink")
        control_root.mkdir(parents=True, exist_ok=True)
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(self._lock_path, flags, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            self._recover_locked()
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _write_file(self, path: Path, content: bytes, *, exclusive: bool = False) -> None:
        cursor = path.parent
        while cursor != self.run_root:
            if cursor.exists() and cursor.is_symlink():
                raise IntegrityError("control-plane output parent cannot be a symlink")
            if cursor == cursor.parent:
                raise IntegrityError("control-plane output is not below the run root")
            cursor = cursor.parent
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise IntegrityError("control-plane outputs cannot be symlinks")
        try:
            path.parent.resolve().relative_to(self.run_root.resolve())
        except ValueError as exc:
            raise IntegrityError("control-plane output escaped the run root") from exc
        _atomic_write(path, content, exclusive=exclusive)

    def _manifest_versions(self) -> list[Path]:
        if not self._history_root.exists():
            return []
        return sorted(self._history_root.glob("*.json"))

    def _event_files(self) -> list[Path]:
        if not self._events_root.exists():
            return []
        return sorted(self._events_root.glob("*.json"))

    def _recover_locked(self) -> None:
        events = self._verified_events_locked()
        committed_hashes = {
            str(event[field])
            for event in events
            for field in ("before_manifest_hash", "after_manifest_hash")
        }
        for path in self._manifest_versions():
            digest = path.stem.split("-", 1)[-1]
            if digest in committed_hashes:
                continue
            orphan_root = self.run_root / ".control" / "orphans" / "manifests"
            orphan_root.mkdir(parents=True, exist_ok=True)
            if orphan_root.is_symlink():
                raise IntegrityError("manifest orphan quarantine cannot be a symlink")
            destination = orphan_root / path.name
            if destination.exists():
                if destination.is_symlink() or destination.read_bytes() != path.read_bytes():
                    raise IntegrityError("manifest orphan quarantine identity drift")
                path.unlink()
            else:
                os.replace(path, destination)
            _fsync_directory(orphan_root)
            _fsync_directory(path.parent)
        if not events:
            return
        expected_hash = str(events[-1]["after_manifest_hash"])
        if self.manifest_path.exists():
            current = self._load_manifest_file(self.manifest_path)
            if current.content_hash() == expected_hash:
                return
        matches = [path for path in self._manifest_versions() if path.stem.endswith(expected_hash)]
        if len(matches) != 1:
            raise IntegrityError("cannot recover authoritative manifest from the event head")
        recovered = self._load_manifest_file(matches[0])
        if recovered.content_hash() != expected_hash:
            raise IntegrityError("recovery manifest hash mismatch")
        self._write_file(self.manifest_path, recovered.canonical_bytes())

    def _load_manifest_file(self, path: Path) -> RunManifest:
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("manifest must be a regular non-symlink file")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise IntegrityError("manifest must contain an object")
        try:
            manifest = RunManifest.from_dict(payload)
        except (TypeError, ValueError) as exc:
            raise IntegrityError(f"invalid run manifest: {exc}") from exc
        if manifest.run_id != self.run_id:
            raise IntegrityError("manifest run_id does not match its run directory")
        return manifest

    def _verified_events_locked(self) -> list[dict[str, Any]]:
        previous_hash: str | None = None
        previous_manifest_hash: str | None = None
        previous_created_at: datetime | None = None
        events: list[dict[str, Any]] = []
        for expected_sequence, path in enumerate(self._event_files(), start=1):
            if path.is_symlink() or not path.is_file():
                raise IntegrityError("audit events must be regular non-symlink files")
            content = path.read_bytes()
            digest = _hash_bytes(content)
            try:
                sequence_text, filename_hash = path.stem.split("-", 1)
            except ValueError as exc:
                raise IntegrityError("invalid audit event filename") from exc
            if int(sequence_text) != expected_sequence or filename_hash != digest:
                raise IntegrityError("audit event sequence or filename hash mismatch")
            payload = json.loads(content)
            if not isinstance(payload, Mapping):
                raise IntegrityError("audit event must contain an object")
            _strict_fields(payload, _EVENT_FIELDS, "audit event")
            if type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
                raise IntegrityError("unsupported audit event schema")
            if isinstance(payload["sequence"], bool) or not isinstance(payload["sequence"], int):
                raise IntegrityError("audit event sequence must be an integer")
            textual = (
                "run_id",
                "event_type",
                "before_manifest_hash",
                "after_manifest_hash",
                "actor_type",
                "actor_id",
                "created_at",
            )
            if any(not isinstance(payload[name], str) for name in textual):
                raise IntegrityError("audit event textual fields must be strings")
            if payload["previous_event_hash"] is not None and not isinstance(payload["previous_event_hash"], str):
                raise IntegrityError("previous_event_hash must be null or a string")
            if not isinstance(payload["payload"], Mapping):
                raise IntegrityError("audit event payload must be an object")
            if payload["event_type"] not in _EVENT_TYPES or payload["actor_type"] not in _ACTOR_TYPES:
                raise IntegrityError("audit event type or actor type is not allowed")
            if payload["actor_type"] not in _EVENT_ACTOR_TYPES[payload["event_type"]]:
                raise IntegrityError("audit event actor is not authorized for its event type")
            if not payload["actor_id"].strip() or len(payload["actor_id"]) > 256:
                raise IntegrityError("audit event actor_id is invalid")
            if payload["sequence"] != expected_sequence or payload["run_id"] != self.run_id:
                raise IntegrityError("audit event identity mismatch")
            if payload["previous_event_hash"] != previous_hash:
                raise IntegrityError("audit event hash chain is broken")
            _require_sha256("before_manifest_hash", payload["before_manifest_hash"])
            _require_sha256("after_manifest_hash", payload["after_manifest_hash"])
            if previous_manifest_hash is not None and payload["before_manifest_hash"] != previous_manifest_hash:
                raise IntegrityError("audit event manifest chain is broken")
            if expected_sequence == 1:
                if payload["event_type"] != "run_created":
                    raise IntegrityError("the first audit event must be run_created")
                if payload["before_manifest_hash"] != payload["after_manifest_hash"]:
                    raise IntegrityError("run_created must bind one initial manifest version")
            elif payload["event_type"] == "run_created":
                raise IntegrityError("run_created may appear only once")
            created_at = _parse_utc(payload["created_at"], "created_at")
            if previous_created_at is not None and created_at < previous_created_at:
                raise IntegrityError("audit event timestamps are not monotonic")
            before_manifest = self._manifest_version_by_hash_locked(
                payload["before_manifest_hash"]
            )
            after_manifest = self._manifest_version_by_hash_locked(
                payload["after_manifest_hash"]
            )
            self._verify_event_semantics_locked(
                payload,
                before_manifest=before_manifest,
                after_manifest=after_manifest,
                completed_stages=tuple(
                    str(event["payload"]["stage"])
                    for event in events
                    if event["event_type"] == "workflow_stage_completed"
                ),
            )
            previous_hash = digest
            previous_manifest_hash = payload["after_manifest_hash"]
            previous_created_at = created_at
            events.append(dict(payload))
        return events

    def _manifest_version_by_hash_locked(self, digest: str) -> RunManifest:
        matches = [
            path
            for path in self._manifest_versions()
            if path.stem.endswith(f"-{digest}")
        ]
        if len(matches) != 1:
            raise IntegrityError("audit event must reference exactly one immutable manifest version")
        manifest = self._load_manifest_file(matches[0])
        if manifest.content_hash() != digest:
            raise IntegrityError("audit event manifest version hash mismatch")
        return manifest

    def _verify_event_semantics_locked(
        self,
        event: Mapping[str, Any],
        *,
        before_manifest: RunManifest,
        after_manifest: RunManifest,
        completed_stages: tuple[str, ...],
    ) -> None:
        event_type = str(event["event_type"])
        actor_type = str(event["actor_type"])
        payload = event["payload"]

        def exact_payload_fields(expected: set[str]) -> None:
            actual = set(payload)
            if actual != expected:
                raise IntegrityError(
                    f"{event_type} payload fields mismatch: "
                    f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
                )

        def only_manifest_fields_may_change(allowed: set[str]) -> None:
            before = before_manifest.to_dict()
            after = after_manifest.to_dict()
            changed = {name for name in before if before[name] != after[name]}
            if not changed.issubset(allowed):
                raise IntegrityError(
                    f"{event_type} changed unauthorized manifest fields: {sorted(changed - allowed)}"
                )
            if _parse_utc(after_manifest.updated_at, "updated_at") < _parse_utc(
                before_manifest.updated_at, "updated_at"
            ):
                raise IntegrityError(f"{event_type} moved manifest updated_at backwards")

        if before_manifest.identity_hash() != after_manifest.identity_hash():
            raise IntegrityError("audit event changed immutable manifest identity")

        if event_type == "run_created":
            exact_payload_fields({"configuration_hash"})
            if payload["configuration_hash"] != after_manifest.configuration_hash():
                raise IntegrityError("run_created configuration hash mismatch")
            if (
                after_manifest.workflow_status is not WorkflowStatus.CREATED
                or after_manifest.validation_status is not ValidationStatus.NOT_RUN
                or after_manifest.promotion_status is not PromotionStatus.INELIGIBLE
            ):
                raise IntegrityError("run_created did not bind an initial-state manifest")
            return

        if event_type == "data_admission_verified":
            exact_payload_fields(
                {
                    "admission_uri", "admission_sha256",
                    "admission_subject_sha256", "plan_sha256",
                    "entitlement_attestation_sha256", "data_manifest_sha256",
                    "execution_contract_sha256", "human_release_sha256",
                    "admission_trust_policy_sha256", "release_trust_policy_sha256",
                }
            )
            if (
                before_manifest.content_hash() != after_manifest.content_hash()
                or before_manifest.workflow_status is not WorkflowStatus.CREATED
                or before_manifest.validation_status is not ValidationStatus.NOT_RUN
                or before_manifest.promotion_status is not PromotionStatus.INELIGIBLE
            ):
                raise IntegrityError("data admission must be recorded before workflow execution")
            for field in payload:
                if field.endswith("sha256"):
                    _require_sha256(field, payload[field])
            return

        if event_type == "workflow_transition":
            exact_payload_fields({"from", "to"})
            only_manifest_fields_may_change({"workflow_status", "updated_at"})
            allowed = {
                WorkflowStatus.CREATED: {WorkflowStatus.RUNNING},
                WorkflowStatus.RUNNING: {WorkflowStatus.COMPLETE, WorkflowStatus.FAILED},
                WorkflowStatus.COMPLETE: set(),
                WorkflowStatus.FAILED: set(),
            }
            if (
                after_manifest.workflow_status not in allowed[before_manifest.workflow_status]
                or payload["from"] != before_manifest.workflow_status.value
                or payload["to"] != after_manifest.workflow_status.value
            ):
                raise IntegrityError("workflow_transition payload or state transition is invalid")
            return

        if event_type == "workflow_stage_completed":
            exact_payload_fields(
                {"stage", "root_run_id", "execution_id", "note_uri", "note_sha256"}
            )
            if before_manifest.content_hash() != after_manifest.content_hash():
                raise IntegrityError("workflow stage events cannot mutate the manifest")
            stage = payload["stage"]
            if len(completed_stages) >= len(WORKFLOW_STAGES):
                raise IntegrityError("too many workflow stage events")
            expected_stage = WORKFLOW_STAGES[len(completed_stages)]
            state_allows_stage = (
                before_manifest.workflow_status is WorkflowStatus.COMPLETE
                and before_manifest.promotion_status is PromotionStatus.APPROVAL_REQUIRED
                if stage == "human_approval"
                else before_manifest.workflow_status is WorkflowStatus.RUNNING
            )
            if (
                not state_allows_stage
                or stage != expected_stage
                or payload["root_run_id"] != self.run_id
            ):
                raise IntegrityError("workflow stage event is out of order or bound incorrectly")
            return

        if event_type == "validation_policy_registered":
            exact_payload_fields({"policy_hash", "policy_uri", "policy_sha256"})
            if (
                before_manifest.content_hash() != after_manifest.content_hash()
                or before_manifest.workflow_status is not WorkflowStatus.CREATED
                or before_manifest.validation_status is not ValidationStatus.NOT_RUN
                or before_manifest.promotion_status is not PromotionStatus.INELIGIBLE
                or payload["policy_hash"] != payload["policy_sha256"]
            ):
                raise IntegrityError("validation policy was not registered before workflow execution")
            return

        if event_type == "validation_artifact_attached":
            exact_payload_fields(
                {"gate_id", "artifact_uri", "artifact_sha256", "runner_trust_policy_hash"}
            )
            if (
                before_manifest.content_hash() != after_manifest.content_hash()
                or before_manifest.workflow_status is not WorkflowStatus.COMPLETE
                or before_manifest.validation_status is not ValidationStatus.NOT_RUN
                or payload["gate_id"] not in REQUIRED_VALIDATION_GATES
            ):
                raise IntegrityError("validation artifact event is not valid for this run state")
            return

        if event_type == "validation_admitted":
            exact_payload_fields(
                {
                    "candidate_id",
                    "bundle_hash",
                    "runner_trust_policy_hash",
                    "validation_status",
                    "promotion_status",
                }
            )
            only_manifest_fields_may_change(
                {"validation_status", "promotion_status", "updated_at"}
            )
            allowed_promotion = {
                ValidationStatus.PASS: (
                    PromotionStatus.APPROVAL_REQUIRED
                    if after_manifest.promotion_eligible
                    else PromotionStatus.INELIGIBLE
                ),
                ValidationStatus.FAIL: PromotionStatus.REJECTED,
            }
            promotion_is_valid = (
                after_manifest.promotion_status
                is allowed_promotion.get(after_manifest.validation_status)
                if after_manifest.validation_status in allowed_promotion
                else (
                    after_manifest.validation_status is ValidationStatus.INCONCLUSIVE
                    and after_manifest.promotion_status
                    in {PromotionStatus.PARKED_FOR_DATA, PromotionStatus.INELIGIBLE}
                )
            )
            if (
                before_manifest.workflow_status is not WorkflowStatus.COMPLETE
                or before_manifest.validation_status is not ValidationStatus.NOT_RUN
                or before_manifest.promotion_status is not PromotionStatus.INELIGIBLE
                or not promotion_is_valid
                or payload["validation_status"] != after_manifest.validation_status.value
                or payload["promotion_status"] != after_manifest.promotion_status.value
            ):
                raise IntegrityError("validation admission payload or state transition is invalid")
            return

        if event_type == "human_decision_verified":
            exact_payload_fields(
                {
                    "approval_id",
                    "decision",
                    "evidence_bundle_hash",
                    "signer_key_id",
                    "trust_policy_hash",
                    "approval_uri",
                    "approval_sha256",
                    "resulting_promotion_status",
                }
            )
            only_manifest_fields_may_change({"promotion_status", "updated_at"})
            expected_promotion = (
                PromotionStatus.APPROVED
                if payload["decision"] == ApprovalDecision.APPROVED.value
                else PromotionStatus.REJECTED
            )
            if (
                actor_type != "authorized_human"
                or before_manifest.validation_status is not ValidationStatus.PASS
                or before_manifest.promotion_status is not PromotionStatus.APPROVAL_REQUIRED
                or after_manifest.promotion_status is not expected_promotion
                or payload["resulting_promotion_status"] != expected_promotion.value
            ):
                raise IntegrityError("human decision payload or state transition is invalid")
            return

        if event_type == "promotion_transition":
            only_manifest_fields_may_change(
                {"validation_status", "promotion_status", "updated_at"}
            )
            if actor_type == "controller":
                exact_payload_fields({"from", "to", "reason_code"})
                if (
                    before_manifest.validation_status is not ValidationStatus.NOT_RUN
                    or after_manifest.validation_status is not ValidationStatus.INCONCLUSIVE
                    or before_manifest.promotion_status is not PromotionStatus.INELIGIBLE
                    or after_manifest.promotion_status is not PromotionStatus.PARKED_FOR_DATA
                    or payload["from"] != PromotionStatus.INELIGIBLE.value
                    or payload["to"] != PromotionStatus.PARKED_FOR_DATA.value
                ):
                    raise IntegrityError("parked-for-data promotion transition is invalid")
            else:
                exact_payload_fields(
                    {
                        "from",
                        "to",
                        "replacement_run_id",
                        "supersession_id",
                        "trust_policy_hash",
                        "supersession_uri",
                        "supersession_sha256",
                    }
                )
                if (
                    before_manifest.validation_status is not ValidationStatus.PASS
                    or after_manifest.validation_status is not ValidationStatus.PASS
                    or before_manifest.promotion_status is not PromotionStatus.APPROVED
                    or after_manifest.promotion_status is not PromotionStatus.SUPERSEDED
                    or payload["from"] != PromotionStatus.APPROVED.value
                    or payload["to"] != PromotionStatus.SUPERSEDED.value
                ):
                    raise IntegrityError("human supersession promotion transition is invalid")
            return

        raise IntegrityError(f"unhandled audit event type: {event_type}")

    def _append_event_locked(
        self,
        *,
        event_type: str,
        before_hash: str,
        after_hash: str,
        actor_type: str,
        actor_id: str,
        payload: Mapping[str, Any],
    ) -> str:
        if event_type not in _EVENT_TYPES or actor_type not in _ACTOR_TYPES:
            raise ValueError("audit event type or actor type is not allowed")
        if not isinstance(actor_id, str) or not actor_id.strip() or len(actor_id) > 256:
            raise ValueError("audit event actor_id is invalid")
        _require_sha256("before_manifest_hash", before_hash)
        _require_sha256("after_manifest_hash", after_hash)
        _canonical_bytes(dict(payload))
        events = self._verified_events_locked()
        sequence = len(events) + 1
        previous_hash = None
        event_files = self._event_files()
        if event_files:
            previous_hash = event_files[-1].stem.split("-", 1)[1]
        event = {
            "schema_version": SCHEMA_VERSION,
            "sequence": sequence,
            "run_id": self.run_id,
            "event_type": event_type,
            "previous_event_hash": previous_hash,
            "before_manifest_hash": before_hash,
            "after_manifest_hash": after_hash,
            "actor_type": actor_type,
            "actor_id": actor_id,
            "payload": dict(payload),
            "created_at": _utc_now(),
        }
        content = _canonical_bytes(event)
        digest = _hash_bytes(content)
        self._write_file(
            self._events_root / f"{sequence:06d}-{digest}.json",
            content,
            exclusive=True,
        )
        return digest

    def _publish_manifest_locked(
        self,
        *,
        before: RunManifest | None,
        after: RunManifest,
        event_type: str,
        actor_type: str,
        actor_id: str,
        payload: Mapping[str, Any],
    ) -> RunManifest:
        after.validate()
        if before is not None and before.identity_hash() != after.identity_hash():
            raise IntegrityError("immutable manifest identity changed")
        before_hash = after.content_hash() if before is None else before.content_hash()
        after_hash = after.content_hash()
        revision = len(self._manifest_versions()) + 1
        version_path = self._history_root / f"{revision:06d}-{after_hash}.json"
        self._write_file(version_path, after.canonical_bytes(), exclusive=True)
        self._append_event_locked(
            event_type=event_type,
            before_hash=before_hash,
            after_hash=after_hash,
            actor_type=actor_type,
            actor_id=actor_id,
            payload=payload,
        )
        self._write_file(self.manifest_path, after.canonical_bytes())
        return after

    def create(self, manifest: RunManifest) -> RunManifest:
        manifest.validate()
        if manifest.run_id != self.run_id:
            raise ValueError("manifest run_id does not match the store")
        if (
            manifest.workflow_status is not WorkflowStatus.CREATED
            or manifest.validation_status is not ValidationStatus.NOT_RUN
            or manifest.promotion_status is not PromotionStatus.INELIGIBLE
            or manifest.created_at != manifest.updated_at
        ):
            raise IllegalTransitionError("new runs must start at created/not_run/ineligible")
        project_vault = Path(__file__).resolve().parents[3] / "vault"
        if manifest.run_kind is not RunKind.RESEARCH and self.vault_root == project_vault.resolve():
            raise PermissionError("test, synthetic, and smoke runs cannot write to the production vault")
        if manifest.parent_run_id is not None:
            parent = ExperimentRunStore(self.vault_root, manifest.parent_run_id).load()
            if parent.run_kind is not manifest.run_kind:
                raise ValueError("parent and child runs must have the same run_kind")
        with self._locked():
            if self.manifest_path.exists():
                existing = self._load_manifest_file(self.manifest_path)
                if existing.configuration_hash() != manifest.configuration_hash():
                    raise IntegrityError("run_id already exists with different immutable configuration")
                return existing
            return self._publish_manifest_locked(
                before=None,
                after=manifest,
                event_type="run_created",
                actor_type="controller",
                actor_id="research_director_controller",
                payload={"configuration_hash": manifest.configuration_hash()},
            )

    def _record_archived_data_admission(
        self,
        evidence: Mapping[str, Any],
        *,
        expected_manifest_hash: str,
        actor_id: str = "data_admission_controller",
    ) -> Path:
        """Persist the signed components; only the verified LSEG entrypoint calls this."""
        archive_fields = {
            "schema_version", "verified_admission", "bundle_base64",
            "entitlement_base64", "data_manifest_base64",
            "execution_contract_base64", "human_release_base64",
        }
        receipt_fields = {
            "admission_id", "admission_bundle_sha256", "admission_subject_sha256",
            "plan_sha256", "legacy_anchor_run_id", "root_run_id",
            "dataset_product_id", "data_manifest_id", "data_asof_utc",
            "feature_manifest_hash", "split_manifest_id", "split_manifest_hash",
            "validation_policy_hash", "code_commit",
            "entitlement_attestation_sha256", "data_manifest_sha256",
            "execution_contract_sha256", "human_release_sha256",
            "admission_trust_policy_sha256", "release_trust_policy_sha256",
        }
        if set(evidence) != archive_fields or not isinstance(
            evidence.get("verified_admission"), Mapping
        ):
            raise ValueError("data-admission evidence has an unexpected schema")
        receipt = evidence["verified_admission"]
        if set(receipt) != receipt_fields:
            raise ValueError("verified data-admission receipt has an unexpected schema")
        if receipt["root_run_id"] != self.run_id:
            raise ValueError("data-admission evidence is bound to another run")
        for field, value in receipt.items():
            if field.endswith("sha256") or field in {
                "feature_manifest_hash", "split_manifest_hash",
                "validation_policy_hash",
            }:
                _require_sha256(field, value)
        content = _canonical_bytes(dict(evidence))
        digest = _hash_bytes(content)
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            self._assert_cas(current, expected_manifest_hash)
            if (
                current.workflow_status is not WorkflowStatus.CREATED
                or current.validation_status is not ValidationStatus.NOT_RUN
                or current.promotion_status is not PromotionStatus.INELIGIBLE
                or current.data_manifest_id != receipt["data_manifest_id"]
                or current.data_asof_utc != receipt["data_asof_utc"]
                or current.feature_manifest_hash != receipt["feature_manifest_hash"]
                or current.code_commit != receipt["code_commit"]
                or current.split_manifest_id != receipt["split_manifest_id"]
            ):
                raise IntegrityError("data-admission evidence differs from the created manifest")
            path = self.run_root / ".control" / "admissions" / f"{digest}.json"
            if path.exists():
                if path.is_symlink() or path.read_bytes() != content:
                    raise IntegrityError("data-admission evidence identity drift")
            else:
                self._write_file(path, content, exclusive=True)
            matching = [
                event
                for event in self._verified_events_locked()
                if event["event_type"] == "data_admission_verified"
            ]
            if matching:
                if len(matching) != 1 or matching[0]["payload"]["admission_sha256"] != digest:
                    raise IntegrityError("a different data admission is already recorded")
                return path
            self._append_event_locked(
                event_type="data_admission_verified",
                before_hash=current.content_hash(),
                after_hash=current.content_hash(),
                actor_type="controller",
                actor_id=actor_id,
                payload={
                    "admission_uri": path.relative_to(self.run_root).as_posix(),
                    "admission_sha256": digest,
                    "admission_subject_sha256": receipt["admission_subject_sha256"],
                    "plan_sha256": receipt["plan_sha256"],
                    "entitlement_attestation_sha256": receipt["entitlement_attestation_sha256"],
                    "data_manifest_sha256": receipt["data_manifest_sha256"],
                    "execution_contract_sha256": receipt["execution_contract_sha256"],
                    "human_release_sha256": receipt["human_release_sha256"],
                    "admission_trust_policy_sha256": receipt["admission_trust_policy_sha256"],
                    "release_trust_policy_sha256": receipt["release_trust_policy_sha256"],
                },
            )
            return path

    def register_validation_policy(
        self,
        runner_trust_store: ValidationRunnerTrustStore,
        *,
        expected_manifest_hash: str,
        actor_id: str = "validation_policy_controller",
    ) -> Path:
        runner_trust_store.assert_external_and_unchanged(self.vault_root)
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            self._assert_cas(current, expected_manifest_hash)
            if (
                current.workflow_status is not WorkflowStatus.CREATED
                or current.validation_status is not ValidationStatus.NOT_RUN
                or current.promotion_status is not PromotionStatus.INELIGIBLE
            ):
                raise IllegalTransitionError(
                    "validation policy must be preregistered while the run is still created"
                )
            path = (
                self.run_root
                / ".control"
                / "policies"
                / f"validation-{runner_trust_store.source_hash}.json"
            )
            if path.exists():
                if path.is_symlink() or path.read_bytes() != runner_trust_store.source_bytes:
                    raise IntegrityError("preregistered validation policy identity drift")
            else:
                self._write_file(path, runner_trust_store.source_bytes, exclusive=True)
            matching = [
                event
                for event in self._verified_events_locked()
                if event["event_type"] == "validation_policy_registered"
            ]
            if matching:
                if (
                    len(matching) != 1
                    or matching[0]["payload"].get("policy_hash")
                    != runner_trust_store.source_hash
                ):
                    raise IntegrityError("a different validation policy is already preregistered")
                return path
            self._append_event_locked(
                event_type="validation_policy_registered",
                before_hash=current.content_hash(),
                after_hash=current.content_hash(),
                actor_type="controller",
                actor_id=actor_id,
                payload={
                    "policy_hash": runner_trust_store.source_hash,
                    "policy_uri": path.relative_to(self.run_root).as_posix(),
                    "policy_sha256": runner_trust_store.source_hash,
                },
            )
            return path

    def load(self) -> RunManifest:
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        with self._locked():
            if not self.manifest_path.exists():
                raise FileNotFoundError(self.manifest_path)
            return self._load_manifest_file(self.manifest_path)

    def _assert_cas(self, current: RunManifest, expected_manifest_hash: str) -> None:
        _require_sha256("expected_manifest_hash", expected_manifest_hash)
        if current.content_hash() != expected_manifest_hash:
            raise ConcurrentUpdateError("stale manifest hash; reload before applying a transition")

    def transition_workflow(
        self,
        target: WorkflowStatus | str,
        *,
        expected_manifest_hash: str,
        actor_id: str,
    ) -> RunManifest:
        desired = _enum(WorkflowStatus, target, "workflow status")
        allowed = {
            WorkflowStatus.CREATED: {WorkflowStatus.RUNNING},
            WorkflowStatus.RUNNING: {WorkflowStatus.COMPLETE, WorkflowStatus.FAILED},
            WorkflowStatus.COMPLETE: set(),
            WorkflowStatus.FAILED: set(),
        }
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            prior = [
                event
                for event in self._verified_events_locked()
                if event["event_type"] == "workflow_transition"
                and event["before_manifest_hash"] == expected_manifest_hash
                and event["payload"].get("to") == desired.value
                and event["actor_id"] == actor_id
            ]
            if prior:
                if len(prior) != 1:
                    raise IntegrityError("workflow transition was recorded more than once")
                return current
            self._assert_cas(current, expected_manifest_hash)
            if desired not in allowed[current.workflow_status]:
                raise IllegalTransitionError(
                    f"illegal workflow transition: {current.workflow_status.value} -> {desired.value}"
                )
            if desired is WorkflowStatus.COMPLETE:
                completed = {
                    event["payload"].get("stage")
                    for event in self._verified_events_locked()
                    if event["event_type"] == "workflow_stage_completed"
                }
                required = {
                    "research_director",
                    "data_steward",
                    "feature_engineer",
                    "modeler",
                    "portfolio_optimizer",
                    "validation_officer",
                    "risk_manager",
                    "reporting_analyst",
                }
                if not required.issubset(completed):
                    raise IllegalTransitionError("workflow cannot complete before every machine stage is recorded")
            updated = replace(current, workflow_status=desired, updated_at=_utc_now())
            return self._publish_manifest_locked(
                before=current,
                after=updated,
                event_type="workflow_transition",
                actor_type="controller",
                actor_id=actor_id,
                payload={"from": current.workflow_status.value, "to": desired.value},
            )

    def write_supporting_note(
        self,
        *,
        stage: str,
        sequence: int,
        title: str,
        body: str,
        tags: tuple[str, ...],
        metadata: Mapping[str, Any],
        expected_manifest_hash: str,
    ) -> Path:
        _validate_identifier("stage", stage)
        if stage not in WORKFLOW_STAGES:
            raise ValueError(f"unknown workflow stage: {stage}")
        if not 1 <= sequence <= 99:
            raise ValueError("note sequence must be between 1 and 99")
        reserved_metadata = {
            "title",
            "type",
            "created",
            "tags",
            "run_id",
            "manifest_identity_hash",
            "workflow_stage",
        }
        if reserved_metadata & set(metadata):
            raise ValueError("supporting note metadata cannot override authoritative fields")
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            completed_events = [
                event
                for event in self._verified_events_locked()
                if event["event_type"] == "workflow_stage_completed"
            ]
            completed_stages = [str(event["payload"].get("stage")) for event in completed_events]
            if stage != "human_approval" and stage in completed_stages:
                matching = [
                    event
                    for event in completed_events
                    if event["payload"].get("stage") == stage
                    and (
                        event["before_manifest_hash"] == expected_manifest_hash
                        or current.content_hash() == expected_manifest_hash
                    )
                ]
                if len(matching) != 1:
                    raise ConcurrentUpdateError(
                        "stale manifest hash does not identify this completed stage"
                    )
            else:
                self._assert_cas(current, expected_manifest_hash)
            if stage == "human_approval":
                if (
                    current.workflow_status is not WorkflowStatus.COMPLETE
                    or current.promotion_status is not PromotionStatus.APPROVAL_REQUIRED
                ):
                    raise IllegalTransitionError("human gate requires completed workflow and approval_required")
            elif stage not in completed_stages and current.workflow_status is not WorkflowStatus.RUNNING:
                raise IllegalTransitionError("machine workflow stages require workflow_status=running")
            if stage in completed_stages:
                if completed_stages.count(stage) != 1:
                    raise IntegrityError(f"workflow stage was recorded more than once: {stage}")
            else:
                expected_stage = WORKFLOW_STAGES[len(completed_stages)]
                if stage != expected_stage:
                    raise IllegalTransitionError(
                        f"workflow stage order violation: expected {expected_stage}, got {stage}"
                    )
            note_path = self.run_root / "notes" / f"{sequence:02d}_{stage}.md"
            frontmatter = {
                "title": title,
                "type": "agent_note",
                "created": current.created_at,
                "tags": list(dict.fromkeys(tags)),
                "run_id": current.run_id,
                "manifest_identity_hash": current.identity_hash(),
                "workflow_stage": stage,
                **dict(metadata),
            }
            lines = ["---"]
            for key, value in frontmatter.items():
                lines.append(
                    f"{key}: {json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)}"
                )
            lines.extend(("---", "", f"# {title}", "", body.rstrip(), ""))
            content = "\n".join(lines).encode("utf-8")
            note_hash = _hash_bytes(content)
            if note_path.exists():
                if note_path.is_symlink() or note_path.read_bytes() != content:
                    raise IntegrityError(f"supporting note identity drift: {note_path.name}")
            else:
                self._write_file(note_path, content, exclusive=True)
            already_recorded = any(
                event["event_type"] == "workflow_stage_completed"
                and event["payload"].get("stage") == stage
                and event["payload"].get("note_sha256") == note_hash
                for event in self._verified_events_locked()
            )
            if not already_recorded:
                self._append_event_locked(
                    event_type="workflow_stage_completed",
                    before_hash=current.content_hash(),
                    after_hash=current.content_hash(),
                    actor_type="agent",
                    actor_id=stage,
                    payload={
                        "stage": stage,
                        "root_run_id": self.run_id,
                        "execution_id": _hash_bytes(f"{self.run_id}:{stage}".encode("utf-8"))[:24],
                        "note_uri": note_path.relative_to(self.run_root).as_posix(),
                        "note_sha256": note_hash,
                    },
                )
            return note_path

    def write_validation_artifact(
        self,
        artifact: ValidationArtifact,
        *,
        runner_trust_store: ValidationRunnerTrustStore,
    ) -> Path:
        artifact.validate()
        if artifact.run_id != self.run_id:
            raise ValueError("validation artifact run_id does not match the store")
        with self._locked():
            manifest = self._load_manifest_file(self.manifest_path)
            if manifest.workflow_status is not WorkflowStatus.COMPLETE:
                raise IllegalTransitionError("validation artifacts require workflow_status=complete")
            if manifest.validation_status is not ValidationStatus.NOT_RUN:
                raise IllegalTransitionError("validation artifacts cannot be changed after admission")
            runner_trust_store.assert_external_and_unchanged(self.vault_root)
            self._validation_policy_event_locked(runner_trust_store.source_hash)
            runner_trust_store.verify(artifact)
            self._assert_artifact_binding(artifact, manifest)
            content = _canonical_bytes(artifact.to_dict())
            artifact_hash = _hash_bytes(content)
            path = (
                self.run_root
                / "validation"
                / "artifacts"
                / artifact.gate_id
                / f"{artifact_hash}.json"
            )
            if path.exists():
                if path.is_symlink() or path.read_bytes() != content:
                    raise IntegrityError(f"validation artifact identity drift: {artifact.gate_id}")
            else:
                self._write_file(path, content, exclusive=True)
            already_recorded = any(
                event["event_type"] == "validation_artifact_attached"
                and event["payload"].get("gate_id") == artifact.gate_id
                and event["payload"].get("artifact_sha256") == artifact_hash
                for event in self._verified_events_locked()
            )
            if not already_recorded:
                self._append_event_locked(
                    event_type="validation_artifact_attached",
                    before_hash=manifest.content_hash(),
                    after_hash=manifest.content_hash(),
                    actor_type="deterministic_python",
                    actor_id=artifact.producer_execution_id,
                    payload={
                        "gate_id": artifact.gate_id,
                        "artifact_uri": path.relative_to(self.run_root).as_posix(),
                        "artifact_sha256": artifact_hash,
                        "runner_trust_policy_hash": runner_trust_store.source_hash,
                    },
                )
            return path

    def _assert_artifact_binding(self, artifact: ValidationArtifact, manifest: RunManifest) -> None:
        expected = {
            "run_id": manifest.run_id,
            "data_manifest_id": manifest.data_manifest_id,
            "data_asof_utc": manifest.data_asof_utc,
            "feature_manifest_hash": manifest.feature_manifest_hash,
            "code_commit": manifest.code_commit,
            "split_manifest_id": manifest.split_manifest_id,
        }
        actual = {
            "run_id": artifact.run_id,
            "data_manifest_id": artifact.data_manifest_id,
            "data_asof_utc": artifact.data_asof_utc,
            "feature_manifest_hash": artifact.feature_manifest_hash,
            "code_commit": artifact.code_commit,
            "split_manifest_id": artifact.split_manifest_id,
        }
        if actual != expected:
            raise IntegrityError("validation artifact is stale or bound to another manifest")
        generated_at = _parse_utc(artifact.generated_at, "generated_at")
        if generated_at < _parse_utc(manifest.created_at, "manifest created_at"):
            raise IntegrityError("validation artifact predates its run manifest")
        if generated_at > datetime.now(timezone.utc) + timedelta(minutes=5):
            raise IntegrityError("validation artifact timestamp is unreasonably far in the future")

    def admit_validation(
        self,
        *,
        candidate_id: str,
        artifact_paths: Mapping[str, str | Path],
        runner_trust_store: ValidationRunnerTrustStore,
        expected_manifest_hash: str,
        actor_id: str = "validation_controller",
    ) -> tuple[RunManifest, ValidationEvidenceBundle]:
        _validate_identifier("candidate_id", candidate_id)
        if set(artifact_paths) != set(REQUIRED_VALIDATION_GATES):
            raise ValueError("validation admission requires the exact deterministic gate set")
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            self._assert_cas(current, expected_manifest_hash)
            if current.workflow_status is not WorkflowStatus.COMPLETE:
                raise IllegalTransitionError("validation can run only after workflow completion")
            if current.validation_status is not ValidationStatus.NOT_RUN:
                raise IllegalTransitionError("validation is forward-only; use a child run for revalidation")
            runner_trust_store.assert_external_and_unchanged(self.vault_root)
            self._validation_policy_event_locked(runner_trust_store.source_hash)
            references: dict[str, EvidenceArtifactReference] = {}
            artifact_timestamps: list[str] = []
            candidate_hashes: set[str] = set()
            resolved_paths: set[Path] = set()
            artifact_inodes: set[tuple[int, int]] = set()
            for gate_id in REQUIRED_VALIDATION_GATES:
                path = Path(artifact_paths[gate_id])
                if not path.is_absolute():
                    path = self.run_root / path
                if path.is_symlink() or not path.is_file():
                    raise IntegrityError(f"validation artifact must be a regular non-symlink file: {gate_id}")
                resolved = path.resolve()
                try:
                    resolved.relative_to(self.run_root.resolve())
                except ValueError as exc:
                    raise IntegrityError(f"validation artifact escaped the run root: {gate_id}") from exc
                if resolved in resolved_paths:
                    raise IntegrityError("validation artifacts must use unique files")
                resolved_paths.add(resolved)
                file_stat = path.stat()
                inode = (file_stat.st_dev, file_stat.st_ino)
                if inode in artifact_inodes:
                    raise IntegrityError("validation artifacts cannot be aliases or hard links")
                artifact_inodes.add(inode)
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(payload, Mapping):
                    raise IntegrityError(f"validation artifact must contain an object: {gate_id}")
                artifact = ValidationArtifact.from_dict(payload)
                if artifact.gate_id != gate_id or artifact.candidate_id != candidate_id:
                    raise IntegrityError(f"validation artifact identity mismatch: {gate_id}")
                self._assert_artifact_binding(artifact, current)
                runner_trust_store.verify(artifact)
                candidate_hashes.add(artifact.candidate_hash)
                artifact_timestamps.append(artifact.generated_at)
                digest, size = _hash_file(path)
                references[gate_id] = EvidenceArtifactReference(
                    artifact_uri=resolved.relative_to(self.run_root.resolve()).as_posix(),
                    sha256=digest,
                    size_bytes=size,
                    status=artifact.status,
                    producer_execution_id=artifact.producer_execution_id,
                )
            if len(candidate_hashes) != 1:
                raise IntegrityError("all validation artifacts must bind one candidate content hash")
            bundle = ValidationEvidenceBundle(
                run_id=self.run_id,
                candidate_id=candidate_id,
                candidate_hash=next(iter(candidate_hashes)),
                manifest_identity_hash=current.identity_hash(),
                runner_trust_policy_hash=runner_trust_store.source_hash,
                artifacts=references,
                created_at=max(artifact_timestamps),
            )
            bundle.validate()
            bundle_hash = bundle.content_hash()
            bundle_path = self.run_root / "validation" / "bundles" / f"{bundle_hash}.json"
            if bundle_path.exists():
                if bundle_path.is_symlink() or bundle_path.read_bytes() != bundle.canonical_bytes():
                    raise IntegrityError("immutable evidence bundle identity drift")
            else:
                self._write_file(bundle_path, bundle.canonical_bytes(), exclusive=True)
            self._write_file(
                self.run_root / "validation" / "evidence_bundle.json",
                bundle.canonical_bytes(),
            )
            validation_status, promotion_status = self._derive_validation_outcome(
                bundle,
                current,
            )
            updated = replace(
                current,
                validation_status=validation_status,
                promotion_status=promotion_status,
                updated_at=_utc_now(),
            )
            result = self._publish_manifest_locked(
                before=current,
                after=updated,
                event_type="validation_admitted",
                actor_type="deterministic_python",
                actor_id=actor_id,
                payload={
                    "candidate_id": candidate_id,
                    "bundle_hash": bundle_hash,
                    "runner_trust_policy_hash": runner_trust_store.source_hash,
                    "validation_status": validation_status.value,
                    "promotion_status": promotion_status.value,
                },
            )
            return result, bundle

    @staticmethod
    def _derive_validation_outcome(
        bundle: ValidationEvidenceBundle,
        manifest: RunManifest,
    ) -> tuple[ValidationStatus, PromotionStatus]:
        validation_status = bundle.overall_status()
        if validation_status is ValidationStatus.PASS and manifest.promotion_eligible:
            promotion_status = PromotionStatus.APPROVAL_REQUIRED
        elif validation_status is ValidationStatus.FAIL:
            promotion_status = PromotionStatus.REJECTED
        elif (
            any(
                bundle.artifacts[gate].status is GateStatus.INCONCLUSIVE
                for gate in ("point_in_time_data_audit", "data_feature_split_lineage")
            )
            and manifest.promotion_eligible
        ):
            promotion_status = PromotionStatus.PARKED_FOR_DATA
        else:
            promotion_status = PromotionStatus.INELIGIBLE
        return validation_status, promotion_status

    def park_for_data(
        self,
        *,
        reason_code: str,
        expected_manifest_hash: str,
        actor_id: str = "data_steward_controller",
    ) -> RunManifest:
        allowed_reasons = {
            "LSEG_ENTITLEMENT_UNAVAILABLE",
            "PIT_HISTORY_UNAVAILABLE",
            "TIMESTAMP_LINEAGE_INCOMPLETE",
            "DATA_EVIDENCE_INCONCLUSIVE",
        }
        if reason_code not in allowed_reasons:
            raise ValueError("unsupported parked-for-data reason code")
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            self._assert_cas(current, expected_manifest_hash)
            if not current.promotion_eligible or current.promotion_status is not PromotionStatus.INELIGIBLE:
                raise IllegalTransitionError("only an eligible, currently ineligible run can park for data")
            updated = replace(
                current,
                validation_status=ValidationStatus.INCONCLUSIVE,
                promotion_status=PromotionStatus.PARKED_FOR_DATA,
                updated_at=_utc_now(),
            )
            return self._publish_manifest_locked(
                before=current,
                after=updated,
                event_type="promotion_transition",
                actor_type="controller",
                actor_id=actor_id,
                payload={
                    "from": current.promotion_status.value,
                    "to": PromotionStatus.PARKED_FOR_DATA.value,
                    "reason_code": reason_code,
                },
            )

    def _load_current_bundle_locked(self) -> ValidationEvidenceBundle:
        path = self.run_root / "validation" / "evidence_bundle.json"
        if path.is_symlink() or not path.is_file():
            raise IntegrityError("current validation evidence bundle is missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise IntegrityError("evidence bundle must contain an object")
        bundle = ValidationEvidenceBundle.from_dict(payload)
        for gate_id, reference in bundle.artifacts.items():
            artifact_path = self.run_root / reference.artifact_uri
            if artifact_path.is_symlink() or not artifact_path.is_file():
                raise IntegrityError(f"evidence artifact missing or not regular: {gate_id}")
            resolved = artifact_path.resolve()
            try:
                resolved.relative_to(self.run_root.resolve())
            except ValueError as exc:
                raise IntegrityError(f"evidence artifact escaped the run root: {gate_id}") from exc
            digest, size = _hash_file(artifact_path)
            if digest != reference.sha256 or size != reference.size_bytes:
                raise IntegrityError(f"evidence artifact hash or size mismatch: {gate_id}")
            payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact = ValidationArtifact.from_dict(payload)
            if artifact.status is not reference.status or artifact.producer_execution_id != reference.producer_execution_id:
                raise IntegrityError(f"evidence artifact reference drift: {gate_id}")
        immutable = self.run_root / "validation" / "bundles" / f"{bundle.content_hash()}.json"
        if immutable.is_symlink() or not immutable.is_file() or immutable.read_bytes() != bundle.canonical_bytes():
            raise IntegrityError("immutable evidence bundle is missing or does not match current bundle")
        return bundle

    def _validation_admission_event_locked(
        self,
        bundle_hash: str,
        runner_trust_policy_hash: str,
    ) -> dict[str, Any]:
        matches = [
            event
            for event in self._verified_events_locked()
            if event["event_type"] == "validation_admitted"
        ]
        if len(matches) != 1:
            raise IntegrityError("a validated run must have exactly one validation_admitted event")
        event = matches[0]
        if event["payload"].get("bundle_hash") != bundle_hash:
            raise IntegrityError("current evidence bundle does not match the admitted bundle hash")
        if event["payload"].get("runner_trust_policy_hash") != runner_trust_policy_hash:
            raise IntegrityError("validation admission runner policy hash mismatch")
        return event

    def _validation_policy_event_locked(self, policy_hash: str) -> dict[str, Any]:
        matches = [
            event
            for event in self._verified_events_locked()
            if event["event_type"] == "validation_policy_registered"
        ]
        if len(matches) != 1:
            raise IntegrityError("run must have exactly one preregistered validation policy")
        event = matches[0]
        if event["payload"].get("policy_hash") != policy_hash:
            raise IntegrityError("active validation policy differs from the preregistered policy")
        self._verify_event_artifact_locked(
            str(event["payload"].get("policy_uri", "")),
            str(event["payload"].get("policy_sha256", "")),
            "preregistered validation policy",
        )
        return event

    def _verify_runner_attestations_locked(
        self,
        bundle: ValidationEvidenceBundle,
        runner_trust_store: ValidationRunnerTrustStore,
    ) -> None:
        runner_trust_store.assert_external_and_unchanged(self.vault_root)
        if bundle.runner_trust_policy_hash != runner_trust_store.source_hash:
            raise IntegrityError("evidence bundle runner policy does not match its historical policy hash")
        self._validation_policy_event_locked(runner_trust_store.source_hash)
        for gate_id, reference in bundle.artifacts.items():
            path = self.run_root / reference.artifact_uri
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, Mapping):
                raise IntegrityError(f"validation artifact must contain an object: {gate_id}")
            artifact = ValidationArtifact.from_dict(payload)
            runner_trust_store.verify(artifact)

    def apply_human_decision(
        self,
        record: HumanApprovalRecord,
        *,
        trust_store: ApprovalTrustStore,
        runner_trust_store: ValidationRunnerTrustStore,
        expected_manifest_hash: str,
    ) -> RunManifest:
        record.validate()
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            self._assert_cas(current, expected_manifest_hash)
            if current.promotion_status is not PromotionStatus.APPROVAL_REQUIRED:
                raise IllegalTransitionError("human decisions require promotion_status=approval_required")
            if current.validation_status is not ValidationStatus.PASS or not current.promotion_eligible:
                raise IllegalTransitionError("only a promotable, validated run can receive a human decision")
            bundle = self._load_current_bundle_locked()
            self._verify_runner_attestations_locked(bundle, runner_trust_store)
            if bundle.overall_status() is not ValidationStatus.PASS:
                raise IntegrityError("human decision cannot use a non-passing evidence bundle")
            expected = {
                "run_id": current.run_id,
                "manifest_identity_hash": current.identity_hash(),
                "candidate_id": bundle.candidate_id,
                "evidence_bundle_hash": bundle.content_hash(),
            }
            actual = {
                "run_id": record.run_id,
                "manifest_identity_hash": record.manifest_identity_hash,
                "candidate_id": record.candidate_id,
                "evidence_bundle_hash": record.evidence_bundle_hash,
            }
            if actual != expected:
                raise ApprovalError("approval is stale, replayed, or bound to another run/evidence bundle")
            admission_event = self._validation_admission_event_locked(
                bundle.content_hash(),
                bundle.runner_trust_policy_hash,
            )
            approval_time = _parse_utc(record.approved_at, "approved_at")
            if approval_time < max(
                _parse_utc(bundle.created_at, "bundle created_at"),
                _parse_utc(str(admission_event["created_at"]), "validation admitted_at"),
            ):
                raise ApprovalError("approval predates its evidence admission")
            approval_path = self.run_root / "approvals" / f"{record.approval_id}.json"
            trust_store.assert_external_and_unchanged(self.vault_root)
            trust_store.verify(record)
            approval_content = _canonical_bytes(record.to_dict())
            if approval_path.exists():
                if approval_path.is_symlink() or approval_path.read_bytes() != approval_content:
                    raise ApprovalError("approval ID has already been used with different content")
            else:
                self._write_file(approval_path, approval_content, exclusive=True)
            target = (
                PromotionStatus.APPROVED
                if record.decision is ApprovalDecision.APPROVED
                else PromotionStatus.REJECTED
            )
            updated = replace(current, promotion_status=target, updated_at=_utc_now())
            return self._publish_manifest_locked(
                before=current,
                after=updated,
                event_type="human_decision_verified",
                actor_type="authorized_human",
                actor_id=record.approved_by,
                payload={
                    "approval_id": record.approval_id,
                    "decision": record.decision.value,
                    "evidence_bundle_hash": record.evidence_bundle_hash,
                    "signer_key_id": record.signer_key_id,
                    "trust_policy_hash": trust_store.source_hash,
                    "approval_uri": approval_path.relative_to(self.run_root).as_posix(),
                    "approval_sha256": _hash_bytes(approval_content),
                    "resulting_promotion_status": target.value,
                },
            )

    def supersede(
        self,
        record: HumanSupersessionRecord,
        *,
        trust_store: ApprovalTrustStore,
        expected_manifest_hash: str,
    ) -> RunManifest:
        record.validate()
        replacement = ExperimentRunStore(self.vault_root, record.replacement_run_id).load()
        if replacement.parent_run_id != self.run_id or replacement.run_kind is not RunKind.RESEARCH:
            raise ApprovalError("replacement must be an explicit research child of the approved run")
        with self._locked():
            current = self._load_manifest_file(self.manifest_path)
            self._assert_cas(current, expected_manifest_hash)
            if current.promotion_status is not PromotionStatus.APPROVED:
                raise IllegalTransitionError("only approved runs can be superseded")
            bundle = self._load_current_bundle_locked()
            expected = {
                "run_id": current.run_id,
                "replacement_run_id": replacement.run_id,
                "manifest_identity_hash": current.identity_hash(),
                "evidence_bundle_hash": bundle.content_hash(),
            }
            actual = {
                "run_id": record.run_id,
                "replacement_run_id": record.replacement_run_id,
                "manifest_identity_hash": record.manifest_identity_hash,
                "evidence_bundle_hash": record.evidence_bundle_hash,
            }
            if actual != expected:
                raise ApprovalError("supersession record is stale or bound to another lineage")
            human_events = [
                event
                for event in self._verified_events_locked()
                if event["event_type"] == "human_decision_verified"
            ]
            if len(human_events) != 1 or _parse_utc(record.approved_at, "approved_at") < _parse_utc(
                str(human_events[0]["created_at"]), "approval event created_at"
            ):
                raise ApprovalError("supersession must follow the signed approval it replaces")
            trust_store.assert_external_and_unchanged(self.vault_root)
            trust_store.verify_supersession(record)
            supersession_path = (
                self.run_root / "approvals" / f"supersession-{record.supersession_id}.json"
            )
            content = _canonical_bytes(record.to_dict())
            if supersession_path.exists():
                if supersession_path.is_symlink() or supersession_path.read_bytes() != content:
                    raise ApprovalError("supersession ID has already been used with different content")
            else:
                self._write_file(supersession_path, content, exclusive=True)
            updated = replace(current, promotion_status=PromotionStatus.SUPERSEDED, updated_at=_utc_now())
            return self._publish_manifest_locked(
                before=current,
                after=updated,
                event_type="promotion_transition",
                actor_type="authorized_human",
                actor_id=record.approved_by,
                payload={
                    "from": PromotionStatus.APPROVED.value,
                    "to": PromotionStatus.SUPERSEDED.value,
                    "replacement_run_id": record.replacement_run_id,
                    "supersession_id": record.supersession_id,
                    "trust_policy_hash": trust_store.source_hash,
                    "supersession_uri": supersession_path.relative_to(self.run_root).as_posix(),
                    "supersession_sha256": _hash_bytes(content),
                },
            )

    def verify_integrity(
        self,
        *,
        trust_store: ApprovalTrustStore | Mapping[str, ApprovalTrustStore] | None = None,
        runner_trust_store: (
            ValidationRunnerTrustStore
            | Mapping[str, ValidationRunnerTrustStore]
            | None
        ) = None,
    ) -> dict[str, Any]:
        with self._locked():
            manifest = self._load_manifest_file(self.manifest_path)
            events = self._verified_events_locked()
            if not events or events[-1]["after_manifest_hash"] != manifest.content_hash():
                raise IntegrityError("manifest does not match the append-only event head")
            history_hashes: set[str] = set()
            for path in self._manifest_versions():
                version = self._load_manifest_file(path)
                digest = version.content_hash()
                if not path.stem.endswith(digest):
                    raise IntegrityError("manifest history filename hash mismatch")
                if version.identity_hash() != manifest.identity_hash():
                    raise IntegrityError("manifest history contains identity drift")
                history_hashes.add(digest)
            for event in events:
                if event["before_manifest_hash"] not in history_hashes or event["after_manifest_hash"] not in history_hashes:
                    raise IntegrityError("audit event references a missing manifest version")
                payload = event["payload"]
                if event["event_type"] == "data_admission_verified":
                    self._verify_event_artifact_locked(
                        str(payload.get("admission_uri", "")),
                        str(payload.get("admission_sha256", "")),
                        "verified data admission",
                    )
                    admission_payload = json.loads(
                        (self.run_root / str(payload["admission_uri"])).read_text(
                            encoding="utf-8"
                        )
                    )
                    receipt = admission_payload.get("verified_admission", {})
                    for field in (
                        "admission_subject_sha256", "plan_sha256",
                        "entitlement_attestation_sha256", "data_manifest_sha256",
                        "execution_contract_sha256", "human_release_sha256",
                        "admission_trust_policy_sha256", "release_trust_policy_sha256",
                    ):
                        if not isinstance(receipt, Mapping) or receipt.get(field) != payload.get(field):
                            raise IntegrityError(
                                "data-admission audit event differs from its evidence artifact"
                            )
                    manifest_bindings = {
                        "root_run_id": manifest.run_id,
                        "data_manifest_id": manifest.data_manifest_id,
                        "data_asof_utc": manifest.data_asof_utc,
                        "feature_manifest_hash": manifest.feature_manifest_hash,
                        "code_commit": manifest.code_commit,
                        "split_manifest_id": manifest.split_manifest_id,
                    }
                    if not isinstance(receipt, Mapping) or any(
                        receipt.get(field) != expected
                        for field, expected in manifest_bindings.items()
                    ):
                        raise IntegrityError(
                            "data-admission evidence differs from the replayed run manifest"
                        )
                elif event["event_type"] == "workflow_stage_completed":
                    self._verify_event_artifact_locked(
                        str(payload.get("note_uri", "")),
                        str(payload.get("note_sha256", "")),
                        "supporting note",
                    )
                elif event["event_type"] == "validation_policy_registered":
                    self._verify_event_artifact_locked(
                        str(payload.get("policy_uri", "")),
                        str(payload.get("policy_sha256", "")),
                        "preregistered validation policy",
                    )
                elif event["event_type"] == "validation_artifact_attached":
                    self._verify_event_artifact_locked(
                        str(payload.get("artifact_uri", "")),
                        str(payload.get("artifact_sha256", "")),
                        "validation artifact",
                    )
                elif event["event_type"] == "human_decision_verified":
                    self._verify_event_artifact_locked(
                        str(payload.get("approval_uri", "")),
                        str(payload.get("approval_sha256", "")),
                        "human approval",
                    )
                    event_trust_store = self._approval_trust_store_for_event(
                        trust_store,
                        str(payload.get("trust_policy_hash", "")),
                    )
                    event_trust_store.assert_external_and_unchanged(self.vault_root)
                    approval_path = self.run_root / str(payload["approval_uri"])
                    approval_payload = json.loads(approval_path.read_text(encoding="utf-8"))
                    if not isinstance(approval_payload, Mapping):
                        raise IntegrityError("human approval record must contain an object")
                    approval = HumanApprovalRecord.from_dict(approval_payload)
                    event_trust_store.verify(approval)
                    bundle = self._load_current_bundle_locked()
                    before = self._manifest_version_by_hash_locked(
                        str(event["before_manifest_hash"])
                    )
                    expected_approval = {
                        "run_id": self.run_id,
                        "approved_by": event["actor_id"],
                        "approval_id": payload.get("approval_id"),
                        "decision": payload.get("decision"),
                        "manifest_identity_hash": before.identity_hash(),
                        "candidate_id": bundle.candidate_id,
                        "evidence_bundle_hash": bundle.content_hash(),
                        "signer_key_id": payload.get("signer_key_id"),
                        "resulting_promotion_status": (
                            PromotionStatus.APPROVED.value
                            if approval.decision is ApprovalDecision.APPROVED
                            else PromotionStatus.REJECTED.value
                        ),
                    }
                    actual_approval = {
                        "run_id": approval.run_id,
                        "approved_by": approval.approved_by,
                        "approval_id": approval.approval_id,
                        "decision": approval.decision.value,
                        "manifest_identity_hash": approval.manifest_identity_hash,
                        "candidate_id": approval.candidate_id,
                        "evidence_bundle_hash": approval.evidence_bundle_hash,
                        "signer_key_id": approval.signer_key_id,
                        "resulting_promotion_status": payload.get(
                            "resulting_promotion_status"
                        ),
                    }
                    if actual_approval != expected_approval:
                        raise IntegrityError("signed human approval does not match its audit transition")
                    admission_event = self._validation_admission_event_locked(
                        bundle.content_hash(),
                        bundle.runner_trust_policy_hash,
                    )
                    if _parse_utc(approval.approved_at, "approved_at") < max(
                        _parse_utc(bundle.created_at, "bundle created_at"),
                        _parse_utc(
                            str(admission_event["created_at"]),
                            "validation admitted_at",
                        ),
                    ):
                        raise IntegrityError("signed human approval predates evidence admission")
                elif (
                    event["event_type"] == "promotion_transition"
                    and payload.get("to") == PromotionStatus.SUPERSEDED.value
                ):
                    self._verify_event_artifact_locked(
                        str(payload.get("supersession_uri", "")),
                        str(payload.get("supersession_sha256", "")),
                        "human supersession",
                    )
                    event_trust_store = self._approval_trust_store_for_event(
                        trust_store,
                        str(payload.get("trust_policy_hash", "")),
                    )
                    event_trust_store.assert_external_and_unchanged(self.vault_root)
                    supersession_path = self.run_root / str(payload["supersession_uri"])
                    supersession_payload = json.loads(supersession_path.read_text(encoding="utf-8"))
                    if not isinstance(supersession_payload, Mapping):
                        raise IntegrityError("human supersession record must contain an object")
                    supersession = HumanSupersessionRecord.from_dict(supersession_payload)
                    event_trust_store.verify_supersession(supersession)
                    bundle = self._load_current_bundle_locked()
                    before = self._manifest_version_by_hash_locked(
                        str(event["before_manifest_hash"])
                    )
                    expected_supersession = {
                        "run_id": self.run_id,
                        "replacement_run_id": payload.get("replacement_run_id"),
                        "supersession_id": payload.get("supersession_id"),
                        "manifest_identity_hash": before.identity_hash(),
                        "evidence_bundle_hash": bundle.content_hash(),
                        "approved_by": event["actor_id"],
                    }
                    actual_supersession = {
                        "run_id": supersession.run_id,
                        "replacement_run_id": supersession.replacement_run_id,
                        "supersession_id": supersession.supersession_id,
                        "manifest_identity_hash": supersession.manifest_identity_hash,
                        "evidence_bundle_hash": supersession.evidence_bundle_hash,
                        "approved_by": supersession.approved_by,
                    }
                    if actual_supersession != expected_supersession:
                        raise IntegrityError(
                            "signed human supersession does not match its audit transition"
                        )
                    human_events = [
                        item
                        for item in events
                        if item["event_type"] == "human_decision_verified"
                    ]
                    if len(human_events) != 1 or _parse_utc(
                        supersession.approved_at, "approved_at"
                    ) < _parse_utc(
                        str(human_events[0]["created_at"]),
                        "approval event created_at",
                    ):
                        raise IntegrityError("signed supersession predates the approval it replaces")
                    replacement = ExperimentRunStore(
                        self.vault_root, supersession.replacement_run_id
                    ).load()
                    if replacement.parent_run_id != self.run_id:
                        raise IntegrityError("supersession replacement lineage is invalid")
            bundle_hash = None
            admission_events = [
                event for event in events if event["event_type"] == "validation_admitted"
            ]
            if admission_events:
                bundle = self._load_current_bundle_locked()
                bundle_hash = bundle.content_hash()
                if bundle.manifest_identity_hash != manifest.identity_hash():
                    raise IntegrityError("evidence bundle manifest identity mismatch")
                self._validation_admission_event_locked(
                    bundle_hash,
                    bundle.runner_trust_policy_hash,
                )
                historical_runner_trust = self._runner_trust_store_for_event(
                    runner_trust_store, bundle.runner_trust_policy_hash
                )
                historical_runner_trust.assert_external_and_unchanged(self.vault_root)
                self._verify_runner_attestations_locked(bundle, historical_runner_trust)
                admission_event = admission_events[0]
                admission_before = self._manifest_version_by_hash_locked(
                    str(admission_event["before_manifest_hash"])
                )
                admission_after = self._manifest_version_by_hash_locked(
                    str(admission_event["after_manifest_hash"])
                )
                derived_validation, derived_promotion = self._derive_validation_outcome(
                    bundle,
                    admission_before,
                )
                if (
                    admission_event["payload"].get("candidate_id") != bundle.candidate_id
                    or admission_event["payload"].get("validation_status")
                    != derived_validation.value
                    or admission_event["payload"].get("promotion_status")
                    != derived_promotion.value
                    or admission_after.validation_status is not derived_validation
                    or admission_after.promotion_status is not derived_promotion
                ):
                    raise IntegrityError(
                        "validation admission outcome does not match the signed evidence bundle"
                    )
            elif manifest.validation_status is not ValidationStatus.NOT_RUN and not (
                manifest.validation_status is ValidationStatus.INCONCLUSIVE
                and manifest.promotion_status is PromotionStatus.PARKED_FOR_DATA
                and any(
                    event["event_type"] == "promotion_transition"
                    and event["actor_type"] == "controller"
                    for event in events
                )
            ):
                raise IntegrityError("validation state has no deterministic admission event")
            return {
                "run_id": self.run_id,
                "manifest_hash": manifest.content_hash(),
                "manifest_identity_hash": manifest.identity_hash(),
                "event_count": len(events),
                "manifest_version_count": len(self._manifest_versions()),
                "evidence_bundle_hash": bundle_hash,
                "status": "pass",
            }

    def _approval_trust_store_for_event(
        self,
        trust_store: ApprovalTrustStore | Mapping[str, ApprovalTrustStore] | None,
        policy_hash: str,
    ) -> ApprovalTrustStore:
        _require_sha256("historical approval trust policy hash", policy_hash)
        if trust_store is None:
            raise IntegrityError(
                "replaying a human decision requires its historical approval trust store"
            )
        if isinstance(trust_store, ApprovalTrustStore):
            selected = trust_store
        elif isinstance(trust_store, Mapping):
            selected = trust_store.get(policy_hash)
            if not isinstance(selected, ApprovalTrustStore):
                raise IntegrityError(
                    "historical approval trust-store mapping is missing the event policy"
                )
        else:
            raise TypeError("trust_store must be an ApprovalTrustStore or hash-keyed mapping")
        if selected.source_hash != policy_hash:
            raise IntegrityError("human event does not match its historical trust policy")
        return selected

    def _runner_trust_store_for_event(
        self,
        trust_store: (
            ValidationRunnerTrustStore
            | Mapping[str, ValidationRunnerTrustStore]
            | None
        ),
        policy_hash: str,
    ) -> ValidationRunnerTrustStore:
        _require_sha256("historical validation-runner trust policy hash", policy_hash)
        if trust_store is None:
            raise IntegrityError(
                "replaying validation requires its historical validation-runner trust store"
            )
        if isinstance(trust_store, ValidationRunnerTrustStore):
            selected = trust_store
        elif isinstance(trust_store, Mapping):
            selected = trust_store.get(policy_hash)
            if not isinstance(selected, ValidationRunnerTrustStore):
                raise IntegrityError(
                    "historical validation trust-store mapping is missing the event policy"
                )
        else:
            raise TypeError(
                "runner_trust_store must be a ValidationRunnerTrustStore or hash-keyed mapping"
            )
        if selected.source_hash != policy_hash:
            raise IntegrityError("validation event does not match its historical trust policy")
        return selected

    def _verify_event_artifact_locked(self, uri: str, expected_hash: str, kind: str) -> None:
        _require_sha256(f"{kind} sha256", expected_hash)
        path = self.run_root / uri
        if not uri or path.is_symlink() or not path.is_file():
            raise IntegrityError(f"{kind} referenced by the audit trail is missing or not regular")
        resolved = path.resolve()
        try:
            resolved.relative_to(self.run_root.resolve())
        except ValueError as exc:
            raise IntegrityError(f"{kind} referenced by the audit trail escaped the run root") from exc
        digest, _ = _hash_file(path)
        if digest != expected_hash:
            raise IntegrityError(f"{kind} referenced by the audit trail failed its hash check")
