from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4


SCHEMA_VERSION = 1
TERMINAL_STATUSES = {"success", "failure", "rejected", "duplicate", "timeout"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_PAYLOAD_KEYS = (
    "prompt",
    "response",
    "secret",
    "credential",
    "api_key",
    "authorization",
    "raw_data",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 hash")


def _safe_payload(value: Any, path: str = "payload") -> Any:
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).lower()
            if any(blocked in normalized for blocked in _FORBIDDEN_PAYLOAD_KEYS):
                raise ValueError(f"sensitive ledger field is forbidden: {path}.{key}")
            result[str(key)] = _safe_payload(item, f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_safe_payload(item, path) for item in value]
    if isinstance(value, str):
        if len(value) > 512:
            raise ValueError(f"ledger text is too long at {path}; store evidence by hash")
        return value
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported ledger payload value at {path}: {type(value).__name__}")


@dataclass(frozen=True)
class TrialReservation:
    trial_id: str
    idempotency_key: str
    campaign: str
    family: str
    parent_trial_id: str | None
    reserved_at: str
    created: bool


@dataclass(frozen=True)
class EvidenceRecord:
    sha256: str
    path: Path
    size_bytes: int


def _check_reservation_identity(row, item):
    for field, value in item.items():
        if row[field] != value:
            raise RuntimeError("existing reservation identity drifted: " + field)


class TrialLedger:
    """Transactional, append-oriented lifetime experiment registry."""

    def __init__(self, db_path: str | Path, evidence_root: str | Path) -> None:
        self.db_path = Path(db_path)
        self.evidence_root = Path(evidence_root)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.evidence_root.mkdir(parents=True, exist_ok=True)
        self._migrate()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 30000")
        return connection

    def _migrate(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_meta (
                    version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trials (
                    trial_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    campaign TEXT NOT NULL,
                    family TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    parent_trial_id TEXT REFERENCES trials(trial_id),
                    reserved_at TEXT NOT NULL,
                    code_hash TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    config_hash TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    model_hash TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trial_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trial_id TEXT NOT NULL REFERENCES trials(trial_id),
                    event_type TEXT NOT NULL,
                    status TEXT,
                    failure_class TEXT,
                    payload_json TEXT NOT NULL,
                    evidence_sha256 TEXT,
                    corrects_event_id INTEGER REFERENCES trial_events(event_id),
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    sha256 TEXT PRIMARY KEY,
                    relative_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS legacy_imports (
                    import_key TEXT PRIMARY KEY,
                    campaign TEXT NOT NULL,
                    source_uri TEXT NOT NULL,
                    source_hash TEXT NOT NULL,
                    attempt_count INTEGER NOT NULL CHECK(attempt_count >= 0),
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS ledger_imports (
                    source_key TEXT PRIMARY KEY,
                    source_uri TEXT NOT NULL,
                    source_fingerprint TEXT NOT NULL,
                    trial_count INTEGER NOT NULL CHECK(trial_count >= 0),
                    event_count INTEGER NOT NULL CHECK(event_count >= 0),
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS candidate_return_series (
                    trial_id TEXT PRIMARY KEY REFERENCES trials(trial_id),
                    candidate_id TEXT NOT NULL,
                    family TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL REFERENCES evidence(sha256),
                    observation_count INTEGER NOT NULL CHECK(observation_count > 0),
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS ix_trials_campaign_family
                    ON trials(campaign, family);
                CREATE INDEX IF NOT EXISTS ix_events_trial
                    ON trial_events(trial_id, event_id);
                """
            )
            row = connection.execute("SELECT version FROM schema_meta").fetchone()
            if row is None:
                connection.execute("INSERT INTO schema_meta(version) VALUES (?)", (SCHEMA_VERSION,))
            elif int(row["version"]) != SCHEMA_VERSION:
                raise RuntimeError(f"unsupported trial ledger schema: {row['version']}")

    def reserve_trial(
        self,
        *,
        idempotency_key: str,
        campaign: str,
        family: str,
        hypothesis_id: str,
        candidate_id: str,
        code_hash: str,
        data_hash: str,
        config_hash: str,
        prompt_hash: str,
        model_hash: str,
        parent_trial_id: str | None = None,
    ) -> TrialReservation:
        if not all(value.strip() for value in (idempotency_key, campaign, family, hypothesis_id, candidate_id)):
            raise ValueError("trial identity fields are required")
        for name, value in {
            "code_hash": code_hash,
            "data_hash": data_hash,
            "config_hash": config_hash,
            "prompt_hash": prompt_hash,
            "model_hash": model_hash,
        }.items():
            _require_sha256(name, value)
        trial_id = str(uuid4())
        reserved_at = _utc_now()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT * FROM trials WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                _check_reservation_identity(existing, {'idempotency_key': idempotency_key, 'campaign': campaign, 'family': family, 'hypothesis_id': hypothesis_id, 'candidate_id': candidate_id, 'parent_trial_id': parent_trial_id, 'code_hash': code_hash, 'data_hash': data_hash, 'config_hash': config_hash, 'prompt_hash': prompt_hash, 'model_hash': model_hash})
                connection.commit()
                return TrialReservation(
                    existing["trial_id"],
                    existing["idempotency_key"],
                    existing["campaign"],
                    existing["family"],
                    existing["parent_trial_id"],
                    existing["reserved_at"],
                    False,
                )
            if parent_trial_id is not None:
                parent = connection.execute(
                    "SELECT 1 FROM trials WHERE trial_id = ?", (parent_trial_id,)
                ).fetchone()
                if parent is None:
                    raise ValueError("parent trial must be reserved first")
            connection.execute(
                """
                INSERT INTO trials(
                    trial_id, idempotency_key, campaign, family, hypothesis_id,
                    candidate_id, parent_trial_id, reserved_at, code_hash, data_hash,
                    config_hash, prompt_hash, model_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    idempotency_key,
                    campaign,
                    family,
                    hypothesis_id,
                    candidate_id,
                    parent_trial_id,
                    reserved_at,
                    code_hash,
                    data_hash,
                    config_hash,
                    prompt_hash,
                    model_hash,
                ),
            )
            connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'reserved', 'reserved', NULL, '{}', NULL, NULL, ?)
                """,
                (trial_id, reserved_at),
            )
            connection.commit()
            return TrialReservation(trial_id, idempotency_key, campaign, family, parent_trial_id, reserved_at, True)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def reserve_trials_atomically(
        self,
        requests: list[Mapping[str, str | None]],
    ) -> tuple[TrialReservation, ...]:
        """Reserve one pre-registered cohort in a single all-or-nothing transaction."""

        if not requests:
            raise ValueError("atomic reservation batch cannot be empty")
        required = {
            "idempotency_key",
            "campaign",
            "family",
            "hypothesis_id",
            "candidate_id",
            "code_hash",
            "data_hash",
            "config_hash",
            "prompt_hash",
            "model_hash",
            "parent_trial_id",
        }
        normalized: list[dict[str, str | None]] = []
        for request in requests:
            if set(request) != required:
                raise ValueError("atomic reservation request fields are incomplete or unexpected")
            item = {key: request[key] for key in required}
            identity = (
                item["idempotency_key"],
                item["campaign"],
                item["family"],
                item["hypothesis_id"],
                item["candidate_id"],
            )
            if not all(isinstance(value, str) and value.strip() for value in identity):
                raise ValueError("trial identity fields are required")
            for name in ("code_hash", "data_hash", "config_hash", "prompt_hash", "model_hash"):
                value = item[name]
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a lowercase SHA-256 hash")
                _require_sha256(name, value)
            parent = item["parent_trial_id"]
            if parent is not None and (not isinstance(parent, str) or not parent.strip()):
                raise ValueError("parent_trial_id must be null or a non-empty string")
            normalized.append(item)
        keys = [str(item["idempotency_key"]) for item in normalized]
        if len(set(keys)) != len(keys):
            raise ValueError("atomic reservation idempotency keys must be unique")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing_rows = {
                row["idempotency_key"]: row
                for row in connection.execute(
                    f"SELECT * FROM trials WHERE idempotency_key IN ({','.join('?' for _ in keys)})",
                    keys,
                ).fetchall()
            }
            if existing_rows and len(existing_rows) != len(normalized):
                raise RuntimeError("atomic reservation batch is only partially present")
            if existing_rows:
                reservations: list[TrialReservation] = []
                for item in normalized:
                    row = existing_rows[str(item["idempotency_key"])]
                    _check_reservation_identity(row, item)
                    reservations.append(
                        TrialReservation(
                            row["trial_id"],
                            row["idempotency_key"],
                            row["campaign"],
                            row["family"],
                            row["parent_trial_id"],
                            row["reserved_at"],
                            False,
                        )
                    )
                connection.commit()
                return tuple(reservations)

            parent_ids = {item["parent_trial_id"] for item in normalized if item["parent_trial_id"]}
            if parent_ids:
                found = {
                    row["trial_id"]
                    for row in connection.execute(
                        f"SELECT trial_id FROM trials WHERE trial_id IN ({','.join('?' for _ in parent_ids)})",
                        tuple(parent_ids),
                    ).fetchall()
                }
                if found != parent_ids:
                    raise ValueError("every atomic reservation parent must already be reserved")

            reserved_at = _utc_now()
            reservations = []
            for item in normalized:
                trial_id = str(uuid4())
                connection.execute(
                    """
                    INSERT INTO trials(
                        trial_id, idempotency_key, campaign, family, hypothesis_id,
                        candidate_id, parent_trial_id, reserved_at, code_hash, data_hash,
                        config_hash, prompt_hash, model_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        trial_id,
                        item["idempotency_key"],
                        item["campaign"],
                        item["family"],
                        item["hypothesis_id"],
                        item["candidate_id"],
                        item["parent_trial_id"],
                        reserved_at,
                        item["code_hash"],
                        item["data_hash"],
                        item["config_hash"],
                        item["prompt_hash"],
                        item["model_hash"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO trial_events(
                        trial_id, event_type, status, failure_class, payload_json,
                        evidence_sha256, corrects_event_id, created_at
                    ) VALUES (?, 'reserved', 'reserved', NULL, '{}', NULL, NULL, ?)
                    """,
                    (trial_id, reserved_at),
                )
                reservations.append(
                    TrialReservation(
                        trial_id,
                        str(item["idempotency_key"]),
                        str(item["campaign"]),
                        str(item["family"]),
                        item["parent_trial_id"],
                        reserved_at,
                        True,
                    )
                )
            connection.commit()
            return tuple(reservations)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_outcome(
        self,
        trial_id: str,
        *,
        status: str,
        failure_class: str | None = None,
        payload: Mapping[str, Any] | None = None,
        evidence_sha256: str | None = None,
    ) -> int:
        if status not in TERMINAL_STATUSES:
            raise ValueError(f"unsupported terminal status: {status}")
        safe = _safe_payload(payload or {})
        if evidence_sha256 is not None:
            self.verify_evidence(evidence_sha256)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone() is None:
                raise KeyError(f"unreserved trial: {trial_id}")
            terminal = connection.execute(
                "SELECT event_id FROM trial_events WHERE trial_id = ? AND event_type = 'outcome'",
                (trial_id,),
            ).fetchone()
            if terminal is not None:
                raise RuntimeError("trial already has a terminal outcome; append a correction event")
            cursor = connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'outcome', ?, ?, ?, ?, NULL, ?)
                """,
                (
                    trial_id,
                    status,
                    failure_class,
                    json.dumps(safe, sort_keys=True, separators=(",", ":")),
                    evidence_sha256,
                    _utc_now(),
                ),
            )
            connection.commit()
            return int(cursor.lastrowid)
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def record_correction(
        self,
        trial_id: str,
        *,
        corrects_event_id: int,
        payload: Mapping[str, Any],
    ) -> int:
        safe = _safe_payload(payload)
        with self._connect() as connection:
            target = connection.execute(
                "SELECT 1 FROM trial_events WHERE event_id = ? AND trial_id = ?",
                (corrects_event_id, trial_id),
            ).fetchone()
            if target is None:
                raise KeyError("correction target does not exist for this trial")
            cursor = connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'correction', NULL, NULL, ?, NULL, ?, ?)
                """,
                (
                    trial_id,
                    json.dumps(safe, sort_keys=True, separators=(",", ":")),
                    corrects_event_id,
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_model_call(self, trial_id: str, record: Any) -> int:
        """Append a redacted model-call receipt; prompt/response bodies are never stored."""

        from quant_lab.research.contracts import ModelCallRecord

        if not isinstance(record, ModelCallRecord):
            raise TypeError("record must be a ModelCallRecord")
        record.validate()
        evidence = self.store_evidence(record.canonical_json().encode("utf-8"), extension=".json")
        safe = {
            "call_id": record.call_id,
            "provider": record.provider,
            "model": record.model,
            "status": record.status,
            "input_tokens": record.input_tokens,
            "output_tokens": record.output_tokens,
            "cost_usd": record.cost_usd,
            "stop_reason": record.stop_reason,
        }
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone() is None:
                raise KeyError(f"unreserved trial: {trial_id}")
            cursor = connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'model_call', ?, NULL, ?, ?, NULL, ?)
                """,
                (
                    trial_id,
                    record.status,
                    json.dumps(safe, sort_keys=True, separators=(",", ":")),
                    evidence.sha256,
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_specialist_review(self, trial_id: str, review: Any, record: Any) -> int:
        """Persist a typed review and redacted model-call lineage as immutable evidence."""

        from quant_lab.research.contracts import ModelCallRecord, SpecialistReview

        if not isinstance(review, SpecialistReview) or not isinstance(record, ModelCallRecord):
            raise TypeError("review and record must use the specialist and model-call contracts")
        review.validate()
        record.validate()
        bundle = json.dumps(
            {"review": review.to_dict(), "model_call": record.to_dict()},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        evidence = self.store_evidence(bundle, extension=".json")
        payload = {
            "review_id": review.review_id,
            "task_id": review.task_id,
            "specialist_id": review.specialist_id,
            "verdict": review.verdict,
            "veto": review.veto,
            "call_id": record.call_id,
            "model": record.model,
            "cost_usd": record.cost_usd,
        }
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone() is None:
                raise KeyError(f"unreserved trial: {trial_id}")
            cursor = connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'specialist_review', ?, NULL, ?, ?, NULL, ?)
                """,
                (
                    trial_id,
                    review.verdict,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    evidence.sha256,
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_specialist_failure(
        self,
        trial_id: str,
        *,
        task_id: str,
        specialist_id: str,
        failure_code: str,
        request_fingerprint: str,
        record: Any | None = None,
    ) -> int:
        """Record a failed review and an optional redacted provider receipt."""

        if not all(value.strip() for value in (task_id, specialist_id, failure_code)):
            raise ValueError("failed review identity and failure code are required")
        _require_sha256("request_fingerprint", request_fingerprint)
        evidence = None
        if record is not None:
            from quant_lab.research.contracts import ModelCallRecord

            if not isinstance(record, ModelCallRecord):
                raise TypeError("failure receipt must be a ModelCallRecord")
            record.validate()
            evidence = self.store_evidence(
                record.canonical_json().encode("utf-8"), extension=".json"
            )
        payload = {
            "task_id": task_id,
            "specialist_id": specialist_id,
            "request_fingerprint": request_fingerprint,
        }
        if record is not None:
            payload.update(
                {
                    "call_id": record.call_id,
                    "provider": record.provider,
                    "model": record.model,
                    "status": record.status,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                    "cost_usd": record.cost_usd,
                    "stop_reason": record.stop_reason,
                }
            )
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone() is None:
                raise KeyError(f"unreserved trial: {trial_id}")
            cursor = connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'specialist_review', 'failure', ?, ?, ?, NULL, ?)
                """,
                (
                    trial_id,
                    failure_code,
                    json.dumps(payload, sort_keys=True, separators=(",", ":")),
                    evidence.sha256 if evidence else None,
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_panel_disposition(self, trial_id: str, payload: Mapping[str, Any]) -> int:
        """Append the deterministic panel disposition and disagreement summary."""

        safe = _safe_payload(payload)
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone() is None:
                raise KeyError(f"unreserved trial: {trial_id}")
            cursor = connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'panel_disposition', ?, NULL, ?, NULL, NULL, ?)
                """,
                (
                    trial_id,
                    (
                        "pass"
                        if bool(safe.get("progression_allowed"))
                        else "backtest_only"
                        if bool(safe.get("backtest_allowed"))
                        else "blocked"
                    ),
                    json.dumps(safe, sort_keys=True, separators=(",", ":")),
                    _utc_now(),
                ),
            )
            return int(cursor.lastrowid)

    def record_research_step(
        self,
        trial_id: str,
        *,
        event_type: str,
        status: str,
        payload: Mapping[str, Any],
        evidence_content: bytes | None = None,
    ) -> int:
        """Append one controller transition with optional content-addressed evidence."""

        allowed = {
            "hypothesis_intake",
            "candidate_extracted",
            "backtest",
            "institutional_audit",
            "candidate_returns",
            "dry_run",
            "mutation_created",
            "decay_monitor",
            "stop",
        }
        if event_type not in allowed or not status.strip():
            raise ValueError("unsupported research controller event")
        safe = _safe_payload(payload)
        evidence = self.store_evidence(evidence_content, extension=".json") if evidence_content else None
        with self._connect() as connection:
            if connection.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone() is None:
                raise KeyError(f"unreserved trial: {trial_id}")
            cursor = connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, ?, ?, NULL, ?, ?, NULL, ?)
                """,
                (
                    trial_id,
                    event_type,
                    status,
                    json.dumps(safe, sort_keys=True, separators=(",", ":")),
                    evidence.sha256 if evidence else None,
                    _utc_now(),
                ),
            )
        return int(cursor.lastrowid)

    def record_candidate_returns(
        self,
        trial_id: str,
        *,
        candidate_id: str,
        family: str,
        returns: Any,
    ) -> EvidenceRecord:
        """Persist one deterministic net-return series for lifetime cross-candidate tests."""

        import math

        import pandas as pd

        if not isinstance(returns, pd.Series) or not isinstance(returns.index, pd.DatetimeIndex):
            raise TypeError("candidate returns require a pandas Series with DatetimeIndex")
        series = returns.astype(float).sort_index()
        if len(series) < 16 or series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError("candidate return series is too short, duplicated, or unsorted")
        if series.isna().any() or not all(math.isfinite(float(value)) for value in series):
            raise ValueError("candidate return series must be finite")
        payload = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "family": family,
            "observations": [
                [timestamp.isoformat(), float(value)] for timestamp, value in series.items()
            ],
        }
        content = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        evidence = self.store_evidence(content, extension=".json")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute("SELECT 1 FROM trials WHERE trial_id = ?", (trial_id,)).fetchone() is None:
                raise KeyError(f"unreserved trial: {trial_id}")
            existing = connection.execute(
                "SELECT evidence_sha256 FROM candidate_return_series WHERE trial_id = ?", (trial_id,)
            ).fetchone()
            if existing is not None:
                if existing["evidence_sha256"] != evidence.sha256:
                    raise RuntimeError("candidate returns changed after their first persistence")
                connection.commit()
                return evidence
            created_at = _utc_now()
            connection.execute(
                """
                INSERT INTO candidate_return_series(
                    trial_id, candidate_id, family, evidence_sha256, observation_count,
                    start_date, end_date, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trial_id,
                    candidate_id,
                    family,
                    evidence.sha256,
                    len(series),
                    series.index.min().isoformat(),
                    series.index.max().isoformat(),
                    created_at,
                ),
            )
            connection.execute(
                """
                INSERT INTO trial_events(
                    trial_id, event_type, status, failure_class, payload_json,
                    evidence_sha256, corrects_event_id, created_at
                ) VALUES (?, 'candidate_returns', 'persisted', NULL, ?, ?, NULL, ?)
                """,
                (
                    trial_id,
                    json.dumps(
                        {
                            "candidate_id": candidate_id,
                            "family": family,
                            "observation_count": len(series),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    evidence.sha256,
                    created_at,
                ),
            )
            connection.commit()
            return evidence
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def candidate_return_matrix(self, *, campaign: str = "malaysia") -> Any:
        """Rebuild active returns; corrected invalid results remain in immutable evidence."""

        import pandas as pd

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.* FROM candidate_return_series r
                JOIN trials t ON t.trial_id = r.trial_id
                WHERE t.campaign = ? ORDER BY t.reserved_at, r.trial_id
                """,
                (campaign,),
            ).fetchall()
            corrections = connection.execute(
                """
                SELECT target.trial_id, correction.payload_json
                FROM trial_events correction
                JOIN trial_events target ON target.event_id = correction.corrects_event_id
                JOIN trials t ON t.trial_id = target.trial_id
                WHERE correction.event_type = 'correction'
                  AND target.event_type = 'candidate_returns'
                  AND t.campaign = ?
                """,
                (campaign,),
            ).fetchall()
        invalidated = {
            row["trial_id"]
            for row in corrections
            if bool(json.loads(row["payload_json"]).get("invalidates_candidate_returns"))
        }
        series: list[pd.Series] = []
        for row in rows:
            if row["trial_id"] in invalidated:
                continue
            evidence = self.verify_evidence(row["evidence_sha256"])
            payload = json.loads(evidence.path.read_text(encoding="utf-8"))
            observations = payload.get("observations", [])
            index = pd.DatetimeIndex([item[0] for item in observations])
            values = [float(item[1]) for item in observations]
            label = f"{row['candidate_id']}::{row['trial_id']}"
            series.append(pd.Series(values, index=index, name=label, dtype=float))
        return pd.concat(series, axis=1).sort_index().fillna(0.0) if series else pd.DataFrame()

    def invalidate_candidate_returns(self, trial_id: str, *, reason_code: str) -> int:
        """Append a correction that removes an invalid series from active analysis."""

        if not re.fullmatch(r"[A-Z][A-Z0-9_]{2,80}", reason_code):
            raise ValueError("candidate-return invalidation requires an uppercase reason code")
        with self._connect() as connection:
            target = connection.execute(
                """
                SELECT event_id FROM trial_events
                WHERE trial_id = ? AND event_type = 'candidate_returns'
                ORDER BY event_id DESC LIMIT 1
                """,
                (trial_id,),
            ).fetchone()
            if target is None:
                raise KeyError("trial has no candidate returns to invalidate")
            corrections = connection.execute(
                """
                SELECT event_id, payload_json FROM trial_events
                WHERE trial_id = ? AND event_type = 'correction' AND corrects_event_id = ?
                ORDER BY event_id
                """,
                (trial_id, int(target["event_id"])),
            ).fetchall()
        for correction in corrections:
            payload = json.loads(correction["payload_json"])
            if bool(payload.get("invalidates_candidate_returns")):
                if payload.get("reason_code") != reason_code:
                    raise RuntimeError("candidate returns were already invalidated for another reason")
                return int(correction["event_id"])
        return self.record_correction(
            trial_id,
            corrects_event_id=int(target["event_id"]),
            payload={
                "invalidates_candidate_returns": True,
                "reason_code": reason_code,
            },
        )

    def store_evidence(self, content: bytes, *, extension: str = ".bin") -> EvidenceRecord:
        if not content:
            raise ValueError("evidence must not be empty")
        if not re.fullmatch(r"\.[a-z0-9]{1,10}", extension):
            raise ValueError("evidence extension is invalid")
        digest = _sha256_bytes(content)
        relative = Path(digest[:2]) / f"{digest}{extension}"
        target = self.evidence_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and _sha256_bytes(target.read_bytes()) != digest:
            raise RuntimeError("existing evidence content does not match its address")
        if not target.exists():
            target.write_bytes(content)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO evidence(sha256, relative_path, size_bytes, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (digest, str(relative), len(content), _utc_now()),
            )
        return EvidenceRecord(digest, target, len(content))

    def verify_evidence(self, sha256: str) -> EvidenceRecord:
        _require_sha256("evidence sha256", sha256)
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM evidence WHERE sha256 = ?", (sha256,)).fetchone()
        if row is None:
            raise FileNotFoundError(f"unknown evidence: {sha256}")
        path = self.evidence_root / row["relative_path"]
        if not path.exists():
            raise FileNotFoundError(f"missing evidence file: {sha256}")
        content = path.read_bytes()
        if _sha256_bytes(content) != sha256 or len(content) != int(row["size_bytes"]):
            raise RuntimeError(f"evidence integrity failure: {sha256}")
        return EvidenceRecord(sha256, path, len(content))

    def import_legacy_malaysia_sweep(self, manifest_path: str | Path) -> bool:
        path = Path(manifest_path)
        content = path.read_bytes()
        payload = json.loads(content)
        if not isinstance(payload, Mapping):
            raise TypeError("legacy sweep manifest must be an object")
        run_id = str(payload.get("run_id", ""))
        count = int(payload.get("candidate_count", -1))
        if not run_id.startswith("MALAYSIA-SWEEP-") or count < 3330:
            raise ValueError("legacy Malaysia sweep manifest is incomplete")
        import_key = f"malaysia_sweep:{run_id}"
        source_hash = _sha256_bytes(content)
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT source_hash, attempt_count FROM legacy_imports WHERE import_key = ?",
                (import_key,),
            ).fetchone()
            if existing is not None:
                if existing["source_hash"] != source_hash or int(existing["attempt_count"]) != count:
                    raise RuntimeError("legacy import key conflicts with prior provenance")
                connection.commit()
                return False
            connection.execute(
                """
                INSERT INTO legacy_imports(
                    import_key, campaign, source_uri, source_hash, attempt_count, imported_at
                ) VALUES (?, 'malaysia', ?, ?, ?, ?)
                """,
                (import_key, str(path.resolve()), source_hash, count, _utc_now()),
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def import_completed_ledger(
        self,
        source_db_path: str | Path,
        source_evidence_root: str | Path,
        *,
        source_key: str,
    ) -> bool:
        """Merge one immutable completed campaign ledger into this lifetime vault."""

        source_db = Path(source_db_path).resolve()
        source_evidence = Path(source_evidence_root).resolve()
        if source_db == self.db_path.resolve():
            raise ValueError("source ledger must differ from the lifetime vault")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{2,160}", source_key):
            raise ValueError("source_key is invalid")
        source = sqlite3.connect(source_db, timeout=30.0)
        source.row_factory = sqlite3.Row
        try:
            version = source.execute("SELECT version FROM schema_meta").fetchone()
            if version is None or int(version["version"]) != SCHEMA_VERSION:
                raise RuntimeError("source ledger schema is unsupported")
            trials = [dict(row) for row in source.execute("SELECT * FROM trials ORDER BY reserved_at, trial_id")]
            events = [dict(row) for row in source.execute("SELECT * FROM trial_events ORDER BY event_id")]
            evidence_rows = [dict(row) for row in source.execute("SELECT * FROM evidence ORDER BY sha256")]
            legacy_rows = [dict(row) for row in source.execute("SELECT * FROM legacy_imports ORDER BY import_key")]
        finally:
            source.close()
        terminal_trials = {
            item["trial_id"] for item in events if item["event_type"] == "outcome" and item["status"] in TERMINAL_STATUSES
        }
        if {item["trial_id"] for item in trials} != terminal_trials:
            raise ValueError("only ledgers whose every trial is terminal may enter the lifetime vault")
        fingerprint = _sha256_bytes(
            json.dumps(
                {"trials": trials, "events": events, "evidence": evidence_rows, "legacy": legacy_rows},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        with self._connect() as connection:
            existing_import = connection.execute(
                "SELECT source_fingerprint FROM ledger_imports WHERE source_key = ?", (source_key,)
            ).fetchone()
        if existing_import is not None:
            if existing_import["source_fingerprint"] != fingerprint:
                raise RuntimeError("ledger import source changed after its first import")
            return False

        for item in evidence_rows:
            relative = Path(item["relative_path"])
            path = (source_evidence / relative).resolve()
            if source_evidence not in path.parents:
                raise ValueError("source evidence path escaped its root")
            content = path.read_bytes()
            if _sha256_bytes(content) != item["sha256"] or len(content) != int(item["size_bytes"]):
                raise RuntimeError(f"source evidence integrity failure: {item['sha256']}")
            stored = self.store_evidence(content, extension=relative.suffix)
            if stored.sha256 != item["sha256"]:
                raise RuntimeError("content-addressed evidence changed during import")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            conflict = connection.execute(
                "SELECT source_fingerprint FROM ledger_imports WHERE source_key = ?", (source_key,)
            ).fetchone()
            if conflict is not None:
                if conflict["source_fingerprint"] != fingerprint:
                    raise RuntimeError("ledger import source changed during concurrent import")
                connection.commit()
                return False
            for item in legacy_rows:
                existing = connection.execute(
                    "SELECT source_hash, attempt_count FROM legacy_imports WHERE import_key = ?",
                    (item["import_key"],),
                ).fetchone()
                if existing is None:
                    connection.execute(
                        """
                        INSERT INTO legacy_imports(
                            import_key, campaign, source_uri, source_hash, attempt_count, imported_at
                        ) VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            item["import_key"], item["campaign"], item["source_uri"], item["source_hash"],
                            item["attempt_count"], item["imported_at"],
                        ),
                    )
                elif existing["source_hash"] != item["source_hash"] or int(existing["attempt_count"]) != int(item["attempt_count"]):
                    raise RuntimeError("legacy denominator conflicts during ledger import")

            pending = {item["trial_id"]: item for item in trials}
            while pending:
                progressed = False
                for trial_id, item in list(pending.items()):
                    parent = item["parent_trial_id"]
                    if parent is not None and connection.execute(
                        "SELECT 1 FROM trials WHERE trial_id = ?", (parent,)
                    ).fetchone() is None:
                        continue
                    id_conflict = connection.execute(
                        "SELECT 1 FROM trials WHERE trial_id = ? OR idempotency_key = ?",
                        (trial_id, item["idempotency_key"]),
                    ).fetchone()
                    if id_conflict is not None:
                        raise RuntimeError("trial identity already exists in lifetime vault")
                    columns = tuple(item)
                    connection.execute(
                        f"INSERT INTO trials({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                        tuple(item[column] for column in columns),
                    )
                    del pending[trial_id]
                    progressed = True
                if not progressed:
                    raise RuntimeError("source ledger contains unresolved parent lineage")

            event_id_map: dict[int, int] = {}
            for item in events:
                old_event_id = int(item["event_id"])
                correction = item["corrects_event_id"]
                mapped_correction = event_id_map.get(int(correction)) if correction is not None else None
                if correction is not None and mapped_correction is None:
                    raise RuntimeError("source correction event precedes its target")
                cursor = connection.execute(
                    """
                    INSERT INTO trial_events(
                        trial_id, event_type, status, failure_class, payload_json,
                        evidence_sha256, corrects_event_id, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item["trial_id"], item["event_type"], item["status"], item["failure_class"],
                        item["payload_json"], item["evidence_sha256"], mapped_correction, item["created_at"],
                    ),
                )
                event_id_map[old_event_id] = int(cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO ledger_imports(
                    source_key, source_uri, source_fingerprint, trial_count, event_count, imported_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source_key, str(source_db), fingerprint, len(trials), len(events), _utc_now()),
            )
            connection.commit()
            return True
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def lifetime_counts(self, *, campaign: str | None = None, family: str | None = None) -> dict[str, int]:
        clauses: list[str] = []
        params: list[str] = []
        if campaign is not None:
            clauses.append("campaign = ?")
            params.append(campaign)
        if family is not None:
            clauses.append("family = ?")
            params.append(family)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._connect() as connection:
            reserved = int(connection.execute(f"SELECT COUNT(*) FROM trials{where}", params).fetchone()[0])
            status_rows = connection.execute(
                f"""
                SELECT e.status, COUNT(*) AS count
                FROM trial_events e JOIN trials t ON t.trial_id = e.trial_id
                WHERE e.event_type = 'outcome'
                {''.join(f' AND t.{clause}' for clause in clauses)}
                GROUP BY e.status
                """,
                params,
            ).fetchall()
            legacy_params: list[str] = []
            legacy_where = ""
            if campaign is not None:
                legacy_where = " WHERE campaign = ?"
                legacy_params.append(campaign)
            legacy = int(
                connection.execute(
                    f"SELECT COALESCE(SUM(attempt_count), 0) FROM legacy_imports{legacy_where}",
                    legacy_params,
                ).fetchone()[0]
            )
        result = {status: 0 for status in sorted(TERMINAL_STATUSES)}
        result.update({row["status"]: int(row["count"]) for row in status_rows})
        result.update({"reserved": reserved, "legacy": legacy, "lifetime_attempts": reserved + legacy})
        return result

    def lineage(self, trial_id: str) -> list[str]:
        result: list[str] = []
        current: str | None = trial_id
        with self._connect() as connection:
            while current is not None:
                row = connection.execute(
                    "SELECT trial_id, parent_trial_id FROM trials WHERE trial_id = ?", (current,)
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown trial in lineage: {current}")
                result.append(row["trial_id"])
                current = row["parent_trial_id"]
        return list(reversed(result))

    def replay_manifest(self, trial_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            trial = connection.execute("SELECT * FROM trials WHERE trial_id = ?", (trial_id,)).fetchone()
            if trial is None:
                raise KeyError(f"unknown trial: {trial_id}")
            events = connection.execute(
                "SELECT * FROM trial_events WHERE trial_id = ? ORDER BY event_id", (trial_id,)
            ).fetchall()
        return {
            "trial": dict(trial),
            "events": [dict(event) for event in events],
        }
