"""Versioned completion package for findings-registry hygiene steps 9--11.

The package is deliberately sidecar-only.  It never writes into the historical
experiment vault.  A build contains:

* one migration-backed manifest projection for every historical experiment;
* a constrained SQLite findings registry and exact evidence lineage;
* an explicit policy for every Markdown file in the workspace;
* a local-only, deduplicated FTS/vector index of canonical research artifacts;
* fail-closed LSEG rerun-readiness plans for the three parked lineages.

All output is content addressed, atomically published, and can be made locally
read-only.  The source v2 overlay remains the authority for historical
classification; these projections do not rewrite history or fabricate approval.
"""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from quant_lab.research.control_plane import (
    ApprovalTrustStore,
    ExperimentRunStore,
    REQUIRED_VALIDATION_GATES,
    ValidationRunnerTrustStore,
    WORKFLOW_STAGES,
)
from quant_lab.research.data_admission import (
    DataAdmissionTrustStore,
    DataReleaseTrustStore,
    VerifiedDataAdmission,
    frozen_research_config_payload,
    frozen_validation_policy_payload,
    verify_archived_data_admission,
    verify_data_admission,
)


SCHEMA_VERSION = 1
HISTORICAL_MARKDOWN_BASELINE = 10_142
EXPECTED_EXPERIMENT_COUNT = 853
EXPECTED_EVIDENCE_COUNT = 2_557
EXPECTED_CLASSIFICATIONS = {
    "test_or_synthetic_quarantined": 841,
    "genuine_rejected_closed": 6,
    "superseded_historical_version": 3,
    "parked_awaiting_point_in_time_data": 3,
}
EXPECTED_PARKED_IDS = (
    "ASEAN-20260704-232302",
    "ASEAN-CONSTRAINED-OPT",
    "MALAYSIA-20260705-082423",
)
REQUIRED_LSEG_DATA = (
    "licensed_point_in_time_universe",
    "delistings",
    "corporate_actions",
    "tradability",
    "liquidity",
    "vendor_publication_and_effective_timestamps",
)
REQUIRED_RERUN_CONTROLS = (
    "frozen_feature_manifest",
    "frozen_model_configuration",
    "sealed_out_of_sample_period",
)
POLICY_RULE_VERSION = "canonical-research-artifact-policy-v1"
CHUNKER_VERSION = "markdown-heading-stream-v1"
RECORD_SOURCE_HISTORICAL = "historical_overlay_v2"

RERUN_RESEARCH_CONTRACTS: dict[str, dict[str, Any]] = {
    "ASEAN-20260704-232302": {
        "hypothesis": "A long-only bottom-volatility ASEAN portfolio retains positive net risk-adjusted return after point-in-time universe repair and 15 bps costs.",
        "candidate_specification": {
            "model_name": "asean_long_top20_lowvol_126",
            "signal": "trailing_126_trading_day_realized_volatility_ascending",
            "allocator": "long_only_top_20_percent",
            "rebalance_frequency": "monthly",
            "cost_bps": 15.0,
        },
        "market_scope": ["XKLS", "XSES", "XIDX", "XBKK", "XPHS", "XSTC", "XHNX"],
        "market_scope_decision": "Use all six legacy ASEAN countries evidenced by the run (Malaysia, Singapore, Indonesia, Thailand, Philippines, Vietnam); this intentionally supersedes the current three-country starter-universe code.",
        "currencies": ["MYR", "SGD", "IDR", "THB", "PHP", "VND", "USD"],
        "replication_window": {"start": "2021-07-04", "end": "2026-07-02"},
        "minimum_lookback_trading_days": 126,
        "target": "next_rebalance_period_total_return_usd",
        "label_horizon": "one_month",
        "feature_sources": [
            "quant_lab/src/quant_lab/research/asean.py",
            "quant_lab/src/quant_lab/data/asean_universe.py",
        ],
    },
    "ASEAN-CONSTRAINED-OPT": {
        "hypothesis": "Country-neutral 252-day ASEAN momentum remains feasible and positive after point-in-time universe repair, monthly rebalancing, six hard portfolio constraints, and 15 bps costs.",
        "candidate_specification": {
            "model_name": "asean_ctry_long_top10_mom_252",
            "signal": "trailing_252_trading_day_momentum",
            "allocator": "country_neutral_long_only_top_10_percent",
            "rebalance_frequency": "monthly",
            "target_volatility": 0.12,
            "max_leverage": 1.5,
            "cost_bps": 15.0,
            "constraints": {
                "max_drawdown_floor": -0.20,
                "max_turnover": 3.0,
                "max_country_weight": 0.40,
                "max_asset_weight": 0.20,
                "cost_stress_multiple": 2.0,
                "minimum_deflated_sharpe": 0.80,
            },
            "additional_required_checks": ["monthly_return_dominance_below_30_percent", "regime_conditioned_stability"],
        },
        "market_scope": ["XKLS", "XSES", "XIDX", "XBKK", "XPHS", "XSTC", "XHNX"],
        "market_scope_decision": "Use the same six-country legacy ASEAN evidence scope; do not inherit the current three-country starter universe.",
        "currencies": ["MYR", "SGD", "IDR", "THB", "PHP", "VND", "USD"],
        "replication_window": {"start": "2021-07-04", "end": "2026-07-02"},
        "minimum_lookback_trading_days": 252,
        "target": "next_rebalance_period_total_return_usd",
        "label_horizon": "one_month",
        "feature_sources": [
            "quant_lab/examples/run_constrained_v1_5.py",
            "quant_lab/src/quant_lab/optimization/net_ir_optimizer.py",
        ],
    },
    "MALAYSIA-20260705-082423": {
        "hypothesis": "A long-only bottom-volatility Bursa Malaysia portfolio retains positive net risk-adjusted return after replacing the curated current-survivor basket with an effective-dated point-in-time universe and 15 bps costs.",
        "candidate_specification": {
            "model_name": "malaysia_long_top33_lowvol_63",
            "signal": "trailing_63_trading_day_realized_volatility_ascending",
            "allocator": "long_only_top_33_percent",
            "rebalance_frequency": "monthly",
            "cost_bps": 15.0,
        },
        "market_scope": ["XKLS"],
        "market_scope_decision": "Use the complete effective-dated Bursa Malaysia listing universe, including inactive and delisted securities; never reuse the curated 30-stock survivor basket.",
        "currencies": ["MYR", "USD"],
        "replication_window": {"start": "2021-07-04", "end": "2026-07-02"},
        "minimum_lookback_trading_days": 63,
        "target": "next_rebalance_period_total_return_usd",
        "label_horizon": "one_month",
        "feature_sources": [
            "quant_lab/src/quant_lab/research/malaysia.py",
            "quant_lab/examples/run_malaysia_backtest.py",
        ],
    },
}
REQUIRED_LSEG_FIELDS = (
    "stable_instrument_id",
    "primary_exchange_mic",
    "listing_effective_at",
    "delisting_effective_at",
    "security_status_effective_at",
    "currency",
    "unadjusted_close",
    "total_return_or_adjustment_factors",
    "cash_dividends",
    "stock_splits",
    "rights_and_other_corporate_actions",
    "trading_volume",
    "shares_outstanding_or_free_float",
    "publication_timestamp_utc",
    "effective_timestamp_utc",
    "revision_or_vintage_id",
    "ingestion_timestamp_utc",
)
VALIDATION_POLICY_SOURCES = (
    "quant_lab/src/quant_lab/validation/splits.py",
    "quant_lab/src/quant_lab/validation/cpcv.py",
    "quant_lab/src/quant_lab/validation/overfit.py",
    "quant_lab/src/quant_lab/validation/metrics.py",
    "quant_lab/src/quant_lab/validation/holdout.py",
    "quant_lab/src/quant_lab/research/_control_plane_contracts.py",
)
LSEG_EXECUTION_SOURCES = (
    "quant_lab/src/quant_lab/__init__.py",
    "quant_lab/src/quant_lab/data/__init__.py",
    "quant_lab/src/quant_lab/research/__init__.py",
    "quant_lab/src/quant_lab/research/_control_plane_contracts.py",
    "quant_lab/src/quant_lab/research/_control_plane_store.py",
    "quant_lab/src/quant_lab/research/_control_plane_trust.py",
    "quant_lab/src/quant_lab/research/control_plane.py",
    "quant_lab/src/quant_lab/research/data_admission.py",
    "quant_lab/src/quant_lab/data/lseg_pit_adapter.py",
    "quant_lab/src/quant_lab/research/lseg_rerun.py",
)
MANIFEST_FIELDS = (
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
)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SEARCH_TERM_RE = re.compile(r"[A-Za-z0-9_]{2,}")


class CompletionError(RuntimeError):
    """Raised when a completion invariant is not satisfied."""


@dataclass(frozen=True)
class CompletionInputs:
    workspace_root: Path
    output_root: Path
    overlay_pointer: Path
    source_findings_database: Path
    paper_database: Path | None = None
    books_database: Path | None = None
    operations_database: Path | None = None
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    historical_markdown_baseline: int = HISTORICAL_MARKDOWN_BASELINE
    approval_trust_stores: Mapping[str, ApprovalTrustStore] | None = None
    validation_runner_trust_stores: (
        Mapping[str, ValidationRunnerTrustStore] | None
    ) = None
    data_admission_trust_stores: Mapping[str, DataAdmissionTrustStore] | None = None
    data_release_trust_stores: Mapping[str, DataReleaseTrustStore] | None = None

    @property
    def experiment_root(self) -> Path:
        return self.workspace_root / "quant_lab" / "vault" / "30_experiments"


@dataclass(frozen=True)
class CompletionResult:
    completion_id: str
    completion_dir: Path
    manifest_path: Path
    verification: Mapping[str, bool]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def semantic_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_experiment_id(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise CompletionError(f"unsafe experiment_id: {value!r}")
    return value


def _load_overlay(inputs: CompletionInputs) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    pointer = json.loads(inputs.overlay_pointer.read_text(encoding="utf-8"))
    migration_dir = Path(pointer["migration_dir"]).resolve()
    manifest_path = migration_dir / "overlay-manifest.json"
    if sha256_file(manifest_path) != pointer["overlay_manifest_sha256"]:
        raise CompletionError("overlay pointer does not match overlay manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    overlay_path = migration_dir / manifest["overlay_json"]["file"]
    if sha256_file(overlay_path) != manifest["overlay_json"]["sha256"]:
        raise CompletionError("overlay JSON hash mismatch")
    payload = json.loads(overlay_path.read_text(encoding="utf-8"))
    if payload["classification_counts"] != EXPECTED_CLASSIFICATIONS:
        raise CompletionError("overlay classification partition is not the approved v2 baseline")
    if manifest["experiment_count"] != EXPECTED_EXPERIMENT_COUNT:
        raise CompletionError("overlay does not cover exactly 853 experiments")
    return migration_dir, manifest, payload


def _source_experiments(path: Path) -> dict[str, dict[str, Any]]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    try:
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise CompletionError(f"source findings database integrity failed: {integrity}")
        return {
            row["experiment_id"]: dict(row)
            for row in connection.execute("SELECT * FROM experiments ORDER BY experiment_id")
        }
    finally:
        connection.close()


def _experiment_folders(root: Path) -> list[str]:
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def _discover_control_plane_runs(
    inputs: CompletionInputs, historical_ids: set[str]
) -> list[dict[str, Any]]:
    """Replay future governed runs into the canonical historical+live union.

    Every non-historical folder must be a genuine control-plane run with an
    append-only event chain. A run containing signed validation or approval
    remains fail-closed until the corresponding root-anchored historical trust
    stores are supplied to the control-plane replay layer.
    """
    vault_root = inputs.workspace_root / "quant_lab" / "vault"
    rows: list[dict[str, Any]] = []
    for folder in sorted(inputs.experiment_root.iterdir(), key=lambda item: item.name):
        if not folder.is_dir() or folder.name in historical_ids:
            continue
        run_id = _safe_experiment_id(folder.name)
        store = ExperimentRunStore(vault_root, run_id)
        try:
            replay = store.verify_integrity(
                trust_store=inputs.approval_trust_stores,
                runner_trust_store=inputs.validation_runner_trust_stores,
            )
            manifest = store.load()
        except Exception as exc:
            raise CompletionError(
                f"non-historical experiment is not a replayable control-plane run: {run_id}: {exc}"
            ) from exc
        event_files = sorted((folder / ".control" / "events").glob("*.json"))
        if not event_files:
            raise CompletionError(f"control-plane run has no event chain: {run_id}")
        audit_head = event_files[-1].stem.split("-", 1)[-1]
        if not _is_sha256(audit_head):
            raise CompletionError(f"control-plane audit head is malformed: {run_id}")
        signed_decision_hash: str | None = None
        admission_events = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in event_files
            if json.loads(path.read_text(encoding="utf-8")).get("event_type")
            == "data_admission_verified"
        ]
        if len(admission_events) > 1:
            raise CompletionError(f"control-plane run has multiple data admissions: {run_id}")
        admission_payload = admission_events[0]["payload"] if admission_events else {}
        if admission_events:
            admission_store = (inputs.data_admission_trust_stores or {}).get(
                admission_payload.get("admission_trust_policy_sha256")
            )
            release_store = (inputs.data_release_trust_stores or {}).get(
                admission_payload.get("release_trust_policy_sha256")
            )
            if not isinstance(admission_store, DataAdmissionTrustStore) or not isinstance(
                release_store, DataReleaseTrustStore
            ):
                raise CompletionError(
                    f"historical data-admission trust policy is unavailable: {run_id}"
                )
            archive_path = folder / str(admission_payload.get("admission_uri", ""))
            try:
                archived = json.loads(archive_path.read_text(encoding="utf-8"))
                replayed_admission = verify_archived_data_admission(
                    archived,
                    admission_trust_store=admission_store,
                    release_trust_store=release_store,
                    vault_root=vault_root,
                )
            except Exception as exc:
                raise CompletionError(
                    f"signed data-admission replay failed: {run_id}: {exc}"
                ) from exc
            admission_manifest_bindings = {
                "root_run_id": manifest.run_id,
                "data_manifest_id": manifest.data_manifest_id,
                "data_asof_utc": manifest.data_asof_utc,
                "feature_manifest_hash": manifest.feature_manifest_hash,
                "code_commit": manifest.code_commit,
                "split_manifest_id": manifest.split_manifest_id,
            }
            mismatched = [
                field
                for field, expected in admission_manifest_bindings.items()
                if getattr(replayed_admission, field) != expected
            ]
            if mismatched:
                raise CompletionError(
                    "cryptographically replayed data admission differs from the run manifest "
                    f"for {run_id}: {', '.join(mismatched)}"
                )
        human_events = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in event_files
            if json.loads(path.read_text(encoding="utf-8")).get("event_type")
            == "human_decision_verified"
        ]
        if human_events:
            if len(human_events) != 1:
                raise CompletionError(f"control-plane run has multiple human decisions: {run_id}")
            signed_decision_hash = human_events[0]["payload"].get("approval_sha256")
            if not _is_sha256(signed_decision_hash):
                raise CompletionError(f"control-plane approval hash is malformed: {run_id}")
        rows.append(
            {
                "run_id": run_id,
                "parent_run_id": manifest.parent_run_id,
                "source_folder": folder.relative_to(inputs.workspace_root).as_posix(),
                "manifest_path": store.manifest_path.relative_to(inputs.workspace_root).as_posix(),
                "manifest_sha256": replay["manifest_hash"],
                "manifest_identity_sha256": replay["manifest_identity_hash"],
                "audit_head_sha256": audit_head,
                "event_count": replay["event_count"],
                "run_kind": manifest.run_kind.value,
                "workflow_status": manifest.workflow_status.value,
                "validation_status": manifest.validation_status.value,
                "promotion_status": manifest.promotion_status.value,
                "promotion_eligible": int(manifest.promotion_eligible),
                "approval_required": int(manifest.promotion_status.value == "approval_required"),
                "accepted": int(manifest.promotion_status.value == "approved"),
                "signed_decision_sha256": signed_decision_hash,
                "admission_subject_sha256": admission_payload.get(
                    "admission_subject_sha256"
                ),
                "admission_plan_sha256": admission_payload.get("plan_sha256"),
                "human_release_sha256": admission_payload.get("human_release_sha256"),
                "admission_trust_policy_sha256": admission_payload.get(
                    "admission_trust_policy_sha256"
                ),
                "release_trust_policy_sha256": admission_payload.get(
                    "release_trust_policy_sha256"
                ),
                "created_at": manifest.created_at,
                "updated_at": manifest.updated_at,
            }
        )
    return rows


def _run_kind(classification: Mapping[str, Any]) -> str:
    if classification["migration_class"] != "test_or_synthetic_quarantined":
        return "research"
    if classification["reason_subcode"] in {
        "documented_synthetic_example_workflow",
        "explicit_synthetic_demo_run",
    }:
        return "synthetic"
    return "test"


def _workflow_status(record: Mapping[str, Any]) -> str:
    source = str(record["original_status"])
    if source in {"workflow_complete_approval_required", "completed_not_accepted"}:
        return "complete"
    if record["migration_class"] in {
        "genuine_rejected_closed",
        "superseded_historical_version",
    }:
        return "complete"
    if source == "rejected_or_blocked":
        return "failed"
    return "failed" if record["migration_class"] == "test_or_synthetic_quarantined" else "running"


def _validation_status(record: Mapping[str, Any]) -> str:
    classification = record["migration_class"]
    if classification == "genuine_rejected_closed":
        return "fail"
    if classification == "parked_awaiting_point_in_time_data":
        return "inconclusive"
    return "not_run"


def _historical_manifest(
    record: Mapping[str, Any], source: Mapping[str, Any], overlay_manifest_hash: str
) -> dict[str, Any]:
    classified_at = str(record["classified_at"])
    created_at = str(source.get("created_at") or classified_at)
    updated_at = str(source.get("updated_at") or classified_at)
    manifest = {
        "run_id": record["experiment_id"],
        "parent_run_id": None,
        "run_kind": _run_kind(record),
        "promotion_eligible": bool(record["promotion_eligible"]),
        "hypothesis": str(source.get("hypothesis") or "legacy_unrecorded_hypothesis"),
        "universe": str(source.get("universe") or "legacy_unrecorded_universe"),
        "data_manifest_id": None,
        "data_asof_utc": None,
        "feature_manifest_hash": None,
        "code_commit": None,
        "target": None,
        "label_horizon": None,
        "split_manifest_id": None,
        # These are deliberately unknown: legacy prose is not replayable control-plane state.
        "workflow_status": None,
        "validation_status": None,
        "promotion_status": record["promotion_status"],
        "created_at": created_at,
        "updated_at": updated_at,
    }
    if tuple(manifest) != MANIFEST_FIELDS:
        raise AssertionError("historical manifest field order/schema drift")
    return {
        "schema_version": SCHEMA_VERSION,
        "projection_kind": "non_executable_historical_migration_manifest",
        "execution_eligible": False,
        "record_source": RECORD_SOURCE_HISTORICAL,
        "not_a_control_plane_run_manifest": True,
        "authoritative_source": {
            "overlay_manifest_sha256": overlay_manifest_hash,
            "snapshot_id": record["snapshot_id"],
            "overlay_record_sha256": semantic_hash(record),
        },
        "manifest": manifest,
        "migration": {
            "lineage_id": record["migration_lineage_id"],
            "classification": record["migration_class"],
            "effective_decision_state": record["effective_decision_state"],
            "reason": record["reason"],
            "reason_subcode": record["reason_subcode"],
            "superseded_by_experiment_id": record["replacement_experiment_id"],
            "action_required": record["action_required"],
            "source_hash": record["source_hash"],
            "approval_required": False,
            "accepted": False,
        },
    }


def _missing_data(record: Mapping[str, Any]) -> dict[str, Any]:
    if record["migration_class"] != "parked_awaiting_point_in_time_data":
        return {}
    contract = RERUN_RESEARCH_CONTRACTS[record["experiment_id"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "checked_at": record["classified_at"],
        "checker": {
            "execution_id": "findings-hygiene-v2-deterministic-migration",
            "authority": "frozen-overlay-and-source-evidence",
            "classification_basis_sha256": record["classification_basis_sha256"],
        },
        "provider": "LSEG",
        "dataset_product_id": None,
        "dataset_product_id_status": "required_from_entitlement_receipt",
        "entitlement_requirements": {
            "aws_region": "ap-southeast-2",
            "requester_pays_account_id": "required",
            "signed_entitlement_attestation": "required",
        },
        "markets": contract["market_scope"],
        "replication_window": contract["replication_window"],
        "missing_items": list(REQUIRED_LSEG_DATA),
        "evidence": [
            {
                "path": item["path"],
                "sha256": item["sha256"],
                "role": item.get("role", ""),
            }
            for item in record["evidence"]
        ],
        "required_admission_artifacts": [
            "signed_entitlement_attestation",
            "signed_data_availability_receipt",
            "admitted_data_manifest",
            "feature_manifest",
            "split_manifest",
            "validation_policy",
            "code_snapshot",
        ],
        "unpark_policy": "all_items_must_be_satisfied_by_one_hashed_data_admission_receipt",
    }


def _init_registry_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE registry_builds(
            build_id TEXT PRIMARY KEY,
            schema_version INTEGER NOT NULL,
            snapshot_id TEXT NOT NULL,
            snapshot_manifest_sha256 TEXT NOT NULL CHECK(length(snapshot_manifest_sha256)=64),
            overlay_manifest_sha256 TEXT NOT NULL CHECK(length(overlay_manifest_sha256)=64),
            overlay_semantic_sha256 TEXT NOT NULL CHECK(length(overlay_semantic_sha256)=64),
            builder_source_sha256 TEXT NOT NULL CHECK(length(builder_source_sha256)=64),
            historical_run_count INTEGER NOT NULL,
            live_run_count INTEGER NOT NULL CHECK(live_run_count>=0),
            lineage_count INTEGER NOT NULL,
            evidence_count INTEGER NOT NULL,
            registry_semantic_sha256 TEXT NOT NULL CHECK(length(registry_semantic_sha256)=64),
            created_at TEXT NOT NULL
        );
        CREATE TABLE lineages(
            lineage_id TEXT PRIMARY KEY,
            historical_anchor_run_id TEXT,
            latest_experiment_id TEXT NOT NULL,
            experiment_count INTEGER NOT NULL CHECK(experiment_count>0),
            current_promotion_status TEXT NOT NULL,
            latest_approved_run_id TEXT CHECK(latest_approved_run_id IS NULL)
        );
        CREATE TABLE experiments(
            experiment_id TEXT PRIMARY KEY,
            lineage_id TEXT NOT NULL REFERENCES lineages(lineage_id),
            record_source TEXT NOT NULL CHECK(record_source IN('historical_overlay_v2','control_plane_v1')),
            source_folder TEXT NOT NULL UNIQUE,
            historical_manifest_path TEXT NOT NULL UNIQUE,
            historical_manifest_sha256 TEXT NOT NULL CHECK(length(historical_manifest_sha256)=64),
            overlay_record_sha256 TEXT NOT NULL CHECK(length(overlay_record_sha256)=64),
            source_record_sha256 TEXT NOT NULL CHECK(length(source_record_sha256)=64),
            snapshot_id TEXT NOT NULL,
            migration_class TEXT NOT NULL CHECK(migration_class IN(
                'test_or_synthetic_quarantined','genuine_rejected_closed',
                'superseded_historical_version','parked_awaiting_point_in_time_data')),
            run_kind TEXT NOT NULL CHECK(run_kind IN('research','test','synthetic','smoke')),
            workflow_status TEXT CHECK(workflow_status IS NULL OR workflow_status IN('created','running','complete','failed')),
            validation_status TEXT CHECK(validation_status IS NULL OR validation_status IN('not_run','pass','fail','inconclusive')),
            promotion_status TEXT NOT NULL CHECK(promotion_status IN(
                'ineligible','parked_for_data','rejected','approval_required','approved','superseded')),
            promotion_eligible INTEGER NOT NULL CHECK(promotion_eligible IN(0,1)),
            excluded_from_action_queue INTEGER NOT NULL CHECK(excluded_from_action_queue IN(0,1)),
            approval_required INTEGER NOT NULL CHECK(approval_required=0),
            accepted INTEGER NOT NULL CHECK(accepted=0),
            reason_code TEXT NOT NULL,
            reason TEXT NOT NULL,
            kill_reason TEXT,
            missing_data_json TEXT NOT NULL,
            legacy_anchor_run_id TEXT,
            parent_run_id TEXT REFERENCES experiments(experiment_id) DEFERRABLE INITIALLY DEFERRED,
            superseded_by_experiment_id TEXT REFERENCES experiments(experiment_id)
                DEFERRABLE INITIALLY DEFERRED,
            action_required TEXT,
            source_hash TEXT NOT NULL CHECK(length(source_hash)=64),
            evidence_count INTEGER NOT NULL CHECK(evidence_count>=0),
            created_at TEXT,
            updated_at TEXT,
            CHECK(migration_class!='genuine_rejected_closed' OR length(kill_reason)>0),
            CHECK(migration_class!='parked_awaiting_point_in_time_data' OR action_required IS NOT NULL),
            CHECK(migration_class!='test_or_synthetic_quarantined' OR excluded_from_action_queue=1),
            CHECK(promotion_status NOT IN('approval_required','approved')),
            CHECK(
              (migration_class='test_or_synthetic_quarantined' AND promotion_status='ineligible') OR
              (migration_class='genuine_rejected_closed' AND promotion_status='rejected') OR
              (migration_class='superseded_historical_version' AND promotion_status='superseded') OR
              (migration_class='parked_awaiting_point_in_time_data' AND promotion_status='parked_for_data')
            )
        );
        CREATE TABLE control_plane_runs(
            run_id TEXT PRIMARY KEY,
            parent_run_id TEXT,
            record_source TEXT NOT NULL CHECK(record_source='control_plane_v1'),
            source_folder TEXT NOT NULL UNIQUE,
            manifest_path TEXT NOT NULL UNIQUE,
            manifest_sha256 TEXT NOT NULL CHECK(length(manifest_sha256)=64),
            manifest_identity_sha256 TEXT NOT NULL CHECK(length(manifest_identity_sha256)=64),
            audit_head_sha256 TEXT NOT NULL CHECK(length(audit_head_sha256)=64),
            event_count INTEGER NOT NULL CHECK(event_count>0),
            run_kind TEXT NOT NULL CHECK(run_kind IN('research','test','synthetic','smoke')),
            workflow_status TEXT NOT NULL CHECK(workflow_status IN('created','running','complete','failed')),
            validation_status TEXT NOT NULL CHECK(validation_status IN('not_run','pass','fail','inconclusive')),
            promotion_status TEXT NOT NULL CHECK(promotion_status IN(
                'ineligible','parked_for_data','rejected','approval_required','approved','superseded')),
            promotion_eligible INTEGER NOT NULL CHECK(promotion_eligible IN(0,1)),
            approval_required INTEGER NOT NULL CHECK(approval_required IN(0,1)),
            accepted INTEGER NOT NULL CHECK(accepted IN(0,1)),
            signed_decision_sha256 TEXT CHECK(signed_decision_sha256 IS NULL OR length(signed_decision_sha256)=64),
            admission_subject_sha256 TEXT CHECK(admission_subject_sha256 IS NULL OR length(admission_subject_sha256)=64),
            admission_plan_sha256 TEXT CHECK(admission_plan_sha256 IS NULL OR length(admission_plan_sha256)=64),
            human_release_sha256 TEXT CHECK(human_release_sha256 IS NULL OR length(human_release_sha256)=64),
            admission_trust_policy_sha256 TEXT CHECK(admission_trust_policy_sha256 IS NULL OR length(admission_trust_policy_sha256)=64),
            release_trust_policy_sha256 TEXT CHECK(release_trust_policy_sha256 IS NULL OR length(release_trust_policy_sha256)=64),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK(approval_required=(promotion_status='approval_required')),
            CHECK(accepted=(promotion_status='approved')),
            CHECK(promotion_status!='approved' OR (
                validation_status='pass' AND promotion_eligible=1 AND signed_decision_sha256 IS NOT NULL
            ))
        );
        CREATE TABLE evidence(
            path TEXT PRIMARY KEY,
            experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id) ON DELETE RESTRICT,
            role TEXT NOT NULL,
            evidence_type TEXT NOT NULL,
            sha256 TEXT NOT NULL CHECK(length(sha256)=64),
            byte_size INTEGER NOT NULL CHECK(byte_size>=0),
            snapshot_id TEXT NOT NULL
        );
        CREATE TABLE lineage_edges(
            from_experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
            to_experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
            edge_type TEXT NOT NULL CHECK(edge_type IN('parent_child','superseded_by','legacy_migration_anchor')),
            basis_sha256 TEXT NOT NULL CHECK(length(basis_sha256)=64),
            PRIMARY KEY(from_experiment_id,to_experiment_id,edge_type),
            CHECK(from_experiment_id<>to_experiment_id)
        );
        CREATE TABLE planned_lineage_edges(
            legacy_anchor_run_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
            reserved_child_run_id TEXT NOT NULL UNIQUE,
            edge_type TEXT NOT NULL CHECK(edge_type='legacy_migration_anchor'),
            readiness_path TEXT NOT NULL,
            readiness_sha256 TEXT NOT NULL CHECK(length(readiness_sha256)=64),
            basis_sha256 TEXT NOT NULL CHECK(length(basis_sha256)=64)
        );
        CREATE TABLE control_plane_lineage_edges(
            run_id TEXT PRIMARY KEY REFERENCES control_plane_runs(run_id),
            parent_run_id TEXT REFERENCES control_plane_runs(run_id)
                DEFERRABLE INITIALLY DEFERRED,
            legacy_anchor_run_id TEXT REFERENCES experiments(experiment_id)
                DEFERRABLE INITIALLY DEFERRED,
            edge_type TEXT NOT NULL CHECK(edge_type IN('parent_child','legacy_migration_anchor')),
            basis_sha256 TEXT NOT NULL CHECK(length(basis_sha256)=64),
            CHECK(
                (edge_type='parent_child' AND parent_run_id IS NOT NULL AND legacy_anchor_run_id IS NULL) OR
                (edge_type='legacy_migration_anchor' AND parent_run_id IS NULL AND legacy_anchor_run_id IS NOT NULL)
            )
        );
        CREATE TABLE rerun_requirements(
            experiment_id TEXT PRIMARY KEY REFERENCES experiments(experiment_id),
            readiness_path TEXT NOT NULL UNIQUE,
            readiness_sha256 TEXT NOT NULL CHECK(length(readiness_sha256)=64),
            execution_authorized INTEGER NOT NULL CHECK(execution_authorized=0),
            root_run_id TEXT CHECK(root_run_id IS NULL),
            required_data_json TEXT NOT NULL,
            required_controls_json TEXT NOT NULL,
            validation_gates_json TEXT NOT NULL,
            workflow_stages_json TEXT NOT NULL
        );
        CREATE INDEX experiments_class_idx ON experiments(migration_class);
        CREATE INDEX experiments_lineage_idx ON experiments(lineage_id);
        CREATE INDEX evidence_experiment_idx ON evidence(experiment_id);
        CREATE VIEW action_queue AS
            SELECT * FROM experiments
            WHERE excluded_from_action_queue=0
              AND promotion_status='parked_for_data'
              AND action_required IS NOT NULL
              AND length(missing_data_json)>2;
        CREATE VIEW operational_runs AS
            SELECT experiment_id AS run_id, parent_run_id, record_source,
                   workflow_status, validation_status, promotion_status,
                   promotion_eligible, approval_required, accepted,
                   created_at, updated_at
            FROM experiments
            UNION ALL
            SELECT run_id, parent_run_id, record_source,
                   workflow_status, validation_status, promotion_status,
                   promotion_eligible, approval_required, accepted,
                   created_at, updated_at
            FROM control_plane_runs;
        CREATE VIEW operational_action_queue AS
            SELECT run_id,record_source,promotion_status FROM operational_runs
            WHERE promotion_status='parked_for_data';
        """
    )


def _build_registry(
    staging: Path,
    inputs: CompletionInputs,
    overlay_manifest: Mapping[str, Any],
    overlay_payload: Mapping[str, Any],
    source_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    folders = _experiment_folders(inputs.experiment_root)
    records = sorted(overlay_payload["classifications"], key=lambda item: item["experiment_id"])
    record_ids = [record["experiment_id"] for record in records]
    if not set(record_ids).issubset(folders) or record_ids != sorted(source_rows):
        raise CompletionError("historical folders, source registry, and overlay are not a bijection")
    live_runs = _discover_control_plane_runs(inputs, set(record_ids))
    if sorted(set(folders) - set(record_ids)) != [row["run_id"] for row in live_runs]:
        raise CompletionError("every post-migration experiment folder must replay as a control-plane run")
    overlay_manifest_path = Path(json.loads(inputs.overlay_pointer.read_text())["overlay_manifest"])
    overlay_manifest_hash = sha256_file(overlay_manifest_path)

    manifest_dir = staging / "historical-manifests"
    manifest_dir.mkdir(parents=True)
    projected: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for record in records:
        experiment_id = _safe_experiment_id(record["experiment_id"])
        source = source_rows[experiment_id]
        if source["source_hash"] != record["source_hash"]:
            raise CompletionError(f"source hash drift for {experiment_id}")
        payload = _historical_manifest(record, source, overlay_manifest_hash)
        path = manifest_dir / f"{experiment_id}.json"
        _write_json(path, payload)
        projected[experiment_id] = (payload, path, sha256_file(path))

    readiness_dir = staging / "rerun-readiness"
    readiness_dir.mkdir()
    readiness: dict[str, tuple[dict[str, Any], Path, str]] = {}
    for experiment_id in EXPECTED_PARKED_IDS:
        record = next(item for item in records if item["experiment_id"] == experiment_id)
        contract = RERUN_RESEARCH_CONTRACTS[experiment_id]
        feature_sources = [
            {"path": rel_path, "sha256": sha256_file(inputs.workspace_root / rel_path)}
            for rel_path in [*contract["feature_sources"], *LSEG_EXECUTION_SOURCES]
        ]
        validation_sources = [
            {"path": rel_path, "sha256": sha256_file(inputs.workspace_root / rel_path)}
            for rel_path in VALIDATION_POLICY_SOURCES
        ]
        evidence_refs = [
            {"path": item["path"], "sha256": item["sha256"], "role": item.get("role", "")}
            for item in record["evidence"]
        ]
        next_run_id = f"LSEG-RERUN-{experiment_id}-R1"
        payload = {
            "schema_version": SCHEMA_VERSION,
            "package_id": f"lseg-rerun-readiness:{experiment_id}:v1",
            "created_at": overlay_manifest["created_at"],
            "package_status": "ready_waiting_for_lseg",
            "lineage_id": record["migration_lineage_id"],
            "legacy_anchor_run_id": experiment_id,
            "next_run_id": next_run_id,
            "next_run_parent_run_id": None,
            "lineage_boundary_policy": "The first governed rerun uses parent_run_id=null and a separate legacy_migration_anchor edge; later governed children use the prior governed run as parent_run_id.",
            "source_bindings": {
                "snapshot_id": record["snapshot_id"],
                "source_record_sha256": record["original_record_sha256"],
                "overlay_record_sha256": semantic_hash(record),
                "classification_basis_sha256": record["classification_basis_sha256"],
                "evidence": evidence_refs,
            },
            "research_contract": {
                "hypothesis": contract["hypothesis"],
                "universe": {
                    "markets": contract["market_scope"],
                    "scope_decision": contract["market_scope_decision"],
                    "point_in_time": True,
                    "include_inactive_and_delisted": True,
                },
                "candidate_specification": contract["candidate_specification"],
                "target": contract["target"],
                "label_horizon": contract["label_horizon"],
                "replication_window": contract["replication_window"],
                "extension_window": None,
                "minimum_lookback_trading_days": contract["minimum_lookback_trading_days"],
            },
            "data_contract": {
                "provider": "LSEG",
                "dataset_product_id": None,
                "dataset_product_id_status": "must_be_bound_from_entitlement_receipt_before_activation",
                "expected_aws_region": "ap-southeast-2",
                "requester_pays_account_id": None,
                "markets": contract["market_scope"],
                "currencies": contract["currencies"],
                "replication_window": contract["replication_window"],
                "required_fields": list(REQUIRED_LSEG_FIELDS),
                "point_in_time_membership": True,
                "include_listing_and_delisting_history": True,
                "corporate_action_policy": "announced/effective/pay timestamps retained; apply only information available by decision cutoff",
                "timestamp_policy": "retain vendor publication, effective, revision/vintage, ingestion and decision-cutoff timestamps in UTC",
                "calendar_policy": "exchange-local sessions mapped to MIC and timezone before UTC normalization",
                "fx_policy": "prior-available spot/total-return conversion to USD with currency lineage; no same-day lookahead",
                "coverage_thresholds": {
                    "stable_identifier_fraction": 0.99,
                    "membership_event_completeness": 1.0,
                    "corporate_action_completeness": 1.0,
                    "required_timestamp_fraction": 1.0,
                    "price_observation_fraction": 0.98,
                },
            },
            "frozen_source_contracts": {
                "feature_sources": feature_sources,
                "split_and_validation_sources": validation_sources,
                "feature_spec_hash": semantic_hash(feature_sources),
                "split_and_validation_policy_hash": semantic_hash(validation_sources),
            },
            "execution_authorized": False,
            "root_run_id": None,
            "root_run_id_policy": (
                "Create one new research run only after every data precondition passes; "
                "forward that immutable root_run_id through every workflow stage."
            ),
            "required_data": list(REQUIRED_LSEG_DATA),
            "required_frozen_controls": list(REQUIRED_RERUN_CONTROLS),
            "data_manifest_preconditions": {
                "entitlement_receipt": {"status": "missing", "artifact_uri": None, "sha256": None},
                "data_admission_manifest": {"status": "missing", "artifact_uri": None, "sha256": None},
                "human_data_available_receipt": {"status": "missing", "artifact_uri": None, "sha256": None},
                **{
                    item: {"status": "missing", "artifact_uri": None, "sha256": None}
                    for item in REQUIRED_LSEG_DATA
                },
            },
            "future_manifest_fields": list(MANIFEST_FIELDS),
            "workflow_stages": list(WORKFLOW_STAGES),
            "deterministic_validation_gates": list(REQUIRED_VALIDATION_GATES),
            "approval_policy": "signed_authorized_human_only_after_measured_bundle_passes",
            "promotion_status_before_data": "parked_for_data",
            "blocked_reason": record["reason"],
            "next_action": record["action_required"],
            "activation_policy": {
                "all_preconditions_required": True,
                "external_calls_before_activation": False,
                "create_exactly_one_control_plane_run": True,
                "root_run_id_must_equal_reserved_next_run_id": next_run_id,
            },
        }
        payload["frozen_source_contracts"]["validation_policy_sha256"] = semantic_hash(
            frozen_validation_policy_payload(payload)
        )
        payload["frozen_source_contracts"]["research_config_sha256"] = semantic_hash(
            frozen_research_config_payload(payload)
        )
        payload["package_semantic_sha256"] = semantic_hash(payload)
        path = readiness_dir / f"{experiment_id}.json"
        _write_json(path, payload)
        readiness[experiment_id] = (payload, path, sha256_file(path))

    db_path = staging / "canonical-findings.sqlite3"
    connection = sqlite3.connect(db_path)
    try:
        _init_registry_schema(connection)
        latest_by_lineage: dict[str, str] = {}
        count_by_lineage: Counter[str] = Counter()
        for record in records:
            lineage = record["migration_lineage_id"]
            count_by_lineage[lineage] += 1
            if record["migration_latest_version"]:
                if lineage in latest_by_lineage:
                    raise CompletionError(f"multiple latest experiments in lineage {lineage}")
                latest_by_lineage[lineage] = record["experiment_id"]
        if set(latest_by_lineage) != set(count_by_lineage):
            raise CompletionError("every lineage must have exactly one latest experiment")
        records_by_id = {record["experiment_id"]: record for record in records}
        for lineage in sorted(count_by_lineage):
            latest_id = latest_by_lineage[lineage]
            connection.execute(
                "INSERT INTO lineages VALUES (?,?,?,?,?,?)",
                (
                    lineage,
                    latest_id if records_by_id[latest_id]["migration_class"] == "parked_awaiting_point_in_time_data" else None,
                    latest_id,
                    count_by_lineage[lineage],
                    records_by_id[latest_id]["promotion_status"],
                    None,
                ),
            )
        for record in records:
            experiment_id = record["experiment_id"]
            payload, manifest_path, manifest_hash = projected[experiment_id]
            relative_manifest = manifest_path.relative_to(staging).as_posix()
            kill_reason = record["reason"] if record["migration_class"] == "genuine_rejected_closed" else None
            connection.execute(
                """INSERT INTO experiments VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    experiment_id,
                    record["migration_lineage_id"],
                    RECORD_SOURCE_HISTORICAL,
                    f"quant_lab/vault/30_experiments/{experiment_id}",
                    relative_manifest,
                    manifest_hash,
                    semantic_hash(record),
                    record["original_record_sha256"],
                    record["snapshot_id"],
                    record["migration_class"],
                    payload["manifest"]["run_kind"],
                    payload["manifest"]["workflow_status"],
                    payload["manifest"]["validation_status"],
                    record["promotion_status"],
                    int(record["promotion_eligible"]),
                    int(record["excluded_from_action_queue"]),
                    0,
                    0,
                    record["reason_subcode"],
                    record["reason"],
                    kill_reason,
                    canonical_json(_missing_data(record)),
                    None,
                    None,
                    record["replacement_experiment_id"],
                    record["action_required"],
                    record["source_hash"],
                    int(record["evidence_count"]),
                    source_rows[experiment_id].get("created_at") or None,
                    source_rows[experiment_id].get("updated_at") or None,
                ),
            )
            for evidence in record["evidence"]:
                evidence_path = inputs.workspace_root / evidence["path"]
                if not evidence_path.is_file() or sha256_file(evidence_path) != evidence["sha256"]:
                    raise CompletionError(f"historical evidence drift: {evidence['path']}")
                connection.execute(
                    "INSERT INTO evidence VALUES (?,?,?,?,?,?,?)",
                    (
                        evidence["path"], experiment_id, evidence.get("role", ""),
                        evidence.get("type", ""), evidence["sha256"],
                        evidence_path.stat().st_size, record["snapshot_id"],
                    ),
                )
        for live in live_runs:
            connection.execute(
                """INSERT INTO control_plane_runs VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    live["run_id"], live["parent_run_id"], "control_plane_v1",
                    live["source_folder"], live["manifest_path"], live["manifest_sha256"],
                    live["manifest_identity_sha256"], live["audit_head_sha256"],
                    live["event_count"], live["run_kind"], live["workflow_status"],
                    live["validation_status"], live["promotion_status"],
                    live["promotion_eligible"], live["approval_required"], live["accepted"],
                    live["signed_decision_sha256"], live["admission_subject_sha256"],
                    live["admission_plan_sha256"], live["human_release_sha256"],
                    live["admission_trust_policy_sha256"],
                    live["release_trust_policy_sha256"], live["created_at"], live["updated_at"],
                ),
            )
        for record in records:
            replacement = record["replacement_experiment_id"]
            if replacement:
                connection.execute(
                    "INSERT INTO lineage_edges VALUES (?,?,?,?)",
                    (
                        record["experiment_id"], replacement, "superseded_by",
                        record["classification_basis_sha256"],
                    ),
                )
        for experiment_id, (payload, readiness_path, readiness_hash) in readiness.items():
            connection.execute(
                "INSERT INTO rerun_requirements VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    experiment_id,
                    readiness_path.relative_to(staging).as_posix(),
                    readiness_hash,
                    0,
                    None,
                    canonical_json(payload["required_data"]),
                    canonical_json(payload["required_frozen_controls"]),
                    canonical_json(payload["deterministic_validation_gates"]),
                    canonical_json(payload["workflow_stages"]),
                ),
            )
            connection.execute(
                "INSERT INTO planned_lineage_edges VALUES (?,?,?,?,?,?)",
                (
                    experiment_id, payload["next_run_id"], "legacy_migration_anchor",
                    readiness_path.relative_to(staging).as_posix(), readiness_hash,
                    payload["source_bindings"]["classification_basis_sha256"],
                ),
            )
        live_ids = {row["run_id"] for row in live_runs}
        planned_roots = {
            payload["next_run_id"]: experiment_id
            for experiment_id, (payload, _, _) in readiness.items()
        }
        for live in live_runs:
            if live["parent_run_id"] is not None:
                if live["parent_run_id"] not in live_ids:
                    raise CompletionError(
                        f"control-plane parent is not in the live registry: {live['run_id']}"
                    )
                connection.execute(
                    "INSERT INTO control_plane_lineage_edges VALUES (?,?,?,?,?)",
                    (
                        live["run_id"], live["parent_run_id"], None,
                        "parent_child", live["audit_head_sha256"],
                    ),
                )
            else:
                anchor = planned_roots.get(live["run_id"])
                if anchor is None:
                    raise CompletionError(
                        f"root control-plane run has no frozen legacy anchor: {live['run_id']}"
                    )
                readiness_hash = readiness[anchor][2]
                if (
                    not all(
                        _is_sha256(live[field])
                        for field in (
                            "admission_subject_sha256", "admission_plan_sha256",
                            "human_release_sha256", "admission_trust_policy_sha256",
                            "release_trust_policy_sha256",
                        )
                    )
                    or live["admission_plan_sha256"] != readiness_hash
                ):
                    raise CompletionError(
                        f"root control-plane run lacks its signed data-admission lineage: {live['run_id']}"
                    )
                connection.execute(
                    "INSERT INTO control_plane_lineage_edges VALUES (?,?,?,?,?)",
                    (
                        live["run_id"], None, anchor,
                        "legacy_migration_anchor", live["audit_head_sha256"],
                    ),
                )
        registry_semantic = semantic_hash(
            {
                "historical": [
                    {
                        "experiment_id": record["experiment_id"],
                        "lineage_id": record["migration_lineage_id"],
                        "migration_class": record["migration_class"],
                        "promotion_status": record["promotion_status"],
                        "source_hash": record["source_hash"],
                        "overlay_record_sha256": semantic_hash(record),
                    }
                    for record in records
                ],
                "control_plane": live_runs,
            }
        )
        connection.execute(
            "INSERT INTO registry_builds VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                f"registry-v1-{registry_semantic[:20]}", SCHEMA_VERSION,
                overlay_manifest["snapshot_id"], overlay_manifest["snapshot_manifest_sha256"],
                overlay_manifest_hash, overlay_manifest["classification_semantic_sha256"],
                sha256_file(Path(__file__)), len(records), len(live_runs), len(count_by_lineage),
                sum(int(record["evidence_count"]) for record in records),
                registry_semantic, records[0]["classified_at"],
            ),
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()
    if integrity != "ok" or foreign_keys:
        raise CompletionError(f"canonical registry failed integrity: {integrity}, {foreign_keys}")
    return {
        "database": db_path.name,
        "database_sha256": sha256_file(db_path),
        "historical_manifest_count": len(projected),
        "historical_manifests_semantic_sha256": semantic_hash(
            {key: value[2] for key, value in sorted(projected.items())}
        ),
        "rerun_readiness_count": len(readiness),
        "rerun_readiness_semantic_sha256": semantic_hash(
            {key: value[2] for key, value in sorted(readiness.items())}
        ),
        "experiment_count": len(records),
        "live_control_plane_run_count": len(live_runs),
        "operational_run_count": len(records) + len(live_runs),
        "evidence_count": sum(int(record["evidence_count"]) for record in records),
        "classification_counts": dict(Counter(record["migration_class"] for record in records)),
        "action_queue_experiment_ids": list(EXPECTED_PARKED_IDS),
        "approval_required_count": 0,
        "accepted_count": 0,
    }


def _indexed_paths(database: Path | None) -> dict[str, str]:
    if database is None or not database.is_file():
        return {}
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(documents)")}
        if not {"rel_path", "sha256"}.issubset(columns):
            return {}
        return {row["rel_path"]: row["sha256"] for row in connection.execute("SELECT rel_path,sha256 FROM documents")}
    finally:
        connection.close()


def _book_aliases(database: Path | None) -> dict[str, dict[str, str]]:
    if database is None or not database.is_file():
        return {}
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True)
    try:
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "aliases" not in tables:
            return {}
        return {
            row[0]: {
                "alias_sha256": row[1],
                "canonical_rel_path": row[2],
                "canonical_sha256": row[3],
            }
            for row in connection.execute(
                """SELECT aliases.alias_path,aliases.sha256,
                          documents.rel_path,documents.sha256
                   FROM aliases JOIN documents USING(doc_id)"""
            )
        }
    finally:
        connection.close()


def _markdown_files(root: Path) -> list[Path]:
    return sorted((path for path in root.rglob("*.md") if path.is_file()), key=lambda p: p.relative_to(root).as_posix())


def _selected_document_category(rel_path: str, experiment_classes: Mapping[str, str]) -> str | None:
    parts = Path(rel_path).parts
    experiment_prefix = ("quant_lab", "vault", "30_experiments")
    if parts[:3] == experiment_prefix and len(parts) >= 5:
        experiment_id = parts[3]
        classification = experiment_classes.get(experiment_id)
        if classification in {"genuine_rejected_closed", "parked_awaiting_point_in_time_data"}:
            name = parts[-1].lower()
            if "validation" in name:
                return "validation_report"
            if "failure" in name or classification == "genuine_rejected_closed":
                return "kill_or_rejection_evidence"
            return "named_experiment_evidence"
        return None
    lowered = rel_path.lower()
    name = Path(rel_path).name.lower()
    if rel_path.startswith("quant_lab/benchmarks/") and name == "benchmark_card.md":
        return "canonical_model_card"
    if rel_path.startswith("quant_lab/docs/") and (
        "findings" in name or "resolution" in name or "strategy_inventory" in name
    ):
        return "canonical_research_memo"
    if lowered.endswith("/codex_supersede_package/final_research_memo.md"):
        return "canonical_research_memo"
    if "promotion" in name and ("minutes" in name or "decision" in name):
        return "promotion_minutes"
    return None


def _base_exclusion(rel_path: str) -> tuple[str, str] | None:
    lowered = rel_path.lower()
    parts = Path(lowered).parts
    if any(part in {".venv", "site-packages", "node_modules", ".git"} for part in parts):
        return "excluded_vendor_or_environment", "Vendored/runtime documentation is not quant research evidence."
    if "quant_lab/artifacts/zai_ocr/" in lowered:
        if "/formula_retries/" in lowered or "/formula_vision_repairs/" in lowered:
            return "excluded_formula_retry", "Formula retry/repair intermediate; canonical paper content is indexed separately."
        return "excluded_zai_ocr_intermediate", "ZAI OCR conversion/intermediate; canonical paper content is indexed separately."
    if any(part in {"__pycache__", ".pytest_cache", ".mypy_cache", ".supergoal"} for part in parts):
        return "excluded_cache_or_temporary", "Cache, automation state, or temporary administration file."
    if rel_path.startswith("quant_lab/artifacts/") or rel_path.startswith("artifacts/"):
        return "excluded_generated_artifact", "Generated report is not in the canonical promotion/experiment registry."
    return None


def _disposition(policy: str) -> str:
    if policy == "canonical_research_artifact":
        return "dense_embed"
    if policy == "exact_duplicate_alias":
        return "alias_only"
    if policy == "covered_existing_index":
        return "existing_index"
    if policy == "registry_evidence_only":
        return "metadata_only"
    return "exclude"


def _title(path: Path) -> str:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _HEADING_RE.match(line.strip())
            if match:
                return match.group(2).strip()
    return path.stem.replace("_", " ").strip()


def _split_text(text: str, maximum: int, overlap: int) -> Iterator[str]:
    start = 0
    while start < len(text):
        end = min(len(text), start + maximum)
        if end < len(text):
            boundary = text.rfind("\n\n", start + maximum // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            yield chunk
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)


def stream_markdown(path: Path, maximum: int = 3600, overlap: int = 400) -> Iterator[tuple[str, str]]:
    heading = "Preamble"
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = _HEADING_RE.match(line.rstrip())
            if match and lines:
                for content in _split_text("".join(lines), maximum, overlap):
                    yield heading, content
                lines = []
                heading = match.group(2).strip()
            elif match:
                heading = match.group(2).strip()
            lines.append(line)
    if lines:
        for content in _split_text("".join(lines), maximum, overlap):
            yield heading, content


def _vector_blob(values: Iterable[float]) -> tuple[bytes, int]:
    parsed = [float(value) for value in values]
    if not parsed or any(not math.isfinite(value) for value in parsed):
        raise CompletionError("embedding provider returned an empty or non-finite vector")
    if not any(value != 0.0 for value in parsed):
        raise CompletionError("embedding provider returned an all-zero vector")
    vector = array("f", parsed)
    return vector.tobytes(), len(vector)


def _init_artifact_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA foreign_keys=ON;
        CREATE TABLE markdown_policy(
            artifact_id TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL,
            sha256 TEXT NOT NULL CHECK(length(sha256)=64),
            byte_size INTEGER NOT NULL CHECK(byte_size>=0),
            mtime_ns INTEGER NOT NULL,
            disposition TEXT NOT NULL CHECK(disposition IN('dense_embed','metadata_only','alias_only','exclude','existing_index')),
            policy TEXT NOT NULL,
            reason TEXT NOT NULL,
            searchable INTEGER NOT NULL CHECK(searchable IN(0,1)),
            canonical_rel_path TEXT,
            source_index TEXT,
            source_index_hash_matches INTEGER CHECK(source_index_hash_matches IN(0,1)),
            experiment_id TEXT,
            lineage_id TEXT,
            is_lineage_head INTEGER CHECK(is_lineage_head IS NULL OR is_lineage_head IN(0,1)),
            artifact_category TEXT,
            rule_version TEXT NOT NULL,
            decision_basis_sha256 TEXT NOT NULL CHECK(length(decision_basis_sha256)=64),
            UNIQUE(rel_path)
        );
        CREATE TABLE documents(
            doc_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL UNIQUE REFERENCES markdown_policy(artifact_id),
            rel_path TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            sha256 TEXT NOT NULL CHECK(length(sha256)=64),
            byte_size INTEGER NOT NULL,
            cloud_allowed INTEGER NOT NULL CHECK(cloud_allowed=0)
        );
        CREATE TABLE aliases(
            alias_artifact_id TEXT PRIMARY KEY REFERENCES markdown_policy(artifact_id),
            canonical_doc_id TEXT REFERENCES documents(doc_id),
            canonical_source_index TEXT NOT NULL CHECK(canonical_source_index IN('completion','books')),
            canonical_rel_path TEXT NOT NULL,
            canonical_sha256 TEXT NOT NULL CHECK(length(canonical_sha256)=64),
            sha256 TEXT NOT NULL CHECK(length(sha256)=64),
            CHECK(
                (canonical_source_index='completion' AND canonical_doc_id IS NOT NULL) OR
                (canonical_source_index='books' AND canonical_doc_id IS NULL)
            )
        );
        CREATE TABLE document_run_links(
            doc_id TEXT NOT NULL REFERENCES documents(doc_id),
            experiment_id TEXT NOT NULL,
            lineage_id TEXT NOT NULL,
            effective_decision_state TEXT NOT NULL,
            PRIMARY KEY(doc_id,experiment_id)
        );
        CREATE TABLE chunks(
            chunk_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            heading TEXT NOT NULL,
            content TEXT NOT NULL,
            chunk_sha256 TEXT NOT NULL CHECK(length(chunk_sha256)=64),
            embedding BLOB NOT NULL,
            embedding_dim INTEGER NOT NULL CHECK(embedding_dim>0),
            UNIQUE(doc_id,ordinal)
        );
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            chunk_id UNINDEXED,title,heading,category,content,
            tokenize='porter unicode61'
        );
        CREATE TABLE index_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE policy_builds(
            policy_build_id TEXT PRIMARY KEY,
            inventory_semantic_sha256 TEXT NOT NULL CHECK(length(inventory_semantic_sha256)=64),
            registry_sha256 TEXT NOT NULL CHECK(length(registry_sha256)=64),
            rule_version TEXT NOT NULL,
            evaluator_source_sha256 TEXT NOT NULL CHECK(length(evaluator_source_sha256)=64),
            artifact_count INTEGER NOT NULL,
            policy_semantic_sha256 TEXT NOT NULL CHECK(length(policy_semantic_sha256)=64)
        );
        CREATE TABLE index_builds(
            index_build_id TEXT PRIMARY KEY,
            policy_build_id TEXT NOT NULL REFERENCES policy_builds(policy_build_id),
            embedding_model TEXT NOT NULL,
            chunker_version TEXT NOT NULL,
            document_count INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            index_semantic_sha256 TEXT NOT NULL CHECK(length(index_semantic_sha256)=64)
        );
        CREATE INDEX markdown_policy_policy_idx ON markdown_policy(policy);
        CREATE INDEX documents_category_idx ON documents(category);
        """
    )


def _build_artifact_index(
    staging: Path,
    inputs: CompletionInputs,
    overlay_payload: Mapping[str, Any],
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]],
    embedding_provider_identity: str,
) -> dict[str, Any]:
    experiment_classes = {
        item["experiment_id"]: item["migration_class"]
        for item in overlay_payload["classifications"]
    }
    experiment_records = {
        item["experiment_id"]: item for item in overlay_payload["classifications"]
    }
    quarantined_hashes = {
        evidence["sha256"]
        for item in overlay_payload["classifications"]
        if item["migration_class"] == "test_or_synthetic_quarantined"
        for evidence in item["evidence"]
    }
    superseded_hashes = {
        evidence["sha256"]
        for item in overlay_payload["classifications"]
        if item["migration_class"] == "superseded_historical_version"
        for evidence in item["evidence"]
    }
    paper = _indexed_paths(inputs.paper_database)
    books = _indexed_paths(inputs.books_database)
    operations = _indexed_paths(inputs.operations_database)
    book_aliases = _book_aliases(inputs.books_database)
    existing = {"papers": paper, "books": books, "operations": operations}
    paths = _markdown_files(inputs.workspace_root)
    file_rows: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    by_selected_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in paths:
        rel_path = path.relative_to(inputs.workspace_root).as_posix()
        digest = sha256_file(path)
        size = path.stat().st_size
        category = _selected_document_category(rel_path, experiment_classes)
        parts = Path(rel_path).parts
        experiment_id = parts[3] if parts[:3] == ("quant_lab", "vault", "30_experiments") and len(parts) >= 5 else None
        row: dict[str, Any] = {
            "rel_path": rel_path,
            "sha256": digest,
            "byte_size": size,
            "policy": "excluded_not_canonical",
            "reason": "Readable Markdown, but not selected by the canonical research-artifact policy.",
            "searchable": 0,
            "canonical_rel_path": None,
            "source_index": None,
            "source_index_hash_matches": None,
            "experiment_id": experiment_id,
            "artifact_category": category,
        }
        classification = experiment_classes.get(experiment_id or "")
        experiment_record = experiment_records.get(experiment_id or "")
        if digest in quarantined_hashes:
            row.update(
                policy="excluded_test_or_synthetic",
                reason="Content hash is bound to quarantined test/synthetic evidence, independent of filename.",
                artifact_category=None,
            )
        elif digest in superseded_hashes:
            row.update(
                policy="excluded_superseded",
                reason="Content hash is bound to a superseded historical artifact; explicit registry lookup is required.",
                artifact_category=None,
            )
        elif category:
            row.update(
                policy="canonical_research_artifact",
                reason="Selected canonical model/research/validation/decision evidence.",
                searchable=1,
            )
            selected.append(row)
            by_selected_hash[digest].append(row)
        elif classification == "test_or_synthetic_quarantined":
            row.update(policy="excluded_test_or_synthetic", reason="Quarantined test/synthetic experiment evidence.")
        elif classification == "superseded_historical_version":
            row.update(policy="excluded_superseded", reason="Superseded historical experiment evidence; available by explicit registry lookup.")
        elif classification in {"genuine_rejected_closed", "parked_awaiting_point_in_time_data"}:
            # This branch is defensive; real named evidence should be selected above.
            row.update(policy="registry_evidence_only", reason="Real experiment evidence represented by the canonical findings registry.")
        elif (exclusion := _base_exclusion(rel_path)) is not None:
            row.update(policy=exclusion[0], reason=exclusion[1])
        else:
            matched_index = next((name for name, mapping in existing.items() if rel_path in mapping), None)
            if matched_index:
                matches = existing[matched_index][rel_path] == digest
                row.update(
                    policy="covered_existing_index" if matches else "existing_index_refresh_required",
                    reason=(
                        f"Exact current content is covered by the {matched_index} index."
                        if matches else f"Path exists in the {matched_index} index but content changed after its build."
                    ),
                    searchable=int(matches),
                    canonical_rel_path=rel_path if matches else None,
                    source_index=matched_index,
                    source_index_hash_matches=int(matches),
                )
            elif rel_path in book_aliases:
                target = book_aliases[rel_path]
                matches = target["alias_sha256"] == digest
                row.update(
                    policy="exact_duplicate_alias" if matches else "existing_index_refresh_required",
                    reason="Exact duplicate alias covered by the local book index." if matches else "Book alias content drifted after indexing.",
                    searchable=0,
                    canonical_rel_path=target["canonical_rel_path"] if matches else None,
                    source_index="books",
                    source_index_hash_matches=int(matches),
                    canonical_sha256=target["canonical_sha256"] if matches else None,
                    canonical_source_index="books" if matches else None,
                )
        file_rows.append(row)

    canonical_rows: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for digest, group in sorted(by_selected_hash.items()):
        ordered = sorted(group, key=lambda row: row["rel_path"])
        canonical = ordered[0]
        canonical_rows.append(canonical)
        for alias in ordered[1:]:
            alias.update(
                policy="exact_duplicate_alias",
                reason="Exact duplicate of a selected canonical research artifact.",
                searchable=0,
                canonical_rel_path=canonical["rel_path"],
                canonical_sha256=canonical["sha256"],
                canonical_source_index="completion",
            )
            aliases.append(alias)

    aliases.extend(
        row
        for row in file_rows
        if row["policy"] == "exact_duplicate_alias"
        and row.get("canonical_source_index") == "books"
    )

    for row in file_rows:
        experiment_record = experiment_records.get(row["experiment_id"] or "")
        row["artifact_id"] = sha256_bytes(row["rel_path"].encode("utf-8"))[:24]
        row["mtime_ns"] = (inputs.workspace_root / row["rel_path"]).stat().st_mtime_ns
        row["disposition"] = _disposition(row["policy"])
        row["lineage_id"] = experiment_record["migration_lineage_id"] if experiment_record else None
        row["is_lineage_head"] = int(experiment_record["migration_latest_version"]) if experiment_record else None
        row["rule_version"] = POLICY_RULE_VERSION
        row["decision_basis_sha256"] = semantic_hash({
            "rel_path": row["rel_path"],
            "sha256": row["sha256"],
            "policy": row["policy"],
            "canonical_rel_path": row["canonical_rel_path"],
            "experiment_id": row["experiment_id"],
            "lineage_id": row["lineage_id"],
            "rule_version": POLICY_RULE_VERSION,
        })

    chunks: list[dict[str, Any]] = []
    for document in canonical_rows:
        path = inputs.workspace_root / document["rel_path"]
        doc_id = sha256_bytes(document["rel_path"].encode("utf-8"))[:24]
        document["doc_id"] = doc_id
        document["title"] = _title(path)
        for ordinal, (heading, content) in enumerate(stream_markdown(path)):
            chunks.append(
                {
                    "chunk_id": sha256_bytes(f"{document['rel_path']}\n{ordinal}\n{content}".encode("utf-8"))[:24],
                    "doc_id": doc_id,
                    "ordinal": ordinal,
                    "heading": heading,
                    "content": content,
                    "title": document["title"],
                    "category": document["artifact_category"],
                }
            )
    if canonical_rows and not chunks:
        raise CompletionError("selected canonical documents produced no searchable chunks")
    vectors: list[Sequence[float]] = []
    for offset in range(0, len(chunks), 64):
        batch = chunks[offset : offset + 64]
        texts = [row["content"] for row in batch]
        batch_vectors = [list(vector) for vector in embed(texts)]
        replay_vectors = [list(vector) for vector in embed(texts)]
        if len(batch_vectors) != len(batch):
            raise CompletionError("embedding provider returned an unexpected vector count")
        if batch_vectors != replay_vectors:
            raise CompletionError("embedding provider is nondeterministic for the frozen chunk batch")
        vectors.extend(batch_vectors)

    policy_path = staging / "markdown-index-policy.jsonl"
    _write_jsonl(policy_path, file_rows)
    database = staging / "canonical-research-artifacts.sqlite3"
    connection = sqlite3.connect(database)
    try:
        _init_artifact_schema(connection)
        for row in file_rows:
            connection.execute(
                "INSERT INTO markdown_policy VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    row["artifact_id"], row["rel_path"], row["sha256"], row["byte_size"],
                    row["mtime_ns"], row["disposition"], row["policy"], row["reason"],
                    row["searchable"], row["canonical_rel_path"],
                    row["source_index"], row["source_index_hash_matches"],
                    row["experiment_id"], row["lineage_id"], row["is_lineage_head"],
                    row["artifact_category"], row["rule_version"], row["decision_basis_sha256"],
                ),
            )
        for row in canonical_rows:
            connection.execute(
                "INSERT INTO documents VALUES (?,?,?,?,?,?,?,0)",
                (
                    row["doc_id"], row["artifact_id"], row["rel_path"], row["title"], row["artifact_category"],
                    row["sha256"], row["byte_size"],
                ),
            )
            if row["experiment_id"]:
                experiment_record = experiment_records[row["experiment_id"]]
                connection.execute(
                    "INSERT INTO document_run_links VALUES (?,?,?,?)",
                    (
                        row["doc_id"], row["experiment_id"], row["lineage_id"],
                        experiment_record["effective_decision_state"],
                    ),
                )
        canonical_doc_ids = {row["rel_path"]: row["doc_id"] for row in canonical_rows}
        for alias in aliases:
            connection.execute(
                "INSERT INTO aliases VALUES (?,?,?,?,?,?)",
                (
                    alias["artifact_id"],
                    canonical_doc_ids.get(alias["canonical_rel_path"]),
                    alias["canonical_source_index"], alias["canonical_rel_path"],
                    alias["canonical_sha256"],
                    alias["sha256"],
                ),
            )
        vector_rows: list[dict[str, Any]] = []
        vector_dimensions: set[int] = set()
        for row, vector in zip(chunks, vectors, strict=True):
            blob, dimension = _vector_blob(vector)
            vector_dimensions.add(dimension)
            vector_rows.append(
                {"chunk_id": row["chunk_id"], "dimension": dimension, "sha256": sha256_bytes(blob)}
            )
            connection.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                (
                    row["chunk_id"], row["doc_id"], row["ordinal"], row["heading"],
                    row["content"], sha256_bytes(row["content"].encode("utf-8")), blob, dimension,
                ),
            )
            connection.execute(
                "INSERT INTO chunks_fts VALUES (?,?,?,?,?)",
                (row["chunk_id"], row["title"], row["heading"], row["category"], row["content"]),
            )
        if len(vector_dimensions) != 1:
            raise CompletionError("embedding provider returned mixed vector dimensions")
        vector_dimension = next(iter(vector_dimensions))
        vector_semantic = semantic_hash(sorted(vector_rows, key=lambda row: row["chunk_id"]))
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "embedding_model": inputs.embedding_model,
            "embedding_provider_identity": embedding_provider_identity,
            "embedding_dimension": vector_dimension,
            "vector_semantic_sha256": vector_semantic,
            "cloud_allowed": "false",
            "workspace_markdown_count": len(file_rows),
            "historical_markdown_baseline": inputs.historical_markdown_baseline,
            "workspace_root": str(inputs.workspace_root),
        }
        connection.executemany("INSERT INTO index_meta VALUES (?,?)", metadata.items())
        policy_semantic = semantic_hash(file_rows)
        inventory_semantic = semantic_hash([
            {"path": row["rel_path"], "sha256": row["sha256"], "size": row["byte_size"]}
            for row in file_rows
        ])
        policy_build_id = f"policy-v1-{policy_semantic[:20]}"
        connection.execute(
            "INSERT INTO policy_builds VALUES (?,?,?,?,?,?,?)",
            (
                policy_build_id, inventory_semantic,
                sha256_file(staging / "canonical-findings.sqlite3"), POLICY_RULE_VERSION,
                sha256_file(Path(__file__)), len(file_rows), policy_semantic,
            ),
        )
        index_semantic = semantic_hash({
            "documents": [(row["rel_path"], row["sha256"]) for row in canonical_rows],
            "chunks": [(row["chunk_id"], sha256_bytes(row["content"].encode("utf-8"))) for row in chunks],
            "vectors": vector_rows,
            "model": inputs.embedding_model,
            "embedding_provider_identity": embedding_provider_identity,
            "chunker": CHUNKER_VERSION,
        })
        connection.execute(
            "INSERT INTO index_builds VALUES (?,?,?,?,?,?,?)",
            (
                f"index-v1-{index_semantic[:20]}", policy_build_id, inputs.embedding_model,
                CHUNKER_VERSION, len(canonical_rows), len(chunks), index_semantic,
            ),
        )
        connection.commit()
        integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
    finally:
        connection.close()
    if integrity != "ok" or foreign_keys:
        raise CompletionError(f"canonical artifact index failed integrity: {integrity}, {foreign_keys}")
    policy_counts = dict(sorted(Counter(row["policy"] for row in file_rows).items()))
    category_counts = dict(sorted(Counter(row["artifact_category"] for row in canonical_rows).items()))
    return {
        "database": database.name,
        "database_sha256": sha256_file(database),
        "policy_file": policy_path.name,
        "policy_file_sha256": sha256_file(policy_path),
        "workspace_markdown_count": len(file_rows),
        "historical_markdown_baseline": inputs.historical_markdown_baseline,
        "markdown_growth_since_baseline": len(file_rows) - inputs.historical_markdown_baseline,
        "policy_counts": policy_counts,
        "canonical_document_count": len(canonical_rows),
        "exact_duplicate_alias_count": len(aliases),
        "chunk_count": len(chunks),
        "category_counts": category_counts,
        "embedding_model": inputs.embedding_model,
        "embedding_provider_identity": embedding_provider_identity,
        "embedding_dimension": vector_dimension,
        "vector_semantic_sha256": vector_semantic,
        "cloud_allowed": False,
        "policy_semantic_sha256": semantic_hash(file_rows),
    }


def _optional_file_hash(path: Path | None) -> str | None:
    return sha256_file(path) if path is not None and path.is_file() else None


def _embedding_provider_identity(
    inputs: CompletionInputs,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None,
    embedding_cache: Path | None,
) -> str:
    if embed is not None:
        def stable_state(value: Any, field: str) -> Any:
            if value is None or isinstance(value, (bool, str, int)):
                return value
            if isinstance(value, float):
                if not math.isfinite(value):
                    raise CompletionError(f"embedding callable {field} is not finite")
                return value
            if isinstance(value, (list, tuple)):
                return [stable_state(item, field) for item in value]
            if isinstance(value, Mapping) and all(isinstance(key, str) for key in value):
                return {
                    key: stable_state(item, f"{field}.{key}")
                    for key, item in sorted(value.items())
                }
            raise CompletionError(
                f"embedding callable {field} has unsupported hidden state: {type(value).__name__}"
            )

        callable_function = embed
        bound_instance: Any | None = None
        if inspect.ismethod(embed):
            callable_function = embed.__func__
            if not inspect.isclass(embed.__self__):
                bound_instance = embed.__self__
        elif not inspect.isfunction(embed):
            call_method = getattr(type(embed), "__call__", None)
            if call_method is None:
                raise CompletionError("embedding callable has no inspectable implementation")
            callable_function = call_method
            bound_instance = embed
        try:
            source = inspect.getsource(callable_function).encode("utf-8")
        except (OSError, TypeError):
            if hasattr(callable_function, "__code__"):
                code = callable_function.__code__
                source = canonical_json(
                    {
                        "bytecode": code.co_code.hex(),
                        "constants": [repr(item) for item in code.co_consts],
                        "names": list(code.co_names),
                        "argcount": code.co_argcount,
                    }
                ).encode("utf-8")
            else:
                raise CompletionError("embedding callable source is not inspectable")
        code = getattr(callable_function, "__code__", None)
        closure_values = []
        if getattr(callable_function, "__closure__", None):
            closure_values = [
                stable_state(cell.cell_contents, f"closure[{index}]")
                for index, cell in enumerate(callable_function.__closure__)
            ]
        referenced_globals = {}
        if code is not None and hasattr(callable_function, "__globals__"):
            for name in sorted(set(code.co_names) & set(callable_function.__globals__)):
                referenced_globals[name] = stable_state(
                    callable_function.__globals__[name], f"global.{name}"
                )
        bound_state = None
        if bound_instance is not None:
            if not hasattr(bound_instance, "__dict__"):
                raise CompletionError(
                    "embedding callable instance has unsupported hidden state"
                )
            bound_state = stable_state(vars(bound_instance), "bound_instance")
        state = {
            "source_sha256": sha256_bytes(source),
            "defaults": stable_state(
                getattr(callable_function, "__defaults__", None), "defaults"
            ),
            "kwdefaults": stable_state(
                getattr(callable_function, "__kwdefaults__", None), "kwdefaults"
            ),
            "closure": closure_values,
            "referenced_globals": referenced_globals,
            "bound_instance": bound_state,
        }
        module = getattr(callable_function, "__module__", type(embed).__module__)
        qualname = getattr(callable_function, "__qualname__", type(embed).__qualname__)
        return (
            f"callable:{module}.{qualname}:"
            f"{semantic_hash(state)}"
        )

    local_rag_home = Path(
        os.environ.get(
            "QUANT_LAB_RAG_HOME",
            str(Path.home() / ".local" / "share" / "quant-lab-rag"),
        )
    ).expanduser()
    local_rag_packages = (
        local_rag_home / "venv" / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    local_distributions: dict[str, importlib.metadata.Distribution] = {}
    if local_rag_packages.is_dir():
        for distribution in importlib.metadata.distributions(path=[str(local_rag_packages)]):
            name = str(distribution.metadata.get("Name", "")).lower().replace("_", "-")
            if name:
                local_distributions[name] = distribution

    runtime_components: list[dict[str, Any]] = []
    for distribution, module_name in (
        ("fastembed", "fastembed"),
        ("onnxruntime", "onnxruntime"),
        ("tokenizers", "tokenizers"),
        ("numpy", "numpy"),
    ):
        installed_distribution: importlib.metadata.Distribution | None
        try:
            installed_distribution = importlib.metadata.distribution(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed_distribution = local_distributions.get(distribution)
        version = installed_distribution.version if installed_distribution else None
        spec = importlib.util.find_spec(module_name)
        origin = Path(spec.origin).resolve() if spec and spec.origin else None
        if origin is None and local_rag_packages.is_dir():
            package_init = local_rag_packages / module_name / "__init__.py"
            if package_init.is_file():
                origin = package_init.resolve()
        installed_files: list[dict[str, Any]] = []
        if installed_distribution is not None:
            for relative in sorted(
                installed_distribution.files or (), key=lambda item: str(item)
            ):
                path = Path(installed_distribution.locate_file(relative))
                if path.is_file() and not path.is_symlink():
                    installed_files.append(
                        {
                            "path": str(relative),
                            "size": path.stat().st_size,
                            "sha256": sha256_file(path),
                        }
                    )
        runtime_components.append(
            {
                "distribution": distribution,
                "version": version,
                "module_origin": str(origin) if origin else None,
                "module_sha256": sha256_file(origin) if origin and origin.is_file() else None,
                "installed_files_semantic_sha256": semantic_hash(installed_files),
            }
        )
    if runtime_components[0]["version"] is None and runtime_components[0]["module_sha256"] is None:
        raise CompletionError("FastEmbed runtime identity cannot be established")
    cache_rows: list[dict[str, Any]] = []
    if embedding_cache is not None and embedding_cache.is_dir():
        for path in sorted(embedding_cache.rglob("*")):
            if path.is_file() and not path.is_symlink():
                cache_rows.append(
                    {
                        "path": path.relative_to(embedding_cache).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    payload = {
        "provider": "fastembed",
        "runtime_components": runtime_components,
        "model": inputs.embedding_model,
        "model_cache_semantic_sha256": semantic_hash(cache_rows),
    }
    return f"fastembed:{semantic_hash(payload)}"


def _completion_identity(
    overlay_manifest: Mapping[str, Any],
    inputs: CompletionInputs,
    markdown_paths: Sequence[Path],
    embedding_provider_identity: str,
    historical_ids: set[str],
) -> str:
    markdown_inventory = [
        {
            "path": path.relative_to(inputs.workspace_root).as_posix(),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in markdown_paths
    ]
    live_inventory: list[dict[str, Any]] = []
    for folder in sorted(inputs.experiment_root.iterdir(), key=lambda item: item.name):
        if not folder.is_dir() or folder.name in historical_ids:
            continue
        _safe_experiment_id(folder.name)
        files = sorted(path for path in folder.rglob("*") if path.is_file())
        if not files:
            raise CompletionError(f"post-migration run has no auditable files: {folder.name}")
        for path in files:
            if path.is_symlink():
                raise CompletionError(f"post-migration run contains a symlink: {path}")
            live_inventory.append(
                {
                    "path": path.relative_to(inputs.workspace_root).as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    implementation_paths = {
        Path(__file__),
        Path(__file__).with_name("data_admission.py"),
        Path(__file__).with_name("lseg_rerun.py"),
        Path(__file__).with_name("strategy_inventory.py"),
        Path(__file__).parents[1] / "data" / "lseg_pit_adapter.py",
    }
    readiness_relative_paths = {
        *LSEG_EXECUTION_SOURCES,
        *VALIDATION_POLICY_SOURCES,
        *(
            path
            for contract in RERUN_RESEARCH_CONTRACTS.values()
            for path in contract["feature_sources"]
        ),
    }
    readiness_sources = [
        {
            "path": path,
            "sha256": sha256_file(inputs.workspace_root / path),
        }
        for path in sorted(readiness_relative_paths)
    ]
    implementation_sources = [
        {"path": path.name, "sha256": sha256_file(path)}
        for path in sorted(implementation_paths, key=lambda item: item.as_posix())
    ]
    overlay_manifest_path = Path(
        json.loads(inputs.overlay_pointer.read_text(encoding="utf-8"))["overlay_manifest"]
    )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "workspace_root": str(inputs.workspace_root),
        "source_paths": {
            "overlay_pointer": str(inputs.overlay_pointer),
            "source_findings_database": str(inputs.source_findings_database),
            "paper_database": str(inputs.paper_database) if inputs.paper_database else None,
            "books_database": str(inputs.books_database) if inputs.books_database else None,
            "operations_database": (
                str(inputs.operations_database) if inputs.operations_database else None
            ),
        },
        "historical_markdown_baseline": inputs.historical_markdown_baseline,
        "overlay_semantic_sha256": overlay_manifest["classification_semantic_sha256"],
        "overlay_manifest_sha256": sha256_file(overlay_manifest_path),
        "overlay_pointer_sha256": sha256_file(inputs.overlay_pointer),
        "source_tree_sha256": overlay_manifest["source_tree_sha256_after"],
        "markdown_inventory_sha256": semantic_hash(markdown_inventory),
        "live_control_plane_inventory_sha256": semantic_hash(live_inventory),
        "approval_trust_policy_hashes": sorted(
            (inputs.approval_trust_stores or {}).keys()
        ),
        "validation_runner_trust_policy_hashes": sorted(
            (inputs.validation_runner_trust_stores or {}).keys()
        ),
        "data_admission_trust_policy_hashes": sorted(
            (inputs.data_admission_trust_stores or {}).keys()
        ),
        "data_release_trust_policy_hashes": sorted(
            (inputs.data_release_trust_stores or {}).keys()
        ),
        "readiness_sources_sha256": semantic_hash(readiness_sources),
        "implementation_sources_sha256": semantic_hash(implementation_sources),
        "embedding_model": inputs.embedding_model,
        "embedding_provider_identity": embedding_provider_identity,
        "source_findings_database_sha256": sha256_file(inputs.source_findings_database),
        "paper_database_sha256": _optional_file_hash(inputs.paper_database),
        "books_database_sha256": _optional_file_hash(inputs.books_database),
        "operations_database_sha256": _optional_file_hash(inputs.operations_database),
        "policy_rule_version": POLICY_RULE_VERSION,
        "chunker_version": CHUNKER_VERSION,
    }
    return f"steps-9-11-v1-{semantic_hash(identity)[:20]}"


def _checksums(directory: Path) -> list[dict[str, str]]:
    return [
        {"path": path.relative_to(directory).as_posix(), "sha256": sha256_file(path)}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name not in {"checksums.sha256", "completion-manifest.json", "verification.json"}
    ]


def _make_read_only(root: Path, immutable: bool) -> bool:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = path.stat().st_mode
        path.chmod(mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)
    immutable_verified = False
    immutable_flag = getattr(stat, "UF_IMMUTABLE", 0)
    if immutable and immutable_flag:
        for path in [*sorted(root.rglob("*"), reverse=True), root]:
            os.chflags(path, path.stat().st_flags | immutable_flag)
        immutable_verified = bool(root.stat().st_flags & immutable_flag)
    return immutable_verified


def _remove_staging(root: Path) -> None:
    """Remove only a builder-owned staging tree, including a protected orphan."""
    if not root.exists():
        return
    if ".staging-" not in root.name or not root.name.startswith(".steps-9-11-"):
        raise CompletionError(f"refusing to remove a non-staging path: {root}")
    immutable_flag = getattr(stat, "UF_IMMUTABLE", 0)
    for path in [root, *sorted(root.rglob("*"), reverse=True)]:
        if immutable_flag:
            os.chflags(path, path.stat().st_flags & ~immutable_flag)
        path.chmod(path.stat().st_mode | stat.S_IWUSR)
    shutil.rmtree(root)


def _package_read_only(root: Path) -> bool:
    return not any(stat.S_IMODE(path.stat().st_mode) & 0o222 for path in [root, *root.rglob("*")])


def _pointer_payload(completion_id: str, completion_dir: Path, immutable: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "completion_id": completion_id,
        "completion_dir": str(completion_dir),
        "completion_manifest": str(completion_dir / "completion-manifest.json"),
        "completion_manifest_sha256": sha256_file(completion_dir / "completion-manifest.json"),
        "package_read_only_verified": _package_read_only(completion_dir),
        "package_local_user_immutable_verified": immutable,
        "updated_at": utc_now(),
    }
    payload["pointer_semantic_sha256"] = semantic_hash(payload)
    return payload


def _default_embedder(model_name: str, cache_dir: Path | None) -> Callable[[Sequence[str]], Sequence[Sequence[float]]]:
    try:
        from fastembed import TextEmbedding
    except ImportError:
        # Reuse the already provisioned local RAG runtime without installing a
        # second model stack into quant_lab's research environment.
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        local_rag_home = Path(
            os.environ.get(
                "QUANT_LAB_RAG_HOME",
                str(Path.home() / ".local" / "share" / "quant-lab-rag"),
            )
        ).expanduser()
        local_rag_packages = (
            local_rag_home / "venv" / "lib"
            / version / "site-packages"
        )
        if not local_rag_packages.is_dir():
            raise CompletionError(
                "fastembed is unavailable; install it or configure QUANT_LAB_RAG_HOME"
            )
        sys.path.append(str(local_rag_packages))
        from fastembed import TextEmbedding

    model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir) if cache_dir else None)

    def embed(texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return list(model.embed(list(texts), batch_size=32))

    return embed


@contextmanager
def _exclusive_build_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def build_completion_package(
    inputs: CompletionInputs,
    *,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    embedding_cache: Path | None = None,
    protect: bool = True,
) -> CompletionResult:
    """Serialize publication while preserving unique crash-recovery staging dirs."""
    lock = inputs.output_root.expanduser().resolve() / ".knowledge-completion-build.lock"
    with _exclusive_build_lock(lock):
        return _build_completion_package_locked(
            inputs,
            embed=embed,
            embedding_cache=embedding_cache,
            protect=protect,
        )


def _build_completion_package_locked(
    inputs: CompletionInputs,
    *,
    embed: Callable[[Sequence[str]], Sequence[Sequence[float]]] | None = None,
    embedding_cache: Path | None = None,
    protect: bool = True,
) -> CompletionResult:
    """Build and atomically publish the full step-9-to-11 completion package."""
    inputs = CompletionInputs(
        **{
            **inputs.__dict__,
            "workspace_root": inputs.workspace_root.resolve(),
            "output_root": inputs.output_root.resolve(),
            "overlay_pointer": inputs.overlay_pointer.resolve(),
            "source_findings_database": inputs.source_findings_database.resolve(),
        }
    )
    _, overlay_manifest, overlay_payload = _load_overlay(inputs)
    source_rows = _source_experiments(inputs.source_findings_database)
    markdown_paths = _markdown_files(inputs.workspace_root)
    provider_identity = _embedding_provider_identity(inputs, embed, embedding_cache)
    completion_id = _completion_identity(
        overlay_manifest, inputs, markdown_paths, provider_identity, set(source_rows)
    )
    packages_root = inputs.output_root / "completion"
    completion_dir = packages_root / completion_id
    pointer_path = inputs.output_root / "knowledge-completion-current.json"
    packages_root.mkdir(parents=True, exist_ok=True)
    for orphan in sorted(packages_root.glob(".steps-9-11-*.staging-*")):
        _remove_staging(orphan)
    if completion_dir.exists():
        report = verify_completion_package(inputs, completion_dir)
        if not all(report.values()):
            raise CompletionError("existing content-addressed completion package failed verification")
        immutable_flag = getattr(stat, "UF_IMMUTABLE", 0)
        immutable_verified = bool(
            immutable_flag and completion_dir.stat().st_flags & immutable_flag
        )
        _write_json(
            pointer_path,
            _pointer_payload(completion_id, completion_dir, immutable_verified),
        )
        return CompletionResult(completion_id, completion_dir, completion_dir / "completion-manifest.json", report)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{completion_id}.staging-", dir=packages_root)
    )
    try:
        registry = _build_registry(staging, inputs, overlay_manifest, overlay_payload, source_rows)
        embedder = embed or _default_embedder(inputs.embedding_model, embedding_cache)
        artifact_index = _build_artifact_index(
            staging, inputs, overlay_payload, embedder, provider_identity
        )
        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "completion_id": completion_id,
            "created_at": overlay_manifest["created_at"],
            "source_mode": "read_only",
            "cloud_allowed": False,
            "workspace_root": str(inputs.workspace_root),
            "source_overlay": {
                "pointer": str(inputs.overlay_pointer),
                "snapshot_id": overlay_manifest["snapshot_id"],
                "migration_id": overlay_manifest["migration_id"],
                "overlay_manifest_sha256": sha256_file(Path(json.loads(inputs.overlay_pointer.read_text())["overlay_manifest"])),
                "source_tree_sha256": overlay_manifest["source_tree_sha256_after"],
            },
            "step_9_registry": registry,
            "step_10_artifact_index": artifact_index,
            "step_11_rerun_readiness": {
                "execution_deferred_until_lseg_data": True,
                "parked_experiment_ids": list(EXPECTED_PARKED_IDS),
                "required_lseg_data": list(REQUIRED_LSEG_DATA),
                "required_controls": list(REQUIRED_RERUN_CONTROLS),
                "workflow_stages": list(WORKFLOW_STAGES),
                "deterministic_validation_gates": list(REQUIRED_VALIDATION_GATES),
            },
        }
        checksums = _checksums(staging)
        checksum_path = staging / "checksums.sha256"
        checksum_path.write_text(
            "".join(f"{item['sha256']}  {item['path']}\n" for item in checksums),
            encoding="utf-8",
        )
        with checksum_path.open("rb") as handle:
            os.fsync(handle.fileno())
        manifest["checksums_file"] = checksum_path.name
        manifest["checksums_file_sha256"] = sha256_file(checksum_path)
        manifest["manifest_semantic_sha256"] = semantic_hash(manifest)
        _write_json(staging / "completion-manifest.json", manifest)
        report = verify_completion_package(inputs, staging)
        _write_json(staging / "verification.json", {
            "verified_at": overlay_manifest["created_at"],
            "checks": report,
            "passed": all(report.values()),
        })
        if not all(report.values()):
            raise CompletionError(f"completion verification failed: {[k for k,v in report.items() if not v]}")
        _fsync_directory(staging)
        _make_read_only(staging, False)
        os.replace(staging, completion_dir)
        _fsync_directory(packages_root)
        immutable_verified = _make_read_only(completion_dir, protect)
        _write_json(
            pointer_path,
            _pointer_payload(completion_id, completion_dir, immutable_verified),
        )
        return CompletionResult(completion_id, completion_dir, completion_dir / "completion-manifest.json", report)
    except Exception:
        if staging.exists():
            _remove_staging(staging)
        raise


def verify_completion_package(inputs: CompletionInputs, completion_dir: Path) -> dict[str, bool]:
    """Verify source lineage, registry semantics, index policy, and LSEG readiness."""
    manifest_path = completion_dir / "completion-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_rows = _source_experiments(inputs.source_findings_database)
    checks: dict[str, bool] = {}
    manifest_without_hash = dict(manifest)
    claimed_manifest_hash = manifest_without_hash.pop("manifest_semantic_sha256", None)
    checks["completion_manifest_semantic_hash"] = (
        _is_sha256(claimed_manifest_hash)
        and semantic_hash(manifest_without_hash) == claimed_manifest_hash
    )
    completion_id = manifest.get("completion_id")
    checks["completion_identity_and_directory"] = (
        isinstance(completion_id, str)
        and (
            completion_dir.name == completion_id
            or completion_dir.name.startswith(f".{completion_id}.staging-")
        )
        and completion_id
        == _completion_identity(
            _load_overlay(inputs)[1],
            inputs,
            _markdown_files(inputs.workspace_root),
            manifest.get("step_10_artifact_index", {}).get(
                "embedding_provider_identity", ""
            ),
            set(source_rows),
        )
    )
    checks["package_governance_fail_closed"] = (
        manifest.get("source_mode") == "read_only"
        and manifest.get("cloud_allowed") is False
        and manifest.get("step_10_artifact_index", {}).get("cloud_allowed") is False
        and manifest.get("step_11_rerun_readiness", {}).get(
            "execution_deferred_until_lseg_data"
        )
        is True
    )
    checksum_path = completion_dir / manifest["checksums_file"]
    checks["checksums_file_hash"] = sha256_file(checksum_path) == manifest["checksums_file_sha256"]
    checksum_rows: list[tuple[str, str]] = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, rel_path = line.split("  ", 1)
        checksum_rows.append((digest, rel_path))
    checks["all_package_checksums"] = bool(checksum_rows) and all(
        (completion_dir / rel_path).is_file()
        and sha256_file(completion_dir / rel_path) == digest
        for digest, rel_path in checksum_rows
    )
    _, overlay_manifest, overlay_payload = _load_overlay(inputs)
    folders = _experiment_folders(inputs.experiment_root)
    registry_path = completion_dir / manifest["step_9_registry"]["database"]
    policy_db = completion_dir / manifest["step_10_artifact_index"]["database"]
    checks["registry_hash"] = sha256_file(registry_path) == manifest["step_9_registry"]["database_sha256"]
    checks["artifact_index_hash"] = sha256_file(policy_db) == manifest["step_10_artifact_index"]["database_sha256"]
    checks["policy_file_hash"] = sha256_file(completion_dir / manifest["step_10_artifact_index"]["policy_file"]) == manifest["step_10_artifact_index"]["policy_file_sha256"]
    checks["overlay_source_tree_binding"] = manifest["source_overlay"]["source_tree_sha256"] == overlay_manifest["source_tree_sha256_after"]

    connection = sqlite3.connect(f"file:{registry_path.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        checks["registry_integrity"] = connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        checks["registry_foreign_keys"] = not connection.execute("PRAGMA foreign_key_check").fetchall()
        ids = [row[0] for row in connection.execute("SELECT experiment_id FROM experiments ORDER BY experiment_id")]
        live_rows = connection.execute(
            "SELECT * FROM control_plane_runs ORDER BY run_id"
        ).fetchall()
        operational_count = connection.execute("SELECT COUNT(*) FROM operational_runs").fetchone()[0]
        counts = dict(connection.execute("SELECT migration_class,COUNT(*) FROM experiments GROUP BY migration_class"))
        queue = [row[0] for row in connection.execute("SELECT experiment_id FROM action_queue ORDER BY experiment_id")]
        source_hashes = dict(connection.execute("SELECT experiment_id,source_hash FROM experiments"))
        rejected_missing_reason = connection.execute("SELECT COUNT(*) FROM experiments WHERE migration_class='genuine_rejected_closed' AND (kill_reason IS NULL OR length(kill_reason)=0)").fetchone()[0]
        parked_rows = connection.execute("SELECT experiment_id,missing_data_json,action_required FROM experiments WHERE migration_class='parked_awaiting_point_in_time_data' ORDER BY experiment_id").fetchall()
        false_approval = connection.execute("SELECT COUNT(*) FROM experiments WHERE approval_required<>0 OR accepted<>0 OR promotion_status IN('approval_required','approved')").fetchone()[0]
        synthetic_queue = connection.execute("SELECT COUNT(*) FROM action_queue WHERE run_kind IN('test','synthetic','smoke')").fetchone()[0]
        bad_supersession = connection.execute(
            """SELECT COUNT(*) FROM experiments old
               LEFT JOIN experiments replacement ON replacement.experiment_id=old.superseded_by_experiment_id
               WHERE old.migration_class='superseded_historical_version'
                 AND (replacement.experiment_id IS NULL OR replacement.lineage_id<>old.lineage_id)"""
        ).fetchone()[0]
        evidence_count = connection.execute("SELECT COUNT(*) FROM evidence").fetchone()[0]
        lineage_count = connection.execute("SELECT COUNT(*) FROM lineages").fetchone()[0]
        non_historical = connection.execute(
            "SELECT COUNT(*) FROM experiments WHERE record_source<>'historical_overlay_v2'"
        ).fetchone()[0]
        bad_edges = connection.execute(
            """SELECT COUNT(*) FROM lineage_edges edge
               JOIN experiments old ON old.experiment_id=edge.from_experiment_id
               JOIN experiments new ON new.experiment_id=edge.to_experiment_id
               WHERE edge.edge_type='superseded_by' AND old.lineage_id<>new.lineage_id"""
        ).fetchone()[0]
        planned_edges = connection.execute(
            "SELECT legacy_anchor_run_id,reserved_child_run_id,readiness_path,readiness_sha256 FROM planned_lineage_edges ORDER BY legacy_anchor_run_id"
        ).fetchall()
        live_edges = connection.execute(
            "SELECT run_id,parent_run_id,legacy_anchor_run_id,edge_type FROM control_plane_lineage_edges ORDER BY run_id"
        ).fetchall()
        registry_build = connection.execute("SELECT * FROM registry_builds").fetchone()
        manifest_rows = connection.execute("SELECT experiment_id,historical_manifest_path,historical_manifest_sha256 FROM experiments").fetchall()
        reruns = connection.execute("SELECT * FROM rerun_requirements ORDER BY experiment_id").fetchall()
    finally:
        connection.close()
    historical_folders = sorted(set(folders) & set(source_rows))
    live_folder_ids = sorted(set(folders) - set(source_rows))
    checks["exact_853_traceability"] = ids == historical_folders == sorted(source_rows) and len(ids) == EXPECTED_EXPERIMENT_COUNT
    checks["historical_plus_control_plane_union"] = (
        [row["run_id"] for row in live_rows] == live_folder_ids
        and operational_count == EXPECTED_EXPERIMENT_COUNT + len(live_rows)
        and registry_build["live_run_count"] == len(live_rows)
        and manifest["step_9_registry"]["live_control_plane_run_count"] == len(live_rows)
    )
    checks["live_lineage_is_total_and_enforced"] = (
        len(live_edges) == len(live_rows)
        and [row["run_id"] for row in live_edges] == [row["run_id"] for row in live_rows]
        and all(
            (
                edge["edge_type"] == "parent_child"
                and edge["parent_run_id"] is not None
                and edge["legacy_anchor_run_id"] is None
            )
            or (
                edge["edge_type"] == "legacy_migration_anchor"
                and edge["parent_run_id"] is None
                and edge["legacy_anchor_run_id"] in EXPECTED_PARKED_IDS
            )
            for edge in live_edges
        )
    )
    planned_hashes = {
        row["reserved_child_run_id"]: row["readiness_sha256"]
        for row in planned_edges
    }
    checks["live_roots_have_signed_data_admission"] = all(
        row["parent_run_id"] is not None
        or (
            row["run_id"] in planned_hashes
            and row["admission_plan_sha256"] == planned_hashes[row["run_id"]]
            and all(
                _is_sha256(row[field])
                for field in (
                    "admission_subject_sha256", "human_release_sha256",
                    "admission_trust_policy_sha256", "release_trust_policy_sha256",
                )
            )
        )
        for row in live_rows
    )
    checks["exact_classification_partition"] = counts == EXPECTED_CLASSIFICATIONS
    checks["exact_three_parked_action_queue"] = queue == sorted(EXPECTED_PARKED_IDS)
    checks["no_test_or_synthetic_action"] = synthetic_queue == 0
    checks["superseded_lineage_connected"] = bad_supersession == 0
    checks["rejected_kill_reasons"] = rejected_missing_reason == 0
    checks["parked_missing_data_specific"] = len(parked_rows) == 3 and all(
        json.loads(row["missing_data_json"])["provider"] == "LSEG"
        and json.loads(row["missing_data_json"])["missing_items"] == list(REQUIRED_LSEG_DATA)
        and json.loads(row["missing_data_json"])["markets"]
        and json.loads(row["missing_data_json"])["replication_window"]
        and json.loads(row["missing_data_json"])["checked_at"]
        and json.loads(row["missing_data_json"])["checker"]["execution_id"]
        and _is_sha256(
            json.loads(row["missing_data_json"])["checker"]["classification_basis_sha256"]
        )
        and json.loads(row["missing_data_json"])["evidence"]
        and all(
            _is_sha256(item["sha256"])
            for item in json.loads(row["missing_data_json"])["evidence"]
        )
        and json.loads(row["missing_data_json"])["entitlement_requirements"]["aws_region"]
        == "ap-southeast-2"
        and row["action_required"]
        for row in parked_rows
    )
    checks["no_false_approval"] = false_approval == 0
    checks["exact_evidence_traceability"] = evidence_count == EXPECTED_EVIDENCE_COUNT
    checks["exact_847_lineages"] = lineage_count == 847
    checks["historical_source_discriminator"] = non_historical == 0
    checks["registry_builder_source_binding"] = (
        registry_build is not None
        and registry_build["builder_source_sha256"] == sha256_file(Path(__file__))
        and registry_build["overlay_semantic_sha256"] == overlay_manifest["classification_semantic_sha256"]
    )
    checks["lineage_edges_same_lineage"] = bad_edges == 0
    checks["source_hashes_match"] = source_hashes == {key: row["source_hash"] for key, row in source_rows.items()}
    checks["all_historical_manifest_hashes"] = len(manifest_rows) == EXPECTED_EXPERIMENT_COUNT and all(
        sha256_file(completion_dir / row["historical_manifest_path"]) == row["historical_manifest_sha256"]
        for row in manifest_rows
    )
    checks["historical_manifests_exact_schema"] = all(
        tuple(json.loads((completion_dir / row["historical_manifest_path"]).read_text())["manifest"]) == MANIFEST_FIELDS
        for row in manifest_rows
    )
    checks["three_rerun_packages"] = [row["experiment_id"] for row in reruns] == sorted(EXPECTED_PARKED_IDS)
    checks["three_planned_legacy_edges"] = (
        [row["legacy_anchor_run_id"] for row in planned_edges] == sorted(EXPECTED_PARKED_IDS)
        and len({row["reserved_child_run_id"] for row in planned_edges}) == 3
        and all(
            sha256_file(completion_dir / row["readiness_path"]) == row["readiness_sha256"]
            for row in planned_edges
        )
    )
    checks["reruns_fail_closed"] = all(row["execution_authorized"] == 0 and row["root_run_id"] is None for row in reruns)
    checks["reruns_share_control_plane_contract"] = all(
        json.loads(row["required_data_json"]) == list(REQUIRED_LSEG_DATA)
        and json.loads(row["required_controls_json"]) == list(REQUIRED_RERUN_CONTROLS)
        and json.loads(row["validation_gates_json"]) == list(REQUIRED_VALIDATION_GATES)
        and json.loads(row["workflow_stages_json"]) == list(WORKFLOW_STAGES)
        for row in reruns
    )
    readiness_payloads = [
        json.loads((completion_dir / row["readiness_path"]).read_text(encoding="utf-8"))
        for row in reruns
    ]
    checks["rerun_package_semantic_hashes"] = all(
        _is_sha256(payload.get("package_semantic_sha256"))
        and semantic_hash({key: value for key, value in payload.items() if key != "package_semantic_sha256"})
        == payload["package_semantic_sha256"]
        for payload in readiness_payloads
    )
    checks["rerun_evidence_bindings"] = all(
        all(
            (inputs.workspace_root / item["path"]).is_file()
            and sha256_file(inputs.workspace_root / item["path"]) == item["sha256"]
            for item in payload["source_bindings"]["evidence"]
        )
        for payload in readiness_payloads
    )

    connection = sqlite3.connect(f"file:{policy_db.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        checks["artifact_index_integrity"] = connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        checks["artifact_index_foreign_keys"] = not connection.execute("PRAGMA foreign_key_check").fetchall()
        policy_rows = connection.execute(
            "SELECT rel_path,sha256,policy,disposition FROM markdown_policy ORDER BY rel_path"
        ).fetchall()
        policy_count = len(policy_rows)
        documents = connection.execute("SELECT rel_path,sha256,category FROM documents ORDER BY rel_path").fetchall()
        chunk_count = connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        chunk_rows = connection.execute(
            """SELECT c.chunk_id,d.title,c.heading,d.category,c.content,
                      c.chunk_sha256,c.embedding,c.embedding_dim
               FROM chunks c JOIN documents d USING(doc_id) ORDER BY c.chunk_id"""
        ).fetchall()
        fts_rows = connection.execute(
            "SELECT chunk_id,title,heading,category,content FROM chunks_fts ORDER BY chunk_id"
        ).fetchall()
        index_meta = dict(connection.execute("SELECT key,value FROM index_meta"))
        forbidden = connection.execute(
            """SELECT COUNT(*) FROM documents WHERE
               lower(rel_path) LIKE '%/zai_ocr/%'
               OR lower(rel_path) LIKE '%/formula_retries/%'
               OR lower(rel_path) LIKE '%/site-packages/%'
               OR rel_path IN (
                    SELECT 'quant_lab/vault/30_experiments/'||experiment_id||'/'||substr(path,length('quant_lab/vault/30_experiments/'||experiment_id||'/')+1)
                    FROM (SELECT '' AS experiment_id,'' AS path) WHERE 0
               )"""
        ).fetchone()[0]
        bad_experiments = connection.execute(
            """SELECT COUNT(*) FROM documents d JOIN markdown_policy p USING(rel_path)
               WHERE p.experiment_id IN (
                 SELECT experiment_id FROM markdown_policy
                 WHERE policy IN('excluded_test_or_synthetic','excluded_superseded')
               )"""
        ).fetchone()[0]
        searchable_real = connection.execute(
            """SELECT COUNT(*) FROM documents d JOIN markdown_policy p USING(rel_path)
               WHERE p.experiment_id IN (?,?,?,?,?,?,?,?,?)""",
            tuple(sorted(
                item["experiment_id"] for item in overlay_payload["classifications"]
                if item["migration_class"] in {"genuine_rejected_closed", "parked_awaiting_point_in_time_data"}
            )),
        ).fetchone()[0]
        duplicate_document_hashes = connection.execute("SELECT sha256 FROM documents GROUP BY sha256 HAVING COUNT(*)>1").fetchall()
        alias_rows = connection.execute(
            """SELECT p.rel_path AS alias_rel_path,p.sha256 AS policy_sha256,
                      p.canonical_rel_path AS policy_canonical_rel_path,
                      a.canonical_source_index,
                      a.canonical_rel_path AS target_rel_path,
                      a.canonical_sha256,a.sha256 AS alias_sha256
               FROM markdown_policy p JOIN aliases a ON a.alias_artifact_id=p.artifact_id
               WHERE p.policy='exact_duplicate_alias' ORDER BY p.rel_path"""
        ).fetchall()
        alias_policy_count = connection.execute(
            "SELECT COUNT(*) FROM markdown_policy WHERE policy='exact_duplicate_alias'"
        ).fetchone()[0]
        policy_build = connection.execute("SELECT * FROM policy_builds").fetchone()
    finally:
        connection.close()
    current_markdown_count = len(_markdown_files(inputs.workspace_root))
    checks["every_current_markdown_has_policy"] = policy_count == current_markdown_count == manifest["step_10_artifact_index"]["workspace_markdown_count"]
    current_markdown_hashes = {
        path.relative_to(inputs.workspace_root).as_posix(): sha256_file(path)
        for path in _markdown_files(inputs.workspace_root)
    }
    checks["markdown_policy_exact_set_and_hashes"] = {
        row["rel_path"]: row["sha256"] for row in policy_rows
    } == current_markdown_hashes
    checks["dense_policy_document_bijection"] = sum(
        row["disposition"] == "dense_embed" for row in policy_rows
    ) == len(documents)
    checks["policy_builder_source_binding"] = (
        policy_build is not None
        and policy_build["evaluator_source_sha256"] == sha256_file(Path(__file__))
        and policy_build["registry_sha256"] == sha256_file(registry_path)
    )
    checks["historical_10142_baseline_reconciled"] = manifest["step_10_artifact_index"]["historical_markdown_baseline"] == HISTORICAL_MARKDOWN_BASELINE and current_markdown_count >= HISTORICAL_MARKDOWN_BASELINE
    checks["canonical_documents_source_hashes"] = all(
        sha256_file(inputs.workspace_root / row["rel_path"]) == row["sha256"] for row in documents
    )
    checks["canonical_findings_searchable"] = searchable_real > 0 and chunk_count > 0 and chunk_count == fts_count
    checks["no_ocr_test_superseded_in_index"] = forbidden == 0 and bad_experiments == 0
    checks["canonical_exact_deduplication"] = not duplicate_document_hashes
    expected_fts = [
        (row["chunk_id"], row["title"], row["heading"], row["category"], row["content"])
        for row in chunk_rows
    ]
    actual_fts = [tuple(row) for row in fts_rows]
    checks["fts_exact_content_bijection"] = (
        chunk_count == fts_count == len(chunk_rows) and expected_fts == actual_fts
        and all(
            sha256_bytes(row["content"].encode("utf-8")) == row["chunk_sha256"]
            for row in chunk_rows
        )
    )
    vector_rows: list[dict[str, Any]] = []
    vector_dimensions: set[int] = set()
    vectors_valid = True
    for row in chunk_rows:
        dimension = row["embedding_dim"]
        blob = row["embedding"]
        vector_dimensions.add(dimension)
        if dimension <= 0 or len(blob) != dimension * 4:
            vectors_valid = False
            continue
        vector = array("f")
        vector.frombytes(blob)
        if any(not math.isfinite(value) for value in vector) or not any(value != 0.0 for value in vector):
            vectors_valid = False
        vector_rows.append(
            {"chunk_id": row["chunk_id"], "dimension": dimension, "sha256": sha256_bytes(blob)}
        )
    checks["all_chunks_have_valid_vectors"] = (
        vectors_valid
        and len(vector_dimensions) == 1
        and int(index_meta.get("embedding_dimension", "-1")) == next(iter(vector_dimensions), -1)
        and semantic_hash(sorted(vector_rows, key=lambda row: row["chunk_id"]))
        == index_meta.get("vector_semantic_sha256")
        == manifest["step_10_artifact_index"]["vector_semantic_sha256"]
    )
    checks["all_aliases_have_canonical_targets"] = (
        alias_policy_count == len(alias_rows)
        == manifest["step_10_artifact_index"]["exact_duplicate_alias_count"]
        and all(
            row["policy_canonical_rel_path"]
            and row["policy_canonical_rel_path"] == row["target_rel_path"]
            and row["policy_sha256"] == row["alias_sha256"]
            and _is_sha256(row["canonical_sha256"])
            and row["canonical_source_index"] in {"completion", "books"}
            for row in alias_rows
        )
    )
    return checks


def search_canonical_artifacts(
    database: Path, query: str, *, limit: int = 10, destination: str = "local"
) -> list[dict[str, Any]]:
    """Run local FTS over the canonical artifact index."""
    if destination != "local":
        raise CompletionError(
            "canonical quant artifacts are local-only; external model egress is disabled"
        )
    terms = list(dict.fromkeys(_SEARCH_TERM_RE.findall(query.lower())))[:20]
    if not terms:
        raise ValueError("query contains no searchable terms")
    expression = " OR ".join(f'"{term}"' for term in terms)
    connection = sqlite3.connect(f"file:{database.resolve()}?mode=ro&immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        workspace_root_row = connection.execute(
            "SELECT value FROM index_meta WHERE key='workspace_root'"
        ).fetchone()
        if not workspace_root_row:
            raise CompletionError("artifact index has no workspace provenance root")
        workspace_root = Path(workspace_root_row[0])
        rows = connection.execute(
            """SELECT d.rel_path,d.title,d.category,d.sha256,c.chunk_sha256,
                      links.experiment_id,links.lineage_id,links.effective_decision_state,
                      c.heading,c.content,
                      bm25(chunks_fts,0.0,2.0,1.5,1.0,3.0) AS score
               FROM chunks_fts JOIN chunks c USING(chunk_id)
               JOIN documents d USING(doc_id)
               LEFT JOIN document_run_links links USING(doc_id)
               WHERE chunks_fts MATCH ? ORDER BY score LIMIT ?""",
            (expression, limit),
        ).fetchall()
        results = [dict(row) for row in rows]
        for row in results:
            source = workspace_root / row["rel_path"]
            if not source.is_file() or sha256_file(source) != row["sha256"]:
                raise CompletionError(f"refusing stale canonical citation: {row['rel_path']}")
            if sha256_bytes(row["content"].encode("utf-8")) != row["chunk_sha256"]:
                raise CompletionError(f"indexed chunk hash mismatch: {row['rel_path']}")
        return results
    finally:
        connection.close()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def validate_rerun_activation(
    *,
    plan_path: Path,
    admission_bundle_path: Path,
    admission_root: Path,
    admission_trust_store: DataAdmissionTrustStore,
    release_trust_store: DataReleaseTrustStore,
    workspace_root: Path,
) -> VerifiedDataAdmission:
    """Return a sealed admission only after file, hash, signature and PIT checks.

    The API intentionally accepts paths rather than caller-provided mappings.
    This removes the authority-forgery seam that previously accepted invented
    hashes and self-asserted booleans.
    """
    try:
        return verify_data_admission(
            plan_path=plan_path,
            bundle_path=admission_bundle_path,
            admission_root=admission_root,
            admission_trust_store=admission_trust_store,
            release_trust_store=release_trust_store,
            workspace_root=workspace_root,
        )
    except Exception as exc:
        if isinstance(exc, CompletionError):
            raise
        raise CompletionError(f"LSEG rerun admission failed closed: {exc}") from exc


def _validate_activation_result(
    result: Any,
    *,
    plan: Mapping[str, Any],
    admission: VerifiedDataAdmission,
    workspace_root: Path,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise CompletionError("activation callback must return the created control-plane receipt")
    required = {
        "run_id", "manifest_path", "manifest_sha256", "manifest_identity_sha256",
        "event_count", "workflow_status", "validation_status", "promotion_status",
        "dataset_descriptor",
    }
    if set(result) != required:
        raise CompletionError("activation callback returned an incomplete control-plane receipt")
    if result["run_id"] != admission.root_run_id or result["run_id"] != plan["next_run_id"]:
        raise CompletionError("activation callback did not preserve the reserved root run ID")
    path = workspace_root.resolve() / str(result["manifest_path"])
    expected_path = (
        workspace_root.resolve() / "quant_lab" / "vault" / "30_experiments"
        / admission.root_run_id / "run_manifest.json"
    )
    if path.resolve() != expected_path.resolve() or path.is_symlink() or not path.is_file():
        raise CompletionError("activation callback did not create the expected run manifest")
    store = ExperimentRunStore(workspace_root.resolve() / "quant_lab" / "vault", admission.root_run_id)
    replay = store.verify_integrity()
    manifest = store.load()
    expected = {
        "manifest_sha256": replay["manifest_hash"],
        "manifest_identity_sha256": replay["manifest_identity_hash"],
        "event_count": 2,
        "workflow_status": "created",
        "validation_status": "not_run",
        "promotion_status": "ineligible",
    }
    if any(result[field] != value for field, value in expected.items()):
        raise CompletionError("created run failed the initial control-plane postcondition")
    descriptor = result["dataset_descriptor"]
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {
            "admission_subject_sha256", "data_manifest_id", "schema_sha256",
            "partition_count", "row_count", "markets", "scanner", "default_batch_size",
        }
        or descriptor["admission_subject_sha256"] != admission.admission_subject_sha256
        or descriptor["data_manifest_id"] != admission.data_manifest_id
        or not _is_sha256(descriptor["schema_sha256"])
        or type(descriptor["partition_count"]) is not int
        or descriptor["partition_count"] < 1
        or type(descriptor["row_count"]) is not int
        or descriptor["row_count"] < 1
        or descriptor["scanner"] != "AdmittedLSEGDataset.iter_batches"
        or descriptor["default_batch_size"] != 65_536
    ):
        raise CompletionError("created run lacks a bounded admitted-dataset scanner contract")
    if (
        manifest.data_manifest_id != admission.data_manifest_id
        or manifest.data_asof_utc != admission.data_asof_utc
        or manifest.feature_manifest_hash != admission.feature_manifest_hash
        or manifest.code_commit != admission.code_commit
        or manifest.split_manifest_id != admission.split_manifest_id
    ):
        raise CompletionError("created run is not bound to the verified admission artifacts")
    return dict(result)


def guarded_activate_rerun(
    *,
    plan_path: Path,
    admission_bundle_path: Path,
    admission_root: Path,
    admission_trust_store: DataAdmissionTrustStore,
    release_trust_store: DataReleaseTrustStore,
    workspace_root: Path,
    activation_ledger: Path,
) -> Any:
    """Create the one built-in local LSEG run at most once after admission.

    There is no caller-supplied callback. An exclusive reservation is persisted
    before the local LSEG adapter/control-plane entrypoint. A crash or failure
    stays fail-closed and requires human recovery; it is never retried
    automatically with a second provider side effect.
    """
    admission = validate_rerun_activation(
        plan_path=plan_path,
        admission_bundle_path=admission_bundle_path,
        admission_root=admission_root,
        admission_trust_store=admission_trust_store,
        release_trust_store=release_trust_store,
        workspace_root=workspace_root,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if activation_ledger.is_symlink():
        raise CompletionError("activation ledger cannot be a symlink")
    activation_ledger.mkdir(parents=True, exist_ok=True)
    if not activation_ledger.is_dir():
        raise CompletionError("activation ledger must be a directory")
    ledger_details = activation_ledger.stat()
    if ledger_details.st_uid != os.getuid() or stat.S_IMODE(ledger_details.st_mode) & 0o022:
        raise CompletionError(
            "activation ledger must be current-user-owned and not group/world writable"
        )
    root_run_id = str(plan["next_run_id"])
    activation_id = sha256_bytes(
        f"{root_run_id}:{admission.admission_subject_sha256}".encode("utf-8")
    )[:24]
    record_path = activation_ledger / f"{activation_id}.json"
    plan_hash = admission.plan_sha256
    receipt_hash = admission.admission_bundle_sha256
    reservation = {
        "schema_version": SCHEMA_VERSION,
        "activation_id": activation_id,
        "root_run_id": root_run_id,
        "plan_sha256": plan_hash,
        "admission_bundle_sha256": receipt_hash,
        "admission_subject_sha256": admission.admission_subject_sha256,
        "verified_admission": admission.to_dict(),
        "status": "reserved",
        "reserved_at": utc_now(),
        "result": None,
    }
    if record_path.is_symlink():
        raise CompletionError("activation record cannot be a symlink")
    try:
        with record_path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(reservation, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(activation_ledger)
    except FileExistsError:
        if record_path.is_symlink() or not record_path.is_file():
            raise CompletionError("existing activation record is not a regular file")
        prior = json.loads(record_path.read_text(encoding="utf-8"))
        if prior.get("plan_sha256") != plan_hash or prior.get("admission_bundle_sha256") != receipt_hash:
            raise CompletionError("reserved run ID is already bound to different activation evidence")
        if prior.get("status") == "completed":
            return _validate_activation_result(
                prior.get("result"),
                plan=plan,
                admission=admission,
                workspace_root=workspace_root,
            )
        raise CompletionError(
            f"activation {activation_id} is {prior.get('status')}; human recovery is required"
        )
    try:
        from quant_lab.research.lseg_rerun import _create_admitted_lseg_run

        result = _validate_activation_result(
            _create_admitted_lseg_run(
                plan,
                admission,
                workspace_root=workspace_root,
                admission_root=admission_root,
                admission_bundle_path=admission_bundle_path,
            ),
            plan=plan,
            admission=admission,
            workspace_root=workspace_root,
        )
        canonical_json(result)
    except Exception as exc:
        failed = {**reservation, "status": "failed", "failed_at": utc_now(), "error_type": type(exc).__name__}
        _write_json(record_path, failed)
        raise
    completed = {**reservation, "status": "completed", "completed_at": utc_now(), "result": result}
    _write_json(record_path, completed)
    record_path.chmod(record_path.stat().st_mode & ~0o222)
    return result


__all__ = [
    "CompletionError",
    "CompletionInputs",
    "CompletionResult",
    "EXPECTED_CLASSIFICATIONS",
    "EXPECTED_PARKED_IDS",
    "HISTORICAL_MARKDOWN_BASELINE",
    "MANIFEST_FIELDS",
    "REQUIRED_LSEG_DATA",
    "REQUIRED_RERUN_CONTROLS",
    "build_completion_package",
    "guarded_activate_rerun",
    "search_canonical_artifacts",
    "validate_rerun_activation",
    "verify_completion_package",
]
