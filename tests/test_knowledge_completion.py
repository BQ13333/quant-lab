from __future__ import annotations

import copy
import base64
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
import quant_lab.data.lseg_pit_adapter as lseg_pit_adapter

from quant_lab.research.knowledge_completion import (
    CompletionError,
    CompletionInputs,
    EXPECTED_CLASSIFICATIONS,
    EXPECTED_PARKED_IDS,
    LSEG_EXECUTION_SOURCES,
    REQUIRED_LSEG_FIELDS,
    build_completion_package,
    guarded_activate_rerun,
    search_canonical_artifacts,
    validate_rerun_activation,
    verify_completion_package,
    _discover_control_plane_runs,
    _embedding_provider_identity,
)
from quant_lab.research.control_plane import ExperimentRunStore, RunKind, RunManifest
from quant_lab.research.data_admission import (
    archive_verified_data_admission,
    canonical_arrow_schema_sha256,
    DataAdmissionTrustStore,
    DataReleaseTrustStore,
    frozen_research_config_payload,
    frozen_validation_policy_payload,
    verify_archived_data_admission,
    DataAdmissionError,
    _verify_inventory,
)
from quant_lab.data.lseg_pit_adapter import open_admitted_lseg_dataset
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


WORKSPACE = Path(
    os.environ.get("QUANT_LAB_PRIVATE_WORKSPACE", "private_corpus")
).expanduser().resolve()
RAG_HOME = Path(
    os.environ.get("QUANT_LAB_RAG_HOME", ".private-rag")
).expanduser().resolve()
KNOWLEDGE = RAG_HOME / "knowledge_layers"


def deterministic_embed(texts):
    return [
        [((len(text) + ordinal * 17) % 101) / 101.0 for ordinal in range(24)]
        for text in texts
    ]


@pytest.fixture(scope="module")
def completion(tmp_path_factory):
    required = (
        KNOWLEDGE / "findings-hygiene-current.json",
        KNOWLEDGE / "findings.sqlite3",
        RAG_HOME / "rag" / "knowledge.sqlite3",
        KNOWLEDGE / "books.sqlite3",
        KNOWLEDGE / "operations.sqlite3",
    )
    if not WORKSPACE.is_dir() or not all(path.is_file() for path in required):
        pytest.skip(
            "private-corpus integration test; configure "
            "QUANT_LAB_PRIVATE_WORKSPACE and QUANT_LAB_RAG_HOME"
        )
    output = tmp_path_factory.mktemp("steps-9-11")
    configured = CompletionInputs(
        workspace_root=WORKSPACE,
        output_root=output,
        overlay_pointer=KNOWLEDGE / "findings-hygiene-current.json",
        source_findings_database=KNOWLEDGE / "findings.sqlite3",
        paper_database=RAG_HOME / "rag" / "knowledge.sqlite3",
        books_database=KNOWLEDGE / "books.sqlite3",
        operations_database=KNOWLEDGE / "operations.sqlite3",
        embedding_model="deterministic-test-embedding-v1",
    )
    result = build_completion_package(configured, embed=deterministic_embed, protect=False)
    return configured, result


def package_manifest(result):
    return json.loads(result.manifest_path.read_text(encoding="utf-8"))


def test_embedding_callable_identity_binds_closure_state(tmp_path):
    configured = CompletionInputs(
        workspace_root=tmp_path,
        output_root=tmp_path / "output",
        overlay_pointer=tmp_path / "overlay.json",
        source_findings_database=tmp_path / "findings.sqlite3",
    )

    def make_embedder(scale):
        def embed(texts):
            return [[float(scale)] for _ in texts]

        return embed

    first = _embedding_provider_identity(configured, make_embedder(1), None)
    second = _embedding_provider_identity(configured, make_embedder(2), None)
    assert first != second


def test_embedding_callable_identity_binds_bound_instance_state(tmp_path):
    configured = CompletionInputs(
        workspace_root=tmp_path,
        output_root=tmp_path / "output",
        overlay_pointer=tmp_path / "overlay.json",
        source_findings_database=tmp_path / "findings.sqlite3",
    )

    class StatefulEmbedder:
        def __init__(self, scale):
            self.scale = scale

        def embed(self, texts):
            return [[float(self.scale)] for _ in texts]

    first = _embedding_provider_identity(configured, StatefulEmbedder(1).embed, None)
    second = _embedding_provider_identity(configured, StatefulEmbedder(2).embed, None)
    assert first != second


def test_repeat_builds_are_byte_deterministic(completion, tmp_path):
    configured, first = completion
    orphan = tmp_path / "repeat-output" / "completion" / ".steps-9-11-v1-orphan.staging-dead"
    orphan.mkdir(parents=True)
    (orphan / "partial.json").write_text("{}", encoding="utf-8")
    repeated_inputs = CompletionInputs(
        **{**configured.__dict__, "output_root": tmp_path / "repeat-output"}
    )
    second = build_completion_package(
        repeated_inputs, embed=deterministic_embed, protect=False
    )
    assert not orphan.exists()
    assert first.completion_id == second.completion_id
    assert (first.completion_dir / "checksums.sha256").read_bytes() == (
        second.completion_dir / "checksums.sha256"
    ).read_bytes()
    assert package_manifest(first)["step_9_registry"]["database_sha256"] == package_manifest(
        second
    )["step_9_registry"]["database_sha256"]
    assert package_manifest(first)["step_10_artifact_index"]["database_sha256"] == package_manifest(
        second
    )["step_10_artifact_index"]["database_sha256"]


def test_registry_exactly_reconciles_frozen_history(completion):
    _, result = completion
    manifest = package_manifest(result)
    assert all(result.verification.values())
    assert manifest["step_9_registry"]["experiment_count"] == 853
    assert manifest["step_9_registry"]["evidence_count"] == 2_557
    assert manifest["step_9_registry"]["classification_counts"] == EXPECTED_CLASSIFICATIONS
    assert sorted(manifest["step_9_registry"]["action_queue_experiment_ids"]) == sorted(EXPECTED_PARKED_IDS)
    assert manifest["step_9_registry"]["approval_required_count"] == 0
    assert manifest["step_9_registry"]["accepted_count"] == 0
    connection = sqlite3.connect(
        result.completion_dir / manifest["step_9_registry"]["database"]
    )
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM control_plane_lineage_edges"
        ).fetchone()[0] == manifest["step_9_registry"]["live_control_plane_run_count"]
    finally:
        connection.close()


def test_nonempty_live_control_plane_discovery_replays_exactly(tmp_path):
    workspace = tmp_path / "workspace"
    vault = workspace / "quant_lab" / "vault"
    run_id = "LSEG-RERUN-FIXTURE-R1"
    manifest = RunManifest.new(
        run_id=run_id,
        parent_run_id=None,
        run_kind=RunKind.RESEARCH,
        hypothesis="Point-in-time fixture hypothesis",
        universe='{"markets":["XKLS"]}',
        data_manifest_id="lseg-fixture-v1",
        data_asof_utc="2026-07-01T00:00:00+00:00",
        feature_manifest_hash="a" * 64,
        code_commit="b" * 40,
        target="one_month_return",
        label_horizon="one_month",
        split_manifest_id="split-fixture-v1",
        created_at="2026-07-02T00:00:00+00:00",
    )
    ExperimentRunStore(vault, run_id).create(manifest)
    configured = CompletionInputs(
        workspace_root=workspace,
        output_root=tmp_path / "output",
        overlay_pointer=tmp_path / "overlay.json",
        source_findings_database=tmp_path / "findings.sqlite3",
    )
    rows = _discover_control_plane_runs(configured, set())
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["event_count"] == 1
    assert rows[0]["workflow_status"] == "created"


def test_registry_constraints_reject_forged_approval(completion, tmp_path):
    _, result = completion
    manifest = package_manifest(result)
    source = result.completion_dir / manifest["step_9_registry"]["database"]
    database = tmp_path / "registry.sqlite3"
    shutil.copy2(source, database)
    database.chmod(0o600)
    connection = sqlite3.connect(database)
    with pytest.raises(sqlite3.IntegrityError):
        connection.execute(
            "UPDATE experiments SET promotion_status='approved' WHERE experiment_id=?",
            (EXPECTED_PARKED_IDS[0],),
        )
    connection.close()


def test_policy_is_exhaustive_and_default_index_is_clean(completion):
    _, result = completion
    manifest = package_manifest(result)
    database = result.completion_dir / manifest["step_10_artifact_index"]["database"]
    connection = sqlite3.connect(database)
    policy_count = connection.execute("SELECT COUNT(*) FROM markdown_policy").fetchone()[0]
    workspace_count = sum(1 for path in WORKSPACE.rglob("*.md") if path.is_file())
    assert policy_count == workspace_count
    assert connection.execute(
        "SELECT COUNT(*) FROM documents WHERE lower(rel_path) LIKE '%zai_ocr%'"
    ).fetchone()[0] == 0
    assert connection.execute(
        """SELECT COUNT(*) FROM documents d JOIN markdown_policy p USING(artifact_id)
           WHERE p.policy IN('excluded_test_or_synthetic','excluded_superseded')"""
    ).fetchone()[0] == 0
    assert connection.execute(
        "SELECT COUNT(*) FROM documents GROUP BY sha256 HAVING COUNT(*)>1"
    ).fetchall() == []
    assert connection.execute("SELECT COUNT(*) FROM aliases").fetchone()[0] == connection.execute(
        "SELECT COUNT(*) FROM markdown_policy WHERE policy='exact_duplicate_alias'"
    ).fetchone()[0]
    connection.close()


def test_completion_manifest_governance_mutation_is_detected(completion, tmp_path):
    configured, result = completion
    copied = tmp_path / result.completion_id
    shutil.copytree(result.completion_dir, copied)
    manifest_path = copied / "completion-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cloud_allowed"] = True
    manifest_path.chmod(0o600)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    report = verify_completion_package(configured, copied)
    assert report["completion_manifest_semantic_hash"] is False
    assert report["package_governance_fail_closed"] is False


def test_canonical_findings_are_searchable_with_provenance(completion):
    _, result = completion
    manifest = package_manifest(result)
    database = result.completion_dir / manifest["step_10_artifact_index"]["database"]
    rows = search_canonical_artifacts(database, "Malaysia survivor point in time", limit=10)
    assert rows
    assert all(row["rel_path"] and row["title"] and row["category"] for row in rows)
    assert all("zai_ocr" not in row["rel_path"].lower() for row in rows)
    with pytest.raises(CompletionError):
        search_canonical_artifacts(
            database, "Malaysia", destination="cerebras-cloud"
        )


def _canonical_bytes(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _write_readonly(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.write_bytes(content)
    path.chmod(0o400)
    return {"path": path.name, "byte_size": len(content), "sha256": _sha(content)}


def _signed(payload, key):
    unsigned = {**payload, "signature": base64.b64encode(b"0" * 64).decode("ascii")}
    signing = dict(unsigned)
    signing.pop("signature")
    unsigned["signature"] = base64.b64encode(key.sign(_canonical_bytes(signing))).decode("ascii")
    return unsigned


def _authority(tmp_path, monkeypatch):
    identities = {
        "entitlement": ("RaaS Entitlement Officer", "authorized_entitlement_verifier"),
        "manifest": ("RaaS Data Steward", "authorized_data_manifest_signer"),
        "execution": ("RaaS Execution Steward", "authorized_execution_contract_signer"),
        "release": ("Research Approver", "authorized_data_release_approver"),
    }
    keys = {name: Ed25519PrivateKey.generate() for name in identities}

    def signer(name):
        identity, role = identities[name]
        public = keys[name].public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        return {
            "identity": identity,
            "key_id": f"{name}-key-v1",
            "role": role,
            "public_key_base64": base64.b64encode(public).decode("ascii"),
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_until": "2027-01-01T00:00:00+00:00",
            "revoked_at": None,
        }

    admission_payload = {
        "schema_version": 1,
        "signers": [signer("entitlement"), signer("manifest"), signer("execution")],
    }
    release_payload = {"schema_version": 1, "signers": [signer("release")]}
    trust_dir = tmp_path / "trust"
    trust_dir.mkdir(mode=0o700)
    admission_path = trust_dir / "admission.json"
    release_path = trust_dir / "release.json"
    admission_path.write_bytes(_canonical_bytes(admission_payload))
    release_path.write_bytes(_canonical_bytes(release_payload))
    admission_hash = _sha(admission_path.read_bytes())
    release_hash = _sha(release_path.read_bytes())
    admission_path.chmod(0o400)
    release_path.chmod(0o400)
    monkeypatch.setattr(
        DataAdmissionTrustStore, "_read_admin_anchor",
        classmethod(lambda cls: frozenset({admission_hash})),
    )
    monkeypatch.setattr(
        DataReleaseTrustStore, "_read_admin_anchor",
        classmethod(lambda cls: frozenset({release_hash})),
    )
    return (
        keys,
        identities,
        DataAdmissionTrustStore(admission_path),
        DataReleaseTrustStore(release_path),
        admission_hash,
        release_hash,
    )


def _valid_admission(completion, tmp_path, monkeypatch):
    _, result = completion
    source_plan = result.completion_dir / "rerun-readiness" / f"{EXPECTED_PARKED_IDS[0]}.json"
    plan_bytes = source_plan.read_bytes()
    plan = json.loads(plan_bytes)
    root = tmp_path / "admission"
    root.mkdir(mode=0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    for row in [
        *plan["frozen_source_contracts"]["feature_sources"],
        *plan["frozen_source_contracts"]["split_and_validation_sources"],
    ]:
        source = WORKSPACE / row["path"]
        target = workspace / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    source_rows = [
        *plan["frozen_source_contracts"]["feature_sources"],
        *plan["frozen_source_contracts"]["split_and_validation_sources"],
    ]
    repository = workspace / "quant_lab"
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    # Runtime control-plane artifacts are deliberately outside source control;
    # mirror the production repository rule so retries do not create source drift.
    (repository / ".gitignore").write_text("vault/30_experiments/\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repository), "config", "user.email", "fixture@example.test"], check=True)
    subprocess.run(["git", "-C", str(repository), "config", "user.name", "Fixture"], check=True)
    subprocess.run(["git", "-C", str(repository), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repository), "commit", "-q", "-m", "frozen fixture"], check=True)
    code_commit = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    git_tree = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD^{tree}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()

    keys, identities, admission_trust, release_trust, admission_hash, release_hash = _authority(
        tmp_path, monkeypatch
    )
    plan_ref = _write_readonly(root / "plan.json", plan_bytes)
    evidence_ref = _write_readonly(root / "entitlement-evidence.txt", b"fixture LSEG access evidence\n")
    source_bundle_hash = _sha(_canonical_bytes(source_rows))
    code_ref = _write_readonly(
        root / "code-snapshot.json",
        _canonical_bytes(
            {
                "schema_version": 1,
                "code_commit": code_commit,
                "git_tree": git_tree,
                "source_bundle_sha256": source_bundle_hash,
                "sources": source_rows,
            }
        ),
    )
    validation_ref = _write_readonly(
        root / "validation-policy.json",
        _canonical_bytes(frozen_validation_policy_payload(plan)),
    )
    research_ref = _write_readonly(
        root / "research-config.json",
        _canonical_bytes(frozen_research_config_payload(plan)),
    )
    partition_dir = root / "data"
    partition_dir.mkdir(mode=0o700)

    now = datetime.now(timezone.utc).replace(microsecond=0)
    t_checked = (now - timedelta(minutes=4)).isoformat()
    t_generated = (now - timedelta(minutes=3)).isoformat()
    t_execution = (now - timedelta(minutes=2)).isoformat()
    t_approved = (now - timedelta(minutes=1)).isoformat()
    t_expiry = (now + timedelta(days=1)).isoformat()
    data_asof = (now - timedelta(minutes=5)).isoformat()
    contract = plan["data_contract"]
    window_start = datetime.fromisoformat(contract["replication_window"]["start"]).replace(
        tzinfo=timezone.utc
    )
    window_end = datetime.fromisoformat(contract["replication_window"]["end"]).replace(
        tzinfo=timezone.utc
    )
    observation_dates = []
    session_cursor = window_start
    while session_cursor <= window_end:
        if session_cursor.weekday() < 5:
            observation_dates.append(session_cursor)
        session_cursor += timedelta(days=1)
    generated_datetime = datetime.fromisoformat(t_generated)
    plan_hash = _sha(plan_bytes)
    entitlement = _signed(
        {
            "schema_version": 1,
            "attestation_id": "entitlement-attestation-v1",
            "provider": "LSEG",
            "customer_legal_entity": "RaaS Motion",
            "entitled_account_id": "raas-account-001",
            "requester_pays_account_id": "123456789012",
            "aws_region": contract["expected_aws_region"],
            "dataset_product_id": "lseg-qa-fixture",
            "delivery_method": "s3_direct",
            "trial_id": "qa-trial-2026",
            "permitted_use": "research_trial",
            "permitted_markets": contract["markets"],
            "redistribution_allowed": False,
            "data_location_constraints": ["local-and-authorized-aws-only"],
            "valid_from": "2026-01-01T00:00:00+00:00",
            "valid_until": "2027-01-01T00:00:00+00:00",
            "checked_at": t_checked,
            "source_evidence_ref": evidence_ref,
            "plan_sha256": plan_hash,
            "legacy_anchor_run_id": plan["legacy_anchor_run_id"],
            "reserved_root_run_id": plan["next_run_id"],
            "attestor_id": identities["entitlement"][0],
            "signer_key_id": "entitlement-key-v1",
            "trust_policy_sha256": admission_hash,
            "signature_algorithm": "Ed25519",
        },
        keys["entitlement"],
    )
    entitlement_ref = _write_readonly(root / "entitlement.json", _canonical_bytes(entitlement))
    calendar_ref = _write_readonly(
        root / "exchange-calendar.json",
        _canonical_bytes(
            {
                "schema_version": 1,
                "markets": {
                    market: [value.date().isoformat() for value in observation_dates]
                    for market in contract["markets"]
                },
            }
        ),
    )
    inventory = []
    for ordinal, market in enumerate(sorted(contract["markets"])):
        relative_path = f"part-{ordinal:03d}-{market}.parquet"
        row_count = len(observation_dates)
        table = pa.table(
            {
                "stable_instrument_id": [f"{market}-FIXTURE"] * row_count,
                "primary_exchange_mic": [market] * row_count,
                "listing_effective_at": [window_start] * row_count,
                "delisting_effective_at": [None] * (row_count - 1) + [window_end],
                "security_status_effective_at": observation_dates,
                "currency": ["USD"] * row_count,
                "unadjusted_close": [100.0 + index for index in range(row_count)],
                "total_return_or_adjustment_factors": [1.0] * row_count,
                "cash_dividends": [0.0] * row_count,
                "stock_splits": [1.0] * row_count,
                "rights_and_other_corporate_actions": ["none"] * row_count,
                "trading_volume": [1_000 + index for index in range(row_count)],
                "shares_outstanding_or_free_float": [10_000] * row_count,
                "publication_timestamp_utc": observation_dates,
                "effective_timestamp_utc": observation_dates,
                "revision_or_vintage_id": ["v1"] * row_count,
                "ingestion_timestamp_utc": [generated_datetime] * row_count,
            }
        )
        sink = pa.BufferOutputStream()
        pq.write_table(table, sink, compression=None)
        partition_ref = _write_readonly(
            partition_dir / relative_path,
            sink.getvalue().to_pybytes(),
        )
        schema_hash = canonical_arrow_schema_sha256(table.schema)
        inventory.append({
            "relative_path": relative_path,
            "byte_size": partition_ref["byte_size"],
            "sha256": partition_ref["sha256"],
            "row_count": table.num_rows,
            "market": market,
            "min_effective_at": observation_dates[0].isoformat(),
            "max_effective_at": observation_dates[-1].isoformat(),
            "min_published_at": observation_dates[0].isoformat(),
            "max_published_at": observation_dates[-1].isoformat(),
            "min_ingested_at": t_generated,
            "max_ingested_at": t_generated,
            "schema_sha256": schema_hash,
        })
    inventory_ref = _write_readonly(root / "inventory.json", _canonical_bytes(inventory))
    content_root = _sha(_canonical_bytes(inventory))
    profile = {
        "schema_version": 1,
        "manifest_id": "lseg-data-manifest-v1",
        "markets": contract["markets"],
        "fields": list(REQUIRED_LSEG_FIELDS),
        "coverage_metrics": {name: 1.0 for name in contract["coverage_thresholds"]},
        "flags": {
            "point_in_time": True,
            "includes_inactive_and_delisted": True,
            "corporate_actions_complete": True,
            "tradability_and_liquidity_complete": True,
            "timestamp_lineage_complete": True,
            "revision_vintages_retained": True,
            "sealed_out_of_sample_period": True,
        },
        "row_count": len(observation_dates) * len(inventory),
        "partition_count": len(inventory),
        "generated_at": t_generated,
        "runner_id": identities["manifest"][0],
    }
    profile_ref = _write_readonly(root / "profiling.json", _canonical_bytes(profile))
    data_manifest = _signed(
        {
            "schema_version": 1,
            "manifest_id": "lseg-data-manifest-v1",
            "provider": "LSEG",
            "dataset_product_id": entitlement["dataset_product_id"],
            "entitled_account_id": entitlement["entitled_account_id"],
            "requester_pays_account_id": entitlement["requester_pays_account_id"],
            "aws_region": entitlement["aws_region"],
            "entitlement_attestation_sha256": entitlement_ref["sha256"],
            "plan_sha256": plan_hash,
            "legacy_anchor_run_id": plan["legacy_anchor_run_id"],
            "reserved_root_run_id": plan["next_run_id"],
            "generated_at": t_generated,
            "data_asof_utc": data_asof,
            "requested_start": contract["replication_window"]["start"],
            "requested_end": contract["replication_window"]["end"],
            "markets": contract["markets"],
            "fields": list(REQUIRED_LSEG_FIELDS),
            "storage_format": "parquet",
            "schema_sha256": schema_hash,
            "inventory_ref": inventory_ref,
            "profiling_ref": profile_ref,
            "calendar_ref": calendar_ref,
            "partition_root": "data",
            "content_root_sha256": content_root,
            "partition_count": len(inventory),
            "row_count": len(observation_dates) * len(inventory),
            "total_bytes": sum(item["byte_size"] for item in inventory),
            "point_in_time": True,
            "includes_inactive_and_delisted": True,
            "corporate_actions_complete": True,
            "tradability_and_liquidity_complete": True,
            "timestamp_lineage_complete": True,
            "revision_vintages_retained": True,
            "sealed_out_of_sample_period": True,
            "coverage_metrics": {name: 1.0 for name in contract["coverage_thresholds"]},
            "calendar_policy_sha256": calendar_ref["sha256"],
            "timezone_policy_sha256": "3" * 64,
            "corporate_action_policy_sha256": "4" * 64,
            "license_classification": "LSEG_QA_RESEARCH_TRIAL",
            "admission_runner_id": identities["manifest"][0],
            "signer_key_id": "manifest-key-v1",
            "trust_policy_sha256": admission_hash,
            "signature_algorithm": "Ed25519",
        },
        keys["manifest"],
    )
    manifest_ref = _write_readonly(root / "data-manifest.json", _canonical_bytes(data_manifest))
    execution = _signed(
        {
            "schema_version": 1,
            "contract_id": "lseg-execution-contract-v1",
            "plan_sha256": plan_hash,
            "legacy_anchor_run_id": plan["legacy_anchor_run_id"],
            "reserved_root_run_id": plan["next_run_id"],
            "source_bundle_sha256": source_bundle_hash,
            "code_commit": code_commit,
            "clean_tree": True,
            "code_snapshot_ref": code_ref,
            "validation_policy_ref": validation_ref,
            "research_config_ref": research_ref,
            "feature_spec_sha256": plan["frozen_source_contracts"]["feature_spec_hash"],
            "split_policy_sha256": plan["frozen_source_contracts"]["split_and_validation_policy_hash"],
            "validation_policy_sha256": validation_ref["sha256"],
            "research_config_sha256": research_ref["sha256"],
            "required_markets": contract["markets"],
            "required_fields": list(REQUIRED_LSEG_FIELDS),
            "required_start": contract["replication_window"]["start"],
            "required_end": contract["replication_window"]["end"],
            "required_coverage": dict(contract["coverage_thresholds"]),
            "created_at": t_execution,
            "signer_id": identities["execution"][0],
            "signer_key_id": "execution-key-v1",
            "trust_policy_sha256": admission_hash,
            "signature_algorithm": "Ed25519",
        },
        keys["execution"],
    )
    execution_ref = _write_readonly(root / "execution.json", _canonical_bytes(execution))
    subject = {
        "plan_sha256": plan_hash,
        "entitlement_attestation_sha256": entitlement_ref["sha256"],
        "admitted_data_manifest_sha256": manifest_ref["sha256"],
        "execution_contract_sha256": execution_ref["sha256"],
        "legacy_anchor_run_id": plan["legacy_anchor_run_id"],
        "reserved_root_run_id": plan["next_run_id"],
    }
    release = _signed(
        {
            "schema_version": 1,
            "release_id": "human-data-release-v1",
            "decision": "authorize_rerun",
            "admission_subject_sha256": _sha(_canonical_bytes(subject)),
            **subject,
            "approved_by": identities["release"][0],
            "approved_at": t_approved,
            "expires_at": t_expiry,
            "limitations_acknowledged": ["Trial data remains licensed and non-redistributable."],
            "signer_key_id": "release-key-v1",
            "trust_policy_sha256": release_hash,
            "signature_algorithm": "Ed25519",
        },
        keys["release"],
    )
    release_ref = _write_readonly(root / "release.json", _canonical_bytes(release))
    bundle = {
        "schema_version": 1,
        "admission_id": "lseg-admission-bundle-v1",
        "plan_ref": plan_ref,
        "entitlement_ref": entitlement_ref,
        "data_manifest_ref": manifest_ref,
        "execution_contract_ref": execution_ref,
        "human_release_ref": release_ref,
        "created_at": t_approved,
    }
    bundle_path = root / "bundle.json"
    _write_readonly(bundle_path, _canonical_bytes(bundle))
    return {
        "plan": plan,
        "plan_path": source_plan,
        "bundle_path": bundle_path,
        "root": root,
        "workspace": workspace,
        "admission_trust": admission_trust,
        "release_trust": release_trust,
    }


def test_all_three_rerun_plans_fail_closed_before_data(completion, tmp_path, monkeypatch):
    _, result = completion
    for experiment_id in EXPECTED_PARKED_IDS:
        plan_path = result.completion_dir / "rerun-readiness" / f"{experiment_id}.json"
        with pytest.raises(CompletionError):
            guarded_activate_rerun(
                plan_path=plan_path,
                admission_bundle_path=tmp_path / "missing-bundle.json",
                admission_root=tmp_path,
                admission_trust_store=None,
                release_trust_store=None,
                workspace_root=WORKSPACE,
                activation_ledger=tmp_path / experiment_id,
            )


def test_fabricated_hex_mapping_cannot_reach_callback(completion, tmp_path, monkeypatch):
    with pytest.raises(CompletionError):
        guarded_activate_rerun(
            plan_path=completion[1].completion_dir / "rerun-readiness" / f"{EXPECTED_PARKED_IDS[0]}.json",
            admission_bundle_path=tmp_path / "fabricated.json",
            admission_root=tmp_path,
            admission_trust_store=None,
            release_trust_store=None,
            workspace_root=WORKSPACE,
            activation_ledger=tmp_path / "activations",
        )


def test_partition_tamper_blocks_before_callback(completion, tmp_path, monkeypatch):
    fixture = _valid_admission(completion, tmp_path, monkeypatch)
    partition = next((fixture["root"] / "data").glob("part-000-*.parquet"))
    partition.chmod(0o600)
    partition.write_bytes(b"tampered\n")
    partition.chmod(0o400)
    with pytest.raises(CompletionError):
        guarded_activate_rerun(
            plan_path=fixture["plan_path"],
            admission_bundle_path=fixture["bundle_path"],
            admission_root=fixture["root"],
            admission_trust_store=fixture["admission_trust"],
            release_trust_store=fixture["release_trust"],
            workspace_root=fixture["workspace"],
            activation_ledger=tmp_path / "activations",
        )


@pytest.mark.parametrize("fault", ["sparse_window", "mic_drift", "missing_field"])
def test_physical_parquet_faults_fail_before_admission(tmp_path, fault):
    root = tmp_path / fault
    data_root = root / "data"
    data_root.mkdir(parents=True, mode=0o700)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 31, tzinfo=timezone.utc)
    dates = (
        [start, end]
        if fault == "sparse_window"
        else [start, datetime(2026, 2, 1, tzinfo=timezone.utc), end]
    )
    count = len(dates)
    physical_market = "XSES" if fault == "mic_drift" else "XKLS"
    table = pa.table(
        {
            "stable_instrument_id": ["FIXTURE"] * count,
            "primary_exchange_mic": [physical_market] * count,
            "listing_effective_at": [start] * count,
            "delisting_effective_at": [None] * (count - 1) + [end],
            "security_status_effective_at": dates,
            "currency": ["MYR"] * count,
            "unadjusted_close": [100.0] * count,
            "total_return_or_adjustment_factors": [1.0] * count,
            "cash_dividends": [0.0] * count,
            "stock_splits": [1.0] * count,
            "rights_and_other_corporate_actions": ["none"] * count,
            "trading_volume": [1_000] * count,
            "shares_outstanding_or_free_float": [10_000] * count,
            "publication_timestamp_utc": dates,
            "effective_timestamp_utc": dates,
            "revision_or_vintage_id": ["v1"] * count,
            "ingestion_timestamp_utc": [end] * count,
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=None)
    reference = _write_readonly(data_root / "part.parquet", sink.getvalue().to_pybytes())
    schema_hash = canonical_arrow_schema_sha256(table.schema)
    inventory = [
        {
            "relative_path": "part.parquet",
            "byte_size": reference["byte_size"],
            "sha256": reference["sha256"],
            "row_count": count,
            "market": "XKLS",
            "min_effective_at": start.isoformat(),
            "max_effective_at": end.isoformat(),
            "min_published_at": start.isoformat(),
            "max_published_at": end.isoformat(),
            "min_ingested_at": end.isoformat(),
            "max_ingested_at": end.isoformat(),
            "schema_sha256": schema_hash,
        }
    ]
    fields = list(REQUIRED_LSEG_FIELDS)
    if fault == "missing_field":
        fields.append("physically_absent_required_field")
    manifest = {
        "partition_root": "data",
        "schema_sha256": schema_hash,
        "fields": fields,
        "markets": ["XKLS"],
        "requested_start": "2026-01-01",
        "requested_end": "2026-03-31",
    }
    calendar_dates = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            calendar_dates.append(cursor.date().isoformat())
        cursor += timedelta(days=1)
    calendar = {
        "schema_version": 1,
        "markets": {"XKLS": calendar_dates},
    }
    with pytest.raises(DataAdmissionError):
        _verify_inventory(
            root, manifest, _canonical_bytes(inventory), _canonical_bytes(calendar), 1,
            {name: 0.98 for name in (
                "stable_identifier_fraction", "membership_event_completeness",
                "corporate_action_completeness", "required_timestamp_fraction",
                "price_observation_fraction",
            )},
        )


def test_physical_coverage_accepts_distinct_point_in_time_revisions(tmp_path):
    root = tmp_path / "revisions"
    data_root = root / "data"
    data_root.mkdir(parents=True, mode=0o700)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    end = datetime(2026, 3, 31, tzinfo=timezone.utc)
    sessions = []
    cursor = start
    while cursor <= end:
        if cursor.weekday() < 5:
            sessions.append(cursor)
        cursor += timedelta(days=1)
    effective = [*sessions, sessions[0]]
    revision_ids = ["v1"] * len(sessions) + ["v2"]
    published = [value + timedelta(hours=16) for value in effective]
    ingested = [value + timedelta(hours=17) for value in effective]
    count = len(effective)
    table = pa.table(
        {
            "stable_instrument_id": ["MY-FIXTURE"] * count,
            "primary_exchange_mic": ["XKLS"] * count,
            "listing_effective_at": [start] * count,
            "delisting_effective_at": [None] * count,
            "security_status_effective_at": effective,
            "currency": ["MYR"] * count,
            "unadjusted_close": [100.0] * count,
            "total_return_or_adjustment_factors": [1.0] * count,
            "cash_dividends": [0.0] * count,
            "stock_splits": [1.0] * count,
            "rights_and_other_corporate_actions": ["none"] * count,
            "trading_volume": [1_000] * count,
            "shares_outstanding_or_free_float": [10_000] * count,
            "publication_timestamp_utc": published,
            "effective_timestamp_utc": effective,
            "revision_or_vintage_id": revision_ids,
            "ingestion_timestamp_utc": ingested,
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, compression=None)
    reference = _write_readonly(data_root / "part.parquet", sink.getvalue().to_pybytes())
    schema_hash = canonical_arrow_schema_sha256(table.schema)
    inventory = [
        {
            "relative_path": "part.parquet",
            "byte_size": reference["byte_size"],
            "sha256": reference["sha256"],
            "row_count": count,
            "market": "XKLS",
            "min_effective_at": min(effective).isoformat(),
            "max_effective_at": max(effective).isoformat(),
            "min_published_at": min(published).isoformat(),
            "max_published_at": max(published).isoformat(),
            "min_ingested_at": min(ingested).isoformat(),
            "max_ingested_at": max(ingested).isoformat(),
            "schema_sha256": schema_hash,
        }
    ]
    manifest = {
        "partition_root": "data",
        "schema_sha256": schema_hash,
        "fields": list(REQUIRED_LSEG_FIELDS),
        "markets": ["XKLS"],
        "requested_start": "2026-01-01",
        "requested_end": "2026-03-31",
    }
    calendar = {
        "schema_version": 1,
        "markets": {"XKLS": [value.date().isoformat() for value in sessions]},
    }
    _, row_count, _, _, coverage = _verify_inventory(
        root,
        manifest,
        _canonical_bytes(inventory),
        _canonical_bytes(calendar),
        20,
        {
            "stable_identifier_fraction": 0.98,
            "membership_event_completeness": 0.98,
            "corporate_action_completeness": 0.98,
            "required_timestamp_fraction": 0.98,
            "price_observation_fraction": 0.98,
        },
    )
    assert row_count == len(sessions) + 1
    assert coverage["price_observation_fraction"] == 1.0


def test_complete_signed_offline_admission_starts_exactly_once(completion, tmp_path, monkeypatch):
    fixture = _valid_admission(completion, tmp_path, monkeypatch)
    admission = validate_rerun_activation(
        plan_path=fixture["plan_path"],
        admission_bundle_path=fixture["bundle_path"],
        admission_root=fixture["root"],
        admission_trust_store=fixture["admission_trust"],
        release_trust_store=fixture["release_trust"],
        workspace_root=fixture["workspace"],
    )
    assert admission.root_run_id == fixture["plan"]["next_run_id"]
    dataset = open_admitted_lseg_dataset(
        admission_root=fixture["root"],
        admission_bundle_path=fixture["bundle_path"],
        verified=admission,
    )
    with monkeypatch.context() as scan_patch:
        scan_patch.setattr(
            lseg_pit_adapter,
            "open_immutable_ref",
            lambda *args, **kwargs: pytest.fail(
                "batch scans must not repeat full-file SHA-256"
            ),
        )
        assert (
            sum(batch.num_rows for batch in dataset.iter_batches(batch_size=7))
            == dataset.row_count
        )
    assert dataset.descriptor()["scanner"] == "AdmittedLSEGDataset.iter_batches"
    ledger = tmp_path / "activations"
    value = guarded_activate_rerun(
        plan_path=fixture["plan_path"],
        admission_bundle_path=fixture["bundle_path"],
        admission_root=fixture["root"],
        admission_trust_store=fixture["admission_trust"],
        release_trust_store=fixture["release_trust"],
        workspace_root=fixture["workspace"],
        activation_ledger=ledger,
    )
    retry = guarded_activate_rerun(
        plan_path=fixture["plan_path"],
        admission_bundle_path=fixture["bundle_path"],
        admission_root=fixture["root"],
        admission_trust_store=fixture["admission_trust"],
        release_trust_store=fixture["release_trust"],
        workspace_root=fixture["workspace"],
        activation_ledger=ledger,
    )
    assert value == retry
    assert value["run_id"] == fixture["plan"]["next_run_id"]
    run_root = (
        fixture["workspace"] / "quant_lab" / "vault" / "30_experiments"
        / fixture["plan"]["next_run_id"]
    )
    assert len(list((run_root / ".control" / "events").glob("*.json"))) == 2
    archive = json.loads(next((run_root / ".control" / "admissions").glob("*.json")).read_text())
    replayed = verify_archived_data_admission(
        archive,
        admission_trust_store=fixture["admission_trust"],
        release_trust_store=fixture["release_trust"],
        vault_root=fixture["workspace"] / "quant_lab" / "vault",
    )
    assert replayed == admission
    forged_archive = copy.deepcopy(archive)
    forged_archive["entitlement_base64"] = base64.b64encode(b"{}").decode("ascii")
    with pytest.raises(DataAdmissionError):
        verify_archived_data_admission(
            forged_archive,
            admission_trust_store=fixture["admission_trust"],
            release_trust_store=fixture["release_trust"],
            vault_root=fixture["workspace"] / "quant_lab" / "vault",
        )


def test_lseg_execution_freeze_covers_control_plane_and_rejects_repository_drift(
    completion, tmp_path, monkeypatch
):
    required_control_plane_sources = {
        "quant_lab/src/quant_lab/research/control_plane.py",
        "quant_lab/src/quant_lab/research/_control_plane_contracts.py",
        "quant_lab/src/quant_lab/research/_control_plane_store.py",
        "quant_lab/src/quant_lab/research/_control_plane_trust.py",
    }
    assert required_control_plane_sources.issubset(LSEG_EXECUTION_SOURCES)

    fixture = _valid_admission(completion, tmp_path, monkeypatch)
    repository = fixture["workspace"] / "quant_lab"
    unexpected = repository / "outside-frozen-source-set.py"
    unexpected.write_text("raise RuntimeError('must never execute')\n", encoding="utf-8")
    with pytest.raises(CompletionError, match="untracked files"):
        validate_rerun_activation(
            plan_path=fixture["plan_path"],
            admission_bundle_path=fixture["bundle_path"],
            admission_root=fixture["root"],
            admission_trust_store=fixture["admission_trust"],
            release_trust_store=fixture["release_trust"],
            workspace_root=fixture["workspace"],
        )


def test_discovery_rejects_authentic_admission_with_reconstructed_mismatched_manifest(
    completion, tmp_path, monkeypatch
):
    fixture = _valid_admission(completion, tmp_path, monkeypatch)
    admission = validate_rerun_activation(
        plan_path=fixture["plan_path"],
        admission_bundle_path=fixture["bundle_path"],
        admission_root=fixture["root"],
        admission_trust_store=fixture["admission_trust"],
        release_trust_store=fixture["release_trust"],
        workspace_root=fixture["workspace"],
    )
    plan = fixture["plan"]
    forged_manifest = RunManifest.new(
        run_id=admission.root_run_id,
        parent_run_id=None,
        run_kind=RunKind.RESEARCH,
        hypothesis=plan["research_contract"]["hypothesis"],
        universe=json.dumps(
            plan["research_contract"]["universe"],
            sort_keys=True,
            separators=(",", ":"),
        ),
        data_manifest_id="forged-local-manifest-id",
        data_asof_utc=admission.data_asof_utc,
        feature_manifest_hash=admission.feature_manifest_hash,
        code_commit=admission.code_commit,
        target=plan["research_contract"]["target"],
        label_horizon=plan["research_contract"]["label_horizon"],
        split_manifest_id=admission.split_manifest_id,
    )
    vault = fixture["workspace"] / "quant_lab" / "vault"
    store = ExperimentRunStore(vault, admission.root_run_id)
    created = store.create(forged_manifest)
    archive = archive_verified_data_admission(
        admission_root=fixture["root"],
        bundle_path=fixture["bundle_path"],
        verified=admission,
    )
    archive_bytes = _canonical_bytes(archive)
    archive_sha256 = _sha(archive_bytes)
    archive_path = store.run_root / ".control" / "admissions" / f"{archive_sha256}.json"
    receipt = archive["verified_admission"]
    with store._locked():
        store._write_file(archive_path, archive_bytes, exclusive=True)
        store._append_event_locked(
            event_type="data_admission_verified",
            before_hash=created.content_hash(),
            after_hash=created.content_hash(),
            actor_type="controller",
            actor_id="malicious-local-controller",
            payload={
                "admission_uri": archive_path.relative_to(store.run_root).as_posix(),
                "admission_sha256": archive_sha256,
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
    configured = CompletionInputs(
        workspace_root=fixture["workspace"],
        output_root=tmp_path / "output",
        overlay_pointer=tmp_path / "overlay.json",
        source_findings_database=tmp_path / "findings.sqlite3",
        data_admission_trust_stores={
            admission.admission_trust_policy_sha256: fixture["admission_trust"]
        },
        data_release_trust_stores={
            admission.release_trust_policy_sha256: fixture["release_trust"]
        },
    )
    with pytest.raises(CompletionError, match="data-admission evidence differs"):
        _discover_control_plane_runs(configured, set())


def test_completed_activation_retry_replays_current_control_plane(
    completion, tmp_path, monkeypatch
):
    fixture = _valid_admission(completion, tmp_path, monkeypatch)
    ledger = tmp_path / "activations"
    value = guarded_activate_rerun(
        plan_path=fixture["plan_path"],
        admission_bundle_path=fixture["bundle_path"],
        admission_root=fixture["root"],
        admission_trust_store=fixture["admission_trust"],
        release_trust_store=fixture["release_trust"],
        workspace_root=fixture["workspace"],
        activation_ledger=ledger,
    )
    run_root = (
        fixture["workspace"] / "quant_lab" / "vault" / "30_experiments"
        / fixture["plan"]["next_run_id"]
    )
    event = next((run_root / ".control" / "events").glob("*.json"))
    event.chmod(0o600)
    event_payload = json.loads(event.read_text(encoding="utf-8"))
    event_payload["actor_id"] = "tampered-after-completed-receipt"
    event.write_text(json.dumps(event_payload), encoding="utf-8")
    event.chmod(0o400)
    with pytest.raises(Exception, match="event|hash|integrity|control-plane"):
        guarded_activate_rerun(
            plan_path=fixture["plan_path"],
            admission_bundle_path=fixture["bundle_path"],
            admission_root=fixture["root"],
            admission_trust_store=fixture["admission_trust"],
            release_trust_store=fixture["release_trust"],
            workspace_root=fixture["workspace"],
            activation_ledger=ledger,
        )
