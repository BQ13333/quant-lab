from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.research.trial_ledger import TrialLedger


HASHES = {
    "code_hash": "a" * 64,
    "data_hash": "b" * 64,
    "config_hash": "c" * 64,
    "prompt_hash": "d" * 64,
    "model_hash": "e" * 64,
}


def ledger(tmp_path):
    return TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")


def reserve(book, key="key", parent=None):
    return book.reserve_trial(
        idempotency_key=key,
        campaign="malaysia",
        family="momentum",
        hypothesis_id="H1",
        candidate_id=f"C-{key}",
        parent_trial_id=parent,
        **HASHES,
    )


def test_reserve_outcome_correction_and_lineage(tmp_path):
    book = ledger(tmp_path)
    parent = reserve(book, "parent")
    child = reserve(book, "child", parent.trial_id)
    evidence = book.store_evidence(b'{"metric":"net_oos_ir"}', extension=".json")
    event = book.record_outcome(
        child.trial_id,
        status="failure",
        failure_class="COST_SENSITIVITY",
        payload={"failure_code": "COST_SENSITIVITY"},
        evidence_sha256=evidence.sha256,
    )
    correction = book.record_correction(
        child.trial_id, corrects_event_id=event, payload={"reason_code": "METRIC_RECALCULATED"}
    )
    assert correction > event
    assert book.lineage(child.trial_id) == [parent.trial_id, child.trial_id]
    assert [row["event_type"] for row in book.replay_manifest(child.trial_id)["events"]] == [
        "reserved",
        "outcome",
        "correction",
    ]


def test_cannot_evaluate_unreserved_or_overwrite_terminal_state(tmp_path):
    book = ledger(tmp_path)
    with pytest.raises(KeyError, match="unreserved"):
        book.record_outcome("missing", status="failure")
    trial = reserve(book)
    book.record_outcome(trial.trial_id, status="success")
    with pytest.raises(RuntimeError, match="terminal outcome"):
        book.record_outcome(trial.trial_id, status="failure")


def test_every_terminal_class_counts_as_lifetime_attempt(tmp_path):
    book = ledger(tmp_path)
    for status in ("success", "failure", "rejected", "duplicate", "timeout"):
        trial = reserve(book, status)
        book.record_outcome(trial.trial_id, status=status, failure_class=status.upper())
    counts = book.lifetime_counts(campaign="malaysia")
    assert counts["reserved"] == 5
    assert counts["lifetime_attempts"] == 5
    assert all(counts[status] == 1 for status in ("success", "failure", "rejected", "duplicate", "timeout"))


def test_idempotent_and_concurrent_reservation(tmp_path):
    book = ledger(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as executor:
        same = list(executor.map(lambda _: reserve(book, "same"), range(20)))
    assert sum(item.created for item in same) == 1
    assert len({item.trial_id for item in same}) == 1
    with ThreadPoolExecutor(max_workers=8) as executor:
        distinct = list(executor.map(lambda index: reserve(book, f"distinct-{index}"), range(20)))
    assert all(item.created for item in distinct)
    assert book.lifetime_counts(campaign="malaysia")["reserved"] == 21


def test_atomic_cohort_reservation_is_all_or_nothing_and_idempotent(tmp_path):
    book = ledger(tmp_path)
    parent = reserve(book, "atomic-parent")
    requests = [
        {
            "idempotency_key": f"atomic-{index}",
            "campaign": "malaysia",
            "family": "v2_pair",
            "hypothesis_id": f"H-{index}",
            "candidate_id": f"C-{index}",
            "parent_trial_id": parent.trial_id,
            **HASHES,
        }
        for index in range(6)
    ]
    created = book.reserve_trials_atomically(requests)
    replayed = book.reserve_trials_atomically(requests)
    assert len(created) == len(replayed) == 6
    assert all(item.created for item in created)
    assert not any(item.created for item in replayed)
    assert [item.trial_id for item in replayed] == [item.trial_id for item in created]
    assert book.lifetime_counts(campaign="malaysia")["reserved"] == 7

    drifted = [dict(item) for item in requests]
    drifted[0]["candidate_id"] = "C-DRIFTED"
    with pytest.raises(RuntimeError, match="identity drifted"):
        book.reserve_trials_atomically(drifted)


def test_atomic_cohort_reservation_rolls_back_when_one_parent_is_missing(tmp_path):
    book = ledger(tmp_path)
    requests = [
        {
            "idempotency_key": f"rollback-{index}",
            "campaign": "malaysia",
            "family": "v2_pair",
            "hypothesis_id": f"H-{index}",
            "candidate_id": f"C-{index}",
            "parent_trial_id": "missing-parent" if index == 5 else None,
            **HASHES,
        }
        for index in range(6)
    ]
    with pytest.raises(ValueError, match="every atomic reservation parent"):
        book.reserve_trials_atomically(requests)
    assert book.lifetime_counts(campaign="malaysia")["reserved"] == 0


def test_evidence_missing_or_corrupt_fails_closed(tmp_path):
    book = ledger(tmp_path)
    evidence = book.store_evidence(b"immutable", extension=".txt")
    assert book.verify_evidence(evidence.sha256).size_bytes == 9
    evidence.path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="integrity"):
        book.verify_evidence(evidence.sha256)
    evidence.path.unlink()
    with pytest.raises(FileNotFoundError, match="missing evidence"):
        book.verify_evidence(evidence.sha256)
    with pytest.raises(FileNotFoundError, match="unknown evidence"):
        book.verify_evidence("f" * 64)


def test_sensitive_payloads_and_long_text_are_rejected(tmp_path):
    book = ledger(tmp_path)
    trial = reserve(book)
    for payload in ({"prompt": "private"}, {"api_key": "secret"}, {"notes": "x" * 513}):
        with pytest.raises(ValueError):
            book.record_outcome(trial.trial_id, status="failure", payload=payload)


def test_legacy_malaysia_sweep_is_idempotent_and_in_lifetime_denominator(tmp_path):
    manifest = tmp_path / "sweep_manifest.json"
    manifest.write_text(
        json.dumps({"run_id": "MALAYSIA-SWEEP-20260705-085432", "candidate_count": 3330}),
        encoding="utf-8",
    )
    book = ledger(tmp_path)
    assert book.import_legacy_malaysia_sweep(manifest) is True
    assert book.import_legacy_malaysia_sweep(manifest) is False
    current = reserve(book, "new")
    book.record_outcome(current.trial_id, status="failure", failure_class="NO_EDGE")
    counts = book.lifetime_counts(campaign="malaysia")
    assert counts == {
        "duplicate": 0,
        "failure": 1,
        "rejected": 0,
        "success": 0,
        "timeout": 0,
        "reserved": 1,
        "legacy": 3330,
        "lifetime_attempts": 3331,
    }


@pytest.mark.skipif(
    not Path(
        "artifacts/malaysia/MALAYSIA-SWEEP-20260705-085432/sweep_manifest.json"
    ).is_file(),
    reason="private historical artifact integration test",
)
def test_real_legacy_manifest_has_at_least_3330_trials(tmp_path):
    path = "artifacts/malaysia/MALAYSIA-SWEEP-20260705-085432/sweep_manifest.json"
    book = ledger(tmp_path)
    assert book.import_legacy_malaysia_sweep(path)
    assert book.lifetime_counts(campaign="malaysia")["legacy"] >= 3330


def test_completed_campaign_ledgers_merge_idempotently_into_lifetime_vault(tmp_path):
    source = ledger(tmp_path / "source")
    parent = reserve(source, "source-parent")
    child = reserve(source, "source-child", parent.trial_id)
    evidence = source.store_evidence(b'{"audit":"immutable"}', extension=".json")
    source.record_outcome(parent.trial_id, status="rejected", failure_class="REVIEW")
    outcome = source.record_outcome(
        child.trial_id,
        status="failure",
        failure_class="AUDIT",
        evidence_sha256=evidence.sha256,
    )
    source.record_correction(child.trial_id, corrects_event_id=outcome, payload={"reason_code": "AUDITED"})

    vault = ledger(tmp_path / "vault")
    assert vault.import_completed_ledger(
        source.db_path, source.evidence_root, source_key="MALAYSIA-CAMPAIGN-001"
    )
    assert not vault.import_completed_ledger(
        source.db_path, source.evidence_root, source_key="MALAYSIA-CAMPAIGN-001"
    )
    assert vault.lifetime_counts(campaign="malaysia")["lifetime_attempts"] == 2
    assert vault.lineage(child.trial_id) == [parent.trial_id, child.trial_id]
    assert vault.verify_evidence(evidence.sha256).path.read_bytes() == b'{"audit":"immutable"}'
    imported_events = vault.replay_manifest(child.trial_id)["events"]
    imported_outcome = next(item for item in imported_events if item["event_type"] == "outcome")
    imported_correction = next(item for item in imported_events if item["event_type"] == "correction")
    assert imported_correction["corrects_event_id"] == imported_outcome["event_id"]


def test_active_campaign_ledger_cannot_enter_lifetime_vault(tmp_path):
    source = ledger(tmp_path / "source")
    reserve(source, "still-active")
    vault = ledger(tmp_path / "vault")
    with pytest.raises(ValueError, match="every trial is terminal"):
        vault.import_completed_ledger(
            source.db_path, source.evidence_root, source_key="MALAYSIA-ACTIVE-001"
        )


def test_candidate_return_series_persist_idempotently_and_rebuild_lifetime_matrix(tmp_path):
    book = ledger(tmp_path)
    first = reserve(book, "returns-one")
    second = reserve(book, "returns-two")
    index = pd.bdate_range("2025-01-02", periods=20)
    one = pd.Series([0.001] * 20, index=index)
    two = pd.Series([0.002] * 20, index=index)
    receipt = book.record_candidate_returns(
        first.trial_id, candidate_id="MYS-C-ONE", family="momentum", returns=one
    )
    assert book.record_candidate_returns(
        first.trial_id, candidate_id="MYS-C-ONE", family="momentum", returns=one
    ).sha256 == receipt.sha256
    book.record_candidate_returns(
        second.trial_id, candidate_id="MYS-C-TWO", family="low_volatility", returns=two
    )
    matrix = book.candidate_return_matrix()
    assert matrix.shape == (20, 2)
    assert tuple(matrix.columns) == (
        f"MYS-C-ONE::{first.trial_id}",
        f"MYS-C-TWO::{second.trial_id}",
    )
    assert sum(
        item["event_type"] == "candidate_returns"
        for item in book.replay_manifest(first.trial_id)["events"]
    ) == 1
    with pytest.raises(RuntimeError, match="changed"):
        book.record_candidate_returns(
            first.trial_id,
            candidate_id="MYS-C-ONE",
            family="momentum",
            returns=one * 2.0,
        )


def test_invalid_candidate_returns_are_corrected_not_deleted(tmp_path):
    book = ledger(tmp_path)
    trial = reserve(book, "invalid-returns")
    index = pd.bdate_range("2025-01-02", periods=20)
    book.record_candidate_returns(
        trial.trial_id,
        candidate_id="MYS-C-INVALID",
        family="residual_momentum",
        returns=pd.Series(0.0, index=index),
    )
    correction = book.invalidate_candidate_returns(
        trial.trial_id, reason_code="INVALID_ALL_NAN_SIGNAL_IMPLEMENTATION"
    )
    assert book.candidate_return_matrix().empty
    assert book.invalidate_candidate_returns(
        trial.trial_id, reason_code="INVALID_ALL_NAN_SIGNAL_IMPLEMENTATION"
    ) == correction
    events = book.replay_manifest(trial.trial_id)["events"]
    return_event = next(item for item in events if item["event_type"] == "candidate_returns")
    correction_event = next(item for item in events if item["event_type"] == "correction")
    assert correction_event["corrects_event_id"] == return_event["event_id"]
