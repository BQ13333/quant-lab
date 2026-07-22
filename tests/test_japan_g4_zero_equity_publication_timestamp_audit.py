from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_zero_equity_publication_timestamp_audit import (
    EXPECTED_PARENT_SHA256,
    EXPECTED_PDF_SHA256,
    produce_japan_g4_zero_equity_publication_timestamp_audit,
)
from quant_lab.data.japan_g4_zero_equity_publication_timestamp_audit_validator import (
    validate_japan_g4_zero_equity_publication_timestamp_audit,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
V16_ROOT = REPO_ROOT / "artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V16"

pytestmark = pytest.mark.skipif(
    not V16_ROOT.is_dir(),
    reason="private JPX artifact integration tests",
)


def _produce(tmp_path: Path):
    return produce_japan_g4_zero_equity_publication_timestamp_audit(
        run_id="PRODUCER-V1",
        v16_root=V16_ROOT,
        artifact_root=tmp_path / "producer",
    )


def test_real_v16_bytes_emit_a_durable_fail_closed_blocker(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    claim = pd.read_csv(result.artifact_dir / "claim_identity.csv", dtype=str, keep_default_na=False)
    candidates = pd.read_csv(
        result.artifact_dir / "timestamp_candidates.csv", dtype=str, keep_default_na=False
    )
    accepted = pd.read_csv(
        result.artifact_dir / "accepted_publication_timestamp.csv",
        dtype=str,
        keep_default_na=False,
    )
    blocker = pd.read_csv(result.blocker_path, dtype=str, keep_default_na=False)

    assert result.manifest["status"] == "blocked_exact_publication_timestamp_absent"
    assert len(claim) == 1
    assert claim.iloc[0]["plan_id"] == "2014-11-delisting_non_trading"
    assert claim.iloc[0]["source_sha256"] == EXPECTED_PDF_SHA256
    assert claim.iloc[0]["zero_events_inferred"] == "False"
    assert len(candidates) == 6
    assert candidates["accepted"].eq("False").all()
    assert candidates["known_at_utc"].eq("").all()
    assert accepted.empty
    assert blocker.iloc[0]["status"] == "blocked_exact_publication_timestamp_absent"
    assert blocker.iloc[0]["pit_complete"] == "False"


def test_retained_bytes_and_candidate_values_are_exact(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    retained = result.artifact_dir / "retained"
    pdf = retained / "2014-11-monthly-report.pdf"
    parent = retained / "2014-11-parent-index.html"
    candidates = pd.read_csv(
        result.artifact_dir / "timestamp_candidates.csv", dtype=str, keep_default_na=False
    ).set_index("candidate_type")

    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == EXPECTED_PDF_SHA256
    assert hashlib.sha256(parent.read_bytes()).hexdigest() == EXPECTED_PARENT_SHA256
    assert candidates.loc["pdf_creation_metadata", "candidate_value"] == "2014-12-19T09:13:32+09:00"
    assert candidates.loc["pdf_modification_metadata", "candidate_value"] == "2014-12-19T09:13:33+09:00"
    assert candidates.loc["parent_index_page_update_date", "candidate_value"] == "Feb. 23, 2015"
    assert candidates.loc["recurring_publication_schedule", "candidate_value"] == "21st"
    assert candidates.loc["archive_parent_index_capture", "candidate_value"] == "2015-03-10T16:09:42+00:00"
    assert candidates.loc["current_http_last_modified", "candidate_value"] == "Wed, 08 Jan 2025 05:15:17 GMT"


def test_independent_validator_confirms_blocker_without_promoting(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    _, validation = validate_japan_g4_zero_equity_publication_timestamp_audit(
        producer_dir=result.artifact_dir,
        validation_run_id="VALIDATION-V1",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "pass_blocker_confirmed"
    assert validation["blocker_confirmed"] is True
    assert validation["publication_timestamp_accepted"] is False
    assert validation["pit_complete"] is False
    assert validation["counts"]["issues"] == 0


def test_validator_fails_when_a_retained_source_is_tampered(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    pdf = result.artifact_dir / "retained/2014-11-monthly-report.pdf"
    pdf.write_bytes(pdf.read_bytes() + b"tampered")
    _, validation = validate_japan_g4_zero_equity_publication_timestamp_audit(
        producer_dir=result.artifact_dir,
        validation_run_id="VALIDATION-TAMPERED",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "fail"
    assert validation["blocker_confirmed"] is False
    assert validation["counts"]["issues"] > 0


def test_producer_rejects_v16_claim_hash_drift(tmp_path: Path) -> None:
    copied = tmp_path / "v16"
    shutil.copytree(V16_ROOT, copied)
    no_event = copied / "no_event_evidence.csv"
    frame = pd.read_csv(no_event, dtype=str, keep_default_na=False)
    frame.loc[frame["month"].eq("2014-11"), "source_sha256"] = "0" * 64
    frame.to_csv(no_event, index=False)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["output_sha256"]["no_event_evidence.csv"] = hashlib.sha256(no_event.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    try:
        produce_japan_g4_zero_equity_publication_timestamp_audit(
            run_id="DRIFTED",
            v16_root=copied,
            artifact_root=tmp_path / "producer",
        )
    except ValueError as exc:
        assert "claim source hash changed" in str(exc)
    else:
        raise AssertionError("producer accepted a drifted source hash")
