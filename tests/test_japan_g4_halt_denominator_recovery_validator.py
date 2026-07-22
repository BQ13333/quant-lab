from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g4_halt_denominator_recovery_validator import (
    DELISTING_CHANNEL,
    HALT_CHANNEL,
    independently_parse_lifecycle_pages,
    validate_japan_g4_halt_denominator_recovery,
    verify_validation_artifact,
)


PAGES = ["""17 Changes in Listed Companies and Issues, Etc. (Jan. 2016)
Delisting
1st Section 2016.01.21 9870 Example
Suspension of Trading
1st Section 2016.01.27 7262 Example
21 Ex-New, Ex-Rights, Etc. (Jan. 2016)
2016.01.28 9999 Must Not Count
"""]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _producer_fixture(tmp_path: Path, halt_count: int = 1) -> Path:
    source = tmp_path / "unresolved.csv"
    original = pd.DataFrame([
        {"gap_id": "d", "gap_type": "source_denominator_uncovered", "scope": DELISTING_CHANNEL, "month": "2016-01"},
        {"gap_id": "h", "gap_type": "source_denominator_uncovered", "scope": HALT_CHANNEL, "month": "2016-01"},
        {"gap_id": "u", "gap_type": "source_denominator_uncovered", "scope": HALT_CHANNEL, "month": "2016-02"},
    ])
    original.to_csv(source, index=False)
    producer = tmp_path / "producer"
    raw = producer / "raw" / "lifecycle"
    raw.mkdir(parents=True)
    pdf = raw / "2016-01.pdf"
    pdf.write_bytes(b"%PDF fixture")
    original.to_csv(producer / "input_denominator.csv", index=False)
    url = "https://www.jpx.co.jp/reports/17_idou1601.pdf"
    pd.DataFrame([{
        "source_kind": "lifecycle_pdf", "source_key": "2016-01", "source_url": url,
        "sha256": _sha(pdf), "validation_status": "exact_month_valid", "local_path": str(pdf.resolve()),
    }]).to_csv(producer / "source_inventory.csv", index=False)
    pd.DataFrame([
        {"gap_id": "d", "month": "2016-01", "channel": DELISTING_CHANNEL, "covered": True, "event_rows": 1, "zero_event_asserted": False, "source_url": url, "source_sha256": _sha(pdf)},
        {"gap_id": "h", "month": "2016-01", "channel": HALT_CHANNEL, "covered": True, "event_rows": halt_count, "zero_event_asserted": False, "source_url": url, "source_sha256": _sha(pdf)},
        {"gap_id": "u", "month": "2016-02", "channel": HALT_CHANNEL, "covered": False, "event_rows": 0, "zero_event_asserted": False, "source_url": "", "source_sha256": ""},
    ]).to_csv(producer / "month_channel_coverage.csv", index=False)
    declared = {
        str(path.relative_to(producer)): _sha(path)
        for path in producer.rglob("*") if path.is_file()
    }
    (producer / "artifact_hashes.json").write_text(json.dumps(declared, sort_keys=True), encoding="utf-8")
    manifest = {
        "source_input": str(source.resolve()),
        "source_input_sha256": _sha(source),
        "artifact_hashes_sha256": _sha(producer / "artifact_hashes.json"),
    }
    (producer / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return producer


def test_independent_parser_stops_at_next_numbered_report() -> None:
    parsed = independently_parse_lifecycle_pages(PAGES, "2016-01")
    assert parsed["exact_month"]
    assert parsed["sections"][DELISTING_CHANNEL]["event_rows"] == 1
    assert parsed["sections"][HALT_CHANNEL]["event_rows"] == 1
    assert not parsed["sections"][HALT_CHANNEL]["explicit_zero"]


def test_validator_passes_partial_fixture_and_self_hashes(tmp_path: Path) -> None:
    producer = _producer_fixture(tmp_path)
    result = validate_japan_g4_halt_denominator_recovery(
        run_id="VALID",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
        require_expected_denominator=False,
    )
    assert result.manifest["status"] == "pass_partial_component"
    assert result.manifest["explicit_section_claims_proven"] == 2
    assert result.manifest["fully_reconciled_claims"] == 2
    assert result.manifest["scorecard_eligible"] is False
    assert verify_validation_artifact(result.artifact_dir) == []


def test_validator_fails_closed_on_count_mismatch_and_tampering(tmp_path: Path) -> None:
    producer = _producer_fixture(tmp_path, halt_count=2)
    result = validate_japan_g4_halt_denominator_recovery(
        run_id="INVALID",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
        require_expected_denominator=False,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "covered_claim_event_count_mismatch" in result.manifest["validation_issue_codes"]
    assert result.manifest["explicit_section_claims_proven"] == 2
    assert result.manifest["fully_reconciled_claims"] == 1
    (producer / "month_channel_coverage.csv").write_text("tampered\n", encoding="utf-8")
    tampered = validate_japan_g4_halt_denominator_recovery(
        run_id="TAMPERED",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
        require_expected_denominator=False,
    )
    assert "producer_output_hash_mismatch" in tampered.manifest["validation_issue_codes"]
