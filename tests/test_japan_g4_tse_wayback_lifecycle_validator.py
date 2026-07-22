from __future__ import annotations

import base64
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

import quant_lab.data.japan_g4_tse_wayback_lifecycle as lifecycle
import quant_lab.data.japan_g4_tse_wayback_lifecycle_validator as lifecycle_validator
from quant_lab.data.japan_g4_tse_wayback_lifecycle import (
    FetchResponse,
    run_japan_g4_tse_wayback_lifecycle,
)
from quant_lab.data.japan_g4_tse_wayback_lifecycle_validator import (
    independently_parse_pdf_pages,
    validate_japan_g4_tse_wayback_lifecycle,
    verify_wayback_lifecycle_validation,
)
from test_japan_g4_tse_wayback_lifecycle import (
    COPIED_MONTH,
    COPIED_PAGES,
    copied_fixture_transport,
    HTML,
    MONTH,
    PAGES,
    PDF,
    fixture_transport,
)


def _cdx_digest(body: bytes) -> str:
    return base64.b32encode(hashlib.sha1(body).digest()).decode("ascii").rstrip("=")


def secure_fixture_transport(url: str) -> FetchResponse:
    response = fixture_transport(url)
    if "/cdx/search/cdx" not in url:
        return response
    payload = json.loads(response.body)
    if len(payload) > 1:
        original = payload[1][1]
        payload[1][3] = _cdx_digest(HTML if original.endswith(".html") else PDF)
    return FetchResponse(
        response.status,
        json.dumps(payload).encode(),
        response.headers,
        response.final_url,
    )


def _producer(tmp_path: Path) -> Path:
    result = run_japan_g4_tse_wayback_lifecycle(
        run_id="PRODUCER",
        months=[MONTH],
        artifact_root=tmp_path,
        max_months=1,
        sleep_seconds=0,
        retries=0,
        transport=secure_fixture_transport,
        page_extractor=lambda body: PAGES,
    )
    return result.artifact_dir


def _copied_producer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(lifecycle, "COPIED_FILE_ROUTE_ENABLED", True)
    result = run_japan_g4_tse_wayback_lifecycle(
        run_id="COPIED-PRODUCER",
        months=[COPIED_MONTH],
        artifact_root=tmp_path / "producer",
        max_months=1,
        sleep_seconds=0,
        retries=0,
        transport=copied_fixture_transport,
        page_extractor=lambda body: COPIED_PAGES,
    )
    return result.artifact_dir


def _rehash_packet(producer: Path) -> None:
    hashes_path = producer / "artifact_hashes.json"
    declared = json.loads(hashes_path.read_text(encoding="utf-8"))
    for relative in declared:
        declared[relative] = hashlib.sha256((producer / relative).read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(declared, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = producer / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes_sha256"] = hashlib.sha256(hashes_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mutate_event_everywhere(producer: Path, field: str, value: object) -> None:
    csv_path = producer / "exact_events.csv"
    parquet_path = producer / "exact_events.parquet"
    csv_events = pd.read_csv(csv_path, dtype=str).fillna("")
    parquet_events = pd.read_parquet(parquet_path)
    csv_events.loc[0, field] = value
    parquet_events.loc[0, field] = value
    csv_events.to_csv(csv_path, index=False)
    parquet_events.to_parquet(parquet_path, index=False)
    checkpoint_path = producer / "acquisition_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["months"][MONTH]["events"][0][field] = value
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _rehash_packet(producer)


def test_independent_parser_produces_exact_nonempty_sets() -> None:
    parsed = independently_parse_pdf_pages(PAGES, MONTH)
    assert parsed["exact_month"]
    assert len(parsed["events"]) == 2
    assert {row["raw_code"] for row in parsed["events"]} == {"2738", "8809"}
    halt = next(row for row in parsed["events"] if row["channel"] == "halt_suspension_resumption")
    assert halt["instrument_class"] == "equity"
    assert halt["normalized_local_code"] == "8809"
    assert halt["halt_start_printed"] == "16:00"
    assert halt["halt_end_printed"] == "16:25"
    assert halt["halt_start_jst"] == "2012-01-19T16:00:00+09:00"
    assert halt["halt_end_jst"] == "2012-01-19T16:25:00+09:00"
    assert halt["source_page"] == 1
    assert halt["source_row_ordinal"] == 1
    assert len(halt["source_record_sha256"]) == 64
    assert len(halt["event_id"]) == 24
    assert parsed["required_sections_valid"]
    assert all(not section["zero_inferred"] for section in parsed["sections"].values())


def test_independent_parser_rejects_out_of_month_rows() -> None:
    pages = [PAGES[0].replace("2012.01.19 8809", "2012.02.19 8809")]
    with pytest.raises(ValueError, match="out-of-month row"):
        independently_parse_pdf_pages(pages, MONTH)


def test_independent_parser_marks_duplicate_required_section_invalid() -> None:
    pages = [PAGES[0].replace("Suspension of Trading", "Suspension of Trading\nSuspension of Trading")]
    parsed = independently_parse_pdf_pages(pages, MONTH)
    assert not parsed["required_sections_valid"]
    assert parsed["sections"]["halt_suspension_resumption"]["heading_count"] == 2
    assert parsed["sections"]["halt_suspension_resumption"]["event_count"] == 0


def test_independent_parser_recognizes_jgb_futures_without_equity_promotion() -> None:
    pages = [
        """
        Changes in Listed Companies and Issues (April 2013)
        Delisting of Stocks
        市場第一部 2013.04.01 1234 TEST COMPANY
        Suspension of Trading
        国債先物 2013.04.05 長期国債先物取引 全限月 13:08～13:18:44
        """
    ]

    parsed = independently_parse_pdf_pages(pages, "2013-04")

    future = next(
        row
        for row in parsed["events"]
        if row["channel"] == "halt_suspension_resumption"
    )
    assert future["instrument_class"] == "jgb_future"
    assert future["instrument_class"] != "equity"
    assert future["ric"] == ""
    assert future["known_at"] == ""
    assert future["admissible_to_shared_halt_history"] is False


def test_validator_reconciles_bytes_cdx_urls_and_event_sets(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="VALID",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "pass_bounded_exact_nonempty_evidence"
    assert result.manifest["capture_valid_month_count"] == 1
    assert result.manifest["independent_exact_event_count"] == 2
    assert result.manifest["reconciled_nonempty_claim_count"] == 2
    assert result.manifest["absence_is_zero"] is False
    reconciliation = pd.read_csv(result.artifact_dir / "event_set_reconciliation.csv")
    assert reconciliation["validated"].astype(bool).all()
    assert verify_wayback_lifecycle_validation(result.artifact_dir) == []


def test_validator_independently_admits_copied_file_transport_and_non_equity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert lifecycle_validator.COPIED_FILE_ROUTE_ENABLED is True
    producer = _copied_producer(tmp_path, monkeypatch)
    monkeypatch.setattr(lifecycle_validator, "COPIED_FILE_ROUTE_ENABLED", True)
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="COPIED-VALIDATION",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: COPIED_PAGES,
    )
    assert result.manifest["status"] == "pass_bounded_exact_nonempty_evidence"
    assert result.manifest["capture_valid_month_count"] == 1
    assert result.manifest["independent_exact_event_count"] == 3
    assert result.manifest["producer_nonempty_claim_count"] == 2
    assert result.manifest["reconciled_nonempty_claim_count"] == 2
    assert result.manifest["live_official_pdf_response_independently_validated"] is True
    assert result.manifest["archived_parent_lineage_strengths"] == [
        lifecycle_validator.DIRECT_ARCHIVED_PDF_LINEAGE
    ]
    events = pd.read_csv(result.artifact_dir / "independent_exact_events.csv", dtype=str)
    assert events["instrument_class"].eq("jgb_future").sum() == 1
    assert verify_wayback_lifecycle_validation(result.artifact_dir) == []


def test_validator_fails_closed_on_event_disagreement(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="DISAGREE",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: [PAGES[0].replace("8809", "8810")],
    )
    assert result.manifest["status"] == "failed_validation"
    assert "global_exact_event_set_disagreement" in result.manifest["validation_issue_codes"]
    assert "channel_exact_event_set_disagreement" in result.manifest["validation_issue_codes"]


def test_validator_rejects_unsafe_hash_inventory_path(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    hashes_path = producer / "artifact_hashes.json"
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    hashes["../escape"] = "0" * 64
    hashes_path.write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")
    manifest_path = producer / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes_sha256"] = hashlib.sha256(hashes_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="UNSAFE",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "unsafe_producer_artifact_path" in result.manifest["validation_issue_codes"]


def test_validator_fails_closed_on_missing_pdf_capture(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    pdf_path = producer / "raw" / "pdfs" / f"{MONTH}.pdf"
    pdf_path.unlink()
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="MISSING",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "missing_capture_bytes" in result.manifest["validation_issue_codes"]
    assert result.manifest["absence_is_zero"] is False


def test_validator_fails_closed_on_missing_required_section(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    pages = [PAGES[0].replace("Suspension of Trading", "Trading Suspensions")]
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="SECTION",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: pages,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "required_section_cardinality_invalid" in result.manifest["validation_issue_codes"]
    assert result.manifest["reconciled_nonempty_claim_count"] == 0


def test_validator_rejects_rehashed_full_event_semantic_tamper(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    _mutate_event_everywhere(producer, "instrument_class", "etf")
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="EVENT-TAMPER",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "source_record_sha256_recomputation_failed" in result.manifest["validation_issue_codes"]
    assert "global_exact_event_set_disagreement" in result.manifest["validation_issue_codes"]


def test_validator_rejects_rehashed_pit_control_tamper(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    _mutate_event_everywhere(producer, "known_at", "2012-04-04T08:24:07Z")
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="PIT-TAMPER",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "producer_event_control_invalid" in result.manifest["validation_issue_codes"]


def test_validator_rejects_csv_parquet_semantic_disagreement(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    parquet_path = producer / "exact_events.parquet"
    events = pd.read_parquet(parquet_path)
    events.loc[0, "halt_end_printed"] = "23:59"
    events.to_parquet(parquet_path, index=False)
    _rehash_packet(producer)
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="FORMAT-TAMPER",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "exact_events_csv_parquet_semantic_mismatch" in result.manifest["validation_issue_codes"]


def test_validator_rejects_manifest_and_checkpoint_count_tamper(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    manifest_path = producer / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["exact_event_count"] = 99
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    checkpoint_path = producer / "acquisition_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    checkpoint["months"][MONTH]["events"].pop()
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_packet(producer)
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="COUNT-TAMPER",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "manifest_count_reconciliation_failed" in result.manifest["validation_issue_codes"]
    assert "checkpoint_table_reconciliation_failed" in result.manifest["validation_issue_codes"]


def test_validator_rejects_rehashed_transport_binding_tamper(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    transport_path = producer / "transport_inventory.csv"
    transport = pd.read_csv(transport_path, dtype=str).fillna("")
    index = transport.index[transport["request_kind"].eq("pdf_capture")][0]
    transport.loc[index, "sha256"] = "0" * 64
    transport.to_csv(transport_path, index=False)
    checkpoint_path = producer / "acquisition_checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    row = next(
        item
        for item in checkpoint["months"][MONTH]["transport"]
        if item["request_kind"] == "pdf_capture"
    )
    row["sha256"] = "0" * 64
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _rehash_packet(producer)
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="TRANSPORT-TAMPER",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "transport_row_binding_invalid" in result.manifest["validation_issue_codes"]


def test_validator_rejects_symlinked_declared_raw_path(tmp_path: Path) -> None:
    producer = _producer(tmp_path / "producer")
    pdf_path = producer / "raw" / "pdfs" / f"{MONTH}.pdf"
    outside = tmp_path / "outside.pdf"
    shutil.copyfile(pdf_path, outside)
    pdf_path.unlink()
    pdf_path.symlink_to(outside)
    _rehash_packet(producer)
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="SYMLINK",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
        page_extractor=lambda body: PAGES,
    )
    assert result.manifest["status"] == "failed_validation"
    assert "producer_artifact_symlink_or_escape" in result.manifest["validation_issue_codes"]
    assert "missing_capture_bytes" in result.manifest["validation_issue_codes"]


def test_validator_accepts_retained_real_packet(tmp_path: Path) -> None:
    packet = (
        Path(__file__).resolve().parents[1]
        / "artifacts"
        / "japan_g4_tse_wayback_lifecycle"
        / "JPN-G4-TSE-WAYBACK-LIFECYCLE-2012-PILOT-20260712-V2"
    )
    if not packet.is_dir():
        pytest.skip("retained institutional packet unavailable")
    result = validate_japan_g4_tse_wayback_lifecycle(
        run_id="REAL-PACKET",
        producer_dir=packet,
        artifact_root=tmp_path / "validation",
    )
    assert result.manifest["status"] == "pass_bounded_exact_nonempty_evidence"
    assert result.manifest["independent_exact_event_count"] == 240
    assert result.manifest["reconciled_nonempty_claim_count"] == 24
