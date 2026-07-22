from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_lifecycle_taxonomy_repair import (
    TARGET_CLASS_BY_CODE,
    run_japan_g4_lifecycle_taxonomy_repair,
    verify_japan_g4_lifecycle_taxonomy_repair,
)
from quant_lab.data.japan_g4_lifecycle_taxonomy_repair_validator import (
    validate_japan_g4_lifecycle_taxonomy_repair,
    verify_japan_g4_lifecycle_taxonomy_repair_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V14_LIFECYCLE = (
    PROJECT_ROOT
    / "artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V14/inputs/"
    "tse_wayback_lifecycle/packets"
)
SOURCE_PACKETS = (
    V14_LIFECYCLE / "packet-0004" / "producer",
    V14_LIFECYCLE / "packet-0005" / "producer",
)
NOTICE_SOURCE = (
    PROJECT_ROOT
    / "artifacts/japan_g4_lifecycle_taxonomy_repair/"
    "JPN-G4-LIFECYCLE-TAXONOMY-REPAIR-V14-20260712-V1/raw/notices/20150302-11.html"
)

pytestmark = pytest.mark.skipif(
    not all(path.is_dir() for path in SOURCE_PACKETS) or not NOTICE_SOURCE.is_file(),
    reason="private JPX artifact integration tests",
)


def test_exact_nine_row_repair_and_independent_validation(tmp_path: Path) -> None:
    producer = run_japan_g4_lifecycle_taxonomy_repair(
        run_id="TAXONOMY-PRODUCER",
        source_packet_dirs=SOURCE_PACKETS,
        official_notice_path=NOTICE_SOURCE,
        artifact_root=tmp_path / "producer",
    )
    assert producer.manifest["repair_rows"] == 9
    assert producer.manifest["warrant_rows"] == 5
    assert producer.manifest["foreign_share_rows"] == 4
    assert producer.manifest["official_pdf_count"] == 7
    assert producer.manifest["official_notice_count"] == 1
    assert producer.manifest["mappings_created"] == 0
    assert producer.manifest["timestamps_created"] == 0
    assert producer.manifest["zero_events_created"] == 0
    assert verify_japan_g4_lifecycle_taxonomy_repair(producer.artifact_dir) == []

    repairs = pd.read_csv(producer.repairs_path, dtype=str).fillna("")
    assert set(repairs["raw_code"]) == set(TARGET_CLASS_BY_CODE)
    assert repairs["event_id"].nunique() == 9
    assert repairs["source_record_sha256"].nunique() == 9
    assert repairs.loc[repairs["raw_code"].isin({"77089", "45639", "76389", "35989", "38109"}), "required_source_marker"].eq("新株予約権").all()
    assert repairs.loc[repairs["raw_code"].isin({"8634", "9399", "5412"}), "required_source_marker"].eq("<外国株>").all()
    notice_row = repairs.loc[repairs["raw_code"].eq("3827")].iloc[0]
    assert notice_row["event_id"] == "765af2e977f955916788a27d"
    assert notice_row["required_source_marker"] == "市場区分:マザーズ(外国株)"
    assert notice_row["taxonomy_evidence_route"] == "separate_official_jpx_delisting_notice"
    for field in (
        "in_research_scope_after",
        "mapping_created",
        "timestamp_created",
        "zero_event_created",
        "classification_inferred",
    ):
        assert repairs[field].str.casefold().eq("false").all()

    validation = validate_japan_g4_lifecycle_taxonomy_repair(
        run_id="TAXONOMY-VALIDATION",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validation.manifest["status"] == "pass_bounded_source_backed_taxonomy_repair"
    assert validation.manifest["repair_rows_reconciled"] == 9
    assert validation.manifest["warrant_rows_reconciled"] == 5
    assert validation.manifest["foreign_share_rows_reconciled"] == 4
    assert validation.manifest["official_pdf_count_verified"] == 7
    assert validation.manifest["official_notice_count_verified"] == 1
    assert validation.manifest["source_record_hashes_recomputed"] is True
    assert validation.manifest["event_ids_recomputed"] is True
    assert verify_japan_g4_lifecycle_taxonomy_repair_validation(validation.artifact_dir) == []


def test_producer_is_write_once(tmp_path: Path) -> None:
    kwargs = {
        "run_id": "WRITE-ONCE",
        "source_packet_dirs": SOURCE_PACKETS,
        "official_notice_path": NOTICE_SOURCE,
        "artifact_root": tmp_path,
    }
    run_japan_g4_lifecycle_taxonomy_repair(**kwargs)
    with pytest.raises(FileExistsError):
        run_japan_g4_lifecycle_taxonomy_repair(**kwargs)


def test_producer_fails_closed_when_required_marker_is_removed(tmp_path: Path) -> None:
    packets = _minimal_packet_copies(tmp_path / "packets")
    csv_path = packets[1] / "exact_events.csv"
    events = pd.read_csv(csv_path, dtype=str).fillna("")
    events.loc[events["raw_code"].eq("8634"), "source_record_text"] = events.loc[
        events["raw_code"].eq("8634"), "source_record_text"
    ].str.replace("<外国株>", "", regex=False)
    events.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="required Japanese source marker missing"):
        run_japan_g4_lifecycle_taxonomy_repair(
            run_id="MISSING-MARKER",
            source_packet_dirs=packets,
            official_notice_path=NOTICE_SOURCE,
            artifact_root=tmp_path / "output",
        )


def test_independent_validator_rejects_semantic_tampering(tmp_path: Path) -> None:
    producer = run_japan_g4_lifecycle_taxonomy_repair(
        run_id="PRODUCER",
        source_packet_dirs=SOURCE_PACKETS,
        official_notice_path=NOTICE_SOURCE,
        artifact_root=tmp_path / "producer",
    )
    repairs_path = producer.artifact_dir / "taxonomy_repairs.csv"
    repairs = pd.read_csv(repairs_path, dtype=str).fillna("")
    repairs.loc[repairs["raw_code"].eq("5412"), "reclassified_instrument_class"] = "equity"
    repairs.to_csv(repairs_path, index=False)
    _rehash(producer.artifact_dir)

    validation = validate_japan_g4_lifecycle_taxonomy_repair(
        run_id="FAILED-VALIDATION",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validation.manifest["status"] == "failed_validation"
    assert "repair_exact_reconciliation_failed" in validation.manifest["validation_issue_codes"]


def test_independent_validator_rejects_notice_tampering(tmp_path: Path) -> None:
    producer = run_japan_g4_lifecycle_taxonomy_repair(
        run_id="NOTICE-PRODUCER",
        source_packet_dirs=SOURCE_PACKETS,
        official_notice_path=NOTICE_SOURCE,
        artifact_root=tmp_path / "producer",
    )
    notice_path = producer.artifact_dir / "raw" / "notices" / "20150302-11.html"
    body = notice_path.read_text(encoding="utf-8")
    assert "マザーズ（外国株）" in body
    notice_path.write_text(body.replace("マザーズ（外国株）", "マザーズ"), encoding="utf-8")
    _rehash(producer.artifact_dir)

    validation = validate_japan_g4_lifecycle_taxonomy_repair(
        run_id="NOTICE-FAILED-VALIDATION",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validation.manifest["status"] == "failed_validation"
    assert "official_notice_hash_mismatch" in validation.manifest["validation_issue_codes"]
    assert "official_notice_semantic_binding_failed" in validation.manifest["validation_issue_codes"]


def _minimal_packet_copies(root: Path) -> tuple[Path, Path]:
    copied: list[Path] = []
    for index, source in enumerate(SOURCE_PACKETS, start=1):
        target = root / f"packet-{index}"
        (target / "raw" / "pdfs").mkdir(parents=True)
        events = pd.read_csv(source / "exact_events.csv", dtype=str).fillna("")
        selected = events.loc[events["raw_code"].isin(TARGET_CLASS_BY_CODE)].copy()
        selected.to_csv(target / "exact_events.csv", index=False)
        for relative in selected["source_path"].unique():
            source_pdf = source / relative
            target_pdf = target / relative
            target_pdf.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_pdf, target_pdf)
        copied.append(target)
    return copied[0], copied[1]


def _rehash(root: Path) -> None:
    hashes_path = root / "artifact_hashes.json"
    inventory = json.loads(hashes_path.read_text(encoding="utf-8"))
    for relative in inventory:
        inventory[relative] = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    hashes_path.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes_sha256"] = hashlib.sha256(hashes_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
