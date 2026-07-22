from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_halt_history import EVENT_COLUMNS
from quant_lab.data.japan_g4_halt_report_admission import normalize_halt_report_events
from quant_lab.data.japan_g4_halt_report_normalization_repair import (
    AGGREGATE_DISPOSITION,
    APPENDIX_DISPOSITION,
    HALT_NORMALIZATION_REPAIRS,
    run_halt_report_normalization_repair,
    verify_halt_report_normalization_repair,
)
from quant_lab.data.japan_g4_halt_report_normalization_repair_validator import (
    validate_halt_report_normalization_repair,
    verify_halt_report_normalization_repair_validation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = {
    "2012-02": PROJECT_ROOT
    / "artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V8/inputs/"
    "tse_wayback_lifecycle/packets/packet-0001/producer/raw/pdfs/2012-02.pdf",
    "2013-09": PROJECT_ROOT
    / "artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V8/inputs/"
    "tse_wayback_lifecycle/packets/packet-0002/producer/raw/pdfs/2013-09.pdf",
    "2020-10": PROJECT_ROOT
    / "artifacts/japan_g4_halt_report_events/JPN-G4-HALT-REPORT-EVENTS-20260712-V2/"
    "raw/lifecycle/2020-10.pdf",
}

pytestmark = pytest.mark.skipif(
    not all(path.is_file() for path in SOURCE_PATHS.values()),
    reason="private JPX artifact integration tests",
)


def _source_rows() -> pd.DataFrame:
    rows = []
    for repair in HALT_NORMALIZATION_REPAIRS:
        row = {
            "event_id": repair.report_event_id,
            "report_month": repair.report_month,
            "channel": "halt_suspension_resumption",
            "event_type": "trading_halt_report_row",
            "event_date": repair.event_date,
            "raw_code": repair.raw_code,
            "normalized_local_code": repair.normalized_local_code,
            "instrument_class": repair.instrument_class,
            "issue_name": repair.issue_name,
            "halt_start_printed": repair.halt_start_printed,
            "halt_end_printed": repair.halt_end_printed,
            "halt_start_jst": (
                f"{repair.event_date}T{repair.halt_start_printed}:00+09:00"
                if repair.halt_start_printed
                else ""
            ),
            "halt_end_jst": (
                f"{repair.event_date}T{repair.halt_end_printed}:00+09:00"
                if repair.halt_end_printed
                else ""
            ),
            "source_url": "https://www.jpx.co.jp/retained.pdf",
            "source_path": f"raw/{repair.report_month}.pdf",
            "source_sha256": repair.source_sha256,
            "source_page": repair.source_page,
            "source_row_ordinal": repair.source_row_ordinal,
            "source_record_text": repair.source_record_text,
            "source_record_sha256": repair.source_record_sha256,
            "known_at": "",
            "ric": "",
            "mapping_status": "pending",
            "timestamp_status": "pending",
            "admissible_to_shared_halt_history": False,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_exact_repairs_create_one_event_and_preserve_nine_evidence_rows() -> None:
    events, admission = normalize_halt_report_events(
        _source_rows(), event_columns=EVENT_COLUMNS, apply_bounded_repairs=True
    )

    assert events["local_code"].tolist() == ["4585"]
    event = events.iloc[0]
    assert event["event_start_jst"] == "2013-09-30T17:10:00+09:00"
    assert event["interval_end_jst"] == "2013-09-30T17:25:00+09:00"
    assert "sealed-byte wrapped-time" in event["reason"]

    statuses = admission["normalization_status"].value_counts().to_dict()
    assert statuses == {
        AGGREGATE_DISPOSITION: 6,
        APPENDIX_DISPOSITION: 3,
        "normalized": 1,
    }
    evidence = admission.loc[admission["normalization_status"].ne("normalized")]
    assert not evidence["in_research_scope"].any()
    assert evidence["local_code"].eq("").all()
    assert evidence["primary_event_id"].eq("").all()


def test_exact_repair_fails_closed_on_semantic_tampering() -> None:
    rows = _source_rows().iloc[[3]].copy()
    rows.loc[:, "issue_name"] = "tampered"
    with pytest.raises(ValueError, match="source-row contract mismatch"):
        normalize_halt_report_events(
            rows, event_columns=EVENT_COLUMNS, apply_bounded_repairs=True
        )


def test_write_once_repair_and_independent_validation_artifacts(tmp_path: Path) -> None:
    producer = run_halt_report_normalization_repair(
        run_id="REPAIR",
        source_paths=SOURCE_PATHS,
        artifact_root=tmp_path / "producer",
    )
    assert producer.manifest["repair_rows"] == 10
    assert producer.manifest["instrument_events_created"] == 1
    assert producer.manifest["evidence_only_rows"] == 9
    assert producer.manifest["unchanged_date_granular_rows"] == 11
    assert verify_halt_report_normalization_repair(producer.artifact_dir) == []
    with pytest.raises(FileExistsError):
        run_halt_report_normalization_repair(
            run_id="REPAIR",
            source_paths=SOURCE_PATHS,
            artifact_root=tmp_path / "producer",
        )

    validation = validate_halt_report_normalization_repair(
        run_id="VALIDATION",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validation.manifest["status"] == "pass_bounded_normalization_repair"
    assert validation.manifest["repair_rows_reconciled"] == 10
    assert validation.manifest["source_hashes_verified"] is True
    assert validation.manifest["source_semantics_verified"] is True
    assert validation.manifest["source_record_hashes_recomputed"] is True
    assert validation.manifest["unchanged_date_granular_rows"] == 11
    assert verify_halt_report_normalization_repair_validation(validation.artifact_dir) == []
