from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_halt_report_events import (
    DELISTING_CHANNEL,
    HALT_CHANNEL,
    parse_report_event_rows,
    run_japan_g4_halt_report_events,
    verify_report_event_artifact,
    _verify_packet,
)
from quant_lab.data.japan_g4_halt_report_events_validator import (
    _audit_packet,
    independently_reparse_report_rows,
    validate_japan_g4_halt_report_events,
    verify_report_event_validation_artifact,
)


PAGES = [
    """17 Changes in Listed Companies and Issues, Etc. (Mar. 2023)
Delisting
Prime 2023.03.31 1234 Example Equity EXAMPLE EQUITY LTD.
Suspension of Trading
Prime 2023.03.20 5678 Halted Equity HALTED EQUITY LTD. 15:01～15:45
指数先物オプション 2023.03.20 全銘柄 16:30～19:10
Index Options 2023.03.21 TOPIX Options All Contracts 15:00
普通債 2023.03.20 00450799 Credit Suisse Group AG Issue 12:30～2023/3/22 12:29
Straight Bonds
21 Ex-New, Ex-Rights, Etc. (Mar. 2023)
"""
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_packet(root: Path, manifest: dict[str, object]) -> None:
    declared = {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "artifact_hashes.json"}
    }
    hashes = root / "artifact_hashes.json"
    hashes.write_text(json.dumps(declared, sort_keys=True), encoding="utf-8")
    manifest["artifact_hashes_sha256"] = _sha(hashes)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _packets(tmp_path: Path) -> tuple[Path, Path]:
    denominator = tmp_path / "denominator"
    raw = denominator / "raw" / "lifecycle"
    raw.mkdir(parents=True)
    months = [str(period) for period in pd.period_range("2010-01", periods=155, freq="M")]
    claims: list[dict[str, object]] = []
    coverage: list[dict[str, object]] = []
    denominator_rows: list[dict[str, object]] = []
    inventory: list[dict[str, object]] = []

    for index, month in enumerate(months):
        pdf = raw / f"{month}.pdf"
        pdf.write_bytes(f"%PDF fixture {month}".encode())
        digest = _sha(pdf)
        inventory.append(
            {
                "source_kind": "lifecycle_pdf",
                "source_key": month,
                "source_url": f"https://www.jpx.co.jp/reports/{month}.pdf",
                "http_status": 200,
                "local_path": str(pdf.resolve()),
                "sha256": digest,
                "validation_status": "exact_month_valid",
                "discovered_links": 0,
            }
        )
        channels = [HALT_CHANNEL]
        if index < 39:
            channels.append(DELISTING_CHANNEL)
        for channel in channels:
            gap_id = f"covered-{index:03d}-{channel}"
            event_rows = 4 if channel == HALT_CHANNEL else 1
            denominator_rows.append(
                {
                    "gap_id": gap_id,
                    "gap_type": "source_denominator_uncovered",
                    "scope": channel,
                    "month": month,
                }
            )
            coverage.append(
                {
                    "gap_id": gap_id,
                    "month": month,
                    "channel": channel,
                    "covered": True,
                    "event_rows": event_rows,
                    "zero_event_asserted": False,
                    "source_url": f"https://www.jpx.co.jp/reports/{month}.pdf",
                    "source_sha256": digest,
                }
            )
            claims.append(
                {
                    "gap_id": gap_id,
                    "month": month,
                    "channel": channel,
                    "producer_event_rows": event_rows,
                    "independent_event_rows": event_rows,
                    "producer_explicit_zero": False,
                    "independent_explicit_zero": False,
                    "exact_month": True,
                    "explicit_section_proven": True,
                    "source_identity_match": True,
                    "event_row_count_match": True,
                    "zero_assertion_match": True,
                    "claim_reconciled": True,
                }
            )

    for index in range(366 - len(denominator_rows)):
        gap_id = f"uncovered-{index:03d}"
        month = str(pd.Period("2023-01", freq="M") + index)
        denominator_rows.append(
            {
                "gap_id": gap_id,
                "gap_type": "source_denominator_uncovered",
                "scope": DELISTING_CHANNEL,
                "month": month,
            }
        )
        coverage.append(
            {
                "gap_id": gap_id,
                "month": month,
                "channel": DELISTING_CHANNEL,
                "covered": False,
                "event_rows": 0,
                "zero_event_asserted": False,
                "source_url": "",
                "source_sha256": "",
            }
        )

    pd.DataFrame(denominator_rows).to_csv(denominator / "input_denominator.csv", index=False)
    pd.DataFrame(coverage).to_csv(denominator / "month_channel_coverage.csv", index=False)
    pd.DataFrame(inventory).to_csv(denominator / "source_inventory.csv", index=False)
    _seal_packet(denominator, {"status": "bounded_denominator"})

    validation = tmp_path / "denominator_validation"
    validation.mkdir()
    pd.DataFrame(claims).to_csv(validation / "claim_reconciliation.csv", index=False)
    _seal_packet(
        validation,
        {
            "status": "pass_partial_component",
            "validation_issue_count": 0,
            "producer_manifest_sha256": _sha(denominator / "manifest.json"),
        },
    )
    return denominator, validation


def test_parser_preserves_non_equity_aggregate_and_code_less_rows() -> None:
    kwargs = {
        "report_month": "2023-03",
        "source_url": "https://www.jpx.co.jp/report.pdf",
        "source_path": "/tmp/report.pdf",
        "source_sha256": "a" * 64,
    }
    rows = parse_report_event_rows(PAGES, channel=HALT_CHANNEL, **kwargs)
    independent = independently_reparse_report_rows(PAGES, channel=HALT_CHANNEL, **kwargs)
    assert len(rows) == 4
    assert [row["source_record_sha256"] for row in rows] == [
        row["source_record_sha256"] for row in independent
    ]
    aggregate = rows[1]
    assert aggregate["raw_code"] == "全銘柄"
    assert aggregate["normalized_local_code"] == ""
    assert aggregate["instrument_class"] == "index_option"
    code_less = rows[2]
    assert code_less["raw_code"] == ""
    assert code_less["issue_name"] == "TOPIX Options All Contracts"
    bond = rows[3]
    assert bond["instrument_class"] == "straight_bond"
    assert bond["halt_start_printed"] == "12:30"
    assert bond["halt_end_printed"] == "2023-03-22 12:29"
    assert bond["known_at"] == "" and bond["ric"] == ""


def test_full_bounded_shape_write_once_and_independent_exact_set(tmp_path: Path) -> None:
    denominator, validation = _packets(tmp_path)
    validation_manifest_path = validation / "manifest.json"
    validation_manifest_text = validation_manifest_path.read_text()
    mixed_upstream = json.loads(validation_manifest_text)
    mixed_upstream["producer_manifest_sha256"] = "f" * 64
    validation_manifest_path.write_text(json.dumps(mixed_upstream))
    with pytest.raises(ValueError, match="lineage"):
        run_japan_g4_halt_report_events(
            run_id="MIXED-UPSTREAM",
            denominator_dir=denominator,
            validation_dir=validation,
            artifact_root=tmp_path / "events",
            page_extractor=lambda _: PAGES,
        )
    validation_manifest_path.write_text(validation_manifest_text)

    producer = run_japan_g4_halt_report_events(
        run_id="EVENTS",
        denominator_dir=denominator,
        validation_dir=validation,
        artifact_root=tmp_path / "events",
        page_extractor=lambda _: PAGES,
    )
    assert producer.manifest["input_denominator_rows"] == 366
    assert producer.manifest["covered_month_channel_claims"] == 194
    assert producer.manifest["unique_reports"] == 155
    assert producer.manifest["known_at_assigned"] == 0
    assert producer.manifest["ric_assigned"] == 0
    assert verify_report_event_artifact(producer.artifact_dir) == []
    with pytest.raises(FileExistsError):
        run_japan_g4_halt_report_events(
            run_id="EVENTS",
            denominator_dir=denominator,
            validation_dir=validation,
            artifact_root=tmp_path / "events",
            page_extractor=lambda _: PAGES,
        )

    checked = validate_japan_g4_halt_report_events(
        run_id="VALID",
        producer_dir=producer.artifact_dir,
        denominator_dir=denominator,
        denominator_validation_dir=validation,
        artifact_root=tmp_path / "event_validation",
        page_extractor=lambda _: PAGES,
    )
    assert checked.manifest["status"] == "pass_bounded_report_event_component"
    assert checked.manifest["input_denominator_rows_verified"] == 366
    assert checked.manifest["covered_claims_reparsed"] == 194
    assert checked.manifest["reports_reparsed"] == 155
    assert checked.manifest["producer_report_rows"] == checked.manifest["independent_report_rows"]
    assert checked.manifest["producer_report_rows"] == checked.manifest["exact_field_reconciled_rows"]
    assert verify_report_event_validation_artifact(checked.artifact_dir) == []

    producer_manifest_path = producer.artifact_dir / "manifest.json"
    producer_manifest_text = producer_manifest_path.read_text()
    mixed_producer = json.loads(producer_manifest_text)
    mixed_producer["denominator_manifest_sha256"] = "e" * 64
    producer_manifest_path.write_text(json.dumps(mixed_producer))
    assert "producer_manifest_lineage_mismatch" in verify_report_event_validation_artifact(
        checked.artifact_dir
    )
    mixed = validate_japan_g4_halt_report_events(
        run_id="MIXED-LINEAGE",
        producer_dir=producer.artifact_dir,
        denominator_dir=denominator,
        denominator_validation_dir=validation,
        artifact_root=tmp_path / "event_validation",
        page_extractor=lambda _: PAGES,
    )
    assert mixed.manifest["status"] == "failed_validation"
    assert (
        "event_producer_denominator_lineage_mismatch"
        in mixed.manifest["validation_issue_codes"]
    )
    producer_manifest_path.write_text(producer_manifest_text)
    assert verify_report_event_validation_artifact(checked.artifact_dir) == []

    claims = pd.read_csv(producer.artifact_dir / "claim_row_reconciliation.csv")
    claims.loc[0, "explicit_report_rows"] = 0
    claims.loc[0, "zero_inferred"] = True
    claims.to_csv(producer.artifact_dir / "claim_row_reconciliation.csv", index=False)
    manifest = json.loads((producer.artifact_dir / "manifest.json").read_text())
    _seal_packet(producer.artifact_dir, manifest)
    rejected = validate_japan_g4_halt_report_events(
        run_id="ZERO-REJECTED",
        producer_dir=producer.artifact_dir,
        denominator_dir=denominator,
        denominator_validation_dir=validation,
        artifact_root=tmp_path / "event_validation",
        page_extractor=lambda _: PAGES,
    )
    assert rejected.manifest["status"] == "failed_validation"
    assert "no_inferred_zero_failed" in rejected.manifest["validation_issue_codes"]


@pytest.mark.parametrize("unsafe", ["../escape.csv", "/tmp/escape.csv"])
def test_packet_hash_inventory_rejects_unsafe_paths(
    tmp_path: Path, unsafe: str
) -> None:
    packet = tmp_path / "packet"
    packet.mkdir()
    hashes = packet / "artifact_hashes.json"
    hashes.write_text(json.dumps({unsafe: "0" * 64}), encoding="utf-8")
    (packet / "manifest.json").write_text(
        json.dumps({"artifact_hashes_sha256": _sha(hashes)}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="unsafe path"):
        _verify_packet(packet, "fixture")
    issues: list[dict[str, str]] = []
    _audit_packet(packet, "fixture", issues)
    assert any(issue["issue_code"] == "packet_hash_path_unsafe" for issue in issues)
