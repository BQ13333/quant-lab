from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_halt_timestamp_recovery_v8 import (
    PAGE_SPECS,
    HttpResponse,
    produce_japan_g4_halt_timestamp_recovery_v8,
)
from quant_lab.data.japan_g4_halt_timestamp_recovery_v8_validator import (
    validate_japan_g4_halt_timestamp_recovery_v8,
)
from quant_lab.data.japan_g4_halt_timestamp_sidecar_admission import (
    ACCEPTED_COLUMNS,
    CONSERVATIVE_DATE_ONLY_RULE,
    OFFICIAL_PRODUCER_RUN_ID,
    OFFICIAL_VALIDATION_RUN_ID,
    TimestampSidecarAdmissionError,
    copy_validated_japan_g4_halt_timestamp_sidecar,
    normalize_accepted_timestamp_sidecar,
    verify_japan_g4_halt_timestamp_sidecar,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _minimal_pdf(text: str) -> bytes:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 8 Tf 40 750 Td ({escaped}) Tj ET".encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    payload = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{index} 0 obj\n".encode("ascii"))
        payload.extend(obj)
        payload.extend(b"\nendobj\n")
    xref = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(payload)


def _responses() -> dict[str, HttpResponse]:
    responses: dict[str, HttpResponse] = {}
    for page in PAGE_SPECS:
        event_text = " ".join(
            fragment for event in page.events for fragment in event.required_fragments
        )
        support_link = (
            f'<a href="{page.support_url}">official attachment</a>'
            if page.support_url
            else ""
        )
        publication = date.fromisoformat(page.publication_date).strftime("%b. %d, %Y")
        body = (
            f"<html><body>Market News {publication} TSE "
            f"<h1>{page.heading_fragment}</h1>{support_link}<p>{event_text}</p>"
            "</body></html>"
        ).encode("utf-8")
        responses[page.url] = HttpResponse(
            payload=body,
            status=200,
            final_url=page.url,
            headers={
                "Content-Type": "text/html; charset=UTF-8",
                "Content-Length": str(len(body)),
                "Date": "Sun, 12 Jul 2026 00:00:00 GMT",
            },
            acquired_at_utc="2026-07-12T00:00:01+00:00",
        )
        if page.support_url:
            pdf = _minimal_pdf(event_text)
            responses[page.support_url] = HttpResponse(
                payload=pdf,
                status=200,
                final_url=page.support_url,
                headers={
                    "Content-Type": "application/pdf",
                    "Content-Length": str(len(pdf)),
                    "Date": "Sun, 12 Jul 2026 00:00:02 GMT",
                },
                acquired_at_utc="2026-07-12T00:00:02+00:00",
            )
    return responses


@pytest.fixture()
def packet(tmp_path: Path) -> tuple[Path, Path]:
    events = tmp_path / "v8" / "halt_suspension_events.parquet"
    events.parent.mkdir()
    pd.DataFrame(
        [
            {
                "event_id": event.event_id,
                "event_type": "delisting_non_trading_interval",
                "local_code": event.local_code,
                "issue_name": event.official_issue_name,
                "event_date": event.event_date,
            }
            for page in PAGE_SPECS
            for event in page.events
        ]
    ).to_parquet(events, index=False)
    (events.parent / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "JPN-G4-HALTS-200607-202606-V8",
                "component": "halt_suspension_events",
                "artifact_type": "japan_g4_halt_suspension_history_producer",
                "output_sha256": {events.name: _sha(events)},
            }
        ),
        encoding="utf-8",
    )
    responses = _responses()
    produced = produce_japan_g4_halt_timestamp_recovery_v8(
        run_id=OFFICIAL_PRODUCER_RUN_ID,
        events_path=events,
        artifact_root=tmp_path / "producer-root",
        fetcher=lambda url, _timeout: responses[url],
    )
    validation_manifest, _ = validate_japan_g4_halt_timestamp_recovery_v8(
        producer_dir=produced.artifact_dir,
        validation_run_id=OFFICIAL_VALIDATION_RUN_ID,
        artifact_root=tmp_path / "validation-root",
    )
    return produced.artifact_dir, validation_manifest.parent


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _rebind_validation_to_producer(producer: Path, validation: Path) -> None:
    manifest_path = producer / "manifest.json"
    validation_manifest_path = validation / "manifest.json"
    manifest = _json(validation_manifest_path)
    manifest["producer_manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha(manifest_path),
        "bytes": manifest_path.stat().st_size,
    }
    _write_json(validation_manifest_path, manifest)


def _rebind_producer_output(producer: Path, relative: str) -> None:
    manifest_path = producer / "manifest.json"
    manifest = _json(manifest_path)
    path = producer / relative
    manifest["outputs"][relative] = {
        "path": str(path),
        "sha256": _sha(path),
        "bytes": path.stat().st_size,
    }
    _write_json(manifest_path, manifest)


def test_valid_packet_is_normalized_and_sealed(
    packet: tuple[Path, Path], tmp_path: Path
) -> None:
    producer, validation = packet
    result = copy_validated_japan_g4_halt_timestamp_sidecar(
        producer, validation, tmp_path / "admitted"
    )

    assert tuple(result.accepted.columns) == ACCEPTED_COLUMNS
    assert len(result.accepted) == 10
    assert result.accepted["known_at_rule"].eq(CONSERVATIVE_DATE_ONLY_RULE).all()
    assert (result.destination / "producer" / "manifest.json").is_file()
    assert (result.destination / "validation" / "manifest.json").is_file()
    assert result.lineage["refinitiv_calls"] is False


@pytest.mark.parametrize("kind", ["declared_output", "raw_source", "response_metadata"])
def test_tamper_is_rejected(packet: tuple[Path, Path], kind: str) -> None:
    producer, validation = packet
    if kind == "declared_output":
        target = producer / "summary.md"
    else:
        sources = pd.read_csv(
            producer / "official_source_manifest.csv", dtype=str, keep_default_na=False
        )
        column = "raw_path" if kind == "raw_source" else "metadata_path"
        target = Path(sources.iloc[0][column])
    target.write_bytes(target.read_bytes() + b"tamper")

    with pytest.raises(TimestampSidecarAdmissionError):
        verify_japan_g4_halt_timestamp_sidecar(producer, validation)


def test_mixed_producer_validator_lineage_is_rejected(
    packet: tuple[Path, Path],
) -> None:
    producer, validation = packet
    manifest_path = validation / "manifest.json"
    manifest = _json(manifest_path)
    manifest["producer_manifest"]["sha256"] = "0" * 64
    _write_json(manifest_path, manifest)

    with pytest.raises(TimestampSidecarAdmissionError, match="lineage"):
        verify_japan_g4_halt_timestamp_sidecar(producer, validation)


def test_declared_count_mismatch_is_rejected(packet: tuple[Path, Path]) -> None:
    producer, validation = packet
    manifest_path = producer / "manifest.json"
    manifest = _json(manifest_path)
    manifest["counts"]["accepted_events"] = 9
    _write_json(manifest_path, manifest)
    _rebind_validation_to_producer(producer, validation)

    with pytest.raises(TimestampSidecarAdmissionError, match="count"):
        verify_japan_g4_halt_timestamp_sidecar(producer, validation)


def test_escaping_declared_source_path_is_rejected(
    packet: tuple[Path, Path], tmp_path: Path
) -> None:
    producer, validation = packet
    source_manifest_path = producer / "official_source_manifest.csv"
    sources = pd.read_csv(source_manifest_path, dtype=str, keep_default_na=False)
    outside = tmp_path / "outside.html"
    outside.write_bytes(Path(sources.iloc[0]["raw_path"]).read_bytes())
    sources.loc[0, "raw_path"] = str(outside)
    sources.to_csv(source_manifest_path, index=False)
    _rebind_producer_output(producer, "official_source_manifest.csv")
    _rebind_validation_to_producer(producer, validation)

    with pytest.raises(TimestampSidecarAdmissionError, match="source_path_escape"):
        verify_japan_g4_halt_timestamp_sidecar(producer, validation)


def test_symlinked_raw_source_is_rejected(
    packet: tuple[Path, Path], tmp_path: Path
) -> None:
    producer, validation = packet
    sources = pd.read_csv(
        producer / "official_source_manifest.csv", dtype=str, keep_default_na=False
    )
    raw_path = Path(sources.iloc[0]["raw_path"])
    outside = tmp_path / "same-bytes.html"
    shutil.copyfile(raw_path, outside)
    raw_path.unlink()
    raw_path.symlink_to(outside)

    with pytest.raises(TimestampSidecarAdmissionError, match="symlink"):
        verify_japan_g4_halt_timestamp_sidecar(producer, validation)


def test_existing_event_overlap_is_rejected(packet: tuple[Path, Path]) -> None:
    producer, validation = packet
    accepted = pd.read_csv(
        producer / "accepted_timestamp_recovery.csv", dtype=str, keep_default_na=False
    )

    with pytest.raises(TimestampSidecarAdmissionError, match="overlap"):
        verify_japan_g4_halt_timestamp_sidecar(
            producer, validation, existing_event_ids=[accepted.iloc[0]["event_id"]]
        )


def test_duplicate_accepted_event_id_is_rejected(packet: tuple[Path, Path]) -> None:
    producer, _ = packet
    accepted = pd.read_csv(
        producer / "accepted_timestamp_recovery.csv", dtype=str, keep_default_na=False
    )
    accepted.loc[1, "event_id"] = accepted.loc[0, "event_id"]

    with pytest.raises(TimestampSidecarAdmissionError):
        normalize_accepted_timestamp_sidecar(accepted)


def test_known_at_after_event_date_is_rejected(packet: tuple[Path, Path]) -> None:
    producer, _ = packet
    accepted = pd.read_csv(
        producer / "accepted_timestamp_recovery.csv", dtype=str, keep_default_na=False
    )
    accepted.loc[0, "event_date"] = "2013-09-13"

    with pytest.raises(TimestampSidecarAdmissionError, match="known_at_after_event_date"):
        normalize_accepted_timestamp_sidecar(accepted)
