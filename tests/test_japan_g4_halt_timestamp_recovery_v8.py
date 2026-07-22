from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g4_halt_timestamp_recovery_v8 import (
    PAGE_SPECS,
    HttpResponse,
    produce_japan_g4_halt_timestamp_recovery_v8,
)
from quant_lab.data.japan_g4_halt_timestamp_recovery_v8_validator import (
    validate_japan_g4_halt_timestamp_recovery_v8,
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
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
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


def _page_date(value: str) -> str:
    return date.fromisoformat(value).strftime("%b. %d, %Y")


def _responses(*, corrupt_page_id: str = "") -> dict[str, HttpResponse]:
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
        body = (
            f"<html><body>Market News {_page_date(page.publication_date)} TSE "
            f"<h1>{page.heading_fragment}</h1>{support_link}<p>{event_text}</p>"
            "</body></html>"
        ).encode("utf-8")
        if page.page_id == corrupt_page_id:
            body = body.replace(page.events[0].local_code.encode(), b"9999")
        responses[page.url] = HttpResponse(
            payload=body,
            status=200,
            final_url=page.url,
            headers={
                "Content-Type": "text/html; charset=UTF-8",
                "Content-Length": str(len(body)),
                "Date": "Sun, 12 Jul 2026 00:00:00 GMT",
                "Last-Modified": "Tue, 28 Jan 2025 00:00:00 GMT",
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
                    "Last-Modified": "Wed, 08 Jan 2025 00:00:00 GMT",
                },
                acquired_at_utc="2026-07-12T00:00:02+00:00",
            )
    return responses


def _v8_fixture(tmp_path: Path) -> Path:
    events_path = tmp_path / "JPN-G4-HALTS-200607-202606-V8" / "halt_suspension_events.parquet"
    events_path.parent.mkdir(parents=True)
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
    ).to_parquet(events_path, index=False)
    manifest = {
        "run_id": "JPN-G4-HALTS-200607-202606-V8",
        "component": "halt_suspension_events",
        "artifact_type": "japan_g4_halt_suspension_history_producer",
        "output_sha256": {events_path.name: _sha(events_path)},
    }
    (events_path.parent / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return events_path


def _produce(tmp_path: Path, *, corrupt_page_id: str = ""):
    events_path = _v8_fixture(tmp_path)
    responses = _responses(corrupt_page_id=corrupt_page_id)
    return produce_japan_g4_halt_timestamp_recovery_v8(
        run_id="PRODUCER",
        events_path=events_path,
        artifact_root=tmp_path / "producer",
        fetcher=lambda url, _timeout: responses[url],
    )


def test_all_ten_claims_are_hashed_bound_and_independently_validated(
    tmp_path: Path,
) -> None:
    result = _produce(tmp_path)
    claims = pd.read_csv(result.claims_path, dtype=str, keep_default_na=False)
    sources = pd.read_csv(
        result.artifact_dir / "official_source_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )

    assert result.manifest["counts"]["accepted_events"] == 10
    assert result.manifest["counts"]["rejected_events"] == 0
    assert result.manifest["counts"]["exact_pages"] == 9
    assert result.manifest["counts"]["retained_source_responses"] == 12
    assert claims["publication_time_jst"].eq("").all()
    assert claims["publication_precision"].eq("official_page_date_only").all()
    assert claims.loc[claims["page_id"].eq("detail_696"), "known_at_utc"].item() == (
        "2013-09-13T15:00:00+00:00"
    )
    assert sources["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    _, validation = validate_japan_g4_halt_timestamp_recovery_v8(
        producer_dir=result.artifact_dir,
        validation_run_id="VALIDATION",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "pass"
    assert validation["counts"]["independently_validated_events"] == 10
    assert validation["counts"]["validated_source_responses"] == 12


def test_nonmatching_page_and_event_row_are_quarantined(tmp_path: Path) -> None:
    result = _produce(tmp_path, corrupt_page_id="detail_696")
    claims = pd.read_csv(result.claims_path, dtype=str, keep_default_na=False)
    sources = pd.read_csv(
        result.artifact_dir / "official_source_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )
    rejected = claims.loc[claims["status"].eq("rejected")]
    page_source = sources.loc[sources["source_id"].eq("detail_696-page")].iloc[0]

    assert result.manifest["counts"]["accepted_events"] == 9
    assert result.manifest["counts"]["rejected_events"] == 1
    assert rejected["event_id"].item() == "jpx-report-d129a47fac42d1338b0cdf10"
    assert rejected["known_at_utc"].item() == ""
    assert page_source["accepted"] == "False"
    assert "/quarantine/raw/" in page_source["raw_path"]

    _, validation = validate_japan_g4_halt_timestamp_recovery_v8(
        producer_dir=result.artifact_dir,
        validation_run_id="QUARANTINE-VALIDATION",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "pass"
    assert validation["counts"]["producer_rejected_events"] == 1


def test_independent_validator_detects_retained_byte_tampering(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    sources = pd.read_csv(
        result.artifact_dir / "official_source_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )
    path = Path(sources.loc[sources["source_id"].eq("detail_696-page"), "raw_path"].item())
    path.write_bytes(b"tampered")

    _, validation = validate_japan_g4_halt_timestamp_recovery_v8(
        producer_dir=result.artifact_dir,
        validation_run_id="TAMPER-VALIDATION",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "failed_validation"
    assert any(
        "source_validation_failed:detail_696-page" in issue
        for issue in validation["validation_issues"]
    )
