from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g4_halt_official_publication_timestamp_v14 import (
    EVENT_SPECS,
    HttpResponse,
    produce_japan_g4_halt_official_publication_timestamp_v14,
)
from quant_lab.data.japan_g4_halt_official_publication_timestamp_v14_validator import (
    validate_japan_g4_halt_official_publication_timestamp_v14,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _v14_fixture(tmp_path: Path, *, bad_source_record_code: str = "") -> Path:
    events_path = (
        tmp_path / "JPN-G4-HALTS-200607-202606-V14" / "halt_suspension_events.parquet"
    )
    events_path.parent.mkdir(parents=True)
    rows = []
    for spec in EVENT_SPECS:
        rows.append(
            {
                "event_id": spec.event_id,
                "event_type": "delisting_non_trading_interval",
                "local_code": spec.local_code,
                "issue_name": spec.issue_name,
                "event_date": spec.event_date,
                "source_id": f"jpx-monthly-report-{spec.event_date[:7]}-delisting_non_trading",
                "source_url": spec.event_source_url,
                "source_sha256": spec.event_source_sha256,
                "source_record_id": (
                    "0" * 64
                    if spec.local_code == bad_source_record_code
                    else spec.source_record_id
                ),
            }
        )
    pd.DataFrame(rows).to_parquet(events_path, index=False)
    manifest = {
        "run_id": "JPN-G4-HALTS-200607-202606-V14",
        "component": "halt_suspension_events",
        "artifact_type": "japan_g4_halt_suspension_history_producer",
        "output_sha256": {events_path.name: _sha(events_path)},
    }
    (events_path.parent / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return events_path


def _responses(
    *,
    corrupt_code: str = "",
    redirected_code: str = "",
    foreign_domain_code: str = "",
) -> dict[str, HttpResponse]:
    responses = {}
    for spec in EVENT_SPECS:
        fragments = list(spec.required_fragments)
        if spec.local_code == corrupt_code:
            fragments[-1] = "2099-01-01"
        body = (
            "<html><body><main><h1>Official JPX notice</h1><p>"
            + "</p><p>".join(fragments)
            + "</p></main></body></html>"
        ).encode("utf-8")
        final_url = spec.notice_url
        if spec.local_code == redirected_code:
            final_url = spec.notice_url.replace(".html", "-redirected.html")
        if spec.local_code == foreign_domain_code:
            final_url = "https://example.com/news/notice.html"
        responses[spec.notice_url] = HttpResponse(
            payload=body,
            status=200,
            final_url=final_url,
            headers={
                "Content-Type": "text/html; charset=UTF-8",
                "Content-Length": str(len(body)),
                "Date": "Sun, 12 Jul 2026 12:00:00 GMT",
                "Last-Modified": "Mon, 01 Jan 2024 00:00:00 GMT",
            },
            acquired_at_utc="2026-07-12T12:00:01+00:00",
        )
    return responses


def _produce(
    tmp_path: Path,
    *,
    bad_source_record_code: str = "",
    corrupt_code: str = "",
    redirected_code: str = "",
    foreign_domain_code: str = "",
):
    events = _v14_fixture(tmp_path, bad_source_record_code=bad_source_record_code)
    responses = _responses(
        corrupt_code=corrupt_code,
        redirected_code=redirected_code,
        foreign_domain_code=foreign_domain_code,
    )
    return produce_japan_g4_halt_official_publication_timestamp_v14(
        run_id="PRODUCER",
        events_path=events,
        artifact_root=tmp_path / "producer",
        fetcher=lambda url, _timeout: responses[url],
    )


def test_exact_six_claims_are_bound_and_independently_validated(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    claims = pd.read_csv(result.claims_path, dtype=str, keep_default_na=False)
    sources = pd.read_csv(
        result.artifact_dir / "official_source_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )

    assert result.manifest["status"] == "complete"
    assert result.manifest["counts"]["accepted_events"] == 6
    assert set(claims["local_code"]) == {"6784", "2712", "9963", "7708", "3587", "4798"}
    assert "4283" not in set(claims["local_code"])
    assert claims["publication_time_jst"].eq("").all()
    assert claims["known_at_rule"].eq(
        "date_only_conservative_upper_bound_next_midnight_jst"
    ).all()
    assert sources["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    _, validation = validate_japan_g4_halt_official_publication_timestamp_v14(
        producer_dir=result.artifact_dir,
        validation_run_id="VALIDATION",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "pass"
    assert validation["counts"]["independently_validated_events"] == 6
    assert validation["counts"]["validated_source_responses"] == 6


def test_first_next_midnight_jst_is_used_for_each_claim(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    claims = pd.read_csv(result.accepted_path, dtype=str, keep_default_na=False)
    expected = {
        "6784": "2014-11-19T15:00:00+00:00",
        "2712": "2015-02-20T15:00:00+00:00",
        "9963": "2015-04-30T15:00:00+00:00",
        "7708": "2015-06-30T15:00:00+00:00",
        "3587": "2015-08-11T15:00:00+00:00",
        "4798": "2015-10-30T15:00:00+00:00",
    }
    assert claims.set_index("local_code")["known_at_utc"].to_dict() == expected


def test_content_mismatch_is_rejected_and_quarantined(tmp_path: Path) -> None:
    result = _produce(tmp_path, corrupt_code="9963")
    claims = pd.read_csv(result.claims_path, dtype=str, keep_default_na=False)
    row = claims.loc[claims["local_code"].eq("9963")].iloc[0]
    assert row["status"] == "rejected"
    assert row["known_at_utc"] == ""
    assert "delisting_date" in row["rejection_reason"]
    assert "/quarantine/raw/" in row["source_path"]


def test_redirect_is_rejected_even_when_domain_remains_jpx(tmp_path: Path) -> None:
    result = _produce(tmp_path, redirected_code="6784")
    claims = pd.read_csv(result.claims_path, dtype=str, keep_default_na=False)
    row = claims.loc[claims["local_code"].eq("6784")].iloc[0]
    assert row["status"] == "rejected"
    assert "exact_final_url" in row["rejection_reason"]


def test_foreign_final_domain_is_rejected(tmp_path: Path) -> None:
    result = _produce(tmp_path, foreign_domain_code="2712")
    claims = pd.read_csv(result.claims_path, dtype=str, keep_default_na=False)
    row = claims.loc[claims["local_code"].eq("2712")].iloc[0]
    assert row["status"] == "rejected"
    assert "official_final_domain" in row["rejection_reason"]


def test_exact_v14_source_record_binding_fails_closed(tmp_path: Path) -> None:
    result = _produce(tmp_path, bad_source_record_code="4798")
    claims = pd.read_csv(result.claims_path, dtype=str, keep_default_na=False)
    row = claims.loc[claims["local_code"].eq("4798")].iloc[0]
    assert row["status"] == "rejected"
    assert "source_record_id" in row["rejection_reason"]


def test_independent_validator_detects_retained_byte_tampering(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    sources = pd.read_csv(
        result.artifact_dir / "official_source_manifest.csv",
        dtype=str,
        keep_default_na=False,
    )
    raw_path = Path(sources.loc[sources["event_id"].eq(EVENT_SPECS[0].event_id), "raw_path"].item())
    raw_path.write_bytes(b"tampered")

    _, validation = validate_japan_g4_halt_official_publication_timestamp_v14(
        producer_dir=result.artifact_dir,
        validation_run_id="TAMPER-VALIDATION",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "failed_validation"
    assert any("source_validation_failed" in issue for issue in validation["validation_issues"])


def test_independent_validator_detects_claim_file_tampering(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    result.claims_path.write_text("tampered\n", encoding="utf-8")

    _, validation = validate_japan_g4_halt_official_publication_timestamp_v14(
        producer_dir=result.artifact_dir,
        validation_run_id="CLAIM-TAMPER-VALIDATION",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "failed_validation"
    assert "producer_output_invalid:timestamp_recovery.csv" in validation["validation_issues"]
