from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g4_halt_issuer_timestamp_4283 import (
    TARGET_EVENT_ID,
    TARGET_RECORD_SHA256,
    TARGET_SOURCE_SHA256,
    produce_japan_g4_halt_issuer_timestamp_4283,
)
from quant_lab.data.japan_g4_halt_issuer_timestamp_4283_validator import (
    validate_japan_g4_halt_issuer_timestamp_4283,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "v16"
    root.mkdir()
    event_source = root / "2015-07.pdf"
    event_source.write_bytes(b"%PDF-fixture-4283")
    source_digest = _sha(event_source)
    events = pd.DataFrame([{
        "event_id": TARGET_EVENT_ID,
        "event_type": "delisting_non_trading_interval",
        "local_code": "4283",
        "issue_name": "パナソニック インフォメーションシステムズ Panasonic Information Systems Co.,Ltd.",
        "event_date": "2015-07-29",
        "source_url": "https://www.jpx.co.jp/english/markets/statistics-equities/monthly/x/17_idou1507.pdf",
        "source_sha256": source_digest,
        "source_record_id": TARGET_RECORD_SHA256,
    }])
    events_path = root / "halt_suspension_events.parquet"
    events.to_parquet(events_path, index=False)
    manifest = {
        "run_id": "JPN-G4-HALTS-200607-202606-V16",
        "component": "halt_suspension_events",
        "artifact_type": "japan_g4_halt_suspension_history_producer",
        "output_sha256": {events_path.name: _sha(events_path)},
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return events_path, event_source


def _patch_event_digest(monkeypatch: object, event_source: Path) -> None:
    digest = _sha(event_source)
    monkeypatch.setattr(
        "quant_lab.data.japan_g4_halt_issuer_timestamp_4283.TARGET_SOURCE_SHA256", digest
    )
    monkeypatch.setattr(
        "quant_lab.data.japan_g4_halt_issuer_timestamp_4283_validator.TARGET_SOURCE_SHA256", digest
    )


def _issuer_html(publication: str = "2015-05-08") -> bytes:
    return f"""<!doctype html><html><head>
    <meta property="article:published_time" content="{publication}T15:00:00+09:00">
    <script type="application/ld+json">{{"@type":"NewsArticle","datePublished":"{publication}"}}</script>
    </head><body>
    <h1>パナソニック インフォメーションシステムズ株式会社</h1>
    <p>証券コード 4283</p><p>上場廃止日は2015年7月29日です。</p>
    </body></html>""".encode()


def test_missing_issuer_bytes_produces_valid_fail_closed_packet(tmp_path: Path, monkeypatch: object) -> None:
    events, event_source = _inputs(tmp_path)
    _patch_event_digest(monkeypatch, event_source)
    producer = produce_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-AUDIT-TEST-V1",
        events_path=events,
        event_source_path=event_source,
        artifact_root=tmp_path / "producer",
    )
    validation = validate_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-AUDIT-VALIDATION-TEST-V1",
        producer_manifest_path=producer.manifest_path,
        artifact_root=tmp_path / "validation",
    )
    claim = pd.read_csv(producer.claim_path, dtype=str, keep_default_na=False)
    assert producer.manifest["status"] == "blocked_fail_closed"
    assert producer.manifest["counts"]["accepted_events"] == 0
    assert "issuer_source_provided" in claim.iloc[0]["rejection_reason"]
    assert claim.iloc[0]["known_at_utc"] == ""
    assert validation.manifest["packet_valid"] is True
    assert validation.manifest["claim_accepted"] is False
    assert validation.manifest["counts"]["validation_issues"] == 0


def test_exact_retained_issuer_bytes_can_be_accepted(tmp_path: Path, monkeypatch: object) -> None:
    events, event_source = _inputs(tmp_path)
    _patch_event_digest(monkeypatch, event_source)
    issuer = tmp_path / "issuer.html"
    issuer.write_bytes(_issuer_html())
    producer = produce_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-ACCEPT-TEST-V1",
        events_path=events,
        event_source_path=event_source,
        issuer_source_path=issuer,
        issuer_source_url="https://is-c.panasonic.co.jp/news/2015/0508.html",
        issuer_source_sha256=_sha(issuer),
        publication_date="2015-05-08",
        artifact_root=tmp_path / "producer",
    )
    validation = validate_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-ACCEPT-VALIDATION-TEST-V1",
        producer_manifest_path=producer.manifest_path,
        artifact_root=tmp_path / "validation",
    )
    claim = pd.read_csv(producer.claim_path, dtype=str, keep_default_na=False)
    assert producer.manifest["status"] == "complete"
    assert producer.manifest["counts"]["accepted_events"] == 1
    assert claim.iloc[0]["known_at_utc"] == "2015-05-08T15:00:00+00:00"
    assert validation.manifest["packet_valid"] is True
    assert validation.manifest["claim_accepted"] is True
    assert validation.manifest["counts"]["validation_issues"] == 0


def test_post_event_publication_fails_closed(tmp_path: Path, monkeypatch: object) -> None:
    events, event_source = _inputs(tmp_path)
    _patch_event_digest(monkeypatch, event_source)
    issuer = tmp_path / "issuer.html"
    issuer.write_bytes(_issuer_html("2015-07-29"))
    producer = produce_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-POST-EVENT-TEST-V1",
        events_path=events,
        event_source_path=event_source,
        issuer_source_path=issuer,
        issuer_source_url="https://is-c.panasonic.co.jp/news/2015/0729.html",
        issuer_source_sha256=_sha(issuer),
        publication_date="2015-07-29",
        artifact_root=tmp_path / "producer",
    )
    claim = pd.read_csv(producer.claim_path, dtype=str, keep_default_na=False)
    assert producer.manifest["counts"]["accepted_events"] == 0
    assert "publication_precedes_event" in claim.iloc[0]["rejection_reason"]


def test_validator_detects_retained_source_tampering(tmp_path: Path, monkeypatch: object) -> None:
    events, event_source = _inputs(tmp_path)
    _patch_event_digest(monkeypatch, event_source)
    issuer = tmp_path / "issuer.html"
    issuer.write_bytes(_issuer_html())
    producer = produce_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-TAMPER-TEST-V1",
        events_path=events,
        event_source_path=event_source,
        issuer_source_path=issuer,
        issuer_source_url="https://is-c.panasonic.co.jp/news/2015/0508.html",
        issuer_source_sha256=_sha(issuer),
        publication_date="2015-05-08",
        artifact_root=tmp_path / "producer",
    )
    claim = pd.read_csv(producer.claim_path, dtype=str, keep_default_na=False).iloc[0]
    Path(claim["issuer_source_path"]).write_bytes(b"tampered")
    validation = validate_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-TAMPER-VALIDATION-TEST-V1",
        producer_manifest_path=producer.manifest_path,
        artifact_root=tmp_path / "validation",
    )
    issues = pd.read_csv(validation.issues_path)
    assert validation.manifest["packet_valid"] is False
    assert "output_hash_mismatch" in set(issues["check_id"])


def test_symlinked_issuer_source_fails_closed(tmp_path: Path, monkeypatch: object) -> None:
    events, event_source = _inputs(tmp_path)
    _patch_event_digest(monkeypatch, event_source)
    issuer = tmp_path / "issuer.html"
    issuer.write_bytes(_issuer_html())
    link = tmp_path / "issuer-link.html"
    link.symlink_to(issuer)
    producer = produce_japan_g4_halt_issuer_timestamp_4283(
        run_id="JPN-G4-4283-ISSUER-SYMLINK-TEST-V1",
        events_path=events,
        event_source_path=event_source,
        issuer_source_path=link,
        issuer_source_url="https://is-c.panasonic.co.jp/news/2015/0508.html",
        issuer_source_sha256=_sha(issuer),
        publication_date="2015-05-08",
        artifact_root=tmp_path / "producer",
    )
    claim = pd.read_csv(producer.claim_path, dtype=str, keep_default_na=False)
    assert producer.manifest["counts"]["accepted_events"] == 0
    assert "issuer_source_not_symlink" in claim.iloc[0]["rejection_reason"]
