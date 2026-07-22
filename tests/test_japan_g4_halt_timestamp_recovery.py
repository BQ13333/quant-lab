from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g4_halt_timestamp_recovery import (
    KNOWN_AT_STATUS,
    produce_japan_g4_halt_timestamp_recovery,
    validate_japan_g4_halt_timestamp_recovery,
)


NOTICE = b"""
<html><body><h1>Decision on Delisting, etc.: Alpha Holdings,Inc.</h1>
<p>Jan. 10, 2020 TSE</p><table><tr><td>Alpha Holdings,Inc. stock</td>
<td>(Code: 1234, Market Division: 1st Section)</td></tr></table></body></html>
"""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    gaps = tmp_path / "unresolved_gaps.csv"
    pd.DataFrame(
        [
            {"gap_id": "gap-1", "gap_type": "historical_pit_timestamp_unresolved", "event_id": "event-1", "local_code": "1234"},
            {"gap_id": "gap-2", "gap_type": "historical_pit_timestamp_unresolved", "event_id": "event-2", "local_code": "5678"},
            {"gap_id": "other", "gap_type": "ric_mapping_unresolved", "event_id": "event-1", "local_code": "1234"},
        ]
    ).to_csv(gaps, index=False)
    events = tmp_path / "events.parquet"
    pd.DataFrame(
        [
            {"event_id": "event-1", "event_type": "delisting_non_trading_interval", "local_code": "1234", "issue_name": "Alpha Holdings,Inc.", "event_date": "2020-01-20"},
            {"event_id": "event-2", "event_type": "delisting_non_trading_interval", "local_code": "5678", "issue_name": "No Notice Co.,Ltd.", "event_date": "2020-02-20"},
        ]
    ).to_parquet(events, index=False)
    archive = tmp_path / "news_ym_202001.json"
    archive.write_text(
        json.dumps(
            [
                {
                    "title": "Decision on Delisting, etc.: Alpha Holdings,Inc.",
                    "url": "/english/news/1021/20200110-11.html",
                    "updated_date": {"year": "2020", "month": "01", "day": "10"},
                }
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "archive_source_manifest.csv"
    pd.DataFrame(
        [
            {
                "source_id": "news_ym_202001",
                "url": "https://www.jpx.co.jp/english/news/news_ym_202001.json",
                "raw_path": str(archive),
                "sha256": _sha(archive),
            }
        ]
    ).to_csv(manifest, index=False)
    return gaps, events, manifest


def test_recovery_is_conservative_hashed_and_fail_closed(tmp_path: Path) -> None:
    gaps, events, archive_manifest = _fixture(tmp_path)
    result = produce_japan_g4_halt_timestamp_recovery(
        run_id="RECOVERY",
        unresolved_gaps_path=gaps,
        events_path=events,
        archive_source_manifest_path=archive_manifest,
        artifact_root=tmp_path / "artifacts",
        expected_target_count=2,
        fetcher=lambda _url, _timeout: NOTICE,
    )
    rows = pd.read_csv(result.recovery_path, dtype=str, keep_default_na=False)
    recovered = rows.loc[rows.event_id.eq("event-1")].iloc[0]
    unresolved = rows.loc[rows.event_id.eq("event-2")].iloc[0]
    assert recovered.known_at_utc == "2020-01-10T15:00:00+00:00"
    assert recovered.known_at_status == KNOWN_AT_STATUS
    assert recovered.known_at_utc[:10] != recovered.event_date
    assert unresolved.known_at_utc == ""
    assert unresolved.unresolved_reason == "no_unique_official_delisting_decision_notice"
    assert result.manifest["counts"]["recovered_rows"] == 1
    assert result.manifest["counts"]["unresolved_rows"] == 1
    assert result.manifest["promotion_eligible"] is False
    assert result.validation_manifest["status"] == "pass"
    inventory = pd.read_csv(result.artifact_dir / "source_inventory.csv")
    assert inventory["sha256"].str.fullmatch(r"[0-9a-f]{64}").all()


def test_notice_without_exact_code_stays_unresolved(tmp_path: Path) -> None:
    gaps, events, archive_manifest = _fixture(tmp_path)
    result = produce_japan_g4_halt_timestamp_recovery(
        run_id="NO-CODE",
        unresolved_gaps_path=gaps,
        events_path=events,
        archive_source_manifest_path=archive_manifest,
        artifact_root=tmp_path / "artifacts",
        expected_target_count=2,
        fetcher=lambda _url, _timeout: NOTICE.replace(b"1234", b"9999"),
    )
    rows = pd.read_csv(result.recovery_path, dtype=str, keep_default_na=False)
    alpha = rows.loc[rows.event_id.eq("event-1")].iloc[0]
    assert alpha.recovery_status == "unresolved"
    assert "code" in alpha.unresolved_reason


def test_independent_validation_detects_notice_tampering(tmp_path: Path) -> None:
    gaps, events, archive_manifest = _fixture(tmp_path)
    result = produce_japan_g4_halt_timestamp_recovery(
        run_id="TAMPER",
        unresolved_gaps_path=gaps,
        events_path=events,
        archive_source_manifest_path=archive_manifest,
        artifact_root=tmp_path / "artifacts",
        expected_target_count=2,
        fetcher=lambda _url, _timeout: NOTICE,
    )
    notice_path = next((result.artifact_dir / "raw").glob("*.html"))
    notice_path.write_bytes(b"tampered")
    _, validation = validate_japan_g4_halt_timestamp_recovery(
        producer_dir=result.artifact_dir,
        validation_dir=result.artifact_dir / "validation-after-tamper",
        expected_target_count=2,
    )
    assert validation["status"] == "failed_validation"
    assert any("row_validation_failed:event-1" in issue for issue in validation["validation_issues"])
