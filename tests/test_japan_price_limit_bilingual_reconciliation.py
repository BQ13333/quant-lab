from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_price_limit_bilingual_reconciliation import (
    PROMOTION_POLICY,
    REQUIRED_COMPLETION_BLOCKERS,
    derive_japanese_counterpart_url,
    parse_japanese_price_limit_notice,
    write_japan_price_limit_bilingual_reconciliation,
)


ENGLISH_URL = "https://www.jpx.co.jp/english/news/1030/20240723-01.html"
JAPANESE_URL = "https://www.jpx.co.jp/news/1030/20240723-01.html"
ENGLISH_BODY = b"""
<p>AIAI Group Corporation (Code: 6557)<br>
Upper daily price limit: 300 yen (Lower limit will remain 150 yen as per normal)<br>
Base price: 746 yen.</p>
"""
JAPANESE_BODY = """
<html><body><p>
 AIAIグループ(株)株式(コード 6557)<br>
 制限値幅(上限):1,600円(下限は通常どおり400円)<br>
 基準値段:1,620円、ストップ高:3,220円、ストップ安:1,220円<br>
 (株)セルシード 株式(コード 7776)<br>
 制限値幅(下限):400円(上限は通常どおり100円)<br>
 基準値段:537円
</p></body></html>
""".encode("utf-8")


def test_derives_official_counterpart_and_parses_values_per_code() -> None:
    assert derive_japanese_counterpart_url(ENGLISH_URL) == JAPANESE_URL
    with pytest.raises(ValueError):
        derive_japanese_counterpart_url("https://example.com/english/news/1030/a.html")

    rows = parse_japanese_price_limit_notice(JAPANESE_BODY, source_url=JAPANESE_URL)
    assert rows == [
        {
            "publication_date": "2024-07-23",
            "code": "6557",
            "side": "UPPER",
            "base_price_yen": 1620,
            "expanded_range_yen": 1600,
            "normal_range_yen": 400,
            "observed_multiplier": 4.0,
            "source_url": JAPANESE_URL,
        },
        {
            "publication_date": "2024-07-23",
            "code": "7776",
            "side": "LOWER",
            "base_price_yen": 537,
            "expanded_range_yen": 400,
            "normal_range_yen": 100,
            "observed_multiplier": 4.0,
            "source_url": JAPANESE_URL,
        },
    ]


def test_reconciles_6557_preserves_both_sources_and_resumes(tmp_path: Path) -> None:
    producer = _write_source_bundle(tmp_path)
    source_hashes_before = _tree_hashes(producer)
    calls: list[str] = []

    def fetcher(url: str) -> tuple[int, bytes, dict[str, str]]:
        calls.append(url)
        return 200, JAPANESE_BODY, {"Content-Type": "text/html; charset=UTF-8"}

    result = write_japan_price_limit_bilingual_reconciliation(
        run_id="BILINGUAL-RUN",
        producer_dir=producer,
        artifact_root=tmp_path / "reconciliation",
        sleep_seconds=0,
        fetcher=fetcher,
    )
    assert calls == [JAPANESE_URL]
    assert result.manifest["status"] == "partial_candidate"
    assert result.manifest["counts"]["detected_mismatch_rows"] == 1
    assert result.manifest["counts"]["corrected_candidate_event_rows"] == 1
    assert result.manifest["promotion_allowed"] is False
    assert result.manifest["promotion_policy"] == PROMOTION_POLICY
    assert result.manifest["remaining_completion_blockers"] == list(REQUIRED_COMPLETION_BLOCKERS)
    assert _tree_hashes(producer) == source_hashes_before

    reconciliation = pd.read_csv(result.artifact_dir / "reconciliation.csv", dtype={"code": str}).iloc[0]
    assert reconciliation["code"] == "6557"
    assert bool(reconciliation["english_regime_mismatch"])
    assert bool(reconciliation["bilingual_limit_numeric_mismatch"])
    assert bool(reconciliation["code_match"])
    assert bool(reconciliation["date_match"])
    assert bool(reconciliation["side_match"])
    assert bool(reconciliation["japanese_regime_match"])
    assert bool(reconciliation["correction_eligible"])
    assert Path(reconciliation["english_raw_path"]).read_bytes() == ENGLISH_BODY
    assert Path(reconciliation["japanese_raw_path"]).read_bytes() == JAPANESE_BODY
    assert reconciliation["english_sha256"] == hashlib.sha256(ENGLISH_BODY).hexdigest()
    assert reconciliation["japanese_sha256"] == hashlib.sha256(JAPANESE_BODY).hexdigest()

    corrected = pd.read_csv(
        result.artifact_dir / "corrected_candidate_events.csv", dtype={"code": str}
    ).iloc[0]
    assert corrected["code"] == "6557"
    assert corrected["base_price_yen"] == 1620
    assert corrected["expanded_range_yen"] == 1600
    assert corrected["normal_range_yen"] == 400
    assert corrected["observed_multiplier"] == 4.0
    assert corrected["english_source_url"] == ENGLISH_URL
    assert corrected["japanese_source_url"] == JAPANESE_URL
    assert corrected["source_url"] == JAPANESE_URL
    assert bool(corrected["candidate_only"])
    assert not bool(corrected["promotion_allowed"])

    def no_refetch(_: str) -> tuple[int, bytes, dict[str, str]]:
        raise AssertionError("successful counterpart should resume from retained raw bytes")

    resumed = write_japan_price_limit_bilingual_reconciliation(
        run_id="BILINGUAL-RUN",
        producer_dir=producer,
        artifact_root=tmp_path / "reconciliation",
        sleep_seconds=0,
        fetcher=no_refetch,
    )
    assert resumed.manifest["counts"]["corrected_candidate_event_rows"] == 1


def test_fails_closed_when_side_does_not_tie_out(tmp_path: Path) -> None:
    producer = _write_source_bundle(tmp_path)
    wrong_side = JAPANESE_BODY.replace("(上限)".encode(), "(下限)".encode(), 1)
    result = write_japan_price_limit_bilingual_reconciliation(
        run_id="SIDE-MISMATCH-RUN",
        producer_dir=producer,
        artifact_root=tmp_path / "reconciliation",
        sleep_seconds=0,
        fetcher=lambda _: (200, wrong_side, {}),
    )
    assert result.manifest["status"] == "failed_candidate"
    assert result.manifest["counts"]["corrected_candidate_event_rows"] == 0
    assert "unresolved_bilingual_mismatches=1" in result.manifest["reconciliation_blockers"]
    corrected = pd.read_csv(result.artifact_dir / "corrected_candidate_events.csv")
    assert corrected.empty
    reconciliation = pd.read_csv(result.artifact_dir / "reconciliation.csv").iloc[0]
    assert reconciliation["reconciliation_status"] == "tie_out_failed"
    assert "side_mismatch" in reconciliation["issues"]


def test_single_writer_lock_and_source_hash_are_enforced(tmp_path: Path) -> None:
    producer = _write_source_bundle(tmp_path)
    output = tmp_path / "reconciliation" / "LOCKED-RUN"
    (output / ".writer.lock").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="single-writer lock"):
        write_japan_price_limit_bilingual_reconciliation(
            run_id="LOCKED-RUN",
            producer_dir=producer,
            artifact_root=tmp_path / "reconciliation",
            sleep_seconds=0,
            fetcher=lambda _: (200, JAPANESE_BODY, {}),
        )

    events = producer / "price_limit_events.csv"
    events.write_bytes(events.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="immutable source output hash mismatch"):
        write_japan_price_limit_bilingual_reconciliation(
            run_id="TAMPERED-RUN",
            producer_dir=producer,
            artifact_root=tmp_path / "reconciliation",
            sleep_seconds=0,
            fetcher=lambda _: (200, JAPANESE_BODY, {}),
        )


def _write_source_bundle(tmp_path: Path) -> Path:
    producer = tmp_path / "producer"
    raw_dir = producer / "raw" / "notice_pages"
    raw_dir.mkdir(parents=True)
    raw = raw_dir / "JPL-00001.html"
    raw.write_bytes(ENGLISH_BODY)
    event = {
        "publication_date": "2024-07-23",
        "archive_publication_date": "2024-06-23",
        "effective_date": "2024-07-24",
        "event_type": "BROADENING",
        "code": "6557",
        "issue_name": "AIAI Group Corporation",
        "side": "UPPER",
        "base_price_yen": 746,
        "expanded_range_yen": 300,
        "normal_range_yen": 150,
        "observed_multiplier": 2.0,
        "source_url": ENGLISH_URL,
        "title": "Daily Price Limits to be Broadened : 1 issue",
    }
    common = {
        "plan_row_id": "JPL-00001",
        "publication_date": event["publication_date"],
        "event_type": event["event_type"],
        "title": event["title"],
        "url": ENGLISH_URL,
        "archive_source_ids": "news_ym_202407",
    }
    pd.DataFrame([common]).to_csv(producer / "notice_plan.csv", index=False)
    pd.DataFrame(
        [
            {
                **common,
                "http_status": 200,
                "terminal_status": "success",
                "content_type": "text/html",
                "raw_path": str(raw),
                "sha256": _digest(raw),
                "bytes": raw.stat().st_size,
                "error": "",
            }
        ]
    ).to_csv(producer / "page_state.csv", index=False)
    pd.DataFrame([event]).to_csv(producer / "price_limit_events.csv", index=False)
    pd.DataFrame(
        [
            {
                "publication_date": "2024-07-23",
                "event_type": "BROADENING",
                "code": "6557",
                "expected_multiplier": 4.0,
                "observed_multiplier": 2.0,
                "regime_multiplier_valid": False,
            }
        ]
    ).to_csv(producer / "event_quality.csv", index=False)
    output_hashes = {filename: _digest(producer / filename) for filename in (
        "notice_plan.csv",
        "page_state.csv",
        "price_limit_events.csv",
        "event_quality.csv",
    )}
    manifest = {
        "run_id": "SOURCE-RUN",
        "artifact_type": "japan_exceptional_price_limit_event_history_candidate",
        "output_sha256": output_hashes,
        "promotion_policy": PROMOTION_POLICY,
    }
    (producer / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return producer


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
