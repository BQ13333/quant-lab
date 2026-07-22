from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_price_limit_bilingual_validation import (
    PROMOTION_POLICY,
    RECONCILIATION_RUN_ID,
    REQUIRED_COMPLETION_BLOCKERS,
    SOURCE_RUN_ID,
    _add_chain_inferred_sides,
    _parse_english_notice,
    _parse_japanese_notice,
    write_japan_price_limit_bilingual_validation,
)


ENGLISH_URL = "https://www.jpx.co.jp/english/news/1030/20240723-01.html"
JAPANESE_URL = "https://www.jpx.co.jp/news/1030/20240723-01.html"
ENGLISH_BODY = b"""
<p>AIAI Group Corporation (Code: 6557)<br>
Upper daily price limit: 300 yen (Lower limit will remain 150 yen as per normal)<br>
Base price: 746 yen.</p>
"""
JAPANESE_BODY = """
<html><body><p>AIAI Group Corporation (コード 6557)<br>
制限値幅（上限）：1,600円（下限は通常どおり400円）<br>
基準値段：1,620円</p></body></html>
""".encode("utf-8")


def test_clean_room_parsers_separate_blank_and_chain_inferred_side() -> None:
    preliminary = _parse_english_notice(
        b"<p>Alpha (Code: 1111) may expand only its upper daily price limit.</p>",
        publication_date="2019-01-02",
        event_type="PRELIMINARY",
        source_url="https://www.jpx.co.jp/english/news/detail/detail_1.html",
        title="preliminary",
    )[0]
    cancelled = _parse_english_notice(
        b"<p>Alpha (Code: 1111) will not have its price limits expanded tomorrow.</p>",
        publication_date="2019-01-03",
        event_type="CANCELLED",
        source_url="https://www.jpx.co.jp/english/news/detail/detail_2.html",
        title="cancelled",
    )[0]
    parsed = _add_chain_inferred_sides(pd.DataFrame([preliminary, cancelled]))
    cancelled_row = parsed.loc[parsed["event_type"].eq("CANCELLED")].iloc[0]
    assert cancelled_row["source_explicit_side"] == ""
    assert cancelled_row["chain_inferred_side"] == "UPPER"
    assert cancelled_row["resolved_side"] == "UPPER"

    japanese = _parse_japanese_notice(JAPANESE_BODY, source_url=JAPANESE_URL)
    assert japanese == [
        {
            "publication_date": "2024-07-23",
            "code": "6557",
            "source_explicit_side": "UPPER",
            "base_price_yen": 1620,
            "expanded_range_yen": 1600,
            "normal_range_yen": 400,
            "observed_multiplier": 4.0,
            "source_url": JAPANESE_URL,
        }
    ]


def test_validator_proves_single_correction_and_closes_discrepancies(tmp_path: Path) -> None:
    producer, overlay = _write_bundles(tmp_path)
    immutable_before = {**_tree_hashes(producer), **_tree_hashes(overlay, prefix="overlay/")}
    result = write_japan_price_limit_bilingual_validation(
        run_id="BILINGUAL-VALIDATOR-RUN",
        producer_dir=producer,
        reconciliation_dir=overlay,
        artifact_root=tmp_path / "validation",
    )

    assert result.manifest["status"] == "pass_partial_component"
    assert result.manifest["validation_issues"] == []
    assert result.manifest["counts"]["proven_correction_rows"] == 1
    assert result.manifest["counts"]["overlay_applied_rows"] == 1
    assert result.manifest["counts"]["unresolved_numeric_discrepancies"] == 0
    assert result.manifest["counts"]["unresolved_regime_discrepancies"] == 0
    assert result.manifest["counts"]["producer_blank_side_rows"] == 1
    assert result.manifest["counts"]["chain_inferred_side_rows"] == 1
    assert result.manifest["correction_proof"]["proven"] is True
    assert result.manifest["correction_proof"]["japanese_base_price_yen"] == 1620
    assert result.manifest["correction_proof"]["japanese_expanded_range_yen"] == 1600
    assert result.manifest["correction_proof"]["japanese_normal_range_yen"] == 400
    assert result.manifest["correction_proof"]["japanese_observed_multiplier"] == 4.0
    assert result.manifest["remaining_completion_blockers"] == list(REQUIRED_COMPLETION_BLOCKERS)
    assert result.manifest["promotion_allowed"] is False
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["component_scorecard_eligible"] is False
    assert result.manifest["promotion_policy"] == PROMOTION_POLICY
    assert {**_tree_hashes(producer), **_tree_hashes(overlay, prefix="overlay/")} == immutable_before

    summary = pd.read_csv(result.artifact_dir / "validation_summary.csv")
    detail = pd.read_csv(result.artifact_dir / "event_validation_detail.csv", keep_default_na=False)
    assert summary["passed"].all()
    cancellation = detail.loc[detail["event_type"].eq("CANCELLED")].iloc[0]
    assert cancellation["producer_side"] == ""
    assert cancellation["source_explicit_side"] == ""
    assert cancellation["chain_inferred_side"] == "UPPER"
    corrected = detail.loc[detail["code"].astype(str).eq("6557")].iloc[0]
    assert bool(corrected["overlay_applied"])
    assert corrected["authoritative_language"] == "japanese"
    assert float(corrected["validated_observed_multiplier"]) == 4.0
    assert not any(result.artifact_dir.glob(".*.tmp"))
    for filename, digest in result.manifest["output_sha256"].items():
        assert _digest(result.artifact_dir / filename) == digest


def test_validator_fails_closed_on_overlay_hash_tampering(tmp_path: Path) -> None:
    producer, overlay = _write_bundles(tmp_path)
    corrected = overlay / "corrected_candidate_events.csv"
    corrected.write_bytes(corrected.read_bytes() + b"\n")

    result = write_japan_price_limit_bilingual_validation(
        run_id="TAMPERED-BILINGUAL-VALIDATOR-RUN",
        producer_dir=producer,
        reconciliation_dir=overlay,
        artifact_root=tmp_path / "validation",
    )
    assert result.manifest["status"] == "failed_validation"
    assert "reconciliation_output_hash_mismatch:corrected_candidate_events.csv" in result.manifest[
        "validation_issues"
    ]
    assert result.manifest["promotion_allowed"] is False
    assert result.manifest["scorecard_eligible"] is False


def _write_bundles(tmp_path: Path) -> tuple[Path, Path]:
    producer = tmp_path / "producer"
    overlay = tmp_path / "overlay"
    raw_source = producer / "raw" / "notice_pages"
    raw_english = overlay / "raw" / "english"
    raw_japanese = overlay / "raw" / "japanese"
    raw_source.mkdir(parents=True)
    raw_english.mkdir(parents=True)
    raw_japanese.mkdir(parents=True)

    notices = [
        {
            "publication_date": "2019-01-02",
            "event_type": "PRELIMINARY",
            "code": "1111",
            "title": "Expansion of daily price limits (Preliminary Notice) : Alpha",
            "url": "https://www.jpx.co.jp/english/news/detail/detail_1.html",
            "body": b"<p>Alpha (Code: 1111) may expand only its upper daily price limit.</p>",
            "side": "UPPER",
        },
        {
            "publication_date": "2019-01-03",
            "event_type": "CANCELLED",
            "code": "1111",
            "title": "Announced Broadening of Daily Price Limits to be Cancelled : Alpha",
            "url": "https://www.jpx.co.jp/english/news/detail/detail_2.html",
            "body": b"<p>Alpha (Code: 1111) will not have its price limits expanded tomorrow.</p>",
            "side": "",
        },
        {
            "publication_date": "2024-07-23",
            "event_type": "BROADENING",
            "code": "6557",
            "title": "Daily Price Limits to be Broadened : 1 issue",
            "url": ENGLISH_URL,
            "body": ENGLISH_BODY,
            "side": "UPPER",
        },
    ]
    plan_rows: list[dict[str, object]] = []
    state_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    quality_rows: list[dict[str, object]] = []
    english_6557_path = Path()
    for index, notice in enumerate(notices, 1):
        row_id = f"JPL-{index:05d}"
        raw = raw_source / f"{row_id}.html"
        raw.write_bytes(notice["body"])
        common = {
            "plan_row_id": row_id,
            "publication_date": notice["publication_date"],
            "event_type": notice["event_type"],
            "title": notice["title"],
            "url": notice["url"],
            "archive_source_ids": "fixture",
        }
        plan_rows.append(common)
        state_rows.append(
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
        )
        parsed = _parse_english_notice(
            notice["body"],
            publication_date=str(notice["publication_date"]),
            event_type=str(notice["event_type"]),
            source_url=str(notice["url"]),
            title=str(notice["title"]),
        )[0]
        event_rows.append(
            {
                "publication_date": parsed["publication_date"],
                "archive_publication_date": parsed["archive_publication_date"],
                "effective_date": parsed["effective_date"],
                "event_type": parsed["event_type"],
                "code": parsed["code"],
                "issue_name": "Alpha" if parsed["code"] == "1111" else "AIAI Group Corporation",
                "side": notice["side"],
                "base_price_yen": parsed["base_price_yen"],
                "expanded_range_yen": parsed["expanded_range_yen"],
                "normal_range_yen": parsed["normal_range_yen"],
                "observed_multiplier": parsed["observed_multiplier"],
                "source_url": parsed["source_url"],
                "title": parsed["title"],
            }
        )
        applies = parsed["event_type"] in {"BROADENING", "CONTINUATION"}
        quality_rows.append(
            {
                "publication_date": parsed["publication_date"],
                "event_type": parsed["event_type"],
                "code": parsed["code"],
                "expected_multiplier": 4.0 if applies else None,
                "observed_multiplier": parsed["observed_multiplier"],
                "regime_multiplier_valid": not applies,
            }
        )
        if notice["code"] == "6557":
            english_6557_path = raw

    pd.DataFrame(plan_rows).to_csv(producer / "notice_plan.csv", index=False)
    pd.DataFrame(state_rows).to_csv(producer / "page_state.csv", index=False)
    pd.DataFrame(event_rows).to_csv(producer / "price_limit_events.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(producer / "event_quality.csv", index=False)
    source_hashes = {
        name: _digest(producer / name)
        for name in ("notice_plan.csv", "page_state.csv", "price_limit_events.csv", "event_quality.csv")
    }
    source_manifest = {
        "schema_version": "1.0-resumable-partial",
        "run_id": SOURCE_RUN_ID,
        "artifact_type": "japan_exceptional_price_limit_event_history_candidate",
        "status": "failed_candidate",
        "component": "temporary_price_limit_events",
        "official_jpx_source": True,
        "research_only": True,
        "scorecard_eligible": False,
        "component_scorecard_eligible": False,
        "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "output_sha256": source_hashes,
        "promotion_policy": PROMOTION_POLICY,
    }
    source_manifest_path = producer / "manifest.json"
    source_manifest_path.write_text(json.dumps(source_manifest), encoding="utf-8")

    retained_english = raw_english / "JPLBR-00001.html"
    retained_english.write_bytes(english_6557_path.read_bytes())
    retained_japanese = raw_japanese / "JPLBR-PAGE-00001.html"
    retained_japanese.write_bytes(JAPANESE_BODY)
    pd.DataFrame(
        [
            {
                "counterpart_id": "JPLBR-PAGE-00001",
                "japanese_source_url": JAPANESE_URL,
                "http_status": 200,
                "terminal_status": "success",
                "content_type": "text/html",
                "raw_path": str(retained_japanese),
                "sha256": _digest(retained_japanese),
                "bytes": retained_japanese.stat().st_size,
                "error": "",
            }
        ]
    ).to_csv(overlay / "counterpart_fetch_state.csv", index=False)
    pd.DataFrame(
        [
            {
                "reconciliation_id": "JPLBR-00001",
                "publication_date": "2024-07-23",
                "event_type": "BROADENING",
                "code": "6557",
                "expected_multiplier": 4.0,
                "english_regime_mismatch": True,
                "english_numeric_consistency_mismatch": False,
                "english_quality_flag_mismatch": True,
                "english_side": "UPPER",
                "english_base_price_yen": 746,
                "english_expanded_range_yen": 300,
                "english_normal_range_yen": 150,
                "english_observed_multiplier": 2.0,
                "english_source_url": ENGLISH_URL,
                "english_raw_path": str(retained_english),
                "english_sha256": _digest(retained_english),
                "english_bytes": retained_english.stat().st_size,
                "japanese_publication_date": "2024-07-23",
                "japanese_side": "UPPER",
                "japanese_base_price_yen": 1620,
                "japanese_expanded_range_yen": 1600,
                "japanese_normal_range_yen": 400,
                "japanese_observed_multiplier": 4.0,
                "japanese_source_url": JAPANESE_URL,
                "japanese_raw_path": str(retained_japanese),
                "japanese_sha256": _digest(retained_japanese),
                "japanese_bytes": retained_japanese.stat().st_size,
                "code_match": True,
                "date_match": True,
                "side_match": True,
                "japanese_regime_match": True,
                "bilingual_limit_numeric_mismatch": True,
                "bilingual_base_price_mismatch": True,
                "correction_eligible": True,
                "reconciliation_status": "corrected_candidate",
                "issues": "",
            }
        ]
    ).to_csv(overlay / "reconciliation.csv", index=False)
    corrected = dict(event_rows[-1])
    corrected.update(
        {
            "base_price_yen": 1620,
            "expanded_range_yen": 1600,
            "normal_range_yen": 400,
            "observed_multiplier": 4.0,
            "source_url": JAPANESE_URL,
            "reconciliation_id": "JPLBR-00001",
            "candidate_status": "bilingual_corrected_candidate",
            "candidate_only": True,
            "promotion_allowed": False,
            "english_source_url": ENGLISH_URL,
            "english_source_sha256": _digest(retained_english),
            "japanese_source_url": JAPANESE_URL,
            "japanese_source_sha256": _digest(retained_japanese),
        }
    )
    pd.DataFrame([corrected]).to_csv(overlay / "corrected_candidate_events.csv", index=False)
    overlay_hashes = {
        name: _digest(overlay / name)
        for name in ("counterpart_fetch_state.csv", "reconciliation.csv", "corrected_candidate_events.csv")
    }
    overlay_manifest = {
        "schema_version": "1.0-immutable-resumable-candidate",
        "run_id": RECONCILIATION_RUN_ID,
        "artifact_type": "japan_exceptional_price_limit_bilingual_reconciliation_candidate",
        "status": "partial_candidate",
        "component": "temporary_price_limit_events",
        "official_jpx_bilingual_sources": True,
        "source_producer_immutable": True,
        "research_only": True,
        "promotion_allowed": False,
        "scorecard_eligible": False,
        "component_scorecard_eligible": False,
        "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "source_manifest_sha256": _digest(source_manifest_path),
        "source_output_sha256": source_hashes,
        "output_sha256": overlay_hashes,
        "promotion_policy": PROMOTION_POLICY,
    }
    (overlay / "manifest.json").write_text(json.dumps(overlay_manifest), encoding="utf-8")
    return producer, overlay


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hashes(root: Path, *, prefix: str = "") -> dict[str, str]:
    return {
        f"{prefix}{path.relative_to(root)}": _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
