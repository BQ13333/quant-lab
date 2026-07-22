from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_price_limit_event_history_validation import (
    PROMOTION_POLICY,
    REQUIRED_COMPLETION_BLOCKERS,
    _classify_archive_title,
    _event_chain_issues,
    _parse_notice_page,
    _regime_multiplier_issues,
    write_japan_price_limit_event_history_validation,
)


def test_independent_title_classification_and_post_2020_parser() -> None:
    assert _classify_archive_title("Broadening of Daily Price Limits (Preliminary Notice) : Alpha") == "PRELIMINARY"
    assert _classify_archive_title("Daily Price Limits to be Broadened : 1 issue") == "BROADENING"
    assert _classify_archive_title("Daily Price Limits Continue to be Broadened : 1 issue") == "CONTINUATION"
    assert _classify_archive_title("Daily Price Limits to be Broadened (ETF/ETN) : 1 issue") == ""

    rows = _parse_notice_page(
        b"""
        <p>Raysum Co.,Ltd. (Code: 8890) will broaden only the upper daily price limit
        on the next business day (September 19) as follows. Upper daily price limit:
        2,800 yen (Lower limit will remain 700 yen as per normal) Base price: 4,445 yen.</p>
        """,
        publication_date="2024-09-18",
        event_type="BROADENING",
        source_url="https://www.jpx.co.jp/english/news/1030/test.html",
        title="Daily Price Limits to be Broadened : 1 issue",
    )
    assert rows[0]["effective_date"] == "2024-09-19"
    assert rows[0]["side"] == "UPPER"
    assert rows[0]["observed_multiplier"] == 4.0
    assert _regime_multiplier_issues(pd.DataFrame(rows)) == []


def test_event_chain_requires_preliminary_before_pre_2020_broadening() -> None:
    events = pd.DataFrame(
        [
            {
                "publication_date": "2019-01-04",
                "effective_date": "2019-01-07",
                "event_type": "BROADENING",
                "code": "1111",
                "source_url": "https://www.jpx.co.jp/english/news/1030/a.html",
            }
        ]
    )
    issues, _ = _event_chain_issues(events)
    assert any(value.startswith("event_chain_broadening_without_preliminary") for value in issues)


def test_validator_rebuilds_archive_denominator_and_retains_blockers(tmp_path: Path) -> None:
    producer = _write_valid_bundle(tmp_path)
    result = write_japan_price_limit_event_history_validation(
        run_id="VALIDATOR-RUN",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
    )
    assert result.manifest["status"] == "pass_partial_component"
    assert result.manifest["validation_issues"] == []
    assert result.manifest["remaining_completion_blockers"] == list(REQUIRED_COMPLETION_BLOCKERS)
    assert result.manifest["promotion_allowed"] is False
    assert result.manifest["promotion_policy"] == PROMOTION_POLICY
    summary = pd.read_csv(result.artifact_dir / "validation_summary.csv")
    assert summary["passed"].all()
    assert result.manifest["title_family_counts"] == {
        "PRELIMINARY": 1,
        "CANCELLED": 0,
        "BROADENING": 1,
        "CONTINUATION": 0,
        "RETURN_NORMAL": 1,
    }


def test_validator_fails_closed_on_hash_and_numeric_tampering(tmp_path: Path) -> None:
    producer = _write_valid_bundle(tmp_path)
    events = pd.read_csv(producer / "price_limit_events.csv", dtype={"code": str})
    events.loc[events["event_type"].eq("BROADENING"), "observed_multiplier"] = 4.0
    events.to_csv(producer / "price_limit_events.csv", index=False)

    result = write_japan_price_limit_event_history_validation(
        run_id="TAMPER-VALIDATOR-RUN",
        producer_dir=producer,
        artifact_root=tmp_path / "validation",
    )
    assert result.manifest["status"] == "failed_validation"
    assert result.manifest["promotion_allowed"] is False
    assert result.manifest["promotion_policy"] == PROMOTION_POLICY
    assert "producer_output_hash_mismatch:price_limit_events.csv" in result.manifest["validation_issues"]
    assert any(value.startswith("event_multiplier_mismatch") for value in result.manifest["validation_issues"])


def _write_valid_bundle(tmp_path: Path) -> Path:
    archive_dir = tmp_path / "archive"
    producer = tmp_path / "producer"
    raw_pages = producer / "raw" / "notice_pages"
    archive_dir.mkdir()
    raw_pages.mkdir(parents=True)
    notices = [
        (
            "2013-09-05",
            "PRELIMINARY",
            "Expansion of daily price limits (preliminary notice) -Alpha-",
            "detail/detail_101.html",
            b"<p>Alpha (Code: 1111) may expand only its upper daily price limit on the next business day (September 6).</p>",
        ),
        (
            "2013-09-06",
            "BROADENING",
            "Expansion of daily price limits (Daily price limit will be expanded) -Alpha-",
            "detail/detail_102.html",
            b"<p>Alpha (Code: 1111) will expand only the upper daily price limit on the next business day (September 9). Upper daily price limit: 600 yen (Lower limit will remain 300 yen as per normal) Base price: 1,200 yen.</p>",
        ),
        (
            "2013-09-09",
            "RETURN_NORMAL",
            "Removal of expanded daily price limits (Return to normal daily price limits) -Alpha-",
            "detail/detail_103.html",
            b"<p>The upper daily price limit for Alpha (Code: 1111) was expanded to twice the range. TSE will return the daily price limits to normal on September 10.</p>",
        ),
    ]
    archive_items = []
    plan_rows = []
    state_rows = []
    event_rows = []
    for index, (published, family, title, target, body) in enumerate(notices, 1):
        stamp = published.split("-")
        archive_items.append(
            {
                "title": title,
                "url": f"/english/news/{target}",
                "updated_date": {"year": int(stamp[0]), "month": int(stamp[1]), "day": int(stamp[2])},
            }
        )
        row_id = f"JPL-{index:05d}"
        url = f"https://www.jpx.co.jp/english/news/{target}"
        raw = raw_pages / f"{row_id}.html"
        raw.write_bytes(body)
        common = {
            "plan_row_id": row_id,
            "publication_date": published,
            "event_type": family,
            "title": title,
            "url": url,
            "archive_source_ids": "news_ym_201309",
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
        parsed = _parse_notice_page(
            body,
            publication_date=published,
            event_type=family,
            source_url=url,
            title=title,
        )[0]
        event_rows.append({**parsed, "issue_name": "Alpha"})

    archive_raw = archive_dir / "news_ym_201309.json"
    archive_raw.write_text(json.dumps({"items": archive_items}), encoding="utf-8")
    archive_manifest = tmp_path / "archive_source_manifest.csv"
    pd.DataFrame(
        [
            {
                "source_id": "news_ym_201309",
                "url": "https://www.jpx.co.jp/english/news/news_ym_201309.json",
                "http_status": 200,
                "content_type": "application/json",
                "raw_path": str(archive_raw),
                "sha256": _digest(archive_raw),
                "bytes": archive_raw.stat().st_size,
            }
        ]
    ).to_csv(archive_manifest, index=False)
    pd.DataFrame(plan_rows).to_csv(producer / "notice_plan.csv", index=False)
    pd.DataFrame(state_rows).to_csv(producer / "page_state.csv", index=False)
    events = pd.DataFrame(event_rows)[
        [
            "publication_date", "archive_publication_date", "effective_date", "event_type", "code", "issue_name", "side",
            "base_price_yen", "expanded_range_yen", "normal_range_yen", "observed_multiplier",
            "source_url", "title",
        ]
    ]
    events.to_csv(producer / "price_limit_events.csv", index=False)
    quality_rows = []
    for row in events.to_dict("records"):
        applies = row["event_type"] in {"BROADENING", "CONTINUATION"} and pd.notna(row["observed_multiplier"])
        quality_rows.append(
            {
                "publication_date": row["publication_date"],
                "event_type": row["event_type"],
                "code": row["code"],
                "expected_multiplier": 2.0 if applies else None,
                "observed_multiplier": row["observed_multiplier"],
                "regime_multiplier_valid": True,
            }
        )
    pd.DataFrame(quality_rows).to_csv(producer / "event_quality.csv", index=False)
    output_hashes = {
        filename: _digest(producer / filename)
        for filename in ("notice_plan.csv", "page_state.csv", "price_limit_events.csv", "event_quality.csv")
    }
    manifest = {
        "schema_version": "1.0-resumable-partial",
        "run_id": "PRODUCER-RUN",
        "artifact_type": "japan_exceptional_price_limit_event_history_candidate",
        "status": "partial_candidate",
        "component": "temporary_price_limit_events",
        "official_jpx_source": True,
        "research_only": True,
        "scorecard_eligible": False,
        "component_scorecard_eligible": False,
        "independent_validation_required": True,
        "cutoff_date": "2013-09-30",
        "counts": {
            "archive_sources": 1,
            "planned_notice_pages": 3,
            "completed_notice_pages": 3,
            "pending_notice_pages": 0,
            "failed_notice_pages": 0,
            "parsed_event_rows": 3,
            "page_parse_failures": 0,
            "regime_multiplier_mismatches": 0,
        },
        "producer_blockers": [],
        "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "critical_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "archive_manifest": str(archive_manifest),
        "archive_manifest_sha256": _digest(archive_manifest),
        "output_sha256": output_hashes,
        "promotion_policy": PROMOTION_POLICY,
    }
    (producer / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return producer


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
