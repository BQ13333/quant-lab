from __future__ import annotations

from pathlib import Path

import pytest

from quant_lab.data.jpx_public_listing_events import (
    discover_archive_pages,
    parse_delisting_html,
    parse_new_listing_html,
    run_jpx_public_listing_events,
)


DELIST = b"""
<html><select class='backnumber'>
<option value='/english/listing/stocks/delisted/index.html'>2026</option>
<option value='/english/listing/stocks/delisted/archives-01.html'>2025</option>
</select><table><tr><th>Date</th><th>Issue Name</th><th>Code</th><th>Market Segment</th><th>Reason</th></tr>
<tr><td>Oct. 01, 2026</td><td>Example Co.</td><td>1234</td><td>Standard</td><td>Merger</td></tr></table></html>
"""

NEW = b"""
<html><select class='backnumber'>
<option value='/english/listing/stocks/new/index.html'>2026</option>
<option value='/english/listing/stocks/new/00-archives-01.html'>2025</option>
</select><table>
<tr><th rowspan='2'>Date</th><th rowspan='2'>Name</th><th>Code</th><th>Outline</th><th>Confirmation</th><th>Range</th><th>Offering</th><th>Unit</th></tr>
<tr><th>Market</th><th>Report</th><th>CG</th><th>Price</th><th>Distribution</th><th>Earnings</th></tr>
<tr><td rowspan='2'>Aug. 04, 2026 (Jun. 30, 2026)</td><td rowspan='2'>New Co.</td><td>607A</td><td></td><td></td><td>-</td><td>100</td><td>100</td></tr>
<tr><td>Growth</td><td></td><td></td><td>-</td><td>-</td><td>-</td></tr>
</table></html>
"""


def test_archive_discovery_and_event_parsers_preserve_missing_known_at():
    pages = discover_archive_pages(DELIST, "https://www.jpx.co.jp/english/listing/stocks/delisted/index.html", "delisting")
    assert pages["archive_year"].tolist() == ["2026", "2025"]
    delisted = parse_delisting_html(DELIST, pages.iloc[0]["url"], "2026")
    new = parse_new_listing_html(NEW, "https://www.jpx.co.jp/new", "2026")
    assert delisted.iloc[0]["event_date"] == "2026-10-01"
    assert delisted.iloc[0]["known_at_utc"] == ""
    assert new.iloc[0]["listing_date"] == "2026-08-04"
    assert new.iloc[0]["approval_date"] == "2026-06-30"
    assert new.iloc[0]["market_segment"] == "Growth"


def test_parsers_cover_may_dates_multiple_tables_and_legacy_six_cell_rows():
    delisting = parse_delisting_html(
        b"""
        <table><tr><td>May 30, 2024</td><td>Samty Co.,Ltd.</td><td>3244</td><td>Prime</td>
        <td>Becoming a wholly owned subsidiary of SAMTY HOLDINGS Co.,Ltd.</td></tr></table>
        <table><tr><td>Jun. 29, 2024</td><td>Alpha</td><td>1111</td><td>Prime</td><td>Other</td></tr></table>
        """,
        "https://example.test/delist",
        "2024",
    )
    assert delisting[["event_date", "code"]].to_dict("records") == [
        {"event_date": "2024-05-30", "code": "3244"},
        {"event_date": "2024-06-29", "code": "1111"},
    ]
    new = parse_new_listing_html(
        b"""
        <table><tr><th>Date</th></tr>
        <tr><td>Jan. 04, 2022</td><td>Dec. 01, 2021</td><td>FUJIKYU HOLDINGS CO.,LTD. **</td>
        <td>7135</td><td>1st Section</td><td></td></tr></table>
        """,
        "https://example.test/new",
        "2022",
    )
    assert new[["listing_date", "approval_date", "code", "market_segment"]].to_dict("records") == [
        {"listing_date": "2022-01-04", "approval_date": "2021-12-01", "code": "7135", "market_segment": "1st Section"}
    ]


def test_bounded_archive_run_is_complete_but_cannot_claim_2008_pit(tmp_path: Path):
    def fetch(url: str):
        return 200, NEW if "/stocks/new/" in url else DELIST, {"Content-Type": "text/html"}

    result = run_jpx_public_listing_events(
        run_id="PUBLIC",
        artifact_root=tmp_path,
        max_pages=4,
        pacing_seconds=0,
        retries=0,
        fetcher=fetch,
    )
    assert result.manifest["status"] == "complete_limited_coverage"
    assert result.manifest["historical_pit_ready"] is False
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["coverage"]["completed_pages"] == 4
    assert result.manifest["coverage"]["delisting_events"] == 2
    assert result.manifest["coverage"]["new_listing_events"] == 2


def test_raw_hash_tampering_fails_resume_rebuild(tmp_path: Path):
    def fetch(url: str):
        return 200, NEW if "/stocks/new/" in url else DELIST, {}

    result = run_jpx_public_listing_events(
        run_id="TAMPER", artifact_root=tmp_path, max_pages=4, pacing_seconds=0, retries=0, fetcher=fetch
    )
    next((result.artifact_dir / "raw").glob("*.html")).write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        run_jpx_public_listing_events(
            run_id="TAMPER", artifact_root=tmp_path, max_pages=1, pacing_seconds=0, retries=0, fetcher=fetch
        )
