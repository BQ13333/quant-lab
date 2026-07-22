from __future__ import annotations

import json
from datetime import date

import pandas as pd

from quant_lab.data.japan_tick_assignment_history import (
    _effective_date,
    _master_snapshot_date,
    _parse_notice_page,
    _walk_news_items,
    archive_urls,
)


def test_master_snapshot_date_is_derived_from_official_rows() -> None:
    master = pd.DataFrame({"Date": ["20260630", "20260630"], "Code": ["1301", "1332"]})
    assert _master_snapshot_date(master) == date(2026, 6, 30)
    assert _master_snapshot_date(pd.DataFrame()) is None


def test_archive_plan_is_bounded_and_complete_to_rolling_window() -> None:
    urls = archive_urls()
    assert len(urls) == 154
    assert urls[0].endswith("news_ym_201309.json")
    assert urls[131].endswith("news_ym_202408.json")
    assert urls[132].endswith("news_ym_01.json")
    assert urls[-1].endswith("news_ym_22.json")
    assert len(urls) == len(set(urls))


def test_news_walker_and_effective_date_handle_compact_month_day() -> None:
    item = {
        "title": "Handling of Tick Sizes from December3 onward",
        "url": "/english/news/example.html",
        "updated_date": {"year": "2020", "month": "11", "day": "30"},
    }
    assert _walk_news_items({"items": [item]}) == [item]
    assert _effective_date(item["title"], "2020-11-30") == "2020-12-03"


def test_notice_parser_preserves_add_remove_scope_and_dates() -> None:
    body = b"""
    <html><body><p>TOPIX500 constituents</p>
    <p>Stocks that will newly be subject to narrow tick sizes</p>
    <table><tr><th>Code</th><th>Name</th></tr><tr><td>285A</td><td>Alpha</td></tr></table>
    <p>Stocks that will no longer be subject to narrow tick sizes</p>
    <table><tr><th>Code</th><th>Name</th></tr><tr><td>8303</td><td>Beta</td></tr></table>
    </body></html>
    """
    item = {
        "title": "Handling of Tick Sizes from September 7 onward",
        "publication_date": "2023-09-05",
        "page_url": "https://www.jpx.co.jp/english/news/test.html",
    }
    rows = _parse_notice_page(body, item)
    assert [(row["action"], row["code"]) for row in rows] == [("ADD", "285A"), ("REMOVE", "8303")]
    assert {row["assignment_scope"] for row in rows} == {"TOPIX500"}
    assert {row["effective_date"] for row in rows} == {"2023-09-07"}
