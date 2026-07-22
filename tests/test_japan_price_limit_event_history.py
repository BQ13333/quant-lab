from __future__ import annotations

from quant_lab.data.japan_price_limit_event_history import (
    classify_notice_title,
    event_quality,
    parse_price_limit_notice,
)


def test_title_classifier_separates_event_types_and_exclusions() -> None:
    assert classify_notice_title("Broadening of Daily Price Limits (Preliminary Notice) : Alpha") == "PRELIMINARY"
    assert classify_notice_title("Announced Broadening of Daily Price Limits to be Cancelled: Alpha") == "CANCELLED"
    assert classify_notice_title("Daily Price Limits to be Returned to Normal : Alpha") == "RETURN_NORMAL"
    assert classify_notice_title("Daily Price Limits Continue to be Broadened : 1 issue") == "CONTINUATION"
    assert classify_notice_title("Daily Price Limits to be Broadened : 2 issues") == "BROADENING"
    assert classify_notice_title("Daily Price Limits to be Broadened (ETF/ETN) : 1 issue") == ""
    assert classify_notice_title("Partial Revision to Broadening of Daily Price Limits") == ""


def test_notice_parser_extracts_post_2020_four_times_regime() -> None:
    body = b"""
    <p>The following issue has fallen under the conditions for two consecutive business days.
    Raysum Co.,Ltd. (Code: 8890) will broaden only the upper daily price limit on the next business day (September 19) as follows.
    Upper daily price limit: 2,800 yen (Lower limit will remain 700 yen as per normal)
    Base price: 4,445 yen; Upper daily price limit: 7,250 yen; Lower daily price limit: 3,745 yen</p>
    """
    rows = parse_price_limit_notice(body, event_type="BROADENING", publication_date="2024-09-18", source_url="https://www.jpx.co.jp/test", title="Daily Price Limits to be Broadened : 1 issue")
    assert len(rows) == 1
    assert rows[0]["code"] == "8890"
    assert rows[0]["effective_date"] == "2024-09-19"
    assert rows[0]["side"] == "UPPER"
    assert rows[0]["observed_multiplier"] == 4.0
    assert bool(event_quality(__import__("pandas").DataFrame(rows)).iloc[0]["regime_multiplier_valid"])


def test_notice_parser_scopes_limits_to_each_issue_and_uses_canonical_url_date() -> None:
    body = b"""
    <p>The following 2 issues will be broadened on the next business day (July 24).
    Alpha Corp. (Code: 6557) Upper daily price limit: 600 yen
    (Lower limit will remain 150 yen as per normal) Base price: 746 yen;
    Upper daily price limit: 1,346 yen; Lower daily price limit: 596 yen
    Beta Corp. (Code: 7776) Lower daily price limit: 400 yen
    (Upper limit will remain 100 yen as per normal) Base price: 537 yen;
    Upper daily price limit: 637 yen; Lower daily price limit: 137 yen</p>
    """
    rows = parse_price_limit_notice(
        body,
        event_type="BROADENING",
        publication_date="2024-06-23",
        source_url="https://www.jpx.co.jp/english/news/1030/20240723-01.html",
        title="Daily Price Limits to be Broadened : 2 issues",
    )
    assert [(row["code"], row["side"], row["observed_multiplier"]) for row in rows] == [
        ("6557", "UPPER", 4.0),
        ("7776", "LOWER", 4.0),
    ]
    assert {row["publication_date"] for row in rows} == {"2024-07-23"}
    assert {row["archive_publication_date"] for row in rows} == {"2024-06-23"}
    assert {row["effective_date"] for row in rows} == {"2024-07-24"}
