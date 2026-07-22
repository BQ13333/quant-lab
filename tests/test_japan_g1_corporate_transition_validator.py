from __future__ import annotations

import pandas as pd

from quant_lab.data.japan_g1_corporate_transition_validator import (
    _canonical_name,
    _english_dates,
    _alias_start_after_direct_route,
    _reason_target,
    _text_mentions_iso_date,
    independently_validate_transition_rows,
)


def test_clean_room_normalization_is_exact_and_not_fuzzy() -> None:
    assert _canonical_name("Hirogin Holdings,Inc.stocks") == "HIROGINHOLDINGSINC"
    assert _canonical_name("Hirogin Holdings, Inc.") == "HIROGINHOLDINGSINC"
    assert _canonical_name("Hirogin Group") != _canonical_name("Hirogin Holdings")
    assert _reason_target(
        "Becoming a wholly owned subsidiary of Hirogin Holdings,Inc. (Stock Transfer)"
    ) == "Hirogin Holdings,Inc."
    assert _english_dates("Sep. 29, 2021 and October 1, 2021") == {
        "2021-09-29",
        "2021-10-01",
    }
    assert _text_mentions_iso_date(
        "6 stocks are scheduled for October 3, 2022 (1 stock on October 1)",
        "2022-10-01",
    )


def test_transition_rows_fail_closed_on_name_or_gap_mismatch() -> None:
    source_url = "https://www.jpx.co.jp/english/news/1030/20210929-01.html"
    old_url = "https://www.jpx.co.jp/english/listing/stocks/delisted/example.html"
    row = {
        "evidence_class": "public_event_only",
        "identity_row_id": "",
        "old_code": "7688",
        "old_issue_name": "Miahelsa Corporation",
        "old_event_date": "2021-09-29",
        "old_event_reason": "Becoming a wholly owned subsidiary of Miahelsa Holdings Corporation",
        "old_event_source_url": old_url,
        "successor_code": "7129",
        "successor_issue_name": "Miahelsa Holdings Corporation",
        "successor_primary_rics": "",
        "official_listing_date": "2021-10-01",
        "listing_source_url": source_url,
        "gap_days": "2",
    }
    delistings = pd.DataFrame(
        [{
            "code": row["old_code"],
            "event_date": row["old_event_date"],
            "issue_name": row["old_issue_name"],
            "reason": row["old_event_reason"],
            "source_url": old_url,
        }]
    )
    source = {
        source_url: "Miahelsa Holdings Corporation stocks (Code: 7129) initial listing October 1, 2021"
    }
    valid = independently_validate_transition_rows(
        pd.DataFrame([row]), delistings=delistings, source_text_by_url=source
    )
    assert valid.iloc[0]["validation_pass"]

    broken = dict(row, successor_issue_name="Unrelated Holdings", gap_days="3")
    invalid = independently_validate_transition_rows(
        pd.DataFrame([broken]), delistings=delistings, source_text_by_url=source
    )
    assert not invalid.iloc[0]["validation_pass"]
    assert "transition_gap_not_exactly_2_to_6_days" in invalid.iloc[0]["validation_issue"]
    assert "successor_name_not_exactly_supported" in invalid.iloc[0]["validation_issue"]


def test_transition_alias_starts_after_validated_direct_route_retires() -> None:
    direct = pd.DataFrame(
        [{"mapping_local_code_override": "5912", "retire_date": "2013-07-13"}]
    )
    assert _alias_start_after_direct_route(direct, "5912") == "2013-07-14"
    assert _alias_start_after_direct_route(direct, "1805") == ""
