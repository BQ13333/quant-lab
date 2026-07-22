from __future__ import annotations

import pandas as pd

from quant_lab.data.japan_g4_halt_mapping_recovery import (
    _unique_candidate,
    parse_layout_quotation_pages,
)


def test_layout_parser_accepts_truncated_label_only_with_continuity() -> None:
    page = """年月 (Year/Month) :2016/04
Stock Quotations (4000 - 4999)
4281　デジタル・アドバタイジング・コン                        サービス業
D.A.Consortium Inc.                          Services
  JQS       貸借           100          819.00    (1)         855.00
4282　ＥＰＳホールディングス　普通株式                        サービス業
EPS Holdings,Inc.                            Services
   1        貸借           100        1,401.00
"""
    parsed = parse_layout_quotation_pages(
        [page], observation_month="2016-04", target_codes={"4281"}
    )["4281"]
    assert len(parsed) == 1
    assert not parsed[0].explicit_ordinary_marker
    assert parsed[0].domestic_market_evidence
    assert parsed[0].continuity_evidence
    assert parsed[0].issue_name_english == "D.A.Consortium Inc."


def test_layout_parser_marks_foreign_quote() -> None:
    page = """年月 (Year/Month) :2016/08
Stock Quotations (8000 - 8999)
8685　アメリカン・インターナショナル・                        保険業
American International Group,Inc.            Insurance
  1外        信用              1       5,630.00
"""
    parsed = parse_layout_quotation_pages(
        [page], observation_month="2016-08", target_codes={"8685"}
    )["8685"]
    assert len(parsed) == 1
    assert parsed[0].foreign_marker
    assert not parsed[0].domestic_market_evidence


def test_candidate_requires_exact_code_tyo_ordinary_and_stable_map() -> None:
    security = pd.DataFrame(
        [
            {
                "local_code": "4281",
                "candidate_ric": "4281.T^I16",
                "refinitiv_ric": "",
                "ric": "4281.T^I16",
                "exchange_code": "TYO",
                "exchange_name": "Tokyo Stock Exchange",
                "rcs_asset_category_leaf": "Ordinary Share",
                "is_primary_issue_ric": "True",
                "document_title": "D A Consortium Ord Shs, Ordinary Share, Delisted, Tokyo Stock Exchange",
                "isin": "JP3549030009",
                "instrument_permid": "8589999568",
                "admitted_source_sha256": "a" * 64,
                "admitted_source_path": "/source/security.csv",
                "source_record_id": "row-1",
                "retire_date": "2016-09-28",
            }
        ]
    )
    symbol_map = pd.DataFrame(
        [
            {
                "local_code": "4281",
                "refinitiv_ric": "4281.T^I16",
                "stable_instrument_id": "isin:JP3549030009",
                "exchange_code": "TYO",
            }
        ]
    )
    candidate, reason = _unique_candidate("4281", symbol_map, security)
    assert reason == ""
    assert candidate["candidate_ric"] == "4281.T^I16"
    assert candidate["stable_instrument_id"] == "isin:JP3549030009"


def test_candidate_rejects_successor_ric() -> None:
    security = pd.DataFrame(
        [
            {
                "local_code": "5289",
                "candidate_ric": "5290.T",
                "refinitiv_ric": "5290.T",
                "ric": "5290.T",
                "exchange_code": "TYO",
                "exchange_name": "Tokyo Stock Exchange",
                "rcs_asset_category_leaf": "Ordinary Share",
                "is_primary_issue_ric": "True",
                "document_title": "Successor, Ordinary Share, Tokyo Stock Exchange",
                "isin": "JP0000000001",
                "instrument_permid": "1",
                "admitted_source_sha256": "a" * 64,
                "admitted_source_path": "/source/security.csv",
                "source_record_id": "row-1",
                "retire_date": "",
            }
        ]
    )
    symbol_map = pd.DataFrame(
        [
            {
                "local_code": "5289",
                "refinitiv_ric": "5290.T",
                "stable_instrument_id": "isin:JP0000000001",
                "exchange_code": "TYO",
            }
        ]
    )
    candidate, reason = _unique_candidate("5289", symbol_map, security)
    assert candidate["candidate_count"] == 0
    assert reason == "no_unique_exact_tokyo_ric_candidate"
