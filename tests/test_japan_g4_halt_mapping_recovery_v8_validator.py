from __future__ import annotations

import pandas as pd

from quant_lab.data.japan_g4_halt_mapping_recovery_v8_validator import (
    _validate_denominator,
    independently_reparse_plain_pages_v8,
)


def test_independent_v8_denominator_recomputes_exact_event_ids() -> None:
    decisions = pd.DataFrame(
        [{"event_id": "event-1", "decision": "accepted"}]
    )
    accepted = decisions.copy()
    rejected = pd.DataFrame(columns=["event_id", "decision"])
    completion = pd.DataFrame(columns=["event_id"])
    declared = pd.DataFrame([{"event_id": "event-1"}])
    context = {
        "report_event_admission": pd.DataFrame(
            [
                {
                    "channel": "halt_suspension_resumption",
                    "instrument_class": "equity",
                    "normalization_status": "normalized",
                    "in_research_scope": "True",
                    "mapping_complete": "False",
                    "primary_event_id": "event-1",
                }
            ]
        ),
        "halt_events": pd.DataFrame([{"event_id": "event-1"}]),
        "mapping_audit": pd.DataFrame(
            [
                {
                    "event_id": "event-1",
                    "mapping_status": "missing_admitted_mapping",
                    "refinitiv_ric": "",
                }
            ]
        ),
    }
    issues: list[dict[str, str]] = []
    checks = _validate_denominator(
        decisions,
        accepted,
        rejected,
        completion,
        declared,
        context,
        issues,
    )
    assert all(row["passed"] for row in checks)
    assert not issues


def test_v8_plain_parser_accepts_ascii_code_separator() -> None:
    page = """年月 (Year/Month) :2013/08
Stock Quotations (3000 - 3999)
3350 フォンツ・ホールディングス 普通 卸売業
FONTZ Holdings, Inc. Wholesale Trade
JQS 貸借 100 108 (1) 119 (6) 80 (30)
3352 バッファロー 普通株式 小売業
BUFFALO CO.LTD Retail Trade
"""
    result = independently_reparse_plain_pages_v8(
        [page], observation_month="2013-08", code="3350"
    )
    assert result["record_count"] == 1
    assert result["unique_record"]["domestic_continuity"]


def test_v8_plain_parser_accepts_split_market_detail() -> None:
    page = """年月 (Year/Month) :2014/04
Stock Quotations (4000 - 4999)
4798　エル・シー・エーホールディングス
LCA Holdings Corporation
サービス業
Services
2
特
信用 100 4 (1) 4 (1) 3 (1) 4 (30)
4799　アグレックス　普通株式
AGREX INC.
"""
    result = independently_reparse_plain_pages_v8(
        [page], observation_month="2014-04", code="4798"
    )
    assert result["record_count"] == 1
    assert result["unique_record"]["domestic_continuity"]
