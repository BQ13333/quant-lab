from __future__ import annotations

from quant_lab.data.japan_exchange_calendar import extract_official_month_session_count


def test_extracts_legacy_official_header_count() -> None:
    value, method = extract_official_month_session_count(
        "Stock Quotations 立会日数(Trading Days)：20 Page: 1", "2006-09"
    )
    assert value == 20
    assert method == "official_header_trading_days"


def test_extracts_modern_max_security_days_traded() -> None:
    text = "\n".join(
        [
            "2022/06 1301 Company ordinary stock P 100 1 2 3 4 22",
            "2022/06 1302 Company ordinary stock P 100 1 2 3 4 19",
            "2022/07 1303 Company ordinary stock P 100 1 2 3 4 20",
        ]
    )
    value, method = extract_official_month_session_count(text, "2022-06")
    assert value == 22
    assert method == "official_max_security_days_traded"
