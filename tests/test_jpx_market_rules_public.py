from __future__ import annotations

from quant_lab.data.jpx_market_rules_public import (
    SOURCES,
    parse_halts,
    parse_market_holidays,
    parse_price_limits,
    parse_tick_sizes,
    parse_trading_unit_policy,
    run_jpx_market_rules_public,
)


CALENDAR = b"<table><tr><td>Jan. 1 (Thu.)</td><td>New Year's Day</td></tr></table><table><tr><td>Jan. 1 (Fri.)</td><td>New Year's Day</td></tr></table>"
UNIT = b"<p>Now trading units are 100 shares for domestic stocks. normalized to 100 shares on Oct. 1, 2018.</p>"
LIMITS = b"<table><tr><td>x</td></tr></table><table><tr><td>x</td></tr></table><table><tr><td>Header</td></tr><tr><td>Less than</td><td>100</td><td>30</td></tr></table>"
TICKS = b"".join(
    b"<table><tr><td>Header</td></tr><tr><td>up to</td><td>1000</td><td>1</td></tr></table>"
    for _ in range(3)
)
HALTS = b"""
<table><tr><th>Header</th></tr><tr><td>Current Co.</td><td>1234</td><td>Jul. 10, 2026</td><td>15:30</td><td>15:45</td><td>15:46</td><td>Merger</td></tr></table>
<table><tr><th>Header</th></tr><tr><td>Old Co.</td><td>5678</td><td>Jun. 01, 2026</td><td>10:00</td><td>10:15</td><td>10:16</td><td>Tender offer</td></tr><tr><td>May Co.</td><td>9012</td><td rowspan="2">May 01, 2026</td><td rowspan="2">11:00</td><td rowspan="2">11:15</td><td rowspan="2">11:16</td><td rowspan="2">News</td></tr><tr><td>May Peer Co.</td><td>3456</td></tr></table>
"""


def test_rule_parsers_preserve_official_values_and_scope():
    assert len(parse_market_holidays(CALENDAR)) == 2
    assert parse_trading_unit_policy(UNIT)["standardization_effective_date"] == "2018-10-01"
    assert parse_price_limits(LIMITS).iloc[0]["daily_limit_yen"] == "30"
    assert set(parse_tick_sizes(TICKS)["category"]) == {"topix500", "etf_unit_one", "other_issues"}
    halts = parse_halts(HALTS)
    assert len(halts) == 4
    assert set(halts["date"]) == {"2026-05-01", "2026-06-01", "2026-07-10"}
    assert set(halts["scope"]) == {"current", "archive"}


def test_market_rule_artifact_is_honestly_incomplete_without_history_and_tca(tmp_path):
    payloads = {
        SOURCES["calendar"]: CALENDAR,
        SOURCES["trading_unit"]: UNIT,
        SOURCES["price_limits"]: LIMITS,
        SOURCES["tick_sizes"]: TICKS,
        SOURCES["halts"]: HALTS,
    }

    def fetch(url: str):
        return 200, payloads[url], {"Content-Type": "text/html"}

    result = run_jpx_market_rules_public(
        run_id="RULES", artifact_root=tmp_path, fetcher=fetch
    )
    assert result.manifest["status"] == "incomplete"
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["official_jpx_calendar"] is False
    assert result.manifest["tca_calibrated"] is False
    assert result.manifest["counts"]["halt_rows"] == 4
    assert result.manifest["counts"]["halt_source_rows"] == 4
    assert len(result.manifest["critical_blockers"]) == 5
