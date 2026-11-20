from __future__ import annotations

import csv
from pathlib import Path

import pytest

from quant_lab.programme.cost_coverage import historical_cost_gate_met, load_cost_coverage
from quant_lab.programme.hk_lots import HkLotUnavailable, lot_size, lot_table_gate_met
from quant_lab.programme.scope import MarketScopeError, load_market_scope

ROOT = Path(__file__).resolve().parents[1]


def test_scope_is_malaysia_china_a_shares_and_hong_kong() -> None:
    scope = load_market_scope()
    assert scope.effective_date == "2026-11-20"
    assert scope.decided_by == "Benjamin Quek"
    assert [(entry.country, entry.exchanges) for entry in scope.active] == [
        ("Malaysia", ("Bursa Malaysia",)),
        ("China", ("SSE", "SZSE")),
        ("Hong Kong", ("HKEX",)),
    ]
    assert scope.require_active("China", "SSE").boards == ("A-share",)
    assert {entry.country for entry in scope.frozen} == {
        "Singapore",
        "Indonesia",
        "Japan",
        "South Korea",
        "United States",
    }
    assert ("China", ("BSE",)) in {(entry.country, entry.exchanges) for entry in scope.excluded}
    assert scope.policy["live_trading"] is False
    assert scope.policy["paid_feeds"] is False
    assert scope.policy["fundamentals"] == "official_public_filings_only"
    assert scope.policy["ziplime"] == "research_sandbox_not_a_dependency"


@pytest.mark.parametrize(
    ("country", "exchange"),
    [
        ("China", "BSE"),
        ("Japan", "JPX"),
        ("United States", "NYSE"),
        ("United States", "NASDAQ"),
        ("Singapore", "SGX"),
        ("Indonesia", "IDX"),
        ("South Korea", "KRX"),
        ("Taiwan", "TWSE"),
    ],
)
def test_frozen_and_excluded_markets_fail_closed(country: str, exchange: str) -> None:
    with pytest.raises(MarketScopeError):
        load_market_scope().require_active(country, exchange)


def test_quant_lab_does_not_depend_on_ziplime_or_a_broker_sdk() -> None:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "ziplime" not in text
    assert "lime-trader" not in text
    assert "lime_trader" not in text


def test_decision_record_names_the_held_prs_and_the_open_items() -> None:
    text = (ROOT / "docs" / "DECISIONS_2026-11-20.md").read_text(encoding="utf-8")
    for phrase in (
        "outbox/box",
        "a5b153ca614c45a468407745f91572342927163c",
        "tests/test_jpx_monthly_quotations.py::test_runner_is_bounded_resumable_and_partitioned",
        "human root-account holder",
        "separate GO",
        "GPL-3.0",
        "lime-trader-sdk",
        "docs/cloud/CLOUD_PRODUCTION_BLUEPRINT.md",
        "/workspace/quant_house/_archive_not_in_use/",
        "HKEXnews",
        "CNINFO",
        "Bursa",
    ):
        assert phrase in text
    assert (ROOT / "docs" / "cloud" / "architecture.png").stat().st_size > 1000
    summary = (ROOT / "docs" / "cloud" / "EXEC_SUMMARY.md").read_text(encoding="utf-8")
    assert "Parquet + DuckDB" in summary
    blueprint = (ROOT / "docs" / "cloud" / "CLOUD_PRODUCTION_BLUEPRINT.md").read_text(encoding="utf-8")
    assert "MY/CN/HK" in blueprint


def test_historical_cost_gate_is_not_met_and_the_hk_lot_is_null() -> None:
    coverage = load_cost_coverage()
    assert historical_cost_gate_met() is False
    assert coverage["certified"] is False
    hk_lot = [row for row in coverage["markets"]["HK"] if row["component"] == "board_lot"]
    assert hk_lot == [
        {
            "component": "board_lot",
            "value": None,
            "effective_from": "2026-11-16",
            "effective_until": None,
            "source_kind": "NONE",
            "source_url": "",
        }
    ]
    assert "BSE" not in coverage["markets"]


def test_hk_lot_table_has_no_invented_rows() -> None:
    table = ROOT / "configs" / "hk_board_lots.csv"
    with table.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows == []
    assert lot_table_gate_met() is False
    with pytest.raises(HkLotUnavailable):
        lot_size("700.HK", "2024-01-02")


def test_a_sourced_lot_row_is_readable_and_a_default_is_not(tmp_path: Path) -> None:
    table = tmp_path / "lots.csv"
    table.write_text(
        "sid,exchange,instrument_code,effective_from,effective_until,known_at,lot_size,source_url,source_kind\n"
        "700.HK,HKEX,00700,2024-01-01,,2024-01-01T00:00:00Z,100,https://www.hkexnews.hk/example,PRIMARY\n",
        encoding="utf-8",
    )
    assert lot_size("700.HK", "2024-06-01", table=table) == 100
    with pytest.raises(HkLotUnavailable):
        lot_size("5.HK", "2024-06-01", table=table)
