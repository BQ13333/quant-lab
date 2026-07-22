from __future__ import annotations

import pandas as pd

from quant_lab.data.refinitiv_discovery import (
    build_refinitiv_discovery_symbol_map_input,
    refinitiv_equity_screener_expression,
    screen_refinitiv_asean_l2_universe,
    summarize_refinitiv_discovery_universe,
)


class FakeScreener:
    def __init__(self, expression: str):
        self.expression = expression

    def __iter__(self):
        if '"SG"' in self.expression:
            return iter(["LOW.SI", "HIGH.SI"])
        return iter([])


class FakeDiscovery:
    Screener = FakeScreener


class FakeLsegData:
    discovery = FakeDiscovery()

    def get_data(self, universe, fields):
        rows = []
        for ric in universe:
            rows.append(
                {
                    "Instrument": ric,
                    "Company Common Name": ric.split(".")[0],
                    "Exchange Name": "Singapore Exchange Securities Trading",
                    "Exchange Country Code": "SG",
                    "Country ISO Code of Headquarters": "SG",
                    "Date": "2026-07-06",
                    "Company Market Cap": 10 if ric == "LOW.SI" else 100,
                    "Free Float (Percent)": 50,
                    "Outstanding Shares": 1000,
                    "Average Daily Volume - 30 Days": 100 if ric == "LOW.SI" else 1000,
                    "Average Daily Value Traded - 30 Days": 1000 if ric == "LOW.SI" else 10000,
                    "Price Close": 1.0,
                    "Volume": 100,
                }
            )
        return pd.DataFrame(rows)


def test_refinitiv_equity_screener_expression_filters_exchange_country():
    expression = refinitiv_equity_screener_expression("sg")

    assert 'IN(TR.ExchangeCountryCode,"SG")' in expression
    assert "Equity(active,public,primary)" in expression


def test_screen_refinitiv_asean_l2_universe_ranks_by_traded_value():
    raw, ranked = screen_refinitiv_asean_l2_universe(
        FakeLsegData(),
        countries=("Singapore",),
        max_per_country=1,
        max_screen_results_per_country=10,
        sleep_seconds=0,
    )

    assert raw["refinitiv_ric"].tolist() == ["LOW.SI", "HIGH.SI"]
    assert ranked.iloc[0]["refinitiv_ric"] == "HIGH.SI"
    assert bool(ranked.iloc[0]["l2_target_candidate"]) is True
    assert bool(ranked.iloc[1]["l2_target_candidate"]) is False


def test_refinitiv_discovery_symbol_map_input_and_summary():
    _, ranked = screen_refinitiv_asean_l2_universe(
        FakeLsegData(),
        countries=("Singapore",),
        max_per_country=1,
        sleep_seconds=0,
    )

    symbol_input = build_refinitiv_discovery_symbol_map_input(ranked)
    summary = summarize_refinitiv_discovery_universe(ranked, max_per_country=1)

    assert symbol_input["refinitiv_ric"].tolist() == ["HIGH.SI"]
    assert symbol_input.loc[0, "symbol_map_source"] == "refinitiv_discovery_screener"
    assert summary.loc[summary["country"].eq("Singapore"), "l2_candidate_rows"].iloc[0] == 1
