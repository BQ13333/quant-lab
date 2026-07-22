from __future__ import annotations

import pandas as pd

from quant_lab.data.asean_universe import default_asean_universe
from quant_lab.data.asean_l2 import (
    ASEAN_L2_COUNTRIES,
    build_refinitiv_l2_symbol_map_input,
    rank_asean_l2_public_candidates,
    summarize_asean_l2_country_targets,
    write_asean_l2_plan_artifacts,
)


def test_rank_asean_l2_public_candidates_prefers_market_cap_then_share_proxy():
    listings = pd.DataFrame(
        [
            {
                "country": "Singapore",
                "exchange": "SGX",
                "ticker": "AAA",
                "yahoo_ticker": "AAA.SI",
                "name": "AAA",
                "is_stock_like": True,
                "security_type": "Common Stock",
                "market_cap": 10,
            },
            {
                "country": "Singapore",
                "exchange": "SGX",
                "ticker": "BBB",
                "yahoo_ticker": "BBB.SI",
                "name": "BBB",
                "is_stock_like": True,
                "security_type": "Common Stock",
                "market_cap": 50,
            },
            {
                "country": "Philippines",
                "exchange": "PSE",
                "ticker": "CCC",
                "yahoo_ticker": "CCC.PS",
                "name": "CCC",
                "is_stock_like": True,
                "security_type": "Common Stock",
                "listed_shares": 1000,
            },
            {
                "country": "Vietnam",
                "exchange": "HNX",
                "ticker": "ETF1",
                "yahoo_ticker": "ETF1.VN",
                "name": "ETF",
                "is_stock_like": True,
                "security_type": "ETF",
                "listed_shares": 999999,
            },
        ]
    )

    ranked = rank_asean_l2_public_candidates(listings, target_per_country=1)

    singapore = ranked[ranked["country"].eq("Singapore")]
    assert singapore.iloc[0]["ticker"] == "BBB"
    assert singapore.iloc[0]["l2_rank_method"] == "market_cap"
    assert "Philippines" not in set(ranked["country"])
    assert "Vietnam" not in set(ranked["country"])
    assert "ETF1" not in set(ranked["ticker"])
    assert int(ranked["l2_target_candidate"].sum()) == 1


def test_default_asean_l2_countries_are_three_tradable_markets():
    assert ASEAN_L2_COUNTRIES == ("Malaysia", "Singapore", "Indonesia")


def test_default_asean_universe_has_only_three_country_scope():
    universe = default_asean_universe()
    countries = {instrument.country for instrument in universe}
    tickers = {instrument.ticker for instrument in universe}

    assert countries == {"Malaysia", "Singapore", "Indonesia"}
    assert {"ASEA", "THD", "EPHE", "VNM", "PSEI.PS", "^SET.BK", "^VNINDEX.VN"}.isdisjoint(tickers)


def test_rank_asean_l2_public_candidates_omits_philippines_and_vietnam():
    listings = pd.DataFrame(
        [
            {
                "country": "Philippines",
                "exchange": "PSE",
                "ticker": "PPP",
                "yahoo_ticker": "PPP.PS",
                "name": "PPP",
                "is_stock_like": True,
                "security_type": "Common Stock",
                "listed_shares": 1000,
            },
            {
                "country": "Vietnam",
                "exchange": "HNX",
                "ticker": "VVV",
                "yahoo_ticker": "VVV.VN",
                "name": "VVV",
                "is_stock_like": True,
                "security_type": "Common Stock",
                "listed_shares": 1000,
            },
        ]
    )

    ranked = rank_asean_l2_public_candidates(listings, target_per_country=1)

    assert ranked.empty


def test_summarize_asean_l2_country_targets_reports_public_and_price_shortfalls():
    ranked = pd.DataFrame(
        [
            {"country": "Singapore", "exchange": "SGX", "ticker": "A", "l2_target_candidate": True},
            {"country": "Singapore", "exchange": "SGX", "ticker": "B", "l2_target_candidate": True},
            {"country": "Indonesia", "exchange": "IDX", "ticker": "C", "l2_target_candidate": True},
        ]
    )
    symbol_map = pd.DataFrame(
        [
            {
                "country": "Singapore",
                "refinitiv_ric": "A.SI",
                "refinitiv_symbol_status": "validated_price_history",
            }
        ]
    )

    summary = summarize_asean_l2_country_targets(ranked, symbol_map=symbol_map, target_per_country=2)
    singapore = summary[summary["country"].eq("Singapore")].iloc[0]
    malaysia = summary[summary["country"].eq("Malaysia")].iloc[0]

    assert singapore["public_l2_candidate_rows"] == 2
    assert singapore["validated_refinitiv_price_symbols"] == 1
    assert singapore["price_symbol_shortfall"] == 1
    assert singapore["readiness"] == "security_master_ready_validate_refinitiv"
    assert malaysia["readiness"] == "vendor_security_master_required"


def test_build_refinitiv_l2_symbol_map_input_adds_asean_curated_seeds():
    ranked = pd.DataFrame(
        [
            {
                "country": "Indonesia",
                "exchange": "IDX",
                "ticker": "BBCA",
                "yahoo_ticker": "BBCA.JK",
                "name": "Bank Central Asia",
                "l2_target_candidate": True,
            }
        ]
    )

    refinitiv_input = build_refinitiv_l2_symbol_map_input(ranked)

    assert "BBCA" in set(refinitiv_input["ticker"])
    assert set(refinitiv_input["country"]).issubset(set(ASEAN_L2_COUNTRIES))
    assert {"Malaysia", "Indonesia", "Singapore"}.issubset(set(refinitiv_input["country"]))
    assert "Philippines" not in set(refinitiv_input["country"])
    assert "Vietnam" not in set(refinitiv_input["country"])
    assert "Thailand" not in set(refinitiv_input["country"])
    assert "China" not in set(refinitiv_input["country"])


def test_write_asean_l2_plan_artifacts_writes_expected_files(tmp_path):
    listings = pd.DataFrame(
        [
            {
                "country": "Singapore",
                "exchange": "SGX",
                "ticker": "D05",
                "yahoo_ticker": "D05.SI",
                "name": "DBS",
                "is_stock_like": True,
                "security_type": "Common Stock",
                "market_cap": 100,
            }
        ]
    )

    artifact = write_asean_l2_plan_artifacts(
        public_listings=listings,
        artifact_root=tmp_path,
        run_id="ASEAN-L2-TEST",
        target_per_country=1,
    )

    assert artifact.country_targets_path.exists()
    assert artifact.refinitiv_symbol_map_input_path.exists()
    assert artifact.vendor_requests_path.exists()
    assert artifact.data_requirements_path.exists()
    assert artifact.markdown_path.exists()
    assert artifact.country_targets["country"].tolist() == list(ASEAN_L2_COUNTRIES)
    assert "Philippines" not in set(artifact.country_targets["country"])
    assert "Vietnam" not in set(artifact.country_targets["country"])
