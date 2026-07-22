from __future__ import annotations

import pandas as pd

from quant_lab.data.refinitiv_field_probe import (
    build_refinitiv_field_catalog,
    build_refinitiv_ric_validation_manifest,
    default_refinitiv_probe_samples,
    recommend_refinitiv_field_map,
    sample_refinitiv_universe,
    summarize_refinitiv_field_probe,
)


def test_default_refinitiv_probe_samples_cover_core_markets():
    samples = default_refinitiv_probe_samples()

    assert {"United States", "Japan", "Hong Kong", "Malaysia"}.issubset(set(samples["country"]))
    assert {"MBBM.KL", "DBSM.SI", "BBRI.JK"}.issubset(set(samples["refinitiv_ric"]))
    assert {"Thailand", "Philippines", "Vietnam", "China", "Taiwan"}.isdisjoint(
        set(samples["country"])
    )
    assert samples["refinitiv_ric"].notna().all()


def test_sample_refinitiv_universe_keeps_defaults_and_listing_samples():
    listings = pd.DataFrame(
        {
            "country": ["Indonesia", "Indonesia"],
            "exchange": ["IDX", "IDX"],
            "ticker": ["BBCA.JK", "BMRI.JK"],
            "refinitiv_ric": ["BBCA.JK", "BMRI.JK"],
        }
    )

    samples = sample_refinitiv_universe(listings, sample_per_market=1)

    assert {"BBCA.JK", "BMRI.JK"} & set(samples["refinitiv_ric"])
    assert "IBM.N" in set(samples["refinitiv_ric"])


def test_sample_refinitiv_universe_can_cap_after_defaults():
    listings = pd.DataFrame(
        {
            "country": ["Indonesia", "Indonesia"],
            "exchange": ["IDX", "IDX"],
            "ticker": ["BBCA.JK", "BMRI.JK"],
            "refinitiv_ric": ["BBCA.JK", "BMRI.JK"],
        }
    )

    samples = sample_refinitiv_universe(listings, sample_per_market=1, max_samples=3)

    assert len(samples) == 3
    assert "IBM.N" in set(samples["refinitiv_ric"])


def test_summarize_and_recommend_refinitiv_field_probe():
    results = pd.DataFrame(
        [
            {
                "country": "Hong Kong",
                "exchange": "HKEX",
                "role": "Close",
                "field": "TRDPRC_1",
                "refinitiv_ric": "0700.HK",
                "coverage": 0.0,
                "non_null_values": 0,
                "error": "",
            },
            {
                "country": "Hong Kong",
                "exchange": "HKEX",
                "role": "Close",
                "field": "TR.PriceClose",
                "refinitiv_ric": "0700.HK",
                "coverage": 1.0,
                "non_null_values": 10,
                "error": "",
            },
        ]
    )

    summary = summarize_refinitiv_field_probe(results)
    recommended = recommend_refinitiv_field_map(summary)

    assert recommended.loc[0, "field"] == "TR.PriceClose"
    assert recommended.loc[0, "status"] == "recommended"


def test_field_probe_builds_validation_manifest_and_catalog():
    results = pd.DataFrame(
        [
            {
                "country": "Malaysia",
                "exchange": "Bursa Malaysia",
                "ticker": "MAYBANK",
                "name": "Maybank",
                "refinitiv_ric": "MBBM.KL",
                "role": "Close",
                "field": "TRDPRC_1",
                "interval": "1D",
                "count": 5,
                "coverage": 1.0,
                "non_null_values": 5,
                "error": "",
                "first_available": pd.Timestamp("2024-01-01"),
                "last_available": pd.Timestamp("2024-01-05"),
            },
            {
                "country": "Malaysia",
                "exchange": "Bursa Malaysia",
                "ticker": "1155",
                "name": "Maybank",
                "refinitiv_ric": "1155.KL",
                "role": "Close",
                "field": "TRDPRC_1",
                "interval": "1D",
                "count": 5,
                "coverage": 0.0,
                "non_null_values": 0,
                "error": "LDError: universe is not found",
                "first_available": pd.NaT,
                "last_available": pd.NaT,
            },
        ]
    )

    validation = build_refinitiv_ric_validation_manifest(results)
    summary = summarize_refinitiv_field_probe(results)
    recommended = recommend_refinitiv_field_map(summary)
    catalog = build_refinitiv_field_catalog(recommended)

    statuses = dict(zip(validation["refinitiv_ric"], validation["status"], strict=True))
    assert statuses["MBBM.KL"] == "validated_price_history"
    assert statuses["1155.KL"] == "no_history_for_candidate_ric"
    assert catalog.loc[0, "field"] == "TRDPRC_1"
