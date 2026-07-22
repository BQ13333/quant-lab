from __future__ import annotations

import pandas as pd

from quant_lab.data.refinitiv_symbol_map import (
    build_refinitiv_symbol_candidates,
    candidate_refinitiv_rics,
    default_refinitiv_symbol_seed_listings,
    sample_symbol_map_listings,
    select_validated_refinitiv_symbol_map,
    validate_refinitiv_symbol_candidates,
)


class FakeLsegData:
    def __init__(self, by_ric_field: dict[tuple[str, str], pd.DataFrame]):
        self.by_ric_field = by_ric_field
        self.calls: list[dict] = []

    def get_history(self, **kwargs):
        self.calls.append(kwargs)
        ric = kwargs["universe"][0]
        field = kwargs["fields"][0]
        if (ric, field) not in self.by_ric_field:
            raise RuntimeError("not found")
        return self.by_ric_field[(ric, field)]


def _candidate_rics(row: dict[str, object]) -> list[str]:
    return [candidate["refinitiv_ric"] for candidate in candidate_refinitiv_rics(row)]


def test_candidate_refinitiv_rics_handles_us_class_shares():
    row = {"country": "United States", "exchange": "NYSE", "ticker": "BRK.B", "name": "Berkshire Hathaway"}

    assert _candidate_rics(row)[0] == "BRKb.N"


def test_candidate_refinitiv_rics_handles_hk_numeric_padding():
    row = {"country": "Hong Kong", "exchange": "HKEX", "ticker": "00700", "name": "Tencent"}

    assert _candidate_rics(row)[0] == "0700.HK"


def test_candidate_refinitiv_rics_handles_vietnam_exchange_suffixes():
    hnx = {"country": "Vietnam", "exchange": "HNX", "ticker": "PVS", "yahoo_ticker": "PVS.VN"}
    hose = {"country": "Vietnam", "exchange": "HOSE", "ticker": "VIC", "yahoo_ticker": "VIC.VN"}

    assert _candidate_rics(hnx)[:2] == ["PVS.HN", "PVS.VN"]
    assert _candidate_rics(hose)[:2] == ["VIC.HM", "VIC.VN"]


def test_candidate_refinitiv_rics_prefers_curated_malaysia_and_singapore_aliases():
    malaysia = {"country": "Malaysia", "exchange": "Bursa Malaysia", "ticker": "1155", "yahoo_ticker": "1155.KL"}
    singapore = {"country": "Singapore", "exchange": "SGX", "ticker": "D05", "yahoo_ticker": "D05.SI"}

    assert _candidate_rics(malaysia)[:2] == ["MBBM.KL", "1155.KL"]
    assert _candidate_rics(singapore)[:2] == ["DBSM.SI", "D05.SI"]


def test_sample_symbol_map_listings_keeps_curated_seeds():
    listings = pd.DataFrame(
        {
            "country": ["Japan", "Japan"],
            "exchange": ["JPX", "JPX"],
            "ticker": ["7203", "6758"],
            "yahoo_ticker": ["7203.T", "6758.T"],
            "name": ["Toyota", "Sony"],
        }
    )

    sampled = sample_symbol_map_listings(listings, sample_per_market=1, max_symbols=20)

    assert "MBBM.KL" in set(sampled["refinitiv_ric"].dropna())
    assert sampled[sampled["country"].eq("Japan") & sampled["symbol_map_source"].ne("curated_seed")].shape[0] == 1


def test_build_candidates_from_default_seed_rows():
    seeds = default_refinitiv_symbol_seed_listings()
    candidates = build_refinitiv_symbol_candidates(seeds)

    assert {"MBBM.KL", "DBSM.SI", "BBRI.JK"}.issubset(set(candidates["refinitiv_ric"]))
    assert {"BDO.PS", "VIC.HM", "PVS.HN", "PTT.BK"}.isdisjoint(set(candidates["refinitiv_ric"]))
    assert {"600000.SS", "000001.SZ", "2330.TW"}.isdisjoint(
        set(candidates["refinitiv_ric"])
    )


def test_validate_and_select_symbol_map_uses_first_valid_candidate():
    dates = pd.date_range("2024-01-01", periods=2)
    candidates = pd.DataFrame(
        [
            {
                "source_symbol_id": "Malaysia|Bursa Malaysia|1155|Maybank",
                "country": "Malaysia",
                "exchange": "Bursa Malaysia",
                "ticker": "1155",
                "name": "Maybank",
                "refinitiv_ric": "MBBM.KL",
                "candidate_rank": 1,
                "candidate_method": "curated_alias",
                "symbol_map_source": "listing",
            },
            {
                "source_symbol_id": "Malaysia|Bursa Malaysia|1155|Maybank",
                "country": "Malaysia",
                "exchange": "Bursa Malaysia",
                "ticker": "1155",
                "name": "Maybank",
                "refinitiv_ric": "1155.KL",
                "candidate_rank": 2,
                "candidate_method": "country_suffix",
                "symbol_map_source": "listing",
            },
        ]
    )
    fake = FakeLsegData(
        {
            ("MBBM.KL", "TRDPRC_1"): pd.DataFrame({"TRDPRC_1": [9.0, 9.1]}, index=dates),
        }
    )

    validations = validate_refinitiv_symbol_candidates(fake, candidates, sleep_seconds=0, progress_every=0)
    symbol_map = select_validated_refinitiv_symbol_map(validations)

    assert symbol_map.loc[0, "refinitiv_ric"] == "MBBM.KL"
    assert symbol_map.loc[0, "refinitiv_symbol_status"] == "validated_price_history"
