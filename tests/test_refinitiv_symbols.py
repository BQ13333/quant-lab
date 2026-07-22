from __future__ import annotations

import pandas as pd

from quant_lab.data.refinitiv_symbols import (
    INFERRED_REFINITIV_RIC_STATUS,
    SUPPLIED_REFINITIV_RIC_STATUS,
    add_refinitiv_ric,
    infer_refinitiv_ric,
)


def test_infer_refinitiv_ric_keeps_exchange_suffix_tickers():
    row = {"ticker": "7203", "yahoo_ticker": "7203.T", "country": "Japan", "exchange": "JPX"}

    assert infer_refinitiv_ric(row) == "7203.T"


def test_infer_refinitiv_ric_maps_us_exchanges():
    assert infer_refinitiv_ric({"ticker": "IBM", "country": "United States", "exchange": "NYSE"}) == "IBM.N"
    assert infer_refinitiv_ric({"ticker": "MSFT", "country": "United States", "exchange": "NASDAQ Global Select"}) == "MSFT.O"
    assert infer_refinitiv_ric({"ticker": "AA", "country": "United States", "exchange": "NYSE American"}) == "AA.A"


def test_add_refinitiv_ric_handles_empty_and_nonempty():
    empty = add_refinitiv_ric(pd.DataFrame())
    assert "refinitiv_ric" in empty.columns

    listings = pd.DataFrame(
        {
            "ticker": ["IBM", "7203"],
            "yahoo_ticker": ["IBM", "7203.T"],
            "country": ["United States", "Japan"],
            "exchange": ["NYSE", "JPX"],
        }
    )
    result = add_refinitiv_ric(listings)

    assert result["refinitiv_ric"].tolist() == ["IBM.N", "7203.T"]
    assert result["refinitiv_ric_status"].tolist() == [
        INFERRED_REFINITIV_RIC_STATUS,
        INFERRED_REFINITIV_RIC_STATUS,
    ]


def test_add_refinitiv_ric_preserves_supplied_values_and_status():
    listings = pd.DataFrame(
        {
            "ticker": ["MAYBANK"],
            "yahoo_ticker": ["1155.KL"],
            "country": ["Malaysia"],
            "exchange": ["Bursa Malaysia"],
            "refinitiv_ric": ["MBBM.KL"],
        }
    )

    result = add_refinitiv_ric(listings)

    assert result.loc[0, "refinitiv_ric"] == "MBBM.KL"
    assert result.loc[0, "refinitiv_ric_status"] == SUPPLIED_REFINITIV_RIC_STATUS
