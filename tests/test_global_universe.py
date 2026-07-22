from __future__ import annotations

from quant_lab.data.global_universe import (
    default_global_etf_universe,
    global_universe_metadata,
    tradable_global_tickers,
)


def test_default_global_etf_universe_is_unique_and_usd():
    universe = default_global_etf_universe()
    tickers = [instrument.ticker for instrument in universe]

    assert "ACWI" in tickers
    assert "THD" not in tickers
    assert "EPHE" not in tickers
    assert "VNM" not in tickers
    assert {"EWT", "MCHI", "FXI"}.isdisjoint(tickers)
    assert len(tickers) == len(set(tickers))
    assert all(instrument.currency == "USD" for instrument in universe)


def test_global_universe_metadata_and_tradables():
    metadata = global_universe_metadata()
    tradable = tradable_global_tickers()

    assert {"ticker", "region", "country", "asset_class", "tier"}.issubset(metadata.columns)
    assert "ACWI" not in tradable
    assert set(tradable).issubset(set(metadata["ticker"]))
