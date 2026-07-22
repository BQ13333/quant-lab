from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


GlobalTier = Literal["benchmark", "regional_etf", "country_etf", "style_etf", "sector_etf"]


@dataclass(frozen=True)
class GlobalInstrument:
    """USD-traded global market proxy.

    This universe is deliberately ETF-first. It is a clean first pass for
    global market research because prices are USD-denominated and tradable
    through most brokers, avoiding local FX and constituent-history claims.
    """

    ticker: str
    name: str
    region: str
    country: str
    currency: str
    exchange: str
    asset_class: str
    tier: GlobalTier


def default_global_etf_universe() -> list[GlobalInstrument]:
    """Return a liquid USD ETF proxy universe for worldwide market research."""

    return [
        GlobalInstrument("ACWI", "iShares MSCI ACWI ETF", "Global", "Global", "USD", "NASDAQ", "Equity", "benchmark"),
        GlobalInstrument("SPY", "SPDR S&P 500 ETF", "North America", "United States", "USD", "NYSE Arca", "Equity", "style_etf"),
        GlobalInstrument("QQQ", "Invesco QQQ Trust", "North America", "United States", "USD", "NASDAQ", "Equity", "style_etf"),
        GlobalInstrument("IWM", "iShares Russell 2000 ETF", "North America", "United States", "USD", "NYSE Arca", "Equity", "style_etf"),
        GlobalInstrument("EFA", "iShares MSCI EAFE ETF", "Developed ex-US", "Developed ex-US", "USD", "NYSE Arca", "Equity", "regional_etf"),
        GlobalInstrument("EEM", "iShares MSCI Emerging Markets ETF", "Emerging Markets", "Emerging Markets", "USD", "NYSE Arca", "Equity", "regional_etf"),
        GlobalInstrument("IEV", "iShares Europe ETF", "Europe", "Europe", "USD", "NYSE Arca", "Equity", "regional_etf"),
        GlobalInstrument("EWJ", "iShares MSCI Japan ETF", "Asia Pacific", "Japan", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWU", "iShares MSCI United Kingdom ETF", "Europe", "United Kingdom", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWG", "iShares MSCI Germany ETF", "Europe", "Germany", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWQ", "iShares MSCI France ETF", "Europe", "France", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWL", "iShares MSCI Switzerland ETF", "Europe", "Switzerland", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWI", "iShares MSCI Italy ETF", "Europe", "Italy", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWP", "iShares MSCI Spain ETF", "Europe", "Spain", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWN", "iShares MSCI Netherlands ETF", "Europe", "Netherlands", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWD", "iShares MSCI Sweden ETF", "Europe", "Sweden", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWC", "iShares MSCI Canada ETF", "North America", "Canada", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWA", "iShares MSCI Australia ETF", "Asia Pacific", "Australia", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWY", "iShares MSCI South Korea ETF", "Asia Pacific", "South Korea", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("INDA", "iShares MSCI India ETF", "Asia Pacific", "India", "USD", "Cboe BZX", "Equity", "country_etf"),
        GlobalInstrument("EPI", "WisdomTree India Earnings Fund", "Asia Pacific", "India", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWZ", "iShares MSCI Brazil ETF", "Latin America", "Brazil", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWW", "iShares MSCI Mexico ETF", "Latin America", "Mexico", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EZA", "iShares MSCI South Africa ETF", "Middle East & Africa", "South Africa", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("TUR", "iShares MSCI Turkey ETF", "Middle East & Africa", "Turkey", "USD", "NASDAQ", "Equity", "country_etf"),
        GlobalInstrument("KSA", "iShares MSCI Saudi Arabia ETF", "Middle East & Africa", "Saudi Arabia", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("UAE", "iShares MSCI UAE ETF", "Middle East & Africa", "United Arab Emirates", "USD", "NASDAQ", "Equity", "country_etf"),
        GlobalInstrument("EWS", "iShares MSCI Singapore ETF", "ASEAN", "Singapore", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EWM", "iShares MSCI Malaysia ETF", "ASEAN", "Malaysia", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("EIDO", "iShares MSCI Indonesia ETF", "ASEAN", "Indonesia", "USD", "NYSE Arca", "Equity", "country_etf"),
        GlobalInstrument("XLF", "Financial Select Sector SPDR Fund", "United States Sector", "United States", "USD", "NYSE Arca", "Equity", "sector_etf"),
        GlobalInstrument("XLK", "Technology Select Sector SPDR Fund", "United States Sector", "United States", "USD", "NYSE Arca", "Equity", "sector_etf"),
        GlobalInstrument("XLE", "Energy Select Sector SPDR Fund", "United States Sector", "United States", "USD", "NYSE Arca", "Equity", "sector_etf"),
        GlobalInstrument("XLV", "Health Care Select Sector SPDR Fund", "United States Sector", "United States", "USD", "NYSE Arca", "Equity", "sector_etf"),
    ]


def global_universe_metadata(instruments: list[GlobalInstrument] | None = None) -> pd.DataFrame:
    """Return the global universe as a DataFrame."""

    rows = [instrument.__dict__ for instrument in (instruments or default_global_etf_universe())]
    return pd.DataFrame(rows)


def tradable_global_tickers(instruments: list[GlobalInstrument] | None = None) -> list[str]:
    """Return all non-benchmark tickers in stable universe order."""

    universe = instruments or default_global_etf_universe()
    return [instrument.ticker for instrument in universe if instrument.tier != "benchmark"]
