from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Tier = Literal["regional_proxy", "country_index", "stock"]


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    country: str
    currency: str
    exchange: str
    tier: Tier


def default_asean_universe() -> list[Instrument]:
    """Current-survivor three-country ASEAN universe with Yahoo Finance tickers.

    This is not a point-in-time constituent database. It is a practical free-data
    starter set covering Malaysia, Singapore, and Indonesia proxies plus liquid
    large-cap stocks where Yahoo historical data is available through yfinance.
    """

    instruments = [
        Instrument("EWS", "iShares MSCI Singapore ETF", "Singapore", "USD", "NYSE Arca", "regional_proxy"),
        Instrument("EIDO", "iShares MSCI Indonesia ETF", "Indonesia", "USD", "NYSE Arca", "regional_proxy"),
        Instrument("EWM", "iShares MSCI Malaysia ETF", "Malaysia", "USD", "NYSE Arca", "regional_proxy"),
        Instrument("^STI", "Straits Times Index", "Singapore", "SGD", "SGX", "country_index"),
        Instrument("^JKSE", "Jakarta Composite Index", "Indonesia", "IDR", "IDX", "country_index"),
        Instrument("^JKLQ45", "LQ45 Index", "Indonesia", "IDR", "IDX", "country_index"),
        Instrument("^KLSE", "FTSE Bursa Malaysia KLCI", "Malaysia", "MYR", "Bursa Malaysia", "country_index"),
    ]

    country_specs = {
        "Singapore": (
            "SGD",
            "SGX",
            [
                "D05.SI",
                "O39.SI",
                "U11.SI",
                "Z74.SI",
                "S68.SI",
                "C6L.SI",
                "S63.SI",
                "BN4.SI",
                "F34.SI",
                "G13.SI",
                "C09.SI",
                "C38U.SI",
                "A17U.SI",
                "9CI.SI",
                "H78.SI",
                "J36.SI",
                "Y92.SI",
                "U14.SI",
                "V03.SI",
                "S58.SI",
                "BS6.SI",
                "M44U.SI",
                "ME8U.SI",
                "N2IU.SI",
                "BUOU.SI",
                "J69U.SI",
                "AJBU.SI",
                "D01.SI",
                "5E2.SI",
                "U96.SI",
            ],
        ),
        "Indonesia": (
            "IDR",
            "IDX",
            [
                "BBCA.JK",
                "BBRI.JK",
                "BMRI.JK",
                "TLKM.JK",
                "ASII.JK",
                "BBNI.JK",
                "UNTR.JK",
                "ICBP.JK",
                "INDF.JK",
                "KLBF.JK",
                "CPIN.JK",
                "SMGR.JK",
                "ANTM.JK",
                "INCO.JK",
                "ADRO.JK",
                "ITMG.JK",
                "MDKA.JK",
                "AMMN.JK",
                "BRPT.JK",
                "TPIA.JK",
                "GOTO.JK",
                "PGAS.JK",
                "EXCL.JK",
                "ISAT.JK",
                "MEDC.JK",
                "AKRA.JK",
                "AMRT.JK",
                "ARTO.JK",
                "INKP.JK",
                "INTP.JK",
            ],
        ),
        "Malaysia": (
            "MYR",
            "Bursa Malaysia",
            [
                "1155.KL",
                "1023.KL",
                "1295.KL",
                "5347.KL",
                "5225.KL",
                "5819.KL",
                "6947.KL",
                "5183.KL",
                "8869.KL",
                "2445.KL",
                "1961.KL",
                "6888.KL",
                "6012.KL",
                "3816.KL",
                "6033.KL",
                "5681.KL",
                "4707.KL",
                "5285.KL",
                "4197.KL",
                "4677.KL",
                "6742.KL",
                "5211.KL",
                "5296.KL",
                "5398.KL",
                "7084.KL",
                "1066.KL",
                "4863.KL",
                "4065.KL",
                "1082.KL",
                "1818.KL",
            ],
        ),
    }

    for country, (currency, exchange, tickers) in country_specs.items():
        for ticker in tickers:
            instruments.append(Instrument(ticker, ticker, country, currency, exchange, "stock"))

    return instruments


def instruments_by_tier(tier: Tier) -> list[Instrument]:
    return [instrument for instrument in default_asean_universe() if instrument.tier == tier]
