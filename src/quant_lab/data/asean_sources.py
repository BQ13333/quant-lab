from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


ASEAN_SOURCE_COLUMNS = [
    "exchange",
    "country",
    "public_source_status",
    "preferred_yahoo_suffixes",
    "notes",
]


@dataclass(frozen=True)
class AseanExchangeSource:
    exchange: str
    country: str
    public_source_status: str
    preferred_yahoo_suffixes: tuple[str, ...]
    notes: str

    def as_record(self) -> dict[str, object]:
        return {
            "exchange": self.exchange,
            "country": self.country,
            "public_source_status": self.public_source_status,
            "preferred_yahoo_suffixes": self.preferred_yahoo_suffixes,
            "notes": self.notes,
        }


_PENDING_STATUS = "pending_public_source_validation"
_PENDING_NOTE = "Pending manual validation of public exchange listing endpoint, coverage, and licensing terms."
_ACTIVE_STATUS = "active_public_adapter"


ASEAN_EXCHANGE_SOURCES: tuple[AseanExchangeSource, ...] = (
    AseanExchangeSource(
        exchange="SGX",
        country="Singapore",
        public_source_status=_ACTIVE_STATUS,
        preferred_yahoo_suffixes=(".SI",),
        notes="Uses SGX securities v1.1 public JSON endpoint; undocumented front-end API, so monitor schema drift and licensing terms.",
    ),
    AseanExchangeSource(
        exchange="Bursa Malaysia",
        country="Malaysia",
        public_source_status=_PENDING_STATUS,
        preferred_yahoo_suffixes=(".KL",),
        notes=_PENDING_NOTE,
    ),
    AseanExchangeSource(
        exchange="IDX",
        country="Indonesia",
        public_source_status=_PENDING_STATUS,
        preferred_yahoo_suffixes=(".JK",),
        notes=_PENDING_NOTE,
    ),
    AseanExchangeSource(
        exchange="SET",
        country="Thailand",
        public_source_status=_PENDING_STATUS,
        preferred_yahoo_suffixes=(".BK",),
        notes="Parser exists for official SET listed-company Excel exports, but live downloads are blocked by the exchange edge service in this environment.",
    ),
    AseanExchangeSource(
        exchange="PSE",
        country="Philippines",
        public_source_status=_ACTIVE_STATUS,
        preferred_yahoo_suffixes=(".PS",),
        notes="Uses PSE EDGE company-directory AJAX HTML endpoint; current listed-company snapshot only. Yahoo price coverage is poor and needs a better vendor.",
    ),
    AseanExchangeSource(
        exchange="HNX/UPCoM",
        country="Vietnam",
        public_source_status=_ACTIVE_STATUS,
        preferred_yahoo_suffixes=(".VN",),
        notes="Uses HNX official AJAX endpoints for HNX listed stocks and UPCoM; HOSE still needs a separate stable adapter or vendor data. Yahoo price coverage is poor.",
    ),
)


def asean_source_registry() -> pd.DataFrame:
    """Return conservative ASEAN source status metadata."""

    return pd.DataFrame.from_records(
        [source.as_record() for source in ASEAN_EXCHANGE_SOURCES],
        columns=ASEAN_SOURCE_COLUMNS,
    )


def asean_yahoo_suffix_map() -> dict[str, tuple[str, ...]]:
    """Return preferred Yahoo Finance suffixes by ASEAN country."""

    return {source.country: source.preferred_yahoo_suffixes for source in ASEAN_EXCHANGE_SOURCES}
