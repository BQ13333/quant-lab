"""Point-in-time coverage for identities already in this repository.

The sheet records what is on disk. It does not download filings, and it does
not treat a missing date as a known date. Official fundamentals are Bursa
announcements, HKEXnews, and CNINFO. Anything else is not a filing.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from quant_lab.data.asean_universe import default_asean_universe
from quant_lab.programme.scope import MarketScope, MarketScopeError, load_market_scope

COLUMNS = (
    "sid",
    "market",
    "exchange",
    "symbol",
    "status",
    "listing_date",
    "delist_date",
    "corporate_actions",
    "fundamentals_source",
    "evidence_pointer",
    "where_found",
)

OFFICIAL_FILING_SOURCES = frozenset({"bursa_announcements", "hkexnews", "cninfo"})
_CORPORATE_ACTIONS = frozenset({"present", "explicitly_absent"})
_REFUSED_FUNDAMENTAL_SOURCES = frozenset(
    {
        "akshare",
        "baostock",
        "bloomberg",
        "factset",
        "lseg",
        "refinitiv",
        "stooq",
        "wind",
        "yahoo",
    }
)
_SUFFIXES = {
    ("Malaysia", "Bursa Malaysia"): ".KL",
    ("Hong Kong", "HKEX"): ".HK",
    ("China", "SSE"): ".SS",
    ("China", "SZSE"): ".SZ",
}


@dataclass(frozen=True)
class CoverageRow:
    sid: str
    market: str
    exchange: str
    symbol: str
    status: str
    listing_date: str
    delist_date: str
    corporate_actions: str
    fundamentals_source: str
    evidence_pointer: str
    where_found: str

    def as_dict(self) -> dict[str, str]:
        return {column: getattr(self, column) for column in COLUMNS}


@dataclass(frozen=True)
class CoverageAssessment:
    sid: str
    admitted: bool
    reasons: tuple[str, ...]


def assess_row(row: CoverageRow, scope: MarketScope | None = None) -> CoverageAssessment:
    """Admit a row only when the identity is explicit and the filing source is allowed."""

    mandate = scope or load_market_scope()
    reasons: list[str] = []
    try:
        mandate.require_active(row.market, row.exchange)
    except MarketScopeError:
        reasons.append("market_not_active")
    if row.exchange == "BSE" or row.symbol.upper().endswith(".BJ"):
        reasons.append("bse_out")
    expected = _SUFFIXES.get((row.market, row.exchange))
    if expected is None or not row.symbol.endswith(expected):
        reasons.append("symbol_does_not_match_exchange")
    if any(token in row.symbol.upper() for token in ("BTC", "ETH", "USDT", "CRYPTO")):
        reasons.append("crypto_out")
    if row.status not in {"active", "inactive"}:
        reasons.append("status_invalid")
    listing = _date(row.listing_date)
    delisted = _date(row.delist_date)
    if row.listing_date and listing is None:
        reasons.append("listing_date_invalid")
    if row.delist_date and delisted is None:
        reasons.append("delist_date_invalid")
    if listing is None:
        reasons.append("listing_date_missing")
    if row.status == "inactive" and delisted is None:
        reasons.append("inactive_without_delist_date")
    if row.status == "active" and row.delist_date:
        reasons.append("active_with_delist_date")
    if listing is not None and delisted is not None and delisted < listing:
        reasons.append("delist_before_listing")
    if row.corporate_actions not in _CORPORATE_ACTIONS:
        reasons.append("corporate_actions_not_explicit")
    source = row.fundamentals_source.strip().lower()
    if source in _REFUSED_FUNDAMENTAL_SOURCES:
        reasons.append("fundamentals_source_not_official")
    elif source == "absent":
        pass
    elif source not in OFFICIAL_FILING_SOURCES:
        reasons.append("fundamentals_source_not_official")
    elif not row.evidence_pointer.strip():
        reasons.append("official_filing_pointer_missing")
    if not row.where_found.strip():
        reasons.append("where_found_missing")
    if not row.sid or row.sid != row.symbol:
        reasons.append("sid_must_match_symbol")
    return CoverageAssessment(row.sid, not reasons, tuple(reasons))


def load_sheet(path: Path | str) -> tuple[CoverageRow, ...]:
    target = Path(path)
    with target.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError("coverage sheet columns do not match the contract")
        rows: list[CoverageRow] = []
        for raw in reader:
            if any(raw.get(column) is None for column in COLUMNS):
                raise ValueError("coverage sheet has a ragged row")
            rows.append(CoverageRow(*(raw[column].strip() for column in COLUMNS)))
    return tuple(rows)


def write_sheet(path: Path | str, rows: tuple[CoverageRow, ...] | list[CoverageRow]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def on_disk_identities() -> tuple[CoverageRow, ...]:
    """Names committed in this repository. Dates and filings are not invented."""

    rows: list[CoverageRow] = []
    for instrument in default_asean_universe():
        if instrument.country != "Malaysia" or instrument.tier != "stock":
            continue
        rows.append(
            _gap_row(
                instrument.ticker,
                "Malaysia",
                "Bursa Malaysia",
                "src/quant_lab/data/asean_universe.py",
            )
        )
    rows.extend(
        [
            _gap_row(
                "MBBM.KL",
                "Malaysia",
                "Bursa Malaysia",
                "src/quant_lab/data/refinitiv_symbol_map.py",
            ),
            _gap_row(
                "0005.HK",
                "Hong Kong",
                "HKEX",
                "tests/test_fx.py",
            ),
            _gap_row(
                "0700.HK",
                "Hong Kong",
                "HKEX",
                "src/quant_lab/data/refinitiv_symbol_map.py",
            ),
            _gap_row(
                "000001.SZ",
                "China",
                "SZSE",
                "tests/test_external_engine_file_exchange.py",
            ),
            _gap_row(
                "600000.SS",
                "China",
                "SSE",
                "docs/HISTORICAL_DATA_LAKE_BACKFILL.md",
            ),
            _gap_row(
                "600004.SS",
                "China",
                "SSE",
                "docs/HISTORICAL_DATA_LAKE_BACKFILL.md",
            ),
            _gap_row(
                "600519.SS",
                "China",
                "SSE",
                "tests/test_external_engine_file_exchange.py",
            ),
        ]
    )
    return tuple(sorted(rows, key=lambda row: (row.market, row.symbol)))


def _gap_row(symbol: str, market: str, exchange: str, where_found: str) -> CoverageRow:
    return CoverageRow(
        sid=symbol,
        market=market,
        exchange=exchange,
        symbol=symbol,
        status="active",
        listing_date="",
        delist_date="",
        corporate_actions="",
        fundamentals_source="absent",
        evidence_pointer="",
        where_found=where_found,
    )


def _date(value: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None
