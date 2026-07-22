from __future__ import annotations

import re

import pandas as pd


INFERRED_REFINITIV_RIC_STATUS = "inferred_not_validated"
SUPPLIED_REFINITIV_RIC_STATUS = "supplied_unvalidated"
MISSING_REFINITIV_RIC_STATUS = "missing"


def _clean_symbol(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def infer_refinitiv_ric(row: pd.Series | dict) -> str | None:
    """Infer a likely equity RIC from existing listing metadata.

    This is intentionally conservative. RIC inference is good enough for smoke
    tests, but serious Tier 2 data needs a vendor security master or an audited
    symbol map.
    """

    data = row if isinstance(row, dict) else row.to_dict()
    ticker = _clean_symbol(data.get("ticker"))
    yahoo_ticker = _clean_symbol(data.get("yahoo_ticker"))
    country = _clean_symbol(data.get("country")) or ""
    exchange = _clean_symbol(data.get("exchange")) or ""
    exchange_code = _clean_symbol(data.get("exchange_code")) or ""

    if yahoo_ticker and "." in yahoo_ticker:
        return yahoo_ticker

    if country == "United States" and ticker:
        exchange_text = f"{exchange} {exchange_code}".upper()
        symbol = re.sub(r"[^A-Z0-9.-]", "", ticker.upper())
        if "NASDAQ" in exchange_text:
            return f"{symbol}.O"
        if "NYSE AMERICAN" in exchange_text or exchange_code == "A":
            return f"{symbol}.A"
        if "NYSE" in exchange_text:
            return f"{symbol}.N"
        if "CBOE" in exchange_text or "BZX" in exchange_text:
            return f"{symbol}.Z"
        return None

    return yahoo_ticker or ticker


def add_refinitiv_ric(
    listings: pd.DataFrame,
    ric_col: str = "refinitiv_ric",
    status_col: str = "refinitiv_ric_status",
) -> pd.DataFrame:
    """Add a proposed Refinitiv RIC column plus its validation status.

    Inferred RICs are useful for smoke tests and field discovery, but they are
    not a vendor security master. Production Refinitiv extraction should use a
    validated symbol map or a validation manifest from a live probe.
    """

    if listings.empty:
        result = listings.copy()
        result[ric_col] = pd.Series(dtype="object")
        result[status_col] = pd.Series(dtype="object")
        return result
    result = listings.copy()
    inferred = result.apply(infer_refinitiv_ric, axis=1)
    supplied = result[ric_col].map(_clean_symbol) if ric_col in result.columns else pd.Series([None] * len(result))
    result[ric_col] = supplied.combine_first(inferred)

    if status_col not in result.columns:
        result[status_col] = MISSING_REFINITIV_RIC_STATUS
    else:
        result[status_col] = result[status_col].fillna(MISSING_REFINITIV_RIC_STATUS)
    result.loc[supplied.notna(), status_col] = result.loc[supplied.notna(), status_col].replace(
        MISSING_REFINITIV_RIC_STATUS,
        SUPPLIED_REFINITIV_RIC_STATUS,
    )
    inferred_mask = supplied.isna() & result[ric_col].notna()
    result.loc[inferred_mask, status_col] = result.loc[inferred_mask, status_col].replace(
        MISSING_REFINITIV_RIC_STATUS,
        INFERRED_REFINITIV_RIC_STATUS,
    )
    return result
