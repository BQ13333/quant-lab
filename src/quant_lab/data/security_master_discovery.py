from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


DEFAULT_DISCOVERY_COUNTRY_CODES: tuple[str, ...] = ("MYS", "SGP", "IDN")
SUPPORTED_DISCOVERY_COUNTRY_CODES: tuple[str, ...] = (
    "MYS",
    "SGP",
    "IDN",
    "CHN",
    "HKG",
    "TWN",
    "KOR",
    "JPN",
    "USA",
    "THA",
)

DISCOVERY_COUNTRY_NAMES: dict[str, str] = {
    "MYS": "Malaysia",
    "SGP": "Singapore",
    "IDN": "Indonesia",
    "CHN": "China",
    "HKG": "Hong Kong",
    "TWN": "Taiwan",
    "KOR": "South Korea",
    "JPN": "Japan",
    "USA": "United States",
    "THA": "Thailand",
}

DISCOVERY_SEARCH_FIELDS: tuple[str, ...] = (
    "RIC",
    "DocumentTitle",
    "CommonName",
    "ExchangeCountry",
    "ExchangeName",
    "ExchangeCode",
    "AssetState",
    "AssetStateName",
    "RCSAssetCategoryLeaf",
    "InstrumentPermID",
    "TickerSymbol",
    "IsPrimaryRIC",
    "IsPrimaryIssueRIC",
)

DISCOVERY_SEARCH_SELECT = ",".join(DISCOVERY_SEARCH_FIELDS)

DISCOVERY_ENRICHMENT_FIELDS: tuple[str, ...] = (
    "TR.CommonName",
    "TR.ExchangeName",
    "TR.ExchangeCountryCode",
    "TR.AssetStatus",
    "TR.RetireDate",
    "TR.FirstTradeDate",
    "TR.ISIN",
    "TR.SEDOL",
    "TR.OrganizationID",
    "TR.PermID",
    "TR.PriceCurrency",
    "TR.RIC",
)

SEARCH_COLUMN_MAP: dict[str, str] = {
    "RIC": "ric",
    "DocumentTitle": "document_title",
    "CommonName": "common_name",
    "ExchangeCountry": "exchange_country",
    "ExchangeName": "exchange_name",
    "ExchangeCode": "exchange_code",
    "AssetState": "asset_state",
    "AssetStateName": "asset_state_name",
    "RCSAssetCategoryLeaf": "rcs_asset_category_leaf",
    "InstrumentPermID": "instrument_permid",
    "TickerSymbol": "ticker_symbol",
    "IsPrimaryRIC": "is_primary_ric",
    "IsPrimaryIssueRIC": "is_primary_issue_ric",
}

ENRICHMENT_COLUMN_MAP: dict[str, str] = {
    "Instrument": "ric",
    "TR.CommonName": "common_name_enriched",
    "Common Name": "common_name_enriched",
    "CommonName": "common_name_enriched",
    "TR.ExchangeName": "exchange_name_enriched",
    "Exchange Name": "exchange_name_enriched",
    "ExchangeName": "exchange_name_enriched",
    "TR.ExchangeCountryCode": "exchange_country_code",
    "Exchange Country Code": "exchange_country_code",
    "ExchangeCountryCode": "exchange_country_code",
    "TR.AssetStatus": "asset_status",
    "Asset Status": "asset_status",
    "AssetStatus": "asset_status",
    "TR.RetireDate": "retire_date",
    "Retire Date": "retire_date",
    "RetireDate": "retire_date",
    "TR.FirstTradeDate": "first_trade_date",
    "First Trade Date": "first_trade_date",
    "FirstTradeDate": "first_trade_date",
    "TR.ISIN": "isin",
    "ISIN": "isin",
    "TR.SEDOL": "sedol",
    "SEDOL": "sedol",
    "TR.OrganizationID": "organization_id",
    "Organization ID": "organization_id",
    "OrganizationID": "organization_id",
    "Organization PermID": "organization_id",
    "TR.PermID": "perm_id",
    "PermID": "perm_id",
    "TR.PriceCurrency": "price_currency",
    "Price Currency": "price_currency",
    "PriceCurrency": "price_currency",
    "TR.RIC": "ric_enriched",
    "RIC": "ric_enriched",
}

SEARCH_NORMALIZED_COLUMNS: tuple[str, ...] = (
    "ric",
    "document_title",
    "common_name",
    "exchange_country",
    "exchange_name",
    "exchange_code",
    "asset_state",
    "asset_state_name",
    "rcs_asset_category_leaf",
    "instrument_permid",
    "ticker_symbol",
    "is_primary_ric",
    "is_primary_issue_ric",
    "active_state_bucket",
    "country_code",
    "country",
    "search_state_bucket",
)

ENTITIES_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "sid",
    "ric",
    "instrument_permid",
    "perm_id",
    "organization_id",
    "issuer_name",
    "document_title",
    "country",
    "country_code",
    "exchange_country",
    "primary_exchange",
    "exchange_code",
    "ticker_symbol",
    "security_type",
    "asset_state",
    "asset_state_name",
    "active_state_bucket",
    "source_vendor",
    "entity_source_scope",
    "source_loaded_at_utc",
)

IDENTIFIER_HISTORY_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "sid",
    "id_type",
    "id_value",
    "effective_from",
    "effective_to",
    "change_type",
    "change_reason",
    "source_timestamp_utc",
    "source_loaded_at_utc",
    "evidence_level",
    "eligible_for_model_denominator",
)

LISTING_WINDOWS_COLUMNS: tuple[str, ...] = (
    "entity_id",
    "sid",
    "ric",
    "ticker",
    "country",
    "country_code",
    "exchange",
    "exchange_code",
    "currency",
    "listing_date",
    "first_trade_date",
    "effective_from",
    "effective_to",
    "delisting_date",
    "last_trade_date",
    "listing_status",
    "asset_state",
    "asset_state_name",
    "delisting_reason",
    "tradable_flag",
    "active_flag",
    "window_source",
    "evidence_level",
    "eligible_for_model_denominator",
    "source_loaded_at_utc",
)

UNIVERSE_MEMBERSHIP_COLUMNS: tuple[str, ...] = (
    "asof_date",
    "entity_id",
    "sid",
    "ric_asof",
    "ticker_asof",
    "country",
    "country_code",
    "exchange",
    "currency",
    "membership_start",
    "membership_end",
    "is_listed_asof",
    "is_investable_asof",
    "eligible_flag",
    "in_discovery_security_master_flag",
    "exclusion_reason",
    "known_at_utc",
    "membership_source",
    "source_loaded_at_utc",
)

QUALITY_COLUMNS: tuple[str, ...] = (
    "country",
    "country_code",
    "asof_year",
    "row_count",
    "active_count",
    "delisted_count",
    "dead_name_count",
    "identifier_change_count",
    "identifier_rows",
    "listing_window_rows",
    "null_ric_rate",
    "dead_name_coverage_rate",
    "first_trade_date_min",
    "retire_date_min",
    "retire_date_max",
    "status",
)


@dataclass(frozen=True)
class SecurityMasterDiscoveryExtraction:
    search_hits: pd.DataFrame
    enrichment: pd.DataFrame
    security_master: pd.DataFrame
    page_log: pd.DataFrame
    pagination_state: dict[str, Any]


@dataclass(frozen=True)
class SecurityMasterDiscoveryTables:
    security_master: pd.DataFrame
    entities: pd.DataFrame
    identifier_history: pd.DataFrame
    listing_windows: pd.DataFrame
    universe_membership: pd.DataFrame
    quality: pd.DataFrame


@dataclass(frozen=True)
class SecurityMasterDiscoveryArtifacts:
    run_id: str
    artifact_dir: Path
    search_hits_path: Path
    enrichment_path: Path
    page_log_path: Path
    security_master_path: Path
    entities_path: Path
    identifier_history_path: Path
    listing_windows_path: Path
    universe_membership_path: Path
    quality_path: Path
    manifest_path: Path
    search_hits: pd.DataFrame
    enrichment: pd.DataFrame
    page_log: pd.DataFrame
    entities: pd.DataFrame
    identifier_history: pd.DataFrame
    listing_windows: pd.DataFrame
    universe_membership: pd.DataFrame
    quality: pd.DataFrame
    manifest: dict[str, Any]


def security_master_discovery_filter(
    country_code: str,
    state_bucket: str,
    primary_only: bool = False,
    extra_filter: str | None = None,
) -> str:
    """Return an LSEG Discovery Search filter for ordinary equities."""

    code = _normalize_country_code(country_code)
    bucket = state_bucket.strip().casefold()
    base = (
        "BusinessEntity eq 'QUOTExEQUITY' and "
        "RCSAssetCategoryLeaf eq 'Ordinary Share' and "
        f"ExchangeCountry eq '{code}'"
    )
    if primary_only:
        base = f"{base} and IsPrimaryRIC eq true and IsPrimaryIssueRIC eq true"
    if extra_filter:
        base = f"{base} and ({extra_filter})"
    if bucket in {"active", "ac"}:
        return f"{base} and AssetState eq 'AC'"
    if bucket in {"inactive", "dead", "retired"}:
        return f"{base} and AssetState ne 'AC' and AssetState ne null"
    raise ValueError(f"Unsupported security-master state bucket: {state_bucket!r}")


def discovery_equity_quotes_view(discovery_client: Any) -> Any:
    """Resolve ld.discovery.Views.EQUITY_QUOTES, with a string fallback for tests."""

    discovery = _discovery_namespace(discovery_client)
    views = getattr(discovery, "Views", None)
    return getattr(views, "EQUITY_QUOTES", "EQUITY_QUOTES")


def search_security_master_discovery(
    discovery_client: Any,
    country_codes: tuple[str, ...] = DEFAULT_DISCOVERY_COUNTRY_CODES,
    state_buckets: tuple[str, ...] = ("active", "inactive"),
    top: int = 1000,
    start_skip: int = 0,
    max_pages: int | None = None,
    max_rows: int | None = None,
    sleep_seconds: float = 0.0,
    search_callable: Callable[..., Any] | None = None,
    primary_only: bool = False,
    extra_filter: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Search EQUITY_QUOTES with top/skip pagination.

    `max_pages` is applied per country/state route. `max_rows` is a global cap
    and can stop mid-page; the returned pagination state records a resume skip.
    """

    if top <= 0:
        raise ValueError("top must be positive")
    if start_skip < 0:
        raise ValueError("start_skip cannot be negative")
    if max_pages is not None and max_pages <= 0:
        raise ValueError("max_pages must be positive when supplied")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be positive when supplied")

    codes = _normalize_country_codes(country_codes)
    frames: list[pd.DataFrame] = []
    page_rows: list[dict[str, Any]] = []
    resume: dict[str, dict[str, Any]] = {}
    stopped_by: str | None = None
    rows_fetched = 0
    pages_fetched = 0
    view = discovery_equity_quotes_view(discovery_client)

    for code in codes:
        for bucket in state_buckets:
            bucket_name = bucket.strip().casefold()
            route_key = f"{code}:{bucket_name}"
            skip = int(start_skip)
            route_pages = 0
            route_exhausted = False
            query_filter = security_master_discovery_filter(
                code,
                bucket_name,
                primary_only=primary_only,
                extra_filter=extra_filter,
            )

            while True:
                if max_rows is not None and rows_fetched >= max_rows:
                    stopped_by = stopped_by or "max_rows"
                    break
                if max_pages is not None and route_pages >= max_pages:
                    stopped_by = stopped_by or "max_pages"
                    break

                raw_page = _call_discovery_search(
                    discovery_client=discovery_client,
                    view=view,
                    query_filter=query_filter,
                    select=DISCOVERY_SEARCH_SELECT,
                    top=top,
                    skip=skip,
                    search_callable=search_callable,
                )
                normalized = normalize_discovery_search_hits(raw_page, country_code=code, state_bucket=bucket_name)
                if len(normalized) > top:
                    normalized = normalized.head(top).copy()

                remaining = None if max_rows is None else max_rows - rows_fetched
                consumed = normalized if remaining is None else normalized.head(max(0, remaining)).copy()
                if not consumed.empty:
                    frames.append(consumed)
                    rows_fetched += int(len(consumed))

                raw_rows = int(len(raw_page))
                consumed_rows = int(len(consumed))
                route_pages += 1
                pages_fetched += 1
                page_rows.append(
                    {
                        "country_code": code,
                        "country": DISCOVERY_COUNTRY_NAMES.get(code, code),
                        "state_bucket": bucket_name,
                        "view": str(view),
                        "filter": query_filter,
                        "select": DISCOVERY_SEARCH_SELECT,
                        "top": int(top),
                        "skip": int(skip),
                        "raw_rows": raw_rows,
                        "rows": consumed_rows,
                    }
                )

                if raw_rows == 0 or raw_rows < top:
                    route_exhausted = True
                    break
                if consumed_rows < len(normalized):
                    stopped_by = stopped_by or "max_rows"
                    skip += consumed_rows
                    break
                skip += top
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

            resume[route_key] = {
                "country_code": code,
                "state_bucket": bucket_name,
                "next_skip": int(skip),
                "exhausted": bool(route_exhausted),
            }
            if stopped_by == "max_rows":
                break
        if stopped_by == "max_rows":
            break

    hits = pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_search_hits()
    if not hits.empty:
        hits = hits.drop_duplicates(["ric", "country_code", "active_state_bucket"], keep="first").reset_index(drop=True)
    page_log = pd.DataFrame(page_rows)
    pagination_state = {
        "top": int(top),
        "start_skip": int(start_skip),
        "max_pages": max_pages,
        "max_rows": max_rows,
        "primary_only": primary_only,
        "extra_filter": extra_filter,
        "pages_fetched": int(pages_fetched),
        "rows_fetched": int(len(hits)),
        "stopped_by": stopped_by,
        "resume": resume,
    }
    return hits, page_log, pagination_state


def normalize_discovery_search_hits(
    frame: pd.DataFrame,
    country_code: str | None = None,
    state_bucket: str | None = None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_search_hits()

    data = _rename_columns(pd.DataFrame(frame).copy(), SEARCH_COLUMN_MAP)
    for column in SEARCH_COLUMN_MAP.values():
        if column not in data.columns:
            data[column] = pd.NA

    if country_code is not None:
        fallback_code = _normalize_country_code(country_code)
    else:
        fallback_code = None
    data["ric"] = _string_series(data["ric"])
    data["asset_state"] = _string_series(data["asset_state"]).str.upper()
    data["exchange_country"] = _string_series(data["exchange_country"]).str.upper()
    data["country_code"] = data["exchange_country"].where(data["exchange_country"].notna(), fallback_code)
    data["country_code"] = _string_series(data["country_code"]).str.upper()
    data["country"] = data["country_code"].map(DISCOVERY_COUNTRY_NAMES)
    data["search_state_bucket"] = state_bucket
    data["active_state_bucket"] = data["asset_state"].map(_active_bucket_from_asset_state)
    if state_bucket is not None:
        bucket = state_bucket.strip().casefold()
        data["active_state_bucket"] = data["active_state_bucket"].where(data["active_state_bucket"].ne("unknown"), bucket)
    data["is_primary_ric"] = data["is_primary_ric"].map(_to_bool)
    data["is_primary_issue_ric"] = data["is_primary_issue_ric"].map(_to_bool)

    for column in SEARCH_NORMALIZED_COLUMNS:
        if column not in data.columns:
            data[column] = pd.NA
    return data[list(SEARCH_NORMALIZED_COLUMNS)].reset_index(drop=True)


def fetch_security_master_discovery_enrichment(
    get_data_client: Any,
    search_hits: pd.DataFrame,
    chunk_size: int = 100,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    if get_data_client is None or search_hits.empty or "ric" not in search_hits.columns:
        return _empty_enrichment()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    getter = getattr(get_data_client, "get_data", None)
    if getter is None and callable(get_data_client):
        getter = get_data_client
    if getter is None:
        raise TypeError("get_data_client must expose get_data(universe=..., fields=...) or be callable")

    rics = [
        value
        for value in dict.fromkeys(_string_series(search_hits["ric"]).dropna().tolist())
        if value and value.strip()
    ]
    frames: list[pd.DataFrame] = []
    for chunk in _chunks(rics, chunk_size):
        if not chunk:
            continue
        frame = getter(universe=chunk, fields=list(DISCOVERY_ENRICHMENT_FIELDS))
        normalized = normalize_security_master_enrichment(frame)
        if not normalized.empty:
            frames.append(normalized)
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    enrichment = pd.concat(frames, ignore_index=True, sort=False) if frames else _empty_enrichment()
    if not enrichment.empty:
        enrichment = enrichment.drop_duplicates("ric", keep="first").reset_index(drop=True)
    return enrichment


def normalize_security_master_enrichment(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return _empty_enrichment()
    data = _rename_columns(pd.DataFrame(frame).copy(), ENRICHMENT_COLUMN_MAP)
    if "ric" not in data.columns:
        if "ric_enriched" in data.columns:
            data["ric"] = data["ric_enriched"]
        elif data.index.name or not isinstance(data.index, pd.RangeIndex):
            data["ric"] = data.index.astype(str)
        else:
            data["ric"] = pd.NA
    if "ric_enriched" in data.columns:
        data["ric"] = data["ric"].where(data["ric"].notna(), data["ric_enriched"])

    for column in (
        "ric",
        "ric_enriched",
        "common_name_enriched",
        "exchange_name_enriched",
        "exchange_country_code",
        "asset_status",
        "retire_date",
        "first_trade_date",
        "isin",
        "sedol",
        "organization_id",
        "perm_id",
        "price_currency",
    ):
        if column not in data.columns:
            data[column] = pd.NA

    data["ric"] = _string_series(data["ric"])
    data["retire_date"] = _normalize_date_series(data["retire_date"])
    data["first_trade_date"] = _normalize_date_series(data["first_trade_date"])
    columns = [
        "ric",
        "ric_enriched",
        "common_name_enriched",
        "exchange_name_enriched",
        "exchange_country_code",
        "asset_status",
        "retire_date",
        "first_trade_date",
        "isin",
        "sedol",
        "organization_id",
        "perm_id",
        "price_currency",
    ]
    return data[columns].reset_index(drop=True)


def extract_security_master_discovery(
    discovery_client: Any,
    get_data_client: Any | None = None,
    country_codes: tuple[str, ...] = DEFAULT_DISCOVERY_COUNTRY_CODES,
    state_buckets: tuple[str, ...] = ("active", "inactive"),
    top: int = 1000,
    start_skip: int = 0,
    max_pages: int | None = None,
    max_rows: int | None = None,
    enrichment_chunk_size: int = 100,
    sleep_seconds: float = 0.0,
    primary_only: bool = False,
    extra_filter: str | None = None,
) -> SecurityMasterDiscoveryExtraction:
    search_hits, page_log, pagination_state = search_security_master_discovery(
        discovery_client=discovery_client,
        country_codes=country_codes,
        state_buckets=state_buckets,
        top=top,
        start_skip=start_skip,
        max_pages=max_pages,
        max_rows=max_rows,
        sleep_seconds=sleep_seconds,
        primary_only=primary_only,
        extra_filter=extra_filter,
    )
    enrichment = fetch_security_master_discovery_enrichment(
        get_data_client=get_data_client,
        search_hits=search_hits,
        chunk_size=enrichment_chunk_size,
        sleep_seconds=sleep_seconds,
    )
    security_master = merge_security_master_discovery_data(search_hits, enrichment)
    return SecurityMasterDiscoveryExtraction(
        search_hits=search_hits,
        enrichment=enrichment,
        security_master=security_master,
        page_log=page_log,
        pagination_state=pagination_state,
    )


def merge_security_master_discovery_data(search_hits: pd.DataFrame, enrichment: pd.DataFrame | None = None) -> pd.DataFrame:
    if search_hits.empty:
        return pd.DataFrame()

    base = search_hits.copy()
    enrich = enrichment.copy() if enrichment is not None else pd.DataFrame()
    if not enrich.empty:
        base = base.merge(enrich, on="ric", how="left", suffixes=("", "_enriched_duplicate"))
    else:
        for column in _empty_enrichment().columns:
            if column != "ric":
                base[column] = pd.NA

    base["common_name"] = _coalesce(base, ["common_name", "common_name_enriched"])
    base["exchange_name"] = _coalesce(base, ["exchange_name", "exchange_name_enriched"])
    base["asset_state_name"] = _coalesce(base, ["asset_state_name", "asset_status"])
    base["first_trade_date"] = _normalize_date_series(_coalesce(base, ["first_trade_date"]))
    base["retire_date"] = _normalize_date_series(_coalesce(base, ["retire_date"]))
    base["price_currency"] = _coalesce(base, ["price_currency"])
    base["country_code"] = _string_series(_coalesce(base, ["country_code", "exchange_country"])).str.upper()
    base["country"] = base["country_code"].map(DISCOVERY_COUNTRY_NAMES)
    base["ticker_symbol"] = _coalesce(base, ["ticker_symbol", "ric"]).map(_ticker_from_ric)
    base["active_state_bucket"] = base["active_state_bucket"].where(
        base["active_state_bucket"].notna(),
        base["asset_state"].map(_active_bucket_from_asset_state),
    )
    base["active_flag"] = base["active_state_bucket"].astype(str).eq("active")
    base["sid"] = base.apply(_sid_for_row, axis=1)
    base["entity_id"] = base.apply(_entity_id_for_row, axis=1)
    return base.reset_index(drop=True)


def build_security_master_discovery_tables(
    security_master: pd.DataFrame,
    requested_country_codes: tuple[str, ...] = DEFAULT_DISCOVERY_COUNTRY_CODES,
    source_loaded_at_utc: str | None = None,
    asof_date: str | None = None,
) -> SecurityMasterDiscoveryTables:
    source_loaded_at_utc = source_loaded_at_utc or _now_utc()
    asof_date = asof_date or datetime.now(timezone.utc).date().isoformat()
    base = security_master.copy()
    if base.empty:
        quality = _build_security_master_quality(
            base,
            pd.DataFrame(columns=IDENTIFIER_HISTORY_COLUMNS),
            requested_country_codes=requested_country_codes,
        )
        return SecurityMasterDiscoveryTables(
            security_master=base,
            entities=pd.DataFrame(columns=ENTITIES_COLUMNS),
            identifier_history=pd.DataFrame(columns=IDENTIFIER_HISTORY_COLUMNS),
            listing_windows=pd.DataFrame(columns=LISTING_WINDOWS_COLUMNS),
            universe_membership=pd.DataFrame(columns=UNIVERSE_MEMBERSHIP_COLUMNS),
            quality=quality,
        )

    for column in ("first_trade_date", "retire_date"):
        if column in base.columns:
            base[column] = _normalize_date_series(base[column])
    base["source_loaded_at_utc"] = source_loaded_at_utc

    entities = _build_entities(base)
    identifier_history = _build_identifier_history(base)
    listing_windows = _build_listing_windows(base)
    universe_membership = _build_universe_membership(base, asof_date=asof_date)
    quality = _build_security_master_quality(base, identifier_history, requested_country_codes=requested_country_codes)

    return SecurityMasterDiscoveryTables(
        security_master=base,
        entities=entities,
        identifier_history=identifier_history,
        listing_windows=listing_windows,
        universe_membership=universe_membership,
        quality=quality,
    )


def write_refinitiv_security_master_discovery_artifacts(
    discovery_client: Any,
    get_data_client: Any,
    artifact_root: str | Path = "artifacts/data_lake/pit/security_master_history",
    run_id: str | None = None,
    country_codes: tuple[str, ...] = DEFAULT_DISCOVERY_COUNTRY_CODES,
    state_buckets: tuple[str, ...] = ("active", "inactive"),
    top: int = 1000,
    start_skip: int = 0,
    max_pages: int | None = None,
    max_rows: int | None = None,
    enrichment_chunk_size: int = 100,
    sleep_seconds: float = 0.0,
    primary_only: bool = False,
    extra_filter: str | None = None,
) -> SecurityMasterDiscoveryArtifacts:
    run_id = run_id or datetime.now(timezone.utc).strftime("REFINITIV-SECURITY-MASTER-DISCOVERY-%Y%m%d-%H%M%S")
    codes = _normalize_country_codes(country_codes)
    artifact_dir = Path(artifact_root) / run_id
    quality_dir = artifact_dir / "quality"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    quality_dir.mkdir(parents=True, exist_ok=True)

    extraction = extract_security_master_discovery(
        discovery_client=discovery_client,
        get_data_client=get_data_client,
        country_codes=codes,
        state_buckets=state_buckets,
        top=top,
        start_skip=start_skip,
        max_pages=max_pages,
        max_rows=max_rows,
        enrichment_chunk_size=enrichment_chunk_size,
        sleep_seconds=sleep_seconds,
        primary_only=primary_only,
        extra_filter=extra_filter,
    )
    source_loaded_at_utc = _now_utc()
    tables = build_security_master_discovery_tables(
        extraction.security_master,
        requested_country_codes=codes,
        source_loaded_at_utc=source_loaded_at_utc,
    )

    search_hits_path = artifact_dir / "discovery_search_hits.csv"
    enrichment_path = artifact_dir / "discovery_enrichment.csv"
    page_log_path = artifact_dir / "discovery_page_log.csv"
    security_master_path = artifact_dir / "security_master.csv"
    entities_path = artifact_dir / "entities.parquet"
    identifier_history_path = artifact_dir / "identifier_history.parquet"
    listing_windows_path = artifact_dir / "listing_windows.parquet"
    universe_membership_path = artifact_dir / "universe_membership_pit.parquet"
    quality_path = quality_dir / "security_master_pit_quality.csv"
    manifest_path = artifact_dir / "manifest.json"

    extraction.search_hits.to_csv(search_hits_path, index=False)
    extraction.enrichment.to_csv(enrichment_path, index=False)
    extraction.page_log.to_csv(page_log_path, index=False)
    tables.security_master.to_csv(security_master_path, index=False)
    tables.entities.to_parquet(entities_path, index=False)
    tables.identifier_history.to_parquet(identifier_history_path, index=False)
    tables.listing_windows.to_parquet(listing_windows_path, index=False)
    tables.universe_membership.to_parquet(universe_membership_path, index=False)
    tables.quality.to_csv(quality_path, index=False)

    manifest = build_security_master_discovery_manifest(
        run_id=run_id,
        artifact_dir=artifact_dir,
        security_master=tables.security_master,
        quality=tables.quality,
        requested_country_codes=codes,
        pagination_state=extraction.pagination_state,
        source_loaded_at_utc=source_loaded_at_utc,
        primary_only=primary_only,
        extra_filter=extra_filter,
        artifacts={
            "discovery_search_hits": search_hits_path,
            "discovery_enrichment": enrichment_path,
            "discovery_page_log": page_log_path,
            "security_master": security_master_path,
            "entities": entities_path,
            "identifier_history": identifier_history_path,
            "listing_windows": listing_windows_path,
            "universe_membership_pit": universe_membership_path,
            "quality": quality_path,
        },
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SecurityMasterDiscoveryArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        search_hits_path=search_hits_path,
        enrichment_path=enrichment_path,
        page_log_path=page_log_path,
        security_master_path=security_master_path,
        entities_path=entities_path,
        identifier_history_path=identifier_history_path,
        listing_windows_path=listing_windows_path,
        universe_membership_path=universe_membership_path,
        quality_path=quality_path,
        manifest_path=manifest_path,
        search_hits=extraction.search_hits,
        enrichment=extraction.enrichment,
        page_log=extraction.page_log,
        entities=tables.entities,
        identifier_history=tables.identifier_history,
        listing_windows=tables.listing_windows,
        universe_membership=tables.universe_membership,
        quality=tables.quality,
        manifest=manifest,
    )


write_security_master_discovery_artifacts = write_refinitiv_security_master_discovery_artifacts


def build_security_master_discovery_manifest(
    run_id: str,
    artifact_dir: Path,
    security_master: pd.DataFrame,
    quality: pd.DataFrame,
    requested_country_codes: tuple[str, ...],
    pagination_state: dict[str, Any],
    source_loaded_at_utc: str,
    artifacts: dict[str, Path],
    primary_only: bool = False,
    extra_filter: str | None = None,
) -> dict[str, Any]:
    codes = _normalize_country_codes(requested_country_codes)
    rows = int(len(security_master))
    country_codes_with_rows = (
        sorted(security_master["country_code"].dropna().astype(str).unique().tolist())
        if not security_master.empty and "country_code" in security_master.columns
        else []
    )
    countries_with_rows = [DISCOVERY_COUNTRY_NAMES.get(code, code) for code in country_codes_with_rows]
    active_rows = _bucket_count(security_master, "active")
    dead_name_rows = _bucket_count(security_master, "inactive")
    country_coverage_complete = set(codes).issubset(set(country_codes_with_rows))
    rows_exist = rows > 0
    capped = bool(pagination_state.get("stopped_by"))
    complete = rows_exist and country_coverage_complete and dead_name_rows > 0 and not capped
    status = "complete" if complete else "partial" if rows_exist else "diagnostic"
    first_dates = _date_values(security_master, "first_trade_date")
    retire_dates = _date_values(security_master, "retire_date")

    artifact_paths = {name: Path(path) for name, path in artifacts.items()}
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": status,
        "rows": rows,
        "security_master_rows": rows,
        "active_rows": int(active_rows),
        "inactive_rows": int(dead_name_rows),
        "dead_name_rows": int(dead_name_rows),
        "delisted_rows": int(dead_name_rows),
        "historical_delisted_names": int(dead_name_rows),
        "requested_country_codes": list(codes),
        "requested_countries": [DISCOVERY_COUNTRY_NAMES.get(code, code) for code in codes],
        "primary_only": bool(primary_only),
        "extra_filter": extra_filter,
        "country_codes_with_rows": country_codes_with_rows,
        "countries": countries_with_rows,
        "country_coverage_complete": bool(country_coverage_complete),
        "date_min": str(first_dates.min().date()) if not first_dates.empty else "",
        "first_trade_date_min": str(first_dates.min().date()) if not first_dates.empty else "",
        "retire_date_min": str(retire_dates.min().date()) if not retire_dates.empty else "",
        "retire_date_max": str(retire_dates.max().date()) if not retire_dates.empty else "",
        "quality_rows": int(len(quality)),
        "pagination": pagination_state,
        "diagnostics": {
            "rows_exist": bool(rows_exist),
            "all_requested_countries_have_rows": bool(country_coverage_complete),
            "inactive_dead_rows_gt_zero": bool(dead_name_rows > 0),
            "pagination_capped": bool(capped),
            "stopped_by": pagination_state.get("stopped_by"),
        },
        "artifacts": {name: str(path.resolve()) for name, path in artifact_paths.items()},
        "output_sha256": {
            name: _sha256_file(path) for name, path in artifact_paths.items()
        },
        "generated_at_utc": source_loaded_at_utc,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_entities(base: pd.DataFrame) -> pd.DataFrame:
    entities = pd.DataFrame(
        {
            "entity_id": base["entity_id"],
            "sid": base["sid"],
            "ric": base["ric"],
            "instrument_permid": _coalesce(base, ["instrument_permid"]),
            "perm_id": _coalesce(base, ["perm_id"]),
            "organization_id": _coalesce(base, ["organization_id"]),
            "issuer_name": _coalesce(base, ["common_name", "document_title"]),
            "document_title": _coalesce(base, ["document_title"]),
            "country": _coalesce(base, ["country"]),
            "country_code": _coalesce(base, ["country_code"]),
            "exchange_country": _coalesce(base, ["exchange_country"]),
            "primary_exchange": _coalesce(base, ["exchange_name"]),
            "exchange_code": _coalesce(base, ["exchange_code"]),
            "ticker_symbol": _coalesce(base, ["ticker_symbol"]),
            "security_type": "ordinary_equity",
            "asset_state": _coalesce(base, ["asset_state"]),
            "asset_state_name": _coalesce(base, ["asset_state_name"]),
            "active_state_bucket": _coalesce(base, ["active_state_bucket"]),
            "source_vendor": "refinitiv_discovery_search",
            "entity_source_scope": "equity_quotes_active_and_inactive_ordinary_shares",
            "source_loaded_at_utc": base["source_loaded_at_utc"],
        }
    )
    return entities[list(ENTITIES_COLUMNS)].drop_duplicates("entity_id", keep="first").reset_index(drop=True)


def _build_identifier_history(base: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    id_specs = (
        ("ric", "ric"),
        ("ticker", "ticker_symbol"),
        ("isin", "isin"),
        ("sedol", "sedol"),
        ("instrument_permid", "instrument_permid"),
        ("organization_id", "organization_id"),
        ("perm_id", "perm_id"),
    )
    for _, item in base.iterrows():
        for id_type, column in id_specs:
            value = item.get(column, pd.NA)
            if not _valid_identifier(value):
                continue
            rows.append(
                {
                    "entity_id": item["entity_id"],
                    "sid": item["sid"],
                    "id_type": id_type,
                    "id_value": str(value).strip(),
                    "effective_from": item.get("first_trade_date", pd.NA),
                    "effective_to": item.get("retire_date", pd.NA) if item.get("active_state_bucket") != "active" else pd.NA,
                    "change_type": "security_master_discovery_identifier",
                    "change_reason": "refinitiv discovery search ordinary equity route",
                    "source_timestamp_utc": pd.NA,
                    "source_loaded_at_utc": item["source_loaded_at_utc"],
                    "evidence_level": "discovery_search_plus_get_data",
                    "eligible_for_model_denominator": True,
                }
            )
    if not rows:
        return pd.DataFrame(columns=IDENTIFIER_HISTORY_COLUMNS)
    return (
        pd.DataFrame(rows)[list(IDENTIFIER_HISTORY_COLUMNS)]
        .drop_duplicates(["entity_id", "id_type", "id_value"], keep="first")
        .reset_index(drop=True)
    )


def _build_listing_windows(base: pd.DataFrame) -> pd.DataFrame:
    active = base["active_state_bucket"].astype(str).eq("active")
    listing_status = _coalesce(base, ["asset_state_name", "asset_status", "active_state_bucket"])
    delisting_date = base["retire_date"].where(~active, pd.NA)
    listing_windows = pd.DataFrame(
        {
            "entity_id": base["entity_id"],
            "sid": base["sid"],
            "ric": base["ric"],
            "ticker": base["ticker_symbol"],
            "country": base["country"],
            "country_code": base["country_code"],
            "exchange": _coalesce(base, ["exchange_name"]),
            "exchange_code": _coalesce(base, ["exchange_code"]),
            "currency": _coalesce(base, ["price_currency"]),
            "listing_date": base["first_trade_date"],
            "first_trade_date": base["first_trade_date"],
            "effective_from": base["first_trade_date"],
            "effective_to": delisting_date,
            "delisting_date": delisting_date,
            "last_trade_date": delisting_date,
            "listing_status": listing_status,
            "asset_state": _coalesce(base, ["asset_state"]),
            "asset_state_name": _coalesce(base, ["asset_state_name"]),
            "delisting_reason": pd.NA,
            "tradable_flag": active,
            "active_flag": active,
            "window_source": "refinitiv_discovery_equity_quotes_search",
            "evidence_level": "discovery_search_plus_get_data",
            "eligible_for_model_denominator": True,
            "source_loaded_at_utc": base["source_loaded_at_utc"],
        }
    )
    return listing_windows[list(LISTING_WINDOWS_COLUMNS)].reset_index(drop=True)


def _build_universe_membership(base: pd.DataFrame, asof_date: str) -> pd.DataFrame:
    active = base["active_state_bucket"].astype(str).eq("active")
    membership_end = base["retire_date"].where(~active, pd.NA)
    membership = pd.DataFrame(
        {
            "asof_date": asof_date,
            "entity_id": base["entity_id"],
            "sid": base["sid"],
            "ric_asof": base["ric"],
            "ticker_asof": base["ticker_symbol"],
            "country": base["country"],
            "country_code": base["country_code"],
            "exchange": _coalesce(base, ["exchange_name"]),
            "currency": _coalesce(base, ["price_currency"]),
            "membership_start": base["first_trade_date"],
            "membership_end": membership_end,
            "is_listed_asof": active,
            "is_investable_asof": active,
            "eligible_flag": True,
            "in_discovery_security_master_flag": True,
            "exclusion_reason": pd.NA,
            "known_at_utc": pd.NA,
            "membership_source": "refinitiv_discovery_equity_quotes_search",
            "source_loaded_at_utc": base["source_loaded_at_utc"],
        }
    )
    membership.loc[~active, "exclusion_reason"] = "inactive_or_retired_name_retained_for_pit_denominator"
    return membership[list(UNIVERSE_MEMBERSHIP_COLUMNS)].reset_index(drop=True)


def _build_security_master_quality(
    base: pd.DataFrame,
    identifier_history: pd.DataFrame,
    requested_country_codes: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    codes = _normalize_country_codes(requested_country_codes)
    for code in codes:
        group = (
            base[base["country_code"].astype("string").eq(code)].copy()
            if not base.empty and "country_code" in base.columns
            else pd.DataFrame()
        )
        identifier_rows = (
            identifier_history[identifier_history["sid"].astype("string").str.startswith(f"{code}:")]
            if not identifier_history.empty and "sid" in identifier_history.columns
            else pd.DataFrame()
        )
        row_count = int(len(group))
        active_count = _bucket_count(group, "active")
        dead_count = _bucket_count(group, "inactive")
        first_dates = _date_values(group, "first_trade_date")
        retire_dates = _date_values(group, "retire_date")
        asof_year = int(first_dates.min().year) if not first_dates.empty else datetime.now(timezone.utc).year
        rics = _string_series(group["ric"]) if row_count and "ric" in group.columns else pd.Series(dtype="string")
        rows.append(
            {
                "country": DISCOVERY_COUNTRY_NAMES.get(code, code),
                "country_code": code,
                "asof_year": asof_year,
                "row_count": row_count,
                "active_count": int(active_count),
                "delisted_count": int(dead_count),
                "dead_name_count": int(dead_count),
                "identifier_change_count": 0,
                "identifier_rows": int(len(identifier_rows)),
                "listing_window_rows": row_count,
                "null_ric_rate": float(rics.isna().mean()) if row_count else 1.0,
                "dead_name_coverage_rate": float(dead_count / row_count) if row_count else 0.0,
                "first_trade_date_min": str(first_dates.min().date()) if not first_dates.empty else "",
                "retire_date_min": str(retire_dates.min().date()) if not retire_dates.empty else "",
                "retire_date_max": str(retire_dates.max().date()) if not retire_dates.empty else "",
                "status": "complete" if row_count > 0 else "missing",
            }
        )
    return pd.DataFrame(rows)[list(QUALITY_COLUMNS)]


def _call_discovery_search(
    discovery_client: Any,
    view: Any,
    query_filter: str,
    select: str,
    top: int,
    skip: int,
    search_callable: Callable[..., Any] | None = None,
) -> pd.DataFrame:
    discovery = _discovery_namespace(discovery_client)
    search = search_callable or getattr(discovery, "search", None)
    if callable(search):
        attempts = (
            {"view": view, "filter": query_filter, "select": select, "top": top, "skip": skip},
            {"query": "", "view": view, "filter": query_filter, "select": select, "top": top, "skip": skip},
            {"view": view, "filter": query_filter, "select": list(DISCOVERY_SEARCH_FIELDS), "top": top, "skip": skip},
        )
        last_type_error: TypeError | None = None
        for kwargs in attempts:
            try:
                return _coerce_to_frame(search(**kwargs))
            except TypeError as exc:
                last_type_error = exc
        if last_type_error is not None:
            raise last_type_error

    search_cls = getattr(discovery, "Search", None)
    if callable(search_cls):
        search_obj = search_cls(view=view, filter=query_filter, select=select, top=top, skip=skip)
        return _coerce_to_frame(search_obj)

    raise TypeError("discovery_client must expose discovery.search(...) or discovery.Search(...)")


def _coerce_to_frame(value: Any) -> pd.DataFrame:
    if value is None:
        return pd.DataFrame()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    for attr in ("to_dataframe", "to_frame", "get_data"):
        method = getattr(value, attr, None)
        if callable(method):
            return _coerce_to_frame(method())
    data = getattr(value, "data", None)
    if data is not None:
        if isinstance(data, pd.DataFrame):
            return data.copy()
        for attr in ("df", "raw", "data"):
            nested = getattr(data, attr, None)
            if nested is not None:
                return _coerce_to_frame(nested)
        if isinstance(data, dict):
            return _coerce_to_frame(data)
    if isinstance(value, dict):
        for key in ("hits", "data", "results", "value"):
            if key in value:
                return pd.DataFrame(value[key])
        return pd.DataFrame(value)
    return pd.DataFrame(value)


def _rename_columns(frame: pd.DataFrame, aliases: dict[str, str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns_by_key = {_column_key(column): column for column in frame.columns}
    rename: dict[Any, str] = {}
    for source, target in aliases.items():
        if source in frame.columns:
            rename[source] = target
            continue
        match = columns_by_key.get(_column_key(source))
        if match is not None:
            rename[match] = target
    return frame.rename(columns=rename)


def _column_key(value: Any) -> str:
    if isinstance(value, tuple):
        value = " ".join(str(part) for part in value if part is not None)
    return "".join(ch for ch in str(value).casefold() if ch.isalnum())


def _normalize_country_codes(country_codes: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    codes: list[str] = []
    for code in country_codes:
        codes.append(_normalize_country_code(code))
    return tuple(dict.fromkeys(codes))


def _normalize_country_code(country_code: str) -> str:
    code = str(country_code).upper().strip()
    aliases = {
        "MY": "MYS",
        "MALAYSIA": "MYS",
        "SG": "SGP",
        "SINGAPORE": "SGP",
        "ID": "IDN",
        "INDONESIA": "IDN",
        "CN": "CHN",
        "CHINA": "CHN",
        "HK": "HKG",
        "HONGKONG": "HKG",
        "HONG KONG": "HKG",
        "TW": "TWN",
        "TAIWAN": "TWN",
        "KR": "KOR",
        "KOREA": "KOR",
        "SOUTH KOREA": "KOR",
        "JP": "JPN",
        "JAPAN": "JPN",
        "US": "USA",
        "UNITED STATES": "USA",
        "UNITED STATES OF AMERICA": "USA",
        "TH": "THA",
        "THAILAND": "THA",
    }
    code = aliases.get(code, code)
    if code not in SUPPORTED_DISCOVERY_COUNTRY_CODES:
        raise ValueError(f"Unsupported Discovery country code: {country_code!r}")
    return code


def _discovery_namespace(client: Any) -> Any:
    discovery = getattr(client, "discovery", None)
    return discovery if discovery is not None else client


def _empty_search_hits() -> pd.DataFrame:
    return pd.DataFrame(columns=SEARCH_NORMALIZED_COLUMNS)


def _empty_enrichment() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "ric",
            "ric_enriched",
            "common_name_enriched",
            "exchange_name_enriched",
            "exchange_country_code",
            "asset_status",
            "retire_date",
            "first_trade_date",
            "isin",
            "sedol",
            "organization_id",
            "perm_id",
            "price_currency",
        ]
    )


def _active_bucket_from_asset_state(value: Any) -> str:
    if pd.isna(value):
        return "unknown"
    text = str(value).strip().upper()
    if not text:
        return "unknown"
    return "active" if text == "AC" else "inactive"


def _to_bool(value: Any) -> bool | pd.NA:
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return pd.NA


def _normalize_date_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        series = values
    else:
        series = pd.Series(values)
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    normalized = parsed.dt.strftime("%Y-%m-%d")
    return normalized.where(parsed.notna(), pd.NA).reset_index(drop=True)


def _string_series(values: Any) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    text = series.astype("string").str.strip()
    return text.mask(text.eq(""))


def _chunks(values: list[str], chunk_size: int) -> list[list[str]]:
    return [values[idx : idx + chunk_size] for idx in range(0, len(values), chunk_size)]


def _coalesce(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    out = pd.Series([pd.NA] * len(frame), index=frame.index, dtype="object")
    for column in columns:
        if column in frame.columns:
            out = out.where(out.notna(), frame[column])
    return out.reset_index(drop=True)


def _ticker_from_ric(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    return text.split(".")[0]


def _sid_for_row(row: pd.Series) -> str:
    code = row.get("country_code")
    ric = row.get("ric")
    return f"{code}:{ric}"


def _entity_id_for_row(row: pd.Series) -> str:
    for prefix, column in (
        ("instrument_permid", "instrument_permid"),
        ("perm_id", "perm_id"),
        ("organization_id", "organization_id"),
        ("ric", "ric"),
    ):
        value = row.get(column, pd.NA)
        if _valid_identifier(value):
            return f"{prefix}:{str(value).strip()}"
    return f"row:{row.name}"


def _valid_identifier(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text and text.casefold() not in {"nan", "none", "<na>", "nat"})


def _bucket_count(frame: pd.DataFrame, bucket: str) -> int:
    if frame.empty or "active_state_bucket" not in frame.columns:
        return 0
    return int(frame["active_state_bucket"].astype(str).eq(bucket).sum())


def _date_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame.empty or column not in frame.columns:
        return pd.Series(dtype="datetime64[ns]")
    return pd.to_datetime(frame[column], errors="coerce").dropna()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DEFAULT_DISCOVERY_COUNTRY_CODES",
    "DISCOVERY_COUNTRY_NAMES",
    "DISCOVERY_ENRICHMENT_FIELDS",
    "DISCOVERY_SEARCH_FIELDS",
    "DISCOVERY_SEARCH_SELECT",
    "SUPPORTED_DISCOVERY_COUNTRY_CODES",
    "SecurityMasterDiscoveryArtifacts",
    "SecurityMasterDiscoveryExtraction",
    "SecurityMasterDiscoveryTables",
    "build_security_master_discovery_manifest",
    "build_security_master_discovery_tables",
    "discovery_equity_quotes_view",
    "extract_security_master_discovery",
    "fetch_security_master_discovery_enrichment",
    "merge_security_master_discovery_data",
    "normalize_discovery_search_hits",
    "normalize_security_master_enrichment",
    "search_security_master_discovery",
    "security_master_discovery_filter",
    "write_refinitiv_security_master_discovery_artifacts",
    "write_security_master_discovery_artifacts",
]
