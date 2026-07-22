from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


USA_ACTIVE_CORE_EXCHANGE_CODES = frozenset({"NYQ", "NSQ", "NMQ", "NAQ", "ASQ"})
USA_LISTED_REFERENCE_EXCHANGE_CODES = frozenset(
    {
        "NYQ",
        "NYS",
        "NSQ",
        "NMQ",
        "NAQ",
        "NMS",
        "NAS",
        "ASQ",
        "ASE",
    }
)
USA_COMMON_SHARE_CATEGORIES = frozenset(
    {
        "Ordinary Share",
        "Fully Paid Ordinary Share",
    }
)


@dataclass(frozen=True)
class UsaResearchUniverseArtifacts:
    run_id: str
    artifact_dir: Path
    active_symbol_map_path: Path
    reference_symbol_map_path: Path
    manifest_path: Path
    manifest: dict[str, object]


@dataclass(frozen=True)
class UsaLargeLiquidUniverseArtifacts:
    run_id: str
    artifact_dir: Path
    snapshot_request_path: Path
    snapshot_field_map_path: Path
    large_liquid_symbol_map_path: Path | None
    manifest_path: Path
    manifest: dict[str, object]


USA_LARGE_LIQUID_UNIVERSE_ID = "USA_LARGE_LIQUID_CORE_V1"
USA_LARGE_LIQUID_DEFAULT_MIN_MARKET_CAP_USD = 1_000_000_000.0
USA_LARGE_LIQUID_DEFAULT_MIN_ADV_VALUE_USD = 5_000_000.0
USA_LARGE_LIQUID_DEFAULT_MIN_FREE_FLOAT_PCT = 10.0


def load_security_master(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.casefold() == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source, low_memory=False)


def _series(data: pd.DataFrame, column: str, default: object = pd.NA) -> pd.Series:
    if column in data.columns:
        return data[column]
    return pd.Series(default, index=data.index, dtype="object")


def _coalesce(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=data.index, dtype="object")
    for column in columns:
        if column in data.columns:
            result = result.where(result.notna(), data[column])
    return result


def _coalesce_numeric(data: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=data.index, dtype="Float64")
    for column in columns:
        if column in data.columns:
            values = pd.to_numeric(data[column], errors="coerce")
            result = result.where(result.notna(), values)
    return result


def _bool_series(data: pd.DataFrame, column: str) -> pd.Series:
    values = _series(data, column, False)
    if values.dtype == bool:
        return values.fillna(False)
    normalized = values.astype("string").str.strip().str.casefold()
    return normalized.isin({"true", "1", "yes", "y"})


def _base_usa_common_shares(security_master: pd.DataFrame) -> pd.DataFrame:
    if "ric" not in security_master.columns:
        raise ValueError("security master must include a ric column")
    data = security_master.copy()
    country_code = _series(data, "country_code").astype("string").str.strip().str.upper()
    country = _series(data, "country").astype("string").str.strip().str.casefold()
    asset_leaf = _series(data, "rcs_asset_category_leaf").astype("string").str.strip()
    ric = _series(data, "ric").astype("string").str.strip()
    base = data.loc[
        (country_code.eq("USA") | country.eq("united states"))
        & asset_leaf.isin(USA_COMMON_SHARE_CATEGORIES)
        & ric.notna()
        & ric.ne("")
    ].copy()
    base["ric"] = base["ric"].astype("string").str.strip()
    return base.drop_duplicates("ric", keep="first").copy()


def build_usa_active_liquid_core(security_master: pd.DataFrame) -> pd.DataFrame:
    """Current operational universe for IBKR-grade US listed common stocks.

    This is not point-in-time Russell membership. It is a current active,
    primary-issue, primary-RIC venue filter that excludes OTC, pink sheet,
    bulletin-board, and secondary feed routes.
    """

    data = _base_usa_common_shares(security_master)
    exchange_code = _series(data, "exchange_code").astype("string").str.strip().str.upper()
    active_bucket = _series(data, "active_state_bucket").astype("string").str.strip().str.casefold()
    primary_ric = _bool_series(data, "is_primary_ric")
    primary_issue = _bool_series(data, "is_primary_issue_ric")
    filtered = data.loc[
        exchange_code.isin(USA_ACTIVE_CORE_EXCHANGE_CODES)
        & active_bucket.eq("active")
        & primary_ric
        & primary_issue
    ].copy()
    filtered["research_universe_role"] = "active_ibkr_operational_core"
    filtered["research_universe_reason"] = (
        "active primary US ordinary share on NYSE/Nasdaq/NYSE American consolidated venue"
    )
    return filtered.sort_values(["exchange_code", "ticker_symbol", "ric"], na_position="last").reset_index(drop=True)


def build_usa_listed_reference_core(security_master: pd.DataFrame) -> pd.DataFrame:
    """Backtest denominator candidate for Russell-suitable US listed venues.

    Inactive/dead names are retained to reduce survivorship bias. The output is
    still not a substitute for point-in-time Russell constituent membership.
    """

    data = _base_usa_common_shares(security_master)
    exchange_code = _series(data, "exchange_code").astype("string").str.strip().str.upper()
    active_bucket = _series(data, "active_state_bucket").astype("string").str.strip().str.casefold()
    primary_ric = _bool_series(data, "is_primary_ric")
    primary_issue = _bool_series(data, "is_primary_issue_ric")
    active_primary_or_dead = active_bucket.ne("active") | (primary_ric & primary_issue)
    filtered = data.loc[exchange_code.isin(USA_LISTED_REFERENCE_EXCHANGE_CODES) & active_primary_or_dead].copy()
    filtered["research_universe_role"] = "listed_reference_backtest_denominator"
    filtered["research_universe_reason"] = (
        "active and inactive US ordinary shares on NYSE/Nasdaq/NYSE American primary or legacy listed venues"
    )
    return filtered.sort_values(["exchange_code", "active_state_bucket", "ticker_symbol", "ric"], na_position="last").reset_index(
        drop=True
    )


def usa_large_liquid_snapshot_field_map() -> pd.DataFrame:
    """Accepted Refinitiv snapshot fields used to gate the USA large-liquid core."""

    rows = [
        ("market_cap", "TR.CompanyMarketCap", "Company Market Cap", True),
        ("adv_value", "TR.AvgDailyValTraded30D", "Average Daily Value Traded - 30 Days", True),
        ("free_float", "TR.FreeFloatPct", "Free Float (Percent)", True),
        ("shares", "TR.SharesOutstanding", "Outstanding Shares", False),
    ]
    return pd.DataFrame(
        [
            {
                "dataset_id": "liquidity_market_cap_free_float",
                "owner_agent": "usa_large_liquid_universe_agent",
                "source_family": "refinitiv_reference_snapshot",
                "field_role": field_role,
                "candidate_field": candidate_field,
                "returned_data_columns": returned_column,
                "required": required,
                "acceptance_status": "accepted_from_asean_live_probe_pending_usa_snapshot",
            }
            for field_role, candidate_field, returned_column, required in rows
        ]
    )


def snapshot_request_from_symbol_map(symbol_map: pd.DataFrame) -> pd.DataFrame:
    """Convert a Refinitiv symbol map into the selected-universe shape used by snapshot extractors."""

    if symbol_map.empty:
        return pd.DataFrame(
            columns=[
                "refinitiv_ric",
                "sid",
                "ticker",
                "local_symbol",
                "name",
                "country",
                "exchange",
                "currency",
                "selected_for_u2",
                "selected_for_u3",
            ]
        )
    data = symbol_map.copy()
    requested = pd.DataFrame(
        {
            "refinitiv_ric": _coalesce(data, ["refinitiv_ric", "ric"]).astype("string").str.strip(),
            "ticker": _coalesce(data, ["ticker", "ticker_symbol", "local_symbol", "refinitiv_ric"]),
            "local_symbol": _coalesce(data, ["ticker", "ticker_symbol", "local_symbol", "refinitiv_ric"]),
            "name": _coalesce(data, ["name", "common_name", "document_title"]),
            "country": _coalesce(data, ["country"]),
            "exchange": _coalesce(data, ["exchange", "exchange_name", "exchange_code"]),
            "currency": _coalesce(data, ["currency", "price_currency"]),
            "selected_for_u2": True,
            "selected_for_u3": True,
        }
    )
    requested = requested.loc[requested["refinitiv_ric"].notna() & requested["refinitiv_ric"].ne("")].copy()
    requested["sid"] = (
        requested["country"].astype("string").fillna("USA")
        + ":"
        + requested["exchange"].astype("string").fillna("")
        + ":"
        + requested["ticker"].astype("string").fillna(requested["refinitiv_ric"])
    )
    return requested.drop_duplicates("refinitiv_ric", keep="first").reset_index(drop=True)


def normalize_usa_liquidity_snapshot(snapshot: pd.DataFrame) -> pd.DataFrame:
    """Normalize Refinitiv liquidity snapshot columns into explicit USD screening fields."""

    if snapshot.empty:
        return pd.DataFrame(
            columns=[
                "refinitiv_ric",
                "market_cap_usd",
                "adv_value_usd",
                "free_float_pct",
                "shares_outstanding",
                "liquidity_snapshot_asof_date",
                "liquidity_snapshot_loaded_at_utc",
            ]
        )
    data = snapshot.copy()
    normalized = pd.DataFrame(
        {
            "refinitiv_ric": _coalesce(data, ["requested_ric", "refinitiv_ric", "Instrument", "instrument"]).astype("string").str.strip(),
            "market_cap_usd": _coalesce_numeric(data, ["market_cap_usd", "Company Market Cap", "TR.CompanyMarketCap"]),
            "adv_value_usd": _coalesce_numeric(
                data,
                [
                    "adv_value_usd",
                    "Average Daily Value Traded - 30 Days",
                    "TR.AvgDailyValTraded30D",
                ],
            ),
            "free_float_pct": _coalesce_numeric(data, ["free_float_pct", "Free Float (Percent)", "TR.FreeFloatPct"]),
            "shares_outstanding": _coalesce_numeric(data, ["shares_outstanding", "Outstanding Shares", "TR.SharesOutstanding"]),
            "liquidity_snapshot_asof_date": _coalesce(data, ["asof_date", "snapshot_asof_date"]),
            "liquidity_snapshot_loaded_at_utc": _coalesce(data, ["source_loaded_at_utc", "loaded_at_utc"]),
        }
    )
    normalized = normalized.loc[normalized["refinitiv_ric"].notna() & normalized["refinitiv_ric"].ne("")].copy()
    return normalized.drop_duplicates("refinitiv_ric", keep="first").reset_index(drop=True)


def build_usa_large_liquid_core(
    security_master: pd.DataFrame,
    liquidity_snapshot: pd.DataFrame,
    min_market_cap_usd: float = USA_LARGE_LIQUID_DEFAULT_MIN_MARKET_CAP_USD,
    min_adv_value_usd: float = USA_LARGE_LIQUID_DEFAULT_MIN_ADV_VALUE_USD,
    min_free_float_pct: float = USA_LARGE_LIQUID_DEFAULT_MIN_FREE_FLOAT_PCT,
) -> pd.DataFrame:
    """Current large-liquid USA core for bounded operational data-lake buildout.

    This filter is intentionally current-only until historical market-cap and
    liquidity fields are backfilled and lagged. It is suitable for controlling
    near-term extraction scope, not for final point-in-time alpha evidence.
    """

    active_core = build_usa_active_liquid_core(security_master)
    normalized = normalize_usa_liquidity_snapshot(liquidity_snapshot)
    if active_core.empty or normalized.empty:
        return active_core.iloc[0:0].copy()
    merged = active_core.merge(normalized, left_on="ric", right_on="refinitiv_ric", how="left")
    market_cap = pd.to_numeric(merged["market_cap_usd"], errors="coerce")
    adv_value = pd.to_numeric(merged["adv_value_usd"], errors="coerce")
    free_float = pd.to_numeric(merged["free_float_pct"], errors="coerce")
    passes = market_cap.ge(min_market_cap_usd) & adv_value.ge(min_adv_value_usd)
    if min_free_float_pct > 0:
        passes = passes & free_float.ge(min_free_float_pct)
    filtered = merged.loc[passes].copy()
    filtered["research_universe_role"] = "usa_large_liquid_operational_core"
    filtered["research_universe_reason"] = (
        "active primary US ordinary share with current Refinitiv market cap, ADV, and free-float gates"
    )
    filtered["large_liquid_filter_pass"] = True
    sort_cols = ["market_cap_usd", "adv_value_usd", "exchange_code", "ticker_symbol", "ric"]
    return filtered.sort_values(sort_cols, ascending=[False, False, True, True, True], na_position="last").reset_index(drop=True)


def symbol_map_from_security_master_rows(
    data: pd.DataFrame,
    source_path: str | Path,
    universe_id: str,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame(
            columns=[
                "country",
                "country_code",
                "exchange",
                "exchange_name",
                "exchange_code",
                "ticker",
                "name",
                "currency",
                "refinitiv_ric",
                "refinitiv_symbol_status",
                "validated_close_field",
                "candidate_method",
                "symbol_map_source",
                "security_master_active_bucket",
                "security_master_active_flag",
                "first_trade_date",
                "retire_date",
                "instrument_permid",
                "perm_id",
                "organization_id",
                "isin",
                "sedol",
                "research_universe_id",
                "research_universe_role",
                "research_universe_reason",
                "market_cap_usd",
                "adv_value_usd",
                "free_float_pct",
                "shares_outstanding",
                "liquidity_snapshot_asof_date",
                "liquidity_snapshot_loaded_at_utc",
            ]
        )

    rows = data.copy()
    rows["refinitiv_ric"] = rows["ric"].astype("string").str.strip()
    rows = rows.loc[rows["refinitiv_ric"].notna() & rows["refinitiv_ric"].ne("")].copy()
    rows = rows.drop_duplicates("refinitiv_ric", keep="first").copy()

    symbol_map = pd.DataFrame(
        {
            "country": _coalesce(rows, ["country"]),
            "country_code": _coalesce(rows, ["country_code", "exchange_country"]),
            "exchange": _coalesce(rows, ["exchange_code", "exchange_name"]),
            "exchange_name": _coalesce(rows, ["exchange_name"]),
            "exchange_code": _coalesce(rows, ["exchange_code"]),
            "ticker": _coalesce(rows, ["ticker_symbol", "refinitiv_ric"]),
            "name": _coalesce(rows, ["common_name", "document_title"]),
            "currency": _coalesce(rows, ["price_currency"]),
            "refinitiv_ric": rows["refinitiv_ric"],
            "refinitiv_symbol_status": "security_master_discovery_candidate",
            "validated_close_field": pd.NA,
            "candidate_method": "usa_research_universe_filter",
            "symbol_map_source": str(Path(source_path).resolve()),
            "security_master_active_bucket": _coalesce(rows, ["active_state_bucket"]),
            "security_master_active_flag": _coalesce(rows, ["active_flag"]),
            "first_trade_date": _coalesce(rows, ["first_trade_date"]),
            "retire_date": _coalesce(rows, ["retire_date"]),
            "instrument_permid": _coalesce(rows, ["instrument_permid"]),
            "perm_id": _coalesce(rows, ["perm_id"]),
            "organization_id": _coalesce(rows, ["organization_id"]),
            "isin": _coalesce(rows, ["isin"]),
            "sedol": _coalesce(rows, ["sedol"]),
            "research_universe_id": universe_id,
            "research_universe_role": _coalesce(rows, ["research_universe_role"]),
            "research_universe_reason": _coalesce(rows, ["research_universe_reason"]),
            "market_cap_usd": _coalesce_numeric(rows, ["market_cap_usd"]),
            "adv_value_usd": _coalesce_numeric(rows, ["adv_value_usd"]),
            "free_float_pct": _coalesce_numeric(rows, ["free_float_pct"]),
            "shares_outstanding": _coalesce_numeric(rows, ["shares_outstanding"]),
            "liquidity_snapshot_asof_date": _coalesce(rows, ["liquidity_snapshot_asof_date"]),
            "liquidity_snapshot_loaded_at_utc": _coalesce(rows, ["liquidity_snapshot_loaded_at_utc"]),
        }
    )
    sort_cols = ["country", "exchange", "security_master_active_bucket", "ticker", "refinitiv_ric"]
    return symbol_map.sort_values(sort_cols, na_position="last").reset_index(drop=True)


def exchange_filter_summary(security_master: pd.DataFrame) -> pd.DataFrame:
    base = _base_usa_common_shares(security_master)
    active = _series(base, "active_state_bucket").astype("string").str.strip().str.casefold().eq("active")
    grouped = (
        base.assign(_active=active)
        .groupby(["exchange_code", "exchange_name"], dropna=False)
        .agg(rows=("ric", "size"), active_rows=("_active", "sum"))
        .reset_index()
    )
    grouped["active_core_exchange"] = grouped["exchange_code"].isin(USA_ACTIVE_CORE_EXCHANGE_CODES)
    grouped["listed_reference_exchange"] = grouped["exchange_code"].isin(USA_LISTED_REFERENCE_EXCHANGE_CODES)
    return grouped.sort_values(["listed_reference_exchange", "active_core_exchange", "rows"], ascending=[False, False, False])


def _manifest_for_map(run_id: str, artifact_dir: Path, symbol_map_path: Path, symbol_map: pd.DataFrame) -> dict[str, object]:
    active_mask = symbol_map["security_master_active_bucket"].astype("string").str.casefold().eq("active")
    return {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "symbol_map_path": str(symbol_map_path.resolve()),
        "rows": int(len(symbol_map)),
        "active_rows": int(active_mask.sum()) if not symbol_map.empty else 0,
        "inactive_rows": int((~active_mask).sum()) if not symbol_map.empty else 0,
        "exchange_counts": symbol_map["exchange"].value_counts(dropna=False).to_dict() if not symbol_map.empty else {},
        "status_counts": symbol_map["refinitiv_symbol_status"].value_counts(dropna=False).to_dict()
        if not symbol_map.empty
        else {},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_usa_research_universe_artifacts(
    security_master_path: str | Path,
    run_id: str,
    artifact_root: str | Path = "artifacts/research_universes",
    symbol_map_artifact_root: str | Path = "artifacts/refinitiv_symbol_map",
) -> UsaResearchUniverseArtifacts:
    source = Path(security_master_path)
    security_master = load_security_master(source)
    active_core = build_usa_active_liquid_core(security_master)
    reference_core = build_usa_listed_reference_core(security_master)
    active_universe_id = "USA_IBKR_ACTIVE_LIQUID_CORE_V1"
    reference_universe_id = "USA_LISTED_REFERENCE_CORE_V1"
    active_symbol_map = symbol_map_from_security_master_rows(active_core, source, active_universe_id)
    reference_symbol_map = symbol_map_from_security_master_rows(reference_core, source, reference_universe_id)

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    active_symbol_map_path = artifact_dir / "usa_ibkr_active_liquid_core_v1_symbol_map.csv"
    reference_symbol_map_path = artifact_dir / "usa_listed_reference_core_v1_symbol_map.csv"
    exchange_summary_path = artifact_dir / "exchange_filter_summary.csv"
    active_symbol_map.to_csv(active_symbol_map_path, index=False)
    reference_symbol_map.to_csv(reference_symbol_map_path, index=False)
    exchange_filter_summary(security_master).to_csv(exchange_summary_path, index=False)

    active_refinitiv_dir = Path(symbol_map_artifact_root) / f"REFINITIV-SYMBOL-MAP-{active_universe_id}-{run_id}"
    reference_refinitiv_dir = Path(symbol_map_artifact_root) / f"REFINITIV-SYMBOL-MAP-{reference_universe_id}-{run_id}"
    active_refinitiv_dir.mkdir(parents=True, exist_ok=True)
    reference_refinitiv_dir.mkdir(parents=True, exist_ok=True)
    active_refinitiv_symbol_map_path = active_refinitiv_dir / "symbol_map.csv"
    reference_refinitiv_symbol_map_path = reference_refinitiv_dir / "symbol_map.csv"
    active_symbol_map.to_csv(active_refinitiv_symbol_map_path, index=False)
    reference_symbol_map.to_csv(reference_refinitiv_symbol_map_path, index=False)

    active_refinitiv_manifest = _manifest_for_map(
        f"REFINITIV-SYMBOL-MAP-{active_universe_id}-{run_id}",
        active_refinitiv_dir,
        active_refinitiv_symbol_map_path,
        active_symbol_map,
    )
    reference_refinitiv_manifest = _manifest_for_map(
        f"REFINITIV-SYMBOL-MAP-{reference_universe_id}-{run_id}",
        reference_refinitiv_dir,
        reference_refinitiv_symbol_map_path,
        reference_symbol_map,
    )
    (active_refinitiv_dir / "manifest.json").write_text(json.dumps(active_refinitiv_manifest, indent=2), encoding="utf-8")
    (reference_refinitiv_dir / "manifest.json").write_text(
        json.dumps(reference_refinitiv_manifest, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "security_master_source": str(source.resolve()),
        "active_universe_id": active_universe_id,
        "reference_universe_id": reference_universe_id,
        "active_symbol_map_path": str(active_symbol_map_path.resolve()),
        "reference_symbol_map_path": str(reference_symbol_map_path.resolve()),
        "active_refinitiv_symbol_map_path": str(active_refinitiv_symbol_map_path.resolve()),
        "reference_refinitiv_symbol_map_path": str(reference_refinitiv_symbol_map_path.resolve()),
        "exchange_summary_path": str(exchange_summary_path.resolve()),
        "security_master_rows": int(len(security_master)),
        "base_usa_common_share_rows": int(len(_base_usa_common_shares(security_master))),
        "active_core_rows": int(len(active_symbol_map)),
        "reference_core_rows": int(len(reference_symbol_map)),
        "active_core_exchange_counts": active_symbol_map["exchange"].value_counts(dropna=False).to_dict(),
        "reference_core_exchange_counts": reference_symbol_map["exchange"].value_counts(dropna=False).to_dict(),
        "filter_notes": [
            "Active core is for current IBKR-grade operations and excludes inactive/dead names.",
            "Listed reference core retains inactive/dead listed names for backtest denominator control.",
            "Neither output is point-in-time Russell membership until official PIT index constituents are extracted.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return UsaResearchUniverseArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        active_symbol_map_path=active_refinitiv_symbol_map_path,
        reference_symbol_map_path=reference_refinitiv_symbol_map_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )


def write_usa_large_liquid_universe_artifacts(
    security_master_path: str | Path,
    run_id: str,
    liquidity_snapshot_path: str | Path | None = None,
    artifact_root: str | Path = "artifacts/research_universes",
    symbol_map_artifact_root: str | Path = "artifacts/refinitiv_symbol_map",
    min_market_cap_usd: float = USA_LARGE_LIQUID_DEFAULT_MIN_MARKET_CAP_USD,
    min_adv_value_usd: float = USA_LARGE_LIQUID_DEFAULT_MIN_ADV_VALUE_USD,
    min_free_float_pct: float = USA_LARGE_LIQUID_DEFAULT_MIN_FREE_FLOAT_PCT,
) -> UsaLargeLiquidUniverseArtifacts:
    source = Path(security_master_path)
    security_master = load_security_master(source)
    active_core = build_usa_active_liquid_core(security_master)
    reference_core = build_usa_listed_reference_core(security_master)
    active_candidate_map = symbol_map_from_security_master_rows(active_core, source, "USA_IBKR_ACTIVE_LIQUID_CORE_V1")
    snapshot_request = snapshot_request_from_symbol_map(active_candidate_map)
    snapshot_field_map = usa_large_liquid_snapshot_field_map()

    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    snapshot_request_path = artifact_dir / "usa_large_liquid_snapshot_request.csv"
    snapshot_field_map_path = artifact_dir / "snapshot_field_map.csv"
    snapshot_request.to_csv(snapshot_request_path, index=False)
    snapshot_field_map.to_csv(snapshot_field_map_path, index=False)

    large_symbol_map_path: Path | None = None
    large_core_rows = 0
    large_core_exchange_counts: dict[str, int] = {}
    status = "needs_refinitiv_liquidity_snapshot"
    snapshot_rows = 0
    if liquidity_snapshot_path:
        snapshot_source = Path(liquidity_snapshot_path)
        liquidity_snapshot = pd.read_parquet(snapshot_source) if snapshot_source.suffix.casefold() == ".parquet" else pd.read_csv(snapshot_source, low_memory=False)
        snapshot_rows = int(len(liquidity_snapshot))
        large_core = build_usa_large_liquid_core(
            security_master,
            liquidity_snapshot,
            min_market_cap_usd=min_market_cap_usd,
            min_adv_value_usd=min_adv_value_usd,
            min_free_float_pct=min_free_float_pct,
        )
        large_symbol_map = symbol_map_from_security_master_rows(large_core, source, USA_LARGE_LIQUID_UNIVERSE_ID)
        large_core_path = artifact_dir / "usa_large_liquid_core_v1.csv"
        local_symbol_map_path = artifact_dir / "usa_large_liquid_core_v1_symbol_map.csv"
        large_core.to_csv(large_core_path, index=False)
        large_symbol_map.to_csv(local_symbol_map_path, index=False)
        refinitiv_dir = Path(symbol_map_artifact_root) / f"REFINITIV-SYMBOL-MAP-{USA_LARGE_LIQUID_UNIVERSE_ID}-{run_id}"
        refinitiv_dir.mkdir(parents=True, exist_ok=True)
        large_symbol_map_path = refinitiv_dir / "symbol_map.csv"
        large_symbol_map.to_csv(large_symbol_map_path, index=False)
        (refinitiv_dir / "manifest.json").write_text(
            json.dumps(_manifest_for_map(f"REFINITIV-SYMBOL-MAP-{USA_LARGE_LIQUID_UNIVERSE_ID}-{run_id}", refinitiv_dir, large_symbol_map_path, large_symbol_map), indent=2),
            encoding="utf-8",
        )
        large_core_rows = int(len(large_symbol_map))
        large_core_exchange_counts = {str(key): int(value) for key, value in large_symbol_map["exchange"].value_counts(dropna=False).to_dict().items()}
        status = "large_liquid_core_ready" if large_core_rows else "large_liquid_core_empty"

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": status,
        "universe_profile": "usa_large_liquid_operational_core",
        "security_master_source": str(source.resolve()),
        "liquidity_snapshot_source": str(Path(liquidity_snapshot_path).resolve()) if liquidity_snapshot_path else "",
        "snapshot_request_path": str(snapshot_request_path.resolve()),
        "snapshot_field_map_path": str(snapshot_field_map_path.resolve()),
        "large_liquid_refinitiv_symbol_map_path": str(large_symbol_map_path.resolve()) if large_symbol_map_path else "",
        "security_master_rows": int(len(security_master)),
        "base_active_candidate_rows": int(len(active_candidate_map)),
        "snapshot_requested_rows": int(len(snapshot_request)),
        "snapshot_rows": snapshot_rows,
        "active_core_rows": large_core_rows,
        "large_liquid_core_rows": large_core_rows,
        "reference_core_rows": int(len(reference_core)),
        "large_liquid_core_exchange_counts": large_core_exchange_counts,
        "min_market_cap_usd": float(min_market_cap_usd),
        "min_adv_value_usd": float(min_adv_value_usd),
        "min_free_float_pct": float(min_free_float_pct),
        "minimum_expected_large_liquid_rows": 500,
        "filter_notes": [
            "Large-liquid USA core is current-only and controls near-term extraction scope.",
            "Final institutional backtests still require lagged/PIT market-cap and liquidity eligibility.",
            "Full security master and listed-reference denominator remain required for survivorship controls.",
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return UsaLargeLiquidUniverseArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        snapshot_request_path=snapshot_request_path,
        snapshot_field_map_path=snapshot_field_map_path,
        large_liquid_symbol_map_path=large_symbol_map_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
