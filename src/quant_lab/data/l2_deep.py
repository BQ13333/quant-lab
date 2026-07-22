from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from quant_lab.data.asean_l2 import ASEAN_L2_COUNTRIES, build_asean_l2_data_requirements
from quant_lab.data.data_depth import load_long_ohlcv

CORPORATE_ACTION_PROXY_COLUMNS = [
    "date",
    "ticker",
    "country",
    "exchange",
    "name",
    "raw_return",
    "adjusted_return",
    "adjustment_ratio",
    "adjustment_change",
    "proxy_event_type",
    "official_action_status",
]


@dataclass(frozen=True)
class L2DeepExtractionResult:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    security_master_path: Path
    universe_tiers_path: Path
    country_summary_path: Path
    ticker_quality_path: Path
    daily_features_path: Path
    corporate_action_proxy_path: Path
    vendor_matrix_path: Path
    markdown_path: Path
    security_master: pd.DataFrame
    universe_tiers: pd.DataFrame
    country_summary: pd.DataFrame
    ticker_quality: pd.DataFrame


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("ASEAN-L2-DEEP-EXTRACTION-%Y%m%d-%H%M%S")


def _pivot_field(data: pd.DataFrame, field: str) -> pd.DataFrame:
    if field not in data.columns:
        return pd.DataFrame()
    frame = (
        data.pivot_table(index="date", columns="ticker", values=field, aggfunc="last")
        .sort_index()
        .astype(float)
    )
    frame.columns.name = None
    return frame


def _max_true_run(mask: pd.Series) -> int:
    if mask.empty:
        return 0
    groups = mask.ne(mask.shift()).cumsum()
    runs = mask.groupby(groups).sum()
    return int(runs.max()) if len(runs) else 0


def _ticker_quality(
    data: pd.DataFrame,
    close: pd.DataFrame,
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
) -> pd.DataFrame:
    total_dates = len(close.index)
    close_missing = close.isna()
    volume_missing = volume.isna() if not volume.empty else pd.DataFrame(True, index=close.index, columns=close.columns)
    zero_volume = (volume.notna() & volume.eq(0.0)) if not volume.empty else pd.DataFrame(False, index=close.index, columns=close.columns)
    stale_close = close.eq(close.shift(1))
    bad_close = close.notna() & close.le(0.0)
    bad_ohlc = pd.DataFrame(False, index=close.index, columns=close.columns)
    if not high.empty and not low.empty:
        bad_ohlc = high.lt(low) | high.lt(close) | low.gt(close)

    grouped = (
        data.groupby("ticker", dropna=False)
        .agg(
            observed_rows=("date", "size"),
            first_date=("date", "min"),
            last_date=("date", "max"),
            sid=("sid", "first"),
            local_symbol=("local_symbol", "first"),
            name=("name", "first"),
            country=("country", "first"),
            exchange=("exchange", "first"),
            currency=("currency", "first"),
            isin=("isin", "first") if "isin" in data.columns else ("ticker", "first"),
        )
        .reset_index()
    )
    grouped["calendar_rows"] = total_dates
    grouped["close_observations"] = close.notna().sum().reindex(grouped["ticker"]).to_numpy()
    grouped["close_missing_pct"] = close_missing.mean().reindex(grouped["ticker"]).to_numpy()
    grouped["volume_missing_pct"] = volume_missing.mean().reindex(grouped["ticker"]).to_numpy()
    grouped["zero_volume_pct"] = zero_volume.mean().reindex(grouped["ticker"]).to_numpy()
    grouped["stale_close_pct"] = stale_close.mean().reindex(grouped["ticker"]).to_numpy()
    grouped["bad_close_rows"] = bad_close.sum().reindex(grouped["ticker"]).to_numpy()
    grouped["bad_ohlc_rows"] = bad_ohlc.sum().reindex(grouped["ticker"]).to_numpy()
    grouped["max_missing_gap"] = [
        _max_true_run(close_missing[ticker]) if ticker in close_missing else 0 for ticker in grouped["ticker"]
    ]
    grouped["max_stale_run"] = [
        _max_true_run(stale_close[ticker].fillna(False)) if ticker in stale_close else 0 for ticker in grouped["ticker"]
    ]
    local_dollar_volume = close * volume
    grouped["latest_local_dollar_volume"] = local_dollar_volume.iloc[-1].reindex(grouped["ticker"]).to_numpy()
    grouped["adv_20_local_lag1_latest"] = (
        local_dollar_volume.shift(1).rolling(20, min_periods=5).mean().iloc[-1].reindex(grouped["ticker"]).to_numpy()
    )
    grouped["adv_60_local_lag1_latest"] = (
        local_dollar_volume.shift(1).rolling(60, min_periods=20).mean().iloc[-1].reindex(grouped["ticker"]).to_numpy()
    )
    grouped["adv_252_local_lag1_latest"] = (
        local_dollar_volume.shift(1).rolling(252, min_periods=60).mean().iloc[-1].reindex(grouped["ticker"]).to_numpy()
    )
    returns = adj_close.pct_change(fill_method=None)
    grouped["vol_63_lag1_latest"] = (
        returns.shift(1).rolling(63, min_periods=20).std().mul(np.sqrt(252)).iloc[-1].reindex(grouped["ticker"]).to_numpy()
    )
    grouped = grouped.sort_values(["country", "exchange", "ticker"], na_position="last").reset_index(drop=True)
    return grouped


def _build_universe_tiers(
    ticker_quality: pd.DataFrame,
    min_rows: int,
    max_missing_ratio: float,
    top_liquid_per_country: int,
) -> pd.DataFrame:
    tiers = ticker_quality.copy()
    tiers["u0_raw_l2"] = True
    tiers["u1_price_coverage"] = (
        tiers["close_observations"].ge(min_rows)
        & tiers["close_missing_pct"].le(max_missing_ratio)
        & tiers["bad_close_rows"].eq(0)
    )
    tiers["u2_backtest_eligible"] = tiers["u1_price_coverage"] & tiers["volume_missing_pct"].le(max_missing_ratio)
    tiers["adv_60_country_rank"] = (
        tiers.groupby("country", dropna=False)["adv_60_local_lag1_latest"]
        .rank(method="first", ascending=False, na_option="bottom")
        .astype("Int64")
    )
    tiers["u3_liquid_proxy"] = (
        tiers["u2_backtest_eligible"]
        & tiers["adv_60_country_rank"].le(top_liquid_per_country)
        & tiers["adv_60_local_lag1_latest"].fillna(0.0).gt(0.0)
    )
    tiers["l2_rejection_reason"] = ""
    tiers.loc[~tiers["u1_price_coverage"], "l2_rejection_reason"] = "insufficient_price_coverage"
    tiers.loc[tiers["u1_price_coverage"] & ~tiers["u2_backtest_eligible"], "l2_rejection_reason"] = "insufficient_volume_coverage"
    tiers.loc[tiers["u2_backtest_eligible"] & ~tiers["u3_liquid_proxy"], "l2_rejection_reason"] = "outside_liquid_proxy_rank"
    return tiers.sort_values(["country", "adv_60_country_rank", "ticker"], na_position="last").reset_index(drop=True)


def _stack_feature(name: str, frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float, name=name)
    return frame.stack(future_stack=True).rename(name)


def _build_daily_features(
    open_: pd.DataFrame,
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    returns = adj_close.pct_change(fill_method=None)
    log_returns = np.log(adj_close.replace(0.0, np.nan)).diff()
    local_dollar_volume = close * volume
    spread_proxy = high.sub(low).div(close.replace(0.0, np.nan))
    overnight_return = open_.div(close.shift(1).replace(0.0, np.nan)).sub(1.0)
    intraday_return = close.div(open_.replace(0.0, np.nan)).sub(1.0)
    amihud = returns.abs().div(local_dollar_volume.replace(0.0, np.nan))

    frames = [
        _stack_feature("open", open_),
        _stack_feature("high", high),
        _stack_feature("low", low),
        _stack_feature("close", close),
        _stack_feature("adj_close", adj_close),
        _stack_feature("volume", volume),
        _stack_feature("simple_return", returns),
        _stack_feature("log_return", log_returns),
        _stack_feature("overnight_return", overnight_return),
        _stack_feature("intraday_return", intraday_return),
        _stack_feature("local_dollar_volume", local_dollar_volume),
        _stack_feature("spread_proxy", spread_proxy),
        _stack_feature("adv_20_local_lag1", local_dollar_volume.shift(1).rolling(20, min_periods=5).mean()),
        _stack_feature("adv_60_local_lag1", local_dollar_volume.shift(1).rolling(60, min_periods=20).mean()),
        _stack_feature("adv_252_local_lag1", local_dollar_volume.shift(1).rolling(252, min_periods=60).mean()),
        _stack_feature("vol_20_lag1", returns.shift(1).rolling(20, min_periods=10).std().mul(np.sqrt(252))),
        _stack_feature("vol_63_lag1", returns.shift(1).rolling(63, min_periods=20).std().mul(np.sqrt(252))),
        _stack_feature("vol_252_lag1", returns.shift(1).rolling(252, min_periods=60).std().mul(np.sqrt(252))),
        _stack_feature("amihud_20_lag1", amihud.shift(1).rolling(20, min_periods=5).mean()),
        _stack_feature("amihud_60_lag1", amihud.shift(1).rolling(60, min_periods=20).mean()),
        _stack_feature("volume_missing_flag", volume.isna().astype(float)),
        _stack_feature("zero_volume_flag", (volume.notna() & volume.eq(0.0)).astype(float)),
        _stack_feature("stale_close_flag", close.eq(close.shift(1)).astype(float)),
        _stack_feature("bad_ohlc_flag", (high.lt(low) | high.lt(close) | low.gt(close)).astype(float)),
    ]
    features = pd.concat(frames, axis=1).reset_index()
    features.columns = ["date", "ticker", *features.columns[2:]]
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features = features.dropna(subset=["date", "ticker"])
    features["year"] = features["date"].dt.year.astype("Int64")

    meta_cols = ["ticker", "sid", "local_symbol", "name", "country", "exchange", "currency"]
    available_meta = [col for col in meta_cols if col in metadata.columns]
    meta = metadata[available_meta].drop_duplicates("ticker", keep="last")
    features = features.merge(meta, on="ticker", how="left")
    ordered = ["date", "ticker", "sid", "local_symbol", "name", "country", "exchange", "currency", "year"]
    return features[[col for col in ordered if col in features.columns] + [col for col in features.columns if col not in ordered]]


def _build_corporate_action_proxy(close: pd.DataFrame, adj_close: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    raw_return = close.pct_change(fill_method=None)
    adjusted_return = adj_close.pct_change(fill_method=None)
    adjustment_ratio = adj_close.div(close.replace(0.0, np.nan))
    adjustment_change = adjustment_ratio.pct_change(fill_method=None)
    mask = (
        (raw_return.abs().gt(0.30) & adjusted_return.abs().lt(0.10))
        | adjustment_change.abs().gt(0.01)
    )
    rows = []
    meta = metadata.set_index("ticker") if "ticker" in metadata.columns else pd.DataFrame()
    stacked = mask.stack(future_stack=True)
    for (date, ticker), flagged in stacked.items():
        if not bool(flagged):
            continue
        meta_row = meta.loc[ticker].to_dict() if ticker in meta.index else {}
        rows.append(
            {
                "date": pd.Timestamp(date),
                "ticker": ticker,
                "country": meta_row.get("country"),
                "exchange": meta_row.get("exchange"),
                "name": meta_row.get("name"),
                "raw_return": raw_return.at[date, ticker] if ticker in raw_return.columns else np.nan,
                "adjusted_return": adjusted_return.at[date, ticker] if ticker in adjusted_return.columns else np.nan,
                "adjustment_ratio": adjustment_ratio.at[date, ticker] if ticker in adjustment_ratio.columns else np.nan,
                "adjustment_change": adjustment_change.at[date, ticker] if ticker in adjustment_change.columns else np.nan,
                "proxy_event_type": "possible_split_or_adjustment",
                "official_action_status": "proxy_only_requires_vendor_corporate_actions",
            }
        )
    return pd.DataFrame(rows, columns=CORPORATE_ACTION_PROXY_COLUMNS)


def _build_country_summary(universe_tiers: pd.DataFrame) -> pd.DataFrame:
    return (
        universe_tiers.groupby("country", dropna=False)
        .agg(
            raw_tickers=("ticker", "nunique"),
            u1_price_coverage=("u1_price_coverage", "sum"),
            u2_backtest_eligible=("u2_backtest_eligible", "sum"),
            u3_liquid_proxy=("u3_liquid_proxy", "sum"),
            median_rows=("close_observations", "median"),
            median_missing_pct=("close_missing_pct", "median"),
            median_zero_volume_pct=("zero_volume_pct", "median"),
            median_adv_60_local=("adv_60_local_lag1_latest", "median"),
        )
        .reset_index()
        .sort_values(["u2_backtest_eligible", "raw_tickers"], ascending=[False, False])
    )


def _build_vendor_matrix(
    symbol_map_path: str | Path | None,
    field_catalog_path: str | Path | None,
    historical_manifest_path: str | Path | None,
) -> pd.DataFrame:
    requirements = build_asean_l2_data_requirements()
    availability = {
        "fuller_equity_universe": "partial_current_survivor_symbol_map" if symbol_map_path else "missing_vendor_security_master",
        "point_in_time_security_master": "missing_pit_identifier_history",
        "corporate_actions": "missing_official_actions_proxy_only",
        "prices": "available_ohlcv_backfill" if historical_manifest_path else "available_from_local_ohlcv_no_manifest",
        "liquidity": "partial_derived_from_price_volume_missing_float_adv",
        "fundamentals": "missing_pit_fundamentals",
        "estimates": "missing_pit_estimates_and_revisions",
        "fx_country_risk": "partial_fx_costs_need_exchange_calendar_tax",
    }
    requirements["current_availability"] = requirements["layer"].map(availability).fillna("unknown")
    requirements["source_artifact"] = ""
    if symbol_map_path:
        requirements.loc[requirements["layer"].isin(["fuller_equity_universe", "point_in_time_security_master"]), "source_artifact"] = str(symbol_map_path)
    if field_catalog_path:
        requirements.loc[requirements["layer"].eq("prices"), "source_artifact"] = str(field_catalog_path)
    if historical_manifest_path:
        requirements.loc[requirements["layer"].eq("prices"), "source_artifact"] = str(historical_manifest_path)
    return requirements


def _write_markdown_summary(
    path: Path,
    run_id: str,
    long_ohlcv_root: str | Path,
    country_summary: pd.DataFrame,
    universe_tiers: pd.DataFrame,
    vendor_matrix: pd.DataFrame,
    daily_features_rows: int,
    corporate_action_proxy_rows: int,
) -> None:
    u2 = int(universe_tiers["u2_backtest_eligible"].sum()) if "u2_backtest_eligible" in universe_tiers else 0
    u3 = int(universe_tiers["u3_liquid_proxy"].sum()) if "u3_liquid_proxy" in universe_tiers else 0
    lines = [
        f"# ASEAN L2 Deep Extraction - {run_id}",
        "",
        "This is a data extraction and enrichment artifact, not a strategy result.",
        "",
        "## Scope",
        "",
        f"- Source OHLCV root: `{long_ohlcv_root}`",
        f"- Raw L2 tickers: `{len(universe_tiers):,}`",
        f"- U2 backtest-eligible tickers: `{u2:,}`",
        f"- U3 liquid-proxy tickers: `{u3:,}`",
        f"- Daily feature rows: `{daily_features_rows:,}`",
        f"- Corporate-action proxy rows: `{corporate_action_proxy_rows:,}`",
        "",
        "## Country Coverage",
        "",
        country_summary.to_string(index=False),
        "",
        "## Data Readiness",
        "",
        vendor_matrix[["layer", "dataset", "current_availability", "research_gate"]].to_string(index=False),
        "",
        "## Backtest Gate",
        "",
        "- Allowed next: OHLCV + derived liquidity/volatility backtests using lagged features.",
        "- Not allowed yet: production-quality PIT fundamentals, analyst revisions, official corporate-action alpha, or delisting-robust claims.",
        "- Key remaining extraction: PIT security master, official splits/dividends, market cap/free float/ADV from vendor, fundamentals, estimates, analyst recommendations, exchange calendars and taxes.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_l2_deep_extraction(
    long_ohlcv_root: str | Path,
    artifact_root: str | Path = "artifacts/l2_deep_extraction",
    run_id: str | None = None,
    include_countries: tuple[str, ...] = ASEAN_L2_COUNTRIES,
    min_rows: int = 2500,
    max_missing_ratio: float = 0.25,
    top_liquid_per_country: int = 50,
    symbol_map_path: str | Path | None = None,
    field_catalog_path: str | Path | None = None,
    historical_manifest_path: str | Path | None = None,
) -> L2DeepExtractionResult:
    """Build a richer L2 OHLCV-derived data layer and extraction readiness matrix."""

    run_id = run_id or _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    panel_dir = artifact_dir / "panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    data = load_long_ohlcv(long_ohlcv_root)
    if include_countries:
        data = data[data["country"].astype("string").isin(include_countries)].copy()
    if data.empty:
        raise ValueError("no L2 OHLCV rows after country filter")
    data["adj_close"] = data["adj_close"].fillna(data["close"])

    open_ = _pivot_field(data, "open")
    high = _pivot_field(data, "high")
    low = _pivot_field(data, "low")
    close = _pivot_field(data, "close")
    adj_close = _pivot_field(data, "adj_close")
    volume = _pivot_field(data, "volume")
    for name, panel in {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": adj_close,
        "volume": volume,
    }.items():
        panel.to_parquet(panel_dir / f"{name}.parquet")

    ticker_quality = _ticker_quality(data, close, adj_close, volume, high, low)
    security_master = ticker_quality[
        [
            "ticker",
            "sid",
            "local_symbol",
            "name",
            "country",
            "exchange",
            "currency",
            "isin",
            "first_date",
            "last_date",
            "observed_rows",
        ]
    ].copy()
    universe_tiers = _build_universe_tiers(ticker_quality, min_rows, max_missing_ratio, top_liquid_per_country)
    country_summary = _build_country_summary(universe_tiers)
    daily_features = _build_daily_features(open_, high, low, close, adj_close, volume, security_master)
    corporate_action_proxy = _build_corporate_action_proxy(close, adj_close, security_master)
    vendor_matrix = _build_vendor_matrix(symbol_map_path, field_catalog_path, historical_manifest_path)

    security_master_path = artifact_dir / "security_master_current_survivor.csv"
    universe_tiers_path = artifact_dir / "eligible_universe_tiers.csv"
    country_summary_path = artifact_dir / "country_summary.csv"
    ticker_quality_path = artifact_dir / "ticker_quality.csv"
    daily_features_path = artifact_dir / "daily_features.parquet"
    corporate_action_proxy_path = artifact_dir / "corporate_action_proxy_events.csv"
    vendor_matrix_path = artifact_dir / "vendor_extraction_matrix.csv"
    markdown_path = artifact_dir / "l2_deep_extraction_summary.md"
    manifest_path = artifact_dir / "manifest.json"

    security_master.to_csv(security_master_path, index=False)
    universe_tiers.to_csv(universe_tiers_path, index=False)
    country_summary.to_csv(country_summary_path, index=False)
    ticker_quality.to_csv(ticker_quality_path, index=False)
    daily_features.to_parquet(daily_features_path, index=False)
    corporate_action_proxy.to_csv(corporate_action_proxy_path, index=False)
    vendor_matrix.to_csv(vendor_matrix_path, index=False)
    _write_markdown_summary(
        markdown_path,
        run_id,
        long_ohlcv_root,
        country_summary,
        universe_tiers,
        vendor_matrix,
        daily_features_rows=len(daily_features),
        corporate_action_proxy_rows=len(corporate_action_proxy),
    )

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "long_ohlcv_root": str(Path(long_ohlcv_root).resolve()),
        "include_countries": list(include_countries),
        "min_rows": int(min_rows),
        "max_missing_ratio": float(max_missing_ratio),
        "top_liquid_per_country": int(top_liquid_per_country),
        "raw_rows": int(data.shape[0]),
        "raw_tickers": int(security_master["ticker"].nunique()),
        "date_min": str(pd.Timestamp(data["date"].min()).date()),
        "date_max": str(pd.Timestamp(data["date"].max()).date()),
        "u2_backtest_eligible": int(universe_tiers["u2_backtest_eligible"].sum()),
        "u3_liquid_proxy": int(universe_tiers["u3_liquid_proxy"].sum()),
        "daily_features_rows": int(len(daily_features)),
        "corporate_action_proxy_rows": int(len(corporate_action_proxy)),
        "artifacts": {
            "security_master_current_survivor": str(security_master_path.resolve()),
            "eligible_universe_tiers": str(universe_tiers_path.resolve()),
            "country_summary": str(country_summary_path.resolve()),
            "ticker_quality": str(ticker_quality_path.resolve()),
            "daily_features": str(daily_features_path.resolve()),
            "corporate_action_proxy_events": str(corporate_action_proxy_path.resolve()),
            "vendor_extraction_matrix": str(vendor_matrix_path.resolve()),
            "summary": str(markdown_path.resolve()),
        },
        "symbol_map_path": str(symbol_map_path) if symbol_map_path else "",
        "field_catalog_path": str(field_catalog_path) if field_catalog_path else "",
        "historical_manifest_path": str(historical_manifest_path) if historical_manifest_path else "",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return L2DeepExtractionResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        manifest_path=manifest_path,
        security_master_path=security_master_path,
        universe_tiers_path=universe_tiers_path,
        country_summary_path=country_summary_path,
        ticker_quality_path=ticker_quality_path,
        daily_features_path=daily_features_path,
        corporate_action_proxy_path=corporate_action_proxy_path,
        vendor_matrix_path=vendor_matrix_path,
        markdown_path=markdown_path,
        security_master=security_master,
        universe_tiers=universe_tiers,
        country_summary=country_summary,
        ticker_quality=ticker_quality,
    )
