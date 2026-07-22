from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.refinitiv_symbol_map import default_refinitiv_symbol_seed_listings


ASEAN_L2_ALL_COUNTRIES: tuple[str, ...] = (
    "Malaysia",
    "Singapore",
    "Indonesia",
)


ASEAN_L2_EXCLUDED_COUNTRIES: tuple[str, ...] = ("Thailand", "Philippines", "Vietnam")


ASEAN_L2_COUNTRIES: tuple[str, ...] = tuple(
    country for country in ASEAN_L2_ALL_COUNTRIES if country not in ASEAN_L2_EXCLUDED_COUNTRIES
)


ASEAN_L2_ALL_MARKET_SPECS: tuple[dict[str, object], ...] = (
    {
        "country": "Malaysia",
        "primary_exchanges": "Bursa Malaysia",
        "target_symbols": 100,
        "public_security_master_status": "vendor_required",
        "price_route": "Refinitiv Workspace/Data Library",
        "next_action": "Pull Bursa top liquid common shares from Refinitiv with RIC, ISIN, listing date, market cap, free float, ADV, and active/delisted flags.",
    },
    {
        "country": "Singapore",
        "primary_exchanges": "SGX",
        "target_symbols": 100,
        "public_security_master_status": "public_adapter_available",
        "price_route": "Refinitiv preferred; Yahoo only for smoke tests",
        "next_action": "Rank SGX common shares/REITs, validate Refinitiv RICs, then backfill adjusted and raw OHLCV.",
    },
    {
        "country": "Indonesia",
        "primary_exchanges": "IDX",
        "target_symbols": 100,
        "public_security_master_status": "vendor_required",
        "price_route": "Refinitiv Workspace/Data Library",
        "next_action": "Pull IDX top liquid common shares from Refinitiv because public IDX discovery is edge-protected here.",
    },
)


ASEAN_L2_MARKET_SPECS: tuple[dict[str, object], ...] = tuple(
    spec for spec in ASEAN_L2_ALL_MARKET_SPECS if str(spec["country"]) in ASEAN_L2_COUNTRIES
)


ASEAN_L2_VENDOR_COLUMNS = (
    "vendor_sid",
    "ric",
    "isin",
    "sedol",
    "local_ticker",
    "company_name",
    "country",
    "exchange",
    "currency",
    "security_type",
    "primary_issue",
    "listing_date",
    "delisting_date",
    "is_active",
    "ticker_change_history",
    "market_cap",
    "free_float_market_cap",
    "free_float_percent",
    "adv_20d",
    "adv_60d",
    "turnover_20d",
    "sector",
    "industry",
)


@dataclass(frozen=True)
class AseanL2PlanArtifact:
    artifact_dir: Path
    ranked_public_candidates_path: Path
    l2_candidate_symbols_path: Path
    refinitiv_symbol_map_input_path: Path
    country_targets_path: Path
    vendor_requests_path: Path
    data_requirements_path: Path
    markdown_path: Path
    manifest_path: Path
    ranked_public_candidates: pd.DataFrame
    l2_candidate_symbols: pd.DataFrame
    refinitiv_symbol_map_input: pd.DataFrame
    country_targets: pd.DataFrame
    vendor_requests: pd.DataFrame
    data_requirements: pd.DataFrame


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return " ".join(str(value).strip().split())


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _truthy_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(True, index=frame.index)
    values = frame[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype("string").str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _empty_ranked_public_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "country",
            "exchange",
            "ticker",
            "yahoo_ticker",
            "name",
            "currency",
            "security_type",
            "asset_type",
            "source",
            "l2_country_rank",
            "l2_exchange_rank",
            "l2_rank_score",
            "l2_rank_method",
            "l2_rank_quality",
            "l2_target_candidate",
            "l2_data_status",
        ]
    )


def rank_asean_l2_public_candidates(
    listings: pd.DataFrame,
    target_per_country: int = 100,
    countries: tuple[str, ...] = ASEAN_L2_COUNTRIES,
) -> pd.DataFrame:
    """Rank public ASEAN listing rows for the first L2 security-master pass.

    The rank uses market cap when an adapter supplies it, then share-count
    proxies, then deterministic symbol order. It is only a discovery rank; ADV
    and free-float market cap should replace it once vendor liquidity data is
    attached.
    """

    if listings is None or listings.empty:
        return _empty_ranked_public_candidates()

    data = listings.copy()
    if "country" not in data.columns:
        return _empty_ranked_public_candidates()

    country = data["country"].astype("string").str.strip()
    data = data[country.isin(countries)].copy()
    if data.empty:
        return _empty_ranked_public_candidates()

    data = data[_truthy_column(data, "is_stock_like")].copy()
    for column in ("asset_type", "security_type", "instrument_type"):
        if column in data.columns:
            text = data[column].astype("string").str.lower()
            excluded = text.str.contains("etf|warrant|right|bond|debenture|certificate|note", na=False)
            data = data[~excluded].copy()
    if data.empty:
        return _empty_ranked_public_candidates()

    data["_market_cap"] = _numeric_column(data, "market_cap")
    share_columns = ["free_float_market_cap", "listed_shares", "outstanding_shares", "shares_outstanding"]
    share_proxy = pd.Series(pd.NA, index=data.index, dtype="Float64")
    for column in share_columns:
        values = _numeric_column(data, column)
        share_proxy = share_proxy.fillna(values)
    data["_share_proxy"] = share_proxy

    data["l2_rank_score"] = data["_market_cap"].fillna(data["_share_proxy"]).fillna(0.0).astype(float)
    data["l2_rank_method"] = "source_symbol_order"
    data.loc[data["_share_proxy"].notna(), "l2_rank_method"] = "share_count_proxy"
    data.loc[data["_market_cap"].notna(), "l2_rank_method"] = "market_cap"
    data["l2_rank_quality"] = data["l2_rank_method"].map(
        {
            "market_cap": "high_public_rank",
            "share_count_proxy": "medium_public_rank",
            "source_symbol_order": "low_public_rank",
        }
    )
    method_priority = {"market_cap": 0, "share_count_proxy": 1, "source_symbol_order": 2}
    data["_method_priority"] = data["l2_rank_method"].map(method_priority).fillna(9).astype(int)
    data["_ticker_sort"] = data.get("ticker", pd.Series("", index=data.index)).astype("string").fillna("")

    data = data.sort_values(
        ["country", "_method_priority", "l2_rank_score", "_ticker_sort"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    data["l2_country_rank"] = data.groupby("country", dropna=False).cumcount() + 1
    if "exchange" in data.columns:
        data["l2_exchange_rank"] = data.groupby(["country", "exchange"], dropna=False).cumcount() + 1
    else:
        data["l2_exchange_rank"] = data["l2_country_rank"]
    data["l2_target_candidate"] = data["l2_country_rank"].le(target_per_country)
    data["l2_data_status"] = "public_security_master_candidate_unvalidated"

    keep_cols = [
        "country",
        "exchange",
        "ticker",
        "yahoo_ticker",
        "name",
        "currency",
        "security_type",
        "asset_type",
        "sector",
        "industry",
        "market_cap",
        "listed_shares",
        "outstanding_shares",
        "source",
        "l2_country_rank",
        "l2_exchange_rank",
        "l2_rank_score",
        "l2_rank_method",
        "l2_rank_quality",
        "l2_target_candidate",
        "l2_data_status",
    ]
    keep_cols = [column for column in keep_cols if column in data.columns]
    return data[keep_cols].reset_index(drop=True)


def build_asean_l2_vendor_requests(target_per_country: int = 100) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in ASEAN_L2_MARKET_SPECS:
        target_symbols = int(target_per_country or spec["target_symbols"])
        rows.append(
            {
                "country": spec["country"],
                "primary_exchanges": spec["primary_exchanges"],
                "target_current_liquid_symbols": target_symbols,
                "target_backtest_years": 20,
                "preferred_route": spec["price_route"],
                "security_master_status": spec["public_security_master_status"],
                "required_filters": "common_shares_or_reits|primary_listing|active_and_delisted|local_equity|exclude_warrants_etfs_funds",
                "required_sort": "free_float_market_cap_desc_then_adv_60d_desc",
                "required_columns": "|".join(ASEAN_L2_VENDOR_COLUMNS),
                "point_in_time_required": True,
                "next_action": spec["next_action"],
            }
        )
    return pd.DataFrame(rows)


def build_asean_l2_data_requirements() -> pd.DataFrame:
    rows = [
        {
            "layer": "fuller_equity_universe",
            "dataset": "current_security_master_l2",
            "minimum_fields": "sid|ric|isin|local_ticker|company_name|country|exchange|currency|security_type|primary_issue|market_cap|adv_60d",
            "point_in_time_rule": "current snapshot is allowed for discovery only",
            "research_gate": "needed_before_large_cross_section_backtest",
            "status": "partial_public_plus_refinitiv_next",
        },
        {
            "layer": "point_in_time_security_master",
            "dataset": "listing_timeline_and_identifier_history",
            "minimum_fields": "sid|ric|isin|local_ticker|effective_from|effective_to|listing_date|delisting_date|ticker_change_reason|active_flag",
            "point_in_time_rule": "as-of date must use identifiers and active universe known at that date",
            "research_gate": "required_before_any_production_quality_claim",
            "status": "vendor_required",
        },
        {
            "layer": "corporate_actions",
            "dataset": "splits_dividends_rights_bonus_actions",
            "minimum_fields": "sid|ex_date|record_date|pay_date|action_type|ratio|cash_amount|currency|adjustment_factor|source_timestamp",
            "point_in_time_rule": "action is usable only after its announcement/source timestamp",
            "research_gate": "required_for_adjusted_raw_return_integrity",
            "status": "vendor_required",
        },
        {
            "layer": "prices",
            "dataset": "daily_raw_and_adjusted_ohlcv",
            "minimum_fields": "sid|date|open|high|low|close|adj_close|volume|turnover|vwap|currency|suspension_flag|provider_timestamp",
            "point_in_time_rule": "daily bar available after market close plus country calendar lag",
            "research_gate": "required_for_backtest",
            "status": "refinitiv_backfill_next",
        },
        {
            "layer": "liquidity",
            "dataset": "tradability_and_capacity",
            "minimum_fields": "sid|date|market_cap|free_float_market_cap|free_float_percent|adv_20d|adv_60d|turnover|bid_ask_proxy|halt_flag|limit_event_flag",
            "point_in_time_rule": "rolling features use prior data only",
            "research_gate": "required_for_weighting_cost_capacity",
            "status": "vendor_required",
        },
        {
            "layer": "fundamentals",
            "dataset": "point_in_time_fundamental_facts",
            "minimum_fields": "sid|fiscal_period|report_date|available_time_utc|revenue|eps|book_value|roe|margins|net_debt|shares",
            "point_in_time_rule": "join by available_time_utc, never fiscal period end",
            "research_gate": "required_for_quality_value_earnings_models",
            "status": "refinitiv_estimates_or_fundamentals_required",
        },
        {
            "layer": "estimates",
            "dataset": "analyst_revisions_recommendations_targets",
            "minimum_fields": "sid|broker|analyst|estimate_item|period|revision_time_utc|old_value|new_value|recommendation|target_price|currency",
            "point_in_time_rule": "event is tradable only after revision_time_utc",
            "research_gate": "required_for_revision_and_event_alpha",
            "status": "refinitiv_estimates_required",
        },
        {
            "layer": "fx_country_risk",
            "dataset": "fx_rates_taxes_fees_calendars",
            "minimum_fields": "date|currency|usd_rate|local_holiday|withholding_tax|stamp_duty|exchange_fee|country_risk_proxy",
            "point_in_time_rule": "use rates and calendars known for the rebalance date",
            "research_gate": "required_for_usd_returns_and_tca",
            "status": "fx_available_costs_need_exchange_detail",
        },
    ]
    return pd.DataFrame(rows)


def summarize_asean_l2_country_targets(
    ranked_public_candidates: pd.DataFrame,
    symbol_map: pd.DataFrame | None = None,
    target_per_country: int = 100,
) -> pd.DataFrame:
    symbol_map = symbol_map if symbol_map is not None else pd.DataFrame()
    rows: list[dict[str, object]] = []
    for spec in ASEAN_L2_MARKET_SPECS:
        country = str(spec["country"])
        target_symbols = int(target_per_country or spec["target_symbols"])
        public_country = (
            ranked_public_candidates[ranked_public_candidates["country"].astype("string").str.strip().eq(country)].copy()
            if not ranked_public_candidates.empty and "country" in ranked_public_candidates.columns
            else pd.DataFrame()
        )
        public_l2 = public_country[public_country.get("l2_target_candidate", pd.Series(False, index=public_country.index)).fillna(False)]
        quality_counts = (
            public_l2["l2_rank_quality"].value_counts(dropna=False).to_dict()
            if not public_l2.empty and "l2_rank_quality" in public_l2.columns
            else {}
        )
        quality_total = max(1, len(public_l2))
        high_or_medium = int(quality_counts.get("high_public_rank", 0)) + int(quality_counts.get("medium_public_rank", 0))
        if not public_l2.empty and int(quality_counts.get("high_public_rank", 0)) / quality_total >= 0.7:
            public_rank_quality = "high_public_rank"
        elif not public_l2.empty and high_or_medium / quality_total >= 0.7:
            public_rank_quality = "medium_public_rank"
        elif not public_l2.empty:
            public_rank_quality = "low_public_rank"
        else:
            public_rank_quality = "no_public_rank"

        validated_symbols = 0
        if not symbol_map.empty and {"country", "refinitiv_ric"}.issubset(symbol_map.columns):
            country_map = symbol_map[symbol_map["country"].astype("string").str.strip().eq(country)].copy()
            if "refinitiv_symbol_status" in country_map.columns:
                country_map = country_map[country_map["refinitiv_symbol_status"].eq("validated_price_history")]
            validated_symbols = int(country_map["refinitiv_ric"].dropna().astype(str).nunique())

        if validated_symbols >= target_symbols:
            readiness = "ready_for_l2_price_backfill"
        elif len(public_l2) >= target_symbols:
            readiness = "security_master_ready_validate_refinitiv"
        elif len(public_country) > 0:
            readiness = "partial_security_master_expand_or_vendor_fill"
        else:
            readiness = "vendor_security_master_required"
        effective_security_master_rows = max(int(len(public_l2)), validated_symbols)

        rows.append(
            {
                "country": country,
                "primary_exchanges": spec["primary_exchanges"],
                "target_symbols": target_symbols,
                "public_security_master_status": spec["public_security_master_status"],
                "public_listing_rows": int(len(public_country)),
                "public_l2_candidate_rows": int(len(public_l2)),
                "public_rank_quality": public_rank_quality,
                "public_rank_quality_counts": "|".join(f"{key}:{value}" for key, value in sorted(quality_counts.items())),
                "validated_refinitiv_price_symbols": validated_symbols,
                "effective_l2_security_master_rows": effective_security_master_rows,
                "security_master_shortfall": max(0, target_symbols - effective_security_master_rows),
                "price_symbol_shortfall": max(0, target_symbols - validated_symbols),
                "readiness": readiness,
                "next_action": spec["next_action"],
            }
        )
    return pd.DataFrame(rows)


def build_refinitiv_l2_symbol_map_input(
    ranked_public_candidates: pd.DataFrame,
    include_curated_seeds: bool = True,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if ranked_public_candidates is not None and not ranked_public_candidates.empty:
        candidates = ranked_public_candidates[ranked_public_candidates["l2_target_candidate"].fillna(False)].copy()
        candidates = candidates[candidates["country"].astype("string").str.strip().isin(ASEAN_L2_COUNTRIES)].copy()
        candidates["l2_input_reason"] = "ranked_public_l2_candidate"
        frames.append(candidates)
    if include_curated_seeds:
        seeds = default_refinitiv_symbol_seed_listings()
        seeds = seeds[seeds["country"].isin(ASEAN_L2_COUNTRIES)].copy()
        seeds["l2_input_reason"] = "curated_refinitiv_seed"
        frames.append(seeds)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True, sort=False)
    dedupe_cols = [column for column in ("country", "exchange", "ticker", "refinitiv_ric") if column in combined.columns]
    if dedupe_cols:
        combined = combined.drop_duplicates(dedupe_cols, keep="first")
    return combined.sort_values(["country", "exchange", "ticker"], na_position="last").reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    data = frame[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in data.iterrows():
        values = [_clean_text(row[column]) for column in columns]
        lines.append("| " + " | ".join(value.replace("|", "/") for value in values) + " |")
    return "\n".join(lines)


def _build_markdown_summary(
    run_id: str,
    country_targets: pd.DataFrame,
    data_requirements: pd.DataFrame,
    artifact: dict[str, str],
) -> str:
    target_cols = [
        "country",
        "target_symbols",
        "public_l2_candidate_rows",
        "public_rank_quality",
        "validated_refinitiv_price_symbols",
        "effective_l2_security_master_rows",
        "security_master_shortfall",
        "price_symbol_shortfall",
        "readiness",
    ]
    requirement_cols = ["layer", "dataset", "research_gate", "status"]
    return "\n".join(
        [
            f"# ASEAN L2 Data Depth Plan - {run_id}",
            "",
            "This artifact is the build plan for the default three-country ASEAN production research universe: Malaysia, Singapore, and Indonesia. Thailand is outside the current scope, and Philippines/Vietnam are removed from the active project universe.",
            "",
            "## Country Targets",
            "",
            _markdown_table(country_targets, target_cols),
            "",
            "## Required Data Layers",
            "",
            _markdown_table(data_requirements, requirement_cols),
            "",
            "## Files",
            "",
            f"- Ranked public candidates: `{artifact['ranked_public_candidates_path']}`",
            f"- L2 candidate symbols: `{artifact['l2_candidate_symbols_path']}`",
            f"- Refinitiv symbol-map input: `{artifact['refinitiv_symbol_map_input_path']}`",
            f"- Vendor security-master requests: `{artifact['vendor_requests_path']}`",
            f"- Data requirements: `{artifact['data_requirements_path']}`",
            "",
            "## Next Commands",
            "",
            "Validate public/seed candidates in Refinitiv:",
            "",
            "```bash",
            "uv run python examples/run_refinitiv_symbol_map.py \\",
            f"  --run-id {run_id}-SYMBOL-MAP \\",
            f"  --listings {artifact['refinitiv_symbol_map_input_path']} \\",
            "  --sample-per-market 100 \\",
            "  --max-symbols 700 \\",
            "  --no-curated-seeds",
            "```",
            "",
            "After the symbol map has at least 50-100 validated symbols per country, run the 20-year historical backfill in country chunks. Do not promote any result beyond research-smoke until delistings, actions, liquidity, fundamentals, estimates, FX, taxes, and transaction-cost assumptions are attached point-in-time.",
        ]
    )


def write_asean_l2_plan_artifacts(
    public_listings: pd.DataFrame,
    symbol_map: pd.DataFrame | None = None,
    artifact_root: str | Path = "artifacts/asean_l2_data_plan",
    run_id: str | None = None,
    target_per_country: int = 100,
) -> AseanL2PlanArtifact:
    run_id = run_id or datetime.now(timezone.utc).strftime("ASEAN-L2-DATA-PLAN-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    ranked = rank_asean_l2_public_candidates(public_listings, target_per_country=target_per_country)
    l2_candidates = ranked[ranked["l2_target_candidate"].fillna(False)].copy() if not ranked.empty else ranked.copy()
    refinitiv_input = build_refinitiv_l2_symbol_map_input(ranked)
    country_targets = summarize_asean_l2_country_targets(
        ranked_public_candidates=ranked,
        symbol_map=symbol_map,
        target_per_country=target_per_country,
    )
    vendor_requests = build_asean_l2_vendor_requests(target_per_country=target_per_country)
    data_requirements = build_asean_l2_data_requirements()

    ranked_public_candidates_path = artifact_dir / "ranked_public_candidates.csv"
    l2_candidate_symbols_path = artifact_dir / "l2_candidate_symbols.csv"
    refinitiv_symbol_map_input_path = artifact_dir / "refinitiv_symbol_map_input.csv"
    country_targets_path = artifact_dir / "country_targets.csv"
    vendor_requests_path = artifact_dir / "vendor_security_master_requests.csv"
    data_requirements_path = artifact_dir / "data_requirements.csv"
    markdown_path = artifact_dir / "asean_l2_data_plan.md"
    manifest_path = artifact_dir / "manifest.json"

    ranked.to_csv(ranked_public_candidates_path, index=False)
    l2_candidates.to_csv(l2_candidate_symbols_path, index=False)
    refinitiv_input.to_csv(refinitiv_symbol_map_input_path, index=False)
    country_targets.to_csv(country_targets_path, index=False)
    vendor_requests.to_csv(vendor_requests_path, index=False)
    data_requirements.to_csv(data_requirements_path, index=False)

    path_map = {
        "ranked_public_candidates_path": str(ranked_public_candidates_path),
        "l2_candidate_symbols_path": str(l2_candidate_symbols_path),
        "refinitiv_symbol_map_input_path": str(refinitiv_symbol_map_input_path),
        "country_targets_path": str(country_targets_path),
        "vendor_requests_path": str(vendor_requests_path),
        "data_requirements_path": str(data_requirements_path),
    }
    markdown_path.write_text(_build_markdown_summary(run_id, country_targets, data_requirements, path_map), encoding="utf-8")

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir),
        "target_per_country": int(target_per_country),
        "public_listing_rows": int(len(public_listings)) if public_listings is not None else 0,
        "ranked_public_candidate_rows": int(len(ranked)),
        "l2_candidate_rows": int(len(l2_candidates)),
        "refinitiv_symbol_map_input_rows": int(len(refinitiv_input)),
        "countries": list(ASEAN_L2_COUNTRIES),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **path_map,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return AseanL2PlanArtifact(
        artifact_dir=artifact_dir,
        ranked_public_candidates_path=ranked_public_candidates_path,
        l2_candidate_symbols_path=l2_candidate_symbols_path,
        refinitiv_symbol_map_input_path=refinitiv_symbol_map_input_path,
        country_targets_path=country_targets_path,
        vendor_requests_path=vendor_requests_path,
        data_requirements_path=data_requirements_path,
        markdown_path=markdown_path,
        manifest_path=manifest_path,
        ranked_public_candidates=ranked,
        l2_candidate_symbols=l2_candidates,
        refinitiv_symbol_map_input=refinitiv_input,
        country_targets=country_targets,
        vendor_requests=vendor_requests,
        data_requirements=data_requirements,
    )
