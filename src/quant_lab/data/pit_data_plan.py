from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ACTIVE_COUNTRIES: tuple[str, ...] = ("Malaysia", "Singapore", "Indonesia")


@dataclass(frozen=True)
class PITDatasetSpec:
    dataset_id: str
    owner_agent: str
    priority: int
    source_family: str
    cadence: str
    point_in_time_key: str
    required_columns: tuple[str, ...]
    partition_by: tuple[str, ...]
    blocks_model_families: tuple[str, ...]
    readiness_gate: str
    notes: str


@dataclass(frozen=True)
class PITDataPlanResult:
    run_id: str
    artifact_dir: Path
    selected_universe: pd.DataFrame
    dataset_requirements: pd.DataFrame
    field_probe_queue: pd.DataFrame
    extraction_batches: pd.DataFrame
    target_artifacts: pd.DataFrame
    readiness_gates: pd.DataFrame
    manifest_path: Path
    summary_path: Path


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("PIT-DATA-PLAN-%Y%m%d-%H%M%S")


def default_pit_dataset_specs() -> list[PITDatasetSpec]:
    """Institutional data sets required before fundamentals/estimates enter backtests."""

    return [
        PITDatasetSpec(
            dataset_id="security_master_pit",
            owner_agent="asean_security_master_agent",
            priority=1,
            source_family="refinitiv_reference_or_discovery",
            cadence="monthly_snapshot_plus_listing_events",
            point_in_time_key="asof_date",
            required_columns=(
                "sid",
                "ric",
                "perm_id",
                "isin",
                "sedol",
                "ticker",
                "company_name",
                "country",
                "exchange",
                "currency",
                "listing_status",
                "first_trade_date",
                "delisting_date",
                "asof_date",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "asof_year"),
            blocks_model_families=("all_equity_backtests",),
            readiness_gate="no_current_survivor_only_universe",
            notes="Needed to eliminate survivorship bias and handle ticker/RIC changes.",
        ),
        PITDatasetSpec(
            dataset_id="corporate_actions",
            owner_agent="corporate_actions_agent",
            priority=1,
            source_family="refinitiv_corporate_actions",
            cadence="event_history",
            point_in_time_key="effective_date,announcement_date,source_loaded_at_utc",
            required_columns=(
                "sid",
                "ric",
                "action_type",
                "announcement_date",
                "effective_date",
                "ex_date",
                "pay_date",
                "split_ratio",
                "cash_amount",
                "currency",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "effective_year"),
            blocks_model_families=("all_return_backtests",),
            readiness_gate="official_adjustment_truth_available",
            notes="Needed before production claims on adjusted returns, dividends, splits, and special distributions.",
        ),
        PITDatasetSpec(
            dataset_id="delistings_listing_history",
            owner_agent="delisting_survivorship_agent",
            priority=1,
            source_family="refinitiv_reference_or_exchange_history",
            cadence="event_history",
            point_in_time_key="event_date,source_loaded_at_utc",
            required_columns=(
                "sid",
                "ric",
                "event_type",
                "event_date",
                "reason",
                "new_ric",
                "new_ticker",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "event_year"),
            blocks_model_families=("all_equity_backtests",),
            readiness_gate="delisted_names_in_universe_or_explicitly_excluded",
            notes="Needed so the backtest denominator includes dead names and listing changes.",
        ),
        PITDatasetSpec(
            dataset_id="fundamentals_pit",
            owner_agent="fundamentals_pit_agent",
            priority=2,
            source_family="worldscope_or_qa_company_fundamentals",
            cadence="fiscal_period_version_history",
            point_in_time_key="period_end_date,release_date,source_loaded_at_utc",
            required_columns=(
                "sid",
                "ric",
                "fiscal_period",
                "period_end_date",
                "release_date",
                "available_time_utc",
                "currency",
                "revenue",
                "gross_profit",
                "ebit",
                "net_income",
                "eps",
                "book_value",
                "total_assets",
                "total_debt",
                "operating_cash_flow",
                "capex",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "fiscal_year"),
            blocks_model_families=("value", "quality", "profitability", "earnings_quality"),
            readiness_gate="release_timestamp_or_safe_lag_policy_present",
            notes="Latest/restated-only fundamentals are not allowed into research-grade models.",
        ),
        PITDatasetSpec(
            dataset_id="ibes_estimates_revisions",
            owner_agent="analyst_revisions_agent",
            priority=2,
            source_family="ibes_history",
            cadence="daily_or_event_revision_history",
            point_in_time_key="available_time_utc",
            required_columns=(
                "sid",
                "ric",
                "measure",
                "fiscal_period",
                "period_end_date",
                "broker_or_contributor_id",
                "estimate_value",
                "estimate_currency",
                "revision_pct",
                "recommendation",
                "target_price",
                "available_time_utc",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "available_year"),
            blocks_model_families=("analyst_revisions", "earnings_momentum", "expectations_revision"),
            readiness_gate="estimate_revision_history_not_current_snapshot",
            notes="Consensus snapshots alone are not enough; we need timestamped revisions.",
        ),
        PITDatasetSpec(
            dataset_id="earnings_events_actuals",
            owner_agent="earnings_events_agent",
            priority=2,
            source_family="ibes_actuals_company_events_or_exchange_calendar",
            cadence="event_history",
            point_in_time_key="release_time_utc,available_time_utc",
            required_columns=(
                "sid",
                "ric",
                "event_type",
                "fiscal_period",
                "event_date",
                "release_time_utc",
                "available_time_utc",
                "actual_eps",
                "consensus_eps_before_release",
                "standardized_unexpected_earnings",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "event_year"),
            blocks_model_families=("earnings_momentum", "post_earnings_drift", "event_driven"),
            readiness_gate="earnings_release_time_available_or_safe_next_day_policy",
            notes="Needed to avoid using after-close releases before they were tradable.",
        ),
        PITDatasetSpec(
            dataset_id="liquidity_market_cap_free_float",
            owner_agent="liquidity_adv_agent",
            priority=2,
            source_family="refinitiv_reference_history_plus_ohlcv",
            cadence="daily_or_monthly_history",
            point_in_time_key="asof_date",
            required_columns=(
                "sid",
                "ric",
                "asof_date",
                "market_cap",
                "free_float_percent",
                "shares_outstanding",
                "turnover_value_local",
                "adv_20_local",
                "adv_60_local",
                "adv_252_local",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "asof_year"),
            blocks_model_families=("capacity", "liquidity", "portfolio_construction"),
            readiness_gate="capacity_filters_use_lagged_adv_and_free_float",
            notes="Needed for USD 5m capacity, country sleeves, and tradability filters.",
        ),
        PITDatasetSpec(
            dataset_id="fx_country_risk",
            owner_agent="fx_benchmark_agent",
            priority=3,
            source_family="lseg_fx_macro_exchange_calendars",
            cadence="daily_history",
            point_in_time_key="date",
            required_columns=(
                "country",
                "date",
                "currency",
                "usd_fx_rate",
                "local_short_rate",
                "market_holiday_flag",
                "withholding_tax_assumption",
                "source_loaded_at_utc",
            ),
            partition_by=("country", "year"),
            blocks_model_families=("usd_returns", "country_allocation", "macro_fx"),
            readiness_gate="local_and_usd_return_paths_reproducible",
            notes="Needed for local-vs-USD returns and country-risk decomposition.",
        ),
    ]


FIELD_PROBE_BUNDLES: dict[str, tuple[dict[str, object], ...]] = {
    "security_master_pit": (
        {"field_role": "permanent_id", "candidate_fields": ("TR.PermID", "TR.OrganizationID"), "required": True},
        {"field_role": "isin", "candidate_fields": ("TR.ISIN", "TR.SecurityISIN"), "required": True},
        {"field_role": "sedol", "candidate_fields": ("TR.SEDOL",), "required": False},
        {"field_role": "listing_status", "candidate_fields": ("TR.ListingStatus", "TR.InstrumentStatus"), "required": True},
        {"field_role": "first_trade_date", "candidate_fields": ("TR.FirstTradeDate", "TR.ListingDate"), "required": True},
        {"field_role": "delisting_date", "candidate_fields": ("TR.DelistingDate",), "required": False},
        {"field_role": "currency", "candidate_fields": ("TR.Currency", "TR.PriceCurrency"), "required": True},
    ),
    "corporate_actions": (
        {"field_role": "dividend_event", "candidate_fields": ("TR.DivExDate", "TR.DivPayDate", "TR.DivUnadjustedGross"), "required": True},
        {"field_role": "split_event", "candidate_fields": ("TR.SplitFactor", "TR.SplitEffectiveDate"), "required": True},
        {"field_role": "action_announcement", "candidate_fields": ("TR.CAAnnouncementDate",), "required": False},
    ),
    "delistings_listing_history": (
        {"field_role": "delisting_date", "candidate_fields": ("TR.DelistingDate",), "required": True},
        {"field_role": "delisting_reason", "candidate_fields": ("TR.DelistingReason",), "required": False},
        {"field_role": "ric_change", "candidate_fields": ("TR.RICChangeDate", "TR.NewRIC"), "required": False},
    ),
    "fundamentals_pit": (
        {"field_role": "revenue", "candidate_fields": ("TR.F.TotRevenue", "TR.Revenue", "QA:1001"), "required": True},
        {"field_role": "gross_profit", "candidate_fields": ("TR.F.GrossProfit", "QA:1100"), "required": True},
        {"field_role": "eps", "candidate_fields": ("TR.F.EPSDilutedExclExtra", "QA:5201"), "required": True},
        {"field_role": "book_value", "candidate_fields": ("TR.F.BookValuePerShare", "TR.BookValuePerShare"), "required": True},
        {"field_role": "release_date", "candidate_fields": ("TR.F.PeriodEndDate", "TR.F.ReportDate", "TR.FilingDate"), "required": True},
    ),
    "ibes_estimates_revisions": (
        {"field_role": "eps_estimate", "candidate_fields": ("IBES:EPS", "TR.EPSMean", "TR.EPSMeanEstimate"), "required": True},
        {"field_role": "revenue_estimate", "candidate_fields": ("IBES:Revenue", "TR.RevenueMean", "TR.SalesMeanEstimate"), "required": True},
        {"field_role": "recommendation", "candidate_fields": ("IBES:Recommendation", "TR.RecMean", "TR.RecommendationMean"), "required": False},
        {"field_role": "target_price", "candidate_fields": ("IBES:TargetPrice", "TR.TargetPriceMean"), "required": False},
        {"field_role": "revision_time", "candidate_fields": ("IBES:EstimateDate", "TR.EstimateDate"), "required": True},
    ),
    "earnings_events_actuals": (
        {"field_role": "earnings_release", "candidate_fields": ("TR.EventStartDate", "TR.EarningsReleaseDate"), "required": True},
        {"field_role": "actual_eps", "candidate_fields": ("IBES:ActualEPS", "TR.EPSActual"), "required": True},
        {"field_role": "surprise", "candidate_fields": ("IBES:Surprise", "TR.EPSSurprise"), "required": True},
    ),
    "liquidity_market_cap_free_float": (
        {"field_role": "market_cap", "candidate_fields": ("TR.CompanyMarketCap", "Company Market Cap"), "required": True},
        {"field_role": "free_float", "candidate_fields": ("TR.FreeFloatPct", "Free Float (Percent)"), "required": True},
        {"field_role": "shares", "candidate_fields": ("TR.SharesOutstanding", "Outstanding Shares"), "required": True},
        {"field_role": "adv_value", "candidate_fields": ("TR.AvgDailyValTraded30D", "Average Daily Value Traded - 30 Days"), "required": True},
    ),
    "fx_country_risk": (
        {"field_role": "fx_spot", "candidate_fields": ("MYR=", "SGD=", "IDR="), "required": True},
        {"field_role": "holiday_calendar", "candidate_fields": ("exchange_calendar",), "required": True},
        {"field_role": "tax_assumption", "candidate_fields": ("manual_policy_table",), "required": True},
    ),
}


def _load_l2_manifest(l2_artifact_dir: Path) -> dict[str, Any]:
    manifest_path = l2_artifact_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_symbol_map(symbol_map_path: str | Path | None, l2_manifest: dict[str, Any]) -> pd.DataFrame:
    candidate = Path(symbol_map_path) if symbol_map_path else None
    if candidate is None and l2_manifest.get("symbol_map_path"):
        candidate = Path(str(l2_manifest["symbol_map_path"]))
    if candidate is None or not candidate.exists():
        return pd.DataFrame()
    return pd.read_csv(candidate, low_memory=False)


def _select_universe(
    l2_artifact_dir: Path,
    symbol_map: pd.DataFrame,
    active_countries: tuple[str, ...],
) -> pd.DataFrame:
    tiers_path = l2_artifact_dir / "eligible_universe_tiers.csv"
    if not tiers_path.exists():
        raise FileNotFoundError(f"missing eligible universe tiers: {tiers_path}")

    tiers = pd.read_csv(tiers_path, low_memory=False)
    required = {"ticker", "country", "u2_backtest_eligible", "u3_liquid_proxy"}
    missing = required - set(tiers.columns)
    if missing:
        raise ValueError(f"eligible universe tiers missing columns: {sorted(missing)}")

    active = {country.strip().lower() for country in active_countries}
    selected = tiers[tiers["country"].astype(str).str.strip().str.lower().isin(active)].copy()
    selected = selected[selected["u2_backtest_eligible"].fillna(False).astype(bool) | selected["u3_liquid_proxy"].fillna(False).astype(bool)]
    if selected.empty:
        raise ValueError("no U2/U3 securities selected for PIT data plan")

    if not symbol_map.empty and {"ticker", "country", "refinitiv_ric"}.issubset(symbol_map.columns):
        smap = symbol_map[["ticker", "country", "refinitiv_ric", "refinitiv_symbol_status", "validated_close_field"]].copy()
        smap = smap.dropna(subset=["ticker", "country", "refinitiv_ric"]).drop_duplicates(["ticker", "country"], keep="first")
        selected = selected.reset_index(drop=True).copy()
        selected["_row_id"] = selected.index
        keyed_maps = []
        for key_col, preference in (("refinitiv_ric", 1), ("ticker", 2)):
            keyed = smap.copy()
            keyed["symbol_key"] = keyed[key_col]
            keyed["match_preference"] = preference
            keyed_maps.append(keyed[["country", "symbol_key", "refinitiv_ric", "refinitiv_symbol_status", "validated_close_field", "match_preference"]])
        smap_keys = pd.concat(keyed_maps, ignore_index=True).dropna(subset=["symbol_key"]).drop_duplicates(
            ["country", "symbol_key"],
            keep="first",
        )

        match_parts = []
        for left_col, preference_offset in (("ticker", 0), ("local_symbol", 10)):
            if left_col not in selected.columns:
                continue
            left = selected[["_row_id", "country", left_col]].rename(columns={left_col: "symbol_key"})
            match = left.merge(smap_keys, on=["country", "symbol_key"], how="left")
            match = match.dropna(subset=["refinitiv_ric"])
            if not match.empty:
                match["match_preference"] = match["match_preference"] + preference_offset
                match_parts.append(match)
        if match_parts:
            matches = (
                pd.concat(match_parts, ignore_index=True)
                .sort_values(["_row_id", "match_preference"])
                .drop_duplicates("_row_id", keep="first")
            )
            selected = selected.merge(
                matches[["_row_id", "refinitiv_ric", "refinitiv_symbol_status", "validated_close_field"]],
                on="_row_id",
                how="left",
            )
        else:
            selected["refinitiv_ric"] = pd.NA
            selected["refinitiv_symbol_status"] = pd.NA
            selected["validated_close_field"] = pd.NA
        selected = selected.drop(columns=["_row_id"])
    else:
        selected["refinitiv_ric"] = pd.NA
        selected["refinitiv_symbol_status"] = pd.NA
        selected["validated_close_field"] = pd.NA

    selected["selected_for_u2"] = selected["u2_backtest_eligible"].fillna(False).astype(bool)
    selected["selected_for_u3"] = selected["u3_liquid_proxy"].fillna(False).astype(bool)
    selected["has_refinitiv_ric"] = selected["refinitiv_ric"].notna()
    keep = [
        "ticker",
        "refinitiv_ric",
        "sid",
        "local_symbol",
        "name",
        "country",
        "exchange",
        "currency",
        "selected_for_u2",
        "selected_for_u3",
        "has_refinitiv_ric",
        "refinitiv_symbol_status",
        "validated_close_field",
        "adv_60_country_rank",
        "adv_60_local_lag1_latest",
        "vol_63_lag1_latest",
    ]
    keep = [col for col in keep if col in selected.columns]
    sort_cols = [col for col in ("country", "selected_for_u3", "adv_60_country_rank", "ticker") if col in selected.columns]
    ascending = [True, False, True, True][: len(sort_cols)]
    return selected[keep].sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _dataset_requirements_frame(specs: list[PITDatasetSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        row = asdict(spec)
        row["required_columns"] = ",".join(spec.required_columns)
        row["partition_by"] = ",".join(spec.partition_by)
        row["blocks_model_families"] = ",".join(spec.blocks_model_families)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["priority", "dataset_id"]).reset_index(drop=True)


def _field_probe_queue(specs: list[PITDatasetSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for entry in FIELD_PROBE_BUNDLES.get(spec.dataset_id, ()):
            for candidate_field in entry["candidate_fields"]:
                rows.append(
                    {
                        "dataset_id": spec.dataset_id,
                        "owner_agent": spec.owner_agent,
                        "source_family": spec.source_family,
                        "field_role": entry["field_role"],
                        "candidate_field": candidate_field,
                        "required": bool(entry["required"]),
                        "validation_status": "needs_live_probe",
                        "validation_method": "LSEG Workspace Data Item Browser or lseg.data probe on a 3-RIC sample.",
                    }
                )
    return pd.DataFrame(rows)


def _year_chunks(start_date: str, end_date: str) -> list[tuple[str, str, int]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks = []
    for year in range(start.year, end.year + 1):
        chunk_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        chunk_end = min(end, pd.Timestamp(year=year, month=12, day=31))
        chunks.append((chunk_start.date().isoformat(), chunk_end.date().isoformat(), year))
    return chunks


def _chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[i : i + size] for i in range(0, len(values), size)]


def _build_extraction_batches(
    selected_universe: pd.DataFrame,
    specs: list[PITDatasetSpec],
    start_date: str,
    end_date: str,
    batch_size: int,
    tiers: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    years = _year_chunks(start_date, end_date)
    time_sliced = {
        "corporate_actions",
        "delistings_listing_history",
        "fundamentals_pit",
        "ibes_estimates_revisions",
        "earnings_events_actuals",
        "liquidity_market_cap_free_float",
        "fx_country_risk",
    }
    reference_snapshot = {"security_master_pit"}

    for spec in specs:
        for tier in tiers:
            tier_col = f"selected_for_{tier.lower()}"
            if tier_col not in selected_universe.columns:
                raise ValueError(f"unknown universe tier in plan: {tier}")
            tier_universe = selected_universe[selected_universe[tier_col].fillna(False).astype(bool)].copy()
            if tier_universe.empty:
                continue
            for country, group in tier_universe.groupby("country", sort=True):
                rics = sorted(group["refinitiv_ric"].dropna().astype(str).unique())
                if spec.dataset_id == "fx_country_risk":
                    rics = [str(country)]
                if not rics:
                    rows.append(
                        {
                            "batch_id": f"{spec.dataset_id}-{tier}-{country}-blocked-no-ric",
                            "dataset_id": spec.dataset_id,
                            "universe_tier": tier,
                            "country": country,
                            "start_date": start_date,
                            "end_date": end_date,
                            "year": pd.NA,
                            "ric_count": 0,
                            "ric_list": "",
                            "status": "blocked_missing_refinitiv_ric",
                            "expected_output": "",
                            "owner_agent": spec.owner_agent,
                        }
                    )
                    continue

                date_chunks = years if spec.dataset_id in time_sliced else [(end_date, end_date, pd.Timestamp(end_date).year)]
                if spec.dataset_id in reference_snapshot:
                    date_chunks = [(end_date, end_date, pd.Timestamp(end_date).year)]
                ric_chunks = [rics] if spec.dataset_id == "fx_country_risk" else _chunked(rics, batch_size)
                for chunk_number, ric_chunk in enumerate(ric_chunks, start=1):
                    for chunk_start, chunk_end, year in date_chunks:
                        batch_id = f"{spec.dataset_id}-{tier}-{str(country).lower().replace(' ', '_')}-{year}-{chunk_number:03d}"
                        rows.append(
                            {
                                "batch_id": batch_id,
                                "dataset_id": spec.dataset_id,
                                "universe_tier": tier,
                                "country": country,
                                "start_date": chunk_start,
                                "end_date": chunk_end,
                                "year": year,
                                "ric_count": len(ric_chunk),
                                "ric_list": ",".join(ric_chunk),
                                "status": "ready_after_field_validation",
                                "expected_output": f"artifacts/data_lake/pit/{spec.dataset_id}/country={country}/tier={tier}/year={year}/{batch_id}.parquet",
                                "owner_agent": spec.owner_agent,
                            }
                        )

    return pd.DataFrame(rows)


def _target_artifacts(specs: list[PITDatasetSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "dataset_id": spec.dataset_id,
                "target_path_template": f"artifacts/data_lake/pit/{spec.dataset_id}/" + "/".join(f"{col}={{{col}}}" for col in spec.partition_by) + "/part.parquet",
                "partition_by": ",".join(spec.partition_by),
                "minimum_schema_columns": ",".join(spec.required_columns),
                "readiness_gate": spec.readiness_gate,
            }
        )
    return pd.DataFrame(rows)


def _readiness_gates(specs: list[PITDatasetSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        rows.append(
            {
                "gate_id": spec.readiness_gate,
                "dataset_id": spec.dataset_id,
                "blocks_model_families": ",".join(spec.blocks_model_families),
                "pass_condition": (
                    "Dataset exists as partitioned parquet, required columns populated, "
                    "known-at timestamp present where applicable, and leakage QA passes."
                ),
                "current_status": "not_started",
            }
        )
    return pd.DataFrame(rows).sort_values(["dataset_id"]).reset_index(drop=True)


def _summary_md(
    run_id: str,
    selected_universe: pd.DataFrame,
    dataset_requirements: pd.DataFrame,
    field_probe_queue: pd.DataFrame,
    extraction_batches: pd.DataFrame,
    lseg_connector_status: str,
) -> str:
    by_country = selected_universe.groupby("country").agg(
        u2_names=("selected_for_u2", "sum"),
        u3_names=("selected_for_u3", "sum"),
        rics=("has_refinitiv_ric", "sum"),
    )
    priority = dataset_requirements[["dataset_id", "priority", "owner_agent", "source_family", "readiness_gate"]]
    blocked_batches = int(extraction_batches["status"].ne("ready_after_field_validation").sum()) if not extraction_batches.empty else 0
    lines = [
        f"# {run_id} PIT Data Acquisition Plan",
        "",
        "## Scope",
        "- Active markets: Malaysia, Singapore, Indonesia.",
        "- Purpose: build the missing point-in-time data lake before fundamentals, estimates, revisions, and corporate actions enter alpha research.",
        f"- LSEG connector probe status: `{lseg_connector_status}`.",
        "",
        "## Selected Universe",
        by_country.to_markdown(),
        "",
        "## Dataset Queue",
        priority.to_markdown(index=False),
        "",
        "## Execution Counts",
        f"- Dataset groups: `{len(dataset_requirements)}`",
        f"- Field-probe rows: `{len(field_probe_queue)}`",
        f"- Extraction batches: `{len(extraction_batches)}`",
        f"- Blocked batches: `{blocked_batches}`",
        "",
        "## Hard Rules",
        "- No fundamental, estimate, recommendation, or earnings feature can enter a research-grade backtest without `available_time_utc` or a documented conservative lag policy.",
        "- Current snapshots are allowed for discovery only, not for model evidence.",
        "- Delisted names and listing changes must either be present or the backtest must be explicitly labeled current-survivor.",
        "- Every batch writes a manifest and quality flags before feature generation.",
        "",
        "## Next Execution",
        "1. Reauthenticate LSEG connector or run through local Workspace `lseg.data` session.",
        "2. Probe candidate fields on a 3-RIC sample per country.",
        "3. Freeze accepted field map.",
        "4. Run priority-1 batches: PIT security master, corporate actions, delistings.",
        "5. Run priority-2 batches: fundamentals, IBES revisions, earnings events, liquidity/free-float.",
    ]
    return "\n".join(lines) + "\n"


def build_pit_data_acquisition_plan(
    l2_artifact_dir: str | Path,
    artifact_root: str | Path = "artifacts/pit_data_plan",
    run_id: str | None = None,
    symbol_map_path: str | Path | None = None,
    active_countries: tuple[str, ...] = ACTIVE_COUNTRIES,
    tiers: tuple[str, ...] = ("u3", "u2"),
    start_date: str = "2006-01-01",
    end_date: str = "2026-07-06",
    batch_size: int = 25,
    lseg_connector_status: str = "not_checked",
) -> PITDataPlanResult:
    """Build a concrete missing-data acquisition queue for institutional backtests."""

    run_id = run_id or _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    l2_artifact = Path(l2_artifact_dir)
    l2_manifest = _load_l2_manifest(l2_artifact)
    symbol_map = _load_symbol_map(symbol_map_path, l2_manifest)
    selected_universe = _select_universe(l2_artifact, symbol_map, active_countries)
    specs = default_pit_dataset_specs()
    dataset_requirements = _dataset_requirements_frame(specs)
    field_probe_queue = _field_probe_queue(specs)
    extraction_batches = _build_extraction_batches(
        selected_universe,
        specs,
        start_date=start_date,
        end_date=end_date,
        batch_size=batch_size,
        tiers=tiers,
    )
    target_artifacts = _target_artifacts(specs)
    readiness_gates = _readiness_gates(specs)

    selected_universe.to_csv(artifact_dir / "selected_universe.csv", index=False)
    dataset_requirements.to_csv(artifact_dir / "dataset_requirements.csv", index=False)
    field_probe_queue.to_csv(artifact_dir / "field_probe_queue.csv", index=False)
    extraction_batches.to_csv(artifact_dir / "extraction_batches.csv", index=False)
    target_artifacts.to_csv(artifact_dir / "target_artifacts.csv", index=False)
    readiness_gates.to_csv(artifact_dir / "readiness_gates.csv", index=False)

    summary_path = artifact_dir / "summary.md"
    summary_path.write_text(
        _summary_md(
            run_id,
            selected_universe,
            dataset_requirements,
            field_probe_queue,
            extraction_batches,
            lseg_connector_status,
        ),
        encoding="utf-8",
    )
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "l2_artifact_dir": str(l2_artifact.resolve()),
        "symbol_map_path": str(Path(symbol_map_path).resolve()) if symbol_map_path else l2_manifest.get("symbol_map_path", ""),
        "active_countries": list(active_countries),
        "tiers": list(tiers),
        "start_date": start_date,
        "end_date": end_date,
        "batch_size": batch_size,
        "lseg_connector_status": lseg_connector_status,
        "selected_names": int(len(selected_universe)),
        "selected_u2": int(selected_universe["selected_for_u2"].sum()) if "selected_for_u2" in selected_universe else 0,
        "selected_u3": int(selected_universe["selected_for_u3"].sum()) if "selected_for_u3" in selected_universe else 0,
        "names_with_refinitiv_ric": int(selected_universe["has_refinitiv_ric"].sum()) if "has_refinitiv_ric" in selected_universe else 0,
        "dataset_count": int(len(dataset_requirements)),
        "field_probe_rows": int(len(field_probe_queue)),
        "extraction_batches": int(len(extraction_batches)),
        "artifacts": {
            "selected_universe": str((artifact_dir / "selected_universe.csv").resolve()),
            "dataset_requirements": str((artifact_dir / "dataset_requirements.csv").resolve()),
            "field_probe_queue": str((artifact_dir / "field_probe_queue.csv").resolve()),
            "extraction_batches": str((artifact_dir / "extraction_batches.csv").resolve()),
            "target_artifacts": str((artifact_dir / "target_artifacts.csv").resolve()),
            "readiness_gates": str((artifact_dir / "readiness_gates.csv").resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return PITDataPlanResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        selected_universe=selected_universe,
        dataset_requirements=dataset_requirements,
        field_probe_queue=field_probe_queue,
        extraction_batches=extraction_batches,
        target_artifacts=target_artifacts,
        readiness_gates=readiness_gates,
        manifest_path=manifest_path,
        summary_path=summary_path,
    )
