from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


INVENTORY_VERSION = "QFORGE20-STRATEGY-INVENTORY-V1"
KAKUSHADZE_SOURCE = "Quant_Lib/151_Trading_Strategies.md"
FINDING_ALPHAS_SOURCE = "Quant_Lib/Finding_Alphas.md"
KAUFMAN_SOURCE = "Quant_Lib/Trading_Systems_and_Methods.md"
CHAN_SOURCE = "Quant_Lib/Quantitative_Trading.md"


@dataclass(frozen=True)
class SourceRecord:
    path: str
    sha256: str
    bytes: int
    line_count: int
    heading_count: int
    canonical_path: str
    duplicate_of: str | None
    source_role: str


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    name: str
    source_path: str
    source_line: int
    source_heading: str
    asset_class: str
    mathematical_cluster: str
    research_status: str
    data_requirements: tuple[str, ...]
    activation_lane: str
    prior_research_overlap: str | None = None
    caution: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized[:96]


def _source_role(filename: str) -> str:
    value = filename.lower()
    if any(
        token in value
        for token in (
            "trading strategies",
            "trading systems",
            "quantitative trading",
            "finding alphas",
            "systematic trading",
            "following the trend",
            "algorithmic trading",
            "expected returns",
            "factor investing",
        )
    ):
        return "strategy_source"
    if any(
        token in value
        for token in (
            "financial machine learning",
            "machine learning",
            "artificial intelligence",
            "optimization",
            "time series",
            "stochastic",
            "brownian",
            "monte carlo",
        )
    ):
        return "methodology_or_math"
    if any(token in value for token in ("microstructure", "liquidity", "dma", "exchanges")):
        return "execution_or_microstructure"
    if any(token in value for token in ("risk", "portfolio", "kelly", "asset management")):
        return "portfolio_or_risk"
    return "supporting_source"


def scan_source_corpus(library_root: Path) -> tuple[SourceRecord, ...]:
    """Inventory an authorized Markdown research corpus without altering it."""

    # Root-level operational notes are not books. Keep the frozen 69-document
    # research corpus stable when readiness/runbook Markdown is added later.
    non_book_root_notes = {"LSEG_MONDAY_READINESS.md"}
    paths = sorted(
        (
            *(path for path in library_root.glob("*.md") if path.name not in non_book_root_notes),
            *(library_root / "Quant_Lib").glob("*.md"),
        )
    )
    raw: list[dict[str, Any]] = []
    by_hash: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        relative = path.relative_to(library_root).as_posix()
        content = path.read_text(encoding="utf-8", errors="replace")
        digest = _sha256_file(path)
        by_hash[digest].append(relative)
        raw.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": path.stat().st_size,
                "line_count": content.count("\n") + (1 if content else 0),
                "heading_count": sum(
                    1 for line in content.splitlines() if re.match(r"^#{1,4}\s+\S", line)
                ),
                "source_role": _source_role(path.name),
            }
        )
    canonical = {
        digest: sorted(group, key=lambda item: (item.count("/"), len(item), item.casefold()))[0]
        for digest, group in by_hash.items()
    }
    return tuple(
        SourceRecord(
            **item,
            canonical_path=canonical[item["sha256"]],
            duplicate_of=(
                None if item["path"] == canonical[item["sha256"]] else canonical[item["sha256"]]
            ),
        )
        for item in raw
    )


def _major_section(section: str | None, fallback: int | None) -> int | None:
    if section:
        return int(section.split(".", maxsplit=1)[0])
    return fallback


def _kakushadze_classification(
    major: int | None, name: str
) -> tuple[str, str, str, tuple[str, ...], str, str | None, str | None]:
    lower = name.casefold()
    if major == 2:
        return (
            "equity_options",
            "options_payoff_or_volatility",
            "kiv_refinitiv",
            ("point_in_time_option_chain", "strike", "expiry", "bid_ask", "implied_volatility"),
            "options_and_volatility",
            None,
            "Daily equity OHLCV cannot reconstruct executable option payoffs.",
        )
    if major == 3:
        if "earnings" in lower:
            return (
                "equity",
                "earnings_revision_or_surprise",
                "kiv_refinitiv",
                ("point_in_time_earnings", "announcement_timestamp", "analyst_consensus"),
                "fundamentals_and_estimates",
                None,
                "Point-in-time timestamps and revisions are mandatory.",
            )
        if lower == "value" or "multifactor" in lower:
            return (
                "equity",
                "fundamental_factor",
                "kiv_refinitiv",
                ("point_in_time_fundamentals", "filing_timestamp", "security_master_history"),
                "fundamentals_and_estimates",
                None,
                "Current OHLCV does not contain point-in-time accounting data.",
            )
        if "implied volatility" in lower:
            return (
                "equity_options",
                "options_volatility",
                "kiv_refinitiv",
                ("point_in_time_option_chain", "implied_volatility", "bid_ask"),
                "options_and_volatility",
                None,
                None,
            )
        if "m&a" in lower or "event-driven" in lower:
            return (
                "equity",
                "event_driven",
                "kiv_refinitiv",
                ("announcement_timestamp", "deal_terms", "deal_status_history"),
                "events_deals_and_index_history",
                None,
                "Survivorship-safe event timestamps are mandatory.",
            )
        if "market-making" in lower:
            return (
                "equity",
                "market_microstructure",
                "kiv_refinitiv",
                ("tick_history", "quotes", "order_book", "fees_and_rebates"),
                "tick_quotes_and_order_book",
                None,
                "Daily bars are not valid evidence for market-making.",
            )
        if "pairs" in lower or "statistical arbitrage" in lower:
            return (
                "equity",
                "relative_value_cointegration",
                "supported_now_research_only",
                ("adj_close", "sector_history", "stable_security_master"),
                "current_daily_ohlcv",
                None,
                "Dry-run execution remains KIV until borrow availability and borrow cost are sourced.",
            )
        if "machine learning" in lower or "knn" in lower:
            return (
                "equity",
                "nonlinear_price_model",
                "supported_now_high_overfit_risk",
                ("lagged_daily_features", "purged_nested_validation"),
                "current_daily_ohlcv",
                None,
                "Not shortlisted until simple new families establish a baseline.",
            )
        overlap_map = (
            (("price-momentum",), "momentum", "MYS-S-MOMENTUM-126"),
            (("low-volatility",), "volatility_defensive", "MYS-S-LOWVOL-126"),
            (("residual momentum",), "residual_momentum", "MYS-S-RESIDMOM-126"),
            (("mean-reversion",), "mean_reversion", "MYS-S-MEANREV-63/MYS-S-BOLLREV-63"),
            (("moving average",), "trend_filter", "MYS-S-EWMAC-16-64"),
            (("support and resistance", "channel"), "breakout_channel", "MYS-S-BREAKOUT-55/MYS-S-CHANNEL-55"),
            (("alpha combos",), "post_hoc_combination", "V1 predeclared combinations"),
        )
        for needles, cluster, overlap in overlap_map:
            if any(needle in lower for needle in needles):
                return (
                    "equity",
                    cluster,
                    "already_researched_or_equivalent",
                    ("adj_close",),
                    "current_daily_ohlcv",
                    overlap,
                    "Do not count horizon or indicator variants as a new hypothesis family.",
                )
    if major == 4:
        return (
            "etf",
            "etf_rotation_or_overlay",
            "kiv_refinitiv",
            ("point_in_time_etf_universe", "constituent_history", "adjusted_prices"),
            "events_deals_and_index_history",
            None,
            "The frozen Malaysia universe contains listed equities, not a validated ETF cohort.",
        )
    section_defaults: Mapping[int, tuple[str, str, tuple[str, ...], str]] = {
        5: ("fixed_income", "rates_and_credit", ("bond_terms", "evaluated_prices", "yield_curves"), "fixed_income_and_curves"),
        6: ("index", "index_arbitrage", ("index_constituent_history", "futures_or_etf_quotes", "dividends"), "events_deals_and_index_history"),
        7: ("volatility", "options_volatility", ("vix_futures_curve", "options", "variance_swaps"), "options_and_volatility"),
        8: ("fx", "fx_carry_or_momentum", ("spot_fx", "forward_points", "interest_rates"), "fx_forwards_and_rates"),
        9: ("commodities", "commodity_carry_or_factor", ("futures_curves", "contract_metadata", "hedging_pressure"), "futures_and_commodities"),
        10: ("futures", "futures_trend_or_spread", ("futures_curves", "roll_calendar", "contract_metadata"), "futures_and_commodities"),
        11: ("structured_credit", "credit_carry_or_relative_value", ("cds_curves", "tranche_terms", "credit_events"), "fixed_income_and_curves"),
        12: ("convertibles", "convertible_arbitrage", ("convertible_terms", "equity_options", "borrow_cost"), "fixed_income_and_curves"),
        13: ("tax", "tax_arbitrage", ("tax_rules_history", "cross_border_instruments"), "legal_and_tax_reference"),
        14: ("macro", "inflation_energy_or_weather", ("macro_series", "swap_curves", "energy_and_weather_data"), "macro_and_economic_calendar"),
        15: ("distressed", "distressed_and_reorganization", ("debt_terms", "court_events", "capital_structure"), "events_deals_and_index_history"),
        16: ("real_estate", "real_estate_factor", ("property_or_reit_history", "fundamentals", "macro_series"), "fundamentals_and_estimates"),
        18: ("cross_asset", "machine_learning_or_sentiment", ("validated_features", "text_or_sentiment_history"), "news_text_and_sentiment"),
        19: ("macro", "macro_momentum_or_event", ("macro_vintages", "economic_release_calendar", "rates"), "macro_and_economic_calendar"),
    }
    if major == 17 and any(token in lower for token in ("money laundering", "pawnbroking", "loan sharking")):
        return (
            "non_investment",
            "non_admissible",
            "not_admissible",
            (),
            "none",
            None,
            "Excluded from investment research on legal, ethical, and mandate grounds.",
        )
    if major == 17:
        return (
            "cash_and_funding",
            "liquidity_or_funding",
            "kiv_refinitiv",
            ("funding_rates", "repo_terms", "cash_flows"),
            "fixed_income_and_curves",
            None,
            None,
        )
    if major in section_defaults:
        asset, cluster, requirements, lane = section_defaults[major]
        return asset, cluster, "kiv_refinitiv", requirements, lane, None, None
    return (
        "unknown",
        "unclassified",
        "kiv_manual_review",
        ("manual_data_contract_review",),
        "manual_review",
        None,
        "Preserved for manual classification; it is not authorized for testing.",
    )


def extract_kakushadze_candidates(library_root: Path) -> tuple[StrategyCandidate, ...]:
    path = library_root / KAKUSHADZE_SOURCE
    numbered = re.compile(r"^##\s+(?P<section>\d+(?:\.\d+)?)\s+(?:Strategy:\s*)?(?P<name>.+?)\s*$")
    unnumbered = re.compile(r"^##\s+Strategy:\s*(?P<name>.+?)\s*$")
    candidates: list[StrategyCandidate] = []
    current_major: int | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        match = numbered.match(line)
        section: str | None = None
        name: str | None = None
        if match:
            section = match.group("section")
            current_major = _major_section(section, current_major)
            if "Strategy:" in line or section == "3.10":
                name = match.group("name")
        else:
            match = unnumbered.match(line)
            if match:
                name = match.group("name")
        if not name:
            continue
        major = _major_section(section, current_major)
        asset, cluster, status, requirements, lane, overlap, caution = _kakushadze_classification(
            major, name
        )
        section_key = section or f"{major or 0}.x"
        candidates.append(
            StrategyCandidate(
                candidate_id=f"k151-{_slug(section_key)}-{_slug(name)}",
                name=name,
                source_path=KAKUSHADZE_SOURCE,
                source_line=line_number,
                source_heading=line.removeprefix("## "),
                asset_class=asset,
                mathematical_cluster=cluster,
                research_status=status,
                data_requirements=requirements,
                activation_lane=lane,
                prior_research_overlap=overlap,
                caution=caution,
            )
        )
    return tuple(candidates)


def _find_heading(library_root: Path, source_path: str, heading: str) -> int:
    path = library_root / source_path
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if line.lstrip("# ").strip().casefold() == heading.casefold():
            return line_number
    raise ValueError(f"source heading not found in {source_path}: {heading}")


def additional_source_candidates(library_root: Path) -> tuple[StrategyCandidate, ...]:
    specifications = (
        (
            "next-price-volume-confirmation",
            "Price-volume confirmation",
            FINDING_ALPHAS_SOURCE,
            "SEEKING PROFITS THROUGH PRICE AND VOLUME",
            "price_volume_interaction",
            ("adj_close", "volume", "adv_60_local_lag1"),
            "Use one signed, lagged price-volume interaction; no indicator menu.",
        ),
        (
            "next-lagged-illiquidity-premium",
            "Lagged illiquidity premium",
            FINDING_ALPHAS_SOURCE,
            "THE ILLIQUIDITY PREMIUM IN ASSET PRICES",
            "liquidity_premium",
            ("amihud_60_lag1", "adv_60_local_lag1", "spread_proxy"),
            "Capacity and turnover must be evaluated as first-class gates.",
        ),
        (
            "next-stock-calendar-seasonality",
            "Stock calendar seasonality",
            KAUFMAN_SOURCE,
            "Seasonal Patterns in Stocks",
            "calendar_seasonality",
            ("date", "adj_close", "simple_return"),
            "Pre-register one calendar effect; do not search days or months post hoc.",
        ),
        (
            "next-cointegrated-pairs",
            "Industry-restricted cointegrated pairs",
            CHAN_SOURCE,
            "STATIONARITY AND COINTEGRATION",
            "relative_value_cointegration",
            ("adj_close", "stable_security_master", "sector_history"),
            "Research is possible now; dry-run execution requires borrow availability and costs.",
        ),
        (
            "next-opening-gap-reversal",
            "Opening-gap reversal",
            KAUFMAN_SOURCE,
            "A Simple Gap Trading Method for Stocks",
            "overnight_intraday_decomposition",
            ("open", "adj_close", "overnight_return", "intraday_return"),
            "Use lagged signals and next executable prices; reject same-bar look-ahead.",
        ),
    )
    return tuple(
        StrategyCandidate(
            candidate_id=candidate_id,
            name=name,
            source_path=source_path,
            source_line=_find_heading(library_root, source_path, heading),
            source_heading=heading,
            asset_class="equity",
            mathematical_cluster=cluster,
            research_status="shortlisted_for_preregistration",
            data_requirements=requirements,
            activation_lane="current_daily_ohlcv",
            caution=caution,
        )
        for candidate_id, name, source_path, heading, cluster, requirements, caution in specifications
    )


def load_prior_research(prior_manifest: Path) -> tuple[dict[str, str], ...]:
    if not prior_manifest.exists():
        return ()
    with prior_manifest.open(newline="", encoding="utf-8") as handle:
        return tuple(dict(row) for row in csv.DictReader(handle))


def refinitiv_kiv_activation_matrix() -> tuple[dict[str, Any], ...]:
    """Data lanes, not commercial product promises; exact entitlements need vendor confirmation."""

    rows = (
        ("security_master_and_corporate_actions", ("security_master_history", "delisted_securities", "listing_events", "corporate_actions"), ("all_equity_families",), 1),
        ("fundamentals_and_estimates", ("point_in_time_financials", "filing_timestamps", "analyst_consensus", "estimate_revisions"), ("value", "quality", "earnings_revision", "fundamental_multifactor"), 2),
        ("events_deals_and_index_history", ("M&A_timestamps_and_terms", "deal_status_history", "index_constituent_history", "spinoffs"), ("event_driven", "merger_arbitrage", "index_rebalance"), 3),
        ("news_text_and_sentiment", ("timestamped_news", "entity_mapping", "novelty", "sentiment"), ("news_momentum", "sentiment", "announcement_reaction"), 4),
        ("securities_lending", ("borrow_availability", "borrow_fee", "short_interest"), ("pairs_trading_execution", "market_neutral_stat_arb", "short_leg_strategies"), 5),
        ("options_and_volatility", ("historical_option_chains", "bid_ask", "strike", "expiry", "implied_volatility", "greeks"), ("options_payoffs", "volatility_risk_premium", "dispersion"), 6),
        ("tick_quotes_and_order_book", ("trades", "quotes", "L1_or_L2_order_book", "exchange_fees"), ("market_making", "intraday_alpha", "execution_models"), 7),
        ("macro_and_economic_calendar", ("vintage_macro_series", "release_timestamps", "policy_rates"), ("macro_momentum", "economic_announcement", "inflation"), 8),
        ("fixed_income_and_curves", ("bond_terms", "evaluated_prices", "yield_curves", "credit_curves", "repo_rates"), ("rates", "credit", "convertibles", "funding"), 9),
        ("fx_forwards_futures_and_commodities", ("spot_and_forwards", "futures_curves", "roll_calendars", "contract_metadata"), ("carry", "term_structure", "commodity_and_fx_strategies"), 10),
    )
    return tuple(
        {
            "activation_lane": lane,
            "required_fields": list(fields),
            "unlocks_clusters": list(clusters),
            "suggested_priority": priority,
            "status": "kiv_until_entitlements_and_pit_qa_pass",
        }
        for lane, fields, clusters, priority in rows
    )


def build_inventory(
    *,
    library_root: Path,
    prior_manifest: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    sources = scan_source_corpus(library_root)
    kakushadze = extract_kakushadze_candidates(library_root)
    shortlist = additional_source_candidates(library_root)
    prior = load_prior_research(prior_manifest)
    all_candidates = (*kakushadze, *shortlist)
    duplicate_groups: dict[str, list[str]] = defaultdict(list)
    for source in sources:
        duplicate_groups[source.sha256].append(source.path)
    duplicates = [sorted(group) for group in duplicate_groups.values() if len(group) > 1]
    ledger_receipt = {
        "path": ledger_path.as_posix(),
        "exists": ledger_path.exists(),
        "sha256": _sha256_file(ledger_path) if ledger_path.exists() else None,
        "bytes": ledger_path.stat().st_size if ledger_path.exists() else None,
        "mutated_by_inventory": False,
    }
    return {
        "schema_version": 1,
        "inventory_version": INVENTORY_VERSION,
        "scope": "Malaysia research inventory only; no trial reservation, backtest, or dry run",
        "source_summary": {
            "markdown_books": len(sources),
            "canonical_books": sum(source.duplicate_of is None for source in sources),
            "exact_duplicate_files": sum(source.duplicate_of is not None for source in sources),
            "exact_duplicate_groups": len(duplicates),
            "source_roles": dict(sorted(Counter(source.source_role for source in sources).items())),
        },
        "source_manifest": [asdict(source) for source in sources],
        "exact_duplicate_groups": duplicates,
        "candidate_summary": {
            "kakushadze_strategy_headings": len(kakushadze),
            "shortlisted_distinct_families": len(shortlist),
            "statuses": dict(sorted(Counter(item.research_status for item in all_candidates).items())),
            "clusters": dict(sorted(Counter(item.mathematical_cluster for item in all_candidates).items())),
        },
        "strategy_candidates": [asdict(item) for item in kakushadze],
        "shortlist": [asdict(item) for item in shortlist],
        "prior_research_manifest": list(prior),
        "prior_research_family_count": len({row.get("family", "") for row in prior}),
        "kiv_activation_matrix": list(refinitiv_kiv_activation_matrix()),
        "ledger_receipt": ledger_receipt,
        "decision": {
            "unsupported_strategies_deleted": False,
            "unsupported_strategies_status": "KIV",
            "backtests_started": False,
            "trials_reserved": 0,
            "next_step": "write and freeze one five-hypothesis preregistration wave before any calculation",
        },
    }


def write_inventory_artifacts(inventory: Mapping[str, Any], output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "source_manifest.json": {
            "inventory_version": inventory["inventory_version"],
            "source_summary": inventory["source_summary"],
            "source_manifest": inventory["source_manifest"],
            "exact_duplicate_groups": inventory["exact_duplicate_groups"],
        },
        "clustered_strategy_inventory.json": {
            "inventory_version": inventory["inventory_version"],
            "candidate_summary": inventory["candidate_summary"],
            "strategy_candidates": inventory["strategy_candidates"],
            "prior_research_manifest": inventory["prior_research_manifest"],
        },
        "refinitiv_kiv_activation_matrix.json": {
            "inventory_version": inventory["inventory_version"],
            "commercial_note": "Confirm exact datasets, history, fields, timestamps, and entitlements with the vendor before purchase.",
            "activation_matrix": inventory["kiv_activation_matrix"],
        },
        "next_cohort_shortlist.json": {
            "inventory_version": inventory["inventory_version"],
            "scope": inventory["scope"],
            "shortlist": inventory["shortlist"],
            "ledger_receipt": inventory["ledger_receipt"],
            "decision": inventory["decision"],
        },
    }
    written: list[Path] = []
    for filename, payload in files.items():
        destination = output_dir / filename
        destination.write_text(
            json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(destination)
    return tuple(written)
