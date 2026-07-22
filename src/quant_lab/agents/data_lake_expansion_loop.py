from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd


CORE_ACCEPTED_COUNTRIES: tuple[str, ...] = ("Malaysia", "Singapore", "Indonesia")
REMOVED_COUNTRIES: tuple[str, ...] = (
    "Vietnam",
    "Philippines",
    "China",
    "Taiwan",
)
DEFAULT_EXPANSION_COUNTRIES: tuple[str, ...] = (
    "Hong Kong",
    "South Korea",
    "Japan",
    "United States",
)
PARKED_COUNTRIES: tuple[str, ...] = ("Thailand",)


@dataclass(frozen=True)
class DataLakeExpansionConfig:
    run_id: str
    artifact_root: Path = Path("artifacts/data_lake_expansion")
    start_date: str = "2006-01-01"
    end_date: str = "2026-07-06"
    include_us: bool = True
    include_parked_thailand: bool = False
    batch_size: int = 6
    smoke_max_plan_rows: int = 1
    symbol_map_sample_per_market: int = 3
    symbol_map_max_symbols: int = 150
    refinitiv_session: str = "desktop.workspace"
    existing_field_catalog_csv: str = (
        "artifacts/refinitiv_field_probe/REFINITIV-FIELD-PROBE-ASEAN-CORRECTED-20260706/field_catalog.csv"
    )
    baseline_symbol_map_csv: str = "artifacts/refinitiv_symbol_map/REFINITIV-SYMBOL-MAP-L1-20260707/symbol_map.csv"


@dataclass(frozen=True)
class DataLakeExpansionArtifacts:
    run_id: str
    artifact_dir: Path
    market_targets_path: Path
    agent_roster_path: Path
    acceptance_gates_path: Path
    extraction_queue_path: Path
    commands_path: Path
    summary_path: Path
    manifest_path: Path
    market_targets: pd.DataFrame
    extraction_queue: pd.DataFrame


def latest_data_lake_expansion_run_id() -> str:
    return datetime.now(timezone.utc).strftime("DATA-LAKE-EXPANSION-%Y%m%d-%H%M%S")


def default_data_lake_expansion_config(run_id: str | None = None) -> DataLakeExpansionConfig:
    return DataLakeExpansionConfig(run_id=run_id or latest_data_lake_expansion_run_id())


def build_market_targets(config: DataLakeExpansionConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {
            "country": "Malaysia",
            "country_code": "MYS",
            "exchange": "Bursa Malaysia",
            "exchange_code": "XKLS",
            "region": "ASEAN",
            "stage": "core_accepted",
            "priority": 0,
            "include_by_default": True,
            "status": "accepted_5_of_5_baseline",
            "target_security_count": 62,
            "default_data_tier": "U2",
            "route_state": "complete_for_current_U2_gate",
            "notes": "Baseline market already accepted by the institutional data-quality loop.",
        },
        {
            "country": "Singapore",
            "country_code": "SGP",
            "exchange": "SGX",
            "exchange_code": "XSES",
            "region": "ASEAN",
            "stage": "core_accepted",
            "priority": 0,
            "include_by_default": True,
            "status": "accepted_5_of_5_baseline",
            "target_security_count": 66,
            "default_data_tier": "U2",
            "route_state": "complete_for_current_U2_gate",
            "notes": "Baseline market already accepted by the institutional data-quality loop.",
        },
        {
            "country": "Indonesia",
            "country_code": "IDN",
            "exchange": "IDX",
            "exchange_code": "XIDX",
            "region": "ASEAN",
            "stage": "core_accepted",
            "priority": 0,
            "include_by_default": True,
            "status": "accepted_5_of_5_baseline",
            "target_security_count": 41,
            "default_data_tier": "U2",
            "route_state": "complete_for_current_U2_gate",
            "notes": "Baseline market already accepted by the institutional data-quality loop.",
        },
    ]
    if config.include_parked_thailand:
        rows.append(
            {
                "country": "Thailand",
                "country_code": "THA",
                "exchange": "SET",
                "exchange_code": "XBKK",
                "region": "ASEAN",
                "stage": "parked_optional",
                "priority": 2,
                "include_by_default": False,
                "status": "parked_pending_tradability_and_refinitiv_route_validation",
                "target_security_count": 100,
                "default_data_tier": "U2_candidate",
                "route_state": "public_listing_adapter_exists_refinitiv_routes_unaccepted",
                "notes": "Optional only; not part of the default production research universe.",
            }
        )
    rows.extend(_north_asia_rows())
    if config.include_us:
        rows.extend(_us_rows())
    targets = pd.DataFrame(rows)
    removed = set(REMOVED_COUNTRIES)
    targets = targets.loc[~targets["country"].isin(removed)].copy()
    return targets.reset_index(drop=True)


def build_agent_roster() -> pd.DataFrame:
    rows = [
        (
            "chief_data_pm",
            "Portfolio-manager owner",
            "Own market admission order, stop alpha promotion below 5/5, and decide when a market graduates.",
            "artifacts/data_lake_expansion, operator-supplied run ledger",
            "Every target market has a 5/5 scorecard or a documented blocker.",
        ),
        (
            "exchange_universe_agent",
            "Universe and listing miner",
            "Mine exchange listings, current names, exchange codes, currency metadata, and public identifiers.",
            "artifacts/universes, artifacts/data_lake/universe",
            "Listing files are partitioned and deduped with Vietnam/Philippines absent.",
        ),
        (
            "refinitiv_symbol_map_agent",
            "RIC map validator",
            "Build and live-validate RIC candidates in bounded samples before full symbol-map promotion.",
            "artifacts/refinitiv_symbol_map",
            "Validated symbol map exists per market with failures isolated.",
        ),
        (
            "historical_price_agent",
            "OHLCV backfill owner",
            "Create plan-only, smoke, then full 20-year historical OHLCV jobs with long partition output.",
            "artifacts/historical_backfill, artifacts/data_lake/prices_long",
            "Every completed batch has coverage, schema, and manifest rows.",
        ),
        (
            "security_master_agent",
            "Survivorship-bias steward",
            "Add active, inactive, delisted, listing windows, identifier history, and RIC-change coverage.",
            "artifacts/data_lake/pit/security_master_history",
            "Dead-name denominator is present before any market enters alpha.",
        ),
        (
            "corporate_actions_agent",
            "Adjustment integrity owner",
            "Backfill splits, dividends, event-grain actions, and adjusted/raw price reconciliation.",
            "artifacts/refinitiv_corporate_actions_events, artifacts/adjusted_return_reconstruction",
            "Adjusted total-return reconstruction passes quarantine thresholds.",
        ),
        (
            "pit_fundamentals_ibes_agent",
            "PIT fundamental and estimates steward",
            "Probe and backfill fundamentals, IBES estimates/revisions, actuals/surprises, targets, and timestamps.",
            "artifacts/refinitiv_pit_historical_backfill, artifacts/refinitiv_ibes_revision_route_probe",
            "Known-at timestamps and leakage lags are validated.",
        ),
        (
            "fx_calendar_tca_agent",
            "FX, calendar, liquidity, and cost steward",
            "Build FX, exchange calendars, halts, ADV/turnover, market cap/free float, and TCA assumptions.",
            "artifacts/fx_country_risk, artifacts/refinitiv_pit_historical_backfill",
            "Local and USD return panels plus liquidity/cost filters are complete.",
        ),
        (
            "data_quality_auditor",
            "Independent QA gatekeeper",
            "Rerun scorecards, schema checks, leakage checks, sample tie-outs, and memory bound checks.",
            "artifacts/institutional_data_quality",
            "Market score is 5/5 with zero critical blockers.",
        ),
        (
            "storage_ops_agent",
            "Data-lake operations",
            "Keep batches bounded, resumable, Drive-syncable, and partitioned without large in-memory joins.",
            "artifacts/data_lake, commands.log",
            "Every heavy job has plan-only, smoke, resume, and command log evidence.",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["agent_id", "role", "mandate", "write_scope", "stop_condition"],
    )


def build_acceptance_gates() -> pd.DataFrame:
    rows = [
        (
            "G1",
            "Universe and security master",
            "All active/inactive listed equities, stable IDs, listing/delist dates, RIC/ticker history, and current-vs-historical denominator.",
            "critical",
            "security_master_history market manifest, listing_windows, identifier_history, universe_membership",
        ),
        (
            "G2",
            "Prices, actions, liquidity, and capacity",
            "20-year raw and adjusted daily OHLCV, split/dividend events, raw-adjusted reconciliation, ADV/turnover, market cap/free float, and capacity flags.",
            "critical",
            "historical_backfill manifest, corporate_actions_events, adjusted_return_reconstruction, liquidity_market_cap_free_float PIT backfill",
        ),
        (
            "G3",
            "PIT fundamentals, IBES, and events",
            "Point-in-time fundamentals, estimates/revisions, actuals/surprises, target/recommendation timestamps, and release calendars.",
            "critical",
            "PIT historical backfill, route history map, IBES revision timestamp probe",
        ),
        (
            "G4",
            "FX, calendars, halts, and TCA",
            "Daily FX, local-vs-USD returns, exchange calendars, closures/halts/suspensions, withholding-tax and cost assumptions.",
            "critical",
            "fx_country_risk artifact, calendar/TCA policy, halt/suspension route evidence",
        ),
        (
            "G5",
            "Leakage, reproducibility, and lake operations",
            "As-of join checks, embargoed decision calendar, no future data, versioned manifests, compaction, partition health, checksums, immutable run IDs, and reproducible commands.",
            "critical",
            "PIT leakage QA report, commands.log, manifest JSON, partition_health.csv, compaction_manifest.csv, market scorecard artifact",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["gate_id", "gate_name", "requirement", "severity", "required_evidence"],
    )


def build_extraction_queue(config: DataLakeExpansionConfig, targets: pd.DataFrame) -> pd.DataFrame:
    countries = _active_expansion_country_string(config)
    rows: list[dict[str, object]] = []

    def add(
        sequence: int,
        agent_id: str,
        phase: str,
        market_group: str,
        command_type: str,
        command: str,
        gate: str,
        bounded: bool = True,
        promotes_alpha: bool = False,
        blocking: bool = True,
        memory_policy: str = "bounded_batches_no_large_cross_market_dataframe",
    ) -> None:
        rows.append(
            {
                "sequence": sequence,
                "agent_id": agent_id,
                "phase": phase,
                "market_group": market_group,
                "countries": countries if market_group != "US" else "United States",
                "command_type": command_type,
                "command": command,
                "acceptance_gate": gate,
                "bounded": bounded,
                "blocking": blocking,
                "promotes_alpha": promotes_alpha,
                "memory_policy": memory_policy,
            }
        )

    add(
        10,
        "exchange_universe_agent",
        "listing_discovery",
        "North Asia",
        "runnable",
        "uv run python examples/run_universe_mining_asia.py --sources hkex,jpx,krx,twse,tpex,sse,szse --continue-on-error",
        "G1",
    )
    add(
        20,
        "exchange_universe_agent",
        "listing_discovery",
        "US",
        "runnable",
        "uv run python examples/run_universe_mining_us.py",
        "G1",
        blocking=config.include_us,
    )
    add(
        30,
        "refinitiv_symbol_map_agent",
        "ric_symbol_map_smoke",
        "North Asia + US",
        "requires_listings_csv",
        (
            "uv run --extra refinitiv python examples/run_refinitiv_symbol_map.py "
            f"--run-id REFINITIV-SYMBOL-MAP-NORTH-ASIA-US-SMOKE-{_date_slug(config.run_id)} "
            "--listings {combined_listings_csv} "
            f"--sample-per-market {config.symbol_map_sample_per_market} "
            f"--max-symbols {config.symbol_map_max_symbols} "
            f"--session {config.refinitiv_session} --count 5 --sleep-seconds 0.05 --progress-every 25"
        ),
        "G1",
    )
    add(
        35,
        "historical_price_agent",
        "jpx_price_smoke_first",
        "North Asia",
        "runnable",
        (
            "uv run --extra refinitiv python examples/run_refinitiv_historical_backfill.py "
            f"--run-id REFINITIV-REMAINING-L1-JPX-QA-{_date_slug(config.run_id)} "
            f"--symbol-map-csv {config.baseline_symbol_map_csv} "
            f"--field-catalog-csv {config.existing_field_catalog_csv} "
            "--include-countries \"Japan\" "
            "--start-date 2021-01-01 "
            f"--end-date {config.end_date} "
            "--window-years 3 --batch-size 2 --max-tickers 2 "
            f"--max-plan-rows {config.smoke_max_plan_rows} --interval 1d"
        ),
        "G2",
    )
    add(
        40,
        "historical_price_agent",
        "refinitiv_20y_plan_only",
        "North Asia + US",
        "requires_validated_symbol_map",
        (
            "uv run --extra refinitiv python examples/run_refinitiv_historical_backfill.py "
            f"--run-id REFINITIV-EXPANSION-HISTORICAL-20Y-PLAN-{_date_slug(config.run_id)} "
            "--symbol-map-csv {validated_symbol_map_csv} "
            f"--field-catalog-csv {config.existing_field_catalog_csv} "
            f"--include-countries \"{countries}\" "
            f"--start-date {config.start_date} --end-date {config.end_date} "
            f"--window-years 5 --batch-size {config.batch_size} --plan-only"
        ),
        "G2",
    )
    add(
        50,
        "historical_price_agent",
        "refinitiv_20y_smoke",
        "North Asia + US",
        "requires_validated_symbol_map",
        (
            "uv run --extra refinitiv python examples/run_refinitiv_historical_backfill.py "
            f"--run-id REFINITIV-EXPANSION-HISTORICAL-20Y-SMOKE-{_date_slug(config.run_id)} "
            "--symbol-map-csv {validated_symbol_map_csv} "
            f"--field-catalog-csv {config.existing_field_catalog_csv} "
            f"--include-countries \"{countries}\" "
            f"--start-date {config.start_date} --end-date {config.end_date} "
            f"--window-years 5 --batch-size {config.batch_size} "
            f"--max-plan-rows {config.smoke_max_plan_rows}"
        ),
        "G2",
    )
    add(
        60,
        "security_master_agent",
        "security_master_route_expansion",
        "North Asia + US",
        "implementation_required",
        (
            "Extend quant_lab.data.security_master_discovery beyond MYS/SGP/IDN, then run one-country active/inactive "
            "Discovery smoke before full dead-name backfill."
        ),
        "G1",
        memory_policy="page_discovery_results_top_100_with_resume_tokens",
    )
    add(
        70,
        "corporate_actions_agent",
        "corporate_actions_smoke",
        "North Asia + US",
        "requires_market_pit_plan",
        (
            "uv run --extra refinitiv python examples/run_refinitiv_corporate_actions_event_backfill.py "
            f"--run-id REFINITIV-CORPACT-EVENTS-EXPANSION-SMOKE-{_date_slug(config.run_id)} "
            "--tier u2 --start-date 2006-01-01 "
            f"--end-date {config.end_date} --batch-size 50 --max-plan-rows 1 --sleep-seconds 0.02"
        ),
        "G2",
    )
    add(
        80,
        "pit_fundamentals_ibes_agent",
        "pit_ibes_route_smoke",
        "North Asia + US",
        "requires_market_pit_plan",
        (
            "Run PIT field, route-history, and IBES revision timestamp probes per market; promote only fields with "
            "known-at dates and non-null historical coverage."
        ),
        "G3",
        memory_policy="sample_per_country_then_chunk_by_dataset_and_window",
    )
    add(
        90,
        "fx_calendar_tca_agent",
        "fx_liquidity_calendar_smoke",
        "North Asia + US",
        "requires_market_currency_map",
        (
            "Build FX/country-risk, exchange-calendar, halt/suspension, ADV/turnover, market-cap/free-float, and TCA "
            "artifacts per market before alpha admission."
        ),
        "G2,G4",
    )
    add(
        95,
        "storage_ops_agent",
        "partition_health_and_compaction",
        "North Asia + US",
        "implementation_required",
        (
            "Generate partition_health.csv and compaction_manifest.csv; compact daily OHLCV toward one file per "
            "provider/interval/adjustment/country/exchange/year where practical before gold/ML reads."
        ),
        "G5",
        blocking=True,
        memory_policy="scan_metadata_and_filtered_pyarrow_datasets_no_full_root_read",
    )
    add(
        100,
        "data_quality_auditor",
        "scorecard_gate",
        "North Asia + US",
        "implementation_required",
        (
            "Create per-market institutional data-quality scorecards mirroring the accepted MYS/SGP/IDN U2 loop; "
            "no market enters alpha until 5/5 with zero critical blockers."
        ),
        "G5",
        promotes_alpha=False,
    )
    add(
        105,
        "data_quality_auditor",
        "tiny_chunk_depth_report_only",
        "North Asia",
        "conditional_runnable_after_jpx_smoke",
        (
            "uv run python examples/run_data_depth_report.py "
            f"artifacts/data_lake/prices_long/REFINITIV-REMAINING-L1-JPX-QA-{_date_slug(config.run_id)}/ohlcv "
            f"--run-id REFINITIV-REMAINING-L1-JPX-QA-{_date_slug(config.run_id)}-DEPTH "
            f"--extraction-manifest-path artifacts/historical_backfill/REFINITIV-REMAINING-L1-JPX-QA-{_date_slug(config.run_id)}/historical_manifest.csv"
        ),
        "G5",
        blocking=False,
        memory_policy="tiny_qa_chunk_only_never_full_lake",
    )
    add(
        110,
        "storage_ops_agent",
        "ops_log_and_resume",
        "North Asia + US",
        "manual_required",
        "Append every command to commands.log, keep plan/smoke/full manifests immutable, and resume failed batches only.",
        "G5",
        blocking=False,
    )
    return pd.DataFrame(rows)


def write_data_lake_expansion_artifacts(
    config: DataLakeExpansionConfig | None = None,
) -> DataLakeExpansionArtifacts:
    config = config or default_data_lake_expansion_config()
    artifact_dir = Path(config.artifact_root) / config.run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    targets = build_market_targets(config)
    agents = build_agent_roster()
    gates = build_acceptance_gates()
    queue = build_extraction_queue(config, targets)

    market_targets_path = artifact_dir / "market_targets.csv"
    agent_roster_path = artifact_dir / "agent_roster.csv"
    acceptance_gates_path = artifact_dir / "acceptance_gates.csv"
    extraction_queue_path = artifact_dir / "extraction_queue.csv"
    commands_path = artifact_dir / "commands.md"
    summary_path = artifact_dir / "summary.md"
    manifest_path = artifact_dir / "manifest.json"

    targets.to_csv(market_targets_path, index=False)
    agents.to_csv(agent_roster_path, index=False)
    gates.to_csv(acceptance_gates_path, index=False)
    queue.to_csv(extraction_queue_path, index=False)
    commands_path.write_text(_commands_md(config, queue), encoding="utf-8")
    summary_path.write_text(_summary_md(config, targets, agents, gates, queue), encoding="utf-8")
    manifest = {
        "run_id": config.run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "generated_at_utc": _now_utc(),
        "status": "ready_for_bounded_data_lake_expansion_loop",
        "ibkr_status": "blocked_not_in_scope",
        "alpha_status": "blocked_until_new_markets_reach_5_of_5",
        "core_accepted_countries": list(CORE_ACCEPTED_COUNTRIES),
        "removed_countries": list(REMOVED_COUNTRIES),
        "default_expansion_countries": list(_active_expansion_countries(config)),
        "market_target_rows": int(len(targets)),
        "extraction_queue_rows": int(len(queue)),
        "bounded_by_default": bool(queue["bounded"].all()),
        "promotes_alpha_rows": int(queue["promotes_alpha"].sum()),
        "artifacts": {
            "market_targets": str(market_targets_path.resolve()),
            "agent_roster": str(agent_roster_path.resolve()),
            "acceptance_gates": str(acceptance_gates_path.resolve()),
            "extraction_queue": str(extraction_queue_path.resolve()),
            "commands": str(commands_path.resolve()),
            "summary": str(summary_path.resolve()),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return DataLakeExpansionArtifacts(
        run_id=config.run_id,
        artifact_dir=artifact_dir,
        market_targets_path=market_targets_path,
        agent_roster_path=agent_roster_path,
        acceptance_gates_path=acceptance_gates_path,
        extraction_queue_path=extraction_queue_path,
        commands_path=commands_path,
        summary_path=summary_path,
        manifest_path=manifest_path,
        market_targets=targets,
        extraction_queue=queue,
    )


def _north_asia_rows() -> list[dict[str, object]]:
    return [
        _target("Hong Kong", "HKG", "HKEX", "XHKG", "North Asia", 1, "U2_candidate", 300),
        _target("South Korea", "KOR", "KRX", "XKRX", "North Asia", 1, "U2_candidate", 300),
        _target("Japan", "JPN", "JPX", "XTKS", "North Asia", 1, "U2_candidate", 500),
    ]


def _us_rows() -> list[dict[str, object]]:
    return [
        _target("United States", "USA", "NYSE", "XNYS", "US", 3, "discovery_candidate", 1000),
        _target("United States", "USA", "NASDAQ", "XNAS", "US", 3, "discovery_candidate", 1000),
        _target("United States", "USA", "NYSE American", "XASE", "US", 3, "discovery_candidate", 300),
    ]


def _target(
    country: str,
    country_code: str,
    exchange: str,
    exchange_code: str,
    region: str,
    priority: int,
    tier: str,
    target_security_count: int,
) -> dict[str, object]:
    return {
        "country": country,
        "country_code": country_code,
        "exchange": exchange,
        "exchange_code": exchange_code,
        "region": region,
        "stage": "expansion_candidate",
        "priority": priority,
        "include_by_default": True,
        "status": "requires_5_of_5_market_admission",
        "target_security_count": target_security_count,
        "default_data_tier": tier,
        "route_state": "listing_miner_exists_refinitiv_symbol_and_pit_routes_need_market_acceptance",
        "notes": "Cannot feed alpha until security-master, corporate-action, liquidity, PIT, FX, and leakage gates pass.",
    }


def _active_expansion_countries(config: DataLakeExpansionConfig) -> tuple[str, ...]:
    countries = list(DEFAULT_EXPANSION_COUNTRIES if config.include_us else DEFAULT_EXPANSION_COUNTRIES[:-1])
    if config.include_parked_thailand:
        countries.insert(0, "Thailand")
    return tuple(country for country in countries if country not in REMOVED_COUNTRIES)


def _active_expansion_country_string(config: DataLakeExpansionConfig) -> str:
    return ",".join(_active_expansion_countries(config))


def _commands_md(config: DataLakeExpansionConfig, queue: pd.DataFrame) -> str:
    runnable = queue[
        queue["command_type"].isin(
            ["runnable", "requires_listings_csv", "requires_validated_symbol_map", "conditional_runnable_after_jpx_smoke"]
        )
    ]
    lines = [
        f"# Data-Lake Expansion Commands: {config.run_id}",
        "",
        "Run these in order. The first two are directly runnable. Later commands require the produced listings or validated symbol-map CSV.",
        "",
        "IBKR and live trading are intentionally out of scope.",
        "",
    ]
    for _, row in runnable.iterrows():
        lines.extend(
            [
                f"## {int(row['sequence'])}. {row['phase']}",
                "",
                "```bash",
                str(row["command"]),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Admission Rule",
            "",
            "A market cannot enter the alpha factory until all acceptance gates are scored 5/5 with zero critical blockers.",
            "",
        ]
    )
    return "\n".join(lines)


def _summary_md(
    config: DataLakeExpansionConfig,
    targets: pd.DataFrame,
    agents: pd.DataFrame,
    gates: pd.DataFrame,
    queue: pd.DataFrame,
) -> str:
    included = ", ".join(_active_expansion_countries(config))
    target_summary = _markdown_table(
        targets.groupby(["region", "stage"], dropna=False)
        .size()
        .reset_index(name="markets")
        .sort_values(["region", "stage"])
    )
    return "\n".join(
        [
            f"# Data-Lake Expansion Loop: {config.run_id}",
            "",
            "## Verdict",
            "",
            "Yes, run another agent loop, but make it a data-quality expansion loop rather than an alpha loop.",
            "The accepted Malaysia/Singapore/Indonesia U2 lake remains the baseline. IBKR/live trading stays blocked.",
            "",
            "## Default Scope",
            "",
            f"- Expansion countries: {included}",
            f"- Removed countries: {', '.join(REMOVED_COUNTRIES)}",
            f"- Parked optional countries: {', '.join(PARKED_COUNTRIES)}",
            f"- Date range target: {config.start_date} to {config.end_date}",
            "",
            "## Market Targets",
            "",
            target_summary,
            "",
            "## Agent Loop",
            "",
            f"- Specialist agents: {len(agents)}",
            f"- Acceptance gates: {len(gates)}",
            f"- Queue rows: {len(queue)}",
            "- Stop condition: every admitted market reaches 5/5 with zero critical blockers.",
            "- Alpha promotion: blocked until the new market scorecards pass.",
            "- Lake operations: partition-health and compaction evidence is required before ML/gold reads.",
            "",
            "## First Move",
            "",
            "Run the public listing miners, then validate a small Refinitiv RIC symbol-map sample. Only after that should the 20-year historical plan-only and smoke jobs run.",
            "",
        ]
    )


def _date_slug(run_id: str) -> str:
    digits = "".join(ch for ch in run_id if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else datetime.now(timezone.utc).strftime("%Y%m%d")


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    values = frame.astype(str)
    columns = list(values.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in values.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join(lines)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CORE_ACCEPTED_COUNTRIES",
    "DEFAULT_EXPANSION_COUNTRIES",
    "PARKED_COUNTRIES",
    "REMOVED_COUNTRIES",
    "DataLakeExpansionArtifacts",
    "DataLakeExpansionConfig",
    "build_acceptance_gates",
    "build_agent_roster",
    "build_extraction_queue",
    "build_market_targets",
    "default_data_lake_expansion_config",
    "latest_data_lake_expansion_run_id",
    "write_data_lake_expansion_artifacts",
]
