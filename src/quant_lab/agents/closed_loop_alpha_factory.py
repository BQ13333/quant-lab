from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quant_lab.models.source_strategies import build_source_strategy_model_suite, source_strategy_manifest
from quant_lab.research.cross_market import CrossMarketSweepResult, run_cross_market_sweep_from_long_ohlcv


UniverseTier = Literal["u2", "u3", "all"]


@dataclass(frozen=True)
class ClosedLoopAlphaFactoryConfig:
    """Configuration for the PM-owned closed-loop source strategy backtest."""

    l2_artifact_dir: str | Path
    artifact_root: str | Path = "artifacts/closed_loop_alpha_factory"
    run_id: str | None = None
    include_countries: tuple[str, ...] = ("Malaysia", "Singapore", "Indonesia")
    universe_tier: UniverseTier = "u3"
    min_net_sharpe: float = 3.0
    min_deflated_sharpe: float = 0.95
    train_size: int = 756
    test_size: int = 63
    embargo: int = 21
    target_volatility: float = 0.12
    max_leverage: float = 2.0
    max_country_weight: float = 0.45
    max_asset_weight: float = 0.08
    drawdown_limit: float = -0.25
    min_rows: int = 2500
    max_missing_ratio: float = 0.25
    cost_bps_grid: tuple[float, ...] = (15.0, 30.0, 50.0)
    max_models: int | None = None
    max_allocators: int | None = None
    precompute_full_signals: bool = True
    write_candidate_weights: bool = True
    candidate_weight_top_n: int = 10


@dataclass(frozen=True)
class ClosedLoopAlphaFactoryResult:
    run_id: str
    artifact_dir: Path
    sweep_result: CrossMarketSweepResult
    candidate_ledger: pd.DataFrame
    pm_promoted: pd.DataFrame
    data_readiness: dict[str, Any]
    agent_trace: list[dict[str, Any]]


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("CLOSED-LOOP-ALPHA-%Y%m%d-%H%M%S")


def closed_loop_allocator_configs(max_asset_weight: float) -> list[dict[str, Any]]:
    """Allocator grid for the first institutional-style loop pass."""

    return [
        {"weighting": "long_only_capped", "top_fraction": 0.10, "max_weight": max_asset_weight, "rebalance_frequency": "D"},
        {"weighting": "long_only_capped", "top_fraction": 0.10, "max_weight": max_asset_weight, "rebalance_frequency": "W"},
        {"weighting": "long_only_capped", "top_fraction": 0.20, "max_weight": max_asset_weight, "rebalance_frequency": "W"},
        {"weighting": "long_only_capped", "top_fraction": 0.20, "max_weight": max_asset_weight, "rebalance_frequency": "M"},
        {
            "weighting": "group_neutral_long_only_capped",
            "top_fraction": 0.20,
            "max_weight": max_asset_weight,
            "rebalance_frequency": "W",
        },
        {
            "weighting": "group_neutral_long_only_capped",
            "top_fraction": 0.20,
            "max_weight": max_asset_weight,
            "rebalance_frequency": "M",
        },
        {"weighting": "group_neutral_dollar_neutral", "top_fraction": 1.0, "rebalance_frequency": "W"},
        {"weighting": "group_neutral_dollar_neutral", "top_fraction": 1.0, "rebalance_frequency": "M"},
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _config_for_json(config: ClosedLoopAlphaFactoryConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["l2_artifact_dir"] = str(config.l2_artifact_dir)
    payload["artifact_root"] = str(config.artifact_root)
    payload["include_countries"] = list(config.include_countries)
    payload["cost_bps_grid"] = list(config.cost_bps_grid)
    return payload


def _tier_column(universe_tier: UniverseTier) -> str | None:
    if universe_tier == "u2":
        return "u2_backtest_eligible"
    if universe_tier == "u3":
        return "u3_liquid_proxy"
    if universe_tier == "all":
        return None
    raise ValueError("universe_tier must be one of 'u2', 'u3', or 'all'")


def _selected_tickers(l2_artifact_dir: Path, universe_tier: UniverseTier, include_countries: tuple[str, ...]) -> list[str] | None:
    tier_col = _tier_column(universe_tier)
    if tier_col is None:
        return None

    tiers_path = l2_artifact_dir / "eligible_universe_tiers.csv"
    if not tiers_path.exists():
        raise FileNotFoundError(f"missing eligible universe tiers: {tiers_path}")

    tiers = pd.read_csv(tiers_path)
    required = {"ticker", "country", tier_col}
    missing = required - set(tiers.columns)
    if missing:
        raise ValueError(f"eligible universe tiers missing columns: {sorted(missing)}")

    country_set = {country.strip().lower() for country in include_countries}
    mask = tiers["country"].astype(str).str.strip().str.lower().isin(country_set)
    mask &= tiers[tier_col].fillna(False).astype(bool)
    return sorted(tiers.loc[mask, "ticker"].astype(str).unique())


def _materialize_filtered_ohlcv(
    long_ohlcv_root: Path,
    run_dir: Path,
    universe_tier: UniverseTier,
    tickers: list[str] | None,
) -> Path:
    if tickers is None:
        return long_ohlcv_root

    ohlcv = pd.read_parquet(long_ohlcv_root)
    filtered = ohlcv[ohlcv["ticker"].astype(str).isin(set(tickers))].copy()
    if filtered.empty:
        raise ValueError(f"no OHLCV rows after {universe_tier} ticker filter")

    path = run_dir / f"filtered_ohlcv_{universe_tier}.parquet"
    filtered.to_parquet(path, index=False)
    return path


def apply_pm_sharpe_gate(
    candidates: pd.DataFrame,
    min_net_sharpe: float = 3.0,
    min_deflated_sharpe: float = 0.95,
) -> pd.DataFrame:
    """Add the PM promotion state and kill reasons to a candidate scoreboard."""

    ledger = candidates.copy()
    if ledger.empty:
        for col in ("goal_min_net_sharpe", "sharpe_goal_pass", "pm_gate_pass", "promotion_state", "kill_reason"):
            ledger[col] = []
        return ledger

    ledger["goal_min_net_sharpe"] = min_net_sharpe
    ledger["goal_min_deflated_sharpe"] = min_deflated_sharpe
    ledger["sharpe_goal_pass"] = ledger["net_sharpe"].fillna(float("-inf")) >= min_net_sharpe
    ledger["deflated_sharpe_goal_pass"] = ledger["deflated_sharpe"].fillna(float("-inf")) >= min_deflated_sharpe
    ledger["net_ir_positive"] = ledger["net_ir"].fillna(float("-inf")) > 0.0

    constraint_pass_cols = [col for col in ledger.columns if col.startswith("constraint_") and col.endswith("_passed")]

    states: list[str] = []
    reasons: list[str] = []
    pm_passes: list[bool] = []
    for _, row in ledger.iterrows():
        row_reasons: list[str] = []
        if not bool(row.get("passed_all", False)):
            failed = [
                col.removeprefix("constraint_").removesuffix("_passed")
                for col in constraint_pass_cols
                if not bool(row.get(col, False))
            ]
            row_reasons.append("failed_constraints=" + ",".join(failed or ["unknown"]))
        if float(row.get("net_sharpe", 0.0)) < min_net_sharpe:
            row_reasons.append(f"net_sharpe_below_{min_net_sharpe:g}")
        if float(row.get("deflated_sharpe", 0.0)) < min_deflated_sharpe:
            row_reasons.append(f"deflated_sharpe_below_{min_deflated_sharpe:g}")
        if float(row.get("net_ir", 0.0)) <= 0.0:
            row_reasons.append("net_ir_not_positive")

        pm_pass = not row_reasons
        pm_passes.append(pm_pass)
        if pm_pass:
            states.append("promotion_candidate_pending_holdout_and_human_approval")
            reasons.append("")
        elif bool(row.get("passed_all", False)):
            states.append("research_survivor_below_sharpe3_goal")
            reasons.append("; ".join(row_reasons))
        else:
            states.append("killed_by_pm_gate")
            reasons.append("; ".join(row_reasons))

    ledger["pm_gate_pass"] = pm_passes
    ledger["promotion_state"] = states
    ledger["kill_reason"] = reasons
    return ledger


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).fillna("")
    columns = list(clipped.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]).replace("|", "\\|") for column in columns) + " |"
        for _, row in clipped.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _write_data_readiness(path: Path, readiness: dict[str, Any]) -> None:
    lines = [
        "# Data Readiness",
        "",
        f"Active countries: {', '.join(readiness['active_countries'])}",
        f"Universe tier: `{readiness['universe_tier']}`",
        f"Selected tickers: `{readiness['selected_tickers']}`",
        f"Price history: `{readiness['price_history_status']}`",
        f"Date range: `{readiness['date_start']}` to `{readiness['date_end']}`",
        "",
        "## Ready",
        "- Daily adjusted close price backtest loop.",
        "- Volume-derived liquidity proxy from the L2 extraction artifact.",
        "- Country-level transaction-cost schedule for Malaysia, Singapore, and Indonesia.",
        "- Source-backed price/technical strategy suite from the Markdown corpus.",
        "",
        "## Not Yet Ready",
        "- Full point-in-time fundamentals.",
        "- Analyst revisions, recommendation changes, and target-price changes.",
        "- Full historical constituent, delisting, and ticker-change security master.",
        "- Verified dividend/split corporate action truth beyond current adjusted price fields.",
        "- Intraday order book, borrow, and market-impact model.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pm_memo(path: Path, ledger: pd.DataFrame, config: ClosedLoopAlphaFactoryConfig, readiness: dict[str, Any]) -> None:
    top_cols = [
        "candidate_id",
        "model_name",
        "family",
        "allocator_name",
        "cost_bps",
        "net_ir",
        "net_sharpe",
        "deflated_sharpe",
        "max_drawdown",
        "turnover",
        "passed_all",
        "pm_gate_pass",
        "promotion_state",
        "kill_reason",
    ]
    top = ledger[[col for col in top_cols if col in ledger.columns]].head(20) if not ledger.empty else pd.DataFrame()
    promoted = int(ledger["pm_gate_pass"].sum()) if "pm_gate_pass" in ledger else 0
    best_sharpe = float(ledger["net_sharpe"].max()) if "net_sharpe" in ledger and not ledger.empty else 0.0
    best_ir = float(ledger["net_ir"].max()) if "net_ir" in ledger and not ledger.empty else 0.0

    if promoted > 0:
        decision = "At least one candidate cleared the Sharpe 3 PM gate, pending sealed holdout review and human approval."
    else:
        decision = "No candidate cleared the Sharpe 3 PM gate. Loop should mutate models, widen data, or improve portfolio construction."

    lines = [
        "# PM Decision Memo",
        "",
        f"Run ID: `{config.run_id or 'auto'}`",
        f"Universe: `{', '.join(config.include_countries)}`",
        f"Tier: `{config.universe_tier}`",
        f"Promotion Sharpe threshold: `{config.min_net_sharpe:.2f}`",
        f"Minimum deflated Sharpe probability: `{config.min_deflated_sharpe:.2f}`",
        f"Selected tickers: `{readiness['selected_tickers']}`",
        "",
        "## Decision",
        decision,
        "",
        "## Summary",
        f"- PM-promoted candidates: `{promoted}`",
        f"- Best observed net Sharpe: `{best_sharpe:.3f}`",
        f"- Best observed net IR: `{best_ir:.3f}`",
        f"- Candidate rows: `{len(ledger)}`",
        "",
        "## Top Candidate Ledger Rows",
        _markdown_table(top),
        "",
        "## Loop Instruction",
        "If the Sharpe 3 gate is not met, route failures to the narrowest owner: data gaps to Data Factory, low net edge to Model Factory, cost/turnover drag to Portfolio PM, and unstable results to Olympiad Math Validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_closed_loop_alpha_factory(config: ClosedLoopAlphaFactoryConfig) -> ClosedLoopAlphaFactoryResult:
    """Run the first PM-owned closed-loop alpha factory backtest pass."""

    run_id = config.run_id or _latest_run_id()
    artifact_dir = Path(config.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    l2_artifact_dir = Path(config.l2_artifact_dir)
    manifest = _read_json(l2_artifact_dir / "manifest.json")
    long_ohlcv_root = Path(manifest["long_ohlcv_root"])
    tickers = _selected_tickers(l2_artifact_dir, config.universe_tier, config.include_countries)
    filtered_ohlcv = _materialize_filtered_ohlcv(long_ohlcv_root, artifact_dir, config.universe_tier, tickers)

    models = build_source_strategy_model_suite(prefix="agentic")
    if config.max_models is not None:
        models = models[: config.max_models]
    allocators = closed_loop_allocator_configs(config.max_asset_weight)
    if config.max_allocators is not None:
        allocators = allocators[: config.max_allocators]

    agent_trace: list[dict[str, Any]] = [
        {
            "agent": "chief_quant_pm",
            "status": "started",
            "decision": "Sharpe 3 is the promotion line, not a curve-fit target.",
            "min_net_sharpe": config.min_net_sharpe,
        },
        {
            "agent": "markdown_librarian",
            "status": "completed",
            "source_strategy_specs": int(len(source_strategy_manifest())),
            "source_strategy_manifest": "source_strategy_manifest.csv",
        },
        {
            "agent": "data_steward",
            "status": "completed",
            "universe_tier": config.universe_tier,
            "selected_tickers": len(tickers) if tickers is not None else "all",
            "filtered_ohlcv": str(filtered_ohlcv),
        },
        {
            "agent": "model_factory",
            "status": "completed",
            "model_count": len(models),
            "allocator_count": len(allocators),
        },
    ]
    source_strategy_manifest().to_csv(artifact_dir / "source_strategy_manifest.csv", index=False)

    sweep = run_cross_market_sweep_from_long_ohlcv(
        long_ohlcv_root=filtered_ohlcv,
        artifact_root=artifact_dir / "backtests",
        run_id=f"{run_id}-{config.universe_tier.upper()}-SOURCE",
        convert_to_usd=False,
        min_rows=config.min_rows,
        max_missing_ratio=config.max_missing_ratio,
        train_size=config.train_size,
        test_size=config.test_size,
        embargo=config.embargo,
        cost_bps_grid=config.cost_bps_grid,
        baseline_cost_bps=config.cost_bps_grid[0],
        target_volatility=config.target_volatility,
        max_leverage=config.max_leverage,
        max_country_weight_limit=config.max_country_weight,
        max_asset_weight_limit=config.max_asset_weight,
        drawdown_limit=config.drawdown_limit,
        min_deflated_sharpe=config.min_deflated_sharpe,
        precompute_full_signals=config.precompute_full_signals,
        write_candidate_weights=config.write_candidate_weights,
        candidate_weight_top_n=config.candidate_weight_top_n,
        models=models,
        allocator_configs=allocators,
        include_countries=config.include_countries,
    )
    agent_trace.append(
        {
            "agent": "backtest_engine",
            "status": "completed",
            "candidate_count": int(len(sweep.all_candidates)),
            "feasible_candidates_before_pm_gate": int(sweep.all_candidates["passed_all"].sum()) if not sweep.all_candidates.empty else 0,
            "artifact_dir": str(sweep.artifact_dir),
        }
    )

    ledger = apply_pm_sharpe_gate(
        sweep.all_candidates,
        min_net_sharpe=config.min_net_sharpe,
        min_deflated_sharpe=config.min_deflated_sharpe,
    )
    ledger.to_csv(artifact_dir / "candidate_ledger.csv", index=False)
    promoted = ledger[ledger["pm_gate_pass"]].copy() if not ledger.empty else pd.DataFrame()
    promoted.to_csv(artifact_dir / "pm_promoted_candidates.csv", index=False)

    readiness = {
        "active_countries": list(config.include_countries),
        "universe_tier": config.universe_tier,
        "selected_tickers": len(tickers) if tickers is not None else int(manifest.get("raw_tickers", 0)),
        "price_history_status": "ready_for_price_volume_loop",
        "date_start": manifest.get("date_min", ""),
        "date_end": manifest.get("date_max", ""),
        "l2_artifact_dir": str(l2_artifact_dir.resolve()),
        "long_ohlcv_input": str(filtered_ohlcv.resolve()),
        "pit_fundamentals_status": "not_ready",
        "analyst_revisions_status": "not_ready",
        "delisting_security_master_status": "not_ready",
    }
    _write_data_readiness(artifact_dir / "data_readiness.md", readiness)
    _write_pm_memo(artifact_dir / "pm_decision.md", ledger, ClosedLoopAlphaFactoryConfig(**{**asdict(config), "run_id": run_id}), readiness)

    agent_trace.append(
        {
            "agent": "chief_quant_pm",
            "status": "completed",
            "pm_promoted_candidates": int(len(promoted)),
            "next_loop": "mutate_or_widen_data" if promoted.empty else "sealed_holdout_and_human_review",
        }
    )

    _write_json(artifact_dir / "run_config.json", _config_for_json(ClosedLoopAlphaFactoryConfig(**{**asdict(config), "run_id": run_id})))
    _write_json(artifact_dir / "agent_trace.json", agent_trace)

    return ClosedLoopAlphaFactoryResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        sweep_result=sweep,
        candidate_ledger=ledger,
        pm_promoted=promoted,
        data_readiness=readiness,
        agent_trace=agent_trace,
    )
