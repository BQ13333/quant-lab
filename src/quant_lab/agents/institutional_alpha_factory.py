from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from quant_lab.agents.closed_loop_alpha_factory import (
    ClosedLoopAlphaFactoryConfig,
    apply_pm_sharpe_gate,
    run_closed_loop_alpha_factory,
)
from quant_lab.research.overfit_audit import build_overfit_audit, write_overfit_audit
from quant_lab.research.pit_ablation import PITEnhancedAblationConfig, run_pit_enhanced_ablation


DEFAULT_DATA_QUALITY_DIR = (
    "artifacts/institutional_data_quality/"
    "DATA-QUALITY-LOOP-MYS-SGP-IDN-U2-20260708"
)
DEFAULT_L2_ARTIFACT_DIR = (
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/"
    "05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)
DEFAULT_PIT_QA_ARTIFACT_DIR = "artifacts/pit_leakage_qa/PIT-LEAKAGE-QA-MYS-SGP-IDN-FULL-U2-20260708"

UniverseTier = Literal["u2", "u3", "all"]


@dataclass(frozen=True)
class InstitutionalAlphaFactoryConfig:
    """Configuration for the Phase 2 alpha/backtest factory.

    This runner is deliberately a research orchestrator. It can discover strong
    candidates, but it cannot approve live trading by itself.
    """

    data_quality_dir: str | Path = DEFAULT_DATA_QUALITY_DIR
    l2_artifact_dir: str | Path | None = None
    pit_qa_artifact_dir: str | Path | None = None
    artifact_root: str | Path = "artifacts/institutional_alpha_factory"
    run_id: str | None = None
    include_countries: tuple[str, ...] = ("Malaysia", "Singapore", "Indonesia")
    universe_tier: UniverseTier = "u2"
    use_accepted_adjusted_returns: bool = True
    accepted_return_column: str = "adj_close_total_return_net"
    start_date: str = "2021-01-01"
    end_date: str = "2026-07-06"
    holdout_start: str = "2024-01-02"
    max_iterations: int = 1
    run_source_only_sweep: bool = True
    source_only_max_models: int | None = 12
    source_only_max_allocators: int | None = 4
    pit_max_source_models: int | None = None
    train_size: int = 504
    test_size: int = 63
    embargo: int = 21
    cost_bps_grid: tuple[float, ...] = (5.0, 15.0, 25.0)
    target_volatility: float = 0.12
    max_leverage: float = 1.0
    max_country_weight: float = 0.45
    max_asset_weight: float = 0.08
    drawdown_limit: float = -0.25
    min_net_sharpe: float = 3.0
    min_deflated_sharpe: float = 0.95
    min_holdout_net_sharpe: float = 1.0
    max_pbo: float = 0.30
    candidate_weight_top_n: int = 10


@dataclass(frozen=True)
class InstitutionalAlphaFactoryResult:
    run_id: str
    artifact_dir: Path
    combined_ledger: pd.DataFrame
    promoted: pd.DataFrame
    mutation_queue: pd.DataFrame
    data_quality_manifest: dict[str, Any]
    overfit_summary: dict[str, Any]
    agent_roster: list[dict[str, Any]]


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("INSTITUTIONAL-ALPHA-FACTORY-%Y%m%d-%H%M%S")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _config_for_json(config: InstitutionalAlphaFactoryConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["data_quality_dir"] = str(config.data_quality_dir)
    payload["l2_artifact_dir"] = str(config.l2_artifact_dir) if config.l2_artifact_dir is not None else None
    payload["pit_qa_artifact_dir"] = str(config.pit_qa_artifact_dir) if config.pit_qa_artifact_dir is not None else None
    payload["artifact_root"] = str(config.artifact_root)
    payload["include_countries"] = list(config.include_countries)
    payload["cost_bps_grid"] = list(config.cost_bps_grid)
    return payload


def load_accepted_data_quality_manifest(data_quality_dir: str | Path) -> dict[str, Any]:
    """Load and enforce the institutional 5/5 data-quality precondition."""

    root = Path(data_quality_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing data-quality manifest: {manifest_path}")

    manifest = _read_json(manifest_path)
    blockers = manifest.get("critical_blockers") or []
    accepted = bool(manifest.get("accepted"))
    score = float(manifest.get("overall_score", 0.0) or 0.0)
    max_score = float(manifest.get("max_score", 5.0) or 5.0)
    if not accepted or score < max_score or blockers:
        raise ValueError(
            "institutional alpha factory is blocked until data quality is accepted 5/5 "
            f"(accepted={accepted}, score={score:g}/{max_score:g}, blockers={len(blockers)})"
        )
    return manifest


def resolve_alpha_factory_paths(config: InstitutionalAlphaFactoryConfig, data_quality_manifest: dict[str, Any]) -> dict[str, Path]:
    paths = data_quality_manifest.get("paths", {})
    l2_path = Path(
        config.l2_artifact_dir
        or paths.get("l2_artifact_dir")
        or DEFAULT_L2_ARTIFACT_DIR
    )
    pit_qa_path = Path(
        config.pit_qa_artifact_dir
        or paths.get("pit_leakage_qa_dir")
        or DEFAULT_PIT_QA_ARTIFACT_DIR
    )
    adjusted_returns_path = Path(
        paths.get("adjusted_return_reconstruction_dir")
        or "artifacts/adjusted_return_reconstruction/ADJRET-RECON-MYS-SGP-IDN-U2-20260708"
    )
    return {
        "data_quality_dir": Path(config.data_quality_dir),
        "l2_artifact_dir": l2_path,
        "pit_qa_artifact_dir": pit_qa_path,
        "adjusted_return_reconstruction_dir": adjusted_returns_path,
    }


def institutional_agent_roster() -> list[dict[str, Any]]:
    """Specialist lane map for the Phase 2 alpha loop."""

    return [
        {
            "agent": "chief_quant_pm",
            "mandate": "Own promotion decisions; Sharpe 3 is a hard gate, not a curve-fit target.",
        },
        {
            "agent": "data_steward",
            "mandate": "Admit only 5/5 data-quality artifacts and PIT/leakage-passed features.",
        },
        {
            "agent": "strategy_librarian",
            "mandate": "Map markdown/book strategies to implemented, source-backed model specs.",
        },
        {
            "agent": "model_factory",
            "mandate": "Generate source, PIT, and ensemble candidates without duplicating highly correlated trials.",
        },
        {
            "agent": "ml_ai_researcher",
            "mandate": "Introduce ML/meta-model ideas only through nested validation and sealed holdout controls.",
        },
        {
            "agent": "olympiad_math_validator",
            "mandate": "Attack false discoveries with deflated Sharpe, PBO, CPCV, decay, and concentration tests.",
        },
        {
            "agent": "portfolio_pm",
            "mandate": "Translate scores into tradable weights with country, asset, turnover, volatility, and cost limits.",
        },
        {
            "agent": "execution_tca_steward",
            "mandate": "Keep transaction costs, turnover drag, FX, and execution-readiness in the loop.",
        },
        {
            "agent": "risk_manager",
            "mandate": "Reject candidates with hidden drawdown, factor, country, liquidity, or concentration risk.",
        },
        {
            "agent": "research_scribe",
            "mandate": "Write immutable ledgers, run configs, mutation queues, and PM-readable summaries.",
        },
    ]


def prefix_candidate_namespace(
    ledger: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    lane: str,
    iteration: int,
    artifact_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ledger.empty:
        return ledger.copy(), returns_matrix.copy()

    prefix = f"{lane}_iter{iteration:02d}"
    out = ledger.copy()
    out["raw_candidate_id"] = out["candidate_id"].astype(str)
    out["candidate_id"] = prefix + "::" + out["raw_candidate_id"]
    out["alpha_lane"] = lane
    out["iteration"] = iteration
    out["source_artifact_dir"] = str(artifact_dir.resolve())

    returns = returns_matrix.copy()
    if not returns.empty:
        returns.columns = [prefix + "::" + str(col) for col in returns.columns]
    return out, returns


def _read_candidate_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    returns = pd.read_parquet(path)
    returns.index = pd.to_datetime(returns.index, errors="coerce")
    return returns.sort_index()


def materialize_accepted_return_l2_proxy(
    l2_artifact_dir: str | Path,
    adjusted_return_reconstruction_dir: str | Path,
    output_dir: str | Path,
    *,
    accepted_return_column: str = "adj_close_total_return_net",
) -> Path:
    """Create an L2-compatible artifact whose `adj_close` is the accepted return authority.

    The existing backtest runners consume long OHLCV data. This adapter keeps
    their API stable while replacing the price field with the production
    adjusted total-return series from the accepted corporate-action
    reconstruction artifact.
    """

    l2_root = Path(l2_artifact_dir)
    adjusted_root = Path(adjusted_return_reconstruction_dir)
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    l2_manifest = _read_json(l2_root / "manifest.json")
    adjusted_manifest = _read_json(adjusted_root / "manifest.json")
    if not bool(adjusted_manifest.get("production_adjusted_return_ready")):
        raise ValueError("adjusted-return artifact is not production ready")

    daily_path = Path(
        adjusted_manifest.get("artifacts", {}).get(
            "daily_adjusted_returns",
            adjusted_root / "daily_adjusted_returns.parquet",
        )
    )
    if not daily_path.exists():
        raise FileNotFoundError(f"missing accepted daily adjusted returns: {daily_path}")

    daily = pd.read_parquet(daily_path)
    required = {"date", "ticker", accepted_return_column}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"accepted daily returns missing columns: {sorted(missing)}")

    base = pd.read_parquet(l2_manifest["long_ohlcv_root"])
    metadata_cols = ["ticker", "sid", "local_symbol", "name", "country", "exchange", "currency"]
    available_metadata = [col for col in metadata_cols if col in base.columns]
    metadata = (
        base[available_metadata]
        .dropna(subset=["ticker"])
        .drop_duplicates("ticker", keep="last")
        .reset_index(drop=True)
    )
    for col in metadata_cols:
        if col not in metadata.columns:
            metadata[col] = pd.NA

    adjusted = daily[["date", "ticker", "close", accepted_return_column]].copy()
    adjusted["date"] = pd.to_datetime(adjusted["date"], errors="coerce")
    adjusted["ticker"] = adjusted["ticker"].astype(str)
    adjusted["adj_close"] = pd.to_numeric(adjusted[accepted_return_column], errors="coerce")
    adjusted = adjusted.merge(metadata[metadata_cols], on="ticker", how="left")
    adjusted["price_authority"] = "accepted_adjusted_return_reconstruction"
    adjusted["price_authority_column"] = accepted_return_column
    adjusted_path = output_root / "accepted_adjusted_long_ohlcv.parquet"
    adjusted.to_parquet(adjusted_path, index=False)

    tiers_src = l2_root / "eligible_universe_tiers.csv"
    tiers_dst = output_root / "eligible_universe_tiers.csv"
    if tiers_src.exists():
        tiers = pd.read_csv(tiers_src, low_memory=False)
        tiers.to_csv(tiers_dst, index=False)

    proxy_manifest = dict(l2_manifest)
    proxy_manifest.update(
        {
            "long_ohlcv_root": str(adjusted_path.resolve()),
            "source_l2_artifact_dir": str(l2_root.resolve()),
            "source_adjusted_return_reconstruction_dir": str(adjusted_root.resolve()),
            "price_authority": "accepted_adjusted_return_reconstruction",
            "price_col": "adj_close",
            "accepted_return_column": accepted_return_column,
            "date_min": str(adjusted["date"].min()),
            "date_max": str(adjusted["date"].max()),
            "raw_tickers": int(adjusted["ticker"].nunique()),
        }
    )
    _write_json(output_root / "manifest.json", proxy_manifest)
    return output_root


def _load_overfit_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return _read_json(path)


def apply_institutional_promotion_gate(
    candidates: pd.DataFrame,
    overfit_summary: dict[str, Any],
    min_net_sharpe: float = 3.0,
    min_deflated_sharpe: float = 0.95,
    min_holdout_net_sharpe: float = 1.0,
    max_pbo: float = 0.30,
) -> pd.DataFrame:
    """Apply PM, holdout, and overfit gates to the combined candidate ledger."""

    gated = apply_pm_sharpe_gate(
        candidates,
        min_net_sharpe=min_net_sharpe,
        min_deflated_sharpe=min_deflated_sharpe,
    )
    if gated.empty:
        for col in (
            "holdout_gate_pass",
            "overfit_audit_pass",
            "institutional_gate_pass",
            "production_state",
        ):
            gated[col] = []
        return gated

    if "holdout_net_sharpe" in gated.columns:
        gated["holdout_gate_pass"] = gated["holdout_net_sharpe"].fillna(float("-inf")).ge(min_holdout_net_sharpe)
    else:
        gated["holdout_net_sharpe"] = pd.NA
        gated["holdout_gate_pass"] = False

    warnings = set(overfit_summary.get("warnings") or [])
    pbo = float(overfit_summary.get("pbo_cscv", overfit_summary.get("pbo_even_odd_proxy", 1.0)) or 0.0)
    hard_overfit_warnings = {
        "high_cscv_pbo",
        "high_even_odd_pbo_proxy",
        "high_candidate_return_correlation",
    }
    overfit_pass = pbo <= max_pbo and not bool(warnings & hard_overfit_warnings)
    gated["overfit_audit_pass"] = overfit_pass
    gated["overfit_pbo_even_odd_proxy"] = pbo
    gated["overfit_pbo_cscv"] = pbo
    gated["overfit_effective_independent_trials"] = overfit_summary.get("effective_independent_trials", 0.0)

    gated["institutional_gate_pass"] = (
        gated["pm_gate_pass"].fillna(False).astype(bool)
        & gated["holdout_gate_pass"].fillna(False).astype(bool)
        & gated["overfit_audit_pass"].fillna(False).astype(bool)
    )
    gated["production_state"] = "research_only"
    gated.loc[gated["institutional_gate_pass"], "production_state"] = (
        "paper_trade_candidate_pending_human_approval"
    )

    failed_holdout = gated["pm_gate_pass"].fillna(False).astype(bool) & ~gated["holdout_gate_pass"].fillna(False).astype(bool)
    gated.loc[failed_holdout, "promotion_state"] = "killed_by_holdout_gate"
    gated.loc[failed_holdout, "kill_reason"] = (
        gated.loc[failed_holdout, "kill_reason"].fillna("").astype(str)
        + "; holdout_net_sharpe_below_"
        + f"{min_holdout_net_sharpe:g}"
    ).str.strip("; ")

    failed_overfit = (
        gated["pm_gate_pass"].fillna(False).astype(bool)
        & gated["holdout_gate_pass"].fillna(False).astype(bool)
        & ~gated["overfit_audit_pass"].fillna(False).astype(bool)
    )
    gated.loc[failed_overfit, "promotion_state"] = "killed_by_overfit_audit"
    gated.loc[failed_overfit, "kill_reason"] = (
        gated.loc[failed_overfit, "kill_reason"].fillna("").astype(str)
        + "; overfit_audit_failed"
    ).str.strip("; ")
    return gated.sort_values(
        ["institutional_gate_pass", "pm_gate_pass", "score", "net_sharpe", "holdout_net_sharpe"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def build_mutation_queue(ledger: pd.DataFrame, overfit_summary: dict[str, Any]) -> pd.DataFrame:
    """Build the next agent queue from observed failure modes."""

    rows: list[dict[str, Any]] = []
    if ledger.empty:
        rows.append(
            {
                "priority": 1,
                "owner_agent": "model_factory",
                "workstream": "candidate_generation",
                "trigger": "empty_candidate_ledger",
                "instruction": "No candidates were produced. Inspect data filters, model registry, and optimizer exceptions.",
            }
        )
        return pd.DataFrame(rows)

    best = ledger.iloc[0]
    promoted = int(ledger.get("institutional_gate_pass", pd.Series(False, index=ledger.index)).sum())
    if promoted == 0:
        rows.append(
            {
                "priority": 1,
                "owner_agent": "chief_quant_pm",
                "workstream": "promotion_gate",
                "trigger": "no_institutional_promotion_candidate",
                "instruction": (
                    "Do not promote. Treat best candidate as research evidence only; continue bounded mutations "
                    "with immutable trial ledger."
                ),
            }
        )

    if float(best.get("net_sharpe", 0.0) or 0.0) < 3.0:
        rows.append(
            {
                "priority": 2,
                "owner_agent": "model_factory",
                "workstream": "alpha_discovery",
                "trigger": "best_net_sharpe_below_3",
                "instruction": (
                    "Search for genuinely orthogonal source/PIT/ML ensemble families. Avoid tiny parameter variants "
                    "unless nested validation proves independent edge."
                ),
            }
        )

    if "constraint_deflated_sharpe_passed" in ledger.columns and float(ledger["constraint_deflated_sharpe_passed"].mean()) < 0.05:
        rows.append(
            {
                "priority": 2,
                "owner_agent": "olympiad_math_validator",
                "workstream": "false_discovery_control",
                "trigger": "deflated_sharpe_pass_rate_below_5pct",
                "instruction": (
                    "Cluster near-duplicate candidates, reduce effective trial inflation, and rerun DSR/PBO/CPCV "
                    "on family-level representatives."
                ),
            }
        )

    if "constraint_max_country_weight_passed" in ledger.columns and float(ledger["constraint_max_country_weight_passed"].mean()) < 0.50:
        rows.append(
            {
                "priority": 1,
                "owner_agent": "portfolio_pm",
                "workstream": "allocator_constraints",
                "trigger": "country_weight_constraint_is_dominant_failure",
                "instruction": (
                    "Repair country-cap handling after volatility targeting or rerun with leverage-consistent "
                    "country limits. This is a portfolio-construction issue before it is an alpha issue."
                ),
            }
        )

    if float(overfit_summary.get("effective_independent_trials", 0.0) or 0.0) < 3.0:
        rows.append(
            {
                "priority": 2,
                "owner_agent": "ml_ai_researcher",
                "workstream": "model_diversity",
                "trigger": "low_effective_independent_trials",
                "instruction": (
                    "Add independent model classes only if they use distinct economic information, then gate them "
                    "through nested walk-forward validation."
                ),
            }
        )

    if "holdout_net_sharpe" in ledger.columns and float(best.get("holdout_net_sharpe", 0.0) or 0.0) >= 3.0:
        rows.append(
            {
                "priority": 3,
                "owner_agent": "risk_manager",
                "workstream": "holdout_forensics",
                "trigger": "strong_holdout_but_no_promotion",
                "instruction": (
                    "Investigate whether holdout strength is concentrated by country, month, sector, or single name "
                    "before any extra optimization is allowed."
                ),
            }
        )

    return pd.DataFrame(rows).sort_values(["priority", "owner_agent"]).reset_index(drop=True)


def _markdown_table(frame: pd.DataFrame, max_rows: int = 30) -> str:
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


def _write_summary(
    path: Path,
    config: InstitutionalAlphaFactoryConfig,
    ledger: pd.DataFrame,
    promoted: pd.DataFrame,
    mutation_queue: pd.DataFrame,
    data_quality_manifest: dict[str, Any],
    overfit_summary: dict[str, Any],
    agent_roster: list[dict[str, Any]],
) -> None:
    top_ranked = ledger.iloc[0].to_dict() if not ledger.empty else {}
    max_sharpe = {}
    max_holdout = {}
    if not ledger.empty and "net_sharpe" in ledger:
        sharpe_values = pd.to_numeric(ledger["net_sharpe"], errors="coerce")
        if sharpe_values.notna().any():
            max_sharpe = ledger.loc[sharpe_values.idxmax()].to_dict()
    if not ledger.empty and "holdout_net_sharpe" in ledger:
        holdout_values = pd.to_numeric(ledger["holdout_net_sharpe"], errors="coerce")
        if holdout_values.notna().any():
            max_holdout = ledger.loc[holdout_values.idxmax()].to_dict()
    decision = (
        "At least one candidate passed institutional research gates. Human review and paper-trading approval are still required."
        if not promoted.empty
        else "No candidate passed the institutional promotion gate. Continue the mutation loop; do not trade."
    )
    display_cols = [
        "candidate_id",
        "alpha_lane",
        "model_name",
        "meta_feature_set",
        "allocator_name",
        "cost_bps",
        "net_ir",
        "net_sharpe",
        "deflated_sharpe",
        "holdout_net_sharpe",
        "passed_all",
        "pm_gate_pass",
        "institutional_gate_pass",
        "promotion_state",
        "kill_reason",
    ]
    top = ledger[[col for col in display_cols if col in ledger.columns]].head(20) if not ledger.empty else pd.DataFrame()
    lines = [
        "# Institutional Alpha Factory",
        "",
        f"Run ID: `{config.run_id}`",
        f"Universe: `{config.universe_tier}` Malaysia/Singapore/Indonesia",
        f"Window: `{config.start_date}` to `{config.end_date}`",
        f"Frozen holdout start: `{config.holdout_start}`",
        f"Data-quality gate: `{data_quality_manifest.get('status')}` at `{data_quality_manifest.get('overall_score')}` / `{data_quality_manifest.get('max_score')}`",
        "",
        "## Verdict",
        "",
        decision,
        "",
        f"- Candidates tested: `{len(ledger):,}`",
        f"- Institutional promotion candidates: `{len(promoted):,}`",
        f"- Top-ranked candidate by gate score: `{top_ranked.get('candidate_id', '')}`",
        f"- Top-ranked net Sharpe: `{float(top_ranked.get('net_sharpe', 0.0) or 0.0):.3f}`",
        f"- Top-ranked deflated Sharpe: `{float(top_ranked.get('deflated_sharpe', 0.0) or 0.0):.3f}`",
        f"- Max net Sharpe candidate: `{max_sharpe.get('candidate_id', '')}`",
        f"- Max net Sharpe: `{float(max_sharpe.get('net_sharpe', 0.0) or 0.0):.3f}`",
        f"- Max holdout net Sharpe candidate: `{max_holdout.get('candidate_id', '')}`",
        f"- Max holdout net Sharpe: `{float(max_holdout.get('holdout_net_sharpe', 0.0) or 0.0):.3f}`",
        f"- PBO CSCV: `{float(overfit_summary.get('pbo_cscv', overfit_summary.get('pbo_even_odd_proxy', 0.0)) or 0.0):.3f}`",
        f"- Effective independent trials: `{float(overfit_summary.get('effective_independent_trials', 0.0) or 0.0):.3f}`",
        "",
        "## Agent Roster",
        "",
        _markdown_table(pd.DataFrame(agent_roster), max_rows=20),
        "",
        "## Top Candidate Ledger",
        "",
        _markdown_table(top, max_rows=20),
        "",
        "## Next Mutation Queue",
        "",
        _markdown_table(mutation_queue, max_rows=20),
        "",
        "## Operating Rule",
        "",
        "This is not a live-trading approval. A Sharpe 3 result only matters if it survives data-quality, deflated-Sharpe, holdout, overfit, risk, cost, and human review gates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_institutional_alpha_factory(config: InstitutionalAlphaFactoryConfig) -> InstitutionalAlphaFactoryResult:
    """Run the Phase 2 institutional alpha factory loop."""

    run_id = config.run_id or _latest_run_id()
    resolved_config = InstitutionalAlphaFactoryConfig(**{**asdict(config), "run_id": run_id})
    artifact_dir = Path(config.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dq_manifest = load_accepted_data_quality_manifest(config.data_quality_dir)
    paths = resolve_alpha_factory_paths(config, dq_manifest)
    l2_for_backtests = paths["l2_artifact_dir"]
    if config.use_accepted_adjusted_returns:
        l2_for_backtests = materialize_accepted_return_l2_proxy(
            paths["l2_artifact_dir"],
            paths["adjusted_return_reconstruction_dir"],
            artifact_dir / "accepted_return_l2_proxy",
            accepted_return_column=config.accepted_return_column,
        )
    roster = institutional_agent_roster()

    ledgers: list[pd.DataFrame] = []
    returns_matrices: list[pd.DataFrame] = []
    iteration_trace: list[dict[str, Any]] = []

    for iteration in range(1, max(1, int(config.max_iterations)) + 1):
        pit_result = run_pit_enhanced_ablation(
            PITEnhancedAblationConfig(
                l2_artifact_dir=l2_for_backtests,
                pit_qa_artifact_dir=paths["pit_qa_artifact_dir"],
                artifact_root=artifact_dir / "pit_enhanced_alpha",
                run_id=f"{run_id}-ITER{iteration:02d}-PIT",
                include_countries=config.include_countries,
                universe_tier=config.universe_tier,
                start_date=config.start_date,
                end_date=config.end_date,
                holdout_start=config.holdout_start,
                train_size=config.train_size,
                test_size=config.test_size,
                embargo=config.embargo,
                cost_bps_grid=config.cost_bps_grid,
                target_volatility=config.target_volatility,
                max_leverage=config.max_leverage,
                max_country_weight=config.max_country_weight,
                max_asset_weight=config.max_asset_weight,
                drawdown_limit=config.drawdown_limit,
                min_deflated_sharpe=config.min_deflated_sharpe,
                candidate_weight_top_n=config.candidate_weight_top_n,
                max_source_models=config.pit_max_source_models,
            )
        )
        pit_returns = _read_candidate_returns(pit_result.artifact_dir / "candidate_returns.parquet")
        pit_ledger, pit_returns = prefix_candidate_namespace(
            pit_result.candidate_ledger,
            pit_returns,
            lane="pit_source_ensemble",
            iteration=iteration,
            artifact_dir=pit_result.artifact_dir,
        )
        ledgers.append(pit_ledger)
        if not pit_returns.empty:
            returns_matrices.append(pit_returns)
        iteration_trace.append(
            {
                "iteration": iteration,
                "lane": "pit_source_ensemble",
                "artifact_dir": str(pit_result.artifact_dir.resolve()),
                "candidates": int(len(pit_ledger)),
            }
        )

        if config.run_source_only_sweep:
            source_result = run_closed_loop_alpha_factory(
                ClosedLoopAlphaFactoryConfig(
                    l2_artifact_dir=l2_for_backtests,
                    artifact_root=artifact_dir / "source_only_alpha",
                    run_id=f"{run_id}-ITER{iteration:02d}-SOURCE",
                    include_countries=config.include_countries,
                    universe_tier=config.universe_tier,
                    min_net_sharpe=config.min_net_sharpe,
                    min_deflated_sharpe=config.min_deflated_sharpe,
                    train_size=config.train_size,
                    test_size=config.test_size,
                    embargo=config.embargo,
                    target_volatility=config.target_volatility,
                    max_leverage=config.max_leverage,
                    max_country_weight=config.max_country_weight,
                    max_asset_weight=config.max_asset_weight,
                    drawdown_limit=config.drawdown_limit,
                    cost_bps_grid=config.cost_bps_grid,
                    max_models=config.source_only_max_models,
                    max_allocators=config.source_only_max_allocators,
                    precompute_full_signals=True,
                    write_candidate_weights=False,
                    candidate_weight_top_n=config.candidate_weight_top_n,
                )
            )
            source_returns = _read_candidate_returns(
                source_result.sweep_result.artifact_dir / "candidate_returns.parquet"
            )
            source_ledger, source_returns = prefix_candidate_namespace(
                source_result.candidate_ledger,
                source_returns,
                lane="source_only",
                iteration=iteration,
                artifact_dir=source_result.artifact_dir,
            )
            ledgers.append(source_ledger)
            if not source_returns.empty:
                returns_matrices.append(source_returns)
            iteration_trace.append(
                {
                    "iteration": iteration,
                    "lane": "source_only",
                    "artifact_dir": str(source_result.artifact_dir.resolve()),
                    "candidates": int(len(source_ledger)),
                }
            )

    combined = pd.concat(ledgers, ignore_index=True, sort=False) if ledgers else pd.DataFrame()
    combined_returns = pd.concat(returns_matrices, axis=1).sort_index() if returns_matrices else pd.DataFrame()
    if not combined_returns.empty:
        combined_returns.to_parquet(artifact_dir / "combined_candidate_returns.parquet")
        audit = write_overfit_audit(combined, combined_returns, artifact_dir)
        overfit_summary = audit.summary
    else:
        audit = build_overfit_audit(combined, combined_returns)
        overfit_summary = audit.summary
        _write_json(artifact_dir / "overfit_audit.json", overfit_summary)

    gated = apply_institutional_promotion_gate(
        combined,
        overfit_summary=overfit_summary,
        min_net_sharpe=config.min_net_sharpe,
        min_deflated_sharpe=config.min_deflated_sharpe,
        min_holdout_net_sharpe=config.min_holdout_net_sharpe,
        max_pbo=config.max_pbo,
    )
    promoted = gated[gated["institutional_gate_pass"]].copy() if not gated.empty else pd.DataFrame()
    mutation_queue = build_mutation_queue(gated, overfit_summary)

    gated.to_csv(artifact_dir / "combined_candidate_ledger.csv", index=False)
    promoted.to_csv(artifact_dir / "institutional_promoted_candidates.csv", index=False)
    mutation_queue.to_csv(artifact_dir / "agent_mutation_queue.csv", index=False)
    _write_json(artifact_dir / "data_quality_manifest_snapshot.json", dq_manifest)
    _write_json(artifact_dir / "agent_roster.json", roster)
    _write_json(artifact_dir / "iteration_trace.json", iteration_trace)
    _write_json(artifact_dir / "run_config.json", _config_for_json(resolved_config))
    _write_summary(
        artifact_dir / "summary.md",
        resolved_config,
        gated,
        promoted,
        mutation_queue,
        dq_manifest,
        overfit_summary,
        roster,
    )

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "data_quality_status": dq_manifest.get("status"),
        "data_quality_score": dq_manifest.get("overall_score"),
        "candidate_count": int(len(gated)),
        "promoted_count": int(len(promoted)),
        "mutation_queue_rows": int(len(mutation_queue)),
        "best_net_sharpe": float(gated["net_sharpe"].max()) if "net_sharpe" in gated and not gated.empty else 0.0,
        "best_deflated_sharpe": float(gated["deflated_sharpe"].max()) if "deflated_sharpe" in gated and not gated.empty else 0.0,
        "best_holdout_net_sharpe": float(gated["holdout_net_sharpe"].max()) if "holdout_net_sharpe" in gated and not gated.empty else 0.0,
        "accepted_for_trading": False,
        "trading_blocker": "requires human review, paper trading, execution/TCA validation, and broker controls",
        "artifacts": {
            "summary": str((artifact_dir / "summary.md").resolve()),
            "combined_candidate_ledger": str((artifact_dir / "combined_candidate_ledger.csv").resolve()),
            "institutional_promoted_candidates": str((artifact_dir / "institutional_promoted_candidates.csv").resolve()),
            "agent_mutation_queue": str((artifact_dir / "agent_mutation_queue.csv").resolve()),
            "overfit_audit": str((artifact_dir / "overfit_audit.json").resolve()),
        },
    }
    _write_json(artifact_dir / "manifest.json", manifest)

    return InstitutionalAlphaFactoryResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        combined_ledger=gated,
        promoted=promoted,
        mutation_queue=mutation_queue,
        data_quality_manifest=dq_manifest,
        overfit_summary=overfit_summary,
        agent_roster=roster,
    )
