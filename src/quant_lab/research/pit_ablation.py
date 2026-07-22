from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.pit_feature_selectors import (
    build_default_pit_factor_matrices,
    load_pit_feature_bundle,
)
from quant_lab.models.ensembles import WeightedEnsembleModel
from quant_lab.models.pit_factors import build_pit_factor_model_suite
from quant_lab.models.source_strategies import build_source_strategy_model_suite
from quant_lab.optimization.net_ir_optimizer import NetIRConstrainedOptimizer
from quant_lab.research.cross_market import _equal_weight_returns, load_long_ohlcv_dataset
from quant_lab.research.overfit_audit import write_overfit_audit
from quant_lab.validation.metrics import max_drawdown, sharpe_ratio


DEFAULT_SOURCE_MODEL_SUFFIXES = (
    "src_mom_126",
    "src_lowvol_126",
    "src_vam_126_vol63",
    "src_resid_mom_126_beta126",
    "src_alpha_combo_mom_lowvol_resid",
)


@dataclass(frozen=True)
class PITEnhancedAblationConfig:
    l2_artifact_dir: str | Path
    pit_qa_artifact_dir: str | Path
    artifact_root: str | Path = "artifacts/pit_enhanced_alpha"
    run_id: str | None = None
    include_countries: tuple[str, ...] = ("Malaysia", "Singapore", "Indonesia")
    universe_tier: str = "u2"
    start_date: str = "2021-01-01"
    end_date: str = "2026-07-06"
    holdout_start: str = "2024-01-02"
    price_col: str = "adj_close"
    min_rows: int = 2500
    max_missing_ratio: float = 0.25
    train_size: int = 504
    test_size: int = 63
    embargo: int = 21
    cost_bps_grid: tuple[float, ...] = (5.0, 15.0, 25.0)
    target_volatility: float = 0.12
    max_leverage: float = 2.0
    max_country_weight: float = 0.45
    max_asset_weight: float = 0.08
    drawdown_limit: float = -0.25
    min_deflated_sharpe: float = 0.95
    candidate_weight_top_n: int = 10
    max_source_models: int | None = None


@dataclass(frozen=True)
class PITEnhancedAblationResult:
    run_id: str
    artifact_dir: Path
    candidate_ledger: pd.DataFrame
    family_summary: pd.DataFrame
    factor_quality: pd.DataFrame
    model_manifest: pd.DataFrame


def run_pit_enhanced_ablation(config: PITEnhancedAblationConfig) -> PITEnhancedAblationResult:
    run_id = config.run_id or datetime.now(timezone.utc).strftime("PIT-ENHANCED-ABLATION-%Y%m%d-%H%M%S")
    artifact_dir = Path(config.artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    l2_dir = Path(config.l2_artifact_dir)
    l2_manifest = _read_json(l2_dir / "manifest.json")
    selected_tickers = _selected_tickers(l2_dir, config.universe_tier, config.include_countries)
    filtered_ohlcv = _materialize_filtered_ohlcv(
        Path(str(l2_manifest["long_ohlcv_root"])),
        artifact_dir / f"filtered_ohlcv_{config.universe_tier}.parquet",
        selected_tickers,
    )
    dataset = load_long_ohlcv_dataset(
        filtered_ohlcv,
        price_col=config.price_col,
        min_rows=config.min_rows,
        max_missing_ratio=config.max_missing_ratio,
        include_countries=config.include_countries,
    )
    full_prices = dataset.prices.loc[pd.Timestamp(config.start_date) : pd.Timestamp(config.end_date)].copy()
    full_prices = full_prices.dropna(how="all")
    research_end = pd.Timestamp(config.holdout_start)
    prices = full_prices.loc[full_prices.index < research_end].copy()
    if prices.empty:
        raise ValueError("no prices in requested PIT ablation date window")
    metadata = dataset.metadata[dataset.metadata["ticker"].isin(prices.columns)].reset_index(drop=True)
    group_map = metadata.set_index("ticker")["country"].reindex(prices.columns)
    prices.to_parquet(artifact_dir / "eligible_prices_local.parquet")
    metadata.to_csv(artifact_dir / "metadata.csv", index=False)

    pit_bundle = load_pit_feature_bundle(config.pit_qa_artifact_dir, require_pass=True)
    matrix_bundle = build_default_pit_factor_matrices(pit_bundle.panel, prices)
    matrix_bundle.feature_quality.to_csv(artifact_dir / "pit_factor_quality.csv", index=False)
    (artifact_dir / "pit_factor_manifest.json").write_text(
        json.dumps(matrix_bundle.source_manifest, indent=2, default=str),
        encoding="utf-8",
    )

    source_models = _select_source_models(
        build_source_strategy_model_suite(prefix="pitabl"),
        max_source_models=config.max_source_models,
    )
    pit_models = build_pit_factor_model_suite(
        matrix_bundle.matrices,
        matrix_bundle.family_columns,
        prefix="pitabl_pit",
        source_artifact=str(Path(config.pit_qa_artifact_dir).resolve()),
    )
    models = [*source_models, *pit_models, *_build_source_pit_ensembles(source_models, pit_models)]
    model_manifest = _model_manifest(models)
    model_manifest.to_csv(artifact_dir / "model_manifest.csv", index=False)

    optimizer = NetIRConstrainedOptimizer(
        drawdown_limit=config.drawdown_limit,
        max_turnover=3.0,
        max_country_weight_limit=config.max_country_weight,
        max_asset_weight_limit=config.max_asset_weight,
        min_deflated_sharpe=config.min_deflated_sharpe,
        target_volatility=config.target_volatility,
        train_size=config.train_size,
        test_size=config.test_size,
        embargo=config.embargo,
        max_leverage=config.max_leverage,
        precompute_full_signals=True,
    )
    allocator_configs = [
        {
            "weighting": "group_neutral_long_only_capped",
            "top_fraction": 0.20,
            "max_weight": config.max_asset_weight,
            "rebalance_frequency": "W",
        },
        {
            "weighting": "group_neutral_long_only_capped",
            "top_fraction": 0.20,
            "max_weight": config.max_asset_weight,
            "rebalance_frequency": "M",
        },
        {
            "weighting": "long_only_capped",
            "top_fraction": 0.15,
            "max_weight": config.max_asset_weight,
            "rebalance_frequency": "W",
        },
    ]
    candidate_results = optimizer.evaluate_all_candidates(
        models=models,
        allocator_configs=allocator_configs,
        cost_bps_grid=config.cost_bps_grid,
        prices=prices,
        group_map=group_map,
        benchmark_returns=_equal_weight_returns(prices),
    )
    ledger = optimizer.results_to_frame(candidate_results)
    returns_matrix = (
        pd.concat({result.candidate_id: result.net_returns for result in candidate_results}, axis=1)
        if candidate_results
        else pd.DataFrame()
    )
    if not returns_matrix.empty:
        returns_matrix.to_parquet(artifact_dir / "candidate_returns.parquet")
        write_overfit_audit(ledger, returns_matrix, artifact_dir)
    ledger = _add_research_audit_columns(
        ledger,
        returns_matrix,
        contaminated_start=config.holdout_start,
        contaminated_through=config.end_date,
    )
    ledger = _add_pit_gate_columns(ledger)
    ledger.to_csv(artifact_dir / "candidate_ledger.csv", index=False)

    _write_candidate_weights(artifact_dir, candidate_results, ledger, top_n=config.candidate_weight_top_n)
    family_summary = _family_summary(ledger)
    family_summary.to_csv(artifact_dir / "family_ablation_summary.csv", index=False)
    constraint_summary = _constraint_pass_summary(ledger)
    constraint_summary.to_csv(artifact_dir / "constraint_pass_summary.csv", index=False)
    resolved_config = PITEnhancedAblationConfig(**{**asdict(config), "run_id": run_id})
    _write_summary(
        artifact_dir / "summary.md",
        ledger,
        family_summary,
        matrix_bundle.feature_quality,
        constraint_summary,
        resolved_config,
    )
    _write_json(artifact_dir / "run_config.json", _config_for_json(resolved_config))

    return PITEnhancedAblationResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        candidate_ledger=ledger,
        family_summary=family_summary,
        factor_quality=matrix_bundle.feature_quality,
        model_manifest=model_manifest,
    )


def _selected_tickers(l2_artifact_dir: Path, universe_tier: str, include_countries: tuple[str, ...]) -> list[str]:
    tiers = pd.read_csv(l2_artifact_dir / "eligible_universe_tiers.csv", low_memory=False)
    country_set = {country.strip().lower() for country in include_countries}
    selected = tiers[tiers["country"].astype(str).str.strip().str.lower().isin(country_set)].copy()
    tier = universe_tier.strip().lower()
    if tier != "all":
        tier_col = {"u2": "u2_backtest_eligible", "u3": "u3_liquid_proxy"}.get(tier, tier)
        selected = selected[selected[tier_col].fillna(False).astype(bool)].copy()
    return sorted(selected["ticker"].dropna().astype(str).unique())


def _materialize_filtered_ohlcv(long_ohlcv_root: Path, output_path: Path, tickers: list[str]) -> Path:
    ohlcv = pd.read_parquet(long_ohlcv_root)
    filtered = ohlcv[ohlcv["ticker"].astype(str).isin(set(tickers))].copy()
    if filtered.empty:
        raise ValueError("no OHLCV rows after PIT ablation ticker filter")
    filtered.to_parquet(output_path, index=False)
    return output_path


def _select_source_models(models: list, max_source_models: int | None) -> list:
    selected = [
        model
        for model in models
        if any(model.name.endswith(suffix) for suffix in DEFAULT_SOURCE_MODEL_SUFFIXES)
    ]
    if max_source_models is not None:
        selected = selected[:max_source_models]
    return selected


def _build_source_pit_ensembles(source_models: list, pit_models: list) -> list[WeightedEnsembleModel]:
    pit_by_feature = {model.feature_set: model for model in pit_models if hasattr(model, "feature_set")}
    pit_all = pit_by_feature.get("all_equal")
    pit_quality = pit_by_feature.get("quality_revision") or pit_all
    if pit_all is None:
        return []
    ensembles: list[WeightedEnsembleModel] = []
    for source in source_models:
        if source.name.endswith("src_resid_mom_126_beta126") or source.name.endswith("src_alpha_combo_mom_lowvol_resid"):
            for pit_model, suffix, weights in (
                (pit_all, "pit_all_70_30", [0.70, 0.30]),
                (pit_quality, "pit_quality_70_30", [0.70, 0.30]),
                (pit_all, "pit_all_50_50", [0.50, 0.50]),
            ):
                ensembles.append(
                    WeightedEnsembleModel(
                        name=f"{source.name}__{suffix}",
                        models=[source, pit_model],
                        weights=weights,
                    )
                )
    return ensembles


def _model_manifest(models: list) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for model in models:
        rows.append(
            {
                "model_name": model.name,
                "class": type(model).__name__,
                "feature_set": getattr(model, "feature_set", ""),
                "component_models": ",".join(component.name for component in getattr(model, "models", []) or []),
                "weights": ",".join(str(value) for value in getattr(model, "weights", []) or []),
            }
        )
    return pd.DataFrame(rows)


def _add_research_audit_columns(
    ledger: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    *,
    contaminated_start: str,
    contaminated_through: str,
) -> pd.DataFrame:
    if ledger.empty or returns_matrix.empty:
        return ledger
    out = ledger.copy()
    rows: list[dict[str, float | str]] = []
    for candidate_id in out["candidate_id"]:
        returns = returns_matrix[str(candidate_id)].dropna() if str(candidate_id) in returns_matrix else pd.Series(dtype=float)
        rows.append(
            {
                "candidate_id": candidate_id,
                "research_net_sharpe": sharpe_ratio(returns, 252) if len(returns) > 2 else 0.0,
                "research_max_drawdown": max_drawdown(returns) if len(returns) > 2 else 0.0,
                "research_days": int(len(returns)),
                "holdout_net_sharpe": pd.NA,
                "holdout_max_drawdown": pd.NA,
                "holdout_days": 0,
                "historical_audit_status": (
                    f"contaminated_not_for_selection:{contaminated_start}:{contaminated_through}"
                ),
                "forward_holdout_status": "insufficient_post_freeze_observations",
            }
        )
    metrics = pd.DataFrame(rows)
    return out.merge(metrics, on="candidate_id", how="left")


def _add_pit_gate_columns(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return ledger
    out = ledger.copy()
    out = _label_feature_sets(out)
    out["research_positive"] = out["research_net_sharpe"].fillna(float("-inf")).gt(0.0)
    out["holdout_positive"] = False
    out["guarded_research_pass"] = out["passed_all"].fillna(False).astype(bool) & out["research_positive"]
    out["guarded_status"] = out["guarded_research_pass"].map({True: "research_survivor", False: "do_not_promote"})
    sort_cols = ["guarded_research_pass", "score", "net_ir", "net_sharpe"]
    return out.sort_values(sort_cols, ascending=[False, False, False, False]).reset_index(drop=True)


def _label_feature_sets(ledger: pd.DataFrame) -> pd.DataFrame:
    out = ledger.copy()
    if "meta_feature_set" not in out.columns:
        out["meta_feature_set"] = pd.NA
    model_names = out["model_name"].astype(str)
    out.loc[out["meta_feature_set"].isna() & model_names.str.contains("__pit_all"), "meta_feature_set"] = "source_plus_pit_all"
    out.loc[out["meta_feature_set"].isna() & model_names.str.contains("__pit_quality"), "meta_feature_set"] = "source_plus_pit_quality"
    out.loc[out["meta_feature_set"].isna() & model_names.str.startswith("pitabl_src_"), "meta_feature_set"] = "source_baseline"
    out["meta_feature_set"] = out["meta_feature_set"].fillna("unlabeled")
    return out


def _family_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    group_col = "meta_feature_set" if "meta_feature_set" in ledger.columns else "meta_family"
    return (
        ledger.groupby(group_col, dropna=False)
        .agg(
            candidates=("candidate_id", "count"),
            feasible=("passed_all", "sum"),
            guarded_survivors=("guarded_research_pass", "sum"),
            best_score=("score", "max"),
            best_net_ir=("net_ir", "max"),
            best_net_sharpe=("net_sharpe", "max"),
            best_holdout_sharpe=("holdout_net_sharpe", "max"),
            median_holdout_sharpe=("holdout_net_sharpe", "median"),
        )
        .reset_index()
        .rename(columns={group_col: "feature_set"})
        .sort_values(["guarded_survivors", "best_score", "best_holdout_sharpe"], ascending=[False, False, False])
    )


def _constraint_pass_summary(ledger: pd.DataFrame) -> pd.DataFrame:
    if ledger.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    pass_cols = [col for col in ledger.columns if col.startswith("constraint_") and col.endswith("_passed")]
    for col in sorted(pass_cols):
        passed = ledger[col].fillna(False).astype(bool)
        rows.append(
            {
                "constraint": col.removeprefix("constraint_").removesuffix("_passed"),
                "passed": int(passed.sum()),
                "failed": int((~passed).sum()),
                "pass_rate": float(passed.mean()),
            }
        )
    return pd.DataFrame(rows)


def _write_candidate_weights(artifact_dir: Path, candidate_results: list, ledger: pd.DataFrame, top_n: int) -> None:
    if not candidate_results or ledger.empty or top_n <= 0:
        return
    selected_ids = set(ledger.head(top_n)["candidate_id"])
    weight_dir = artifact_dir / "candidate_weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    for result in candidate_results:
        if result.candidate_id not in selected_ids:
            continue
        safe_name = result.candidate_id.replace("|", "__").replace("=", "-").replace("/", "_").replace(" ", "_")
        result.weights.to_parquet(weight_dir / f"{safe_name}.parquet")


def _write_summary(
    path: Path,
    ledger: pd.DataFrame,
    family_summary: pd.DataFrame,
    factor_quality: pd.DataFrame,
    constraint_summary: pd.DataFrame,
    config: PITEnhancedAblationConfig,
) -> None:
    best = ledger.iloc[0].to_dict() if not ledger.empty else {}
    survivor_count = int(ledger["guarded_research_pass"].sum()) if "guarded_research_pass" in ledger else 0
    lines = [
        "# PIT-Enhanced Alpha Ablation",
        "",
        f"Run ID: `{config.run_id}`",
        f"Universe: `{config.universe_tier}` Malaysia/Singapore/Indonesia",
        f"Research window: `{config.start_date}` to before `{config.holdout_start}`",
        f"Historical contaminated audit window: `{config.holdout_start}` to `{config.end_date}`",
        "",
        "## Verdict",
        "",
        f"- Candidates tested: `{len(ledger):,}`",
        f"- Guarded research survivors: `{survivor_count:,}`",
        f"- Best candidate: `{best.get('candidate_id', '')}`",
        f"- Best research-period net Sharpe: `{float(best.get('net_sharpe', 0.0)):.3f}`",
        "- Terminal holdout: `not evaluated; requires post-freeze observations`",
        "",
        "This is a research ablation. It is not production approval because corporate actions, delistings, and remaining PIT event/estimate fields are still incomplete.",
        "",
        "## Family Ablation",
        "",
        _markdown_table(family_summary),
        "",
        "## Blocking Gates",
        "",
        _markdown_table(constraint_summary),
        "",
        "## PIT Factor Coverage",
        "",
        _markdown_table(factor_quality),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).fillna("")
    columns = list(clipped.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = ["| " + " | ".join(str(row[col]).replace("|", "\\|") for col in columns) + " |" for _, row in clipped.iterrows()]
    return "\n".join([header, separator, *rows])


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _config_for_json(config: PITEnhancedAblationConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["l2_artifact_dir"] = str(config.l2_artifact_dir)
    payload["pit_qa_artifact_dir"] = str(config.pit_qa_artifact_dir)
    payload["artifact_root"] = str(config.artifact_root)
    payload["include_countries"] = list(config.include_countries)
    payload["cost_bps_grid"] = list(config.cost_bps_grid)
    return payload
