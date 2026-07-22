from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.models.base import AlphaModel
from quant_lab.models.baselines import (
    LowVolatilityModel,
    MeanReversionModel,
    MomentumModel,
    VolatilityAdjustedMomentumModel,
)
from quant_lab.models.signal_factory import signal_manifest
from quant_lab.models.zoo import (
    BreakoutModel,
    TimeSeriesMomentumModel,
    TrendLowVolEnsembleModel,
    VolatilityCompressionModel,
)
from quant_lab.notes.obsidian import ObsidianNote, write_note
from quant_lab.optimization.net_ir_optimizer import NetIRConstrainedOptimizer
from quant_lab.regimes import VolTrendRegimeDetector
from quant_lab.research.asean import _benchmark_row, _equal_weight_benchmark, _markdown_table
from quant_lab.research.malaysia import _eligible_malaysia_prices, _ewm_benchmark
from quant_lab.validation.metrics import deflated_sharpe_ratio


@dataclass(frozen=True)
class MalaysiaSweepResult:
    run_id: str
    artifact_dir: Path
    all_candidates: pd.DataFrame
    champions: pd.DataFrame
    feasible_candidates: pd.DataFrame
    killed_strategies: pd.DataFrame
    benchmarks: pd.DataFrame
    note_paths: list[Path]


@dataclass
class CachedRegimeConditionedModel(AlphaModel):
    """Regime-conditioned wrapper that caches regime labels per price index."""

    model: AlphaModel | None = None
    regime: str = ""
    detector: VolTrendRegimeDetector | None = None
    _cached_index: pd.Index | None = None
    _cached_regimes: pd.Series | None = None

    def fit(self, prices: pd.DataFrame, start: int, end: int) -> "CachedRegimeConditionedModel":
        if self.model is None:
            raise ValueError("model must be provided")
        if not self.regime:
            raise ValueError("regime must be provided")
        self.model.fit(prices, start, end)
        if self._cached_index is None or not self._cached_index.equals(prices.index):
            detector = self.detector or VolTrendRegimeDetector()
            self._cached_regimes = detector.describe(prices)
            self._cached_index = prices.index.copy()
        return self

    def predict(self, prices: pd.DataFrame, start: int, end: int):
        self.fit(prices, start, end)
        if self._cached_regimes is None:
            raise ValueError("cached regimes were not initialized")
        model_signal = self.model.predict(prices, start, end)
        scores = model_signal.scores.copy()
        window_regimes = self._cached_regimes.iloc[start:end]
        mask = window_regimes.apply(lambda value: self.regime in value if pd.notna(value) else False)
        if not mask.any():
            scores[:] = 0.0
        else:
            scores.loc[~mask] = 0.0
        return type(model_signal)(
            scores=scores,
            model_name=self.name,
            horizon=self.horizon,
            metadata={
                "family": "regime_conditioned",
                "regime": self.regime,
                "wrapped_model": self.model.name,
            },
        )


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("MALAYSIA-SWEEP-%Y%m%d-%H%M%S")


def build_malaysia_full_model_suite(prefix: str = "malaysia") -> list[AlphaModel]:
    """Build a broad price-only Malaysia strategy suite.

    Volume-confirmed and country-relative models are intentionally excluded:
    the current Malaysia artifact has adjusted prices, not reliable volume, and
    all stocks are from one country.
    """

    models: list[AlphaModel] = []

    for lookback in (21, 63, 126, 252):
        models.append(LowVolatilityModel(name=f"{prefix}_lowvol_{lookback}", lookback=lookback))

    for lookback in (21, 63, 126, 252):
        models.append(MomentumModel(name=f"{prefix}_mom_{lookback}", lookback=lookback))

    for lookback, vol_lookback in ((21, 21), (63, 21), (63, 63), (126, 63), (252, 126)):
        models.append(
            VolatilityAdjustedMomentumModel(
                name=f"{prefix}_vam_{lookback}_vol{vol_lookback}",
                lookback=lookback,
                volatility_lookback=vol_lookback,
            )
        )

    for lookback in (5, 10, 21, 63):
        models.append(MeanReversionModel(name=f"{prefix}_rev_{lookback}", lookback=lookback))

    for lookback, scale_by_vol in ((21, False), (63, False), (63, True), (126, True), (252, True)):
        suffix = "vol" if scale_by_vol else "raw"
        models.append(
            TimeSeriesMomentumModel(
                name=f"{prefix}_tsmom_{lookback}_{suffix}",
                lookback=lookback,
                scale_by_vol=scale_by_vol,
            )
        )

    for lookback in (20, 55, 126):
        models.append(BreakoutModel(name=f"{prefix}_breakout_{lookback}", lookback=lookback))

    for vol_lookback, compression_lookback in ((21, 63), (42, 126), (63, 252)):
        models.append(
            VolatilityCompressionModel(
                name=f"{prefix}_volcomp_{vol_lookback}_{compression_lookback}",
                vol_lookback=vol_lookback,
                compression_lookback=compression_lookback,
            )
        )

    for lookback in (63, 126, 252):
        models.append(
            TrendLowVolEnsembleModel(
                name=f"{prefix}_trend_lowvol_{lookback}",
                trend_model=MomentumModel(name=f"{prefix}_trend_mom_{lookback}", lookback=lookback),
                lowvol_model=LowVolatilityModel(name=f"{prefix}_trend_lowvol_leg_{lookback}", lookback=lookback),
            )
        )

    regime_specs = [
        ("trend_up_mom_126", MomentumModel(name=f"{prefix}_regime_inner_mom_126", lookback=126), "trend_up"),
        ("trend_up_tsmom_126", TimeSeriesMomentumModel(name=f"{prefix}_regime_inner_tsmom_126", lookback=126, scale_by_vol=True), "trend_up"),
        ("trend_down_lowvol_63", LowVolatilityModel(name=f"{prefix}_regime_inner_lowvol_63", lookback=63), "trend_down"),
        ("high_vol_lowvol_63", LowVolatilityModel(name=f"{prefix}_regime_inner_highvol_lowvol_63", lookback=63), "high_vol"),
        ("low_vol_mom_63", MomentumModel(name=f"{prefix}_regime_inner_lowvol_mom_63", lookback=63), "low_vol"),
        ("trend_down_rev_21", MeanReversionModel(name=f"{prefix}_regime_inner_rev_21", lookback=21), "trend_down"),
    ]
    for label, inner, regime in regime_specs:
        models.append(
            CachedRegimeConditionedModel(
                name=f"{prefix}_regime_{label}",
                model=inner,
                regime=regime,
            )
        )

    return models


def build_malaysia_allocator_configs(
    top_fractions: tuple[float, ...] = (0.10, 0.20, 0.33, 0.50),
    rebalance_frequencies: tuple[str, ...] = ("D", "W", "M"),
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for top_fraction in top_fractions:
        for frequency in rebalance_frequencies:
            configs.append(
                {
                    "weighting": "long_only",
                    "top_fraction": top_fraction,
                    "rebalance_frequency": frequency,
                }
            )
    for frequency in rebalance_frequencies:
        configs.append(
            {
                "weighting": "dollar_neutral",
                "top_fraction": 1.0,
                "rebalance_frequency": frequency,
            }
        )
    return configs


def _family_from_model_name(model_name: str) -> str:
    if "regime_" in model_name:
        return "regime_conditioned"
    if "trend_lowvol" in model_name:
        return "trend_low_vol_ensemble"
    if "lowvol" in model_name:
        return "low_volatility"
    if "volcomp" in model_name:
        return "volatility_compression"
    if "_vam_" in model_name:
        return "volatility_adjusted_momentum"
    if "_tsmom_" in model_name:
        return "time_series_momentum"
    if "_rev_" in model_name:
        return "mean_reversion"
    if "_breakout_" in model_name:
        return "breakout"
    if "_mom_" in model_name:
        return "momentum"
    return "other"


def _candidate_key(model: AlphaModel, config: dict[str, Any], cost_bps: float) -> str:
    return (
        f"{model.name}|{config['weighting']}|"
        f"{config['top_fraction']}|{config['rebalance_frequency']}|{cost_bps}"
    )


def _failure_reasons(row: pd.Series) -> str:
    reasons = []
    if not bool(row.get("constraint_max_drawdown_passed", True)):
        reasons.append("drawdown")
    if not bool(row.get("constraint_turnover_passed", True)):
        reasons.append("turnover")
    if not bool(row.get("constraint_max_asset_weight_passed", True)):
        reasons.append("asset_concentration")
    if not bool(row.get("constraint_cost_stress_2x_passed", True)):
        reasons.append("2x_cost_stress")
    if not bool(row.get("constraint_deflated_sharpe_passed", True)):
        reasons.append("deflated_sharpe")
    if not reasons and row.get("passed_all") is False:
        reasons.append("unspecified_gate")
    return "; ".join(reasons)


def _refresh_multiple_testing_metrics(
    rows: list[dict[str, Any]],
    returns_by_key: dict[str, pd.Series],
    trial_sharpes: list[float],
    optimizer: NetIRConstrainedOptimizer,
) -> pd.DataFrame:
    refreshed = []
    for row in rows:
        result = dict(row)
        returns = returns_by_key[result["candidate_key"]]
        dsr = deflated_sharpe_ratio(returns, trial_sharpes=trial_sharpes, periods_per_year=optimizer.periods_per_year)
        result["deflated_sharpe"] = dsr
        result["constraint_deflated_sharpe"] = dsr
        result["constraint_deflated_sharpe_passed"] = dsr >= optimizer.min_deflated_sharpe

        violations = [
            max(0.0, optimizer.drawdown_limit - float(result.get("max_drawdown", 0.0))),
            max(0.0, float(result.get("turnover", 0.0)) - optimizer.max_turnover),
            max(0.0, float(result.get("constraint_max_country_weight", 0.0)) - optimizer.max_country_weight_limit),
            max(0.0, float(result.get("constraint_max_asset_weight", 0.0)) - optimizer.max_asset_weight_limit),
            max(0.0, -float(result.get("constraint_cost_stress_2x", 0.0))),
            max(0.0, optimizer.min_deflated_sharpe - dsr),
        ]
        result["constraint_violation"] = sum(violations)
        result["passed_all"] = all(
            bool(result.get(column, False))
            for column in (
                "constraint_max_drawdown_passed",
                "constraint_turnover_passed",
                "constraint_max_country_weight_passed",
                "constraint_max_asset_weight_passed",
                "constraint_cost_stress_2x_passed",
                "constraint_deflated_sharpe_passed",
            )
        )
        result["score"] = float(result.get("net_ir", 0.0)) - 10.0 * float(result["constraint_violation"])
        result["failure_reasons"] = _failure_reasons(pd.Series(result))
        refreshed.append(result)

    frame = pd.DataFrame(refreshed)
    if frame.empty:
        return frame
    return frame.sort_values(
        ["passed_all", "score", "net_ir", "net_sharpe", "deflated_sharpe"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def _summarize_killed_strategies(all_candidates: pd.DataFrame) -> pd.DataFrame:
    if all_candidates.empty:
        return pd.DataFrame()
    rows = []
    for model_name, group in all_candidates.groupby("model_name", sort=False):
        ranked = group.sort_values(["passed_all", "score", "net_ir"], ascending=[False, False, False])
        best = ranked.iloc[0]
        rows.append(
            {
                "model_name": model_name,
                "family": best["family"],
                "candidate_count": len(group),
                "passed_candidates": int(group["passed_all"].sum()),
                "best_score": best["score"],
                "best_net_ir": best["net_ir"],
                "best_net_sharpe": best["net_sharpe"],
                "best_deflated_sharpe": best["deflated_sharpe"],
                "best_allocator_name": best["allocator_name"],
                "best_cost_bps": best["cost_bps"],
                "status": "survived" if bool(group["passed_all"].any()) else "killed",
                "primary_failure_reasons": best.get("failure_reasons", ""),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["status", "best_score", "best_net_ir"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def _benchmarks(prices: pd.DataFrame, source_prices: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _benchmark_row("malaysia_equal_weight_usd", _equal_weight_benchmark(prices)),
        _benchmark_row("malaysia_ewm_etf_usd", _ewm_benchmark(source_prices.reindex(prices.index))),
    ]
    frame = pd.DataFrame(rows)
    return frame.rename(columns={"sharpe": "net_sharpe"})


def _write_summary_note(
    vault_root: Path,
    run_id: str,
    artifact_dir: Path,
    source_artifact: Path,
    all_candidates: pd.DataFrame,
    champions: pd.DataFrame,
    feasible_candidates: pd.DataFrame,
    killed_strategies: pd.DataFrame,
    benchmarks: pd.DataFrame,
    eligible_prices: pd.DataFrame,
    baseline_cost_bps: float,
) -> list[Path]:
    experiment_dir = Path("30_experiments") / run_id
    best = champions.iloc[0].to_dict() if not champions.empty else {}
    killed_count = int((killed_strategies["status"] == "killed").sum()) if not killed_strategies.empty else 0
    survived_count = int((killed_strategies["status"] == "survived").sum()) if not killed_strategies.empty else 0
    artifact_link = artifact_dir.resolve()

    body = [
        f"Run id: `{run_id}`",
        f"Source ASEAN artifact: `{source_artifact.resolve()}`",
        f"Artifact directory: `{artifact_link}`",
        f"Eligible Malaysia stocks: `{eligible_prices.shape[1]}`",
        f"Dates: `{eligible_prices.index.min()}` to `{eligible_prices.index.max()}`",
        f"Candidates evaluated: `{len(all_candidates)}`",
        f"Feasible candidates: `{len(feasible_candidates)}`",
        f"Survived model families/models: `{survived_count}`",
        f"Killed models: `{killed_count}`",
        f"Baseline cost for champion view: `{baseline_cost_bps}` bps",
        "",
        "## Champion",
        "No champion was produced." if not best else "\n".join(f"- `{key}`: `{value}`" for key, value in best.items()),
        "",
        "## Champions",
        _markdown_table(champions.head(25)) if not champions.empty else "_No rows._",
        "",
        "## Benchmarks",
        _markdown_table(benchmarks) if not benchmarks.empty else "_No rows._",
        "",
        "## Killed Strategies",
        _markdown_table(killed_strategies.head(40)) if not killed_strategies.empty else "_No rows._",
        "",
        "## Caveats",
        "- This is broader than the Malaysia pilot, but it still uses a curated liquid current-survivor Malaysia basket.",
        "- It is not a point-in-time full Bursa database and does not include delistings.",
        "- Volume-confirmed strategies were skipped because the source artifact is price-only.",
        "- Country-relative and country-neutral strategies are not meaningful in a single-country Malaysia-only universe.",
        "- Results remain research-only until liquidity, borrow, corporate actions, taxes, and live execution checks are added.",
    ]

    path = write_note(
        vault_root,
        experiment_dir / "index.md",
        ObsidianNote(
            title=f"{run_id} Malaysia Full Strategy Sweep",
            note_type="experiment",
            tags=["quant-lab", "malaysia", "full-sweep", "backtest"],
            metadata={"experiment_id": run_id, "status": "completed", "artifact_dir": str(artifact_link)},
            body="\n".join(body),
        ),
    )
    return [path]


def run_malaysia_full_sweep_from_asean_artifact(
    asean_artifact_dir: str | Path = "artifacts/asean/ASEAN-20260704-232302",
    artifact_root: str | Path = "artifacts/malaysia",
    vault_root: str | Path = "vault",
    run_id: str | None = None,
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 25.0, 50.0),
    baseline_cost_bps: float = 15.0,
    train_size: int = 504,
    test_size: int = 63,
    embargo: int = 5,
    min_rows: int = 1000,
    max_missing_ratio: float = 0.05,
    models: list[AlphaModel] | None = None,
    allocator_configs: list[dict[str, Any]] | None = None,
    drawdown_limit: float = -0.20,
    max_turnover: float = 3.0,
    max_asset_weight_limit: float = 0.25,
    min_deflated_sharpe: float = 0.80,
    progress_interval: int | None = 250,
) -> MalaysiaSweepResult:
    source_path = Path(asean_artifact_dir)
    prices_path = source_path / "daily_usd_prices.parquet"
    metadata_path = source_path / "universe.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"missing prices: {prices_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing universe metadata: {metadata_path}")

    source_prices = pd.read_parquet(prices_path).sort_index()
    metadata = pd.read_csv(metadata_path)
    eligible_prices = _eligible_malaysia_prices(
        source_prices,
        metadata,
        min_rows=min_rows,
        max_missing_ratio=max_missing_ratio,
    )

    run_id = run_id or _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    malaysia_metadata = metadata.loc[
        metadata["ticker"].isin(eligible_prices.columns)
        | ((metadata["country"] == "Malaysia") & metadata["tier"].isin(["regional_proxy", "country_index"]))
    ].copy()
    malaysia_metadata.to_csv(artifact_dir / "universe.csv", index=False)
    eligible_prices.to_parquet(artifact_dir / "eligible_usd_prices.parquet")

    if eligible_prices.shape[1] < 5 or eligible_prices.shape[0] < train_size + test_size:
        raise ValueError("not enough eligible Malaysia price history for the requested sweep")

    models = models or build_malaysia_full_model_suite(prefix="malaysia")
    allocator_configs = allocator_configs or build_malaysia_allocator_configs()
    benchmark_returns = _equal_weight_benchmark(eligible_prices)
    optimizer = NetIRConstrainedOptimizer(
        drawdown_limit=drawdown_limit,
        max_turnover=max_turnover,
        max_country_weight_limit=1.0,
        max_asset_weight_limit=max_asset_weight_limit,
        min_deflated_sharpe=min_deflated_sharpe,
        target_volatility=0.12,
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
    )

    rows: list[dict[str, Any]] = []
    returns_by_key: dict[str, pd.Series] = {}
    trial_sharpes: list[float] = []
    failures: list[dict[str, Any]] = []
    total_attempts = len(models) * len(allocator_configs) * len(cost_bps_grid)
    attempts = 0

    for model in models:
        for config in allocator_configs:
            for cost_bps in cost_bps_grid:
                attempts += 1
                key = _candidate_key(model, config, cost_bps)
                try:
                    candidate = optimizer.evaluate_candidate(
                        model=model,
                        weighting=config["weighting"],
                        top_fraction=float(config["top_fraction"]),
                        rebalance_frequency=config["rebalance_frequency"],
                        cost_bps=cost_bps,
                        prices=eligible_prices,
                        group_map=None,
                        benchmark_returns=benchmark_returns,
                        trial_sharpes=[],
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "candidate_key": key,
                            "model_name": model.name,
                            "allocator_config": json.dumps(config, sort_keys=True),
                            "cost_bps": cost_bps,
                            "error": str(exc),
                        }
                    )
                    continue

                row = candidate.to_dict()
                row["candidate_key"] = key
                row["family"] = _family_from_model_name(candidate.model_name)
                trial_sharpes.append(candidate.net_sharpe)
                returns_by_key[key] = candidate.net_returns
                rows.append(row)
                if progress_interval and attempts % progress_interval == 0:
                    print(
                        f"Malaysia sweep progress: {attempts}/{total_attempts} candidates "
                        f"({len(failures)} failed)",
                        flush=True,
                    )

    if progress_interval:
        print(
            f"Malaysia sweep progress: {attempts}/{total_attempts} candidates "
            f"({len(failures)} failed)",
            flush=True,
        )

    all_candidates = _refresh_multiple_testing_metrics(rows, returns_by_key, trial_sharpes, optimizer)
    if not all_candidates.empty:
        all_candidates.to_csv(artifact_dir / "all_candidates.csv", index=False)
    else:
        pd.DataFrame().to_csv(artifact_dir / "all_candidates.csv", index=False)

    feasible_candidates = all_candidates.loc[all_candidates["passed_all"]].copy() if not all_candidates.empty else pd.DataFrame()
    feasible_candidates.to_csv(artifact_dir / "feasible_candidates.csv", index=False)

    baseline = all_candidates.loc[all_candidates["cost_bps"] == baseline_cost_bps].copy() if not all_candidates.empty else pd.DataFrame()
    champion_source = baseline if not baseline.empty else all_candidates
    champions = champion_source.head(50).copy() if not champion_source.empty else pd.DataFrame()
    champions.to_csv(artifact_dir / "champions.csv", index=False)

    best_by_cost = (
        all_candidates.sort_values(["cost_bps", "passed_all", "score", "net_ir"], ascending=[True, False, False, False])
        .groupby("cost_bps", as_index=False)
        .head(10)
        if not all_candidates.empty
        else pd.DataFrame()
    )
    best_by_cost.to_csv(artifact_dir / "best_by_cost.csv", index=False)

    best_by_family = (
        all_candidates.sort_values(["family", "passed_all", "score", "net_ir"], ascending=[True, False, False, False])
        .groupby("family", as_index=False)
        .head(5)
        if not all_candidates.empty
        else pd.DataFrame()
    )
    best_by_family.to_csv(artifact_dir / "best_by_family.csv", index=False)

    best_by_allocator = (
        all_candidates.sort_values(["allocator_name", "passed_all", "score", "net_ir"], ascending=[True, False, False, False])
        .groupby("allocator_name", as_index=False)
        .head(5)
        if not all_candidates.empty
        else pd.DataFrame()
    )
    best_by_allocator.to_csv(artifact_dir / "best_by_allocator.csv", index=False)

    killed_strategies = _summarize_killed_strategies(all_candidates)
    killed_strategies.to_csv(artifact_dir / "killed_strategies.csv", index=False)

    benchmarks = _benchmarks(eligible_prices, source_prices)
    benchmarks.to_csv(artifact_dir / "benchmarks.csv", index=False)

    signal_manifest(models).to_csv(artifact_dir / "signal_manifest.csv", index=False)
    pd.DataFrame(allocator_configs).to_csv(artifact_dir / "allocator_manifest.csv", index=False)
    pd.DataFrame(failures).to_csv(artifact_dir / "failed_evaluations.csv", index=False)

    manifest = {
        "run_id": run_id,
        "source_artifact": str(source_path.resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "eligible_tickers": int(eligible_prices.shape[1]),
        "dates": {
            "start": str(eligible_prices.index.min()),
            "end": str(eligible_prices.index.max()),
            "rows": int(eligible_prices.shape[0]),
        },
        "models": len(models),
        "allocator_configs": len(allocator_configs),
        "cost_bps_grid": list(cost_bps_grid),
        "candidate_count": len(all_candidates),
        "failed_evaluations": len(failures),
        "feasible_candidates": len(feasible_candidates),
        "baseline_cost_bps": baseline_cost_bps,
        "constraints": {
            "drawdown_limit": drawdown_limit,
            "max_turnover": max_turnover,
            "max_asset_weight_limit": max_asset_weight_limit,
            "min_deflated_sharpe": min_deflated_sharpe,
            "train_size": train_size,
            "test_size": test_size,
            "embargo": embargo,
        },
    }
    (artifact_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    note_paths = _write_summary_note(
        vault_root=Path(vault_root),
        run_id=run_id,
        artifact_dir=artifact_dir,
        source_artifact=source_path,
        all_candidates=all_candidates,
        champions=champions,
        feasible_candidates=feasible_candidates,
        killed_strategies=killed_strategies,
        benchmarks=benchmarks,
        eligible_prices=eligible_prices,
        baseline_cost_bps=baseline_cost_bps,
    )

    return MalaysiaSweepResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        all_candidates=all_candidates,
        champions=champions,
        feasible_candidates=feasible_candidates,
        killed_strategies=killed_strategies,
        benchmarks=benchmarks,
        note_paths=note_paths,
    )
