from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_lab.models.baselines import LowVolatilityModel, MomentumModel, VolatilityAdjustedMomentumModel
from quant_lab.models.zoo import (
    BreakoutModel,
    CountryRelativeMomentumModel,
    CrossCountryRankingModel,
    MeanReversionModel21,
    MeanReversionModel63,
    TimeSeriesMomentumModel,
    TrendLowVolEnsembleModel,
    WithinCountryRankingModel,
)
from quant_lab.optimization.net_ir_optimizer import NetIRConstrainedOptimizer
from quant_lab.research.asean import _equal_weight_benchmark, _metadata_from_artifact
from quant_lab.validation.metrics import deflated_sharpe_ratio


def _model_from_leaderboard_row(row: pd.Series, group_map: pd.Series | None) -> Any:
    """Instantiate the correct model class from a leaderboard row."""
    name = row["model_name"]

    # Extract model type from name
    if "lowvol_63" in name:
        return LowVolatilityModel(name=name, lookback=63)
    if "lowvol_126" in name:
        return LowVolatilityModel(name=name, lookback=126)
    if "lowvol_252" in name:
        return LowVolatilityModel(name=name, lookback=252)
    if "mom_21" in name:
        return MomentumModel(name=name, lookback=21)
    if "mom_63" in name:
        return MomentumModel(name=name, lookback=63)
    if "mom_126" in name:
        return MomentumModel(name=name, lookback=126)
    if "mom_252" in name:
        return MomentumModel(name=name, lookback=252)
    if "vam_63" in name:
        return VolatilityAdjustedMomentumModel(name=name, lookback=63, volatility_lookback=63)
    if "vam_126" in name:
        return VolatilityAdjustedMomentumModel(name=name, lookback=126, volatility_lookback=63)
    if "vam_vol21" in name:
        return VolatilityAdjustedMomentumModel(name=name, lookback=63, volatility_lookback=21)
    if "rev_5" in name:
        from quant_lab.models.baselines import MeanReversionModel
        return MeanReversionModel(name=name, lookback=5)
    if "rev_21" in name:
        return MeanReversionModel21(name=name, lookback=21)
    if "rev_63" in name:
        return MeanReversionModel63(name=name, lookback=63)
    if "tsmom_63" in name:
        return TimeSeriesMomentumModel(name=name, lookback=63)
    if "tsmom_126" in name:
        return TimeSeriesMomentumModel(name=name, lookback=126, scale_by_vol=True)
    if "breakout_20" in name:
        return BreakoutModel(name=name, lookback=20)
    if "trend_lowvol_ensemble" in name:
        return TrendLowVolEnsembleModel(
            name=name,
            trend_model=MomentumModel(name="trend_mom_63", lookback=63),
            lowvol_model=LowVolatilityModel(name="trend_lowvol_63", lookback=63),
        )
    if "ctry_rel_mom_63" in name and group_map is not None:
        return CountryRelativeMomentumModel(name=name, lookback=63, groups=group_map)
    if "within_ctry_63" in name and group_map is not None:
        return WithinCountryRankingModel(name=name, lookback=63, groups=group_map)
    if "cross_ctry_63" in name and group_map is not None:
        return CrossCountryRankingModel(name=name, lookback=63, groups=group_map)

    # Fallback: try basic momentum
    return MomentumModel(name=name, lookback=63)


def _allocator_from_leaderboard_row(row: pd.Series) -> dict[str, Any]:
    """Extract allocator config from leaderboard row."""
    weighting = row["weighting"]
    top_fraction = row.get("top_fraction", 0.33)
    # Default to daily rebalancing; we will test D and M
    return {
        "weighting": weighting,
        "top_fraction": top_fraction,
        "rebalance_frequency": "D",
    }


def main() -> None:
    artifact_dir = Path("artifacts/asean/ASEAN-20260704-232302")
    prices_path = artifact_dir / "eligible_stock_usd_prices.parquet"
    eligible_prices = pd.read_parquet(prices_path)
    metadata = _metadata_from_artifact(artifact_dir)
    group_map = metadata.set_index("ticker")["country"].reindex(eligible_prices.columns)
    benchmark_returns = _equal_weight_benchmark(eligible_prices)

    # Read the upgraded leaderboard
    upgraded_dir = Path("artifacts/asean/upgraded")
    leaderboard = pd.read_csv(upgraded_dir / "leaderboard.csv")

    # Filter out benchmark rows, keep only top 15 models
    models_only = leaderboard[
        ~leaderboard["model_name"].isin(["asean_equal_weight_usd", "asean_equal_country_usd"])
    ].head(15)

    print("=" * 70)
    print("FOCUSED NET IR CONSTRAINED OPTIMIZATION")
    print("=" * 70)
    print(f"Evaluating top {len(models_only)} models from upgraded leaderboard")
    print(f"Models: {list(models_only['model_name'])}")
    print()

    optimizer = NetIRConstrainedOptimizer(
        drawdown_limit=-0.20,
        max_turnover=3.0,
        max_country_weight_limit=0.40,
        max_asset_weight_limit=0.20,
        min_deflated_sharpe=0.80,
        target_volatility=0.12,
        train_size=504,
        test_size=63,
        embargo=5,
    )

    # Build candidate list from leaderboard rows
    results = []
    for _, row in models_only.iterrows():
        model = _model_from_leaderboard_row(row, group_map)
        base_config = _allocator_from_leaderboard_row(row)

        # Test this model with its base config at 3 cost levels
        # Plus a monthly rebalance variant for the top models
        configs = [base_config]
        if row["sharpe"] > 1.5:
            monthly_config = base_config.copy()
            monthly_config["rebalance_frequency"] = "M"
            configs.append(monthly_config)

        for config in configs:
            for cost_bps in (0.0, 15.0, 50.0):
                try:
                    candidate = optimizer.evaluate_candidate(
                        model=model,
                        weighting=config["weighting"],
                        top_fraction=config["top_fraction"],
                        rebalance_frequency=config["rebalance_frequency"],
                        cost_bps=cost_bps,
                        prices=eligible_prices,
                        group_map=group_map,
                        benchmark_returns=benchmark_returns,
                        trial_sharpes=[],  # Will be filled in second pass
                    )
                    results.append(candidate)
                    print(f"  Evaluated: {model.name} | {config['weighting']}_top{int(config['top_fraction']*100)}_{config['rebalance_frequency']} | {cost_bps:.0f} bps | IR={candidate.net_ir:.3f} | Passed={candidate.passed_all}")
                except Exception as e:
                    print(f"  ERROR: {model.name} | {config} | {cost_bps} bps: {e}")

    # Second pass: recompute deflated Sharpe with full trial set
    trial_sharpes = [r.net_sharpe for r in results]
    final_results = []
    for result in results:
        constraints = optimizer._evaluate_constraints(
            result.net_returns,
            result.gross_returns,
            result.weights,
            group_map,
            result.cost_bps,
            trial_sharpes,
        )
        from quant_lab.optimization.net_ir_optimizer import OptimizedCandidateResult
        final_results.append(
            OptimizedCandidateResult(
                model_name=result.model_name,
                allocator_name=result.allocator_name,
                cost_bps=result.cost_bps,
                net_ir=result.net_ir,
                net_sharpe=result.net_sharpe,
                gross_sharpe=result.gross_sharpe,
                deflated_sharpe=result.net_sharpe,  # Will be recomputed below
                annual_return=result.annual_return,
                annual_volatility=result.annual_volatility,
                max_drawdown=result.max_drawdown,
                turnover=result.turnover,
                annual_excess_return=result.annual_excess_return,
                tracking_error=result.tracking_error,
                information_ratio=result.information_ratio,
                constraints=constraints,
                net_returns=result.net_returns,
                weights=result.weights,
                cost_returns=result.cost_returns,
                gross_returns=result.gross_returns,
                metadata=result.metadata,
            )
        )

    # Recompute DSR for all
    for i, result in enumerate(final_results):
        dsr = deflated_sharpe_ratio(result.net_returns, trial_sharpes=trial_sharpes, periods_per_year=252)
        # Rebuild with corrected DSR
        constraints = optimizer._evaluate_constraints(
            result.net_returns,
            result.gross_returns,
            result.weights,
            group_map,
            result.cost_bps,
            trial_sharpes,
        )
        final_results[i] = OptimizedCandidateResult(
            model_name=result.model_name,
            allocator_name=result.allocator_name,
            cost_bps=result.cost_bps,
            net_ir=result.net_ir,
            net_sharpe=result.net_sharpe,
            gross_sharpe=result.gross_sharpe,
            deflated_sharpe=dsr,
            annual_return=result.annual_return,
            annual_volatility=result.annual_volatility,
            max_drawdown=result.max_drawdown,
            turnover=result.turnover,
            annual_excess_return=result.annual_excess_return,
            tracking_error=result.tracking_error,
            information_ratio=result.information_ratio,
            constraints=constraints,
            net_returns=result.net_returns,
            weights=result.weights,
            cost_returns=result.cost_returns,
            gross_returns=result.gross_returns,
            metadata=result.metadata,
        )

    final_results.sort(key=lambda r: r.score, reverse=True)

    df = pd.DataFrame([r.to_dict() for r in final_results])
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    output_dir = Path("artifacts/asean/constrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "constrained_results.csv", index=False)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total evaluations: {len(df)}")
    print()

    # Top 15 by score
    print("--- TOP 15 BY COMPOSITE SCORE (Net IR - 10×Violation) ---")
    top15 = df.head(15)
    display_cols = [
        "model_name", "allocator_name", "cost_bps", "net_ir", "net_sharpe",
        "deflated_sharpe", "annual_return", "max_drawdown", "turnover",
        "passed_all", "score", "constraint_violation",
    ]
    print(top15[[c for c in display_cols if c in top15.columns]].to_string(index=False))
    print()

    # Feasible
    feasible = df[df["passed_all"] == True]
    print(f"--- FEASIBLE SOLUTIONS: {len(feasible)} of {len(df)} ---")
    if not feasible.empty:
        print(feasible.head(10)[[c for c in display_cols if c in feasible.columns]].to_string(index=False))
    else:
        print("No feasible solutions found.")
    print()

    # Best by cost
    print("--- BEST BY COST LEVEL ---")
    for cost in (0.0, 15.0, 50.0):
        subset = df[df["cost_bps"] == cost].head(1)
        if not subset.empty:
            print(f"  {cost:.0f} bps: {subset.iloc[0]['model_name']} | "
                  f"IR={subset.iloc[0]['net_ir']:.3f} | "
                  f"Sharpe={subset.iloc[0]['net_sharpe']:.3f} | "
                  f"DSR={subset.iloc[0]['deflated_sharpe']:.3f} | "
                  f"DD={subset.iloc[0]['max_drawdown']:.1%} | "
                  f"TO={subset.iloc[0]['turnover']:.3f} | "
                  f"Passed={subset.iloc[0]['passed_all']}")
    print()

    # Constraint breakdown
    print("--- CONSTRAINT BREAKDOWN ---")
    for c in ["max_drawdown", "turnover", "max_country_weight", "max_asset_weight", "cost_stress_2x", "deflated_sharpe"]:
        col = f"constraint_{c}_passed"
        if col in df.columns:
            passed = df[col].sum()
            total = len(df)
            print(f"  {c}: {passed}/{total} passed ({passed/total*100:.1f}%)")
    print()

    # Best feasible by cost
    print("--- BEST FEASIBLE BY COST LEVEL ---")
    for cost in (0.0, 15.0, 50.0):
        subset = feasible[feasible["cost_bps"] == cost].head(1)
        if not subset.empty:
            print(f"  {cost:.0f} bps: {subset.iloc[0]['model_name']} | "
                  f"IR={subset.iloc[0]['net_ir']:.3f} | "
                  f"Sharpe={subset.iloc[0]['net_sharpe']:.3f} | "
                  f"DSR={subset.iloc[0]['deflated_sharpe']:.3f}")
        else:
            print(f"  {cost:.0f} bps: No feasible solution")
    print()

    print("Results saved to:", output_dir / "constrained_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
