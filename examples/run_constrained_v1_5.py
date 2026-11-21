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


def _model_from_leaderboard_row(name: str, group_map: pd.Series | None):
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
    return MomentumModel(name=name, lookback=63)


def main() -> None:
    artifact_dir = Path("artifacts/asean/ASEAN-20260704-232302")
    eligible_prices = pd.read_parquet(artifact_dir / "eligible_stock_usd_prices.parquet")
    metadata = _metadata_from_artifact(artifact_dir)
    group_map = metadata.set_index("ticker")["country"].reindex(eligible_prices.columns)
    benchmark_returns = _equal_weight_benchmark(eligible_prices)

    upgraded_dir = Path("artifacts/asean/upgraded")
    leaderboard = pd.read_csv(upgraded_dir / "leaderboard.csv")
    models_only = leaderboard[
        ~leaderboard["model_name"].isin(["asean_equal_weight_usd", "asean_equal_country_usd"])
    ].head(15)

    print("=" * 70)
    print("CONSTRAINED OPTIMIZATION — MAX LEVERAGE = 1.5")
    print("=" * 70)
    print(f"Evaluating top {len(models_only)} models")
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
        max_leverage=1.5,
    )

    results = []
    for _, row in models_only.iterrows():
        model = _model_from_leaderboard_row(row["model_name"], group_map)
        weighting = row["weighting"]
        top_fraction = row.get("top_fraction", 0.33)
        for freq in ("D", "M"):
            for cost_bps in (0.0, 15.0, 50.0):
                try:
                    r = optimizer.evaluate_candidate(
                        model, weighting, top_fraction, freq, cost_bps,
                        eligible_prices, group_map, benchmark_returns, [],
                    )
                    results.append(r)
                except Exception as e:
                    print(f"  ERROR: {model.name} | {freq} | {cost_bps}bps: {e}")

    trial_sharpes = [r.net_sharpe for r in results]
    from quant_lab.optimization.net_ir_optimizer import OptimizedCandidateResult
    final_results = []
    for r in results:
        constraints = optimizer._evaluate_constraints(
            r.net_returns, r.gross_returns, r.weights, group_map, r.cost_bps, trial_sharpes
        )
        dsr = deflated_sharpe_ratio(r.net_returns, trial_sharpes, 252)
        final_results.append(OptimizedCandidateResult(
            model_name=r.model_name, allocator_name=r.allocator_name, cost_bps=r.cost_bps,
            net_ir=r.net_ir, net_sharpe=r.net_sharpe, gross_sharpe=r.gross_sharpe,
            deflated_sharpe=dsr, annual_return=r.annual_return, annual_volatility=r.annual_volatility,
            max_drawdown=r.max_drawdown, turnover=r.turnover, annual_excess_return=r.annual_excess_return,
            tracking_error=r.tracking_error, information_ratio=r.information_ratio,
            constraints=constraints, net_returns=r.net_returns, weights=r.weights,
            cost_returns=r.cost_returns, gross_returns=r.gross_returns, metadata=r.metadata,
        ))

    final_results.sort(key=lambda x: x.score, reverse=True)
    df = pd.DataFrame([r.to_dict() for r in final_results])
    if not df.empty:
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

    output_dir = Path("artifacts/asean/constrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "constrained_results_v1_5.csv", index=False)

    print(f"Total evaluations: {len(df)}")
    print()

    print("--- TOP 15 BY COMPOSITE SCORE ---")
    top15 = df.head(15)
    cols = ["model_name","allocator_name","cost_bps","net_ir","net_sharpe",
            "deflated_sharpe","annual_return","max_drawdown","turnover","passed_all","score","constraint_violation"]
    print(top15[[c for c in cols if c in top15.columns]].to_string(index=False))
    print()

    feasible = df[df["passed_all"] == True]
    print(f"--- FEASIBLE SOLUTIONS: {len(feasible)} of {len(df)} ---")
    if not feasible.empty:
        print(feasible.head(15)[[c for c in cols if c in feasible.columns]].to_string(index=False))
    else:
        print("No feasible solutions found.")
    print()

    print("--- BEST BY COST LEVEL ---")
    for cost in (0.0, 15.0, 50.0):
        subset = df[df["cost_bps"] == cost].head(1)
        if not subset.empty:
            print(f"  {cost:.0f} bps: {subset.iloc[0]['model_name']} | {subset.iloc[0]['allocator_name']} | "
                  f"IR={subset.iloc[0]['net_ir']:.3f} | Sharpe={subset.iloc[0]['net_sharpe']:.3f} | "
                  f"DSR={subset.iloc[0]['deflated_sharpe']:.3f} | "
                  f"DD={subset.iloc[0]['max_drawdown']:.1%} | TO={subset.iloc[0]['turnover']:.3f} | "
                  f"Passed={subset.iloc[0]['passed_all']}")
    print()

    print("--- BEST FEASIBLE BY COST LEVEL ---")
    for cost in (0.0, 15.0, 50.0):
        subset = feasible[feasible["cost_bps"] == cost].head(1)
        if not subset.empty:
            print(f"  {cost:.0f} bps: {subset.iloc[0]['model_name']} | {subset.iloc[0]['allocator_name']} | "
                  f"IR={subset.iloc[0]['net_ir']:.3f} | Sharpe={subset.iloc[0]['net_sharpe']:.3f} | "
                  f"DSR={subset.iloc[0]['deflated_sharpe']:.3f}")
        else:
            print(f"  {cost:.0f} bps: No feasible solution")
    print()

    print("--- CONSTRAINT BREAKDOWN ---")
    for c in ["max_drawdown","turnover","max_country_weight","max_asset_weight","cost_stress_2x","deflated_sharpe"]:
        col = f"constraint_{c}_passed"
        if col in df.columns:
            p = df[col].sum()
            print(f"  {c}: {p}/{len(df)} passed ({p/len(df)*100:.1f}%)")
    print()

    print("Results saved to:", output_dir / "constrained_results_v1_5.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
