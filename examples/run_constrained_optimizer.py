from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_lab.models.baselines import LowVolatilityModel, MomentumModel, VolatilityAdjustedMomentumModel
from quant_lab.models.zoo import (
    BreakoutModel,
    CountryRelativeMomentumModel,
    CrossCountryRankingModel,
    TimeSeriesMomentumModel,
    TrendLowVolEnsembleModel,
    WithinCountryRankingModel,
)
from quant_lab.optimization.net_ir_optimizer import NetIRConstrainedOptimizer
from quant_lab.research.asean import _equal_weight_benchmark, _metadata_from_artifact
from quant_lab.validation.gates import ValidationGate


def main() -> None:
    artifact_dir = Path("artifacts/asean/ASEAN-20260704-232302")
    prices_path = artifact_dir / "eligible_stock_usd_prices.parquet"
    eligible_prices = pd.read_parquet(prices_path)
    metadata = _metadata_from_artifact(artifact_dir)
    group_map = metadata.set_index("ticker")["country"].reindex(eligible_prices.columns)
    benchmark_returns = _equal_weight_benchmark(eligible_prices)

    # Pre-select top-performing models from the upgraded run
    models = [
        # Low volatility family (best performers)
        LowVolatilityModel(name="lowvol_63", lookback=63),
        LowVolatilityModel(name="lowvol_126", lookback=126),
        LowVolatilityModel(name="lowvol_252", lookback=252),
        # Momentum family
        MomentumModel(name="mom_63", lookback=63),
        MomentumModel(name="mom_126", lookback=126),
        MomentumModel(name="mom_252", lookback=252),
        # Volatility-adjusted momentum
        VolatilityAdjustedMomentumModel(name="vam_63", lookback=63, volatility_lookback=63),
        VolatilityAdjustedMomentumModel(name="vam_126", lookback=126, volatility_lookback=63),
        # Time-series momentum
        TimeSeriesMomentumModel(name="tsmom_63", lookback=63),
        # Trend + low vol ensemble
        TrendLowVolEnsembleModel(
            name="trend_lowvol_ensemble",
            trend_model=MomentumModel(name="trend_mom_63", lookback=63),
            lowvol_model=LowVolatilityModel(name="trend_lowvol_63", lookback=63),
        ),
        # Breakout
        BreakoutModel(name="breakout_20", lookback=20),
    ]

    # Add group-aware models
    models.extend([
        CountryRelativeMomentumModel(name="ctry_rel_mom_63", lookback=63, groups=group_map),
        WithinCountryRankingModel(name="within_ctry_63", lookback=63, groups=group_map),
        CrossCountryRankingModel(name="cross_ctry_63", lookback=63, groups=group_map),
    ])

    allocator_configs = [
        {"weighting": "long_only", "top_fraction": 0.10, "rebalance_frequency": "D"},
        {"weighting": "long_only", "top_fraction": 0.10, "rebalance_frequency": "M"},
        {"weighting": "long_only", "top_fraction": 0.20, "rebalance_frequency": "D"},
        {"weighting": "long_only", "top_fraction": 0.20, "rebalance_frequency": "M"},
        {"weighting": "long_only", "top_fraction": 0.33, "rebalance_frequency": "D"},
        {"weighting": "long_only", "top_fraction": 0.33, "rebalance_frequency": "M"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.10, "rebalance_frequency": "D"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.10, "rebalance_frequency": "M"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.20, "rebalance_frequency": "D"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.20, "rebalance_frequency": "M"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.33, "rebalance_frequency": "D"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.33, "rebalance_frequency": "M"},
        {"weighting": "dollar_neutral", "top_fraction": 1.0, "rebalance_frequency": "D"},
        {"weighting": "group_neutral_dollar_neutral", "top_fraction": 1.0, "rebalance_frequency": "D"},
    ]

    cost_bps_grid = (0.0, 5.0, 10.0, 15.0, 25.0)

    print("=" * 70)
    print("NET IR CONSTRAINED OPTIMIZATION")
    print("=" * 70)
    print(f"Models: {len(models)}")
    print(f"Allocator configs: {len(allocator_configs)}")
    print(f"Cost levels: {cost_bps_grid}")
    print(f"Total evaluations: {len(models) * len(allocator_configs) * len(cost_bps_grid)}")
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

    df = optimizer.optimize(
        models=models,
        allocator_configs=allocator_configs,
        cost_bps_grid=cost_bps_grid,
        prices=eligible_prices,
        group_map=group_map,
        benchmark_returns=benchmark_returns,
    )

    output_dir = Path("artifacts/asean/constrained")
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "constrained_results.csv", index=False)

    print(f"Evaluations completed: {len(df)}")
    print()

    # Top 20 by net IR
    print("--- TOP 20 BY NET INFORMATION RATIO ---")
    top20 = df.head(20)
    display_cols = [
        "model_name", "allocator_name", "cost_bps", "net_ir", "net_sharpe",
        "deflated_sharpe", "annual_return", "max_drawdown", "turnover",
        "passed_all", "score",
    ]
    print(top20[[c for c in display_cols if c in top20.columns]].to_string(index=False))
    print()

    # Feasible solutions only
    feasible = df[df["passed_all"] == True]
    print(f"--- FEASIBLE SOLUTIONS: {len(feasible)} of {len(df)} ---")
    if not feasible.empty:
        print(feasible.head(20)[[c for c in display_cols if c in feasible.columns]].to_string(index=False))
    else:
        print("No feasible solutions found.")
    print()

    # Best by cost level
    print("--- BEST BY COST LEVEL ---")
    for cost in cost_bps_grid:
        subset = df[df["cost_bps"] == cost].head(1)
        if not subset.empty:
            print(f"Cost {cost:.0f} bps: {subset.iloc[0]['model_name']} | "
                  f"IR={subset.iloc[0]['net_ir']:.3f} | "
                  f"Sharpe={subset.iloc[0]['net_sharpe']:.3f} | "
                  f"DSR={subset.iloc[0]['deflated_sharpe']:.3f} | "
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

    # Compare to equal-weight benchmark
    print("--- BENCHMARK COMPARISON ---")
    print(f"Equal-weight Sharpe: {benchmark_returns.mean() * 252 / (benchmark_returns.std(ddof=1) * 252**0.5):.3f}")
    print(f"Equal-weight Ann Return: {benchmark_returns.mean() * 252:.1%}")
    print()

    print("=" * 70)
    print("Results saved to:", output_dir / "constrained_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
