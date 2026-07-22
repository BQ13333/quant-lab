from __future__ import annotations

import argparse

from quant_lab.agents.closed_loop_alpha_factory import (
    ClosedLoopAlphaFactoryConfig,
    run_closed_loop_alpha_factory,
)


DEFAULT_L2_ARTIFACT = (
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/"
    "05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the PM-owned closed-loop alpha factory source-strategy backtest."
    )
    parser.add_argument("--l2-artifact-dir", default=DEFAULT_L2_ARTIFACT)
    parser.add_argument("--artifact-root", default="artifacts/closed_loop_alpha_factory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--universe-tier", choices=("u2", "u3", "all"), default="u3")
    parser.add_argument("--include-countries", default="Malaysia,Singapore,Indonesia")
    parser.add_argument("--min-net-sharpe", type=float, default=3.0)
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.95)
    parser.add_argument("--train-size", type=int, default=756)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=21)
    parser.add_argument("--target-volatility", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--max-country-weight", type=float, default=0.45)
    parser.add_argument("--max-asset-weight", type=float, default=0.08)
    parser.add_argument("--drawdown-limit", type=float, default=-0.25)
    parser.add_argument("--min-rows", type=int, default=2500)
    parser.add_argument("--max-missing-ratio", type=float, default=0.25)
    parser.add_argument("--cost-bps-grid", default="15,30,50")
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Optional bounded first pass. Omit for the full source strategy suite.",
    )
    parser.add_argument(
        "--max-allocators",
        type=int,
        default=None,
        help="Optional bounded allocator grid for quick proof runs.",
    )
    parser.add_argument("--no-precompute-full-signals", action="store_true")
    parser.add_argument("--no-candidate-weights", action="store_true")
    parser.add_argument("--candidate-weight-top-n", type=int, default=10)
    args = parser.parse_args()

    countries = tuple(country.strip() for country in args.include_countries.split(",") if country.strip())
    costs = tuple(float(value.strip()) for value in args.cost_bps_grid.split(",") if value.strip())
    config = ClosedLoopAlphaFactoryConfig(
        l2_artifact_dir=args.l2_artifact_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        include_countries=countries,
        universe_tier=args.universe_tier,
        min_net_sharpe=args.min_net_sharpe,
        min_deflated_sharpe=args.min_deflated_sharpe,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
        target_volatility=args.target_volatility,
        max_leverage=args.max_leverage,
        max_country_weight=args.max_country_weight,
        max_asset_weight=args.max_asset_weight,
        drawdown_limit=args.drawdown_limit,
        min_rows=args.min_rows,
        max_missing_ratio=args.max_missing_ratio,
        cost_bps_grid=costs,
        max_models=args.max_models,
        max_allocators=args.max_allocators,
        precompute_full_signals=not args.no_precompute_full_signals,
        write_candidate_weights=not args.no_candidate_weights,
        candidate_weight_top_n=args.candidate_weight_top_n,
    )
    result = run_closed_loop_alpha_factory(config)

    print("=" * 80)
    print("CLOSED-LOOP ALPHA FACTORY PASS COMPLETE")
    print("=" * 80)
    print(f"Run ID:              {result.run_id}")
    print(f"Artifacts:           {result.artifact_dir}")
    print(f"Universe tier:       {args.universe_tier}")
    print(f"Candidates:          {len(result.candidate_ledger)}")
    print(f"PM promoted:         {len(result.pm_promoted)}")
    if not result.candidate_ledger.empty:
        display_cols = [
            "model_name",
            "family",
            "allocator_name",
            "cost_bps",
            "net_ir",
            "net_sharpe",
            "deflated_sharpe",
            "passed_all",
            "pm_gate_pass",
            "promotion_state",
        ]
        cols = [col for col in display_cols if col in result.candidate_ledger.columns]
        print(result.candidate_ledger[cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
