from __future__ import annotations

import argparse

from quant_lab.research.cross_market import run_cross_market_sweep_from_long_ohlcv


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact cross-market sweep from long OHLCV data-lake partitions.")
    parser.add_argument("long_ohlcv_root", help="Path to prices_long/<run_id>/ohlcv.")
    parser.add_argument("--artifact-root", default="artifacts/cross_market")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--local-currency", action="store_true", help="Skip FX conversion and evaluate local-currency prices.")
    parser.add_argument("--fx-period", default="5y")
    parser.add_argument("--min-rows", type=int, default=1000)
    parser.add_argument("--max-missing-ratio", type=float, default=0.15)
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=5)
    parser.add_argument("--target-volatility", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=3.0)
    parser.add_argument("--max-country-weight", type=float, default=0.35)
    parser.add_argument("--max-asset-weight", type=float, default=0.10)
    parser.add_argument("--drawdown-limit", type=float, default=-0.20)
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.80)
    parser.add_argument("--write-candidate-weights", action="store_true")
    parser.add_argument("--candidate-weight-top-n", type=int, default=None)
    parser.add_argument(
        "--include-countries",
        default="",
        help="Comma-separated country allowlist, for example 'Malaysia,Singapore,Indonesia'.",
    )
    parser.add_argument(
        "--exclude-countries",
        default="",
        help="Comma-separated country denylist, for example 'United States'.",
    )
    args = parser.parse_args()

    include_countries = [country.strip() for country in args.include_countries.split(",") if country.strip()]
    exclude_countries = [country.strip() for country in args.exclude_countries.split(",") if country.strip()]

    result = run_cross_market_sweep_from_long_ohlcv(
        long_ohlcv_root=args.long_ohlcv_root,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        convert_to_usd=not args.local_currency,
        fx_period=args.fx_period,
        min_rows=args.min_rows,
        max_missing_ratio=args.max_missing_ratio,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
        target_volatility=args.target_volatility,
        max_leverage=args.max_leverage,
        max_country_weight_limit=args.max_country_weight,
        max_asset_weight_limit=args.max_asset_weight,
        drawdown_limit=args.drawdown_limit,
        min_deflated_sharpe=args.min_deflated_sharpe,
        write_candidate_weights=args.write_candidate_weights,
        candidate_weight_top_n=args.candidate_weight_top_n,
        include_countries=include_countries,
        exclude_countries=exclude_countries,
    )

    print("=" * 80)
    print("CROSS-MARKET SWEEP COMPLETE")
    print("=" * 80)
    print(f"Run ID:              {result.run_id}")
    print(f"Artifacts:           {result.artifact_dir}")
    print(f"Eligible tickers:    {result.prices.shape[1]}")
    print(f"Rows:                {result.prices.shape[0]}")
    print(f"Candidates:          {len(result.all_candidates)}")
    print(f"Feasible candidates: {len(result.feasible_candidates)}")
    print(f"FX tickers:          {', '.join(result.fx_tickers) if result.fx_tickers else 'none'}")
    if not result.champions.empty:
        display_cols = [
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
            "score",
        ]
        print(result.champions[[col for col in display_cols if col in result.champions.columns]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
