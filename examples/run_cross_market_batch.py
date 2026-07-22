from __future__ import annotations

import argparse
from dataclasses import replace

from quant_lab.research.cross_market_batch import (
    ASIA_L1_COUNTRIES,
    default_asia_l1_sweep_specs,
    run_cross_market_batch,
    select_sweep_specs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run sequential cross-market sweep batches and write one consolidated scoreboard."
    )
    parser.add_argument("long_ohlcv_root", help="Path to prices_long/<run_id>/ohlcv.")
    parser.add_argument("--artifact-root", default="artifacts/cross_market_batches")
    parser.add_argument("--batch-id", default=None)
    parser.add_argument(
        "--include-countries",
        default=",".join(ASIA_L1_COUNTRIES),
        help="Comma-separated country allowlist. Defaults to ASEAN + broader Asia.",
    )
    parser.add_argument(
        "--strategy-suite",
        choices=("core", "source"),
        default="core",
        help="Use the compact core zoo or the source-backed book/markdown strategy suite.",
    )
    parser.add_argument(
        "--spec-ids",
        default=None,
        help="Comma-separated sweep spec ids for bounded smoke runs or targeted reruns.",
    )
    parser.add_argument("--max-specs", type=int, default=None, help="Run only the first N selected sweep specs.")
    parser.add_argument(
        "--drawdown-limit",
        type=float,
        default=None,
        help="Override the max-drawdown gate for all selected specs, for example -0.35 for EM-realistic stress.",
    )
    parser.add_argument(
        "--target-volatility",
        type=float,
        default=None,
        help="Override the annualized target volatility for all selected specs.",
    )
    parser.add_argument(
        "--max-leverage",
        type=float,
        default=None,
        help="Override the volatility-targeting leverage cap for all selected specs.",
    )
    args = parser.parse_args()

    include_countries = [country.strip() for country in args.include_countries.split(",") if country.strip()]
    requested_spec_ids = [spec_id.strip() for spec_id in args.spec_ids.split(",") if spec_id.strip()] if args.spec_ids else None
    specs = select_sweep_specs(default_asia_l1_sweep_specs(), spec_ids=requested_spec_ids, max_specs=args.max_specs)
    if args.drawdown_limit is not None:
        specs = [replace(spec, drawdown_limit=args.drawdown_limit) for spec in specs]
    if args.target_volatility is not None:
        specs = [replace(spec, target_volatility=args.target_volatility) for spec in specs]
    if args.max_leverage is not None:
        specs = [replace(spec, max_leverage=args.max_leverage) for spec in specs]
    result = run_cross_market_batch(
        long_ohlcv_root=args.long_ohlcv_root,
        artifact_root=args.artifact_root,
        batch_id=args.batch_id,
        specs=specs,
        include_countries=include_countries,
        strategy_suite=args.strategy_suite,
    )

    print("=" * 80)
    print("CROSS-MARKET BATCH COMPLETE")
    print("=" * 80)
    print(f"Batch ID:            {result.batch_id}")
    print(f"Artifacts:           {result.artifact_dir}")
    print(f"Strategy suite:      {args.strategy_suite}")
    print(f"Candidates:          {len(result.scoreboard)}")
    print(
        "Feasible candidates: "
        f"{int(result.scoreboard['passed_all'].sum()) if not result.scoreboard.empty else 0}"
    )
    print(f"Failed runs:         {len(result.failures)}")

    if not result.scoreboard.empty:
        display_cols = [
            "batch_spec_id",
            "model_name",
            "family",
            "allocator_name",
            "cost_bps",
            "price_currency",
            "net_ir",
            "net_sharpe",
            "deflated_sharpe",
            "max_drawdown",
            "turnover",
            "passed_all",
            "score",
        ]
        cols = [col for col in display_cols if col in result.scoreboard.columns]
        print(result.scoreboard[cols].head(25).to_string(index=False))


if __name__ == "__main__":
    main()
