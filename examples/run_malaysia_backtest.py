from __future__ import annotations

import argparse

from quant_lab.research.malaysia import run_malaysia_research_from_asean_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Malaysia-only research backtest from an ASEAN artifact.")
    parser.add_argument(
        "--source-artifact",
        default="artifacts/asean/ASEAN-20260704-232302",
        help="ASEAN artifact directory containing daily_usd_prices.parquet and universe.csv.",
    )
    parser.add_argument("--artifact-root", default="artifacts/malaysia")
    parser.add_argument("--baseline-cost-bps", type=float, default=15.0)
    args = parser.parse_args()

    result = run_malaysia_research_from_asean_artifact(
        asean_artifact_dir=args.source_artifact,
        artifact_root=args.artifact_root,
        baseline_cost_bps=args.baseline_cost_bps,
    )

    print("=" * 70)
    print("MALAYSIA BACKTEST COMPLETE")
    print("=" * 70)
    print(f"Run ID:             {result.run_id}")
    print(f"Artifacts:          {result.artifact_dir}")
    print(f"Eligible tickers:   {result.eligible_prices.shape[1]}")
    print(f"Dates:              {result.eligible_prices.index.min()} to {result.eligible_prices.index.max()}")
    print(f"Leaderboard rows:   {len(result.leaderboard)}")
    print()

    if not result.leaderboard.empty:
        cols = [
            "model_name",
            "sharpe",
            "deflated_sharpe",
            "annual_return",
            "max_drawdown",
            "turnover",
            "information_ratio",
            "weighting",
            "top_fraction",
            "cost_bps",
        ]
        print(result.leaderboard[[c for c in cols if c in result.leaderboard.columns]].head(20).to_string(index=False))

    print()
    print("Obsidian notes:")
    for path in result.note_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
