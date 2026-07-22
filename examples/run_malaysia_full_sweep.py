from __future__ import annotations

import argparse

from quant_lab.research.malaysia_sweep import run_malaysia_full_sweep_from_asean_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Malaysia full strategy sweep.")
    parser.add_argument(
        "--source-artifact",
        default="artifacts/asean/ASEAN-20260704-232302",
        help="ASEAN artifact directory containing daily_usd_prices.parquet and universe.csv.",
    )
    parser.add_argument("--artifact-root", default="artifacts/malaysia")
    parser.add_argument("--baseline-cost-bps", type=float, default=15.0)
    parser.add_argument("--max-asset-weight", type=float, default=0.25)
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.80)
    args = parser.parse_args()

    result = run_malaysia_full_sweep_from_asean_artifact(
        asean_artifact_dir=args.source_artifact,
        artifact_root=args.artifact_root,
        baseline_cost_bps=args.baseline_cost_bps,
        max_asset_weight_limit=args.max_asset_weight,
        min_deflated_sharpe=args.min_deflated_sharpe,
    )

    print("=" * 80)
    print("MALAYSIA FULL STRATEGY SWEEP COMPLETE")
    print("=" * 80)
    print(f"Run ID:               {result.run_id}")
    print(f"Artifacts:            {result.artifact_dir}")
    print(f"Candidates:           {len(result.all_candidates)}")
    print(f"Feasible candidates:  {len(result.feasible_candidates)}")
    print()

    if not result.champions.empty:
        display_cols = [
            "model_name",
            "family",
            "allocator_name",
            "cost_bps",
            "net_ir",
            "net_sharpe",
            "deflated_sharpe",
            "annual_return",
            "max_drawdown",
            "turnover",
            "passed_all",
            "score",
        ]
        print(result.champions[[c for c in display_cols if c in result.champions.columns]].head(20).to_string(index=False))

    print()
    print("Obsidian notes:")
    for path in result.note_paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
