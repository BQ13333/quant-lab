from __future__ import annotations

from quant_lab.research.global_markets import run_global_research


def main() -> None:
    result = run_global_research(
        artifact_root="artifacts/global",
        vault_root="vault",
        period="5y",
        cost_bps_grid=(0.0, 5.0, 15.0, 50.0),
        baseline_cost_bps=5.0,
    )

    print(f"Run ID: {result.run_id}")
    print(f"Artifacts: {result.artifact_dir}")
    print(f"Leaderboard rows: {len(result.leaderboard)}")
    if not result.leaderboard.empty:
        cols = [
            "model_name",
            "sharpe",
            "deflated_sharpe",
            "annual_return",
            "max_drawdown",
            "turnover",
            "information_ratio",
            "cost_bps",
        ]
        print(result.leaderboard[[c for c in cols if c in result.leaderboard.columns]].head(10).to_string(index=False))
    print("Notes:")
    for path in result.note_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
