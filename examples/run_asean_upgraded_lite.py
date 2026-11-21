from __future__ import annotations

from pathlib import Path

from quant_lab.research.asean_upgraded import run_asean_upgraded_research


def main() -> None:
    artifact_dir = Path("artifacts/asean/ASEAN-20260704-232302")
    result = run_asean_upgraded_research(
        artifact_dir=artifact_dir,
        output_dir="artifacts/asean/upgraded",
        vault_root="vault",
        cost_bps_grid=(0.0, 15.0, 50.0),
    )

    print("=" * 70)
    print("ASEAN UPGRADED RESEARCH COMPLETE")
    print("=" * 70)
    print(f"Run ID:        {result['run_id']}")
    print(f"Output dir:    {result['output_dir']}")
    print(f"Leaderboard:   {len(result['leaderboard'])} rows")
    print(f"Cost grid:     {len(result['cost_sensitivity'])} rows")
    print(f"Promoted:      {len(result['promoted_models'])} models")
    print(f"Failed:        {len(result['failed_models'])} models")
    print()

    if not result["leaderboard"].empty:
        print("--- TOP 10 MODELS (BASELINE 15 BPS) ---")
        top10 = result["leaderboard"].head(10)
        display_cols = [
            "model_name", "sharpe", "deflated_sharpe",
            "annual_return", "max_drawdown", "turnover",
            "annual_excess_return", "information_ratio",
        ]
        print(top10[[c for c in display_cols if c in top10.columns]].to_string(index=False))
        print()

    if not result["best_by_cost"].empty:
        print("--- BEST BY COST ---")
        best_cols = ["cost_bps", "model_name", "sharpe", "deflated_sharpe", "annual_return", "max_drawdown"]
        print(result["best_by_cost"][[c for c in best_cols if c in result["best_by_cost"].columns]].head(20).to_string(index=False))
        print()

    print("--- PROMOTED MODELS ---")
    if result["promoted_models"]:
        for m in result["promoted_models"]:
            print(f"  ✓ {m}")
    else:
        print("  None")
    print()

    print("--- OBSIDIAN NOTES ---")
    for p in result["note_paths"]:
        print(f"  {p.resolve()}")
    print()

    print("=" * 70)


if __name__ == "__main__":
    main()
