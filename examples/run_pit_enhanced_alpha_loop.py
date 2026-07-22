from __future__ import annotations

import argparse

from quant_lab.research.pit_ablation import PITEnhancedAblationConfig, run_pit_enhanced_ablation


DEFAULT_L2_ARTIFACT = (
    "artifacts/codex_backtests/CODEX_ASEAN_L2_DEEP_20260707/"
    "05_l2_deep_extraction/ASEAN-L2-DEEP-MYS-SGP-IDN-20260708"
)
DEFAULT_PIT_QA_ARTIFACT = "artifacts/pit_leakage_qa/PIT-LEAKAGE-QA-MYS-SGP-IDN-FULL-U2-20260708"


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run guarded PIT-enhanced alpha ablations on ASEAN U2.")
    parser.add_argument("--l2-artifact-dir", default=DEFAULT_L2_ARTIFACT)
    parser.add_argument("--pit-qa-artifact-dir", default=DEFAULT_PIT_QA_ARTIFACT)
    parser.add_argument("--artifact-root", default="artifacts/pit_enhanced_alpha")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--universe-tier", default="u2", choices=("u2", "u3", "all"))
    parser.add_argument("--include-countries", default="Malaysia,Singapore,Indonesia")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--holdout-start", default="2024-01-02")
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=21)
    parser.add_argument("--cost-bps-grid", default="5,15,25")
    parser.add_argument("--target-volatility", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=2.0)
    parser.add_argument("--max-country-weight", type=float, default=0.45)
    parser.add_argument("--max-asset-weight", type=float, default=0.08)
    parser.add_argument("--drawdown-limit", type=float, default=-0.25)
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.95)
    parser.add_argument("--candidate-weight-top-n", type=int, default=10)
    parser.add_argument("--max-source-models", type=int, default=None)
    args = parser.parse_args()

    config = PITEnhancedAblationConfig(
        l2_artifact_dir=args.l2_artifact_dir,
        pit_qa_artifact_dir=args.pit_qa_artifact_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        include_countries=_parse_csv(args.include_countries),
        universe_tier=args.universe_tier,
        start_date=args.start_date,
        end_date=args.end_date,
        holdout_start=args.holdout_start,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
        cost_bps_grid=_parse_float_csv(args.cost_bps_grid),
        target_volatility=args.target_volatility,
        max_leverage=args.max_leverage,
        max_country_weight=args.max_country_weight,
        max_asset_weight=args.max_asset_weight,
        drawdown_limit=args.drawdown_limit,
        min_deflated_sharpe=args.min_deflated_sharpe,
        candidate_weight_top_n=args.candidate_weight_top_n,
        max_source_models=args.max_source_models,
    )
    result = run_pit_enhanced_ablation(config)

    print(f"Run ID: {result.run_id}")
    print(f"Artifact dir: {result.artifact_dir}")
    print(f"Candidates: {len(result.candidate_ledger)}")
    guarded = int(result.candidate_ledger["guarded_research_pass"].sum()) if not result.candidate_ledger.empty else 0
    print(f"Guarded research survivors: {guarded}")
    if not result.candidate_ledger.empty:
        cols = [
            "candidate_id",
            "meta_feature_set",
            "allocator_name",
            "cost_bps",
            "net_ir",
            "net_sharpe",
            "holdout_net_sharpe",
            "passed_all",
            "guarded_research_pass",
        ]
        print(result.candidate_ledger[[col for col in cols if col in result.candidate_ledger.columns]].head(20).to_string(index=False))
    print(f"Summary: {result.artifact_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
