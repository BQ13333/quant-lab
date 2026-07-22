from __future__ import annotations

import argparse

from quant_lab.agents.institutional_alpha_factory import (
    DEFAULT_DATA_QUALITY_DIR,
    DEFAULT_L2_ARTIFACT_DIR,
    DEFAULT_PIT_QA_ARTIFACT_DIR,
    InstitutionalAlphaFactoryConfig,
    run_institutional_alpha_factory,
)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_float_csv(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 2 institutional alpha factory on the accepted MYS/SGP/IDN U2 data lake."
    )
    parser.add_argument("--data-quality-dir", default=DEFAULT_DATA_QUALITY_DIR)
    parser.add_argument("--l2-artifact-dir", default=DEFAULT_L2_ARTIFACT_DIR)
    parser.add_argument("--pit-qa-artifact-dir", default=DEFAULT_PIT_QA_ARTIFACT_DIR)
    parser.add_argument("--artifact-root", default="artifacts/institutional_alpha_factory")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--include-countries", default="Malaysia,Singapore,Indonesia")
    parser.add_argument("--universe-tier", choices=("u2", "u3", "all"), default="u2")
    parser.add_argument("--start-date", default="2021-01-01")
    parser.add_argument("--end-date", default="2026-07-06")
    parser.add_argument("--holdout-start", default="2024-01-02")
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--skip-source-only-sweep", action="store_true")
    parser.add_argument("--source-only-max-models", type=int, default=12)
    parser.add_argument("--source-only-max-allocators", type=int, default=4)
    parser.add_argument("--pit-max-source-models", type=int, default=None)
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=63)
    parser.add_argument("--embargo", type=int, default=21)
    parser.add_argument("--cost-bps-grid", default="5,15,25")
    parser.add_argument("--target-volatility", type=float, default=0.12)
    parser.add_argument("--max-leverage", type=float, default=1.0)
    parser.add_argument("--max-country-weight", type=float, default=0.45)
    parser.add_argument("--max-asset-weight", type=float, default=0.08)
    parser.add_argument("--drawdown-limit", type=float, default=-0.25)
    parser.add_argument("--min-net-sharpe", type=float, default=3.0)
    parser.add_argument("--min-deflated-sharpe", type=float, default=0.95)
    parser.add_argument("--min-holdout-net-sharpe", type=float, default=1.0)
    parser.add_argument("--max-pbo", type=float, default=0.30)
    parser.add_argument("--candidate-weight-top-n", type=int, default=10)
    args = parser.parse_args()

    config = InstitutionalAlphaFactoryConfig(
        data_quality_dir=args.data_quality_dir,
        l2_artifact_dir=args.l2_artifact_dir,
        pit_qa_artifact_dir=args.pit_qa_artifact_dir,
        artifact_root=args.artifact_root,
        run_id=args.run_id,
        include_countries=_parse_csv(args.include_countries),
        universe_tier=args.universe_tier,
        start_date=args.start_date,
        end_date=args.end_date,
        holdout_start=args.holdout_start,
        max_iterations=args.max_iterations,
        run_source_only_sweep=not args.skip_source_only_sweep,
        source_only_max_models=args.source_only_max_models,
        source_only_max_allocators=args.source_only_max_allocators,
        pit_max_source_models=args.pit_max_source_models,
        train_size=args.train_size,
        test_size=args.test_size,
        embargo=args.embargo,
        cost_bps_grid=_parse_float_csv(args.cost_bps_grid),
        target_volatility=args.target_volatility,
        max_leverage=args.max_leverage,
        max_country_weight=args.max_country_weight,
        max_asset_weight=args.max_asset_weight,
        drawdown_limit=args.drawdown_limit,
        min_net_sharpe=args.min_net_sharpe,
        min_deflated_sharpe=args.min_deflated_sharpe,
        min_holdout_net_sharpe=args.min_holdout_net_sharpe,
        max_pbo=args.max_pbo,
        candidate_weight_top_n=args.candidate_weight_top_n,
    )
    result = run_institutional_alpha_factory(config)

    print("=" * 88)
    print("INSTITUTIONAL ALPHA FACTORY PASS COMPLETE")
    print("=" * 88)
    print(f"Run ID:             {result.run_id}")
    print(f"Artifacts:          {result.artifact_dir}")
    print(f"Candidates:         {len(result.combined_ledger)}")
    print(f"Promoted:           {len(result.promoted)}")
    print(f"Mutation queue:     {len(result.mutation_queue)}")
    if not result.combined_ledger.empty:
        cols = [
            "candidate_id",
            "alpha_lane",
            "net_ir",
            "net_sharpe",
            "deflated_sharpe",
            "holdout_net_sharpe",
            "passed_all",
            "institutional_gate_pass",
            "promotion_state",
        ]
        print(result.combined_ledger[[col for col in cols if col in result.combined_ledger.columns]].head(20).to_string(index=False))
    print(f"Summary:            {result.artifact_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
