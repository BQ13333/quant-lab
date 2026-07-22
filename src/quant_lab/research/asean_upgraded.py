from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_lab.data.asean_universe import default_asean_universe
from quant_lab.models.baselines import (
    LowVolatilityModel,
    MeanReversionModel,
    MomentumModel,
    VolatilityAdjustedMomentumModel,
)
from quant_lab.models.zoo import (
    BreakoutModel,
    CountryRelativeMomentumModel,
    CrossCountryRankingModel,
    LowVolatilityModel126,
    LowVolatilityModel252,
    MeanReversionModel21,
    MeanReversionModel63,
    MomentumModel126,
    MomentumModel21,
    MomentumModel252,
    TimeSeriesMomentumModel,
    TrendLowVolEnsembleModel,
    VolatilityAdjustedMomentumModel126,
    VolatilityAdjustedMomentumModelVol21,
    WithinCountryRankingModel,
)
from quant_lab.notes.obsidian import ObsidianNote, write_note
from quant_lab.portfolio.costs import country_cost_schedule
from quant_lab.research.asean import (
    _equal_country_benchmark,
    _equal_weight_benchmark,
    _metadata,
    _metadata_from_artifact,
)
from quant_lab.research.diagnostics import (
    generate_cost_sensitivity,
    generate_diagnostics_summary,
    generate_model_cards,
)
from quant_lab.research.tournament import ModelTournament
from quant_lab.validation.gates import ValidationGate
from quant_lab.validation.metrics import max_drawdown, sharpe_ratio


def _upgraded_model_suite(prefix: str = "asean", groups: pd.Series | None = None) -> list:
    """Build expanded model zoo for ASEAN upgraded research."""
    models = [
        # Low volatility
        LowVolatilityModel(name=f"{prefix}_lowvol_63", lookback=63),
        LowVolatilityModel126(name=f"{prefix}_lowvol_126", lookback=126),
        LowVolatilityModel252(name=f"{prefix}_lowvol_252", lookback=252),
        # Momentum
        MomentumModel21(name=f"{prefix}_mom_21", lookback=21),
        MomentumModel(name=f"{prefix}_mom_63", lookback=63),
        MomentumModel126(name=f"{prefix}_mom_126", lookback=126),
        MomentumModel252(name=f"{prefix}_mom_252", lookback=252),
        # Volatility-adjusted momentum
        VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_63", lookback=63, volatility_lookback=63),
        VolatilityAdjustedMomentumModel126(name=f"{prefix}_vam_126", lookback=126, volatility_lookback=63),
        VolatilityAdjustedMomentumModelVol21(name=f"{prefix}_vam_vol21", lookback=63, volatility_lookback=21),
        # Mean reversion
        MeanReversionModel(name=f"{prefix}_rev_5", lookback=5),
        MeanReversionModel21(name=f"{prefix}_rev_21", lookback=21),
        MeanReversionModel63(name=f"{prefix}_rev_63", lookback=63),
        # Time-series momentum
        TimeSeriesMomentumModel(name=f"{prefix}_tsmom_63", lookback=63),
        TimeSeriesMomentumModel(name=f"{prefix}_tsmom_126", lookback=126, scale_by_vol=True),
        # Breakout
        BreakoutModel(name=f"{prefix}_breakout_20", lookback=20),
        # Trend + low vol ensemble
        TrendLowVolEnsembleModel(
            name=f"{prefix}_trend_lowvol_ensemble",
            trend_model=MomentumModel(name=f"{prefix}_trend_mom_63", lookback=63),
            lowvol_model=LowVolatilityModel(name=f"{prefix}_trend_lowvol_63", lookback=63),
        ),
    ]

    # Add group-aware models if country mapping is available
    if groups is not None and not groups.empty:
        models.extend([
            CountryRelativeMomentumModel(name=f"{prefix}_ctry_rel_mom_63", lookback=63, groups=groups),
            WithinCountryRankingModel(name=f"{prefix}_within_ctry_63", lookback=63, groups=groups),
            CrossCountryRankingModel(name=f"{prefix}_cross_ctry_63", lookback=63, groups=groups),
        ])

    return models


def _benchmark_row(name: str, returns: pd.Series) -> dict[str, float | str]:
    return {
        "model_name": name,
        "sharpe": sharpe_ratio(returns),
        "sortino": float("nan"),
        "calmar": float("nan"),
        "max_drawdown": max_drawdown(returns),
        "annual_return": returns.mean() * 252,
        "annual_volatility": returns.std(ddof=1) * (252**0.5),
        "turnover": 0.0,
        "deflated_sharpe": float("nan"),
        "annual_excess_return": float("nan"),
        "tracking_error": float("nan"),
        "information_ratio": float("nan"),
        "splits": 0,
    }


def _run_upgraded_tournaments(
    eligible_prices: pd.DataFrame,
    group_map: pd.Series,
    benchmark_returns: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    """Run expanded model suite across multiple allocator configurations."""
    tournament_specs = [
        ("dn", "dollar_neutral", None, 0.33),
        ("long_top33", "long_only", None, 0.33),
        ("long_top20", "long_only", None, 0.20),
        ("long_top10", "long_only", None, 0.10),
        ("ctry_dn", "group_neutral_dollar_neutral", group_map, 0.33),
        ("ctry_long_top33", "group_neutral_long_only", group_map, 0.33),
        ("ctry_long_top20", "group_neutral_long_only", group_map, 0.20),
        ("ctry_long_top10", "group_neutral_long_only", group_map, 0.10),
    ]
    frames = []
    for label, weighting, groups, top_fraction in tournament_specs:
        models = _upgraded_model_suite(prefix=f"asean_{label}", groups=groups)
        tournament = ModelTournament(
            models=models,
            target_volatility=0.12,
            cost_bps=cost_bps,
            weighting=weighting,
            group_map=groups,
            top_fraction=top_fraction,
        )
        frame = tournament.run(
            eligible_prices,
            train_size=504,
            test_size=63,
            embargo=5,
            benchmark_returns=benchmark_returns,
        )
        if not frame.empty:
            frame["weighting"] = weighting
            frame["top_fraction"] = top_fraction
            frame["cost_bps"] = cost_bps
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_asean_upgraded_research(
    artifact_dir: str | Path,
    output_dir: str | Path | None = None,
    vault_root: str | Path = "vault",
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 25.0, 50.0),
) -> dict[str, Any]:
    """Run upgraded ASEAN research using an existing artifact's data.

    This avoids re-downloading data and focuses on the expanded model zoo,
    portfolio configurations, cost sensitivity, and diagnostics.
    """
    artifact_path = Path(artifact_dir)
    prices_path = artifact_path / "eligible_stock_usd_prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"Missing eligible prices: {prices_path}")

    eligible_prices = pd.read_parquet(prices_path)
    metadata = _metadata_from_artifact(artifact_path)
    group_map = metadata.set_index("ticker")["country"].reindex(eligible_prices.columns)
    benchmark_returns = _equal_weight_benchmark(eligible_prices)
    equal_country_returns = _equal_country_benchmark(eligible_prices, group_map)

    run_id = datetime.now(timezone.utc).strftime("ASEAN-UPGRADED-%Y%m%d-%H%M%S")
    output_path = Path(output_dir) if output_dir else artifact_path / "upgraded"
    output_path.mkdir(parents=True, exist_ok=True)
    vault_path = Path(vault_root)

    # ------------------------------------------------------------------
    # 1. Run tournaments at multiple cost levels
    # ------------------------------------------------------------------
    cost_frames = []
    for cost_bps in cost_bps_grid:
        frame = _run_upgraded_tournaments(
            eligible_prices=eligible_prices,
            group_map=group_map,
            benchmark_returns=benchmark_returns,
            cost_bps=cost_bps,
        )
        if not frame.empty:
            cost_frames.append(frame)

    cost_sensitivity = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
    if not cost_sensitivity.empty:
        cost_sensitivity = cost_sensitivity.sort_values(
            ["cost_bps", "deflated_sharpe", "sharpe"], ascending=[True, False, False]
        )
        best_by_cost = cost_sensitivity.groupby("cost_bps", as_index=False).head(5)
    else:
        best_by_cost = pd.DataFrame()

    cost_sensitivity.to_csv(output_path / "cost_sensitivity.csv", index=False)
    best_by_cost.to_csv(output_path / "best_by_cost.csv", index=False)

    # ------------------------------------------------------------------
    # 2. Build leaderboard at baseline cost (15 bps)
    # ------------------------------------------------------------------
    baseline_cost = 15.0
    baseline_frame = cost_sensitivity.loc[cost_sensitivity["cost_bps"] == baseline_cost].copy()
    if not baseline_frame.empty:
        baseline_frame = baseline_frame.sort_values(["deflated_sharpe", "sharpe"], ascending=False)

    leaderboard = pd.concat(
        [
            baseline_frame,
            pd.DataFrame(
                [
                    _benchmark_row("asean_equal_weight_usd", benchmark_returns),
                    _benchmark_row("asean_equal_country_usd", equal_country_returns),
                ]
            ),
        ]
    ).reset_index(drop=True)
    leaderboard.to_csv(output_path / "leaderboard.csv", index=False)

    # ------------------------------------------------------------------
    # 3. Diagnostics
    # ------------------------------------------------------------------
    diagnostics = {}
    if not leaderboard.empty:
        diag_summary = generate_diagnostics_summary(
            leaderboard=leaderboard,
            save_dir=output_path,
        )
        diagnostics["diagnostics_summary"] = diag_summary

    # ------------------------------------------------------------------
    # 4. Promotion gate assessment
    # ------------------------------------------------------------------
    gate = ValidationGate()
    promoted_models = []
    failed_models = []
    failed_reasons = []
    gaps = []

    if not baseline_frame.empty:
        for _, row in baseline_frame.head(20).iterrows():
            promote_row = row.to_dict()
            decision = gate.promote(promote_row)
            if decision.passed:
                promoted_models.append(row["model_name"])
            else:
                failed_models.append(row["model_name"])
                failed_reasons.append("; ".join(decision.reasons))

    if not promoted_models:
        gaps.append("No models passed promotion gates at 15 bps baseline cost.")
        gaps.append("Consider lower-cost execution, longer holding periods, or alternative data.")
    else:
        gaps.append("Continue stress testing with FX shocks and liquidity constraints.")
        gaps.append("Evaluate regime-conditioned models for drawdown control.")

    # ------------------------------------------------------------------
    # 5. Write Obsidian notes
    # ------------------------------------------------------------------
    note_paths = _write_upgraded_obsidian_notes(
        vault_root=vault_path,
        run_id=run_id,
        artifact_dir=output_path,
        leaderboard=leaderboard,
        cost_sensitivity=cost_sensitivity,
        best_by_cost=best_by_cost,
        promoted_models=promoted_models,
        failed_models=failed_models,
        failed_reasons=failed_reasons,
        gaps=gaps,
        eligible_prices=eligible_prices,
    )

    return {
        "run_id": run_id,
        "output_dir": output_path,
        "leaderboard": leaderboard,
        "cost_sensitivity": cost_sensitivity,
        "best_by_cost": best_by_cost,
        "promoted_models": promoted_models,
        "failed_models": failed_models,
        "note_paths": note_paths,
    }


def _write_upgraded_obsidian_notes(
    vault_root: Path,
    run_id: str,
    artifact_dir: Path,
    leaderboard: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    best_by_cost: pd.DataFrame,
    promoted_models: list[str],
    failed_models: list[str],
    failed_reasons: list[str],
    gaps: list[str],
    eligible_prices: pd.DataFrame,
) -> list[Path]:
    experiment_dir = Path("30_experiments") / run_id
    paths: list[Path] = []
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    artifact_link = artifact_dir.resolve()

    # Index note
    summary = [
        f"Run id: `{run_id}`",
        f"Artifact directory: `{artifact_link}`",
        f"Eligible stock universe: `{eligible_prices.shape[1]}` tickers across `{eligible_prices.shape[0]}` dates.",
        "",
        "## Best Result",
        "No leaderboard was produced." if not best else "\n".join(f"- `{key}`: `{value}`" for key, value in best.items()),
        "",
        "## Promoted Models",
        "_No models passed promotion gates._" if not promoted_models else "\n".join(f"- `{m}`" for m in promoted_models),
        "",
        "## Caveats",
        "- Current-survivor free-data universe; not a point-in-time constituent database.",
        "- Default ASEAN scope is Malaysia, Singapore, and Indonesia only.",
        "- USD conversion uses Yahoo LC-per-USD FX closes and prior-only forward fill.",
        "- Expanded model zoo includes 20+ alpha models vs. 9 in the baseline.",
        "- Cost sensitivity tested at 0, 5, 10, 15, 25, 50 bps.",
        "- Promotion gates: Sharpe > 0.8, DSR > 0.80, positive excess return & IR, survives 2x cost stress.",
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "index.md",
            ObsidianNote(
                title=f"{run_id} ASEAN Upgraded Backtest",
                note_type="experiment",
                tags=["quant-lab", "asean", "upgraded", "backtest"],
                metadata={"experiment_id": run_id, "status": "completed", "artifact_dir": str(artifact_link)},
                body="\n".join(summary),
            ),
        )
    )

    # Validation report
    validation_body = [
        "## Leaderboard",
        _markdown_table(leaderboard) if not leaderboard.empty else "No leaderboard.",
        "",
        "## Cost Sensitivity (Best By Cost)",
        _markdown_table(best_by_cost.head(40)) if not best_by_cost.empty else "No cost sensitivity data.",
        "",
        "## Promotion Gate Results",
        "### Passed",
        "_None_" if not promoted_models else "\n".join(f"- `{m}`" for m in promoted_models),
        "",
        "### Failed",
        "_None_" if not failed_models else "\n".join(
            f"- `{m}`: {reason}" for m, reason in zip(failed_models, failed_reasons)
        ),
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "validation_report.md",
            ObsidianNote(
                title=f"{run_id} Validation Report",
                note_type="validation_report",
                tags=["quant-lab", "validation", "asean", "upgraded"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(validation_body),
            ),
        )
    )

    # Cost sensitivity
    cost_body = [
        f"Artifact directory: `{artifact_link}`",
        "",
        "## Best By Cost",
        _markdown_table(best_by_cost.head(40)) if not best_by_cost.empty else "No data.",
        "",
        "## Interpretation",
        "- If the winner changes dramatically between 0 and 15 bps, the edge is friction-sensitive.",
        "- A robust candidate should retain positive active return and acceptable drawdown at stressed costs.",
        "- Compare annual_excess_return, tracking_error, and information_ratio, not only Sharpe.",
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "cost_sensitivity.md",
            ObsidianNote(
                title=f"{run_id} Cost Sensitivity",
                note_type="cost_sensitivity",
                tags=["quant-lab", "asean", "costs", "upgraded"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(cost_body),
            ),
        )
    )

    # Failure modes
    failure_body = [
        f"Run ID: `{run_id}`",
        "",
        "## Failed Models",
        "_No models failed promotion gates._" if not failed_models else "\n".join(
            f"- **{m}**: {reason}" for m, reason in zip(failed_models, failed_reasons)
        ),
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "failure_modes.md",
            ObsidianNote(
                title=f"{run_id} Failure Modes",
                note_type="failure_modes",
                tags=["quant-lab", "asean", "failure-modes"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(failure_body),
            ),
        )
    )

    # Next research queue
    queue_body = [
        f"Run ID: `{run_id}`",
        "",
        "## Passed Models",
        "_None_" if not promoted_models else "\n".join(f"- {m}" for m in promoted_models),
        "",
        "## Recommended Next Steps",
        "_None_" if not gaps else "\n".join(f"- {g}" for g in gaps),
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "next_research_queue.md",
            ObsidianNote(
                title=f"{run_id} Next Research Queue",
                note_type="research_queue",
                tags=["quant-lab", "asean", "research-queue"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(queue_body),
            ),
        )
    )

    return paths


def _markdown_table(frame: pd.DataFrame, max_rows: int = 80) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).fillna("")
    columns = list(clipped.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in clipped.iterrows():
        rows.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join([header, separator, *rows])
