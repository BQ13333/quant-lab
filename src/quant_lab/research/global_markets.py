from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.global_universe import (
    GlobalInstrument,
    default_global_etf_universe,
    global_universe_metadata,
)
from quant_lab.data.yfinance_loader import download_price_history
from quant_lab.models.signal_factory import SignalFactoryConfig, build_signal_models
from quant_lab.notes.obsidian import ObsidianNote, write_note
from quant_lab.research.tournament import ModelTournament
from quant_lab.validation.gates import ValidationGate
from quant_lab.validation.metrics import max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class GlobalResearchResult:
    run_id: str
    artifact_dir: Path
    leaderboard: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    best_by_cost: pd.DataFrame
    coverage: pd.DataFrame
    note_paths: list[Path]


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("GLOBAL-%Y%m%d-%H%M%S")


def _markdown_table(frame: pd.DataFrame, max_rows: int = 60) -> str:
    if frame.empty:
        return "_No rows._"
    clipped = frame.head(max_rows).fillna("")
    columns = list(clipped.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in clipped.iterrows()
    ]
    return "\n".join([header, separator, *rows])


def _eligible_prices(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    min_rows: int,
    max_missing_ratio: float,
) -> pd.DataFrame:
    tradable = metadata.loc[metadata["tier"] != "benchmark", "ticker"].tolist()
    stock_prices = prices.reindex(columns=tradable).dropna(how="all")
    row_counts = stock_prices.notna().sum()
    missing_ratio = stock_prices.isna().mean()
    eligible = row_counts.ge(min_rows) & missing_ratio.le(max_missing_ratio)
    return stock_prices.loc[:, eligible].ffill(limit=5).dropna(how="all")


def _equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    return prices.pct_change(fill_method=None).fillna(0.0).mean(axis=1)


def _benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    if "ACWI" in prices.columns and prices["ACWI"].notna().sum() > 10:
        return prices["ACWI"].pct_change(fill_method=None).fillna(0.0)
    return _equal_weight_benchmark(prices)


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
        "weighting": "benchmark",
        "top_fraction": float("nan"),
        "cost_bps": 0.0,
    }


def _model_suite(prefix: str = "global") -> list:
    return build_signal_models(SignalFactoryConfig(prefix=prefix, profile="lite", include_group_models=False))


def _run_tournaments(
    eligible_prices: pd.DataFrame,
    group_map: pd.Series,
    benchmark_returns: pd.Series,
    cost_bps: float,
    train_size: int,
    test_size: int,
    embargo: int,
) -> pd.DataFrame:
    specs = [
        ("long_top33", "long_only", None, 0.33),
        ("long_top20", "long_only", None, 0.20),
        ("region_long_top33", "group_neutral_long_only", group_map, 0.33),
        ("region_long_top20", "group_neutral_long_only", group_map, 0.20),
        ("dn", "dollar_neutral", None, 0.33),
    ]

    frames = []
    for label, weighting, groups, top_fraction in specs:
        tournament = ModelTournament(
            models=_model_suite(prefix=f"global_{label}"),
            target_volatility=0.12,
            cost_bps=cost_bps,
            weighting=weighting,
            group_map=groups,
            top_fraction=top_fraction,
        )
        frame = tournament.run(
            eligible_prices,
            train_size=train_size,
            test_size=test_size,
            embargo=embargo,
            benchmark_returns=benchmark_returns,
        )
        if not frame.empty:
            frame["weighting"] = weighting
            frame["top_fraction"] = top_fraction
            frame["cost_bps"] = cost_bps
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def run_global_research_from_prices(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    artifact_dir: str | Path,
    vault_root: str | Path = "vault",
    run_id: str | None = None,
    coverage: pd.DataFrame | None = None,
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 15.0, 50.0),
    baseline_cost_bps: float = 5.0,
    train_size: int = 504,
    test_size: int = 63,
    embargo: int = 5,
    min_rows: int = 756,
    max_missing_ratio: float = 0.20,
) -> GlobalResearchResult:
    """Run the global market tournament from an existing USD price panel."""

    if prices.empty:
        raise ValueError("prices must not be empty")
    if metadata.empty or "ticker" not in metadata.columns:
        raise ValueError("metadata must include ticker rows")

    run_id = run_id or _latest_run_id()
    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)

    metadata = metadata.copy()
    prices = prices.sort_index()
    prices.to_parquet(artifact_path / "daily_usd_prices.parquet")
    metadata.to_csv(artifact_path / "universe.csv", index=False)

    if coverage is None:
        coverage = pd.DataFrame(
            {
                "ticker": prices.columns,
                "rows": [int(prices[column].dropna().shape[0]) for column in prices.columns],
                "start": [prices[column].dropna().index.min() for column in prices.columns],
                "end": [prices[column].dropna().index.max() for column in prices.columns],
                "interval": "1d",
                "period_used": "provided",
            }
        )
    coverage.to_csv(artifact_path / "coverage.csv", index=False)

    eligible_prices = _eligible_prices(prices, metadata, min_rows=min_rows, max_missing_ratio=max_missing_ratio)
    eligible_prices.to_parquet(artifact_path / "eligible_usd_prices.parquet")

    if eligible_prices.shape[1] < 5:
        cost_sensitivity = pd.DataFrame()
        best_by_cost = pd.DataFrame()
        leaderboard = pd.DataFrame()
    else:
        group_map = metadata.set_index("ticker")["region"].reindex(eligible_prices.columns)
        benchmark = _benchmark_returns(prices.reindex(eligible_prices.index))
        cost_frames = []
        for cost_bps in cost_bps_grid:
            frame = _run_tournaments(
                eligible_prices=eligible_prices,
                group_map=group_map,
                benchmark_returns=benchmark,
                cost_bps=cost_bps,
                train_size=train_size,
                test_size=test_size,
                embargo=embargo,
            )
            if not frame.empty:
                cost_frames.append(frame)

        cost_sensitivity = pd.concat(cost_frames, ignore_index=True) if cost_frames else pd.DataFrame()
        if cost_sensitivity.empty:
            best_by_cost = pd.DataFrame()
            baseline_frame = pd.DataFrame()
        else:
            cost_sensitivity = cost_sensitivity.sort_values(
                ["cost_bps", "deflated_sharpe", "sharpe"],
                ascending=[True, False, False],
            )
            best_by_cost = cost_sensitivity.groupby("cost_bps", as_index=False).head(5)
            baseline_frame = cost_sensitivity.loc[cost_sensitivity["cost_bps"] == baseline_cost_bps].copy()
            baseline_frame = baseline_frame.sort_values(["deflated_sharpe", "sharpe"], ascending=False)

        benchmark_rows = [
            _benchmark_row("global_equal_weight_usd", _equal_weight_benchmark(eligible_prices)),
            _benchmark_row("global_acwi_or_equal_weight", benchmark),
        ]
        leaderboard = pd.concat([baseline_frame, pd.DataFrame(benchmark_rows)], ignore_index=True)

    cost_sensitivity.to_csv(artifact_path / "cost_sensitivity.csv", index=False)
    best_by_cost.to_csv(artifact_path / "best_by_cost.csv", index=False)
    leaderboard.to_csv(artifact_path / "leaderboard.csv", index=False)

    note_paths = _write_global_notes(
        vault_root=Path(vault_root),
        run_id=run_id,
        artifact_dir=artifact_path,
        leaderboard=leaderboard,
        cost_sensitivity=cost_sensitivity,
        best_by_cost=best_by_cost,
        coverage=coverage,
        eligible_prices=eligible_prices,
        baseline_cost_bps=baseline_cost_bps,
    )

    return GlobalResearchResult(
        run_id=run_id,
        artifact_dir=artifact_path,
        leaderboard=leaderboard,
        cost_sensitivity=cost_sensitivity,
        best_by_cost=best_by_cost,
        coverage=coverage,
        note_paths=note_paths,
    )


def run_global_research(
    artifact_root: str | Path = "artifacts/global",
    vault_root: str | Path = "vault",
    period: str = "10y",
    interval: str = "1d",
    instruments: list[GlobalInstrument] | None = None,
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 15.0, 50.0),
    baseline_cost_bps: float = 5.0,
) -> GlobalResearchResult:
    """Download USD ETF proxies and run the global market tournament."""

    run_id = _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    universe = instruments or default_global_etf_universe()
    metadata = global_universe_metadata(universe)
    tickers = metadata["ticker"].tolist()

    downloaded = download_price_history(
        tickers=tickers,
        period=period,
        interval=interval,
        output_path=artifact_dir / "daily_usd_prices.parquet",
        fallback_periods=["7y", "5y", "3y", "1y"],
        ohlcv=True,
    )

    return run_global_research_from_prices(
        prices=downloaded.prices,
        metadata=metadata,
        artifact_dir=artifact_dir,
        vault_root=vault_root,
        run_id=run_id,
        coverage=downloaded.coverage,
        cost_bps_grid=cost_bps_grid,
        baseline_cost_bps=baseline_cost_bps,
    )


def _write_global_notes(
    vault_root: Path,
    run_id: str,
    artifact_dir: Path,
    leaderboard: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    best_by_cost: pd.DataFrame,
    coverage: pd.DataFrame,
    eligible_prices: pd.DataFrame,
    baseline_cost_bps: float,
) -> list[Path]:
    experiment_dir = Path("30_experiments") / run_id
    artifact_link = artifact_dir.resolve()
    paths: list[Path] = []
    gate = ValidationGate()

    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    promoted = []
    failed = []
    if not leaderboard.empty:
        candidates = leaderboard[leaderboard.get("weighting", "") != "benchmark"].head(20)
        for _, row in candidates.iterrows():
            decision = gate.promote(row.to_dict())
            if decision.passed:
                promoted.append(str(row["model_name"]))
            else:
                failed.append((str(row["model_name"]), "; ".join(decision.reasons)))

    index_body = [
        f"Run id: `{run_id}`",
        f"Artifact directory: `{artifact_link}`",
        f"Eligible global ETF universe: `{eligible_prices.shape[1]}` tickers across `{eligible_prices.shape[0]}` dates.",
        f"Baseline cost: `{baseline_cost_bps}` bps.",
        "",
        "## Best Baseline Result",
        "No leaderboard was produced." if not best else "\n".join(f"- `{key}`: `{value}`" for key, value in best.items()),
        "",
        "## Promotion Status",
        "_No models passed promotion gates._" if not promoted else "\n".join(f"- `{name}`" for name in promoted),
        "",
        "## Caveats",
        "- This is a USD ETF proxy universe, not direct local-exchange single-stock coverage.",
        "- Free-data history is not a point-in-time constituent database.",
        "- Live trading remains blocked; use this only for research and paper-trading candidates.",
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "index.md",
            ObsidianNote(
                title=f"{run_id} Global Market Backtest",
                note_type="experiment",
                tags=["quant-lab", "global", "backtest", "etf-proxy"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(index_body),
            ),
        )
    )

    validation_body = [
        "## Leaderboard",
        _markdown_table(leaderboard),
        "",
        "## Best By Cost",
        _markdown_table(best_by_cost),
        "",
        "## Failed Promotion Reasons",
        "_None_" if not failed else "\n".join(f"- **{name}**: {reason}" for name, reason in failed),
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "validation_report.md",
            ObsidianNote(
                title=f"{run_id} Global Validation Report",
                note_type="validation_report",
                tags=["quant-lab", "global", "validation"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(validation_body),
            ),
        )
    )

    coverage_body = [
        "## Coverage",
        _markdown_table(coverage, max_rows=100),
        "",
        "## Cost Sensitivity Sample",
        _markdown_table(cost_sensitivity.head(40)) if not cost_sensitivity.empty else "_No rows._",
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "coverage_and_costs.md",
            ObsidianNote(
                title=f"{run_id} Global Coverage And Costs",
                note_type="data_quality",
                tags=["quant-lab", "global", "coverage", "costs"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(coverage_body),
            ),
        )
    )

    return paths
