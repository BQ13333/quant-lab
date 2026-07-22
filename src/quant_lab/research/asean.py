from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.data.asean_universe import Instrument, default_asean_universe
from quant_lab.data.fx import convert_local_prices_to_usd, fx_tickers_for
from quant_lab.data.yfinance_loader import DownloadResult, download_price_history
from quant_lab.models.baselines import LowVolatilityModel, MeanReversionModel, MomentumModel, VolatilityAdjustedMomentumModel
from quant_lab.notes.obsidian import ObsidianNote, write_note
from quant_lab.research.tournament import ModelTournament
from quant_lab.validation.metrics import max_drawdown, sharpe_ratio


@dataclass(frozen=True)
class AseanResearchResult:
    run_id: str
    artifact_dir: Path
    leaderboard: pd.DataFrame
    coverage: pd.DataFrame
    note_paths: list[Path]


@dataclass(frozen=True)
class AseanArtifactAnalysisResult:
    artifact_dir: Path
    cost_sensitivity: pd.DataFrame
    best_by_cost: pd.DataFrame


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("ASEAN-%Y%m%d-%H%M%S")


def _metadata(instruments: list[Instrument]) -> pd.DataFrame:
    return pd.DataFrame([instrument.__dict__ for instrument in instruments])


def _metadata_from_artifact(artifact_dir: Path) -> pd.DataFrame:
    universe_path = artifact_dir / "universe.csv"
    if universe_path.exists():
        return pd.read_csv(universe_path)
    return _metadata(default_asean_universe())


def _eligible_prices(
    prices_usd: pd.DataFrame,
    metadata: pd.DataFrame,
    min_rows: int,
    max_missing_ratio: float,
) -> pd.DataFrame:
    stock_tickers = metadata.loc[metadata["tier"] == "stock", "ticker"].tolist()
    stock_prices = prices_usd.reindex(columns=stock_tickers)
    stock_prices = stock_prices.dropna(how="all")
    row_counts = stock_prices.notna().sum()
    missing_ratio = stock_prices.isna().mean()
    eligible = row_counts.ge(min_rows) & missing_ratio.le(max_missing_ratio)
    return stock_prices.loc[:, eligible].ffill(limit=5).dropna(how="all")


def _equal_weight_benchmark(prices: pd.DataFrame) -> pd.Series:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    return returns.mean(axis=1)


def _equal_country_benchmark(prices: pd.DataFrame, group_map: pd.Series) -> pd.Series:
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    aligned_groups = group_map.reindex(returns.columns)
    country_returns = []
    for country in aligned_groups.dropna().unique():
        columns = aligned_groups[aligned_groups == country].index
        country_returns.append(returns.loc[:, columns].mean(axis=1).rename(country))
    if not country_returns:
        return pd.Series(dtype=float)
    return pd.concat(country_returns, axis=1).mean(axis=1)


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


def _coverage_with_metadata(coverage: pd.DataFrame, metadata: pd.DataFrame, tier: str) -> pd.DataFrame:
    enriched = coverage.merge(metadata, on="ticker", how="left")
    enriched["download_tier"] = tier
    return enriched


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


def _model_suite(prefix: str = "asean") -> list:
    return [
        MomentumModel(name=f"{prefix}_mom_21", lookback=21),
        MomentumModel(name=f"{prefix}_mom_63", lookback=63),
        MomentumModel(name=f"{prefix}_mom_126", lookback=126),
        VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_63", lookback=63, volatility_lookback=63),
        VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_126", lookback=126, volatility_lookback=63),
        MeanReversionModel(name=f"{prefix}_rev_5", lookback=5),
        MeanReversionModel(name=f"{prefix}_rev_21", lookback=21),
        LowVolatilityModel(name=f"{prefix}_lowvol_63", lookback=63),
        LowVolatilityModel(name=f"{prefix}_lowvol_126", lookback=126),
    ]


def _run_tournaments(
    eligible_prices: pd.DataFrame,
    group_map: pd.Series,
    benchmark_returns: pd.Series,
    cost_bps: float,
) -> pd.DataFrame:
    tournament_specs = [
        ("dn", "dollar_neutral", None, 0.33),
        ("long_top33", "long_only", None, 0.33),
        ("long_top20", "long_only", None, 0.20),
        ("ctry_dn", "group_neutral_dollar_neutral", group_map, 0.33),
        ("ctry_long_top33", "group_neutral_long_only", group_map, 0.33),
        ("ctry_long_top20", "group_neutral_long_only", group_map, 0.20),
    ]
    frames = []
    for label, weighting, groups, top_fraction in tournament_specs:
        tournament = ModelTournament(
            models=_model_suite(prefix=f"asean_{label}"),
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


def run_asean_cost_sensitivity(
    artifact_dir: str | Path,
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 10.0, 15.0, 25.0, 50.0),
) -> AseanArtifactAnalysisResult:
    """Re-run the model suite from a saved ASEAN artifact at multiple cost levels."""

    artifact_path = Path(artifact_dir)
    prices_path = artifact_path / "eligible_stock_usd_prices.parquet"
    if not prices_path.exists():
        raise FileNotFoundError(f"missing eligible prices: {prices_path}")

    eligible_prices = pd.read_parquet(prices_path)
    metadata = _metadata_from_artifact(artifact_path)
    group_map = metadata.set_index("ticker")["country"].reindex(eligible_prices.columns)
    benchmark_returns = _equal_weight_benchmark(eligible_prices)

    frames = []
    for cost_bps in cost_bps_grid:
        frame = _run_tournaments(
            eligible_prices=eligible_prices,
            group_map=group_map,
            benchmark_returns=benchmark_returns,
            cost_bps=cost_bps,
        )
        if not frame.empty:
            frames.append(frame)

    cost_sensitivity = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if cost_sensitivity.empty:
        best_by_cost = pd.DataFrame()
    else:
        cost_sensitivity = cost_sensitivity.sort_values(["cost_bps", "deflated_sharpe", "sharpe"], ascending=[True, False, False])
        best_by_cost = cost_sensitivity.groupby("cost_bps", as_index=False).head(5)

    cost_sensitivity.to_csv(artifact_path / "cost_sensitivity.csv", index=False)
    best_by_cost.to_csv(artifact_path / "best_by_cost.csv", index=False)
    _write_cost_sensitivity_note(artifact_path, cost_sensitivity, best_by_cost)

    return AseanArtifactAnalysisResult(
        artifact_dir=artifact_path,
        cost_sensitivity=cost_sensitivity,
        best_by_cost=best_by_cost,
    )


def run_asean_research(
    artifact_root: str | Path = "artifacts/asean",
    vault_root: str | Path = "vault",
    stock_period: str = "5y",
    index_period: str = "10y",
    intraday_period: str = "60d",
    intraday_interval: str = "1h",
) -> AseanResearchResult:
    run_id = _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    instruments = default_asean_universe()
    metadata = _metadata(instruments)
    metadata.to_csv(artifact_dir / "universe.csv", index=False)

    index_tickers = metadata.loc[metadata["tier"].isin(["regional_proxy", "country_index"]), "ticker"].tolist()
    stock_tickers = metadata.loc[metadata["tier"] == "stock", "ticker"].tolist()
    fx_tickers = fx_tickers_for(instruments)

    daily_fallbacks = ["5y", "2y", "1y", "6mo"]
    index_daily = download_price_history(
        index_tickers,
        period=index_period,
        interval="1d",
        output_path=artifact_dir / "index_daily.parquet",
        fallback_periods=daily_fallbacks,
    )
    stock_daily = download_price_history(
        stock_tickers,
        period=stock_period,
        interval="1d",
        output_path=artifact_dir / "stock_daily.parquet",
        fallback_periods=["2y", "1y", "6mo"],
    )
    fx_daily = download_price_history(
        fx_tickers,
        period=stock_period,
        interval="1d",
        output_path=artifact_dir / "fx_daily.parquet",
        fallback_periods=["2y", "1y", "6mo"],
    )

    intraday_sample = [
        "ASEA",
        "^STI",
        "^JKSE",
        "^KLSE",
        "D05.SI",
        "BBCA.JK",
        "1155.KL",
    ]
    intraday = download_price_history(
        intraday_sample,
        period=intraday_period,
        interval=intraday_interval,
        output_path=artifact_dir / f"intraday_{intraday_interval}.parquet",
    )

    all_daily = pd.concat([index_daily.prices, stock_daily.prices], axis=1).sort_index()
    usd_prices = convert_local_prices_to_usd(all_daily, instruments, fx_daily.prices)
    usd_prices.to_parquet(artifact_dir / "daily_usd_prices.parquet")

    eligible_prices = _eligible_prices(usd_prices, metadata, min_rows=500, max_missing_ratio=0.2)
    eligible_prices.to_parquet(artifact_dir / "eligible_stock_usd_prices.parquet")

    if eligible_prices.shape[1] < 5 or eligible_prices.shape[0] < 756:
        leaderboard = pd.DataFrame()
    else:
        group_map = metadata.set_index("ticker")["country"].reindex(eligible_prices.columns)
        benchmark_returns = _equal_weight_benchmark(eligible_prices)
        equal_country_returns = _equal_country_benchmark(eligible_prices, group_map)
        leaderboard = _run_tournaments(
            eligible_prices=eligible_prices,
            group_map=group_map,
            benchmark_returns=benchmark_returns,
            cost_bps=15.0,
        )
        if not leaderboard.empty:
            leaderboard = leaderboard.sort_values(["deflated_sharpe", "sharpe"], ascending=False)
        leaderboard = pd.concat(
            [
                leaderboard,
                pd.DataFrame(
                    [
                        _benchmark_row("asean_equal_weight_usd", benchmark_returns),
                        _benchmark_row("asean_equal_country_usd", equal_country_returns),
                    ]
                ),
            ]
        )
        leaderboard = leaderboard.reset_index(drop=True)

    leaderboard.to_csv(artifact_dir / "leaderboard.csv", index=False)

    diagnostics = _write_diagnostics(artifact_dir, leaderboard, eligible_prices, metadata)

    coverage = pd.concat(
        [
            _coverage_with_metadata(index_daily.coverage, metadata, "index_daily"),
            _coverage_with_metadata(stock_daily.coverage, metadata, "stock_daily"),
            fx_daily.coverage.assign(country="FX", currency="USD", exchange="Yahoo FX", tier="fx", download_tier="fx_daily"),
            _coverage_with_metadata(intraday.coverage, metadata, "intraday_sample"),
        ],
        ignore_index=True,
    )
    coverage.to_csv(artifact_dir / "coverage.csv", index=False)

    note_paths = _write_obsidian_notes(
        vault_root=Path(vault_root),
        run_id=run_id,
        artifact_dir=artifact_dir,
        leaderboard=leaderboard,
        coverage=coverage,
        eligible_prices=eligible_prices,
        diagnostics=diagnostics,
        failures={
            "index_daily": index_daily.failures,
            "stock_daily": stock_daily.failures,
            "fx_daily": fx_daily.failures,
            "intraday": intraday.failures,
        },
    )

    return AseanResearchResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        leaderboard=leaderboard,
        coverage=coverage,
        note_paths=note_paths,
    )


def _write_obsidian_notes(
    vault_root: Path,
    run_id: str,
    artifact_dir: Path,
    leaderboard: pd.DataFrame,
    coverage: pd.DataFrame,
    eligible_prices: pd.DataFrame,
    diagnostics: dict[str, Path],
    failures: dict[str, dict[str, str]],
) -> list[Path]:
    experiment_dir = Path("30_experiments") / run_id
    paths: list[Path] = []
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    artifact_link = artifact_dir.resolve()

    summary = [
        f"Run id: `{run_id}`",
        f"Artifact directory: `{artifact_link}`",
        f"Eligible stock universe: `{eligible_prices.shape[1]}` tickers across `{eligible_prices.shape[0]}` dates.",
        f"Diagnostics: `{diagnostics.get('diagnostics_summary', '')}`",
        "",
        "## Best Result",
        "No leaderboard was produced." if not best else "\n".join(f"- `{key}`: `{value}`" for key, value in best.items()),
        "",
        "## Caveats",
        "- Current-survivor free-data universe; not a point-in-time constituent database.",
        "- Default ASEAN scope is Malaysia, Singapore, and Indonesia only.",
        "- USD conversion uses Yahoo LC-per-USD FX closes and prior-only forward fill.",
        "- Intraday data is extracted for inspection, not used for model selection in this first pass.",
        "- Benchmark-relative metrics compare models to an equal-weight USD stock basket.",
    ]
    paths.append(
        write_note(
            vault_root,
            experiment_dir / "index.md",
            ObsidianNote(
                title=f"{run_id} ASEAN Backtest",
                note_type="experiment",
                tags=["quant-lab", "asean", "backtest"],
                metadata={"experiment_id": run_id, "status": "completed", "artifact_dir": str(artifact_link)},
                body="\n".join(summary),
            ),
        )
    )

    validation_body = [
        "## Leaderboard",
        _markdown_table(leaderboard) if not leaderboard.empty else "No leaderboard.",
        "",
        "## Coverage Snapshot",
        _markdown_table(coverage[["ticker", "rows", "start", "end", "country", "download_tier"]]),
        "",
        "## Failures",
    ]
    for tier, tier_failures in failures.items():
        validation_body.append(f"### {tier}")
        validation_body.append("None" if not tier_failures else "\n".join(f"- `{ticker}`: {reason}" for ticker, reason in tier_failures.items()))

    paths.append(
        write_note(
            vault_root,
            experiment_dir / "validation_report.md",
            ObsidianNote(
                title=f"{run_id} Validation Report",
                note_type="validation_report",
                tags=["quant-lab", "validation", "asean"],
                metadata={"experiment_id": run_id, "artifact_dir": str(artifact_link)},
                body="\n".join(validation_body),
            ),
        )
    )

    paths.append(
        write_note(
            vault_root,
            "10_maps/ASEAN Quant Research Map.md",
            ObsidianNote(
                title="ASEAN Quant Research Map",
                note_type="map",
                tags=["quant-lab", "map", "asean"],
                metadata={"latest_experiment": run_id},
                body=(
                    f"- Latest experiment: [[{run_id}/index|{run_id} ASEAN Backtest]]\n"
                    "- Permanent concepts: [[PERM - Model Tournament]], [[PERM - Purged Walk Forward Validation]], [[PERM - Deflated Sharpe Ratio]]\n"
                    "- Machine artifacts are stored outside the vault under the run artifact directory."
                ),
            ),
        )
    )

    return paths


def _write_cost_sensitivity_note(
    artifact_dir: Path,
    cost_sensitivity: pd.DataFrame,
    best_by_cost: pd.DataFrame,
    vault_root: str | Path = "vault",
) -> Path | None:
    run_id = artifact_dir.name
    if cost_sensitivity.empty:
        return None

    body = [
        f"Artifact directory: `{artifact_dir.resolve()}`",
        "",
        "## Best By Cost",
        _markdown_table(best_by_cost.head(40)),
        "",
        "## Interpretation Prompts",
        "- If the winner changes dramatically between `0` and `15` bps, the edge is friction-sensitive.",
        "- A robust candidate should retain positive active return and acceptable drawdown at stressed costs.",
        "- Compare `annual_excess_return`, `tracking_error`, and `information_ratio`, not only Sharpe.",
    ]
    return write_note(
        vault_root,
        Path("30_experiments") / run_id / "cost_sensitivity.md",
        ObsidianNote(
            title=f"{run_id} Cost Sensitivity",
            note_type="cost_sensitivity",
            tags=["quant-lab", "asean", "costs"],
            metadata={"experiment_id": run_id, "artifact_dir": str(artifact_dir.resolve())},
            body="\n".join(body),
        ),
    )


def _write_diagnostics(
    artifact_dir: Path,
    leaderboard: pd.DataFrame,
    eligible_prices: pd.DataFrame,
    metadata: pd.DataFrame,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    if eligible_prices.empty:
        return paths

    returns = eligible_prices.pct_change(fill_method=None).fillna(0.0)
    group_map = metadata.set_index("ticker")["country"].reindex(eligible_prices.columns)
    country_returns = {}
    for country in group_map.dropna().unique():
        columns = group_map[group_map == country].index
        country_returns[country] = returns.loc[:, columns].mean(axis=1)
    country_frame = pd.DataFrame(country_returns)
    country_summary = pd.DataFrame(
        {
            "annual_return": country_frame.mean() * 252,
            "annual_volatility": country_frame.std(ddof=1) * (252**0.5),
            "sharpe": country_frame.apply(sharpe_ratio),
            "max_drawdown": country_frame.apply(max_drawdown),
        }
    ).sort_values("sharpe", ascending=False)
    country_path = artifact_dir / "country_benchmark_summary.csv"
    country_summary.to_csv(country_path)
    paths["country_benchmark_summary"] = country_path

    if not leaderboard.empty:
        ranked = leaderboard.copy()
        ranked["beats_equal_weight"] = ranked["annual_excess_return"].fillna(-999.0) > 0
        summary_path = artifact_dir / "diagnostics_summary.csv"
        ranked.to_csv(summary_path, index=False)
        paths["diagnostics_summary"] = summary_path

    return paths
