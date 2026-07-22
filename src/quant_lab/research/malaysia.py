from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from quant_lab.models.baselines import (
    LowVolatilityModel,
    MeanReversionModel,
    MomentumModel,
    VolatilityAdjustedMomentumModel,
)
from quant_lab.models.zoo import BreakoutModel, TimeSeriesMomentumModel, TrendLowVolEnsembleModel
from quant_lab.notes.obsidian import ObsidianNote, write_note
from quant_lab.research.asean import _benchmark_row, _equal_weight_benchmark, _markdown_table
from quant_lab.research.tournament import ModelTournament


@dataclass(frozen=True)
class MalaysiaResearchResult:
    run_id: str
    artifact_dir: Path
    leaderboard: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    best_by_cost: pd.DataFrame
    eligible_prices: pd.DataFrame
    note_paths: list[Path]


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("MALAYSIA-%Y%m%d-%H%M%S")


def _model_suite(prefix: str) -> list:
    return [
        MomentumModel(name=f"{prefix}_mom_63", lookback=63),
        MomentumModel(name=f"{prefix}_mom_126", lookback=126),
        MomentumModel(name=f"{prefix}_mom_252", lookback=252),
        VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_126", lookback=126, volatility_lookback=63),
        TimeSeriesMomentumModel(name=f"{prefix}_tsmom_126", lookback=126, scale_by_vol=True),
        LowVolatilityModel(name=f"{prefix}_lowvol_63", lookback=63),
        LowVolatilityModel(name=f"{prefix}_lowvol_126", lookback=126),
        MeanReversionModel(name=f"{prefix}_rev_21", lookback=21),
        BreakoutModel(name=f"{prefix}_breakout_55", lookback=55),
        TrendLowVolEnsembleModel(
            name=f"{prefix}_trend_lowvol_126",
            trend_model=MomentumModel(name=f"{prefix}_trend_mom_126", lookback=126),
            lowvol_model=LowVolatilityModel(name=f"{prefix}_trend_lowvol_leg_126", lookback=126),
        ),
    ]


def _eligible_malaysia_prices(
    prices: pd.DataFrame,
    metadata: pd.DataFrame,
    min_rows: int = 1000,
    max_missing_ratio: float = 0.05,
) -> pd.DataFrame:
    malaysia_stocks = metadata.loc[
        (metadata["country"] == "Malaysia") & (metadata["tier"] == "stock"),
        "ticker",
    ].tolist()
    stock_prices = prices.reindex(columns=malaysia_stocks).dropna(how="all")
    row_counts = stock_prices.notna().sum()
    missing_ratio = stock_prices.isna().mean()
    eligible = row_counts.ge(min_rows) & missing_ratio.le(max_missing_ratio)
    return stock_prices.loc[:, eligible].ffill(limit=5).dropna(how="all")


def _ewm_benchmark(prices: pd.DataFrame) -> pd.Series:
    if "EWM" not in prices.columns:
        return pd.Series(dtype=float)
    return prices["EWM"].pct_change(fill_method=None).fillna(0.0)


def _run_tournaments(
    eligible_prices: pd.DataFrame,
    benchmark_returns: pd.Series,
    cost_bps: float,
    train_size: int,
    test_size: int,
    embargo: int,
) -> pd.DataFrame:
    specs = [
        ("dn", "dollar_neutral", 0.33),
        ("long_top33", "long_only", 0.33),
        ("long_top20", "long_only", 0.20),
        ("long_top10", "long_only", 0.10),
    ]
    frames = []
    for label, weighting, top_fraction in specs:
        tournament = ModelTournament(
            models=_model_suite(prefix=f"malaysia_{label}"),
            target_volatility=0.12,
            cost_bps=cost_bps,
            weighting=weighting,
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


def run_malaysia_research_from_asean_artifact(
    asean_artifact_dir: str | Path = "artifacts/asean/ASEAN-20260704-232302",
    artifact_root: str | Path = "artifacts/malaysia",
    vault_root: str | Path = "vault",
    run_id: str | None = None,
    cost_bps_grid: tuple[float, ...] = (0.0, 5.0, 15.0, 25.0, 50.0),
    baseline_cost_bps: float = 15.0,
    train_size: int = 504,
    test_size: int = 63,
    embargo: int = 5,
    min_rows: int = 1000,
    max_missing_ratio: float = 0.05,
) -> MalaysiaResearchResult:
    """Run a Malaysia-only research tournament from an existing ASEAN artifact."""

    source_path = Path(asean_artifact_dir)
    prices_path = source_path / "daily_usd_prices.parquet"
    metadata_path = source_path / "universe.csv"
    if not prices_path.exists():
        raise FileNotFoundError(f"missing prices: {prices_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing universe metadata: {metadata_path}")

    prices = pd.read_parquet(prices_path).sort_index()
    metadata = pd.read_csv(metadata_path)
    eligible_prices = _eligible_malaysia_prices(
        prices,
        metadata,
        min_rows=min_rows,
        max_missing_ratio=max_missing_ratio,
    )

    run_id = run_id or _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    malaysia_metadata = metadata.loc[
        metadata["ticker"].isin(eligible_prices.columns)
        | ((metadata["country"] == "Malaysia") & metadata["tier"].isin(["regional_proxy", "country_index"]))
    ].copy()
    malaysia_metadata.to_csv(artifact_dir / "universe.csv", index=False)
    eligible_prices.to_parquet(artifact_dir / "eligible_usd_prices.parquet")

    coverage_path = source_path / "coverage.csv"
    if coverage_path.exists():
        coverage = pd.read_csv(coverage_path)
        coverage = coverage.loc[coverage["ticker"].isin(malaysia_metadata["ticker"])].copy()
        coverage.to_csv(artifact_dir / "coverage.csv", index=False)

    if eligible_prices.shape[1] < 5 or eligible_prices.shape[0] < train_size + test_size:
        cost_sensitivity = pd.DataFrame()
        best_by_cost = pd.DataFrame()
        leaderboard = pd.DataFrame()
    else:
        equal_weight_returns = _equal_weight_benchmark(eligible_prices)
        frames = []
        for cost_bps in cost_bps_grid:
            frame = _run_tournaments(
                eligible_prices=eligible_prices,
                benchmark_returns=equal_weight_returns,
                cost_bps=cost_bps,
                train_size=train_size,
                test_size=test_size,
                embargo=embargo,
            )
            if not frame.empty:
                frames.append(frame)
        cost_sensitivity = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
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
            _benchmark_row("malaysia_equal_weight_usd", equal_weight_returns),
            _benchmark_row("malaysia_ewm_etf_usd", _ewm_benchmark(prices.reindex(eligible_prices.index))),
        ]
        leaderboard = pd.concat([baseline_frame, pd.DataFrame(benchmark_rows)], ignore_index=True)

    cost_sensitivity.to_csv(artifact_dir / "cost_sensitivity.csv", index=False)
    best_by_cost.to_csv(artifact_dir / "best_by_cost.csv", index=False)
    leaderboard.to_csv(artifact_dir / "leaderboard.csv", index=False)

    note_paths = _write_notes(
        vault_root=Path(vault_root),
        run_id=run_id,
        artifact_dir=artifact_dir,
        source_artifact=source_path,
        leaderboard=leaderboard,
        cost_sensitivity=cost_sensitivity,
        best_by_cost=best_by_cost,
        eligible_prices=eligible_prices,
        baseline_cost_bps=baseline_cost_bps,
    )

    return MalaysiaResearchResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        leaderboard=leaderboard,
        cost_sensitivity=cost_sensitivity,
        best_by_cost=best_by_cost,
        eligible_prices=eligible_prices,
        note_paths=note_paths,
    )


def _write_notes(
    vault_root: Path,
    run_id: str,
    artifact_dir: Path,
    source_artifact: Path,
    leaderboard: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
    best_by_cost: pd.DataFrame,
    eligible_prices: pd.DataFrame,
    baseline_cost_bps: float,
) -> list[Path]:
    experiment_dir = Path("30_experiments") / run_id
    best = leaderboard.iloc[0].to_dict() if not leaderboard.empty else {}
    body = [
        f"Run id: `{run_id}`",
        f"Source ASEAN artifact: `{source_artifact.resolve()}`",
        f"Artifact directory: `{artifact_dir.resolve()}`",
        f"Eligible Malaysia stocks: `{eligible_prices.shape[1]}`",
        f"Dates: `{eligible_prices.index.min()}` to `{eligible_prices.index.max()}`",
        f"Baseline cost: `{baseline_cost_bps}` bps",
        "",
        "## Best Result",
        "No leaderboard was produced." if not best else "\n".join(f"- `{key}`: `{value}`" for key, value in best.items()),
        "",
        "## Leaderboard",
        _markdown_table(leaderboard.head(30)) if not leaderboard.empty else "_No rows._",
        "",
        "## Best By Cost",
        _markdown_table(best_by_cost.head(30)) if not best_by_cost.empty else "_No rows._",
        "",
        "## Caveats",
        "- Malaysia pilot uses a curated liquid Bursa Malaysia basket, not the full Bursa market.",
        "- Prices are Yahoo-derived and converted into USD in the source ASEAN artifact.",
        "- Universe is current-survivor and not point-in-time.",
        "- Treat this as research, not a live-trading approval.",
    ]
    note = write_note(
        vault_root,
        experiment_dir / "index.md",
        ObsidianNote(
            title=f"{run_id} Malaysia Backtest",
            note_type="experiment",
            tags=["quant-lab", "malaysia", "backtest"],
            metadata={"experiment_id": run_id, "artifact_dir": str(artifact_dir.resolve())},
            body="\n".join(body),
        ),
    )
    return [note]
