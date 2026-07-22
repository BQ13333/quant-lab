from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.asean_universe import Instrument
from quant_lab.data.fx import convert_local_prices_to_usd, fx_tickers_for
from quant_lab.data.yfinance_loader import download_price_history
from quant_lab.models.baselines import LowVolatilityModel, MomentumModel, VolatilityAdjustedMomentumModel
from quant_lab.models.zoo import BreakoutModel, TimeSeriesMomentumModel, TrendLowVolEnsembleModel, VolatilityCompressionModel
from quant_lab.research.overfit_audit import write_overfit_audit


@dataclass(frozen=True)
class CrossMarketDataset:
    prices: pd.DataFrame
    metadata: pd.DataFrame
    group_map: pd.Series


@dataclass(frozen=True)
class CrossMarketSweepResult:
    run_id: str
    artifact_dir: Path
    prices: pd.DataFrame
    metadata: pd.DataFrame
    all_candidates: pd.DataFrame
    feasible_candidates: pd.DataFrame
    champions: pd.DataFrame
    fx_tickers: list[str]


def _latest_run_id() -> str:
    return datetime.now(timezone.utc).strftime("CROSS-MARKET-SWEEP-%Y%m%d-%H%M%S")


def _equal_weight_returns(prices: pd.DataFrame) -> pd.Series:
    returns = prices.pct_change(fill_method=None)
    return returns.mean(axis=1).fillna(0.0)


def _family_from_model_name(model_name: str) -> str:
    if "trend_lowvol" in model_name:
        return "trend_low_vol_ensemble"
    if "lowvol" in model_name:
        return "low_volatility"
    if "_vam_" in model_name:
        return "volatility_adjusted_momentum"
    if "_tsmom_" in model_name:
        return "time_series_momentum"
    if "_breakout_" in model_name:
        return "breakout"
    if "_volcomp_" in model_name:
        return "volatility_compression"
    if "_mom_" in model_name:
        return "momentum"
    return "other"


def _metadata_from_long_ohlcv(long_ohlcv: pd.DataFrame) -> pd.DataFrame:
    metadata_cols = ["ticker", "sid", "local_symbol", "name", "country", "exchange", "currency"]
    available = [col for col in metadata_cols if col in long_ohlcv.columns]
    metadata = (
        long_ohlcv[available]
        .dropna(subset=["ticker"])
        .drop_duplicates("ticker", keep="last")
        .sort_values("ticker")
        .reset_index(drop=True)
    )
    for col in metadata_cols:
        if col not in metadata.columns:
            metadata[col] = pd.NA
    metadata["tier"] = "stock"
    return metadata[metadata_cols + ["tier"]]


def _metadata_to_instruments(metadata: pd.DataFrame) -> list[Instrument]:
    instruments = []
    for row in metadata.itertuples(index=False):
        ticker = str(row.ticker)
        instruments.append(
            Instrument(
                ticker=ticker,
                name=str(row.name) if pd.notna(row.name) else ticker,
                country=str(row.country) if pd.notna(row.country) else "Unknown",
                currency=str(row.currency) if pd.notna(row.currency) else "USD",
                exchange=str(row.exchange) if pd.notna(row.exchange) else "Unknown",
                tier="stock",
            )
        )
    return instruments


def _normalize_country_set(countries: list[str] | tuple[str, ...] | set[str] | None) -> set[str] | None:
    if not countries:
        return None
    normalized = {str(country).strip().lower() for country in countries if str(country).strip()}
    return normalized or None


def filter_dataset_by_countries(
    dataset: CrossMarketDataset,
    include_countries: list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_countries: list[str] | tuple[str, ...] | set[str] | None = None,
) -> CrossMarketDataset:
    """Filter an already-loaded cross-market dataset by metadata country."""

    include = _normalize_country_set(include_countries)
    exclude = _normalize_country_set(exclude_countries)
    if include is None and exclude is None:
        return dataset

    metadata = dataset.metadata.copy()
    country_key = metadata["country"].astype(str).str.strip().str.lower()
    mask = pd.Series(True, index=metadata.index)
    if include is not None:
        mask &= country_key.isin(include)
    if exclude is not None:
        mask &= ~country_key.isin(exclude)
    metadata = metadata.loc[mask].reset_index(drop=True)

    tickers = [ticker for ticker in dataset.prices.columns if ticker in set(metadata["ticker"])]
    prices = dataset.prices.loc[:, tickers]
    group_map = metadata.set_index("ticker")["country"].reindex(prices.columns)
    if prices.empty:
        raise ValueError("no eligible price columns after country filters")
    return CrossMarketDataset(prices=prices, metadata=metadata, group_map=group_map)


def load_long_ohlcv_dataset(
    long_ohlcv_root: str | Path,
    price_col: str = "adj_close",
    min_rows: int = 1000,
    max_missing_ratio: float = 0.15,
    ffill_limit: int = 5,
    include_countries: list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_countries: list[str] | tuple[str, ...] | set[str] | None = None,
) -> CrossMarketDataset:
    """Load long OHLCV partitions into an eligible wide price matrix."""

    long_ohlcv = pd.read_parquet(long_ohlcv_root)
    if long_ohlcv.empty:
        raise ValueError("long OHLCV dataset is empty")
    if price_col not in long_ohlcv.columns:
        raise ValueError(f"missing price column: {price_col}")

    long_ohlcv = long_ohlcv.copy()
    long_ohlcv["date"] = pd.to_datetime(long_ohlcv["date"], errors="coerce")
    long_ohlcv[price_col] = pd.to_numeric(long_ohlcv[price_col], errors="coerce")
    long_ohlcv = long_ohlcv.dropna(subset=["date", "ticker", price_col])

    metadata = _metadata_from_long_ohlcv(long_ohlcv)
    prices = (
        long_ohlcv.pivot_table(index="date", columns="ticker", values=price_col, aggfunc="last")
        .sort_index()
        .astype(float)
    )
    prices = prices.ffill(limit=ffill_limit)
    eligible = prices.notna().sum().ge(min_rows) & prices.isna().mean().le(max_missing_ratio)
    prices = prices.loc[:, eligible].dropna(how="all")
    metadata = metadata[metadata["ticker"].isin(prices.columns)].reset_index(drop=True)
    group_map = metadata.set_index("ticker")["country"].reindex(prices.columns)

    if prices.empty:
        raise ValueError("no eligible price columns after coverage filters")
    return filter_dataset_by_countries(
        CrossMarketDataset(prices=prices, metadata=metadata, group_map=group_map),
        include_countries=include_countries,
        exclude_countries=exclude_countries,
    )


def convert_dataset_to_usd(
    dataset: CrossMarketDataset,
    fx_closes: pd.DataFrame,
    min_rows: int = 1000,
    max_missing_ratio: float = 0.15,
) -> CrossMarketDataset:
    instruments = _metadata_to_instruments(dataset.metadata)
    usd_prices = convert_local_prices_to_usd(dataset.prices, instruments=instruments, fx_closes=fx_closes)
    eligible = usd_prices.notna().sum().ge(min_rows) & usd_prices.isna().mean().le(max_missing_ratio)
    usd_prices = usd_prices.loc[:, eligible].dropna(how="all")
    metadata = dataset.metadata[dataset.metadata["ticker"].isin(usd_prices.columns)].reset_index(drop=True)
    group_map = metadata.set_index("ticker")["country"].reindex(usd_prices.columns)
    if usd_prices.empty:
        raise ValueError("no eligible USD price columns after FX conversion")
    return CrossMarketDataset(prices=usd_prices, metadata=metadata, group_map=group_map)


def build_cross_market_model_suite(prefix: str = "cross") -> list:
    return [
        MomentumModel(name=f"{prefix}_mom_63", lookback=63),
        MomentumModel(name=f"{prefix}_mom_126", lookback=126),
        LowVolatilityModel(name=f"{prefix}_lowvol_63", lookback=63),
        LowVolatilityModel(name=f"{prefix}_lowvol_126", lookback=126),
        VolatilityAdjustedMomentumModel(name=f"{prefix}_vam_63_vol63", lookback=63, volatility_lookback=63),
        TimeSeriesMomentumModel(name=f"{prefix}_tsmom_126_vol", lookback=126, scale_by_vol=True),
        BreakoutModel(name=f"{prefix}_breakout_55", lookback=55),
        VolatilityCompressionModel(name=f"{prefix}_volcomp_42_126", vol_lookback=42, compression_lookback=126),
        TrendLowVolEnsembleModel(
            name=f"{prefix}_trend_lowvol_126",
            trend_model=MomentumModel(name=f"{prefix}_trend_mom_126", lookback=126),
            lowvol_model=LowVolatilityModel(name=f"{prefix}_trend_lowvol_leg_126", lookback=126),
        ),
    ]


def build_cross_market_allocator_configs() -> list[dict[str, Any]]:
    return [
        {"weighting": "long_only", "top_fraction": 0.10, "rebalance_frequency": "W"},
        {"weighting": "long_only", "top_fraction": 0.20, "rebalance_frequency": "W"},
        {"weighting": "long_only", "top_fraction": 0.20, "rebalance_frequency": "M"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.20, "rebalance_frequency": "W"},
        {"weighting": "group_neutral_long_only", "top_fraction": 0.20, "rebalance_frequency": "M"},
        {"weighting": "group_neutral_dollar_neutral", "top_fraction": 1.0, "rebalance_frequency": "W"},
    ]


def run_cross_market_sweep_from_long_ohlcv(
    long_ohlcv_root: str | Path,
    artifact_root: str | Path = "artifacts/cross_market",
    run_id: str | None = None,
    convert_to_usd: bool = True,
    fx_period: str = "5y",
    price_col: str = "adj_close",
    min_rows: int = 1000,
    max_missing_ratio: float = 0.15,
    train_size: int = 504,
    test_size: int = 63,
    embargo: int = 5,
    cost_bps_grid: tuple[float, ...] = (5.0, 15.0, 25.0),
    baseline_cost_bps: float = 15.0,
    target_volatility: float = 0.12,
    max_leverage: float = 3.0,
    max_country_weight_limit: float = 0.35,
    max_asset_weight_limit: float = 0.10,
    drawdown_limit: float = -0.20,
    min_deflated_sharpe: float = 0.80,
    precompute_full_signals: bool = False,
    write_candidate_weights: bool = False,
    candidate_weight_top_n: int | None = None,
    models: list | None = None,
    allocator_configs: list[dict[str, Any]] | None = None,
    include_countries: list[str] | tuple[str, ...] | set[str] | None = None,
    exclude_countries: list[str] | tuple[str, ...] | set[str] | None = None,
) -> CrossMarketSweepResult:
    run_id = run_id or _latest_run_id()
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_long_ohlcv_dataset(
        long_ohlcv_root,
        price_col=price_col,
        min_rows=min_rows,
        max_missing_ratio=max_missing_ratio,
        include_countries=include_countries,
        exclude_countries=exclude_countries,
    )
    fx_tickers: list[str] = []
    fx_closes = pd.DataFrame()
    price_currency = "local"
    if convert_to_usd:
        instruments = _metadata_to_instruments(dataset.metadata)
        fx_tickers = fx_tickers_for(instruments)
        if fx_tickers:
            fx_result = download_price_history(
                fx_tickers,
                period=fx_period,
                interval="1d",
                output_path=artifact_dir / "fx_closes.parquet",
                fallback_periods=["3y", "1y"],
            )
            fx_closes = fx_result.prices
            fx_result.coverage.to_csv(artifact_dir / "fx_coverage.csv", index=False)
        dataset = convert_dataset_to_usd(
            dataset,
            fx_closes=fx_closes,
            min_rows=min_rows,
            max_missing_ratio=max_missing_ratio,
        )
        price_currency = "USD"

    prices = dataset.prices
    metadata = dataset.metadata
    group_map = dataset.group_map
    prices.to_parquet(artifact_dir / f"eligible_prices_{price_currency.lower()}.parquet")
    metadata.to_csv(artifact_dir / "metadata.csv", index=False)

    models = models or build_cross_market_model_suite(prefix="cross")
    allocator_configs = allocator_configs or build_cross_market_allocator_configs()
    from quant_lab.optimization.net_ir_optimizer import NetIRConstrainedOptimizer

    optimizer = NetIRConstrainedOptimizer(
        drawdown_limit=drawdown_limit,
        max_turnover=3.0,
        max_country_weight_limit=max_country_weight_limit,
        max_asset_weight_limit=max_asset_weight_limit,
        min_deflated_sharpe=min_deflated_sharpe,
        target_volatility=target_volatility,
        train_size=train_size,
        test_size=test_size,
        embargo=embargo,
        max_leverage=max_leverage,
        precompute_full_signals=precompute_full_signals,
    )
    candidate_results = optimizer.evaluate_all_candidates(
        models=models,
        allocator_configs=allocator_configs,
        cost_bps_grid=cost_bps_grid,
        prices=prices,
        group_map=group_map,
        benchmark_returns=_equal_weight_returns(prices),
    )
    candidates = optimizer.results_to_frame(candidate_results)
    if not candidates.empty:
        fallback_family = candidates["model_name"].map(_family_from_model_name)
        if "meta_family" in candidates.columns:
            candidates["family"] = candidates["meta_family"].fillna(fallback_family)
        else:
            candidates["family"] = fallback_family
        candidates = candidates.sort_values(
            ["passed_all", "score", "net_ir", "net_sharpe", "deflated_sharpe"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)

    weight_paths: dict[str, str] = {}
    if write_candidate_weights and candidate_results:
        weight_dir = artifact_dir / "candidate_weights"
        weight_dir.mkdir(parents=True, exist_ok=True)
        if candidates.empty:
            selected_ids = {result.candidate_id for result in candidate_results}
        elif candidate_weight_top_n is None:
            selected_ids = set(candidates["candidate_id"])
        else:
            selected_ids = set(candidates.head(candidate_weight_top_n)["candidate_id"])
        for result in candidate_results:
            if result.candidate_id not in selected_ids:
                continue
            safe_name = (
                result.candidate_id.replace("|", "__")
                .replace("=", "-")
                .replace("/", "_")
                .replace(" ", "_")
            )
            path = weight_dir / f"{safe_name}.parquet"
            result.weights.to_parquet(path)
            weight_paths[result.candidate_id] = str(path.resolve())
    if weight_paths and not candidates.empty:
        candidates["candidate_weights_path"] = candidates["candidate_id"].map(weight_paths).fillna("")
    candidates.to_csv(artifact_dir / "all_candidates.csv", index=False)

    returns_matrix = pd.concat(
        {result.candidate_id: result.net_returns for result in candidate_results},
        axis=1,
    ) if candidate_results else pd.DataFrame()
    if not returns_matrix.empty:
        returns_matrix.to_parquet(artifact_dir / "candidate_returns.parquet")
        write_overfit_audit(candidates, returns_matrix, artifact_dir)

    feasible = candidates[candidates["passed_all"]].copy() if not candidates.empty else pd.DataFrame()
    feasible.to_csv(artifact_dir / "feasible_candidates.csv", index=False)
    baseline = candidates[candidates["cost_bps"] == baseline_cost_bps].copy() if not candidates.empty else pd.DataFrame()
    champions = (baseline if not baseline.empty else candidates).head(50).copy() if not candidates.empty else pd.DataFrame()
    champions.to_csv(artifact_dir / "champions.csv", index=False)

    manifest = {
        "run_id": run_id,
        "long_ohlcv_root": str(Path(long_ohlcv_root).resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "price_currency": price_currency,
        "eligible_tickers": int(prices.shape[1]),
        "rows": int(prices.shape[0]),
        "date_start": str(prices.index.min()),
        "date_end": str(prices.index.max()),
        "countries": sorted(str(country) for country in metadata["country"].dropna().unique()),
        "include_countries": sorted(include_countries) if include_countries else [],
        "exclude_countries": sorted(exclude_countries) if exclude_countries else [],
        "fx_tickers": fx_tickers,
        "models": len(models),
        "allocator_configs": len(allocator_configs),
        "cost_bps_grid": list(cost_bps_grid),
        "candidate_count": int(len(candidates)),
        "feasible_candidates": int(len(feasible)),
        "candidate_returns_path": str((artifact_dir / "candidate_returns.parquet").resolve()) if not returns_matrix.empty else "",
        "overfit_audit_path": str((artifact_dir / "overfit_audit.json").resolve()) if not returns_matrix.empty else "",
        "write_candidate_weights": write_candidate_weights,
        "candidate_weight_top_n": candidate_weight_top_n,
        "candidate_weight_count": len(weight_paths),
        "constraints": {
            "train_size": train_size,
            "test_size": test_size,
            "embargo": embargo,
            "target_volatility": target_volatility,
            "max_leverage": max_leverage,
            "drawdown_limit": drawdown_limit,
            "max_country_weight_limit": max_country_weight_limit,
            "max_asset_weight_limit": max_asset_weight_limit,
            "min_deflated_sharpe": min_deflated_sharpe,
            "precompute_full_signals": precompute_full_signals,
        },
    }
    (artifact_dir / "sweep_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return CrossMarketSweepResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        prices=prices,
        metadata=metadata,
        all_candidates=candidates,
        feasible_candidates=feasible,
        champions=champions,
        fx_tickers=fx_tickers,
    )
