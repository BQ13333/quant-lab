from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.models.source_strategies import build_source_strategy_model_suite, source_strategy_manifest
from quant_lab.research.cross_market import build_cross_market_model_suite
from quant_lab.research.cross_market import run_cross_market_sweep_from_long_ohlcv


ASIA_L1_ALL_COUNTRIES: tuple[str, ...] = (
    "Hong Kong",
    "Indonesia",
    "Japan",
    "Malaysia",
    "Singapore",
    "South Korea",
    "Thailand",
)


ASIA_L1_EXCLUDED_COUNTRIES: tuple[str, ...] = ("Thailand",)


ASIA_L1_COUNTRIES: tuple[str, ...] = tuple(
    country for country in ASIA_L1_ALL_COUNTRIES if country not in ASIA_L1_EXCLUDED_COUNTRIES
)


@dataclass(frozen=True)
class CrossMarketSweepSpec:
    spec_id: str
    convert_to_usd: bool
    min_rows: int
    max_missing_ratio: float
    train_size: int
    test_size: int
    embargo: int
    max_country_weight: float
    max_asset_weight: float
    min_deflated_sharpe: float
    drawdown_limit: float = -0.20
    target_volatility: float = 0.12
    max_leverage: float = 3.0
    fx_period: str = "1y"

    @property
    def currency_label(self) -> str:
        return "usd" if self.convert_to_usd else "local"


@dataclass(frozen=True)
class CrossMarketBatchResult:
    batch_id: str
    artifact_dir: Path
    scoreboard: pd.DataFrame
    constraint_summary: pd.DataFrame
    failures: pd.DataFrame


def _latest_batch_id() -> str:
    return datetime.now(timezone.utc).strftime("CROSS-MARKET-BATCH-%Y%m%d-%H%M%S")


def _models_for_strategy_suite(strategy_suite: str):
    if strategy_suite == "core":
        return build_cross_market_model_suite(prefix="cross")
    if strategy_suite == "source":
        return build_source_strategy_model_suite(prefix="source")
    raise ValueError("strategy_suite must be 'core' or 'source'")


def default_asia_l1_sweep_specs() -> list[CrossMarketSweepSpec]:
    """Small but broad L1 grid sized for a one-year daily OHLCV dataset."""

    return [
        CrossMarketSweepSpec(
            spec_id="asia_l1_local_strict_126",
            convert_to_usd=False,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=126,
            test_size=21,
            embargo=2,
            max_country_weight=0.35,
            max_asset_weight=0.10,
            min_deflated_sharpe=0.50,
        ),
        CrossMarketSweepSpec(
            spec_id="asia_l1_local_balanced_126",
            convert_to_usd=False,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=126,
            test_size=21,
            embargo=2,
            max_country_weight=0.40,
            max_asset_weight=0.15,
            min_deflated_sharpe=0.45,
        ),
        CrossMarketSweepSpec(
            spec_id="asia_l1_local_relaxed_126",
            convert_to_usd=False,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=126,
            test_size=21,
            embargo=2,
            max_country_weight=0.45,
            max_asset_weight=0.25,
            min_deflated_sharpe=0.40,
        ),
        CrossMarketSweepSpec(
            spec_id="asia_l1_local_fast_84",
            convert_to_usd=False,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=84,
            test_size=21,
            embargo=2,
            max_country_weight=0.40,
            max_asset_weight=0.20,
            min_deflated_sharpe=0.40,
        ),
        CrossMarketSweepSpec(
            spec_id="asia_l1_local_slow_168",
            convert_to_usd=False,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=168,
            test_size=21,
            embargo=2,
            max_country_weight=0.40,
            max_asset_weight=0.20,
            min_deflated_sharpe=0.40,
        ),
        CrossMarketSweepSpec(
            spec_id="asia_l1_usd_balanced_126",
            convert_to_usd=True,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=126,
            test_size=21,
            embargo=2,
            max_country_weight=0.40,
            max_asset_weight=0.20,
            min_deflated_sharpe=0.40,
        ),
        CrossMarketSweepSpec(
            spec_id="asia_l1_usd_relaxed_126",
            convert_to_usd=True,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=126,
            test_size=21,
            embargo=2,
            max_country_weight=0.45,
            max_asset_weight=0.25,
            min_deflated_sharpe=0.35,
        ),
        CrossMarketSweepSpec(
            spec_id="asia_l1_usd_fast_84",
            convert_to_usd=True,
            min_rows=120,
            max_missing_ratio=0.35,
            train_size=84,
            test_size=21,
            embargo=2,
            max_country_weight=0.45,
            max_asset_weight=0.25,
            min_deflated_sharpe=0.35,
        ),
    ]


def select_sweep_specs(
    specs: list[CrossMarketSweepSpec],
    spec_ids: list[str] | tuple[str, ...] | None = None,
    max_specs: int | None = None,
) -> list[CrossMarketSweepSpec]:
    """Filter sweep specs for bounded smoke runs and targeted reruns."""

    selected = list(specs)
    if spec_ids:
        requested = [spec_id.strip() for spec_id in spec_ids if spec_id.strip()]
        by_id = {spec.spec_id: spec for spec in selected}
        missing = [spec_id for spec_id in requested if spec_id not in by_id]
        if missing:
            raise ValueError(f"unknown sweep spec ids: {missing}")
        selected = [by_id[spec_id] for spec_id in requested]
    if max_specs is not None:
        if max_specs <= 0:
            raise ValueError("max_specs must be positive")
        selected = selected[:max_specs]
    if not selected:
        raise ValueError("no sweep specs selected")
    return selected


def _constraint_summary(scoreboard: pd.DataFrame) -> pd.DataFrame:
    if scoreboard.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    constraint_cols = [col for col in scoreboard.columns if col.startswith("constraint_") and col.endswith("_passed")]
    for spec_id, group in scoreboard.groupby("batch_spec_id", sort=True):
        row: dict[str, Any] = {
            "batch_spec_id": spec_id,
            "candidate_count": int(len(group)),
            "feasible_candidates": int(group["passed_all"].sum()) if "passed_all" in group else 0,
            "best_score": float(group["score"].max()) if "score" in group else 0.0,
            "best_net_ir": float(group["net_ir"].max()) if "net_ir" in group else 0.0,
            "best_deflated_sharpe": float(group["deflated_sharpe"].max()) if "deflated_sharpe" in group else 0.0,
        }
        for col in constraint_cols:
            name = col.removeprefix("constraint_").removesuffix("_passed")
            row[f"{name}_pass_rate"] = float(group[col].mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["feasible_candidates", "best_score", "best_net_ir"],
        ascending=[False, False, False],
    )


def run_cross_market_batch(
    long_ohlcv_root: str | Path,
    artifact_root: str | Path = "artifacts/cross_market_batches",
    batch_id: str | None = None,
    specs: list[CrossMarketSweepSpec] | None = None,
    include_countries: tuple[str, ...] | list[str] = ASIA_L1_COUNTRIES,
    strategy_suite: str = "core",
) -> CrossMarketBatchResult:
    """Run multiple cross-market sweeps sequentially and build one scoreboard."""

    specs = specs or default_asia_l1_sweep_specs()
    batch_id = batch_id or _latest_batch_id()
    artifact_dir = Path(artifact_root) / batch_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if strategy_suite == "source":
        source_strategy_manifest().to_csv(artifact_dir / "source_strategy_manifest.csv", index=False)

    candidate_frames: list[pd.DataFrame] = []
    failure_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for spec in specs:
        run_id = f"{batch_id}-{spec.spec_id}"
        try:
            models = _models_for_strategy_suite(strategy_suite)
            result = run_cross_market_sweep_from_long_ohlcv(
                long_ohlcv_root=long_ohlcv_root,
                artifact_root=artifact_dir / "runs",
                run_id=run_id,
                convert_to_usd=spec.convert_to_usd,
                fx_period=spec.fx_period,
                min_rows=spec.min_rows,
                max_missing_ratio=spec.max_missing_ratio,
                train_size=spec.train_size,
                test_size=spec.test_size,
                embargo=spec.embargo,
                target_volatility=spec.target_volatility,
                max_leverage=spec.max_leverage,
                max_country_weight_limit=spec.max_country_weight,
                max_asset_weight_limit=spec.max_asset_weight,
                drawdown_limit=spec.drawdown_limit,
                min_deflated_sharpe=spec.min_deflated_sharpe,
                include_countries=include_countries,
                models=models,
            )
            candidates = result.all_candidates.copy()
            candidates.insert(0, "batch_spec_id", spec.spec_id)
            candidates.insert(1, "batch_run_id", result.run_id)
            candidates.insert(2, "batch_artifact_dir", str(result.artifact_dir))
            candidates.insert(3, "price_currency", "USD" if spec.convert_to_usd else "local")
            for key, value in asdict(spec).items():
                candidates[f"spec_{key}"] = value
            candidate_frames.append(candidates)
            run_rows.append(
                {
                    "batch_spec_id": spec.spec_id,
                    "run_id": result.run_id,
                    "artifact_dir": str(result.artifact_dir),
                    "price_currency": "USD" if spec.convert_to_usd else "local",
                    "eligible_tickers": int(result.prices.shape[1]),
                    "rows": int(result.prices.shape[0]),
                    "candidate_count": int(len(result.all_candidates)),
                    "feasible_candidates": int(len(result.feasible_candidates)),
                    "model_count": len(models),
                    "fx_tickers": ",".join(result.fx_tickers),
                    "status": "completed",
                }
            )
        except Exception as exc:
            failure_rows.append(
                {
                    "batch_spec_id": spec.spec_id,
                    "run_id": run_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            run_rows.append(
                {
                    "batch_spec_id": spec.spec_id,
                    "run_id": run_id,
                    "artifact_dir": "",
                    "price_currency": "USD" if spec.convert_to_usd else "local",
                    "eligible_tickers": 0,
                    "rows": 0,
                    "candidate_count": 0,
                    "feasible_candidates": 0,
                    "model_count": 0,
                    "fx_tickers": "",
                    "status": "failed",
                }
            )

    scoreboard = pd.concat(candidate_frames, ignore_index=True) if candidate_frames else pd.DataFrame()
    if not scoreboard.empty:
        scoreboard = scoreboard.sort_values(
            ["passed_all", "score", "net_ir", "net_sharpe", "deflated_sharpe"],
            ascending=[False, False, False, False, False],
        ).reset_index(drop=True)
    constraint_summary = _constraint_summary(scoreboard)
    failures = pd.DataFrame(failure_rows)
    runs = pd.DataFrame(run_rows)

    scoreboard.to_csv(artifact_dir / "scoreboard.csv", index=False)
    constraint_summary.to_csv(artifact_dir / "constraint_summary.csv", index=False)
    failures.to_csv(artifact_dir / "failures.csv", index=False)
    runs.to_csv(artifact_dir / "runs.csv", index=False)

    manifest = {
        "batch_id": batch_id,
        "long_ohlcv_root": str(Path(long_ohlcv_root).resolve()),
        "artifact_dir": str(artifact_dir.resolve()),
        "include_countries": list(include_countries),
        "strategy_suite": strategy_suite,
        "model_count_per_run": int(runs["model_count"].max()) if not runs.empty else 0,
        "specs": [asdict(spec) for spec in specs],
        "run_count": len(specs),
        "completed_runs": int((runs["status"] == "completed").sum()) if not runs.empty else 0,
        "failed_runs": int((runs["status"] == "failed").sum()) if not runs.empty else 0,
        "candidate_count": int(len(scoreboard)),
        "feasible_candidates": int(scoreboard["passed_all"].sum()) if not scoreboard.empty else 0,
    }
    (artifact_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CrossMarketBatchResult(
        batch_id=batch_id,
        artifact_dir=artifact_dir,
        scoreboard=scoreboard,
        constraint_summary=constraint_summary,
        failures=failures,
    )
