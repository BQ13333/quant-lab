from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from quant_lab.data.usa_exchange_calendar_tca import TCAAssumptions


CALIBRATION_TYPE = "market_data_calibrated_research"
SPREAD_FORMULA = (
    "half_spread_bps = 0.5 * (high - low) / ((high + low) / 2) * 10000"
)
IMPACT_FORMULA = (
    "impact_bps = amihud_60_lag1 * adv_60_local_lag1 "
    "* reference_participation_rate * 10000"
)
REQUIRED_DAILY_COLUMNS = (
    "date",
    "ticker",
    "high",
    "low",
    "adv_60_local_lag1",
    "amihud_60_lag1",
    "bad_ohlc_flag",
)


@dataclass(frozen=True)
class FeeInput:
    """An explicit fee scalar and its independent provenance."""

    bps: float
    source_name: str
    source_quality: str

    def validate(self, label: str) -> None:
        if not math.isfinite(self.bps) or self.bps < 0:
            raise ValueError(f"{label}.bps must be a finite non-negative number")
        if not self.source_name.strip():
            raise ValueError(f"{label}.source_name cannot be empty")
        if not self.source_quality.strip():
            raise ValueError(f"{label}.source_quality cannot be empty")


@dataclass(frozen=True)
class USATCACalibrationArtifacts:
    run_id: str
    artifact_dir: Path
    assumptions_json_path: Path
    assumptions_csv_path: Path
    evidence_path: Path
    stress_grid_path: Path
    manifest_path: Path
    assumptions: TCAAssumptions
    stress_grid: pd.DataFrame
    evidence: dict[str, Any]
    manifest: dict[str, Any]


class _BoundedReservoir:
    """Algorithm R reservoir with fixed storage and deterministic random state."""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("max_calibration_samples must be positive")
        self.capacity = int(capacity)
        self._values = np.empty(self.capacity, dtype=np.float64)
        self._rng = np.random.default_rng(seed)
        self.count = 0
        self.retained = 0

    def add(self, values: np.ndarray) -> None:
        clean = np.asarray(values, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        for value in clean:
            self.count += 1
            if self.retained < self.capacity:
                self._values[self.retained] = value
                self.retained += 1
                continue
            replacement = int(self._rng.integers(0, self.count))
            if replacement < self.capacity:
                self._values[replacement] = value

    def values(self) -> np.ndarray:
        return self._values[: self.retained].copy()


def calibrate_usa_tca(
    daily_features_path: str | Path,
    eligible_universe_tiers_path: str | Path,
    *,
    start_date: str | date,
    end_date: str | date,
    commission: FeeInput,
    exchange_fee: FeeInput,
    regulatory_sell_fee: FeeInput,
    artifact_root: str | Path = "artifacts/usa_tca_calibration",
    run_id: str | None = None,
    reference_participation_rate: float = 0.10,
    max_participation_rate: float = 0.10,
    market_data_source_name: str = "usa_l2_daily_features_u2",
    market_data_source_quality: str = "derived_ohlcv_market_data",
    winsor_lower_quantile: float = 0.01,
    winsor_upper_quantile: float = 0.99,
    stress_quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99),
    batch_size: int = 65_536,
    max_calibration_samples: int = 250_000,
    random_seed: int = 17,
    overwrite: bool = False,
) -> USATCACalibrationArtifacts:
    """Calibrate research-only USA TCA scalars from U2 daily market data.

    The parquet scan is predicate-pushed and batch bounded. Distributional statistics
    use a deterministic fixed-capacity reservoir; total observation and coverage
    counts are still measured over every filtered row.
    """

    daily_path = Path(daily_features_path)
    tiers_path = Path(eligible_universe_tiers_path)
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if end < start:
        raise ValueError("end_date cannot be before start_date")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0 < reference_participation_rate <= 1:
        raise ValueError("reference_participation_rate must be in (0, 1]")
    if not 0 < max_participation_rate <= 1:
        raise ValueError("max_participation_rate must be in (0, 1]")
    if max_participation_rate < reference_participation_rate:
        raise ValueError("max_participation_rate cannot be below reference_participation_rate")
    if not 0 <= winsor_lower_quantile < winsor_upper_quantile <= 1:
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")
    if not stress_quantiles or any(not 0 <= value <= 1 for value in stress_quantiles):
        raise ValueError("stress_quantiles must contain values in [0, 1]")
    if not market_data_source_name.strip() or not market_data_source_quality.strip():
        raise ValueError("market data source name and quality cannot be empty")
    for label, fee in (
        ("commission", commission),
        ("exchange_fee", exchange_fee),
        ("regulatory_sell_fee", regulatory_sell_fee),
    ):
        fee.validate(label)

    run_id = run_id or datetime.now(timezone.utc).strftime("USA-TCA-CALIBRATION-%Y%m%d-%H%M%S")
    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"artifact directory is not empty: {artifact_dir}")

    u2_tickers = _load_u2_tickers(tiers_path)
    scan = _scan_daily_calibration_values(
        daily_path=daily_path,
        u2_tickers=u2_tickers,
        start=start,
        end=end,
        reference_participation_rate=reference_participation_rate,
        batch_size=batch_size,
        max_calibration_samples=max_calibration_samples,
        random_seed=random_seed,
    )
    spread_statistics = _robust_statistics(
        scan["spread_values"],
        total_count=scan["spread_sample_count"],
        winsor_lower=winsor_lower_quantile,
        winsor_upper=winsor_upper_quantile,
        requested_quantiles=stress_quantiles,
    )
    impact_statistics = _robust_statistics(
        scan["impact_values"],
        total_count=scan["impact_sample_count"],
        winsor_lower=winsor_lower_quantile,
        winsor_upper=winsor_upper_quantile,
        requested_quantiles=stress_quantiles,
    )

    assumptions = TCAAssumptions(
        commission_bps=float(commission.bps),
        exchange_fee_bps=float(exchange_fee.bps),
        regulatory_sell_fee_bps=float(regulatory_sell_fee.bps),
        half_spread_bps=float(spread_statistics["calibrated_median_bps"]),
        impact_bps_at_reference_participation=float(
            impact_statistics["calibrated_median_bps"]
        ),
        reference_participation_rate=float(reference_participation_rate),
        max_participation_rate=float(max_participation_rate),
        source_quality=CALIBRATION_TYPE,
        source_name=market_data_source_name,
    )
    assumptions.validate()

    input_checksums = {
        "daily_features": _sha256(daily_path),
        "eligible_universe_tiers": _sha256(tiers_path),
    }
    stress_grid = _build_stress_grid(
        spread_statistics=spread_statistics,
        impact_statistics=impact_statistics,
        quantiles=stress_quantiles,
        commission=commission,
        exchange_fee=exchange_fee,
        regulatory_sell_fee=regulatory_sell_fee,
    )
    evidence = {
        "calibration_type": CALIBRATION_TYPE,
        "research_only": True,
        "live_trading_ready": False,
        "ibkr_ready": False,
        "production_ready": False,
        "market_data_source": {
            "name": market_data_source_name,
            "quality": market_data_source_quality,
            "daily_features_path": str(daily_path.resolve()),
            "eligible_universe_tiers_path": str(tiers_path.resolve()),
            "sha256": input_checksums,
        },
        "filters": {
            "universe_tier": "U2",
            "eligibility_column": "u2_backtest_eligible",
            "requested_start_date": start.isoformat(),
            "requested_end_date": end.isoformat(),
            "eligible_entities_in_tier_file": len(u2_tickers),
        },
        "coverage": {
            "filtered_rows": scan["filtered_rows"],
            "entities": scan["entities"],
            "entity_count": len(scan["entities"]),
            "observed_date_min": scan["observed_date_min"],
            "observed_date_max": scan["observed_date_max"],
            "spread_entities": scan["spread_entity_count"],
            "impact_entities": scan["impact_entity_count"],
        },
        "formulas": {
            "half_spread_bps": SPREAD_FORMULA,
            "impact_bps_at_reference_participation": IMPACT_FORMULA,
            "spread_interpretation": (
                "Conservative one-way research proxy: half the normalized daily range. "
                "The range contains intraday volatility as well as spread, so it should "
                "not be interpreted as a quoted or realized spread."
            ),
            "impact_interpretation": (
                "Lagged 60-session Amihud price response multiplied by lagged 60-session "
                "ADV and the configured participation rate."
            ),
        },
        "robust_estimation": {
            "method": "winsorized_median_from_deterministic_bounded_reservoir",
            "winsor_lower_quantile": winsor_lower_quantile,
            "winsor_upper_quantile": winsor_upper_quantile,
            "random_seed": random_seed,
            "max_calibration_samples_per_metric": max_calibration_samples,
            "spread": spread_statistics,
            "impact": impact_statistics,
        },
        "fees": {
            "commission": asdict(commission),
            "exchange_fee": asdict(exchange_fee),
            "regulatory_sell_fee": asdict(regulatory_sell_fee),
        },
        "scanner": {
            "engine": "pyarrow.dataset.Scanner",
            "predicate_pushdown": ["ticker in U2", "date >= start_date", "date <= end_date"],
            "use_threads": False,
            "configured_batch_size": batch_size,
            "batches_processed": scan["batches_processed"],
            "max_batch_rows_observed": scan["max_batch_rows_observed"],
            "memory_bounded": True,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    artifact_dir.mkdir(parents=True, exist_ok=True)
    assumptions_json_path = artifact_dir / "tca_assumptions.json"
    assumptions_csv_path = artifact_dir / "tca_assumptions.csv"
    evidence_path = artifact_dir / "calibration_evidence.json"
    stress_grid_path = artifact_dir / "stress_grid.csv"
    manifest_path = artifact_dir / "manifest.json"

    scalar_payload = asdict(assumptions)
    assumptions_json_path.write_text(
        json.dumps(scalar_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pd.DataFrame([scalar_payload]).to_csv(assumptions_csv_path, index=False)
    stress_grid.to_csv(stress_grid_path, index=False)
    evidence_path.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    artifact_paths = {
        "tca_assumptions_json": assumptions_json_path,
        "tca_assumptions_csv": assumptions_csv_path,
        "calibration_evidence": evidence_path,
        "stress_grid": stress_grid_path,
    }
    artifact_checksums = {name: _sha256(path) for name, path in artifact_paths.items()}
    manifest = {
        "run_id": run_id,
        "status": CALIBRATION_TYPE,
        "calibration_type": CALIBRATION_TYPE,
        "research_only": True,
        "live_trading_ready": False,
        "ibkr_ready": False,
        "production_ready": False,
        "assumptions_compatibility": "quant_lab.data.usa_exchange_calendar_tca.TCAAssumptions",
        "sample_count": {
            "filtered_rows": scan["filtered_rows"],
            "spread_valid": scan["spread_sample_count"],
            "impact_valid": scan["impact_sample_count"],
            "spread_retained": spread_statistics["sample_count_retained"],
            "impact_retained": impact_statistics["sample_count_retained"],
        },
        "entities": {
            "count": len(scan["entities"]),
            "values": scan["entities"],
        },
        "dates": {
            "requested_start": start.isoformat(),
            "requested_end": end.isoformat(),
            "observed_start": scan["observed_date_min"],
            "observed_end": scan["observed_date_max"],
        },
        "quantiles": {
            "spread_bps": spread_statistics["winsorized_quantiles_bps"],
            "impact_bps": impact_statistics["winsorized_quantiles_bps"],
        },
        "formulas": evidence["formulas"],
        "scanner": evidence["scanner"],
        "input_sha256": input_checksums,
        "artifacts": {name: str(path.resolve()) for name, path in artifact_paths.items()},
        "artifact_sha256": artifact_checksums,
        "generated_at_utc": evidence["generated_at_utc"],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return USATCACalibrationArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        assumptions_json_path=assumptions_json_path,
        assumptions_csv_path=assumptions_csv_path,
        evidence_path=evidence_path,
        stress_grid_path=stress_grid_path,
        manifest_path=manifest_path,
        assumptions=assumptions,
        stress_grid=stress_grid,
        evidence=evidence,
        manifest=manifest,
    )


def _load_u2_tickers(path: Path) -> tuple[str, ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    header = pd.read_csv(path, nrows=0)
    required = {"ticker", "u2_backtest_eligible"}
    missing = required - set(header.columns)
    if missing:
        raise ValueError(f"eligible universe is missing columns: {sorted(missing)}")
    eligible_tickers: set[str] = set()
    chunks = pd.read_csv(
        path,
        usecols=["ticker", "u2_backtest_eligible"],
        dtype={"ticker": "string"},
        chunksize=50_000,
    )
    for tiers in chunks:
        eligible = _coerce_boolean(tiers["u2_backtest_eligible"])
        eligible_tickers.update(tiers.loc[eligible, "ticker"].dropna().astype(str))
    tickers = tuple(sorted(eligible_tickers))
    if not tickers:
        raise ValueError("eligible universe contains no U2 tickers")
    return tickers


def _scan_daily_calibration_values(
    *,
    daily_path: Path,
    u2_tickers: tuple[str, ...],
    start: date,
    end: date,
    reference_participation_rate: float,
    batch_size: int,
    max_calibration_samples: int,
    random_seed: int,
) -> dict[str, Any]:
    if not daily_path.exists():
        raise FileNotFoundError(daily_path)
    dataset = ds.dataset(daily_path, format="parquet")
    missing = set(REQUIRED_DAILY_COLUMNS) - set(dataset.schema.names)
    if missing:
        raise ValueError(f"daily features are missing columns: {sorted(missing)}")
    date_type = dataset.schema.field("date").type
    start_scalar = _temporal_scalar(start, date_type, end_of_day=False)
    end_scalar = _temporal_scalar(end, date_type, end_of_day=True)
    predicate = (
        ds.field("ticker").isin(u2_tickers)
        & (ds.field("date") >= start_scalar)
        & (ds.field("date") <= end_scalar)
    )
    scanner = dataset.scanner(
        columns=list(REQUIRED_DAILY_COLUMNS),
        filter=predicate,
        batch_size=batch_size,
        batch_readahead=1,
        fragment_readahead=1,
        use_threads=False,
    )
    spread_reservoir = _BoundedReservoir(max_calibration_samples, random_seed)
    impact_reservoir = _BoundedReservoir(max_calibration_samples, random_seed + 1)
    entities: set[str] = set()
    spread_entities: set[str] = set()
    impact_entities: set[str] = set()
    filtered_rows = 0
    batches_processed = 0
    max_batch_rows_observed = 0
    observed_min: pd.Timestamp | None = None
    observed_max: pd.Timestamp | None = None

    for batch in scanner.to_batches():
        rows = batch.num_rows
        if rows == 0:
            continue
        batches_processed += 1
        filtered_rows += rows
        max_batch_rows_observed = max(max_batch_rows_observed, rows)
        frame = batch.to_pandas()
        tickers = frame["ticker"].astype("string")
        entities.update(tickers.dropna().astype(str))
        dates = pd.to_datetime(frame["date"], errors="coerce")
        if dates.notna().any():
            batch_min = dates.min()
            batch_max = dates.max()
            observed_min = batch_min if observed_min is None else min(observed_min, batch_min)
            observed_max = batch_max if observed_max is None else max(observed_max, batch_max)

        high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
        low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
        bad_ohlc = _coerce_boolean(frame["bad_ohlc_flag"]).to_numpy(dtype=bool)
        midpoint = (high + low) / 2.0
        spread = 0.5 * (high - low) / midpoint * 10_000.0
        spread_valid = (
            np.isfinite(spread)
            & np.isfinite(high)
            & np.isfinite(low)
            & (high >= low)
            & (low > 0)
            & (midpoint > 0)
            & ~bad_ohlc
            & (spread >= 0)
        )
        spread_reservoir.add(spread[spread_valid])
        spread_entities.update(tickers.loc[spread_valid].dropna().astype(str))

        amihud = pd.to_numeric(frame["amihud_60_lag1"], errors="coerce").to_numpy(dtype=float)
        adv = pd.to_numeric(frame["adv_60_local_lag1"], errors="coerce").to_numpy(dtype=float)
        impact = amihud * adv * reference_participation_rate * 10_000.0
        impact_valid = (
            np.isfinite(impact)
            & np.isfinite(amihud)
            & np.isfinite(adv)
            & (amihud >= 0)
            & (adv > 0)
            & (impact >= 0)
        )
        impact_reservoir.add(impact[impact_valid])
        impact_entities.update(tickers.loc[impact_valid].dropna().astype(str))

    if filtered_rows == 0:
        raise ValueError("no daily feature rows matched the U2 and date filters")
    if spread_reservoir.count == 0:
        raise ValueError("no valid daily high-low observations matched the calibration filters")
    if impact_reservoir.count == 0:
        raise ValueError("no valid lagged Amihud and ADV observations matched the filters")
    return {
        "spread_values": spread_reservoir.values(),
        "impact_values": impact_reservoir.values(),
        "spread_sample_count": spread_reservoir.count,
        "impact_sample_count": impact_reservoir.count,
        "filtered_rows": filtered_rows,
        "entities": sorted(entities),
        "spread_entity_count": len(spread_entities),
        "impact_entity_count": len(impact_entities),
        "observed_date_min": observed_min.date().isoformat() if observed_min is not None else None,
        "observed_date_max": observed_max.date().isoformat() if observed_max is not None else None,
        "batches_processed": batches_processed,
        "max_batch_rows_observed": max_batch_rows_observed,
    }


def _robust_statistics(
    values: np.ndarray,
    *,
    total_count: int,
    winsor_lower: float,
    winsor_upper: float,
    requested_quantiles: Iterable[float],
) -> dict[str, Any]:
    if values.size == 0:
        raise ValueError("cannot estimate statistics from an empty sample")
    lower_value, upper_value = np.quantile(
        values, [winsor_lower, winsor_upper], method="linear"
    )
    winsorized = np.clip(values, lower_value, upper_value)
    levels = tuple(sorted(set((winsor_lower, 0.50, winsor_upper, *requested_quantiles))))
    raw = np.quantile(values, levels, method="linear")
    robust = np.quantile(winsorized, levels, method="linear")
    return {
        "sample_count_total": int(total_count),
        "sample_count_retained": int(values.size),
        "bounded_sample_approximation": bool(total_count > values.size),
        "winsor_lower_value_bps": float(lower_value),
        "winsor_upper_value_bps": float(upper_value),
        "raw_quantiles_bps": {
            _quantile_label(level): float(value) for level, value in zip(levels, raw)
        },
        "winsorized_quantiles_bps": {
            _quantile_label(level): float(value) for level, value in zip(levels, robust)
        },
        "calibrated_median_bps": float(np.median(winsorized)),
    }


def _build_stress_grid(
    *,
    spread_statistics: dict[str, Any],
    impact_statistics: dict[str, Any],
    quantiles: Iterable[float],
    commission: FeeInput,
    exchange_fee: FeeInput,
    regulatory_sell_fee: FeeInput,
) -> pd.DataFrame:
    rows = []
    spread_quantiles = spread_statistics["winsorized_quantiles_bps"]
    impact_quantiles = impact_statistics["winsorized_quantiles_bps"]
    for quantile in sorted(set(quantiles)):
        label = _quantile_label(quantile)
        spread_bps = float(spread_quantiles[label])
        impact_bps = float(impact_quantiles[label])
        common = commission.bps + exchange_fee.bps + spread_bps + impact_bps
        rows.append(
            {
                "percentile": quantile,
                "percentile_label": label,
                "half_spread_bps": spread_bps,
                "impact_bps_at_reference_participation": impact_bps,
                "commission_bps": commission.bps,
                "exchange_fee_bps": exchange_fee.bps,
                "regulatory_sell_fee_bps": regulatory_sell_fee.bps,
                "estimated_buy_cost_bps": common,
                "estimated_sell_cost_bps": common + regulatory_sell_fee.bps,
                "calibration_type": CALIBRATION_TYPE,
                "live_trading_ready": False,
                "ibkr_ready": False,
            }
        )
    return pd.DataFrame(rows)


def _temporal_scalar(value: date, arrow_type: pa.DataType, *, end_of_day: bool) -> Any:
    if pa.types.is_date(arrow_type):
        return value
    if pa.types.is_timestamp(arrow_type):
        clock = time.max if end_of_day else time.min
        return datetime.combine(value, clock)
    raise TypeError(f"daily features date column must be date or timestamp, got {arrow_type}")


def _parse_date(value: str | date, field_name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"{field_name} must be a valid date")
    return parsed.date()


def _coerce_boolean(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(values, errors="coerce")
    return normalized.isin({"true", "1", "1.0", "yes", "y", "t"}) | (
        numeric.notna() & numeric.ne(0)
    )


def _quantile_label(value: float) -> str:
    percentage = value * 100
    if float(percentage).is_integer():
        return f"p{int(percentage):02d}"
    return "p" + f"{percentage:.4f}".rstrip("0").rstrip(".").replace(".", "_")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
