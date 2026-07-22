from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.usa_exchange_calendar_tca import TCAAssumptions
from quant_lab.data.usa_tca_calibration import (
    FeeInput,
    _build_stress_grid,
    _load_u2_tickers,
    _robust_statistics,
    _scan_daily_calibration_values,
)


CALIBRATION_TYPE = "japan_liquidity_aware_market_data_calibration"


@dataclass(frozen=True)
class JapanG4TCACalibrationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def calibrate_japan_g4_tca(
    daily_features_path: str | Path,
    eligible_universe_tiers_path: str | Path,
    admitted_market_data_manifest_path: str | Path,
    *,
    start_date: str | date,
    end_date: str | date,
    commission: FeeInput,
    exchange_fee: FeeInput,
    regulatory_sell_fee: FeeInput,
    artifact_root: str | Path = "artifacts/japan_g4_tca_calibration",
    run_id: str = "JPN-G4-TCA-CALIBRATION",
    reference_participation_rate: float = 0.10,
    max_participation_rate: float = 0.10,
    winsor_lower_quantile: float = 0.01,
    winsor_upper_quantile: float = 0.99,
    stress_quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95, 0.99),
    batch_size: int = 65_536,
    max_calibration_samples: int = 250_000,
    random_seed: int = 17,
    minimum_calibration_rows: int = 1_000,
    minimum_entities: int = 10,
) -> JapanG4TCACalibrationArtifacts:
    """Produce a research-only Japan TCA candidate from admitted lagged liquidity data."""

    _path_safe(run_id)
    daily = Path(daily_features_path).resolve()
    tiers = Path(eligible_universe_tiers_path).resolve()
    admitted_manifest_path = Path(admitted_market_data_manifest_path).resolve()
    admitted = _read_json(admitted_manifest_path)
    start = _date_value(start_date, "start_date")
    end = _date_value(end_date, "end_date")
    if end < start:
        raise ValueError("end_date cannot be before start_date")
    blockers = _admitted_market_data_blockers(
        daily,
        tiers,
        admitted_manifest_path,
        admitted,
        start=start,
        end=end,
    )
    if blockers:
        raise ValueError("Japan TCA input admission failed: " + "; ".join(blockers))
    if minimum_calibration_rows <= 0 or minimum_entities <= 0:
        raise ValueError("minimum calibration rows and entities must be positive")
    for label, fee in (
        ("commission", commission),
        ("exchange_fee", exchange_fee),
        ("regulatory_sell_fee", regulatory_sell_fee),
    ):
        fee.validate(label)
    if not 0 < reference_participation_rate <= max_participation_rate <= 1:
        raise ValueError("participation rates must satisfy 0 < reference <= maximum <= 1")
    if not 0 <= winsor_lower_quantile < winsor_upper_quantile <= 1:
        raise ValueError("winsor quantiles must satisfy 0 <= lower < upper <= 1")

    root = (Path(artifact_root) / run_id).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)

    tickers = _load_u2_tickers(tiers)
    scan = _scan_daily_calibration_values(
        daily_path=daily,
        u2_tickers=tickers,
        start=start,
        end=end,
        reference_participation_rate=reference_participation_rate,
        batch_size=batch_size,
        max_calibration_samples=max_calibration_samples,
        random_seed=random_seed,
    )
    spread = _robust_statistics(
        scan["spread_values"],
        total_count=scan["spread_sample_count"],
        winsor_lower=winsor_lower_quantile,
        winsor_upper=winsor_upper_quantile,
        requested_quantiles=stress_quantiles,
    )
    impact = _robust_statistics(
        scan["impact_values"],
        total_count=scan["impact_sample_count"],
        winsor_lower=winsor_lower_quantile,
        winsor_upper=winsor_upper_quantile,
        requested_quantiles=stress_quantiles,
    )
    assumptions = TCAAssumptions(
        commission_bps=commission.bps,
        exchange_fee_bps=exchange_fee.bps,
        regulatory_sell_fee_bps=regulatory_sell_fee.bps,
        half_spread_bps=spread["calibrated_median_bps"],
        impact_bps_at_reference_participation=impact["calibrated_median_bps"],
        reference_participation_rate=reference_participation_rate,
        max_participation_rate=max_participation_rate,
        source_quality=CALIBRATION_TYPE,
        source_name="admitted_japan_daily_liquidity",
    )
    assumptions.validate()
    enough_samples = (
        scan["filtered_rows"] >= minimum_calibration_rows
        and len(scan["entities"]) >= minimum_entities
    )
    try:
        observed_start = _date_value(scan["observed_date_min"], "observed_date_min")
        observed_end = _date_value(scan["observed_date_max"], "observed_date_max")
        coverage_complete = observed_start <= start and observed_end >= end
    except ValueError:
        coverage_complete = False

    stress = _build_stress_grid(
        spread_statistics=spread,
        impact_statistics=impact,
        quantiles=stress_quantiles,
        commission=commission,
        exchange_fee=exchange_fee,
        regulatory_sell_fee=regulatory_sell_fee,
    )
    stress["calibration_type"] = CALIBRATION_TYPE
    evidence = {
        "calibration_type": CALIBRATION_TYPE,
        "currency": "JPY",
        "liquidity_aware": True,
        "research_only": True,
        "input_paths": {
            "daily_features": str(daily),
            "eligible_universe_tiers": str(tiers),
            "admitted_market_data_manifest": str(admitted_manifest_path),
        },
        "input_sha256": {
            "daily_features": _sha256(daily),
            "eligible_universe_tiers": _sha256(tiers),
            "admitted_market_data_manifest": _sha256(admitted_manifest_path),
        },
        "configuration": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "reference_participation_rate": reference_participation_rate,
            "max_participation_rate": max_participation_rate,
            "winsor_lower_quantile": winsor_lower_quantile,
            "winsor_upper_quantile": winsor_upper_quantile,
            "stress_quantiles": list(stress_quantiles),
            "batch_size": batch_size,
            "max_calibration_samples": max_calibration_samples,
            "random_seed": random_seed,
            "minimum_calibration_rows": minimum_calibration_rows,
            "minimum_entities": minimum_entities,
        },
        "fees": {
            "commission": asdict(commission),
            "exchange_fee": asdict(exchange_fee),
            "regulatory_sell_fee": asdict(regulatory_sell_fee),
        },
        "counts": {
            "calibration_rows": scan["filtered_rows"],
            "spread_rows": scan["spread_sample_count"],
            "impact_rows": scan["impact_sample_count"],
            "entities": len(scan["entities"]),
        },
        "coverage": {
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "observed_start_date": scan["observed_date_min"],
            "observed_end_date": scan["observed_date_max"],
            "complete": coverage_complete,
        },
        "minimum_sample_requirements_met": enough_samples,
        "spread_statistics": spread,
        "impact_statistics": impact,
    }
    assumptions_path = root / "tca_assumptions.json"
    evidence_path = root / "calibration_evidence.json"
    stress_path = root / "stress_grid.csv"
    _write_json(assumptions_path, asdict(assumptions))
    _write_json(evidence_path, evidence)
    stress.to_csv(stress_path, index=False)
    outputs = [assumptions_path, evidence_path, stress_path]
    manifest = {
        "schema_version": "1.0-candidate",
        "run_id": run_id,
        "status": "complete_candidate" if enough_samples and coverage_complete else "partial_candidate",
        "component": "tca_assumptions",
        "component_scorecard_eligible": False,
        "scorecard_eligible": False,
        "independent_validation_required": True,
        "currency": "JPY",
        "calibration_type": CALIBRATION_TYPE,
        "liquidity_aware": True,
        "minimum_sample_requirements_met": enough_samples,
        "coverage": evidence["coverage"],
        "counts": evidence["counts"],
        "input_paths": evidence["input_paths"],
        "input_sha256": evidence["input_sha256"],
        "artifacts": {path.stem: str(path) for path in outputs},
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "critical_blockers": [
            *([] if coverage_complete else ["requested_date_coverage_incomplete"]),
            *([] if enough_samples else ["minimum_calibration_sample_not_met"]),
        ],
        "promotion_policy": _promotion_policy(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return JapanG4TCACalibrationArtifacts(run_id, root, manifest_path, manifest)


def validate_japan_g4_tca_calibration(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_tca_calibration_validation",
) -> JapanG4TCACalibrationArtifacts:
    """Independently recompute a Japan TCA candidate and emit admission evidence."""

    _path_safe(run_id)
    producer = Path(producer_dir).resolve()
    producer_manifest_path = producer / "manifest.json"
    candidate = _read_json(producer_manifest_path)
    blockers: list[str] = []
    if run_id == candidate.get("run_id"):
        blockers.append("independence_failure_same_run_id")
    if not (
        candidate.get("status") == "complete_candidate"
        and candidate.get("component") == "tca_assumptions"
        and candidate.get("independent_validation_required") is True
        and candidate.get("currency") == "JPY"
        and candidate.get("liquidity_aware") is True
        and not candidate.get("critical_blockers")
    ):
        blockers.append("producer_not_complete_candidate")
    blockers.extend(_producer_output_blockers(producer, candidate))

    evidence_path = producer / "calibration_evidence.json"
    assumptions_path = producer / "tca_assumptions.json"
    evidence = _read_json(evidence_path)
    assumptions_payload = _read_json(assumptions_path)
    blockers.extend(_input_hash_blockers(evidence))
    if not blockers:
        paths = evidence["input_paths"]
        config = evidence["configuration"]
        blockers.extend(
            _admitted_market_data_blockers(
                Path(paths["daily_features"]),
                Path(paths["eligible_universe_tiers"]),
                Path(paths["admitted_market_data_manifest"]),
                _read_json(Path(paths["admitted_market_data_manifest"])),
                start=_date_value(config["start_date"], "start_date"),
                end=_date_value(config["end_date"], "end_date"),
            )
        )
    recomputed: dict[str, Any] | None = None
    if not blockers:
        try:
            recomputed = _recompute(evidence)
            _compare_assumptions(assumptions_payload, recomputed["assumptions"], blockers)
            _compare_recomputed_metadata(candidate, evidence, recomputed, blockers)
        except (FileNotFoundError, TypeError, ValueError, KeyError) as exc:
            blockers.append(f"independent_recompute_failed:{type(exc).__name__}:{exc}")

    root = (Path(artifact_root) / run_id).resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"artifact directory is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    validated_path = root / "validated_tca_assumptions.json"
    if blockers:
        _write_json(validated_path, {})
    else:
        shutil.copy2(assumptions_path, validated_path)
    summary_path = root / "validation_summary.csv"
    pd.DataFrame(
        [{"check": "independent_recompute", "passed": not blockers, "blockers": "|".join(blockers)}]
    ).to_csv(summary_path, index=False)
    outputs = [validated_path, summary_path]
    counts = candidate.get("counts", {})
    manifest = {
        "schema_version": "1.0-independent",
        "run_id": run_id,
        "status": "pass_component" if not blockers else "failed_validation",
        "component": "tca_assumptions",
        "component_scorecard_eligible": not blockers,
        "scorecard_eligible": False,
        "independent_validation": True,
        "currency": "JPY",
        "calibration_type": CALIBRATION_TYPE,
        "liquidity_aware": True,
        "minimum_sample_requirements_met": candidate.get("minimum_sample_requirements_met") is True,
        "coverage": candidate.get("coverage", {}),
        "counts": counts,
        "critical_blocker_count": len(blockers),
        "critical_blockers": blockers,
        "producer_manifest": str(producer_manifest_path),
        "producer_manifest_sha256": _sha256(producer_manifest_path) if producer_manifest_path.is_file() else "",
        "artifacts": {path.stem: str(path) for path in outputs},
        "output_sha256": {path.name: _sha256(path) for path in outputs},
        "promotion_policy": _promotion_policy(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return JapanG4TCACalibrationArtifacts(run_id, root, manifest_path, manifest)


def _recompute(evidence: dict[str, Any]) -> dict[str, Any]:
    paths = evidence["input_paths"]
    config = evidence["configuration"]
    fees = evidence["fees"]
    tickers = _load_u2_tickers(Path(paths["eligible_universe_tiers"]))
    scan = _scan_daily_calibration_values(
        daily_path=Path(paths["daily_features"]),
        u2_tickers=tickers,
        start=_date_value(config["start_date"], "start_date"),
        end=_date_value(config["end_date"], "end_date"),
        reference_participation_rate=float(config["reference_participation_rate"]),
        batch_size=int(config["batch_size"]),
        max_calibration_samples=int(config["max_calibration_samples"]),
        random_seed=int(config["random_seed"]),
    )
    spread = _robust_statistics(
        scan["spread_values"],
        total_count=scan["spread_sample_count"],
        winsor_lower=float(config["winsor_lower_quantile"]),
        winsor_upper=float(config["winsor_upper_quantile"]),
        requested_quantiles=tuple(config["stress_quantiles"]),
    )
    impact = _robust_statistics(
        scan["impact_values"],
        total_count=scan["impact_sample_count"],
        winsor_lower=float(config["winsor_lower_quantile"]),
        winsor_upper=float(config["winsor_upper_quantile"]),
        requested_quantiles=tuple(config["stress_quantiles"]),
    )
    counts = {
        "calibration_rows": scan["filtered_rows"],
        "spread_rows": scan["spread_sample_count"],
        "impact_rows": scan["impact_sample_count"],
        "entities": len(scan["entities"]),
    }
    enough_samples = (
        counts["calibration_rows"] >= int(config["minimum_calibration_rows"])
        and counts["entities"] >= int(config["minimum_entities"])
    )
    return {
        "assumptions": {
            "commission_bps": float(fees["commission"]["bps"]),
            "exchange_fee_bps": float(fees["exchange_fee"]["bps"]),
            "regulatory_sell_fee_bps": float(fees["regulatory_sell_fee"]["bps"]),
            "half_spread_bps": spread["calibrated_median_bps"],
            "impact_bps_at_reference_participation": impact["calibrated_median_bps"],
            "reference_participation_rate": float(config["reference_participation_rate"]),
            "max_participation_rate": float(config["max_participation_rate"]),
            "source_quality": CALIBRATION_TYPE,
            "source_name": "admitted_japan_daily_liquidity",
        },
        "counts": counts,
        "coverage": {
            "start_date": config["start_date"],
            "end_date": config["end_date"],
            "observed_start_date": scan["observed_date_min"],
            "observed_end_date": scan["observed_date_max"],
            "complete": True,
        },
        "minimum_sample_requirements_met": enough_samples,
    }


def _compare_recomputed_metadata(
    candidate: dict[str, Any],
    evidence: dict[str, Any],
    recomputed: dict[str, Any],
    blockers: list[str],
) -> None:
    if candidate.get("counts") != evidence.get("counts") or candidate.get("counts") != recomputed["counts"]:
        blockers.append("calibration_count_reconciliation_failed")
    if candidate.get("coverage") != evidence.get("coverage") or candidate.get("coverage") != recomputed["coverage"]:
        blockers.append("calibration_coverage_reconciliation_failed")
    flags = {
        candidate.get("minimum_sample_requirements_met"),
        evidence.get("minimum_sample_requirements_met"),
        recomputed["minimum_sample_requirements_met"],
    }
    if flags != {True}:
        blockers.append("calibration_minimum_sample_reconciliation_failed")


def _compare_assumptions(
    observed: dict[str, Any], expected: dict[str, Any], blockers: list[str]
) -> None:
    if set(observed) != set(expected):
        blockers.append("assumption_schema_mismatch")
        return
    for key, value in expected.items():
        actual = observed.get(key)
        if isinstance(value, float):
            try:
                matches = math.isclose(float(actual), value, rel_tol=1e-12, abs_tol=1e-12)
            except (TypeError, ValueError):
                matches = False
        else:
            matches = actual == value
        if not matches:
            blockers.append(f"assumption_recompute_mismatch:{key}")


def _admitted_market_data_blockers(
    daily: Path,
    tiers: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    *,
    start: date,
    end: date,
) -> list[str]:
    blockers = []
    if not (
        manifest.get("status") in {"complete", "pass_component"}
        and manifest.get("currency") == "JPY"
        and manifest.get("admitted_for_research") is True
        and manifest.get("liquidity_complete") is True
        and not manifest.get("critical_blockers")
    ):
        blockers.append("admitted_market_data_contract_invalid")
    if not manifest_path.is_file():
        blockers.append("admitted_market_data_manifest_missing")
    artifacts = manifest.get("artifacts", {})
    declared = [Path(str(value)).resolve() for value in artifacts.values()] if isinstance(artifacts, dict) else []
    for label, path in (("daily_features", daily), ("eligible_universe_tiers", tiers)):
        if path not in declared:
            blockers.append(f"{label}_not_declared_by_admitted_manifest")
        expected = _declared_digest(manifest, path)
        if not path.is_file() or expected is None or _sha256(path) != expected:
            blockers.append(f"{label}_admitted_hash_mismatch")
    coverage = manifest.get("coverage", {})
    try:
        admitted_start = date.fromisoformat(str(coverage.get("start_date"))[:10])
        admitted_end = date.fromisoformat(str(coverage.get("end_date"))[:10])
    except ValueError:
        admitted_start = admitted_end = None
    if not (
        isinstance(coverage, dict)
        and coverage.get("complete") is True
        and admitted_start is not None
        and admitted_start <= start
        and admitted_end is not None
        and admitted_end >= end
    ):
        blockers.append("admitted_market_data_date_coverage_incomplete")
    return blockers


def _declared_digest(manifest: dict[str, Any], path: Path) -> str | None:
    for key in ("sha256", "output_sha256", "artifact_sha256"):
        values = manifest.get(key, {})
        if not isinstance(values, dict):
            continue
        for name, digest in values.items():
            if Path(str(name)).name == path.name and len(str(digest)) == 64:
                return str(digest)
    return None


def _producer_output_blockers(root: Path, manifest: dict[str, Any]) -> list[str]:
    hashes = manifest.get("output_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        return ["producer_hash_inventory_empty"]
    return [
        f"producer_output_hash_mismatch:{name}"
        for name, digest in hashes.items()
        if not (root / name).is_file() or _sha256(root / name) != digest
    ]


def _input_hash_blockers(evidence: dict[str, Any]) -> list[str]:
    paths = evidence.get("input_paths", {})
    hashes = evidence.get("input_sha256", {})
    blockers = []
    for key in ("daily_features", "eligible_universe_tiers", "admitted_market_data_manifest"):
        path = Path(str(paths.get(key, "")))
        if not path.is_file() or _sha256(path) != hashes.get(key):
            blockers.append(f"calibration_input_hash_mismatch:{key}")
    return blockers


def _promotion_policy() -> dict[str, bool]:
    return {
        "alpha": False,
        "historical_backtest": False,
        "production": False,
        "ibkr": False,
        "live_trading": False,
    }


def _path_safe(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")


def _date_value(value: str | date, name: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid date") from exc


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
