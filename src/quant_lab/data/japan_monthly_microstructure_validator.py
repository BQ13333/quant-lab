from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data.japan_monthly_microstructure import _preferred_unit_source


UTC = timezone.utc
VERSION = "1.0-independent"
VALID_UNITS = {1, 5, 10, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 10000}


@dataclass(frozen=True)
class JapanMonthlyMicrostructureValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def write_japan_monthly_microstructure_validation(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_monthly_microstructure_validation",
) -> JapanMonthlyMicrostructureValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be path-safe")
    producer = Path(producer_dir)
    root = Path(artifact_root) / run_id
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    source_manifest_path = producer / "manifest.json"
    source_manifest = _json(source_manifest_path)
    blockers = []
    if run_id == source_manifest.get("run_id"):
        blockers.append("independence_failure_same_run_id")
    if not (
        source_manifest.get("status") == "complete_candidate"
        and source_manifest.get("historical_lot_size_candidate") is True
        and source_manifest.get("scorecard_eligible") is False
        and source_manifest.get("independent_validation_required") is True
    ):
        blockers.append("producer_not_complete_candidate")
    for filename, expected in source_manifest.get("output_sha256", {}).items():
        path = producer / filename
        if not path.is_file() or _sha256(path) != expected:
            blockers.append(f"producer_output_hash_mismatch:{filename}")
    plan_path = producer / "file_plan.csv"
    if not plan_path.is_file() or _sha256(plan_path) != source_manifest.get("input_plan_sha256"):
        blockers.append("producer_plan_hash_mismatch")
    history_path = producer / "monthly_lot_size_history.parquet"
    raw_path = producer / "raw_lot_size_observations.parquet"
    state_path = producer / "file_quality.csv"
    unresolved_path = producer / "unresolved_lot_sizes.csv"
    conflicts_path = producer / "lot_size_conflicts.csv"
    venue_path = producer / "venue_specific_lot_size_variants.csv"
    required = [history_path, raw_path, state_path, unresolved_path, conflicts_path, venue_path]
    if not all(path.is_file() for path in required):
        blockers.append("producer_required_artifacts_missing")
    history = pd.read_parquet(history_path) if history_path.is_file() else pd.DataFrame()
    raw = pd.read_parquet(raw_path) if raw_path.is_file() else pd.DataFrame()
    state = pd.read_csv(state_path, dtype=str).fillna("") if state_path.is_file() else pd.DataFrame()
    unresolved = pd.read_csv(unresolved_path) if unresolved_path.is_file() else pd.DataFrame([{}])
    conflicts = pd.read_csv(conflicts_path) if conflicts_path.is_file() else pd.DataFrame([{}])
    venue = pd.read_csv(venue_path) if venue_path.is_file() else pd.DataFrame()
    blockers.extend(_history_blockers(history))
    blockers.extend(_raw_reconciliation_blockers(history, raw))
    if not unresolved.empty:
        blockers.append(f"unresolved_rows={len(unresolved)}")
    if not conflicts.empty:
        blockers.append(f"same_venue_conflict_rows={len(conflicts)}")
    if state.empty or not state["status"].eq("completed").all():
        blockers.append("file_state_not_fully_complete")
    else:
        expected_rows = pd.to_numeric(state["expected_rows"], errors="coerce")
        parsed_rows = pd.to_numeric(state["parsed_rows"], errors="coerce")
        parser_conflicts = pd.to_numeric(state["conflict_codes"], errors="coerce")
        if expected_rows.isna().any() or parsed_rows.isna().any() or not expected_rows.eq(parsed_rows).all():
            blockers.append("file_denominator_not_exact")
        if parser_conflicts.isna().any() or not parser_conflicts.eq(0).all():
            blockers.append("file_parser_conflicts_present")
    if not venue.empty and not venue["unit_variants"].astype(int).gt(1).all():
        blockers.append("venue_variant_ledger_invalid")
    output_history = root / "validated_monthly_lot_size_history.parquet"
    if not blockers:
        shutil.copy2(history_path, output_history)
    else:
        pd.DataFrame(columns=history.columns).to_parquet(output_history, index=False)
    status = "pass_component" if not blockers else "failed_validation"
    manifest = {
        "schema_version": VERSION,
        "run_id": run_id,
        "validator_run_id": run_id,
        "producer_run_id": source_manifest.get("run_id"),
        "artifact_type": "japan_monthly_microstructure_independent_validation",
        "status": status,
        "component": "lot_sizes",
        "component_scorecard_eligible": not blockers,
        "scorecard_eligible": False,
        "independent_validation": True,
        "official_jpx_source": True,
        "evidence_scope": "full_2006_07_to_2026_06_monthly_issuer_lot_sizes" if not blockers else "failed_no_admission",
        "critical_blocker_count": len(blockers),
        "critical_blockers": blockers,
        "counts": {
            "history_rows": len(history),
            "raw_source_rows": len(raw),
            "months": int(history["observation_month"].nunique()) if "observation_month" in history else 0,
            "venue_specific_variant_rows": len(venue),
        },
        "producer_manifest": str(source_manifest_path.resolve()),
        "producer_manifest_sha256": _sha256(source_manifest_path) if source_manifest_path.is_file() else "",
        "artifacts": {"validated_monthly_lot_size_history": str(output_history.resolve())},
        "output_sha256": {"validated_monthly_lot_size_history.parquet": _sha256(output_history)},
        "promotion_policy": {"alpha": False, "historical_backtest": False, "production": False, "ibkr": False, "live_trading": False},
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(manifest_path, manifest)
    return JapanMonthlyMicrostructureValidationArtifacts(run_id, root, manifest_path, manifest)


def _history_blockers(history: pd.DataFrame) -> list[str]:
    required = {
        "observation_month", "code", "known_at_utc", "source_families",
        "preferred_source_family", "trading_unit", "unit_mapping_status", "pit_usable",
    }
    if history.empty or not required.issubset(history.columns):
        return ["history_schema_or_rows_invalid"]
    blockers = []
    if history.duplicated(["observation_month", "code"]).any():
        blockers.append("history_duplicate_code_month_keys")
    periods = pd.PeriodIndex(history["observation_month"], freq="M")
    if list(sorted(periods.unique())) != list(pd.period_range("2006-07", "2026-06", freq="M")):
        blockers.append("history_month_denominator_invalid")
    units = pd.to_numeric(history["trading_unit"], errors="coerce")
    if units.isna().any() or not set(units.astype(int)).issubset(VALID_UNITS):
        blockers.append("history_trading_units_invalid")
    if not history["unit_mapping_status"].eq("direct_official_monthly_pdf").all():
        blockers.append("history_contains_non_direct_mapping")
    if not history["pit_usable"].astype(str).str.lower().isin({"true", "1", "yes"}).all():
        blockers.append("history_contains_non_pit_rows")
    expected_source = history["source_families"].map(_preferred_unit_source)
    if not expected_source.eq(history["preferred_source_family"]).all():
        blockers.append("history_preferred_venue_routing_invalid")
    known = pd.to_datetime(history["known_at_utc"], errors="coerce", utc=True)
    month_end = periods.to_timestamp(how="end").tz_localize("UTC")
    if known.isna().any() or not (known.reset_index(drop=True) > pd.Series(month_end)).all():
        blockers.append("history_known_at_not_after_observation_month")
    return blockers


def _raw_reconciliation_blockers(history: pd.DataFrame, raw: pd.DataFrame) -> list[str]:
    required = {"observation_month", "code", "trading_unit", "source_family", "unit_observed"}
    if raw.empty or not required.issubset(raw.columns) or history.empty:
        return ["raw_observation_schema_or_rows_invalid"]
    valid = raw.loc[raw["unit_observed"].astype(bool) & raw["trading_unit"].notna()].copy()
    by_source = valid.groupby(["observation_month", "code", "source_family"], as_index=False).agg(
        observed_unit=("trading_unit", "first"),
        variants=("trading_unit", "nunique"),
    )
    if by_source["variants"].gt(1).any():
        return ["raw_same_venue_unit_conflicts"]
    check = history.merge(
        by_source.rename(columns={"source_family": "preferred_source_family"}),
        on=["observation_month", "code", "preferred_source_family"],
        how="left",
        validate="one_to_one",
    )
    actual = pd.to_numeric(check["trading_unit"], errors="coerce")
    observed = pd.to_numeric(check["observed_unit"], errors="coerce")
    return [] if observed.notna().all() and actual.eq(observed).all() else ["history_does_not_reconcile_to_preferred_raw_venue"]


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
