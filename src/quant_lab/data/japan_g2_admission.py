from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PASS_STATUSES = {"accepted", "complete", "pass", "passed"}
_LIQUIDITY_DATASET_ALIASES = {
    "liquidity_pit",
    "liquidity_market_cap_free_float",
    "pit_liquidity",
}


@dataclass(frozen=True)
class JapanG2OHLCVRun:
    historical_dir: Path
    depth_dir: Path
    compaction_dir: Path
    hash_inventory_path: Path
    row_equality_dir: Path | None = None


@dataclass(frozen=True)
class JapanG2AdmissionInputs:
    g1_security_master_dir: Path
    g1_validation_dir: Path
    active_symbol_map_dir: Path
    reference_symbol_map_dir: Path
    active_ohlcv: JapanG2OHLCVRun
    reference_ohlcv: JapanG2OHLCVRun
    corporate_actions_dir: Path
    adjusted_returns_dir: Path
    liquidity_dir: Path
    component_hash_inventory_path: Path


@dataclass(frozen=True)
class JapanG2AdmissionResult:
    run_id: str
    artifact_dir: Path
    admitted: bool
    manifest_path: Path
    checks_path: Path
    source_hashes_path: Path
    summary_path: Path
    manifest: dict[str, Any]
    checks: pd.DataFrame


class _Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        component: str,
        check_id: str,
        passed: Any,
        *,
        numerator: Any = None,
        denominator: Any = None,
        detail: str = "",
    ) -> bool:
        value = bool(passed)
        self.rows.append(
            {
                "component": component,
                "check_id": check_id,
                "passed": value,
                "numerator": numerator,
                "denominator": denominator,
                "detail": detail,
            }
        )
        return value

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self.rows,
            columns=["component", "check_id", "passed", "numerator", "denominator", "detail"],
        )


def write_japan_g2_admission(
    inputs: JapanG2AdmissionInputs,
    artifact_root: str | Path = "artifacts/japan_g2_admission",
    run_id: str | None = None,
) -> JapanG2AdmissionResult:
    """Publish an immutable, fail-closed aggregate admission contract for Japan G2."""

    run_id = run_id or datetime.now(timezone.utc).strftime("JPN-G2-ADMISSION-%Y%m%d-%H%M%S")
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be one non-empty path component")
    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists():
        raise FileExistsError(f"immutable G2 admission artifact already exists: {artifact_dir}")

    checks = _Checks()
    required_hashes: dict[str, set[Path]] = {"active": set(), "reference": set(), "components": set()}
    source_state: dict[str, Any] = {}

    denominator, role_windows = _evaluate_g1(inputs, checks, required_hashes["components"])
    active_raw, active_run_id = _evaluate_ohlcv_lane(
        "active", inputs.active_ohlcv, role_windows.get("active", denominator), checks, required_hashes["active"]
    )
    reference_raw, reference_run_id = _evaluate_ohlcv_lane(
        "reference",
        inputs.reference_ohlcv,
        role_windows.get("reference", denominator),
        checks,
        required_hashes["reference"],
    )
    checks.add(
        "ohlcv",
        "separate_active_reference_runs",
        bool(active_run_id and reference_run_id and active_run_id != reference_run_id),
        detail=f"active={active_run_id or '<missing>'}; reference={reference_run_id or '<missing>'}",
    )
    checks.add(
        "ohlcv",
        "separate_active_reference_hash_inventories",
        inputs.active_ohlcv.hash_inventory_path.resolve()
        != inputs.reference_ohlcv.hash_inventory_path.resolve(),
    )

    corporate_events = _evaluate_corporate_actions(
        inputs.corporate_actions_dir, denominator, checks, required_hashes["components"]
    )
    raw_union = _union_raw_keys(active_raw, reference_raw)
    _evaluate_adjusted_returns(
        inputs.adjusted_returns_dir,
        corporate_events,
        raw_union,
        checks,
        required_hashes["components"],
    )
    _evaluate_liquidity(
        inputs.liquidity_dir, denominator, checks, required_hashes["components"]
    )

    inventories = {
        "active": inputs.active_ohlcv.hash_inventory_path,
        "reference": inputs.reference_ohlcv.hash_inventory_path,
        "components": inputs.component_hash_inventory_path,
    }
    verified_files: dict[str, dict[Path, str]] = {}
    for name, inventory_path in inventories.items():
        mapping, issues = _verify_hash_inventory(inventory_path, required_hashes[name])
        verified_files[name] = mapping
        checks.add(
            f"hashes_{name}",
            "sha256_inventory_complete_and_verified",
            not issues,
            numerator=len(mapping),
            denominator=len(required_hashes[name]),
            detail=" | ".join(issues),
        )
        source_state[name] = {
            "inventory_path": str(inventory_path.resolve()),
            "inventory_sha256": _sha256(inventory_path) if inventory_path.is_file() else "",
            "files": {str(path): digest for path, digest in sorted(mapping.items(), key=lambda item: str(item[0]))},
        }

    stable_issues = _source_stability_issues(verified_files)
    checks.add("hashes", "sources_unchanged_during_evaluation", not stable_issues, detail=" | ".join(stable_issues))

    check_frame = checks.frame()
    admitted = bool(not check_frame.empty and check_frame["passed"].all())
    blocker_rows = check_frame.loc[~check_frame["passed"]]
    blockers = [f"{row.component}:{row.check_id}" for row in blocker_rows.itertuples(index=False)]
    source_checksums_verified = _checks_pass(
        check_frame,
        (
            ("hashes_active", "sha256_inventory_complete_and_verified"),
            ("hashes_reference", "sha256_inventory_complete_and_verified"),
            ("hashes_components", "sha256_inventory_complete_and_verified"),
            ("hashes", "sources_unchanged_during_evaluation"),
        ),
    )
    window_coverage_complete = _checks_pass(
        check_frame,
        (
            ("g1", "full_master_denominator_covered_by_role_maps"),
            ("ohlcv_active", "g1_ric_window_plan_coverage"),
            ("ohlcv_reference", "g1_ric_window_plan_coverage"),
            ("liquidity", "full_g1_ric_window_plan_coverage"),
        ),
    )
    raw_compacted_parity = _checks_pass(
        check_frame,
        (
            ("ohlcv_active", "raw_compacted_row_parity"),
            ("ohlcv_active", "independent_row_equality_passed"),
            ("ohlcv_reference", "raw_compacted_row_parity"),
            ("ohlcv_reference", "independent_row_equality_passed"),
        ),
    )
    official_actions_reconciled = _checks_pass(
        check_frame,
        (
            ("corporate_actions", "official_full_terminal_plan"),
            ("corporate_actions", "full_g1_ric_plan_coverage"),
            ("corporate_actions", "quality_denominators_complete"),
            ("adjusted_returns", "manifest_reconciliation_complete"),
            ("adjusted_returns", "corporate_action_reconciliation_denominator"),
            ("adjusted_returns", "raw_adjusted_daily_key_parity"),
        ),
    )
    pit_liquidity_complete = _checks_pass(
        check_frame,
        (
            ("liquidity", "full_terminal_liquidity_plan"),
            ("liquidity", "quality_denominators_complete"),
            ("liquidity", "pit_history_denominator_reconciles"),
            ("liquidity", "pit_timestamp_coverage_complete"),
        ),
    )

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=artifact_dir.parent))
    try:
        checks_path = temp_dir / "component_contract.csv"
        source_hashes_path = temp_dir / "source_hashes.json"
        summary_path = temp_dir / "summary.md"
        manifest_path = temp_dir / "manifest.json"
        check_frame.to_csv(checks_path, index=False)
        source_hashes_payload = {
            "algorithm": "sha256",
            "coverage_complete": admitted,
            "immutable_source_verification": True,
            "inventories": source_state,
        }
        source_hashes_path.write_text(json.dumps(source_hashes_payload, indent=2), encoding="utf-8")
        summary_path.write_text(_summary(run_id, admitted, check_frame, blockers), encoding="utf-8")

        component_counts = (
            check_frame.groupby("component")["passed"]
            .agg(checks="size", passed="sum")
            .astype(int)
            .to_dict("index")
        )
        manifest = {
            "run_id": run_id,
            "artifact_type": "japan_g2_aggregate_admission",
            "schema_version": "1.0",
            "status": "pass" if admitted else "failed_admission",
            "evidence_scope": "full" if admitted else "failed_no_admission",
            "admission_passed": admitted,
            "g2_complete": admitted,
            "component_scorecard_eligible": admitted,
            "scorecard_eligible": admitted,
            "promotion_allowed": admitted,
            "partial": not admitted,
            "immutable": True,
            "atomic_publish": True,
            "source_checksums_verified": source_checksums_verified,
            "window_coverage_complete": window_coverage_complete,
            "raw_compacted_parity": raw_compacted_parity,
            "official_corporate_actions_reconciled": official_actions_reconciled,
            "pit_liquidity_complete": pit_liquidity_complete,
            "critical_blocker_count": len(blockers),
            "critical_blockers": blockers,
            "check_count": int(len(check_frame)),
            "passed_check_count": int(check_frame["passed"].sum()),
            "component_counts": component_counts,
            "denominators": {
                "g1_listing_windows": int(len(denominator)),
                "g1_rics": int(denominator["ric"].nunique()) if not denominator.empty else 0,
                "active_raw_ohlcv_rows": int(len(active_raw)),
                "reference_raw_ohlcv_rows": int(len(reference_raw)),
                "raw_union_rows": int(len(raw_union)),
                "corporate_action_rows": int(len(corporate_events)),
            },
            "promotion_policy": {
                "g2_scorecard_credit": admitted,
                "alpha": False,
                "historical_backtest": False,
                "production": False,
                "ibkr": False,
                "live_trading": False,
            },
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_sha256": {
                "component_contract": _sha256(checks_path),
                "source_hashes": _sha256(source_hashes_path),
                "summary": _sha256(summary_path),
            },
            "artifacts": {
                "component_contract": "component_contract.csv",
                "source_hashes": "source_hashes.json",
                "summary": "summary.md",
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        os.replace(temp_dir, artifact_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return JapanG2AdmissionResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        admitted=admitted,
        manifest_path=artifact_dir / "manifest.json",
        checks_path=artifact_dir / "component_contract.csv",
        source_hashes_path=artifact_dir / "source_hashes.json",
        summary_path=artifact_dir / "summary.md",
        manifest=manifest,
        checks=check_frame,
    )


def build_japan_g2_admission(
    inputs: JapanG2AdmissionInputs,
    artifact_root: str | Path = "artifacts/japan_g2_admission",
    run_id: str | None = None,
) -> JapanG2AdmissionResult:
    """Compatibility name for callers that describe sidecar publication as a build."""

    return write_japan_g2_admission(inputs=inputs, artifact_root=artifact_root, run_id=run_id)


JapanG2OHLCVPaths = JapanG2OHLCVRun
JapanG2AdmissionPaths = JapanG2AdmissionInputs


def _evaluate_g1(
    inputs: JapanG2AdmissionInputs, checks: _Checks, required: set[Path]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    security_manifest_path = inputs.g1_security_master_dir / "manifest.json"
    validation_manifest_path = inputs.g1_validation_dir / "manifest.json"
    active_manifest_path = inputs.active_symbol_map_dir / "manifest.json"
    reference_manifest_path = inputs.reference_symbol_map_dir / "manifest.json"
    active_map_path = _first_existing(inputs.active_symbol_map_dir, ("symbol_map.csv", "symbol_map.parquet"))
    reference_map_path = _first_existing(inputs.reference_symbol_map_dir, ("symbol_map.csv", "symbol_map.parquet"))
    listing_path = _first_existing(inputs.g1_security_master_dir, ("listing_windows.parquet", "listing_windows.csv"))
    required.update(
        path
        for path in (
            security_manifest_path,
            validation_manifest_path,
            active_manifest_path,
            reference_manifest_path,
            active_map_path,
            reference_map_path,
            listing_path,
        )
        if path is not None
    )

    security = _read_json(security_manifest_path)
    validation = _read_json(validation_manifest_path)
    active_manifest = _read_json(active_manifest_path)
    reference_manifest = _read_json(reference_manifest_path)
    checks.add("g1", "security_master_full_and_admitted", _full_manifest(security))
    checks.add(
        "g1",
        "independent_validation_passed",
        _full_manifest(validation)
        and _text(_value(validation, "validation_status", "status")).lower() in _PASS_STATUSES
        and _integer(_value(validation, "critical_blocker_count"), -1) == 0
        and _truthy(_value(validation, "source_checksums_verified")),
    )
    checks.add("g1", "active_map_full_and_validated", _map_manifest_ready(active_manifest, "active"))
    checks.add(
        "g1", "reference_map_full_and_validated", _map_manifest_ready(reference_manifest, "reference")
    )

    denominator = _normalize_windows(_read_table(listing_path))
    active = _normalize_windows(_read_table(active_map_path))
    reference = _normalize_windows(_read_table(reference_map_path))
    checks.add("g1", "listing_window_denominator_nonempty", not denominator.empty, numerator=len(denominator))
    checks.add("g1", "active_window_denominator_nonempty", not active.empty, numerator=len(active))
    checks.add("g1", "reference_window_denominator_nonempty", not reference.empty, numerator=len(reference))
    for name, frame in (("listing", denominator), ("active", active), ("reference", reference)):
        checks.add(
            "g1",
            f"{name}_windows_valid_unique",
            _windows_valid_unique(frame),
            numerator=len(frame),
        )

    denominator_keys = _window_keys(denominator)
    active_keys = _window_keys(active)
    reference_keys = _window_keys(reference)
    checks.add("g1", "active_windows_within_master", active_keys.issubset(denominator_keys))
    checks.add("g1", "reference_windows_within_master", reference_keys.issubset(denominator_keys))
    checks.add(
        "g1",
        "full_master_denominator_covered_by_role_maps",
        bool(denominator_keys and active_keys.union(reference_keys) == denominator_keys),
        numerator=len(active_keys.union(reference_keys)),
        denominator=len(denominator_keys),
    )
    return denominator, {"active": active, "reference": reference}


def _evaluate_ohlcv_lane(
    role: str,
    paths: JapanG2OHLCVRun,
    expected_windows: pd.DataFrame,
    checks: _Checks,
    required: set[Path],
) -> tuple[pd.DataFrame, str]:
    component = f"ohlcv_{role}"
    run_manifest_path = paths.historical_dir / "historical_run_manifest.json"
    plan_path = paths.historical_dir / "historical_plan.csv"
    terminal_path = paths.historical_dir / "historical_manifest.csv"
    depth_manifest_path = paths.depth_dir / "data_depth_manifest.json"
    depth_path = paths.depth_dir / "overall.csv"
    compaction_path = paths.compaction_dir / "manifest.json"
    row_equality_path = paths.row_equality_dir / "manifest.json" if paths.row_equality_dir else None
    required.update(
        {
            run_manifest_path,
            plan_path,
            terminal_path,
            depth_manifest_path,
            depth_path,
            compaction_path,
        }
    )
    if row_equality_path is not None:
        required.add(row_equality_path)
    run = _read_json(run_manifest_path)
    plan = _read_table(plan_path)
    terminal = _read_table(terminal_path)
    depth = _read_table(depth_path)
    compaction = _read_json(compaction_path)
    row_equality = _read_json(row_equality_path) if row_equality_path is not None else {}
    run_id = _text(_value(run, "run_id"))

    planned = _integer(_value(run, "planned_rows"), -1)
    executed = _integer(_value(run, "executed_rows", "executed_manifest_rows", "terminal_rows"), len(terminal))
    completed = _integer(_value(run, "completed_rows", "terminal_rows"), -1)
    failed = _integer(_value(run, "failed_rows"), -1)
    empty = _integer(_value(run, "empty_rows"), 0)
    checks.add(component, "run_identity_present", bool(run_id))
    checks.add(
        component,
        "run_denominator_complete",
        planned > 0 and planned == len(plan) == len(terminal) == executed == completed,
        numerator=completed,
        denominator=planned,
    )
    checks.add(component, "zero_failed_or_empty_plan_rows", failed == 0 and empty == 0, numerator=failed + empty)

    plan_ids = _id_series(plan, "plan_row_id")
    terminal_ids = _id_series(terminal, "plan_row_id")
    checks.add(
        component,
        "plan_terminal_one_to_one",
        bool(
            len(plan_ids) == len(plan)
            and len(terminal_ids) == len(terminal)
            and not plan_ids.duplicated().any()
            and not terminal_ids.duplicated().any()
            and set(plan_ids) == set(terminal_ids)
        ),
    )
    statuses = terminal.get("status", pd.Series(dtype=object)).astype(str)
    payload_rows = pd.to_numeric(terminal.get("long_ohlcv_rows", pd.Series(dtype=float)), errors="coerce")
    request_failures = pd.to_numeric(
        terminal.get("failures", pd.Series(0, index=terminal.index)), errors="coerce"
    ).fillna(-1)
    terminal_payload_ok, payload_detail = _terminal_payload_contract(
        terminal,
        expected_windows,
        paths.historical_dir,
        required,
    )
    checks.add(
        component,
        "all_terminal_rows_have_complete_payload",
        terminal_payload_ok,
        detail=payload_detail,
    )
    plan_coverage, plan_outside = _plan_covers_windows(plan, expected_windows)
    checks.add(component, "g1_ric_window_plan_coverage", plan_coverage, denominator=len(expected_windows))
    checks.add(component, "zero_plan_rics_outside_g1_role", plan_outside == 0, numerator=plan_outside)

    raw_root = _path_from_manifest(paths.historical_dir, run, ("long_ohlcv_root",), "long_ohlcv")
    raw_files = _parquet_files(raw_root)
    required.update(raw_files)
    raw = _normalize_market_rows(_read_parquet_files(raw_files))
    checks.add(component, "raw_ohlcv_nonempty", not raw.empty, numerator=len(raw))
    raw_duplicates = int(raw.duplicated(["ric", "date"]).sum()) if not raw.empty else -1
    raw_outside = _outside_window_rows(raw, expected_windows)
    checks.add(component, "zero_duplicate_raw_rows", raw_duplicates == 0, numerator=raw_duplicates)
    checks.add(component, "zero_raw_rows_outside_g1_windows", raw_outside == 0, numerator=raw_outside)
    expected_rics = set(expected_windows.get("ric", pd.Series(dtype=str)).astype(str))
    raw_rics = set(raw.get("ric", pd.Series(dtype=str)).astype(str))
    checks.add(
        component,
        "all_role_rics_have_raw_rows",
        bool(expected_rics and raw_rics == expected_rics),
        numerator=len(raw_rics),
        denominator=len(expected_rics),
    )
    checks.add(
        component,
        "terminal_raw_row_denominator_reconciles",
        bool(not payload_rows.empty and int(payload_rows.sum()) == len(raw)),
        numerator=len(raw),
        denominator=int(payload_rows.sum()) if payload_rows.notna().all() else None,
    )
    close = pd.to_numeric(raw.get("close", pd.Series(dtype=float)), errors="coerce")
    volume = pd.to_numeric(raw.get("volume", pd.Series(dtype=float)), errors="coerce")
    checks.add(
        component,
        "raw_price_volume_quality",
        bool(not raw.empty and close.notna().all() and close.gt(0).all() and volume.notna().all()),
    )

    depth_ok = False
    if len(depth) == 1:
        row = depth.iloc[0]
        declared_duplicate = _integer(_series_value(row, "duplicate_ohlcv_rows", "duplicate_rows"), 0)
        declared_outside = _integer(
            _series_value(row, "outside_listing_window_rows", "outside_window_rows"), 0
        )
        depth_ok = bool(
            _integer(_series_value(row, "rows"), -1) == len(raw)
            and _integer(_series_value(row, "tickers", "rics"), -1) == len(raw_rics)
            and _integer(_series_value(row, "non_positive_close_rows"), -1) == 0
            and _number(_series_value(row, "volume_missing_pct"), -1.0) == 0.0
            and declared_duplicate == 0
            and declared_outside == 0
            and raw_duplicates == 0
            and raw_outside == 0
        )
    checks.add(component, "depth_qa_reconciles_to_raw", depth_ok)
    checks.add(component, "depth_manifest_present", bool(_read_json(depth_manifest_path)))

    source_rows = _integer(_value(compaction, "source_rows", "raw_rows"), -1)
    output_rows = _integer(_value(compaction, "output_rows", "compacted_rows"), -1)
    failed_groups = _integer(_value(compaction, "failed_groups", "failed_rows"), -1)
    compact_root = _path_from_manifest(paths.compaction_dir, compaction, ("output_root",), "compacted")
    compact_files = _parquet_files(compact_root)
    required.update(compact_files)
    compacted = _normalize_market_rows(_read_parquet_files(compact_files))
    compact_keys = _market_keys(compacted)
    raw_keys = _market_keys(raw)
    checks.add(
        component,
        "raw_compacted_row_parity",
        bool(
            _text(_value(compaction, "status")).lower() == "complete"
            and len(raw) > 0
            and source_rows == len(raw) == output_rows == len(compacted)
            and failed_groups == 0
            and raw_keys == compact_keys
            and len(compacted) == len(compact_keys)
        ),
        numerator=output_rows,
        denominator=source_rows,
    )
    equality_source = Path(_text(_value(row_equality, "source_root"))).resolve() if _value(row_equality, "source_root") else None
    equality_compacted = Path(_text(_value(row_equality, "compacted_root"))).resolve() if _value(row_equality, "compacted_root") else None
    checks.add(
        component,
        "independent_row_equality_passed",
        bool(
            _text(_value(row_equality, "status")).lower() in _PASS_STATUSES
            and _truthy(_value(row_equality, "row_equality_passed"))
            and _integer(_value(row_equality, "source_rows"), -1) == len(raw)
            and _integer(_value(row_equality, "compacted_rows"), -1) == len(compacted)
            and not _value(row_equality, "failed_years")
            and equality_source == raw_root.resolve()
            and equality_compacted == compact_root.resolve()
        ),
    )
    return raw, run_id


def _evaluate_corporate_actions(
    root: Path, windows: pd.DataFrame, checks: _Checks, required: set[Path]
) -> pd.DataFrame:
    component = "corporate_actions"
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    plan_path = _path_from_manifest(root, manifest, ("event_plan",), "event_plan.csv")
    terminal_path = _path_from_manifest(root, manifest, ("event_manifest",), "event_manifest.csv")
    quality_path = _path_from_manifest(root, manifest, ("event_quality",), "event_quality.csv")
    event_files = _preferred_data_files(root / "events")
    required.update({manifest_path, plan_path, terminal_path, quality_path, *event_files})
    plan = _read_table(plan_path)
    terminal = _read_table(terminal_path)
    quality = _read_table(quality_path)
    events = _read_data_files(event_files)

    planned = _integer(_value(manifest, "planned_rows"), -1)
    completed = _integer(_value(manifest, "completed_rows"), -1)
    failed = _integer(_value(manifest, "failed_rows"), -1)
    empty = _integer(_value(manifest, "empty_rows"), -1)
    event_rows = _integer(_value(manifest, "event_rows", "reconciled_rows"), -1)
    official = _truthy(_value(manifest, "official_source", "official_jpx_source", "official_exchange_source"))
    checks.add(
        component,
        "official_full_terminal_plan",
        bool(
            official
            and _text(_value(manifest, "status")).lower() in _PASS_STATUSES
            and planned > 0
            and planned == len(plan) == len(terminal) == completed
            and failed == 0
            and 0 <= empty <= completed
        ),
        numerator=completed,
        denominator=planned,
    )
    plan_ids = _id_series(plan, "plan_row_id")
    terminal_ids = _id_series(terminal, "plan_row_id")
    plan_rics = {
        ric
        for value in plan.get("ric_list", pd.Series(dtype=str))
        for ric in _split_rics(value)
    }
    expected_rics = set(windows.get("ric", pd.Series(dtype=str)).astype(str))
    checks.add(
        component,
        "full_g1_ric_plan_coverage",
        bool(expected_rics and plan_rics == expected_rics),
        numerator=len(plan_rics),
        denominator=len(expected_rics),
    )
    checks.add(
        component,
        "plan_terminal_denominator_reconciles",
        bool(
            len(plan_ids) == len(plan)
            and len(terminal_ids) == len(terminal)
            and not plan_ids.duplicated().any()
            and not terminal_ids.duplicated().any()
            and set(plan_ids) == set(terminal_ids)
        ),
    )
    numeric = _numeric_columns(
        terminal, ("event_rows", "known_at_rows", "valid_event_date_rows", "alpha_usable_rows")
    )
    terminal_denominators = bool(
        not terminal.empty
        and terminal.get("status", pd.Series(dtype=object)).astype(str).isin(
            {"completed", "completed_empty"}
        ).all()
        and "response_success" in terminal
        and terminal["response_success"].map(_truthy).all()
        and numeric.notna().all().all()
        and numeric["event_rows"].ge(0).all()
        and numeric["known_at_rows"].eq(numeric["event_rows"]).all()
        and numeric["valid_event_date_rows"].eq(numeric["event_rows"]).all()
        and numeric["alpha_usable_rows"].eq(numeric["event_rows"]).all()
    )
    checks.add(component, "all_terminal_event_denominators", terminal_denominators)
    quality_ok = False
    if not quality.empty:
        q = _numeric_columns(
            quality,
            (
                "plan_rows",
                "completed_rows",
                "failed_rows",
                "empty_rows",
                "event_rows",
                "known_at_rows",
                "valid_event_date_rows",
                "alpha_usable_rows",
                "min_coverage",
            ),
        )
        quality_ok = bool(
            q.notna().all().all()
            and q["plan_rows"].gt(0).all()
            and q["completed_rows"].eq(q["plan_rows"]).all()
            and q["failed_rows"].eq(0).all()
            and q["empty_rows"].ge(0).all()
            and q["empty_rows"].le(q["completed_rows"]).all()
            and q["event_rows"].ge(0).all()
            and q["known_at_rows"].eq(q["event_rows"]).all()
            and q["valid_event_date_rows"].eq(q["event_rows"]).all()
            and q["alpha_usable_rows"].eq(q["event_rows"]).all()
            and q["min_coverage"].eq(1.0).all()
            and int(q["plan_rows"].sum()) == planned
            and int(q["event_rows"].sum()) == event_rows
        )
    checks.add(component, "quality_denominators_complete", quality_ok)

    normalized = _normalize_event_rows(events)
    event_keys = normalized.get("event_key", pd.Series(dtype=str))
    duplicate_rows = int(event_keys.duplicated().sum()) if len(event_keys) else -1
    outside = _outside_window_rows(normalized.rename(columns={"event_date": "date"}), windows)
    known = normalized.get("known_at", pd.Series(dtype="datetime64[ns]"))
    alpha = normalized.get("alpha_usable", pd.Series(False, index=normalized.index)).map(_truthy)
    checks.add(
        component,
        "event_rows_reconcile_and_are_pit_usable",
        bool(
            not normalized.empty
            and len(normalized) == event_rows
            and duplicate_rows == 0
            and normalized["ric"].notna().all()
            and normalized["event_date"].notna().all()
            and known.notna().all()
            and alpha.all()
        ),
        numerator=len(normalized),
        denominator=event_rows,
    )
    checks.add(component, "zero_duplicate_or_outside_window_events", duplicate_rows == 0 and outside == 0, numerator=max(duplicate_rows, 0) + max(outside, 0))
    return normalized


def _evaluate_adjusted_returns(
    root: Path,
    corporate_events: pd.DataFrame,
    raw_keys: set[tuple[str, pd.Timestamp]],
    checks: _Checks,
    required: set[Path],
) -> None:
    component = "adjusted_returns"
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    normalized_path = _path_from_manifest(
        root, manifest, ("corporate_action_events_normalized",), "corporate_action_events_normalized.parquet"
    )
    reconciliation_path = _path_from_manifest(
        root, manifest, ("event_price_reconciliation",), "event_price_reconciliation.parquet"
    )
    daily_path = _path_from_manifest(root, manifest, ("daily_adjusted_returns",), "daily_adjusted_returns.parquet")
    quality_path = _path_from_manifest(root, manifest, ("reconciliation_quality",), "reconciliation_quality.csv")
    required.update({manifest_path, normalized_path, reconciliation_path, daily_path, quality_path})
    events = _read_table(normalized_path)
    reconciliation = _read_table(reconciliation_path)
    daily = _normalize_market_rows(_read_table(daily_path))
    quality = _read_table(quality_path)

    event_rows = _integer(_value(manifest, "event_rows", "reconciled_rows"), -1)
    valid = _integer(_value(manifest, "valid_event_rows", "reconciled_rows"), -1)
    matched = _integer(_value(manifest, "matched_event_rows"), -1)
    if matched < 0:
        matched = valid - _integer(_value(manifest, "unmatched_event_rows"), valid)
    quarantined = _integer(_value(manifest, "quarantined_event_rows", "quarantined_rows"), -1)
    ambiguous = _integer(_value(manifest, "ambiguous_factor_rows"), 0)
    daily_rows = _integer(_value(manifest, "daily_adjusted_return_rows", "adjusted_return_rows"), -1)
    checks.add(
        component,
        "manifest_reconciliation_complete",
        bool(
            _text(_value(manifest, "status")).lower() in _PASS_STATUSES
            and _truthy(_value(manifest, "production_adjusted_return_ready", "reconciliation_complete"))
            and event_rows > 0
            and event_rows == valid == matched
            and _number(_value(manifest, "matched_event_share"), -1.0) == 1.0
            and quarantined == 0
            and ambiguous == 0
            and _number(_value(manifest, "quarantine_share"), -1.0) == 0.0
        ),
        numerator=matched,
        denominator=valid,
    )
    event_key = _column(events, "event_key")
    reconciliation_key = _column(reconciliation, "event_key")
    corporate_key = _column(corporate_events, "event_key")
    keys_ok = bool(
        len(events) == event_rows == len(reconciliation)
        and event_key.notna().all()
        and reconciliation_key.notna().all()
        and not event_key.duplicated().any()
        and not reconciliation_key.duplicated().any()
        and set(event_key.astype(str)) == set(reconciliation_key.astype(str)) == set(corporate_key.astype(str))
    )
    checks.add(component, "corporate_action_reconciliation_denominator", keys_ok)
    statuses = reconciliation.get("reconciliation_status", pd.Series(dtype=object)).astype(str)
    checks.add(
        component,
        "zero_unmatched_quarantined_or_duplicate_reconciliations",
        bool(
            len(statuses) == len(reconciliation)
            and statuses.ne("quarantined").all()
            and reconciliation_key.duplicated().sum() == 0
            and reconciliation.get("ex_trade_date", pd.Series(dtype=object)).notna().all()
        ),
    )
    quality_ok = False
    if not quality.empty:
        q = _numeric_columns(
            quality,
            (
                "event_rows",
                "valid_event_rows",
                "matched_event_rows",
                "quarantined_event_rows",
                "matched_event_share",
                "quarantine_share",
            ),
        )
        quality_ok = bool(
            q.notna().all().all()
            and q["event_rows"].eq(q["valid_event_rows"]).all()
            and q["valid_event_rows"].eq(q["matched_event_rows"]).all()
            and q["quarantined_event_rows"].eq(0).all()
            and q["matched_event_share"].eq(1.0).all()
            and q["quarantine_share"].eq(0.0).all()
            and int(q["event_rows"].sum()) == event_rows
        )
    checks.add(component, "quality_denominators_complete", quality_ok)
    daily_keys = _market_keys(daily)
    checks.add(
        component,
        "raw_adjusted_daily_key_parity",
        bool(
            raw_keys
            and len(daily) == daily_rows == len(daily_keys)
            and daily_keys == raw_keys
        ),
        numerator=len(daily_keys),
        denominator=len(raw_keys),
    )


def _evaluate_liquidity(
    root: Path, windows: pd.DataFrame, checks: _Checks, required: set[Path]
) -> None:
    component = "liquidity"
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    plan_path = _path_from_manifest(root, manifest, ("historical_plan",), "historical_plan.csv")
    terminal_path = _path_from_manifest(root, manifest, ("historical_manifest",), "historical_manifest.csv")
    quality_path = _path_from_manifest(root, manifest, ("historical_quality",), "historical_quality.csv")
    history_root = _path_from_manifest(root, manifest, ("output_root",), "history")
    history_files = _parquet_files(history_root)
    required.update({manifest_path, plan_path, terminal_path, quality_path, *history_files})
    plan = _read_table(plan_path)
    terminal = _read_table(terminal_path)
    quality = _read_table(quality_path)
    history = _normalize_liquidity_rows(_read_parquet_files(history_files))

    planned = _integer(_value(manifest, "planned_rows"), -1)
    completed = _integer(_value(manifest, "completed_rows"), -1)
    failed = _integer(_value(manifest, "failed_rows"), -1)
    empty = _integer(_value(manifest, "empty_rows"), -1)
    history_rows = _integer(_value(manifest, "history_rows"), -1)
    plan_datasets = {_canonical_liquidity_dataset(value) for value in _column(plan, "dataset_id").dropna()}
    checks.add(
        component,
        "full_terminal_liquidity_plan",
        bool(
            planned > 0
            and planned == len(plan) == len(terminal) == completed
            and failed == 0
            and empty == 0
            and plan_datasets == {"liquidity_pit"}
        ),
        numerator=completed,
        denominator=planned,
    )
    plan_ids = _id_series(plan, "plan_row_id")
    terminal_ids = _id_series(terminal, "plan_row_id")
    terminal_status = terminal.get("status", pd.Series(dtype=object)).astype(str)
    terminal_outside = pd.to_numeric(
        terminal.get("outside_requested_rows", pd.Series(0, index=terminal.index)), errors="coerce"
    ).fillna(-1)
    terminal_history = pd.to_numeric(
        terminal.get("history_rows", pd.Series(dtype=float)), errors="coerce"
    )
    checks.add(
        component,
        "plan_terminal_denominator_reconciles",
        bool(
            len(plan_ids) == len(plan)
            and len(terminal_ids) == len(terminal)
            and not plan_ids.duplicated().any()
            and not terminal_ids.duplicated().any()
            and set(plan_ids) == set(terminal_ids)
            and terminal_status.eq("completed").all()
            and terminal_history.notna().all()
            and terminal_history.gt(0).all()
            and terminal_outside.eq(0).all()
            and int(terminal_history.sum()) == history_rows
        ),
    )
    coverage, plan_outside = _plan_covers_windows(plan, windows)
    checks.add(component, "full_g1_ric_window_plan_coverage", coverage, denominator=len(windows))
    checks.add(component, "zero_plan_rics_outside_g1", plan_outside == 0, numerator=plan_outside)

    quality_ok = False
    if not quality.empty:
        coverage_col = "min_coverage" if "min_coverage" in quality else "coverage"
        average_col = "avg_coverage" if "avg_coverage" in quality else coverage_col
        q = _numeric_columns(
            quality,
            (
                "plan_rows",
                "completed_rows",
                "failed_rows",
                "empty_rows",
                "history_rows",
                coverage_col,
                average_col,
            ),
        )
        quality_datasets = {
            _canonical_liquidity_dataset(value) for value in _column(quality, "dataset_id").dropna()
        }
        quality_ok = bool(
            quality_datasets == {"liquidity_pit"}
            and q.notna().all().all()
            and q["plan_rows"].gt(0).all()
            and q["completed_rows"].eq(q["plan_rows"]).all()
            and q["failed_rows"].eq(0).all()
            and q["empty_rows"].eq(0).all()
            and q["history_rows"].gt(0).all()
            and q[coverage_col].eq(1.0).all()
            and q[average_col].eq(1.0).all()
            and int(q["plan_rows"].sum()) == planned
            and int(q["history_rows"].sum()) == history_rows
        )
    checks.add(component, "quality_denominators_complete", quality_ok)

    duplicate_rows = (
        int(history.duplicated(["ric", "asof_date", "field_role"]).sum()) if not history.empty else -1
    )
    outside = _outside_window_rows(history.rename(columns={"asof_date": "date"}), windows)
    expected_rics = set(windows.get("ric", pd.Series(dtype=str)).astype(str))
    actual_rics = set(history.get("ric", pd.Series(dtype=str)).astype(str))
    history_ids = set(_id_series(history, "plan_row_id"))
    checks.add(
        component,
        "pit_history_denominator_reconciles",
        bool(
            not history.empty
            and len(history) == history_rows
            and actual_rics == expected_rics
            and history_ids == set(plan_ids)
            and history["asof_date"].notna().all()
            and history["value"].notna().all()
            and history["field_role"].astype(str).str.strip().ne("").all()
        ),
        numerator=len(history),
        denominator=history_rows,
    )
    checks.add(
        component,
        "zero_duplicate_or_outside_window_liquidity_rows",
        duplicate_rows == 0 and outside == 0,
        numerator=max(duplicate_rows, 0) + max(outside, 0),
    )
    declared_known = _value(manifest, "known_at_coverage", "pit_timestamp_coverage")
    declared_violations = _value(manifest, "pit_timestamp_violations", "outside_window_rows")
    declared_ok = (declared_known is None or _number(declared_known, -1.0) == 1.0) and (
        declared_violations is None or _integer(declared_violations, -1) == 0
    )
    checks.add(
        component,
        "pit_timestamp_coverage_complete",
        bool(not history.empty and history["asof_date"].notna().all() and outside == 0 and declared_ok),
        numerator=int(history["asof_date"].notna().sum()) if not history.empty else 0,
        denominator=len(history),
    )


def _verify_hash_inventory(path: Path, required: set[Path]) -> tuple[dict[Path, str], list[str]]:
    payload = _read_json(path)
    issues: list[str] = []
    if not payload:
        return {}, [f"missing_or_invalid_inventory:{path}"]
    if _text(payload.get("algorithm", "sha256")).lower() != "sha256":
        issues.append("algorithm_is_not_sha256")
    if payload.get("immutable") is not True:
        issues.append("immutable_not_true")
    if payload.get("coverage_complete") is not True:
        issues.append("coverage_complete_not_true")
    raw_files = payload.get("files")
    if not isinstance(raw_files, Mapping) or not raw_files:
        return {}, [*issues, "files_mapping_missing"]
    mapping: dict[Path, str] = {}
    for raw_path, raw_digest in raw_files.items():
        digest = _text(raw_digest).lower()
        file_path = Path(_text(raw_path))
        if not file_path.is_absolute():
            file_path = path.parent / file_path
        file_path = file_path.resolve()
        if not _SHA256_RE.fullmatch(digest):
            issues.append(f"invalid_digest:{raw_path}")
        elif not file_path.is_file():
            issues.append(f"missing_file:{raw_path}")
        elif _sha256(file_path) != digest:
            issues.append(f"digest_mismatch:{raw_path}")
        mapping[file_path] = digest
    required_resolved = {item.resolve() for item in required}
    missing = required_resolved - set(mapping)
    if missing:
        issues.append(f"required_hash_coverage_missing:{len(missing)}")
    return mapping, issues


def _source_stability_issues(inventories: Mapping[str, Mapping[Path, str]]) -> list[str]:
    issues: list[str] = []
    for inventory, files in inventories.items():
        for path, expected in files.items():
            if not path.is_file() or _sha256(path) != expected:
                issues.append(f"{inventory}:{path}")
    return issues


def _checks_pass(checks: pd.DataFrame, keys: Sequence[tuple[str, str]]) -> bool:
    if checks.empty:
        return False
    indexed = checks.set_index(["component", "check_id"])["passed"]
    return all(key in indexed.index and bool(indexed.loc[key]) for key in keys)


def _terminal_payload_contract(
    terminal: pd.DataFrame,
    expected_windows: pd.DataFrame,
    historical_dir: Path,
    required: set[Path],
) -> tuple[bool, str]:
    if terminal.empty:
        return False, "terminal manifest is empty"

    window_groups = {
        str(ric): group.reset_index(drop=True)
        for ric, group in expected_windows.groupby("ric", dropna=False)
        if pd.notna(ric) and str(ric)
    }
    issues: list[str] = []
    accepted_rows = 0
    structural_missing = 0
    unexplained_missing = 0
    now = pd.Timestamp.now(tz="UTC")

    for row in terminal.to_dict("records"):
        plan_row_id = str(_mapping_value(row, "plan_row_id") or "<missing>")
        status = str(_mapping_value(row, "status") or "")
        payload_rows = _integer(_mapping_value(row, "long_ohlcv_rows"), -1)
        failures = _integer(_mapping_value(row, "failures"), -1)
        tickers = _split_rics(_mapping_value(row, "tickers", "ric_list"))

        if status == "completed":
            if payload_rows > 0 and failures == 0:
                accepted_rows += 1
            else:
                issues.append(f"{plan_row_id}: completed row has payload={payload_rows}, failures={failures}")
            continue

        if status not in {
            "completed_partial",
            "completed_structural_partial",
            "completed_prelisting",
            "completed_postretirement",
        }:
            issues.append(f"{plan_row_id}: unsupported terminal status {status or '<missing>'}")
            continue

        coverage_path = _resolve_evidence_path(
            historical_dir,
            _mapping_value(row, "canonical_coverage_path", "coverage_path"),
        )
        if coverage_path is None or not coverage_path.is_file():
            issues.append(f"{plan_row_id}: coverage file is missing")
            continue
        required.add(coverage_path)
        coverage = _read_table(coverage_path)
        coverage_tickers = _id_series(coverage, "ticker")
        coverage_rows = pd.to_numeric(
            coverage.get("rows", pd.Series(dtype=float)), errors="coerce"
        )
        if (
            not tickers
            or len(coverage) != len(tickers)
            or len(coverage_tickers) != len(coverage)
            or coverage_tickers.duplicated().any()
            or set(coverage_tickers) != set(tickers)
            or coverage_rows.isna().any()
        ):
            issues.append(f"{plan_row_id}: coverage denominator does not equal requested ticker set")
            continue

        missing_tickers = coverage.loc[coverage_rows.eq(0), "ticker"].astype(str).tolist()
        nonzero_tickers = int(coverage_rows.gt(0).sum())
        if failures != len(missing_tickers):
            issues.append(
                f"{plan_row_id}: manifest failures={failures} but coverage zero rows={len(missing_tickers)}"
            )
            continue
        if status in {"completed_partial", "completed_structural_partial"}:
            if payload_rows <= 0 or nonzero_tickers <= 0 or not missing_tickers:
                issues.append(f"{plan_row_id}: partial row lacks mixed usable/zero coverage")
                continue
        elif payload_rows != 0 or nonzero_tickers != 0 or len(missing_tickers) != len(tickers):
            issues.append(f"{plan_row_id}: structural-empty row has payload or nonzero ticker coverage")
            continue

        request_start = pd.to_datetime(_mapping_value(row, "start_date", "window_start"), errors="coerce")
        request_end = pd.to_datetime(_mapping_value(row, "end_date", "window_end"), errors="coerce")
        if pd.isna(request_start) or pd.isna(request_end):
            issues.append(f"{plan_row_id}: request window is invalid")
            continue

        unexplained = [
            ric
            for ric in missing_tickers
            if not _zero_history_is_structural(
                window_groups.get(ric),
                pd.Timestamp(request_start).normalize(),
                pd.Timestamp(request_end).normalize(),
                now,
            )
        ]
        structural_missing += len(missing_tickers) - len(unexplained)
        unexplained_missing += len(unexplained)
        if unexplained:
            issues.append(f"{plan_row_id}: unexplained zero-history RICs={','.join(unexplained)}")
            continue
        accepted_rows += 1

    detail = (
        f"accepted_terminal_rows={accepted_rows}/{len(terminal)}; "
        f"structural_missing_rics={structural_missing}; unexplained_missing_rics={unexplained_missing}"
    )
    if issues:
        detail += "; issues=" + " | ".join(issues[:25])
        if len(issues) > 25:
            detail += f" | ... {len(issues) - 25} additional"
    return accepted_rows == len(terminal) and not issues, detail


def _zero_history_is_structural(
    windows: pd.DataFrame | None,
    request_start: pd.Timestamp,
    request_end: pd.Timestamp,
    now: pd.Timestamp,
) -> bool:
    if windows is None or windows.empty or not windows["pit_usable"].map(_truthy).all():
        return False
    for item in windows.itertuples(index=False):
        if pd.isna(item.start) or pd.isna(item.end):
            return False
        effective_end = pd.Timestamp.max.normalize() if bool(item.right_open) else pd.Timestamp(item.end)
        if pd.Timestamp(item.start) <= request_end and effective_end >= request_start:
            return False
        if pd.Timestamp(item.end) < request_start and not bool(item.right_open):
            known_at = item.window_end_known_at
            if pd.isna(known_at) or pd.Timestamp(known_at) > now:
                return False
    return True


def _resolve_evidence_path(root: Path, value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute() or path.exists():
        return path
    candidate = root / path
    return candidate if candidate.exists() else path


def _normalize_windows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["ric", "start", "end", "right_open", "window_end_known_at", "pit_usable"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    ric = _column(frame, "refinitiv_ric", "ric", "ticker", "requested_ric").astype("string")
    start = pd.to_datetime(_column(frame, "listing_start", "window_start", "start_date"), errors="coerce").dt.normalize()
    end = pd.to_datetime(_column(frame, "listing_end", "window_end", "end_date"), errors="coerce").dt.normalize()
    right_open = _optional_column(frame, "right_open", default=False).map(_truthy)
    window_end_known_at = pd.to_datetime(
        _optional_column(frame, "window_end_known_at", default=pd.NA), errors="coerce", utc=True
    )
    pit_usable = _optional_column(frame, "pit_usable", default=True).map(_truthy)
    result = pd.DataFrame(
        {
            "ric": ric,
            "start": start,
            "end": end,
            "right_open": right_open,
            "window_end_known_at": window_end_known_at,
            "pit_usable": pit_usable,
        }
    )
    return result.reset_index(drop=True)


def _windows_valid_unique(frame: pd.DataFrame) -> bool:
    return bool(
        not frame.empty
        and frame[["ric", "start", "end"]].notna().all().all()
        and frame["ric"].astype(str).str.strip().ne("").all()
        and frame["start"].le(frame["end"]).all()
        and not frame.duplicated(["ric", "start", "end"]).any()
    )


def _window_keys(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp, pd.Timestamp]]:
    if frame.empty:
        return set()
    return {
        (str(row.ric), pd.Timestamp(row.start), pd.Timestamp(row.end))
        for row in frame.itertuples(index=False)
        if pd.notna(row.ric) and pd.notna(row.start) and pd.notna(row.end)
    }


def _plan_covers_windows(plan: pd.DataFrame, windows: pd.DataFrame) -> tuple[bool, int]:
    if plan.empty or windows.empty:
        return False, 0
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for row in plan.to_dict("records"):
        start = pd.to_datetime(_mapping_value(row, "window_start", "start_date", "listing_start"), errors="coerce")
        end = pd.to_datetime(_mapping_value(row, "window_end", "end_date", "listing_end"), errors="coerce")
        if pd.isna(start) or pd.isna(end) or start > end:
            continue
        for ric in _split_rics(_mapping_value(row, "tickers", "ric_list", "refinitiv_ric", "requested_ric")):
            intervals.setdefault(ric, []).append((pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()))
    expected_rics = set(windows["ric"].astype(str))
    outside = len(set(intervals) - expected_rics)
    for item in windows.itertuples(index=False):
        candidates = sorted(intervals.get(str(item.ric), []))
        cursor = pd.Timestamp(item.start)
        target = pd.Timestamp(item.end)
        for start, end in candidates:
            clipped_start = max(start, cursor)
            clipped_end = min(end, target)
            if clipped_end < cursor or clipped_start > cursor:
                continue
            cursor = max(cursor, clipped_end + pd.Timedelta(days=1))
            if cursor > target:
                break
        if cursor <= target:
            return False, outside
    return True, outside


def _normalize_market_rows(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["ric", "date", "close", "volume"]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    result = frame.copy()
    result["ric"] = _column(result, "ticker", "refinitiv_ric", "ric", "requested_ric").astype("string")
    result["date"] = pd.to_datetime(_column(result, "date", "asof_date"), errors="coerce").dt.normalize()
    if "close" not in result:
        result["close"] = pd.NA
    if "volume" not in result:
        result["volume"] = pd.NA
    return result


def _normalize_event_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["event_key", "ric", "event_date", "known_at", "alpha_usable"])
    result = frame.copy()
    result["event_key"] = _column(result, "event_key", "event_id")
    result["ric"] = _column(result, "ticker", "requested_ric", "refinitiv_ric", "ric").astype("string")
    result["event_date"] = pd.to_datetime(
        _column(result, "event_date", "price_event_date", "ex_trade_date"), errors="coerce"
    ).dt.normalize()
    result["known_at"] = pd.to_datetime(
        _column(result, "known_at_date", "known_at", "source_loaded_at_utc"), errors="coerce"
    ).dt.normalize()
    if "alpha_usable" not in result:
        result["alpha_usable"] = False
    return result


def _normalize_liquidity_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["ric", "asof_date", "field_role", "value", "plan_row_id"])
    result = frame.copy()
    result["ric"] = _column(result, "requested_ric", "ticker", "refinitiv_ric", "ric").astype("string")
    result["asof_date"] = pd.to_datetime(
        _column(result, "asof_date", "date", "known_at_date"), errors="coerce"
    ).dt.normalize()
    if "field_role" not in result:
        result["field_role"] = pd.NA
    if "value" not in result:
        result["value"] = pd.NA
    if "plan_row_id" not in result:
        result["plan_row_id"] = pd.NA
    return result


def _outside_window_rows(rows: pd.DataFrame, windows: pd.DataFrame) -> int:
    if rows.empty:
        return -1
    if windows.empty or not {"ric", "date"}.issubset(rows):
        return len(rows)
    by_ric: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for item in windows.itertuples(index=False):
        end = pd.Timestamp.max.normalize() if bool(getattr(item, "right_open", False)) else pd.Timestamp(item.end)
        by_ric.setdefault(str(item.ric), []).append((pd.Timestamp(item.start), end))
    outside = 0
    for ric, date in rows[["ric", "date"]].itertuples(index=False, name=None):
        if pd.isna(ric) or pd.isna(date):
            outside += 1
            continue
        if not any(start <= pd.Timestamp(date) <= end for start, end in by_ric.get(str(ric), [])):
            outside += 1
    return outside


def _union_raw_keys(*frames: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    keys: set[tuple[str, pd.Timestamp]] = set()
    for frame in frames:
        keys.update(_market_keys(frame))
    return keys


def _market_keys(frame: pd.DataFrame) -> set[tuple[str, pd.Timestamp]]:
    if frame.empty or not {"ric", "date"}.issubset(frame):
        return set()
    return {
        (str(ric), pd.Timestamp(date))
        for ric, date in frame[["ric", "date"]].itertuples(index=False, name=None)
        if pd.notna(ric) and pd.notna(date)
    }


def _map_manifest_ready(manifest: Mapping[str, Any], role: str) -> bool:
    return bool(
        _full_manifest(manifest)
        and _text(_value(manifest, "map_role", "role")).lower() == role
        and _text(_value(manifest, "validation_status", "status")).lower() in _PASS_STATUSES
        and _integer(_value(manifest, "unresolved_rows"), 0) == 0
        and _integer(_value(manifest, "overlap_conflict_rows", "duplicate_window_rows"), 0) == 0
    )


def _full_manifest(manifest: Mapping[str, Any]) -> bool:
    if not manifest:
        return False
    labels = " ".join(
        _text(_value(manifest, name)) for name in ("run_id", "artifact_type", "mode", "scope")
    ).lower()
    return bool(
        _text(_value(manifest, "status")).lower() in _PASS_STATUSES
        and _truthy(_value(manifest, "scorecard_eligible", "component_scorecard_eligible"))
        and _text(_value(manifest, "evidence_scope")).lower() == "full"
        and not any(_truthy(_value(manifest, name)) for name in ("partial", "smoke", "is_smoke_test"))
        and not any(marker in labels for marker in ("partial", "smoke", "stub", "policy-only"))
    )


def _path_from_manifest(root: Path, manifest: Mapping[str, Any], keys: Sequence[str], default: str) -> Path:
    value: Any = None
    artifacts = manifest.get("artifacts")
    for key in keys:
        if isinstance(artifacts, Mapping) and artifacts.get(key):
            value = artifacts[key]
            break
        if manifest.get(key):
            value = manifest[key]
            break
    path = Path(_text(value)) if value else root / default
    if path.is_absolute():
        return path
    rooted = root / path
    return rooted if rooted.exists() or not path.exists() else path


def _first_existing(root: Path, names: Sequence[str]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return root / names[0] if names else None


def _preferred_data_files(root: Path) -> list[Path]:
    parquet = _parquet_files(root)
    return parquet or sorted(path for path in root.rglob("*.csv") if path.is_file()) if root.exists() else []


def _parquet_files(root: Path) -> list[Path]:
    if root.is_file() and root.suffix.lower() in {".parquet", ".pq"}:
        return [root]
    return sorted(path for path in root.rglob("*.parquet") if path.is_file()) if root.is_dir() else []


def _read_parquet_files(paths: Sequence[Path]) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except (OSError, ValueError, ImportError):
            return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _read_data_files(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [_read_table(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}


def _numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            name: pd.to_numeric(frame[name], errors="coerce")
            if name in frame
            else pd.Series(float("nan"), index=frame.index)
            for name in columns
        },
        index=frame.index,
    )


def _id_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if frame.empty or name not in frame:
        return pd.Series(dtype=str)
    values = frame[name]
    return values.loc[values.notna() & values.astype(str).str.strip().ne("")].astype(str)


def _column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame:
            return frame[name]
    return pd.Series(pd.NA, index=frame.index, dtype="object")


def _optional_column(frame: pd.DataFrame, name: str, default: Any) -> pd.Series:
    return frame[name] if name in frame else pd.Series(default, index=frame.index)


def _value(mapping: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    return _value(mapping, *names)


def _series_value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name in row.index:
            return row[name]
    return None


def _split_rics(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        values: Iterable[Any] = value
    else:
        values = _text(value).replace("|", ",").split(",")
    return [str(item).strip() for item in values if str(item).strip()]


def _canonical_liquidity_dataset(value: Any) -> str:
    text = _text(value).strip().lower()
    return "liquidity_pit" if text in _LIQUIDITY_DATASET_ALIASES else text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _text(value).strip().lower() in {"1", "true", "yes", "pass", "passed", "complete"}


def _integer(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _summary(run_id: str, admitted: bool, checks: pd.DataFrame, blockers: Sequence[str]) -> str:
    lines = [
        "# Japan G2 Aggregate Admission",
        "",
        f"Run ID: `{run_id}`",
        f"Status: `{'pass' if admitted else 'failed_admission'}`",
        f"Scorecard eligible: `{admitted}`",
        f"Checks passed: `{int(checks['passed'].sum())}/{len(checks)}`",
        "",
        "This sidecar grants only G2 component credit. Alpha, production, IBKR, and live-trading promotion remain false.",
        "",
        "## Blockers",
        "",
        *([f"- `{blocker}`" for blocker in blockers] or ["- None."]),
        "",
    ]
    return "\n".join(lines)
