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


EXPECTED_G1_RICS = 5_752
EXPECTED_G1_LISTING_WINDOWS = 9_706
G3_DATASETS = (
    "fundamentals_pit",
    "ibes_estimates_revisions",
    "earnings_events_actuals",
)
G3_LANES = ("fundamentals", "consensus", "earnings", "analyst_detail")

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PASS_STATUSES = {"accepted", "accepted_backtest", "complete", "pass", "passed"}
_TERMINAL_STATUSES = {"completed", "structural_not_applicable"}
_INVALID_SCOPE_MARKERS = ("bounded", "partial", "qa", "smoke", "stub", "synthetic")
_LANE_PLAN_KEYS = {
    "fundamentals": "fundamentals_plan",
    "consensus": "consensus_plan",
    "earnings": "earnings_plan",
    "analyst_detail": "analyst_detail_plan",
}
_LANE_DATASETS = {
    "fundamentals": "fundamentals_pit",
    "consensus": "ibes_estimates_revisions",
    "earnings": "earnings_events_actuals",
    "analyst_detail": "ibes_estimates_revisions",
}
_ORDINARY_LANES = {"fundamentals", "consensus", "earnings"}
_STRUCTURAL_REASONS = {
    "outside_listing_window",
    "pre_listing_window",
    "post_delisting_window",
}


@dataclass(frozen=True)
class JapanG3TerminalRun:
    lane: str
    producer_dir: Path
    validator_dir: Path


@dataclass(frozen=True)
class JapanG3AdmissionInputs:
    g1_validation_dir: Path
    g2_admission_dir: Path
    pit_plan_dir: Path
    execution_plan_dir: Path
    sector_routing_path: Path
    decision_calendar_path: Path
    terminal_runs: tuple[JapanG3TerminalRun, ...]


@dataclass(frozen=True)
class JapanG3AdmissionResult:
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


class _Sources:
    """Capture each source before use and detect any later mutation."""

    def __init__(self) -> None:
        self.hashes: dict[Path, str] = {}
        self.missing: set[Path] = set()

    def add(self, path: str | Path | None) -> Path | None:
        if path is None:
            return None
        resolved = Path(path).expanduser().resolve()
        if resolved in self.hashes or resolved in self.missing:
            return resolved
        if not resolved.is_file():
            self.missing.add(resolved)
            return resolved
        self.hashes[resolved] = _sha256(resolved)
        return resolved

    def stability_issues(self) -> list[str]:
        issues = [f"missing source: {path}" for path in sorted(self.missing, key=str)]
        for path, expected in sorted(self.hashes.items(), key=lambda item: str(item[0])):
            if not path.is_file():
                issues.append(f"source disappeared: {path}")
                continue
            actual = _sha256(path)
            if actual != expected:
                issues.append(f"source changed during validation: {path}")
        return issues


def write_japan_g3_admission(
    inputs: JapanG3AdmissionInputs,
    artifact_root: str | Path = "artifacts/japan_g3_admission",
    run_id: str | None = None,
) -> JapanG3AdmissionResult:
    """Publish an immutable, atomic, fail-closed aggregate admission for Japan G3."""

    run_id = run_id or datetime.now(timezone.utc).strftime("JPN-G3-ADMISSION-%Y%m%d-%H%M%S")
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("run_id must be one non-empty path component")
    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists():
        raise FileExistsError(f"immutable G3 admission artifact already exists: {artifact_dir}")

    checks = _Checks()
    sources = _Sources()
    lanes = {run.lane: run for run in inputs.terminal_runs}
    checks.add(
        "lanes",
        "exactly_one_terminal_run_per_required_lane",
        len(inputs.terminal_runs) == len(G3_LANES)
        and len(lanes) == len(G3_LANES)
        and set(lanes) == set(G3_LANES),
        numerator=len(lanes),
        denominator=len(G3_LANES),
        detail=f"lanes={sorted(lanes)}",
    )

    g1_manifest_path = sources.add(inputs.g1_validation_dir / "manifest.json")
    g2_manifest_path = sources.add(inputs.g2_admission_dir / "manifest.json")
    pit_manifest_path = sources.add(inputs.pit_plan_dir / "manifest.json")
    execution_manifest_path = sources.add(inputs.execution_plan_dir / "manifest.json")
    g1_manifest = _read_json(g1_manifest_path)
    g2_manifest = _read_json(g2_manifest_path)
    pit_manifest = _read_json(pit_manifest_path)
    execution_manifest = _read_json(execution_manifest_path)

    _evaluate_g1(g1_manifest, checks)
    _evaluate_g2(g2_manifest, inputs.g2_admission_dir, checks, sources)
    selected, windows = _evaluate_pit_plan(
        pit_manifest, inputs.pit_plan_dir, g1_manifest_path, checks, sources
    )
    plans = _evaluate_execution_plan(
        execution_manifest,
        inputs.execution_plan_dir,
        pit_manifest_path,
        inputs.pit_plan_dir,
        selected,
        checks,
        sources,
    )
    sector = _evaluate_sector_routing(
        inputs.sector_routing_path, selected, plans.get("fundamentals", pd.DataFrame()), checks, sources
    )
    decision_calendar = _evaluate_decision_calendar(
        inputs.decision_calendar_path, selected, checks, sources
    )

    lane_stats: dict[str, dict[str, int]] = {}
    for lane in G3_LANES:
        run = lanes.get(lane)
        lane_stats[lane] = _evaluate_lane(
            lane,
            run,
            plans.get(lane, pd.DataFrame()),
            windows,
            sector,
            decision_calendar,
            checks,
            sources,
        )

    _evaluate_cross_lane_separation(plans, lanes, checks)
    stable_issues = sources.stability_issues()
    checks.add(
        "hashes",
        "all_sources_present_and_unchanged_during_validation",
        not stable_issues,
        numerator=len(sources.hashes),
        denominator=len(sources.hashes) + len(sources.missing),
        detail=" | ".join(stable_issues),
    )

    check_frame = checks.frame()
    admitted = bool(not check_frame.empty and check_frame["passed"].all())
    blockers = [
        f"{row.component}:{row.check_id}"
        for row in check_frame.loc[~check_frame["passed"]].itertuples(index=False)
    ]
    component_counts = (
        check_frame.groupby("component")["passed"]
        .agg(checks="size", passed="sum")
        .astype(int)
        .to_dict("index")
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
            "source_count": len(sources.hashes),
            "missing_source_count": len(sources.missing),
            "files": {str(path): digest for path, digest in sorted(sources.hashes.items(), key=lambda x: str(x[0]))},
            "missing_files": [str(path) for path in sorted(sources.missing, key=str)],
        }
        source_hashes_path.write_text(
            json.dumps(source_hashes_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        summary_path.write_text(_summary(run_id, admitted, check_frame, blockers), encoding="utf-8")

        manifest = {
            "run_id": run_id,
            "artifact_type": "japan_g3_aggregate_admission",
            "schema_version": "1.0",
            "status": "pass" if admitted else "failed_admission",
            "evidence_scope": "full" if admitted else "failed_no_admission",
            "admission_passed": admitted,
            "g3_complete": admitted,
            "component_scorecard_eligible": admitted,
            "scorecard_eligible": admitted,
            "promotion_allowed": admitted,
            "partial": not admitted,
            "immutable": True,
            "atomic_publish": True,
            "source_checksums_verified": admitted,
            "producer_validator_independence_verified": _check_passed(
                check_frame, "validators", "all_lane_validators_are_independent"
            ),
            "decision_calendar_eligibility_recomputed": _check_passed(
                check_frame, "pit", "all_rows_known_at_or_before_eligible_decision"
            ),
            "critical_blocker_count": len(blockers),
            "critical_blockers": blockers,
            "check_count": int(len(check_frame)),
            "passed_check_count": int(check_frame["passed"].sum()),
            "component_counts": component_counts,
            "denominators": {
                "g1_rics": int(selected["ric"].nunique()) if not selected.empty else 0,
                "g1_listing_windows": int(len(windows)),
                "execution_plan_rows": int(sum(len(plan) for plan in plans.values())),
                "terminal_rows": int(sum(item["terminal_rows"] for item in lane_stats.values())),
                "history_rows": int(sum(item["history_rows"] for item in lane_stats.values())),
            },
            "datasets": list(G3_DATASETS),
            "lanes": list(G3_LANES),
            "promotion_policy": {
                "g3_scorecard_credit": admitted,
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
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        final_stability_issues = sources.stability_issues()
        if final_stability_issues != stable_issues:
            raise RuntimeError("G3 admission sources changed while the packet was being published")
        os.replace(temp_dir, artifact_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    return JapanG3AdmissionResult(
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


def build_japan_g3_admission(
    inputs: JapanG3AdmissionInputs,
    artifact_root: str | Path = "artifacts/japan_g3_admission",
    run_id: str | None = None,
) -> JapanG3AdmissionResult:
    return write_japan_g3_admission(inputs=inputs, artifact_root=artifact_root, run_id=run_id)


def _evaluate_g1(manifest: Mapping[str, Any], checks: _Checks) -> None:
    independent = manifest.get("independent_validation") is True or bool(
        manifest.get("independent_audit") is True
        and manifest.get("validator_type") == "independent_process"
    )
    checks.add(
        "g1",
        "independent_full_admission",
        _full_manifest(manifest)
        and independent
        and int(manifest.get("critical_blocker_count", -1) or 0) == 0
        and manifest.get("source_checksums_verified") is True,
    )
    producer = str(manifest.get("producer_run_id", "")).strip()
    validator = str(manifest.get("validator_run_id", manifest.get("run_id", ""))).strip()
    checks.add(
        "g1",
        "producer_validator_identity_is_separate",
        bool(producer and validator and producer != validator),
        detail=f"producer={producer or '<missing>'}; validator={validator or '<missing>'}",
    )


def _evaluate_g2(
    manifest: Mapping[str, Any],
    directory: Path,
    checks: _Checks,
    sources: _Sources,
) -> None:
    denominators = manifest.get("denominators", {})
    strict = bool(
        manifest.get("artifact_type") == "japan_g2_aggregate_admission"
        and str(manifest.get("status", "")).lower() == "pass"
        and manifest.get("admission_passed") is True
        and manifest.get("g2_complete") is True
        and manifest.get("scorecard_eligible") is True
        and manifest.get("evidence_scope") == "full"
        and manifest.get("partial") is False
        and manifest.get("immutable") is True
        and manifest.get("atomic_publish") is True
        and manifest.get("source_checksums_verified") is True
        and manifest.get("window_coverage_complete") is True
        and manifest.get("raw_compacted_parity") is True
        and manifest.get("official_corporate_actions_reconciled") is True
        and manifest.get("pit_liquidity_complete") is True
        and int(manifest.get("critical_blocker_count", -1)) == 0
        and int(manifest.get("check_count", 0)) > 0
        and int(manifest.get("passed_check_count", -1)) == int(manifest.get("check_count", 0))
    )
    checks.add("g2", "strict_aggregate_admission", strict)
    checks.add(
        "g2",
        "admitted_g1_denominator_is_exact",
        _integer(denominators.get("g1_rics"), -1) == EXPECTED_G1_RICS
        and _integer(denominators.get("g1_listing_windows"), -1) == EXPECTED_G1_LISTING_WINDOWS,
        numerator=f"{_integer(denominators.get('g1_rics'), -1)}/{_integer(denominators.get('g1_listing_windows'), -1)}",
        denominator=f"{EXPECTED_G1_RICS}/{EXPECTED_G1_LISTING_WINDOWS}",
    )
    issues = _verify_declared_artifacts(manifest, directory, sources)
    checks.add("g2", "packet_artifact_hashes_verified", not issues, detail=" | ".join(issues))


def _evaluate_pit_plan(
    manifest: Mapping[str, Any],
    directory: Path,
    g1_manifest_path: Path | None,
    checks: _Checks,
    sources: _Sources,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_path = _artifact_path(manifest, "selected_universe", directory, "selected_universe.csv")
    sources.add(selected_path)
    selected_raw = _read_table(selected_path)
    selected = _normalise_windows(selected_raw)
    full = bool(
        _full_manifest(manifest)
        and manifest.get("point_in_time_plan") is True
        and manifest.get("current_survivor") is False
        and manifest.get("window_denominator_preserved") is True
        and manifest.get("g1_independent_validation_admitted") is True
        and set(manifest.get("datasets", [])) == set(G3_DATASETS)
        and _integer(manifest.get("selected_names"), -1) == EXPECTED_G1_RICS
        and _integer(manifest.get("selected_listing_windows"), -1) == EXPECTED_G1_LISTING_WINDOWS
    )
    checks.add("pit_plan", "full_admitted_plan_manifest", full)
    checks.add(
        "pit_plan",
        "selected_universe_exact_denominator",
        len(selected) == EXPECTED_G1_LISTING_WINDOWS
        and selected["ric"].nunique() == EXPECTED_G1_RICS,
        numerator=f"{selected['ric'].nunique() if not selected.empty else 0}/{len(selected)}",
        denominator=f"{EXPECTED_G1_RICS}/{EXPECTED_G1_LISTING_WINDOWS}",
    )
    checks.add(
        "pit_plan",
        "listing_windows_are_unique_and_pit_usable",
        not selected.empty
        and not selected.duplicated(["ric", "start", "end"]).any()
        and selected[["ric", "start", "known_at"]].notna().all().all()
        and selected["pit_usable"].all(),
    )
    hash_issues = _verify_declared_artifact(manifest, "selected_universe", selected_path)
    checks.add("pit_plan", "selected_universe_hash_verified", not hash_issues, detail=" | ".join(hash_issues))
    provenance = manifest.get("provenance", {}).get("g1_independent_validation", {})
    expected_g1_hash = str(provenance.get("sha256", "")).lower()
    checks.add(
        "pit_plan",
        "plan_binds_current_g1_validation",
        g1_manifest_path is not None
        and g1_manifest_path.is_file()
        and expected_g1_hash == _sha256(g1_manifest_path),
    )
    return selected, selected.copy()


def _evaluate_execution_plan(
    manifest: Mapping[str, Any],
    directory: Path,
    pit_manifest_path: Path | None,
    pit_directory: Path,
    selected: pd.DataFrame,
    checks: _Checks,
    sources: _Sources,
) -> dict[str, pd.DataFrame]:
    ready = bool(
        str(manifest.get("status", "")).lower() in {"complete", "ready"}
        and manifest.get("evidence_scope") == "full_execution_plan"
        and manifest.get("scorecard_eligible") is True
        and manifest.get("g1_independent_validation_admitted") is True
        and _integer(manifest.get("selected_names"), -1) == EXPECTED_G1_RICS
    )
    checks.add("execution_plan", "full_execution_plan_admitted", ready)

    plans: dict[str, pd.DataFrame] = {}
    hash_issues: list[str] = []
    schema_issues: list[str] = []
    all_ids: list[str] = []
    selected_rics = set(selected["ric"].dropna().astype(str))
    for lane, key in _LANE_PLAN_KEYS.items():
        path = _artifact_path(manifest, key, directory, f"{key}.csv")
        sources.add(path)
        plan = _read_table(path)
        plan.attrs["source_path"] = str(path.resolve())
        plan.attrs["source_sha256"] = _sha256(path) if path.is_file() else ""
        plans[lane] = plan
        hash_issues.extend(f"{lane}: {issue}" for issue in _verify_declared_artifact(manifest, key, path))
        required = {"plan_row_id", "dataset_id", "start_date", "end_date", "ric_list", "route"}
        missing = sorted(required - set(plan.columns))
        if missing:
            schema_issues.append(f"{lane}: missing columns {missing}")
            continue
        ids = plan["plan_row_id"].astype(str)
        if ids.eq("").any() or ids.duplicated().any():
            schema_issues.append(f"{lane}: blank or duplicate plan_row_id")
        all_ids.extend(ids.tolist())
        expected_dataset = _LANE_DATASETS[lane]
        if not plan["dataset_id"].astype(str).eq(expected_dataset).all():
            schema_issues.append(f"{lane}: wrong dataset_id")
        lane_rics = set(_plan_rics(plan))
        if lane_rics != selected_rics:
            schema_issues.append(
                f"{lane}: RIC coverage {len(lane_rics)} does not equal selected {len(selected_rics)}"
            )
        expected_count = _integer(manifest.get(f"{key}_rows"), -1)
        if expected_count != len(plan):
            schema_issues.append(f"{lane}: manifest row count {expected_count} != {len(plan)}")

    checks.add("execution_plan", "all_plan_hashes_verified", not hash_issues, detail=" | ".join(hash_issues))
    checks.add(
        "execution_plan", "all_lanes_cover_full_selected_universe", not schema_issues, detail=" | ".join(schema_issues)
    )
    checks.add(
        "execution_plan",
        "plan_row_ids_globally_unique",
        bool(all_ids) and len(all_ids) == len(set(all_ids)),
        numerator=len(set(all_ids)),
        denominator=len(all_ids),
    )
    declared_total = _integer(manifest.get("total_plan_rows"), -1)
    actual_total = sum(len(plan) for plan in plans.values())
    checks.add(
        "execution_plan",
        "total_plan_denominator_reconciles",
        declared_total == actual_total,
        numerator=actual_total,
        denominator=declared_total,
    )

    selected_path = pit_directory / "selected_universe.csv"
    input_hashes = manifest.get("input_sha256", {})
    binding_issues: list[str] = []
    for path in (pit_manifest_path, selected_path.resolve()):
        if path is None or not path.is_file():
            binding_issues.append(f"missing PIT binding source {path}")
            continue
        expected = _hash_for_path(input_hashes, path)
        if expected != _sha256(path):
            binding_issues.append(f"execution plan does not bind {path}")
    checks.add(
        "execution_plan", "execution_plan_binds_full_pit_plan", not binding_issues, detail=" | ".join(binding_issues)
    )
    return plans


def _evaluate_sector_routing(
    path: Path,
    selected: pd.DataFrame,
    fundamentals_plan: pd.DataFrame,
    checks: _Checks,
    sources: _Sources,
) -> pd.DataFrame:
    sources.add(path)
    sector = _read_table(path)
    required = {"ric", "fundamental_route_class"}
    schema_ok = required.issubset(sector.columns)
    selected_rics = set(selected["ric"].dropna().astype(str))
    sector_rics = set(sector["ric"].dropna().astype(str)) if "ric" in sector else set()
    route_values = set(sector["fundamental_route_class"].dropna().astype(str)) if schema_ok else set()
    checks.add(
        "sector_routing",
        "one_complete_route_per_admitted_ric",
        schema_ok
        and not sector["ric"].astype(str).duplicated().any()
        and sector_rics == selected_rics
        and route_values.issubset({"general", "bank", "insurance"})
        and len(sector) == EXPECTED_G1_RICS,
        numerator=len(sector_rics),
        denominator=EXPECTED_G1_RICS,
    )
    route_issues: list[str] = []
    if schema_ok and {"ric_list", "route_class"}.issubset(fundamentals_plan.columns):
        route_map = sector.set_index("ric")["fundamental_route_class"].astype(str).to_dict()
        for row in fundamentals_plan.itertuples(index=False):
            planned = _split_rics(getattr(row, "ric_list"))
            wrong = [ric for ric in planned if route_map.get(ric) != str(getattr(row, "route_class"))]
            if wrong:
                route_issues.append(f"{getattr(row, 'plan_row_id')}: mismatched RICs {wrong[:5]}")
    else:
        route_issues.append("fundamentals plan lacks route_class/ric_list")
    checks.add(
        "sector_routing",
        "fundamentals_plan_uses_complete_sector_routes",
        not route_issues,
        detail=" | ".join(route_issues),
    )
    return sector


def _evaluate_decision_calendar(
    path: Path,
    selected: pd.DataFrame,
    checks: _Checks,
    sources: _Sources,
) -> pd.DataFrame:
    sources.add(path)
    calendar = _read_table(path)
    time_col = _first_column(calendar, ("decision_time_utc", "decision_at_utc"))
    entity_col = _first_column(calendar, ("ric", "refinitiv_ric", "entity"))
    if time_col is None:
        checks.add("decision_calendar", "timezone_aware_decision_times", False, detail="missing decision time")
        return pd.DataFrame()
    times, valid = _strict_utc(calendar[time_col])
    normalized = pd.DataFrame({"decision_time_utc": times})
    if entity_col is not None:
        normalized["ric"] = calendar[entity_col].astype("string")
    checks.add(
        "decision_calendar",
        "timezone_aware_decision_times",
        not calendar.empty and valid.all() and not normalized["decision_time_utc"].duplicated().all(),
        numerator=int(valid.sum()),
        denominator=len(valid),
    )
    if "ric" in normalized:
        selected_rics = set(selected["ric"].dropna().astype(str))
        covered = set(normalized["ric"].dropna().astype(str))
        checks.add(
            "decision_calendar",
            "calendar_covers_full_admitted_ric_denominator",
            selected_rics.issubset(covered),
            numerator=len(selected_rics & covered),
            denominator=len(selected_rics),
        )
        duplicate_subset = ["ric", "decision_time_utc"]
    else:
        checks.add("decision_calendar", "calendar_covers_full_admitted_ric_denominator", True, detail="global calendar")
        duplicate_subset = ["decision_time_utc"]
    checks.add(
        "decision_calendar",
        "zero_duplicate_decisions",
        not normalized.duplicated(duplicate_subset).any(),
    )
    return normalized.dropna(subset=["decision_time_utc"]).sort_values("decision_time_utc").reset_index(drop=True)


def _evaluate_lane(
    lane: str,
    run: JapanG3TerminalRun | None,
    plan: pd.DataFrame,
    windows: pd.DataFrame,
    sector: pd.DataFrame,
    decision_calendar: pd.DataFrame,
    checks: _Checks,
    sources: _Sources,
) -> dict[str, int]:
    stats = {"terminal_rows": 0, "history_rows": 0}
    component = f"lane_{lane}"
    if run is None:
        checks.add(component, "full_non_qa_producer_manifest", False, detail="missing lane")
        checks.add(component, "terminal_plan_denominator_reconciles", False, detail="missing lane")
        checks.add(component, "terminal_payloads_are_valid", False, detail="missing lane")
        checks.add(component, "independent_validator_passed", False, detail="missing lane")
        return stats

    producer_manifest_path = sources.add(run.producer_dir / "manifest.json")
    validator_manifest_path = sources.add(run.validator_dir / "manifest.json")
    producer = _read_json(producer_manifest_path)
    validator = _read_json(validator_manifest_path)
    producer_ok = bool(
        _full_manifest(producer)
        and str(producer.get("lane", "")) == lane
        and _integer(producer.get("critical_blocker_count"), 0) == 0
        and not any(producer.get(flag) is True for flag in ("partial", "smoke", "qa", "research_qa"))
    )
    checks.add(component, "full_non_qa_producer_manifest", producer_ok)

    plan_key = _LANE_PLAN_KEYS[lane]
    expected_plan_hash = _sha256_from_frame_source(plan, run.producer_dir, producer, plan_key)
    actual_plan_hash = str(producer.get("plan_sha256", "")).lower()
    checks.add(
        component,
        "producer_binds_exact_execution_plan",
        bool(expected_plan_hash and actual_plan_hash == expected_plan_hash),
        detail=f"declared={actual_plan_hash or '<missing>'}; expected={expected_plan_hash or '<missing>'}",
    )

    row_manifest_path = _artifact_path(producer, "row_manifest", run.producer_dir, "row_manifest.csv")
    sources.add(row_manifest_path)
    terminal = _read_table(row_manifest_path)
    stats["terminal_rows"] = len(terminal)
    required = {"plan_row_id", "dataset_id", "status", "output_path", "history_rows"}
    schema_ok = required.issubset(terminal.columns)
    plan_ids = set(plan["plan_row_id"].astype(str)) if "plan_row_id" in plan else set()
    terminal_ids = set(terminal["plan_row_id"].astype(str)) if "plan_row_id" in terminal else set()
    terminal_statuses = set(terminal["status"].astype(str)) if "status" in terminal else set()
    terminal_ok = bool(
        schema_ok
        and not terminal["plan_row_id"].astype(str).duplicated().any()
        and terminal_ids == plan_ids
        and terminal_statuses.issubset(_TERMINAL_STATUSES)
        and len(terminal) == len(plan)
    )
    checks.add(
        component,
        "terminal_plan_denominator_reconciles",
        terminal_ok,
        numerator=len(terminal_ids),
        denominator=len(plan_ids),
        detail=f"statuses={sorted(terminal_statuses)}",
    )
    declared_rows = _integer(producer.get("planned_rows"), -1)
    declared_terminal = _integer(producer.get("terminal_rows"), -1)
    checks.add(
        component,
        "producer_terminal_counts_reconcile",
        declared_rows == len(plan) and declared_terminal == len(terminal),
        detail=f"planned={declared_rows}/{len(plan)}; terminal={declared_terminal}/{len(terminal)}",
    )
    row_hash_issues = _verify_declared_artifact(producer, "row_manifest", row_manifest_path)
    checks.add(component, "terminal_manifest_hash_verified", not row_hash_issues, detail=" | ".join(row_hash_issues))

    plan_by_id = (
        plan.assign(plan_row_id=plan["plan_row_id"].astype(str)).set_index("plan_row_id", drop=False)
        if "plan_row_id" in plan and not plan.empty
        else pd.DataFrame()
    )
    payload_issues: list[str] = []
    all_data: list[pd.DataFrame] = []
    structural_count = 0
    if schema_ok:
        for row in terminal.itertuples(index=False):
            row_id = str(getattr(row, "plan_row_id"))
            status = str(getattr(row, "status"))
            if row_id not in plan_by_id.index:
                payload_issues.append(f"{row_id}: not in plan")
                continue
            plan_row = plan_by_id.loc[row_id]
            if isinstance(plan_row, pd.DataFrame):
                payload_issues.append(f"{row_id}: duplicate plan row")
                continue
            if str(getattr(row, "dataset_id")) != _LANE_DATASETS[lane]:
                payload_issues.append(f"{row_id}: wrong dataset")
            if status == "structural_not_applicable":
                structural_count += 1
                payload_issues.extend(_validate_structural_terminal(row, plan_row, windows))
                continue
            if status != "completed":
                payload_issues.append(f"{row_id}: non-terminal status {status}")
                continue
            output_path = _resolve_output_path(getattr(row, "output_path"), run.producer_dir)
            sources.add(output_path)
            data = _read_table(output_path)
            if data.empty:
                payload_issues.append(f"{row_id}: completed output is empty")
                continue
            declared_history = _integer(getattr(row, "history_rows"), -1)
            if declared_history != len(data):
                payload_issues.append(f"{row_id}: history_rows {declared_history} != {len(data)}")
            data = data.copy()
            data["_plan_row_id"] = row_id
            data["_lane"] = lane
            all_data.append(data)
            stats["history_rows"] += len(data)

    combined = pd.concat(all_data, ignore_index=True, sort=False) if all_data else pd.DataFrame()
    data_issues, metrics = _validate_payload_rows(
        lane, combined, plan_by_id, windows, decision_calendar
    )
    payload_issues.extend(data_issues)
    checks.add(
        component,
        "terminal_payloads_are_valid",
        not payload_issues and (not terminal_ok or len(terminal) == structural_count or not combined.empty),
        numerator=stats["history_rows"],
        denominator=stats["history_rows"],
        detail=" | ".join(payload_issues[:50]),
    )
    checks.add(
        component,
        "zero_duplicate_payload_rows",
        metrics["duplicate_rows"] == 0,
        numerator=metrics["duplicate_rows"],
        denominator=stats["history_rows"],
    )
    checks.add(
        component,
        "zero_rows_outside_plan_or_listing_windows",
        metrics["outside_rows"] == 0,
        numerator=metrics["outside_rows"],
        denominator=stats["history_rows"],
    )
    checks.add(
        component,
        "known_at_decision_eligibility_recomputed",
        metrics["pit_rows"] == stats["history_rows"] and metrics["pit_violations"] == 0,
        numerator=metrics["pit_rows"] - metrics["pit_violations"],
        denominator=stats["history_rows"],
    )

    validator_ok, validator_detail = _independent_validator_ok(
        validator, producer, producer_manifest_path, validator_manifest_path
    )
    checks.add(component, "independent_validator_passed", validator_ok, detail=validator_detail)
    return stats


def _evaluate_cross_lane_separation(
    plans: Mapping[str, pd.DataFrame],
    lanes: Mapping[str, JapanG3TerminalRun],
    checks: _Checks,
) -> None:
    issues: list[str] = []
    for lane, plan in plans.items():
        if "route" not in plan:
            issues.append(f"{lane}: missing route")
            continue
        routes = set(plan["route"].dropna().astype(str))
        if lane == "analyst_detail":
            if routes != {"get_data_parameterized_self_keyed"}:
                issues.append(f"analyst_detail: routes={sorted(routes)}")
        elif "get_data_parameterized_self_keyed" in routes:
            issues.append(f"{lane}: contains analyst-detail route")
    resolved_dirs = {lane: run.producer_dir.resolve() for lane, run in lanes.items()}
    if len(resolved_dirs) != len(set(resolved_dirs.values())):
        issues.append("producer directories are shared across lanes")
    checks.add("lanes", "ordinary_and_analyst_detail_are_separate", not issues, detail=" | ".join(issues))
    validator_checks = [
        row
        for row in checks.rows
        if row["component"].startswith("lane_") and row["check_id"] == "independent_validator_passed"
    ]
    checks.add(
        "validators",
        "all_lane_validators_are_independent",
        len(validator_checks) == len(G3_LANES) and all(row["passed"] for row in validator_checks),
        numerator=sum(int(row["passed"]) for row in validator_checks),
        denominator=len(G3_LANES),
    )
    pit_checks = [
        row
        for row in checks.rows
        if row["component"].startswith("lane_")
        and row["check_id"] == "known_at_decision_eligibility_recomputed"
    ]
    checks.add(
        "pit",
        "all_rows_known_at_or_before_eligible_decision",
        len(pit_checks) == len(G3_LANES) and all(row["passed"] for row in pit_checks),
        numerator=sum(int(row["passed"]) for row in pit_checks),
        denominator=len(G3_LANES),
    )


def _validate_structural_terminal(
    row: Any, plan_row: pd.Series, windows: pd.DataFrame
) -> list[str]:
    row_id = str(getattr(row, "plan_row_id"))
    reason = str(getattr(row, "structural_reason", "")).strip().lower()
    raw_output = getattr(row, "output_path", "")
    output = "" if raw_output is None or pd.isna(raw_output) else str(raw_output).strip()
    history_rows = _integer(getattr(row, "history_rows", -1), -1)
    issues: list[str] = []
    if reason not in _STRUCTURAL_REASONS:
        issues.append(f"{row_id}: invalid structural reason {reason or '<missing>'}")
    if output or history_rows != 0:
        issues.append(f"{row_id}: structural terminal must have no output and zero rows")
    planned_rics = _split_rics(plan_row.get("ric_list"))
    start = pd.to_datetime(plan_row.get("start_date"), errors="coerce")
    end = pd.to_datetime(plan_row.get("end_date"), errors="coerce")
    if not planned_rics or pd.isna(start) or pd.isna(end):
        issues.append(f"{row_id}: structural applicability cannot be recomputed")
        return issues
    overlaps = 0
    for ric in planned_rics:
        ric_windows = windows.loc[windows["ric"].eq(ric)]
        for window in ric_windows.itertuples(index=False):
            window_end = pd.Timestamp.max.normalize() if window.right_open else window.end
            if window.start <= end and window_end >= start:
                overlaps += 1
                break
    if overlaps:
        issues.append(f"{row_id}: structural state overlaps {overlaps} admitted listing windows")
    return issues


def _validate_payload_rows(
    lane: str,
    data: pd.DataFrame,
    plan_by_id: pd.DataFrame,
    windows: pd.DataFrame,
    decision_calendar: pd.DataFrame,
) -> tuple[list[str], dict[str, int]]:
    metrics = {"duplicate_rows": 0, "outside_rows": 0, "pit_rows": 0, "pit_violations": 0}
    if data.empty:
        return [], metrics
    issues: list[str] = []
    ric_col = _first_column(data, ("ric", "refinitiv_ric", "instrument"))
    known_col = _first_column(data, ("available_time_utc", "source_known_at_utc", "known_at"))
    event_col = _first_column(
        data,
        ("observation_date", "estimate_date", "calc_date", "asof_date", "event_date", "period_end_date"),
    )
    if ric_col is None or known_col is None or event_col is None:
        missing = [name for name, value in (("ric", ric_col), ("known_at", known_col), ("event_date", event_col)) if value is None]
        issues.append(f"{lane}: payload missing semantic columns {missing}")
        metrics["pit_violations"] = len(data)
        return issues, metrics

    normalized = data.copy()
    normalized["_ric"] = normalized[ric_col].astype("string")
    normalized["_event_date"] = pd.to_datetime(normalized[event_col], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize()
    known, valid_known = _strict_utc(normalized[known_col])
    normalized["_known_at"] = known
    metrics["pit_rows"] = len(normalized)
    metrics["pit_violations"] += int((~valid_known).sum())
    if "pit_timestamp_violation" in normalized:
        metrics["pit_violations"] += int(_bool_series(normalized["pit_timestamp_violation"]).sum())

    duplicate_cols = ["_plan_row_id", "_ric", "_event_date", "_known_at"]
    for candidate in (
        "field_role",
        "source_field",
        "frequency",
        "fiscal_period",
        "estimate_type",
        "forecast_period",
        "analyst_name",
        "broker_name",
    ):
        if candidate in normalized:
            duplicate_cols.append(candidate)
    metrics["duplicate_rows"] = int(normalized.duplicated(duplicate_cols, keep=False).sum())

    outside = normalized["_event_date"].isna().copy()
    planned_ric_violations = 0
    for row_id, indexes in normalized.groupby("_plan_row_id", sort=False).groups.items():
        if row_id not in plan_by_id.index:
            outside.loc[indexes] = True
            continue
        plan_row = plan_by_id.loc[row_id]
        allowed = set(_split_rics(plan_row.get("ric_list")))
        start = pd.to_datetime(plan_row.get("start_date"), errors="coerce")
        end = pd.to_datetime(plan_row.get("end_date"), errors="coerce")
        group = normalized.loc[indexes]
        invalid = ~group["_ric"].astype(str).isin(allowed)
        planned_ric_violations += int(invalid.sum())
        invalid |= group["_event_date"].lt(start) | group["_event_date"].gt(end)
        outside.loc[indexes] |= invalid.to_numpy()
    outside |= _outside_listing_windows(normalized["_ric"], normalized["_event_date"], windows)
    metrics["outside_rows"] = int(outside.sum())
    if planned_ric_violations:
        issues.append(f"{lane}: {planned_ric_violations} rows use RICs outside their plan row")

    eligible, claimed_mismatch = _decision_eligibility(normalized, decision_calendar)
    metrics["pit_violations"] += int((~eligible).sum()) + claimed_mismatch
    if claimed_mismatch:
        issues.append(f"{lane}: {claimed_mismatch} claimed eligible decisions do not recompute")
    return issues, metrics


def _decision_eligibility(data: pd.DataFrame, calendar: pd.DataFrame) -> tuple[pd.Series, int]:
    eligible = pd.Series(False, index=data.index, dtype=bool)
    assigned = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    if calendar.empty or "decision_time_utc" not in calendar:
        return eligible, 0
    if "ric" in calendar:
        calendar_groups = {
            str(ric): group["decision_time_utc"].dropna().sort_values().reset_index(drop=True)
            for ric, group in calendar.groupby("ric", sort=False)
        }
        groups: Iterable[tuple[Any, Any]] = data.groupby("_ric", sort=False).groups.items()
    else:
        global_times = calendar["decision_time_utc"].dropna().sort_values().reset_index(drop=True)
        calendar_groups = {"__global__": global_times}
        groups = [("__global__", data.index)]
    for ric, indexes in groups:
        times = calendar_groups.get(str(ric))
        if times is None or times.empty:
            continue
        known = data.loc[indexes, "_known_at"]
        positions = times.searchsorted(known, side="left")
        valid = known.notna() & (positions < len(times))
        valid_indexes = known.index[valid]
        if len(valid_indexes):
            chosen = times.iloc[positions[valid].astype(int)].array
            assigned.loc[valid_indexes] = chosen
            eligible.loc[valid_indexes] = known.loc[valid_indexes].le(assigned.loc[valid_indexes])
    claimed_col = _first_column(data, ("eligible_decision_time_utc", "decision_time_utc"))
    mismatch = 0
    if claimed_col is not None:
        claimed, valid = _strict_utc(data[claimed_col])
        mismatch = int((~valid | claimed.ne(assigned)).sum())
    return eligible, mismatch


def _outside_listing_windows(rics: pd.Series, dates: pd.Series, windows: pd.DataFrame) -> pd.Series:
    outside = pd.Series(True, index=rics.index, dtype=bool)
    grouped_windows = {str(ric): group for ric, group in windows.groupby("ric", sort=False)}
    for ric, indexes in rics.groupby(rics, sort=False).groups.items():
        candidates = grouped_windows.get(str(ric))
        if candidates is None or candidates.empty:
            continue
        values = dates.loc[indexes]
        inside = pd.Series(False, index=indexes, dtype=bool)
        for window in candidates.itertuples(index=False):
            within = values.ge(window.start)
            if not window.right_open:
                within &= values.le(window.end)
            inside |= within.fillna(False)
        outside.loc[indexes] = ~inside
    return outside


def _independent_validator_ok(
    validator: Mapping[str, Any],
    producer: Mapping[str, Any],
    producer_manifest_path: Path | None,
    validator_manifest_path: Path | None,
) -> tuple[bool, str]:
    producer_id = str(producer.get("run_id", "")).strip()
    validator_id = str(validator.get("validator_run_id", validator.get("run_id", ""))).strip()
    declared_producer = str(validator.get("producer_run_id", "")).strip()
    independent = validator.get("independent_validation") is True or bool(
        validator.get("independent_audit") is True
        and validator.get("validator_type") == "independent_process"
    )
    expected_manifest_hash = str(validator.get("producer_manifest_sha256", "")).lower()
    actual_manifest_hash = (
        _sha256(producer_manifest_path)
        if producer_manifest_path is not None and producer_manifest_path.is_file()
        else ""
    )
    valid = bool(
        _full_manifest(validator)
        and independent
        and validator.get("source_checksums_verified") is True
        and _integer(validator.get("critical_blocker_count"), -1) == 0
        and producer_id
        and declared_producer == producer_id
        and validator_id
        and validator_id != producer_id
        and producer_manifest_path is not None
        and validator_manifest_path is not None
        and producer_manifest_path.parent != validator_manifest_path.parent
        and _SHA256_RE.fullmatch(expected_manifest_hash) is not None
        and expected_manifest_hash == actual_manifest_hash
    )
    detail = (
        f"producer={producer_id or '<missing>'}; validator={validator_id or '<missing>'}; "
        f"declared_producer={declared_producer or '<missing>'}; manifest_hash_bound={expected_manifest_hash == actual_manifest_hash}"
    )
    return valid, detail


def _normalise_windows(frame: pd.DataFrame) -> pd.DataFrame:
    ric_col = _first_column(frame, ("refinitiv_ric", "ric"))
    start_col = _first_column(frame, ("listing_start", "start_date", "valid_from"))
    end_col = _first_column(frame, ("listing_end", "end_date", "valid_to"))
    known_col = _first_column(frame, ("known_at", "source_known_at_utc", "available_time_utc"))
    if None in (ric_col, start_col, end_col, known_col):
        return pd.DataFrame(columns=["ric", "start", "end", "known_at", "right_open", "pit_usable"])
    known, _valid = _strict_utc(frame[known_col])
    result = pd.DataFrame(
        {
            "ric": frame[ric_col].astype("string").str.strip(),
            "start": pd.to_datetime(frame[start_col], errors="coerce").dt.tz_localize(None).dt.normalize(),
            "end": pd.to_datetime(frame[end_col], errors="coerce").dt.tz_localize(None).dt.normalize(),
            "known_at": known,
            "right_open": _bool_series(frame.get("right_open", pd.Series(False, index=frame.index))),
            "pit_usable": _bool_series(frame.get("pit_usable", pd.Series(False, index=frame.index))),
        }
    )
    return result


def _full_manifest(manifest: Mapping[str, Any]) -> bool:
    status = str(manifest.get("status", "")).lower()
    scope = str(manifest.get("evidence_scope", "")).lower()
    labels = " ".join(
        str(manifest.get(key, "")) for key in ("artifact_type", "mode", "run_id", "scope")
    ).lower()
    return bool(
        status in _PASS_STATUSES
        and manifest.get("scorecard_eligible") is True
        and scope == "full"
        and manifest.get("partial") is not True
        and not any(marker in labels for marker in _INVALID_SCOPE_MARKERS)
    )


def _verify_declared_artifacts(
    manifest: Mapping[str, Any], directory: Path, sources: _Sources
) -> list[str]:
    artifacts = manifest.get("artifacts", {})
    hashes = manifest.get("artifact_sha256", {})
    if not isinstance(artifacts, Mapping) or not isinstance(hashes, Mapping):
        return ["artifact path/hash mappings are missing"]
    issues: list[str] = []
    for key, expected in hashes.items():
        raw_path = artifacts.get(key)
        if raw_path is None:
            issues.append(f"{key}: path not declared")
            continue
        path = _resolve_declared_path(raw_path, directory)
        sources.add(path)
        if not path.is_file():
            issues.append(f"{key}: file missing")
        elif str(expected).lower() != _sha256(path):
            issues.append(f"{key}: SHA-256 mismatch")
    return issues


def _verify_declared_artifact(
    manifest: Mapping[str, Any], key: str, path: Path
) -> list[str]:
    artifacts = manifest.get("artifacts", {})
    hashes = manifest.get("artifact_sha256", {})
    issues: list[str] = []
    if not isinstance(artifacts, Mapping) or key not in artifacts:
        issues.append(f"{key}: artifact path not declared")
    else:
        declared = _resolve_declared_path(artifacts[key], path.parent)
        if declared != path.resolve():
            issues.append(f"{key}: declared path mismatch")
    expected = str(hashes.get(key, "")).lower() if isinstance(hashes, Mapping) else ""
    if _SHA256_RE.fullmatch(expected) is None:
        issues.append(f"{key}: valid SHA-256 not declared")
    elif not path.is_file() or _sha256(path) != expected:
        issues.append(f"{key}: SHA-256 mismatch")
    return issues


def _artifact_path(
    manifest: Mapping[str, Any], key: str, directory: Path, default_name: str
) -> Path:
    artifacts = manifest.get("artifacts", {})
    raw = artifacts.get(key) if isinstance(artifacts, Mapping) else None
    return _resolve_declared_path(raw, directory) if raw else (directory / default_name).resolve()


def _resolve_declared_path(raw: Any, directory: Path) -> Path:
    path = Path(str(raw)).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def _resolve_output_path(raw: Any, producer_dir: Path) -> Path:
    text = "" if raw is None or pd.isna(raw) else str(raw).strip()
    if not text:
        return (producer_dir / "__missing_output__").resolve()
    return _resolve_declared_path(text, producer_dir)


def _sha256_from_frame_source(
    plan: pd.DataFrame,
    producer_dir: Path,
    producer: Mapping[str, Any],
    plan_key: str,
) -> str:
    source_path = Path(str(plan.attrs.get("source_path", ""))).resolve()
    source_hash = str(plan.attrs.get("source_sha256", "")).lower()
    if not source_path.is_file() or _SHA256_RE.fullmatch(source_hash) is None:
        return ""
    raw = producer.get("plan_path")
    if raw:
        candidate = _resolve_declared_path(raw, producer_dir)
        if candidate != source_path:
            return ""
    declared = str(
        producer.get("plan_sha256")
        or producer.get("expected_plan_sha256")
        or producer.get(f"{plan_key}_sha256", "")
    ).lower()
    return source_hash if declared == source_hash else ""


def _hash_for_path(mapping: Any, path: Path) -> str:
    if not isinstance(mapping, Mapping):
        return ""
    resolved = path.resolve()
    for raw, digest in mapping.items():
        try:
            candidate = Path(str(raw)).expanduser().resolve()
        except (OSError, RuntimeError):
            continue
        if candidate == resolved:
            return str(digest).lower()
    return ""


def _read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.is_file():
        return pd.DataFrame()
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if path.suffix.lower() in {".csv", ".txt"}:
            return pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError):
        return pd.DataFrame()
    return pd.DataFrame()


def _strict_utc(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    parsed: list[pd.Timestamp | pd.NaT] = []
    valid: list[bool] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
            ok = not pd.isna(timestamp) and timestamp.tzinfo is not None
            parsed.append(timestamp.tz_convert("UTC") if ok else pd.NaT)
            valid.append(bool(ok))
        except (TypeError, ValueError, OverflowError):
            parsed.append(pd.NaT)
            valid.append(False)
    return (
        pd.Series(parsed, index=values.index, dtype="datetime64[ns, UTC]"),
        pd.Series(valid, index=values.index, dtype=bool),
    )


def _first_column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    return next((candidate for candidate in candidates if candidate in frame.columns), None)


def _plan_rics(plan: pd.DataFrame) -> list[str]:
    if "ric_list" not in plan:
        return []
    return sorted({ric for value in plan["ric_list"] for ric in _split_rics(value)})


def _split_rics(value: Any) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple, set)):
        values: Iterable[Any] = value
    else:
        values = str(value).replace("|", ",").split(",")
    return [str(item).strip() for item in values if str(item).strip()]


def _bool_series(values: pd.Series) -> pd.Series:
    truthy = {"1", "true", "yes", "y", "pass", "passed"}
    return values.map(
        lambda value: value if isinstance(value, bool) else str(value).strip().lower() in truthy
    ).fillna(False).astype(bool)


def _integer(value: Any, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_passed(checks: pd.DataFrame, component: str, check_id: str) -> bool:
    rows = checks.loc[checks["component"].eq(component) & checks["check_id"].eq(check_id), "passed"]
    return bool(len(rows) == 1 and rows.iloc[0])


def _summary(run_id: str, admitted: bool, checks: pd.DataFrame, blockers: Sequence[str]) -> str:
    lines = [
        "# Japan G3 Aggregate Admission",
        "",
        f"Run ID: `{run_id}`",
        f"Status: `{'pass' if admitted else 'failed_admission'}`",
        f"Scorecard eligible: `{admitted}`",
        f"Checks passed: `{int(checks['passed'].sum())}/{len(checks)}`",
        "",
        "The packet binds the admitted 5,752-RIC / 9,706-window G1 denominator and strict G2 admission to the full Japan G3 PIT plan.",
        "It grants only G3 component credit; alpha, production, IBKR, and live-trading promotion remain false.",
        "",
        "## Blockers",
        "",
        *([f"- `{blocker}`" for blocker in blockers] or ["- None."]),
        "",
    ]
    return "\n".join(lines)
