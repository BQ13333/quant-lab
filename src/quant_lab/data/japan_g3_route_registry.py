from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


REQUIRED_ROLES: dict[str, tuple[str, ...]] = {
    "fundamentals_pit": ("revenue", "gross_profit", "eps", "book_value", "release_date"),
    "ibes_estimates_revisions": (
        "eps_estimate",
        "revenue_estimate",
        "recommendation",
        "target_price",
        "revision_time",
    ),
    "earnings_events_actuals": ("earnings_release", "actual_eps", "surprise"),
}


@dataclass(frozen=True)
class JapanG3RouteRegistryArtifacts:
    run_id: str
    artifact_dir: Path
    registry_path: Path
    unresolved_path: Path
    manifest_path: Path
    registry: pd.DataFrame
    unresolved: pd.DataFrame


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()


def build_japan_g3_route_registry(
    historical_probe_dir: str | Path,
    critical_probe_dir: str | Path,
    revision_probe_dir: str | Path,
    fundamentals_qa_dir: str | Path,
    sector_snapshot_dir: str | Path | None = None,
    ibes_events_qa_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    historical_path = Path(historical_probe_dir) / "history_field_map.csv"
    critical_path = Path(critical_probe_dir) / "route_probe_results.csv"
    revision_path = Path(revision_probe_dir) / "accepted_revision_time_map.csv"
    fundamentals_quality_path = Path(fundamentals_qa_dir) / "quality.csv"
    fundamentals_manifest_path = Path(fundamentals_qa_dir) / "manifest.json"
    inputs = [
        historical_path,
        critical_path,
        revision_path,
        fundamentals_quality_path,
        fundamentals_manifest_path,
    ]
    sector_profile_path = (
        Path(sector_snapshot_dir) / "sector_routing_profile.csv" if sector_snapshot_dir is not None else None
    )
    if sector_profile_path is not None:
        inputs.append(sector_profile_path)
    ibes_events_manifest_path = (
        Path(ibes_events_qa_dir) / "manifest.json" if ibes_events_qa_dir is not None else None
    )
    ibes_events_quality_path = (
        Path(ibes_events_qa_dir) / "quality.csv" if ibes_events_qa_dir is not None else None
    )
    if ibes_events_manifest_path is not None:
        inputs.extend([ibes_events_manifest_path, ibes_events_quality_path])
    historical = _read_csv(historical_path)
    critical = _read_csv(critical_path)
    revision = _read_csv(revision_path)
    fundamentals = _read_csv(fundamentals_quality_path)
    ibes_events_manifest = {}
    if ibes_events_manifest_path is not None and ibes_events_manifest_path.exists():
        ibes_events_manifest = json.loads(ibes_events_manifest_path.read_text(encoding="utf-8"))
    ibes_events_quality = (
        _read_csv(ibes_events_quality_path) if ibes_events_quality_path is not None else pd.DataFrame()
    )
    ibes_events_qa_ready = bool(
        ibes_events_manifest.get("status") == "completed_partial"
        and ibes_events_manifest.get("self_keyed_analyst_fields") is True
        and ibes_events_manifest.get("mixed_target_and_recommendation_query_allowed") is False
        and float(ibes_events_manifest.get("known_at_coverage", 0.0)) == 1.0
        and int(ibes_events_manifest.get("pit_timestamp_violations", -1)) == 0
        and int(ibes_events_manifest.get("revision_event_rows", 0)) > 0
        and int(ibes_events_manifest.get("earnings_event_rows", 0)) > 0
        and set(ibes_events_quality.get("dataset_id", pd.Series(dtype=str)).astype(str))
        >= {"ibes_estimates_revisions", "earnings_events_actuals"}
    )

    rows: list[dict[str, object]] = []
    for role in ("revenue", "gross_profit", "eps", "book_value"):
        match = fundamentals.loc[fundamentals.get("field_role", pd.Series(dtype=str)).astype(str).eq(role)]
        if match.empty:
            continue
        item = match.iloc[0]
        accepted = bool(
            int(pd.to_numeric(item.get("history_rows"), errors="coerce") or 0) > 0
            and float(pd.to_numeric(item.get("known_at_coverage"), errors="coerce") or 0.0) == 1.0
            and int(pd.to_numeric(item.get("pit_timestamp_violations"), errors="coerce") or 0) == 0
        )
        if accepted:
            source_field = {
                "revenue": "TR.Revenue|TR.BankTotalRevenue",
                "gross_profit": "TR.GrossProfit",
                "eps": "TR.EPSActValue",
                "book_value": "TR.BookValuePerShare",
            }[role]
            rows.append(
                {
                    "dataset_id": "fundamentals_pit",
                    "field_role": role,
                    "route": "get_data_parameterized",
                    "source_field": source_field,
                    "known_at_field": (
                        "TR.Revenue.calcdate|TR.BankTotalRevenue.calcdate"
                        if role == "revenue"
                        else f"{source_field}.calcdate"
                    ),
                    "applicability_scope": "general_or_bank_fallback" if role == "revenue" else "general_only" if role == "gross_profit" else "all",
                    "acceptance_status": "qa_route_validated",
                    "evidence_path": str(fundamentals_quality_path.resolve()),
                }
            )
    if rows:
        rows.append(
            {
                "dataset_id": "fundamentals_pit",
                "field_role": "release_date",
                "route": "get_data_parameterized",
                "source_field": "metric.calcdate",
                "known_at_field": "metric.calcdate_then_next_weekday_00_jst",
                "applicability_scope": "all_returned_metrics",
                "acceptance_status": "qa_route_validated",
                "evidence_path": str(fundamentals_quality_path.resolve()),
            }
        )

    history_roles = {
        "ibes_estimates_revisions": ("eps_estimate", "revenue_estimate", "recommendation"),
    }
    if not ibes_events_qa_ready:
        history_roles["earnings_events_actuals"] = ("earnings_release",)
    for dataset_id, roles in history_roles.items():
        for role in roles:
            matches = historical.loc[
                historical.get("dataset_id", pd.Series(dtype=str)).astype(str).eq(dataset_id)
                & historical.get("field_role", pd.Series(dtype=str)).astype(str).eq(role)
            ]
            if matches.empty:
                continue
            item = matches.sort_values("coverage", ascending=False).iloc[0]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "field_role": role,
                    "route": "get_history",
                    "source_field": str(item["candidate_field"]),
                    "known_at_field": "history_observation_timestamp",
                    "applicability_scope": "all_with_vendor_coverage",
                    "acceptance_status": "qa_route_validated",
                    "evidence_path": str(historical_path.resolve()),
                }
            )

    critical_roles = {
        "ibes_estimates_revisions": ("target_price",),
    }
    if not ibes_events_qa_ready:
        critical_roles["earnings_events_actuals"] = ("actual_eps", "surprise")
    for dataset_id, roles in critical_roles.items():
        for role in roles:
            matches = critical.loc[
                critical.get("dataset_id", pd.Series(dtype=str)).astype(str).eq(dataset_id)
                & critical.get("field_role", pd.Series(dtype=str)).astype(str).eq(role)
                & critical.get("status", pd.Series(dtype=str)).astype(str).eq("historical_validated")
            ]
            if matches.empty:
                continue
            item = matches.sort_values(["coverage", "non_null_values"], ascending=False).iloc[0]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "field_role": role,
                    "route": "get_history",
                    "source_field": str(item["candidate_field"]),
                    "known_at_field": "history_observation_timestamp",
                    "applicability_scope": "all_with_vendor_coverage",
                    "acceptance_status": "qa_route_validated",
                    "evidence_path": str(critical_path.resolve()),
                }
            )

    if ibes_events_qa_ready:
        rows.append(
            {
                "dataset_id": "ibes_estimates_revisions",
                "field_role": "revision_time",
                "route": "get_data_parameterized_self_keyed",
                "source_field": "TR.RecEstValue.*|TR.TPEstValue.*",
                "known_at_field": "estimate_date_then_next_weekday_00_jst",
                "applicability_scope": "all_with_ibes_detail_coverage",
                "acceptance_status": "qa_route_validated",
                "evidence_path": str(ibes_events_manifest_path.resolve()),
            }
        )
        for role, source_field in (
            ("actual_eps", "TR.EPSActValue"),
            ("surprise", "TR.EPSActSurprise"),
            ("earnings_release", "TR.EPSActValue.calcdate"),
        ):
            rows.append(
                {
                    "dataset_id": "earnings_events_actuals",
                    "field_role": role,
                    "route": "get_data_parameterized",
                    "source_field": source_field,
                    "known_at_field": "metric.calcdate_then_next_weekday_00_jst",
                    "applicability_scope": "all_with_ibes_actual_coverage",
                    "acceptance_status": "qa_route_validated",
                    "evidence_path": str(ibes_events_manifest_path.resolve()),
                }
            )
    else:
        revision_matches = revision.loc[
            revision.get("country", pd.Series(dtype=str)).astype(str).eq("JPN")
            & revision.get("status", pd.Series(dtype=str)).astype(str).eq("accepted")
        ]
        if revision_matches.empty:
            revision_matches = pd.DataFrame()
    if not ibes_events_qa_ready and not revision_matches.empty:
        item = revision_matches.sort_values("unique_date_count", ascending=False).iloc[0]
        rows.append(
            {
                "dataset_id": "ibes_estimates_revisions",
                "field_role": "revision_time",
                "route": "get_data_parameterized",
                "source_field": str(item["field"]),
                "known_at_field": str(item["field"]),
                "applicability_scope": "all_with_ibes_coverage",
                "acceptance_status": "qa_route_validated",
                "evidence_path": str(revision_path.resolve()),
            }
        )

    registry = pd.DataFrame(rows).drop_duplicates(["dataset_id", "field_role"], keep="first")
    required = {(dataset, role) for dataset, roles in REQUIRED_ROLES.items() for role in roles}
    present = set(zip(registry.get("dataset_id", []), registry.get("field_role", [])))
    unresolved = pd.DataFrame(
        [
            {"dataset_id": dataset, "field_role": role, "status": "unresolved_required_route"}
            for dataset, role in sorted(required - present)
        ],
        columns=["dataset_id", "field_role", "status"],
    )
    return registry.sort_values(["dataset_id", "field_role"]).reset_index(drop=True), unresolved, inputs


def write_japan_g3_route_registry_artifacts(
    historical_probe_dir: str | Path,
    critical_probe_dir: str | Path,
    revision_probe_dir: str | Path,
    fundamentals_qa_dir: str | Path,
    artifact_root: str | Path,
    run_id: str,
    sector_snapshot_dir: str | Path | None = None,
    ibes_events_qa_dir: str | Path | None = None,
) -> JapanG3RouteRegistryArtifacts:
    registry, unresolved, inputs = build_japan_g3_route_registry(
        historical_probe_dir,
        critical_probe_dir,
        revision_probe_dir,
        fundamentals_qa_dir,
        sector_snapshot_dir,
        ibes_events_qa_dir,
    )
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    registry_path = artifact_dir / "accepted_route_registry.csv"
    unresolved_path = artifact_dir / "unresolved_roles.csv"
    registry.to_csv(registry_path, index=False)
    unresolved.to_csv(unresolved_path, index=False)

    input_hashes = {str(path.resolve()): _sha256(path) for path in inputs if path.exists()}
    manifest = {
        "run_id": run_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": "complete_qa_routes" if unresolved.empty else "incomplete_routes",
        "scope": "route_registry",
        "evidence_scope": "qa",
        "scorecard_eligible": False,
        "partial": True,
        "research_only": True,
        "production_ready": False,
        "live_execution_ready": False,
        "ibkr_ready": False,
        "required_roles": sum(len(roles) for roles in REQUIRED_ROLES.values()),
        "accepted_roles": int(len(registry)),
        "unresolved_roles": int(len(unresolved)),
        "period_end_as_release_date_allowed": False,
        "sector_schema_note": "Gross profit is structurally inapplicable to banks; use sector-aware completeness, never zero imputation.",
        "sector_routing_profile_included": bool(sector_snapshot_dir is not None),
        "self_keyed_ibes_events_qa_included": bool(ibes_events_qa_dir is not None),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sha256": input_hashes,
        "artifact_sha256": {
            "accepted_route_registry": _sha256(registry_path),
            "unresolved_roles": _sha256(unresolved_path),
        },
        "artifacts": {
            "accepted_route_registry": str(registry_path.resolve()),
            "unresolved_roles": str(unresolved_path.resolve()),
        },
    }
    manifest_path = artifact_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return JapanG3RouteRegistryArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        registry_path=registry_path,
        unresolved_path=unresolved_path,
        manifest_path=manifest_path,
        registry=registry,
        unresolved=unresolved,
    )
