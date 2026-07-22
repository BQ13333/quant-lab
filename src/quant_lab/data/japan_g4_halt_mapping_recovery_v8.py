from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data import japan_g4_halt_mapping_recovery as v1


UTC = timezone.utc
V8_RUN_ID = "JPN-G4-HALTS-200607-202606-V8"
QUOTATION_RUN_ID = "JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2"
PRIOR_RUN_ID = "JPN-G4-HALT-MAPPING-RECOVERY-20260712-V1"
PRIOR_VALIDATION_RUN_ID = (
    "JPN-G4-HALT-MAPPING-RECOVERY-INDEPENDENT-VALIDATION-20260712-V1"
)


def recover_japan_g4_halt_mappings_v8(
    *,
    run_id: str,
    v8_dir: str | Path,
    quotation_dir: str | Path,
    prior_producer_dir: str | Path,
    prior_validation_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_mapping_recovery",
    page_extractor: v1.PageExtractor | None = None,
) -> v1.JapanG4HaltMappingRecoveryArtifacts:
    """Seal event-specific RIC recoveries for the exact unresolved V8 halt set."""
    v1._require_safe_run_id(run_id)
    v8 = Path(v8_dir).resolve()
    quotation = Path(quotation_dir).resolve()
    prior = Path(prior_producer_dir).resolve()
    prior_validation = Path(prior_validation_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    if root.exists():
        raise FileExistsError(f"write-once recovery artifact already exists: {root}")

    paths = {
        "v8_manifest": v8 / "manifest.json",
        "report_event_admission": v8 / "report_event_admission.csv",
        "halt_events": v8 / "halt_suspension_events.parquet",
        "mapping_audit": v8 / "mapping_audit.csv",
        "g1_reference_symbol_map": v8 / "inputs" / "g1_reference_symbol_map.csv",
        "g1_security_master_candidates": v8
        / "inputs"
        / "g1_security_master_candidates.csv",
        "g1_lineage": v8 / "inputs" / "g1_lineage.json",
        "quotation_manifest": quotation / "manifest.json",
        "quotation_request_plan": quotation / "request_plan.csv",
        "quotation_request_plan_hash": quotation / "request_plan.sha256",
        "quotation_file_status": quotation / "file_status.csv",
        "quotation_file_quality": quotation / "file_quality.csv",
        "prior_producer_manifest": prior / "manifest.json",
        "prior_mapping_decisions": prior / "mapping_decisions.csv",
        "prior_validation_manifest": prior_validation / "manifest.json",
        "prior_validation_hashes": prior_validation / "artifact_hashes.json",
        "producer_code": Path(__file__).resolve(),
        "v1_parser_code": Path(v1.__file__).resolve(),
    }
    manifests = _validate_manifests(paths)
    inventory = _input_inventory(paths, manifests)
    failed = inventory.loc[~inventory["hash_match"], "input_id"].tolist()
    if failed:
        raise ValueError(f"input hash verification failed: {failed}")

    admission = _csv(paths["report_event_admission"])
    events = pd.read_parquet(paths["halt_events"]).fillna("")
    mapping_audit = _csv(paths["mapping_audit"])
    symbol_map = _csv(paths["g1_reference_symbol_map"])
    security = _csv(paths["g1_security_master_candidates"])
    prior_decisions = _csv(paths["prior_mapping_decisions"])
    plan = _csv(paths["quotation_request_plan"])
    status = _csv(paths["quotation_file_status"])
    quality = _csv(paths["quotation_file_quality"])

    scoped = v1._scoped_admission(admission)
    scoped_events = v1._join_scoped_events(scoped, events)
    denominator = _denominator_frame(scoped, scoped_events, mapping_audit)
    source_index = v1._source_index(plan, status, quality, quotation)
    extract_pages = page_extractor or v1._extract_layout_pages
    page_cache: dict[Path, list[str]] = {}
    parse_cache: dict[tuple[Path, str, str], list[v1.QuotationRecord]] = {}
    decisions = [
        v1._decide_event(
            event,
            symbol_map=symbol_map,
            security=security,
            source_index=source_index,
            extract_pages=extract_pages,
            page_cache=page_cache,
            parse_cache=parse_cache,
            paths=paths,
            input_inventory=inventory,
        )
        for event in scoped_events.to_dict("records")
    ]
    decision_frame = pd.DataFrame(decisions, columns=v1.DECISION_COLUMNS).sort_values(
        ["event_date", "event_id"]
    )
    accepted = decision_frame.loc[decision_frame["decision"].eq("accepted")].copy()
    rejected = decision_frame.loc[decision_frame["decision"].eq("rejected")].copy()
    completion = v1._completion_plan(rejected)
    sources = v1._source_inventory(decision_frame)
    checks = _producer_checks(
        scoped,
        scoped_events,
        denominator,
        decision_frame,
        sources,
        prior_decisions,
    )

    root.mkdir(parents=True)
    outputs = {
        "denominator_event_ids.csv": denominator,
        "mapping_decisions.csv": decision_frame,
        "accepted_mappings.csv": accepted,
        "rejected_mappings.csv": rejected,
        "completion_plan.csv": completion,
        "source_observations.csv": sources,
        "input_inventory.csv": inventory,
        "producer_checks.csv": checks,
    }
    for name, frame in outputs.items():
        v1._write_csv(root / name, frame)

    hashes = v1._artifact_hashes(
        root, exclude={"artifact_hashes.json", "manifest.json"}
    )
    hashes_path = root / "artifact_hashes.json"
    v1._write_json(hashes_path, hashes)
    passed = bool(checks["passed"].all())
    manifest = {
        "run_id": run_id,
        "artifact_type": "japan_g4_halt_mapping_recovery_v8_packet",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_candidate" if passed else "failed_producer_checks",
        "source_v8_run_id": V8_RUN_ID,
        "source_v8_manifest": str(paths["v8_manifest"]),
        "source_v8_manifest_sha256": v1._sha_path(paths["v8_manifest"]),
        "quotation_run_id": QUOTATION_RUN_ID,
        "quotation_manifest": str(paths["quotation_manifest"]),
        "quotation_manifest_sha256": v1._sha_path(paths["quotation_manifest"]),
        "prior_recovery_run_id": PRIOR_RUN_ID,
        "prior_recovery_manifest_sha256": v1._sha_path(
            paths["prior_producer_manifest"]
        ),
        "prior_validation_run_id": PRIOR_VALIDATION_RUN_ID,
        "prior_validation_manifest_sha256": v1._sha_path(
            paths["prior_validation_manifest"]
        ),
        "denominator_definition": (
            "V8 admitted normalized in-scope halt_suspension_resumption equity rows "
            "with mapping_complete=false and a unique primary_event_id"
        ),
        "denominator_rows": int(len(decision_frame)),
        "denominator_event_ids_sha256": v1._sha_path(
            root / "denominator_event_ids.csv"
        ),
        "accepted_rows": int(len(accepted)),
        "rejected_rows": int(len(rejected)),
        "prior_audit_estimate_rows": 13,
        "prior_audit_estimate_delta": int(len(accepted) - 13),
        "accepted_event_ids_unique": bool(accepted["event_id"].is_unique),
        "fail_closed": True,
        "refinitiv_calls": False,
        "vendor_api_calls": False,
        "inferred_code_t_allowed": False,
        "successor_ric_allowed": False,
        "window_extension": False,
        "source_known_at_required_on_or_before_event": True,
        "latest_eligible_official_observation_only": True,
        "candidate_uniqueness_required": True,
        "independent_validation_required": True,
        "independent_validation_status": "not_run",
        "scorecard_eligible": False,
        "promotion_recommended": False,
        "controller_touched": False,
        "write_once": True,
        "producer_checks_pass": passed,
        "producer_code_sha256": v1._sha_path(paths["producer_code"]),
        "v1_parser_code_sha256": v1._sha_path(paths["v1_parser_code"]),
        "artifact_hashes_sha256": v1._sha_path(hashes_path),
        "outputs": hashes,
    }
    manifest_path = root / "manifest.json"
    v1._write_json(manifest_path, manifest)
    return v1.JapanG4HaltMappingRecoveryArtifacts(
        run_id,
        root,
        manifest_path,
        root / "accepted_mappings.csv",
        root / "rejected_mappings.csv",
        root / "completion_plan.csv",
        manifest,
    )


def _validate_manifests(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    values = {
        key: v1._read_json(paths[key])
        for key in (
            "v8_manifest",
            "quotation_manifest",
            "prior_producer_manifest",
            "prior_validation_manifest",
        )
    }
    v8 = values["v8_manifest"]
    quotation = values["quotation_manifest"]
    prior = values["prior_producer_manifest"]
    validation = values["prior_validation_manifest"]
    if (
        v8.get("run_id") != V8_RUN_ID
        or v8.get("component") != "halt_suspension_events"
        or v8.get("g1_lineage_admitted") is not True
    ):
        raise ValueError("V8 halt manifest is not the admitted bounded input")
    if (
        quotation.get("run_id") != QUOTATION_RUN_ID
        or quotation.get("official_jpx_source") is not True
        or quotation.get("status") != "complete_candidate_source"
        or quotation.get("completed_files") != quotation.get("planned_files")
    ):
        raise ValueError("quotation manifest is not the complete official JPX source")
    if prior.get("run_id") != PRIOR_RUN_ID or prior.get("status") != "complete_candidate":
        raise ValueError("prior recovery packet is not the admitted V1 baseline")
    if (
        validation.get("run_id") != PRIOR_VALIDATION_RUN_ID
        or validation.get("status") != "pass"
        or validation.get("producer_manifest_sha256")
        != v1._sha_path(paths["prior_producer_manifest"])
    ):
        raise ValueError("prior independent validation is not a passing V1 baseline")
    return values


def _input_inventory(
    paths: dict[str, Path], manifests: dict[str, dict[str, Any]]
) -> pd.DataFrame:
    v8_hashes = manifests["v8_manifest"].get("output_sha256", {})
    prior_hashes = manifests["prior_producer_manifest"].get("outputs", {})
    expected = {
        "report_event_admission": v8_hashes.get("report_event_admission.csv", ""),
        "halt_events": v8_hashes.get("halt_suspension_events.parquet", ""),
        "mapping_audit": v8_hashes.get("mapping_audit.csv", ""),
        "g1_reference_symbol_map": v8_hashes.get(
            "inputs/g1_reference_symbol_map.csv", ""
        ),
        "g1_security_master_candidates": v8_hashes.get(
            "inputs/g1_security_master_candidates.csv", ""
        ),
        "g1_lineage": v8_hashes.get("inputs/g1_lineage.json", ""),
        "quotation_request_plan": manifests["quotation_manifest"].get(
            "request_plan_sha256", ""
        ),
        "prior_mapping_decisions": prior_hashes.get("mapping_decisions.csv", ""),
    }
    rows: list[dict[str, Any]] = []
    for input_id, path in paths.items():
        observed = v1._sha_path(path)
        wanted = expected.get(input_id, observed)
        rows.append(
            {
                "input_id": input_id,
                "path": str(path),
                "expected_sha256": wanted,
                "observed_sha256": observed,
                "hash_match": bool(wanted) and wanted == observed,
            }
        )
    return pd.DataFrame(rows)


def _denominator_frame(
    scoped: pd.DataFrame, events: pd.DataFrame, mapping_audit: pd.DataFrame
) -> pd.DataFrame:
    denominator = events[
        ["event_id", "report_event_id", "event_date", "local_code", "issue_name"]
    ].copy()
    audit = mapping_audit[
        ["event_id", "refinitiv_ric", "mapping_status", "mapping_route"]
    ]
    denominator = denominator.merge(audit, on="event_id", how="left", validate="one_to_one")
    expected_ids = set(scoped["primary_event_id"])
    if set(denominator["event_id"]) != expected_ids:
        raise ValueError("V8 denominator event IDs do not reconcile")
    if (
        denominator["refinitiv_ric"].ne("").any()
        or not denominator["mapping_status"].eq("missing_admitted_mapping").all()
        or not denominator["mapping_route"].eq("no_unique_admitted_candidate").all()
    ):
        raise ValueError("V8 denominator contains a mapped or non-missing row")
    return denominator.sort_values(["event_date", "event_id"])


def _producer_checks(
    scoped: pd.DataFrame,
    events: pd.DataFrame,
    denominator: pd.DataFrame,
    decisions: pd.DataFrame,
    sources: pd.DataFrame,
    prior_decisions: pd.DataFrame,
) -> pd.DataFrame:
    base = v1._producer_checks(scoped, events, decisions, sources)
    ids = set(decisions["event_id"])
    prior_accepted = set(
        prior_decisions.loc[prior_decisions["decision"].eq("accepted"), "event_id"]
    )
    prior_rejected = set(
        prior_decisions.loc[prior_decisions["decision"].eq("rejected"), "event_id"]
    )
    extra = pd.DataFrame(
        [
            ("denominator_event_ids_exact", ids == set(denominator["event_id"]), str(len(ids))),
            ("denominator_mapping_status_missing", denominator["refinitiv_ric"].eq("").all(), "all blank"),
            ("prior_accepted_absent", not bool(ids & prior_accepted), str(len(ids & prior_accepted))),
            ("prior_rejections_preserved", len(ids & prior_rejected) == 15, str(len(ids & prior_rejected))),
        ],
        columns=["check", "passed", "detail"],
    )
    return pd.concat([base, extra], ignore_index=True)


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)
