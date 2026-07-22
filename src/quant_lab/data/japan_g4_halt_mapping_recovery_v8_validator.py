from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from quant_lab.data import japan_g4_halt_mapping_recovery_validator as independent


UTC = timezone.utc
V8_RUN_ID = "JPN-G4-HALTS-200607-202606-V8"
QUOTATION_RUN_ID = "JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2"
PRIOR_RUN_ID = "JPN-G4-HALT-MAPPING-RECOVERY-20260712-V1"
PRIOR_VALIDATION_RUN_ID = (
    "JPN-G4-HALT-MAPPING-RECOVERY-INDEPENDENT-VALIDATION-20260712-V1"
)
_PLAIN_CODE_LINE = re.compile(
    r"(?m)^(?P<code>[1-9][0-9A-Z]{3})(?:[ \t]|\u3000)"
)
_FOREIGN_MARKERS = ("<foreign>", "外国株", "外国", "1外", "2外")
_MARKET_TOKEN = re.compile(
    r"^(?:JQG|JQS|TPM|Mothers|Prime|Standard|Growth|PRM|STD|GRT|[12M])(?:\s|$)"
)


def validate_japan_g4_halt_mapping_recovery_v8(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = (
        "artifacts/japan_g4_halt_mapping_recovery_validation"
    ),
    page_extractor: independent.PlainPageExtractor | None = None,
) -> independent.JapanG4HaltMappingRecoveryValidationArtifacts:
    """Independently reconstruct and validate a sealed V8 recovery packet."""
    independent._safe_run_id(run_id)
    producer = Path(producer_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    if root.exists():
        raise FileExistsError(f"write-once validation artifact already exists: {root}")
    root.mkdir(parents=True)
    issues: list[dict[str, str]] = []

    manifest = independent._json(producer / "manifest.json", issues, "producer_manifest")
    declared = independent._json(
        producer / "artifact_hashes.json", issues, "producer_artifact_hashes"
    )
    producer_hash_rows = independent._validate_producer_hashes(
        producer, manifest, declared, issues
    )
    decisions = independent._csv(
        producer / "mapping_decisions.csv", issues, "mapping_decisions"
    )
    accepted = independent._csv(
        producer / "accepted_mappings.csv", issues, "accepted_mappings"
    )
    rejected = independent._csv(
        producer / "rejected_mappings.csv", issues, "rejected_mappings"
    )
    completion = independent._csv(
        producer / "completion_plan.csv", issues, "completion_plan"
    )
    denominator = independent._csv(
        producer / "denominator_event_ids.csv", issues, "denominator_event_ids"
    )
    inputs = independent._csv(
        producer / "input_inventory.csv", issues, "input_inventory"
    )
    sources = independent._csv(
        producer / "source_observations.csv", issues, "source_observations"
    )

    context, lineage_rows = _load_and_rehash_inputs(inputs, manifest, issues)
    denominator_rows = _validate_denominator(
        decisions,
        accepted,
        rejected,
        completion,
        denominator,
        context,
        issues,
    )
    source_rows = independent._validate_source_inventory(sources, context, issues)
    claim_rows = _validate_claims_v8(
        decisions,
        context,
        page_extractor or independent._extract_plain_pages,
        issues,
    )

    independent._write_csv(
        root / "producer_hash_validation.csv", pd.DataFrame(producer_hash_rows)
    )
    independent._write_csv(root / "lineage_validation.csv", pd.DataFrame(lineage_rows))
    independent._write_csv(
        root / "denominator_validation.csv", pd.DataFrame(denominator_rows)
    )
    independent._write_csv(root / "source_validation.csv", pd.DataFrame(source_rows))
    claim_frame = pd.DataFrame(claim_rows)
    independent._write_csv(root / "claim_validation.csv", claim_frame)
    issue_frame = pd.DataFrame(issues, columns=["issue_code", "scope", "detail"])
    independent._write_csv(root / "validation_issues.csv", issue_frame)

    accepted_rows = int(decisions.get("decision", pd.Series(dtype=str)).eq("accepted").sum())
    valid_rows = int(
        claim_frame.get("claim_valid", pd.Series(dtype=bool)).fillna(False).sum()
    )
    passed = bool(not issues and accepted_rows == valid_rows)
    hashes = independent._artifact_hashes(
        root, {"artifact_hashes.json", "manifest.json"}
    )
    hashes_path = root / "artifact_hashes.json"
    independent._write_json(hashes_path, hashes)
    validation_manifest = {
        "run_id": run_id,
        "artifact_type": (
            "japan_g4_halt_mapping_recovery_v8_independent_validation"
        ),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if passed else "failed_validation",
        "independent_validation": True,
        "independent_parser": "pypdf_sequential_text_chunk_reparse_v1",
        "producer_dir": str(producer),
        "producer_manifest_sha256": independent._sha(producer / "manifest.json"),
        "source_v8_run_id": V8_RUN_ID,
        "quotation_run_id": QUOTATION_RUN_ID,
        "denominator_rows": int(len(decisions)),
        "accepted_rows": accepted_rows,
        "accepted_rows_validated": valid_rows,
        "rejected_rows": int(len(decisions) - accepted_rows),
        "source_files_rehashed": int(len(source_rows)),
        "validation_issue_count": int(len(issues)),
        "validation_issue_codes": sorted({row["issue_code"] for row in issues}),
        "all_lineage_hashes_verified": bool(
            lineage_rows and all(bool(row["passed"]) for row in lineage_rows)
        ),
        "exact_event_id_denominator_recomputed": True,
        "known_at_recomputed": True,
        "candidate_uniqueness_recomputed": True,
        "source_records_independently_reparsed": True,
        "unsafe_inventory_paths_rejected": True,
        "mixed_manifests_rejected": True,
        "refinitiv_calls": False,
        "vendor_api_calls": False,
        "inferred_code_t_allowed": False,
        "successor_ric_allowed": False,
        "window_extension": False,
        "controller_touched": False,
        "scorecard_touched": False,
        "component_validation_pass": passed,
        "scorecard_eligible": False,
        "promotion_recommended": passed,
        "write_once": True,
        "validator_code_sha256": independent._sha(Path(__file__).resolve()),
        "artifact_hashes_sha256": independent._sha(hashes_path),
        "outputs": hashes,
    }
    manifest_path = root / "manifest.json"
    independent._write_json(manifest_path, validation_manifest)
    return independent.JapanG4HaltMappingRecoveryValidationArtifacts(
        run_id, root, manifest_path, validation_manifest
    )


def independently_reparse_plain_pages_v8(
    pages: list[str], *, observation_month: str, code: str
) -> dict[str, Any]:
    """Reparse JPX rows from sequential text without using layout coordinates."""
    expected_month = observation_month.replace("-", "/")
    records: list[dict[str, Any]] = []
    for page_number, text in enumerate(pages, start=1):
        if expected_month not in text or "Stock Quotations" not in text:
            continue
        matches = list(_PLAIN_CODE_LINE.finditer(text))
        for position, match in enumerate(matches):
            if match.group("code") != code:
                continue
            end = matches[position + 1].start() if position + 1 < len(matches) else len(text)
            chunk = text[match.start():end].rstrip()
            lines = [line.strip() for line in chunk.splitlines() if line.strip()]
            code_line = lines[0] if lines else ""
            english_index = next(
                (
                    index
                    for index, line in enumerate(lines[1:9], start=1)
                    if re.search(r"[A-Za-z]{3}", line)
                ),
                -1,
            )
            english = lines[english_index] if english_index >= 0 else ""
            details = lines[english_index + 1 : english_index + 14]
            folded = chunk.casefold()
            foreign = any(marker in folded for marker in _FOREIGN_MARKERS)
            market_token = any(_MARKET_TOKEN.match(line) for line in details)
            quote_detail = any(
                re.search(r"\d", line)
                and (
                    "信用" in line
                    or "貸借" in line
                    or bool(_MARKET_TOKEN.match(line))
                )
                for line in details
            )
            records.append(
                {
                    "page_number": page_number,
                    "chunk": chunk,
                    "english_name": english,
                    "explicit_ordinary": "普通" in code_line,
                    "domestic_continuity": bool(
                        english and market_token and quote_detail and not foreign
                    ),
                    "foreign_marker": foreign,
                }
            )
    return {
        "record_count": len(records),
        "records": records,
        "unique_record": records[0] if len(records) == 1 else {},
    }


def _validate_claims_v8(
    decisions: pd.DataFrame,
    context: dict[str, Any],
    extract_pages: independent.PlainPageExtractor,
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pages_cache: dict[Path, list[str]] = {}
    plan = context.get("quotation_request_plan", pd.DataFrame())
    status = context.get("quotation_file_status", pd.DataFrame())
    security = context.get("g1_security_master_candidates", pd.DataFrame())
    symbol_map = context.get("g1_reference_symbol_map", pd.DataFrame())
    for decision in decisions.to_dict("records"):
        event_id = str(decision.get("event_id", ""))
        code = str(decision.get("local_code", ""))
        event_date = pd.Timestamp(str(decision.get("event_date", "")), tz="UTC")
        ric, candidate_count, stable = independent._independent_unique_candidate(
            code, security, symbol_map
        )
        source_request = str(decision.get("source_request_id", ""))
        latest = independent._latest_eligible_request(
            code, event_date, plan, status
        )
        latest_match = bool(source_request and latest == source_request)
        known_at_ok = False
        record_hash_ok = False
        reparse_count = 0
        reparse_domestic = False
        reparse_foreign = False
        name_match = False
        source_hash_ok = False
        path_safe = False
        if source_request:
            raw = Path(str(decision.get("source_pdf_path", ""))).resolve()
            quotation_root = context.get("quotation_root")
            path_safe = isinstance(quotation_root, Path) and independent._within(
                raw, quotation_root
            )
            expected_raw_hash = str(decision.get("source_pdf_sha256", ""))
            source_hash_ok = raw.is_file() and independent._sha(raw) == expected_raw_hash
            known_at = pd.Timestamp(str(decision.get("source_known_at_utc", "")))
            known_at_ok = known_at <= event_date
            record_text = str(decision.get("source_record_text", ""))
            record_hash_ok = independent._sha_text(record_text) == decision.get(
                "source_record_sha256"
            )
            if source_hash_ok:
                pages_cache.setdefault(raw, extract_pages(raw))
                reparsed = independently_reparse_plain_pages_v8(
                    pages_cache[raw],
                    observation_month=str(decision.get("source_month", "")),
                    code=code,
                )
                reparse_count = int(reparsed["record_count"])
                if reparse_count == 1:
                    record = reparsed["unique_record"]
                    reparse_domestic = bool(record["domestic_continuity"])
                    reparse_foreign = bool(record["foreign_marker"])
                    name_match = independent._simple_name_match(
                        str(decision.get("event_issue_name", "")),
                        str(record.get("english_name", "")),
                        str(decision.get("candidate_document_title", "")),
                    )
        accepted = decision.get("decision") == "accepted"
        claim_valid = bool(
            accepted
            and candidate_count == 1
            and ric == decision.get("candidate_ric")
            and stable
            and latest_match
            and known_at_ok
            and record_hash_ok
            and source_hash_ok
            and path_safe
            and reparse_count == 1
            and reparse_domestic
            and not reparse_foreign
            and name_match
        )
        if accepted and not claim_valid:
            independent._issue(
                issues,
                "accepted_claim_not_independently_proven",
                event_id,
                (
                    f"candidate={candidate_count},latest={latest_match},"
                    f"records={reparse_count}"
                ),
            )
        rows.append(
            {
                "event_id": event_id,
                "decision": decision.get("decision", ""),
                "candidate_count_recomputed": candidate_count,
                "candidate_ric_recomputed": ric,
                "stable_identity_recomputed": stable,
                "latest_eligible_request_id": latest,
                "latest_request_match": latest_match,
                "known_at_on_or_before_event": known_at_ok,
                "source_record_hash_match": record_hash_ok,
                "source_pdf_hash_match": source_hash_ok,
                "source_path_safe": path_safe,
                "independent_record_count": reparse_count,
                "independent_domestic_continuity": reparse_domestic,
                "independent_foreign_marker": reparse_foreign,
                "independent_name_match": name_match,
                "claim_valid": claim_valid,
            }
        )
    return rows


def _load_and_rehash_inputs(
    inputs: pd.DataFrame,
    producer_manifest: dict[str, Any],
    issues: list[dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    context: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    by_id: dict[str, Path] = {}
    for item in inputs.to_dict("records"):
        input_id = str(item.get("input_id", ""))
        path = Path(str(item.get("path", ""))).resolve()
        expected = str(item.get("observed_sha256", ""))
        passed = path.is_file() and independent._sha(path) == expected
        rows.append({"lineage_id": input_id, "path": str(path), "passed": passed})
        by_id[input_id] = path
        if not passed:
            independent._issue(
                issues, "input_lineage_hash_mismatch", input_id, str(path)
            )

    required_ids = {
        "v8_manifest",
        "report_event_admission",
        "halt_events",
        "mapping_audit",
        "g1_reference_symbol_map",
        "g1_security_master_candidates",
        "g1_lineage",
        "quotation_manifest",
        "quotation_request_plan",
        "quotation_file_status",
        "quotation_file_quality",
        "prior_producer_manifest",
        "prior_validation_manifest",
    }
    missing = sorted(required_ids - set(by_id))
    for input_id in missing:
        independent._issue(issues, "required_input_missing", input_id, "")

    csv_ids = (
        "report_event_admission",
        "mapping_audit",
        "g1_reference_symbol_map",
        "g1_security_master_candidates",
        "quotation_request_plan",
        "quotation_file_status",
        "quotation_file_quality",
    )
    for input_id in csv_ids:
        context[input_id] = independent._csv(
            by_id.get(input_id, Path("")), issues, input_id
        )
    try:
        context["halt_events"] = pd.read_parquet(by_id["halt_events"]).fillna("")
    except Exception as exc:
        independent._issue(
            issues, "input_parquet_read_failed", "halt_events", type(exc).__name__
        )
        context["halt_events"] = pd.DataFrame()

    manifests = {
        key: _read_manifest(by_id.get(key), issues, key)
        for key in (
            "v8_manifest",
            "quotation_manifest",
            "prior_producer_manifest",
            "prior_validation_manifest",
        )
    }
    if (
        manifests["v8_manifest"].get("run_id") != V8_RUN_ID
        or manifests["v8_manifest"].get("g1_lineage_admitted") is not True
    ):
        independent._issue(issues, "mixed_v8_manifest", "v8_manifest", "")
    quotation = manifests["quotation_manifest"]
    if (
        quotation.get("run_id") != QUOTATION_RUN_ID
        or quotation.get("official_jpx_source") is not True
        or quotation.get("completed_files") != quotation.get("planned_files")
    ):
        independent._issue(
            issues, "mixed_quotation_manifest", "quotation_manifest", ""
        )
    if manifests["prior_producer_manifest"].get("run_id") != PRIOR_RUN_ID:
        independent._issue(issues, "mixed_prior_manifest", "prior_producer_manifest", "")
    prior_validation = manifests["prior_validation_manifest"]
    if (
        prior_validation.get("run_id") != PRIOR_VALIDATION_RUN_ID
        or prior_validation.get("status") != "pass"
    ):
        independent._issue(
            issues, "mixed_prior_validation", "prior_validation_manifest", ""
        )
    if independent._sha(by_id["v8_manifest"]) != producer_manifest.get(
        "source_v8_manifest_sha256"
    ):
        independent._issue(
            issues, "producer_manifest_lineage_mismatch", "v8_manifest", ""
        )
    if independent._sha(by_id["quotation_manifest"]) != producer_manifest.get(
        "quotation_manifest_sha256"
    ):
        independent._issue(
            issues, "producer_manifest_lineage_mismatch", "quotation_manifest", ""
        )

    context["quotation_manifest"] = quotation
    context["quotation_root"] = by_id["quotation_manifest"].parent.resolve()
    _rehash_external_g1_lineage(by_id.get("g1_lineage"), rows, issues)
    return context, rows


def _validate_denominator(
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    completion: pd.DataFrame,
    declared: pd.DataFrame,
    context: dict[str, Any],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    admission = context.get("report_event_admission", pd.DataFrame())
    mapping = context.get("mapping_audit", pd.DataFrame())
    events = context.get("halt_events", pd.DataFrame())
    mask = (
        admission.get("channel", pd.Series(dtype=str)).eq(
            "halt_suspension_resumption"
        )
        & admission.get("instrument_class", pd.Series(dtype=str)).eq("equity")
        & admission.get("normalization_status", pd.Series(dtype=str)).eq("normalized")
        & admission.get("in_research_scope", pd.Series(dtype=str)).map(independent._bool)
        & ~admission.get("mapping_complete", pd.Series(dtype=str)).map(independent._bool)
        & admission.get("primary_event_id", pd.Series(dtype=str)).ne("")
    )
    expected_ids = set(admission.loc[mask, "primary_event_id"])
    decision_ids = set(decisions.get("event_id", pd.Series(dtype=str)))
    declared_ids = set(declared.get("event_id", pd.Series(dtype=str)))
    event_ids = set(
        events.loc[
            events.get("event_id", pd.Series(dtype=str)).isin(expected_ids), "event_id"
        ]
    )
    mapping_rows = mapping.loc[
        mapping.get("event_id", pd.Series(dtype=str)).isin(expected_ids)
    ]
    checks = [
        ("exact_v8_denominator", decision_ids == expected_ids, f"expected={len(expected_ids)},observed={len(decision_ids)}"),
        ("declared_denominator_exact", declared_ids == expected_ids, str(len(declared_ids))),
        ("event_table_exact", event_ids == expected_ids, str(len(event_ids))),
        ("mapping_rows_exact", set(mapping_rows.get("event_id", pd.Series(dtype=str))) == expected_ids, str(len(mapping_rows))),
        ("mapping_status_missing", mapping_rows.get("mapping_status", pd.Series(dtype=str)).eq("missing_admitted_mapping").all(), "missing_admitted_mapping"),
        ("mapping_ric_blank", mapping_rows.get("refinitiv_ric", pd.Series(dtype=str)).eq("").all(), "all blank"),
        ("decision_ids_unique", decisions.get("event_id", pd.Series(dtype=str)).is_unique, str(len(decisions))),
        ("accepted_partition", set(accepted.get("event_id", pd.Series(dtype=str))) == set(decisions.loc[decisions.get("decision", pd.Series(dtype=str)).eq("accepted"), "event_id"]), str(len(accepted))),
        ("rejected_partition", set(rejected.get("event_id", pd.Series(dtype=str))) == set(decisions.loc[decisions.get("decision", pd.Series(dtype=str)).eq("rejected"), "event_id"]), str(len(rejected))),
        ("completion_exact_rejections", set(completion.get("event_id", pd.Series(dtype=str))) == set(rejected.get("event_id", pd.Series(dtype=str))), str(len(completion))),
    ]
    result = [
        {"check": check, "passed": bool(passed), "detail": detail}
        for check, passed, detail in checks
    ]
    for row in result:
        if not row["passed"]:
            independent._issue(
                issues,
                "denominator_reconciliation_failed",
                str(row["check"]),
                str(row["detail"]),
            )
    return result


def _rehash_external_g1_lineage(
    lineage_path: Path | None,
    rows: list[dict[str, Any]],
    issues: list[dict[str, str]],
) -> None:
    if lineage_path is None or not lineage_path.is_file():
        independent._issue(issues, "g1_lineage_missing", "g1_lineage", "")
        return
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    external = [
        (
            "g1_producer_manifest",
            lineage.get("g1_producer_manifest"),
            lineage.get("g1_producer_manifest_sha256"),
        ),
        (
            "g1_validation_manifest",
            lineage.get("g1_validation_manifest"),
            lineage.get("g1_validation_manifest_sha256"),
        ),
        (
            "reference_symbol_map",
            lineage.get("reference_symbol_map"),
            lineage.get("reference_symbol_map_sha256"),
        ),
    ]
    for source in lineage.get("security_master_sources", []):
        key = str(source.get("source_key", "security_source"))
        external.append(
            (f"{key}:manifest", source.get("manifest_path"), source.get("manifest_sha256"))
        )
        external.append(
            (
                f"{key}:data",
                source.get("security_master_path"),
                source.get("security_master_sha256"),
            )
        )
    for lineage_id, raw_path, expected in external:
        path = Path(str(raw_path)).resolve()
        passed = path.is_file() and independent._sha(path) == expected
        rows.append({"lineage_id": lineage_id, "path": str(path), "passed": passed})
        if not passed:
            independent._issue(
                issues, "external_g1_lineage_hash_mismatch", lineage_id, str(path)
            )


def _read_manifest(
    path: Path | None, issues: list[dict[str, str]], scope: str
) -> dict[str, Any]:
    if path is None:
        independent._issue(issues, "manifest_missing", scope, "")
        return {}
    return independent._json(path, issues, scope)
