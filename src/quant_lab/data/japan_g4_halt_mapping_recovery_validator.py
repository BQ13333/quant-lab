from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pypdf import PdfReader


UTC = timezone.utc
TARGET_CHANNEL = "halt_suspension_resumption"
V5_RUN_ID = "JPN-G4-HALTS-200607-202606-V5"
QUOTATION_RUN_ID = "JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2"
_PLAIN_CODE_LINE = re.compile(r"(?m)^(?P<code>[1-9][0-9A-Z]{3})\u3000")
_FOREIGN = ("<foreign>", "外国株", "外国", "1外", "2外")

PlainPageExtractor = Callable[[Path], list[str]]


@dataclass(frozen=True)
class JapanG4HaltMappingRecoveryValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def independently_reparse_plain_pages(
    pages: list[str], *, observation_month: str, code: str
) -> dict[str, Any]:
    """Reparse a code with sequential PDF text, separate from layout extraction."""
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
            english = next(
                (line for line in lines[1:8] if re.search(r"[A-Za-z]{3}", line)),
                "",
            )
            market = next(
                (
                    line
                    for line in lines[1:12]
                    if re.match(r"^(?:JQG|JQS|TPM|Mothers|Prime|Standard|Growth|PRM|STD|GRT|[12M])(?:\s|外)", line)
                    and (re.search(r"\d", line) or " - " in line)
                ),
                "",
            )
            folded = chunk.casefold()
            foreign = any(marker in folded for marker in _FOREIGN)
            records.append(
                {
                    "page_number": page_number,
                    "chunk": chunk,
                    "english_name": english,
                    "explicit_ordinary": "普通" in code_line,
                    "domestic_continuity": bool(english and market and not foreign),
                    "foreign_marker": foreign,
                }
            )
    return {
        "record_count": len(records),
        "records": records,
        "unique_record": records[0] if len(records) == 1 else {},
    }


def validate_japan_g4_halt_mapping_recovery(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_mapping_recovery_validation",
    page_extractor: PlainPageExtractor | None = None,
) -> JapanG4HaltMappingRecoveryValidationArtifacts:
    _safe_run_id(run_id)
    producer = Path(producer_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    if root.exists():
        raise FileExistsError(f"write-once validation artifact already exists: {root}")
    root.mkdir(parents=True)
    issues: list[dict[str, str]] = []

    producer_manifest = _json(producer / "manifest.json", issues, "producer_manifest")
    declared_hashes = _json(
        producer / "artifact_hashes.json", issues, "producer_artifact_hashes"
    )
    producer_hash_rows = _validate_producer_hashes(
        producer, producer_manifest, declared_hashes, issues
    )
    decisions = _csv(producer / "mapping_decisions.csv", issues, "mapping_decisions")
    accepted = _csv(producer / "accepted_mappings.csv", issues, "accepted_mappings")
    rejected = _csv(producer / "rejected_mappings.csv", issues, "rejected_mappings")
    completion = _csv(producer / "completion_plan.csv", issues, "completion_plan")
    inputs = _csv(producer / "input_inventory.csv", issues, "input_inventory")
    sources = _csv(producer / "source_observations.csv", issues, "source_observations")

    lineage_rows = _validate_lineage(inputs, producer_manifest, decisions, issues)
    context = _load_original_inputs(inputs, issues)
    denominator_rows = _validate_denominator(
        decisions, accepted, rejected, completion, context, issues
    )
    source_rows = _validate_source_inventory(sources, context, issues)
    claim_rows = _validate_claims(
        decisions,
        context,
        page_extractor or _extract_plain_pages,
        issues,
    )

    _write_csv(root / "producer_hash_validation.csv", pd.DataFrame(producer_hash_rows))
    _write_csv(root / "lineage_validation.csv", pd.DataFrame(lineage_rows))
    _write_csv(root / "denominator_validation.csv", pd.DataFrame(denominator_rows))
    _write_csv(root / "source_validation.csv", pd.DataFrame(source_rows))
    claim_frame = pd.DataFrame(claim_rows)
    _write_csv(root / "claim_validation.csv", claim_frame)
    issue_frame = pd.DataFrame(issues, columns=["issue_code", "scope", "detail"])
    _write_csv(root / "validation_issues.csv", issue_frame)

    accepted_rows = int(decisions.get("decision", pd.Series(dtype=str)).eq("accepted").sum())
    accepted_valid = int(
        claim_frame.get("claim_valid", pd.Series(dtype=bool)).fillna(False).sum()
    )
    pass_status = bool(not issues and accepted_rows == accepted_valid)
    hashes = _artifact_hashes(root, {"artifact_hashes.json", "manifest.json"})
    hashes_path = root / "artifact_hashes.json"
    _write_json(hashes_path, hashes)
    manifest = {
        "run_id": run_id,
        "artifact_type": "japan_g4_halt_mapping_recovery_independent_validation",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "pass" if pass_status else "failed_validation",
        "independent_validation": True,
        "independent_parser": "pypdf_sequential_text_chunk_reparse_v1",
        "producer_dir": str(producer),
        "producer_manifest_sha256": _sha(producer / "manifest.json"),
        "denominator_rows": int(len(decisions)),
        "accepted_rows": accepted_rows,
        "accepted_rows_validated": accepted_valid,
        "rejected_rows": int(len(decisions) - accepted_rows),
        "source_files_rehashed": int(len(source_rows)),
        "validation_issue_count": int(len(issues)),
        "validation_issue_codes": sorted({row["issue_code"] for row in issues}),
        "all_lineage_hashes_verified": bool(
            lineage_rows and all(bool(row["passed"]) for row in lineage_rows)
        ),
        "known_at_recomputed": True,
        "candidate_uniqueness_recomputed": True,
        "unsafe_inventory_paths_rejected": True,
        "mixed_manifests_rejected": True,
        "refinitiv_calls": False,
        "controller_touched": False,
        "scorecard_touched": False,
        "component_validation_pass": pass_status,
        "scorecard_eligible": False,
        "promotion_recommended": pass_status,
        "write_once": True,
        "artifact_hashes_sha256": _sha(hashes_path),
        "outputs": hashes,
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return JapanG4HaltMappingRecoveryValidationArtifacts(
        run_id, root, manifest_path, manifest
    )


def verify_validation_artifact(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir)
    problems: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        inventory_path = root / "artifact_hashes.json"
        if _sha(inventory_path) != manifest.get("artifact_hashes_sha256"):
            problems.append("artifact_hash_inventory_mismatch")
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        for relative, expected in inventory.items():
            path = root / relative
            if not path.is_file() or _sha(path) != expected:
                problems.append(f"artifact_hash_mismatch:{relative}")
    except Exception as exc:
        problems.append(f"artifact_parse_error:{type(exc).__name__}")
    return problems


def _validate_claims(
    decisions: pd.DataFrame,
    context: dict[str, Any],
    extract_pages: PlainPageExtractor,
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
        ric, candidate_count, stable = _independent_unique_candidate(
            code, security, symbol_map
        )
        source_request = str(decision.get("source_request_id", ""))
        latest = _latest_eligible_request(code, event_date, plan, status)
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
            path_safe = isinstance(quotation_root, Path) and _within(raw, quotation_root)
            expected_raw_hash = str(decision.get("source_pdf_sha256", ""))
            source_hash_ok = raw.is_file() and _sha(raw) == expected_raw_hash
            known_at = pd.Timestamp(str(decision.get("source_known_at_utc", "")))
            known_at_ok = known_at <= event_date
            record_text = str(decision.get("source_record_text", ""))
            record_hash_ok = _sha_text(record_text) == decision.get("source_record_sha256")
            if source_hash_ok:
                pages_cache.setdefault(raw, extract_pages(raw))
                reparsed = independently_reparse_plain_pages(
                    pages_cache[raw],
                    observation_month=str(decision.get("source_month", "")),
                    code=code,
                )
                reparse_count = int(reparsed["record_count"])
                if reparse_count == 1:
                    record = reparsed["unique_record"]
                    reparse_domestic = bool(record["domestic_continuity"])
                    reparse_foreign = bool(record["foreign_marker"])
                    name_match = _simple_name_match(
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
            _issue(
                issues,
                "accepted_claim_not_independently_proven",
                event_id,
                f"candidate={candidate_count},latest={latest_match},records={reparse_count}",
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


def _validate_lineage(
    inputs: pd.DataFrame,
    producer_manifest: dict[str, Any],
    decisions: pd.DataFrame,
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in inputs.to_dict("records"):
        path = Path(str(item.get("path", "")))
        observed = _sha(path) if path.is_file() else ""
        passed = bool(
            observed
            and observed == item.get("observed_sha256")
            and observed == item.get("expected_sha256")
        )
        rows.append(
            {"lineage_id": item.get("input_id", ""), "path": str(path), "passed": passed}
        )
        if not passed:
            _issue(issues, "input_lineage_hash_mismatch", str(item.get("input_id", "")), observed)
    by_id = {str(row["input_id"]): row for row in inputs.to_dict("records")}
    for key, manifest_key in (
        ("v5_manifest", "source_v5_manifest_sha256"),
        ("quotation_manifest", "quotation_manifest_sha256"),
    ):
        item = by_id.get(key, {})
        passed = item.get("observed_sha256") == producer_manifest.get(manifest_key)
        rows.append({"lineage_id": f"producer_{key}", "path": item.get("path", ""), "passed": passed})
        if not passed:
            _issue(issues, "producer_manifest_lineage_mismatch", key, "")

    lineage_item = by_id.get("g1_lineage", {})
    lineage_path = Path(str(lineage_item.get("path", "")))
    if lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
        external = [
            ("g1_producer_manifest", lineage.get("g1_producer_manifest"), lineage.get("g1_producer_manifest_sha256")),
            ("g1_validation_manifest", lineage.get("g1_validation_manifest"), lineage.get("g1_validation_manifest_sha256")),
            ("reference_symbol_map", lineage.get("reference_symbol_map"), lineage.get("reference_symbol_map_sha256")),
        ]
        for source in lineage.get("security_master_sources", []):
            key = str(source.get("source_key", "security_source"))
            external.append((f"{key}:manifest", source.get("manifest_path"), source.get("manifest_sha256")))
            external.append((f"{key}:data", source.get("security_master_path"), source.get("security_master_sha256")))
        for lineage_id, raw_path, expected in external:
            path = Path(str(raw_path))
            passed = path.is_file() and _sha(path) == expected
            rows.append({"lineage_id": lineage_id, "path": str(path), "passed": passed})
            if not passed:
                _issue(issues, "external_g1_lineage_hash_mismatch", lineage_id, str(path))

    for path_value, expected in decisions.loc[
        decisions.get("identity_source_path", pd.Series(dtype=str)).ne(""),
        ["identity_source_path", "identity_source_sha256"],
    ].drop_duplicates().itertuples(index=False, name=None):
        path = Path(str(path_value))
        passed = path.is_file() and _sha(path) == expected
        rows.append({"lineage_id": "decision_identity_source", "path": str(path), "passed": passed})
        if not passed:
            _issue(issues, "decision_identity_source_hash_mismatch", str(path), "")
    return rows


def _load_original_inputs(
    inputs: pd.DataFrame, issues: list[dict[str, str]]
) -> dict[str, Any]:
    context: dict[str, Any] = {}
    paths = {str(row["input_id"]): Path(str(row["path"])) for row in inputs.to_dict("records")}
    csv_ids = (
        "report_event_admission",
        "g1_reference_symbol_map",
        "g1_security_master_candidates",
        "quotation_request_plan",
        "quotation_file_status",
        "quotation_file_quality",
    )
    for input_id in csv_ids:
        context[input_id] = _csv(paths.get(input_id, Path("")), issues, input_id)
    halt_path = paths.get("halt_events", Path(""))
    try:
        context["halt_events"] = pd.read_parquet(halt_path).fillna("")
    except Exception as exc:
        _issue(issues, "input_parquet_read_failed", "halt_events", type(exc).__name__)
        context["halt_events"] = pd.DataFrame()
    quotation_manifest_path = paths.get("quotation_manifest", Path(""))
    if quotation_manifest_path.is_file():
        manifest = json.loads(quotation_manifest_path.read_text(encoding="utf-8"))
        context["quotation_manifest"] = manifest
        context["quotation_root"] = quotation_manifest_path.parent.resolve()
        if manifest.get("run_id") != QUOTATION_RUN_ID:
            _issue(issues, "mixed_quotation_manifest", str(quotation_manifest_path), str(manifest.get("run_id")))
    v5_manifest_path = paths.get("v5_manifest", Path(""))
    if v5_manifest_path.is_file():
        manifest = json.loads(v5_manifest_path.read_text(encoding="utf-8"))
        context["v5_manifest"] = manifest
        if manifest.get("run_id") != V5_RUN_ID:
            _issue(issues, "mixed_v5_manifest", str(v5_manifest_path), str(manifest.get("run_id")))
    return context


def _validate_denominator(
    decisions: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    completion: pd.DataFrame,
    context: dict[str, Any],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    admission = context.get("report_event_admission", pd.DataFrame())
    if admission.empty:
        expected_ids: set[str] = set()
    else:
        mask = (
            admission["channel"].eq(TARGET_CHANNEL)
            & admission["instrument_class"].eq("equity")
            & admission["normalization_status"].eq("normalized")
            & admission["in_research_scope"].map(_bool)
            & ~admission["mapping_complete"].map(_bool)
            & admission["primary_event_id"].ne("")
        )
        expected_ids = set(admission.loc[mask, "primary_event_id"])
    observed_ids = set(decisions.get("event_id", pd.Series(dtype=str)))
    checks = [
        {"check": "exact_v5_denominator", "passed": observed_ids == expected_ids, "detail": f"expected={len(expected_ids)},observed={len(observed_ids)}"},
        {"check": "decision_ids_unique", "passed": decisions.get("event_id", pd.Series(dtype=str)).is_unique, "detail": str(len(decisions))},
        {"check": "accepted_partition", "passed": set(accepted.get("event_id", pd.Series(dtype=str))) == set(decisions.loc[decisions.get("decision", pd.Series(dtype=str)).eq("accepted"), "event_id"]), "detail": str(len(accepted))},
        {"check": "rejected_partition", "passed": set(rejected.get("event_id", pd.Series(dtype=str))) == set(decisions.loc[decisions.get("decision", pd.Series(dtype=str)).eq("rejected"), "event_id"]), "detail": str(len(rejected))},
        {"check": "completion_exact_rejections", "passed": set(completion.get("event_id", pd.Series(dtype=str))) == set(rejected.get("event_id", pd.Series(dtype=str))), "detail": str(len(completion))},
    ]
    for check in checks:
        if not check["passed"]:
            _issue(issues, "denominator_reconciliation_failed", check["check"], check["detail"])
    return checks


def _validate_source_inventory(
    sources: pd.DataFrame, context: dict[str, Any], issues: list[dict[str, str]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    root = context.get("quotation_root")
    plan = context.get("quotation_request_plan", pd.DataFrame())
    status = context.get("quotation_file_status", pd.DataFrame())
    if plan.empty or status.empty:
        return rows
    joined = plan.merge(status, on="request_id", how="inner", validate="one_to_one").set_index("request_id")
    for source in sources.to_dict("records"):
        request_id = str(source.get("source_request_id", ""))
        raw = Path(str(source.get("source_pdf_path", ""))).resolve()
        normalized = Path(str(source.get("source_normalized_path", ""))).resolve()
        in_inventory = request_id in joined.index
        row = joined.loc[request_id] if in_inventory else pd.Series(dtype=str)
        safe = isinstance(root, Path) and _within(raw, root) and _within(normalized, root)
        raw_hash = raw.is_file() and _sha(raw) == source.get("source_pdf_sha256") == row.get("sha256")
        normalized_hash = normalized.is_file() and _sha(normalized) == source.get("source_normalized_sha256") == row.get("normalized_sha256")
        exact = bool(
            in_inventory
            and row.get("observation_month") == source.get("source_month")
            and row.get("known_at_utc") == source.get("source_known_at_utc")
            and row.get("source_url") == source.get("source_url")
        )
        passed = safe and raw_hash and normalized_hash and exact
        rows.append({"request_id": request_id, "path_safe": safe, "raw_hash_match": raw_hash, "normalized_hash_match": normalized_hash, "inventory_exact": exact, "passed": passed})
        if not passed:
            _issue(issues, "source_inventory_validation_failed", request_id, str(raw))
    return rows


def _independent_unique_candidate(
    code: str, security: pd.DataFrame, symbol_map: pd.DataFrame
) -> tuple[str, int, bool]:
    if security.empty or symbol_map.empty:
        return "", 0, False
    rows = security.loc[security["local_code"].eq(code)].copy()
    rows["resolved_ric"] = rows["candidate_ric"].where(rows["candidate_ric"].ne(""), rows["refinitiv_ric"])
    rows["resolved_ric"] = rows["resolved_ric"].where(rows["resolved_ric"].ne(""), rows["ric"])
    pattern = re.compile(rf"^{re.escape(code)}\.T(?:\^[A-Z0-9]+)?$", re.I)
    rows = rows.loc[
        rows["resolved_ric"].map(lambda value: bool(pattern.fullmatch(str(value))))
        & rows["exchange_code"].eq("TYO")
        & rows["exchange_name"].eq("Tokyo Stock Exchange")
        & rows["rcs_asset_category_leaf"].eq("Ordinary Share")
        & rows["is_primary_issue_ric"].map(_bool)
    ]
    rics = sorted(rows["resolved_ric"].unique())
    if len(rics) != 1:
        return "", len(rics), False
    ric = rics[0]
    rows = rows.loc[rows["resolved_ric"].eq(ric)]
    isins = sorted(value for value in rows["isin"].unique() if value)
    permids = sorted(value for value in rows["instrument_permid"].unique() if value)
    source_hashes = sorted(value for value in rows["admitted_source_sha256"].unique() if value)
    map_rows = symbol_map.loc[
        symbol_map["local_code"].eq(code)
        & symbol_map["refinitiv_ric"].eq(ric)
        & symbol_map["exchange_code"].eq("TYO")
    ]
    stable = bool(
        len(isins) == len(permids) == len(source_hashes) == 1
        and set(map_rows["stable_instrument_id"]) == {f"isin:{isins[0]}"}
    )
    return ric, 1, stable


def _latest_eligible_request(
    code: str, event_date: pd.Timestamp, plan: pd.DataFrame, status: pd.DataFrame
) -> str:
    if plan.empty or status.empty:
        return ""
    joined = plan.merge(status[["request_id", "status"]], on="request_id", validate="one_to_one")
    eligible = joined.loc[
        joined["status"].eq("completed")
        & joined["code_band"].astype(str).isin([code[0], "all"])
        & pd.to_datetime(joined["known_at_utc"], utc=True).le(event_date)
    ].sort_values(["observation_month", "request_id"])
    return str(eligible.iloc[-1]["request_id"]) if not eligible.empty else ""


def _simple_name_match(event: str, source: str, candidate: str) -> bool:
    def tokens(value: str) -> set[str]:
        stop = {"co", "corp", "corporation", "company", "inc", "ltd", "limited", "holdings", "holding", "group", "ordinary", "share", "ord", "shs"}
        return {token for token in re.findall(r"[a-z0-9]+", value.casefold()) if len(token) >= 3 and token not in stop}
    event_tokens = tokens(event)
    source_tokens = tokens(source)
    return bool(event_tokens & source_tokens)


def _validate_producer_hashes(
    producer: Path,
    manifest: dict[str, Any],
    declared: dict[str, str],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inventory_path = producer / "artifact_hashes.json"
    inventory_ok = inventory_path.is_file() and _sha(inventory_path) == manifest.get("artifact_hashes_sha256")
    rows.append({"path": "artifact_hashes.json", "passed": inventory_ok})
    if not inventory_ok:
        _issue(issues, "producer_hash_inventory_mismatch", str(inventory_path), "")
    for relative, expected in sorted(declared.items()):
        path = producer / relative
        passed = path.is_file() and _sha(path) == expected
        rows.append({"path": relative, "passed": passed})
        if not passed:
            _issue(issues, "producer_output_hash_mismatch", relative, "")
    actual = {str(path.relative_to(producer)) for path in producer.rglob("*") if path.is_file()}
    expected_files = set(declared) | {"manifest.json", "artifact_hashes.json"}
    for relative in sorted(actual - expected_files):
        _issue(issues, "undeclared_producer_output", relative, "")
    for relative in sorted(expected_files - actual):
        _issue(issues, "missing_producer_output", relative, "")
    return rows


def _extract_plain_pages(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(path).pages]


def _json(path: Path, issues: list[dict[str, str]], scope: str) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _issue(issues, "json_read_failed", scope, type(exc).__name__)
        return {}


def _csv(path: Path, issues: list[dict[str, str]], scope: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    except Exception as exc:
        _issue(issues, "csv_read_failed", scope, type(exc).__name__)
        return pd.DataFrame()


def _issue(
    issues: list[dict[str, str]], code: str, scope: str, detail: str
) -> None:
    issues.append({"issue_code": code, "scope": scope, "detail": detail})


def _bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _safe_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a path-safe name")


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }
