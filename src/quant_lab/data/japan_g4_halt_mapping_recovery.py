from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from pypdf import PdfReader


UTC = timezone.utc
TARGET_CHANNEL = "halt_suspension_resumption"
TARGET_INSTRUMENT_CLASS = "equity"
JPX_RUN_ID = "JPN-JPX-MONTHLY-QUOTATIONS-200712-202606-V2"
V5_RUN_ID = "JPN-G4-HALTS-200607-202606-V5"
KNOWN_AT_POLICY = (
    "official_7th_business_day_schedule_lagged_to_21st_calendar_day_00_jst"
)
_CODE_LINE = re.compile(r"^\s*(?P<code>[1-9][0-9A-Z]{3})(?:\s|\u3000)")
_EXACT_TOKYO_RIC = r"^{code}\.T(?:\^[A-Z0-9]+)?$"
_FOREIGN_MARKERS = ("<foreign>", "外国株", "外国", "外貨", "1外", "2外")
_NON_EQUITY_MARKERS = (
    "etf",
    "exchange traded fund",
    "investment trust",
    "investment certificate",
    "preferred share",
    "preference share",
    "subscription right",
    "warrant",
    "bond",
)

DECISION_COLUMNS = (
    "event_id",
    "report_event_id",
    "event_date",
    "local_code",
    "event_issue_name",
    "decision",
    "decision_reason",
    "candidate_ric",
    "candidate_count",
    "candidate_document_title",
    "candidate_security_record_id",
    "candidate_isin",
    "candidate_instrument_permid",
    "candidate_retire_date",
    "identity_stable_instrument_id",
    "identity_map_record_count",
    "identity_source_path",
    "identity_source_sha256",
    "identity_source_record_hash",
    "identity_map_path",
    "identity_map_sha256",
    "identity_lineage_path",
    "identity_lineage_sha256",
    "source_month",
    "source_known_at_utc",
    "source_request_id",
    "source_url",
    "source_pdf_path",
    "source_pdf_sha256",
    "source_normalized_path",
    "source_normalized_sha256",
    "source_page_number",
    "source_record_text",
    "source_record_sha256",
    "source_issue_name_japanese",
    "source_issue_name_english",
    "ordinary_share_evidence",
    "continuity_evidence",
    "source_parser",
)


@dataclass(frozen=True)
class JapanG4HaltMappingRecoveryArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    accepted_path: Path
    rejected_path: Path
    completion_plan_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class QuotationRecord:
    code: str
    page_number: int
    record_text: str
    record_sha256: str
    issue_name_japanese: str
    issue_name_english: str
    explicit_ordinary_marker: bool
    domestic_market_evidence: bool
    foreign_marker: bool
    continuity_evidence: str


PageExtractor = Callable[[Path], list[str]]


def parse_layout_quotation_pages(
    pages: Iterable[str], *, observation_month: str, target_codes: set[str]
) -> dict[str, list[QuotationRecord]]:
    """Parse contiguous quotation records from pypdf layout text."""
    parsed: dict[str, list[QuotationRecord]] = {code: [] for code in target_codes}
    expected_header = observation_month.replace("-", "/")
    for page_number, text in enumerate(pages, start=1):
        lines = [line.rstrip() for line in text.splitlines()]
        page_text = "\n".join(lines)
        if expected_header not in page_text or "Stock Quotations" not in page_text:
            continue
        starts = [index for index, line in enumerate(lines) if _CODE_LINE.match(line)]
        for position, start in enumerate(starts):
            match = _CODE_LINE.match(lines[start])
            assert match is not None
            code = match.group("code").upper()
            if code not in target_codes:
                continue
            end = starts[position + 1] if position + 1 < len(starts) else len(lines)
            record_lines = _trim_record_lines(lines[start:end])
            record_text = "\n".join(record_lines)
            nonblank = [line.strip() for line in record_lines if line.strip()]
            code_line = nonblank[0] if nonblank else ""
            english_line = _first_english_name_line(nonblank[1:])
            detail_line = _first_market_detail_line(nonblank[1:])
            folded = record_text.casefold()
            foreign = any(marker in folded for marker in _FOREIGN_MARKERS)
            domestic = bool(detail_line) and not foreign
            explicit_ordinary = "普通" in code_line
            continuity = ""
            if english_line and detail_line and domestic:
                continuity = (
                    "same_page_exact_code_then_english_issue_then_domestic_quote_detail"
                )
            parsed[code].append(
                QuotationRecord(
                    code=code,
                    page_number=page_number,
                    record_text=record_text,
                    record_sha256=_sha_text(record_text),
                    issue_name_japanese=_japanese_name(code_line, code),
                    issue_name_english=_english_name(english_line),
                    explicit_ordinary_marker=explicit_ordinary,
                    domestic_market_evidence=domestic,
                    foreign_marker=foreign,
                    continuity_evidence=continuity,
                )
            )
    return parsed


def recover_japan_g4_halt_mappings(
    *,
    run_id: str,
    v5_dir: str | Path,
    quotation_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_mapping_recovery",
    page_extractor: PageExtractor | None = None,
) -> JapanG4HaltMappingRecoveryArtifacts:
    """Recover event-specific mappings without mutating the V5 halt artifact."""
    _require_safe_run_id(run_id)
    v5 = Path(v5_dir).resolve()
    quotation = Path(quotation_dir).resolve()
    root = (Path(artifact_root) / run_id).resolve()
    if root.exists():
        raise FileExistsError(f"write-once recovery artifact already exists: {root}")
    root.mkdir(parents=True)

    v5_manifest_path = v5 / "manifest.json"
    quotation_manifest_path = quotation / "manifest.json"
    v5_manifest = _read_json(v5_manifest_path)
    quotation_manifest = _read_json(quotation_manifest_path)
    _validate_input_manifests(v5_manifest, quotation_manifest)

    paths = {
        "v5_manifest": v5_manifest_path,
        "report_event_admission": v5 / "report_event_admission.csv",
        "halt_events": v5 / "halt_suspension_events.parquet",
        "g1_reference_symbol_map": v5 / "inputs" / "g1_reference_symbol_map.csv",
        "g1_security_master_candidates": v5
        / "inputs"
        / "g1_security_master_candidates.csv",
        "g1_lineage": v5 / "inputs" / "g1_lineage.json",
        "quotation_manifest": quotation_manifest_path,
        "quotation_request_plan": quotation / "request_plan.csv",
        "quotation_request_plan_hash": quotation / "request_plan.sha256",
        "quotation_file_status": quotation / "file_status.csv",
        "quotation_file_quality": quotation / "file_quality.csv",
    }
    input_inventory = _build_input_inventory(paths, v5_manifest, quotation_manifest)
    if not input_inventory["hash_match"].all():
        failed = input_inventory.loc[~input_inventory["hash_match"], "input_id"].tolist()
        raise ValueError(f"input hash verification failed: {failed}")

    admission = pd.read_csv(paths["report_event_admission"], dtype=str).fillna("")
    events = pd.read_parquet(paths["halt_events"]).fillna("")
    symbol_map = pd.read_csv(paths["g1_reference_symbol_map"], dtype=str).fillna("")
    security = pd.read_csv(
        paths["g1_security_master_candidates"], dtype=str
    ).fillna("")
    plan = pd.read_csv(paths["quotation_request_plan"], dtype=str).fillna("")
    status = pd.read_csv(paths["quotation_file_status"], dtype=str).fillna("")
    quality = pd.read_csv(paths["quotation_file_quality"], dtype=str).fillna("")

    scoped = _scoped_admission(admission)
    scoped_events = _join_scoped_events(scoped, events)
    source_index = _source_index(plan, status, quality, quotation)
    extract_pages = page_extractor or _extract_layout_pages
    page_cache: dict[Path, list[str]] = {}
    parse_cache: dict[tuple[Path, str, str], list[QuotationRecord]] = {}
    decisions: list[dict[str, Any]] = []
    for event in scoped_events.to_dict("records"):
        decisions.append(
            _decide_event(
                event,
                symbol_map=symbol_map,
                security=security,
                source_index=source_index,
                extract_pages=extract_pages,
                page_cache=page_cache,
                parse_cache=parse_cache,
                paths=paths,
                input_inventory=input_inventory,
            )
        )
    decision_frame = pd.DataFrame(decisions, columns=DECISION_COLUMNS).sort_values(
        ["event_date", "event_id"]
    )
    accepted = decision_frame.loc[decision_frame["decision"].eq("accepted")].copy()
    rejected = decision_frame.loc[decision_frame["decision"].eq("rejected")].copy()
    completion = _completion_plan(rejected)
    source_inventory = _source_inventory(decision_frame)
    checks = _producer_checks(scoped, scoped_events, decision_frame, source_inventory)

    all_path = root / "mapping_decisions.csv"
    accepted_path = root / "accepted_mappings.csv"
    rejected_path = root / "rejected_mappings.csv"
    completion_path = root / "completion_plan.csv"
    source_path = root / "source_observations.csv"
    input_path = root / "input_inventory.csv"
    checks_path = root / "producer_checks.csv"
    _write_csv(all_path, decision_frame)
    _write_csv(accepted_path, accepted)
    _write_csv(rejected_path, rejected)
    _write_csv(completion_path, completion)
    _write_csv(source_path, source_inventory)
    _write_csv(input_path, input_inventory)
    _write_csv(checks_path, checks)

    artifacts = _artifact_hashes(root, exclude={"artifact_hashes.json", "manifest.json"})
    hashes_path = root / "artifact_hashes.json"
    _write_json(hashes_path, artifacts)
    all_checks_pass = bool(checks["passed"].all())
    manifest = {
        "run_id": run_id,
        "artifact_type": "japan_g4_halt_mapping_recovery_packet",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "complete_candidate" if all_checks_pass else "failed_producer_checks",
        "source_v5_run_id": v5_manifest["run_id"],
        "source_v5_manifest": str(v5_manifest_path),
        "source_v5_manifest_sha256": _sha_path(v5_manifest_path),
        "quotation_run_id": quotation_manifest["run_id"],
        "quotation_manifest": str(quotation_manifest_path),
        "quotation_manifest_sha256": _sha_path(quotation_manifest_path),
        "denominator_rows": int(len(decision_frame)),
        "accepted_rows": int(len(accepted)),
        "rejected_rows": int(len(rejected)),
        "accepted_event_ids_unique": bool(accepted["event_id"].is_unique),
        "fail_closed": True,
        "refinitiv_calls": False,
        "window_extension": False,
        "successor_ric_allowed": False,
        "source_known_at_required_on_or_before_event": True,
        "latest_eligible_official_observation_only": True,
        "independent_validation_required": True,
        "independent_validation_status": "not_run",
        "scorecard_eligible": False,
        "promotion_recommended": False,
        "write_once": True,
        "producer_checks_pass": all_checks_pass,
        "artifact_hashes_sha256": _sha_path(hashes_path),
        "outputs": artifacts,
    }
    manifest_path = root / "manifest.json"
    _write_json(manifest_path, manifest)
    return JapanG4HaltMappingRecoveryArtifacts(
        run_id,
        root,
        manifest_path,
        accepted_path,
        rejected_path,
        completion_path,
        manifest,
    )


def verify_recovery_artifact(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir)
    issues: list[str] = []
    try:
        manifest = _read_json(root / "manifest.json")
        inventory_path = root / "artifact_hashes.json"
        if _sha_path(inventory_path) != manifest.get("artifact_hashes_sha256"):
            issues.append("artifact_hash_inventory_mismatch")
        inventory = _read_json(inventory_path)
        for relative, expected in inventory.items():
            path = root / relative
            if not path.is_file() or _sha_path(path) != expected:
                issues.append(f"artifact_hash_mismatch:{relative}")
    except Exception as exc:
        issues.append(f"artifact_parse_error:{type(exc).__name__}")
    return issues


def _decide_event(
    event: dict[str, Any],
    *,
    symbol_map: pd.DataFrame,
    security: pd.DataFrame,
    source_index: pd.DataFrame,
    extract_pages: PageExtractor,
    page_cache: dict[Path, list[str]],
    parse_cache: dict[tuple[Path, str, str], list[QuotationRecord]],
    paths: dict[str, Path],
    input_inventory: pd.DataFrame,
) -> dict[str, Any]:
    code = str(event["local_code"])
    event_date = pd.Timestamp(str(event["event_date"]), tz="UTC")
    base = {column: "" for column in DECISION_COLUMNS}
    base.update(
        {
            "event_id": event["event_id"],
            "report_event_id": event["report_event_id"],
            "event_date": str(event["event_date"]),
            "local_code": code,
            "event_issue_name": event["issue_name"],
            "identity_map_path": str(paths["g1_reference_symbol_map"]),
            "identity_map_sha256": _inventory_hash(
                input_inventory, "g1_reference_symbol_map"
            ),
            "identity_lineage_path": str(paths["g1_lineage"]),
            "identity_lineage_sha256": _inventory_hash(input_inventory, "g1_lineage"),
            "source_parser": "pypdf_layout_contiguous_record_v1",
        }
    )

    candidate, candidate_reason = _unique_candidate(code, symbol_map, security)
    base["candidate_count"] = candidate.get("candidate_count", 0)
    for key, target in (
        ("candidate_ric", "candidate_ric"),
        ("document_title", "candidate_document_title"),
        ("source_record_id", "candidate_security_record_id"),
        ("isin", "candidate_isin"),
        ("instrument_permid", "candidate_instrument_permid"),
        ("retire_date", "candidate_retire_date"),
        ("stable_instrument_id", "identity_stable_instrument_id"),
        ("identity_map_record_count", "identity_map_record_count"),
        ("admitted_source_path", "identity_source_path"),
        ("admitted_source_sha256", "identity_source_sha256"),
        ("identity_source_record_hash", "identity_source_record_hash"),
    ):
        base[target] = candidate.get(key, "")
    if candidate_reason:
        return _reject(base, candidate_reason)

    eligible = source_index.loc[
        pd.to_datetime(source_index["known_at_utc"], utc=True).le(event_date)
        & source_index["code_band"].isin([code[0], "all"])
    ].sort_values(["observation_month", "request_id"])
    if eligible.empty:
        return _reject(base, "no_official_observation_known_before_event")
    source = eligible.iloc[-1].to_dict()
    _add_source_fields(base, source)
    raw_path = Path(str(source["raw_path"])).resolve()
    cache_key = (raw_path, str(source["observation_month"]), code)
    if cache_key not in parse_cache:
        if raw_path not in page_cache:
            page_cache[raw_path] = extract_pages(raw_path)
        parsed = parse_layout_quotation_pages(
            page_cache[raw_path],
            observation_month=str(source["observation_month"]),
            target_codes={code},
        )
        parse_cache[cache_key] = parsed.get(code, [])
    records = parse_cache[cache_key]
    if len(records) != 1:
        reason = "latest_eligible_source_record_missing" if not records else (
            "latest_eligible_source_record_not_unique"
        )
        return _reject(base, reason)
    record = records[0]
    _add_record_fields(base, record)
    if record.foreign_marker or not str(candidate.get("isin", "")).startswith("JP"):
        return _reject(base, "foreign_share_excluded")
    if not record.domestic_market_evidence:
        return _reject(base, "domestic_market_continuity_not_proven")
    if not record.explicit_ordinary_marker and not record.continuity_evidence:
        return _reject(base, "truncated_ordinary_label_without_continuity")
    if not _identity_name_matches(
        str(event["issue_name"]),
        record.issue_name_japanese,
        record.issue_name_english,
        str(candidate.get("document_title", "")),
    ):
        return _reject(base, "issue_identity_name_mismatch")
    known = pd.Timestamp(str(source["known_at_utc"]))
    if known > event_date:
        return _reject(base, "source_known_at_after_event")
    base["ordinary_share_evidence"] = (
        "explicit_pdf_ordinary_marker_plus_g1_ordinary_share"
        if record.explicit_ordinary_marker
        else "g1_ordinary_share_plus_pdf_contiguous_domestic_quote_record"
    )
    base["decision"] = "accepted"
    base["decision_reason"] = "exact_event_pit_mapping_source_backed"
    return base


def _unique_candidate(
    code: str, symbol_map: pd.DataFrame, security: pd.DataFrame
) -> tuple[dict[str, Any], str]:
    rows = security.loc[security["local_code"].eq(code)].copy()
    if rows.empty:
        return {"candidate_count": 0}, "no_g1_identity_candidate"
    rows["_ric"] = rows["candidate_ric"].mask(
        rows["candidate_ric"].eq(""), rows["refinitiv_ric"]
    ).mask(lambda series: series.eq(""), rows["ric"])
    exact = rows.loc[
        rows["_ric"].str.match(
            _EXACT_TOKYO_RIC.format(code=re.escape(code)), case=False, na=False
        )
        & rows["exchange_code"].eq("TYO")
        & rows["exchange_name"].eq("Tokyo Stock Exchange")
        & rows["rcs_asset_category_leaf"].eq("Ordinary Share")
        & rows["is_primary_issue_ric"].map(_bool_value)
    ].copy()
    exact = exact.loc[
        ~exact["document_title"].str.casefold().apply(
            lambda value: any(marker in value for marker in _NON_EQUITY_MARKERS)
        )
    ]
    rics = sorted(exact["_ric"].drop_duplicates())
    if len(rics) != 1:
        return {"candidate_count": len(rics)}, (
            "no_unique_exact_tokyo_ric_candidate"
            if not rics
            else "multiple_exact_tokyo_ric_candidates"
        )
    ric = rics[0]
    exact = exact.loc[exact["_ric"].eq(ric)]
    isins = sorted(value for value in exact["isin"].unique() if value)
    permids = sorted(value for value in exact["instrument_permid"].unique() if value)
    source_hashes = sorted(
        value for value in exact["admitted_source_sha256"].unique() if value
    )
    source_paths = sorted(
        value for value in exact["admitted_source_path"].unique() if value
    )
    if len(isins) != 1 or len(permids) != 1 or len(source_hashes) != 1 or len(source_paths) != 1:
        return {"candidate_count": 1, "candidate_ric": ric}, (
            "unstable_g1_identity_evidence"
        )
    map_rows = symbol_map.loc[
        symbol_map["local_code"].eq(code)
        & symbol_map["refinitiv_ric"].eq(ric)
        & symbol_map["stable_instrument_id"].eq(f"isin:{isins[0]}")
        & symbol_map["exchange_code"].eq("TYO")
    ]
    stable_ids = sorted(value for value in map_rows["stable_instrument_id"].unique() if value)
    if stable_ids != [f"isin:{isins[0]}"]:
        return {"candidate_count": 1, "candidate_ric": ric}, (
            "g1_map_security_identity_not_stable"
        )
    row = exact.sort_values(["source_record_id"]).iloc[0].to_dict()
    identity_record = "|".join(
        [
            ric,
            code,
            str(row["document_title"]),
            isins[0],
            permids[0],
            str(row["source_record_id"]),
            source_hashes[0],
        ]
    )
    return {
        "candidate_count": 1,
        "candidate_ric": ric,
        "document_title": row["document_title"],
        "source_record_id": row["source_record_id"],
        "isin": isins[0],
        "instrument_permid": permids[0],
        "retire_date": row["retire_date"],
        "stable_instrument_id": stable_ids[0],
        "identity_map_record_count": len(map_rows),
        "admitted_source_path": source_paths[0],
        "admitted_source_sha256": source_hashes[0],
        "identity_source_record_hash": _sha_text(identity_record),
    }, ""


def _source_index(
    plan: pd.DataFrame,
    status: pd.DataFrame,
    quality: pd.DataFrame,
    quotation_root: Path,
) -> pd.DataFrame:
    if not plan["request_id"].is_unique or not status["request_id"].is_unique:
        raise ValueError("quotation request/status inventory is not unique")
    merged = plan.merge(status, on="request_id", how="inner", validate="one_to_one")
    merged = merged.merge(quality, on="request_id", how="inner", validate="one_to_one")
    completed = merged.loc[
        merged["status"].eq("completed")
        & merged["quality_pass"].map(_bool_value)
        & merged["source_official_jpx"].map(_bool_value)
        & merged["known_at_policy"].eq(KNOWN_AT_POLICY)
    ].copy()
    for column in ("raw_path", "normalized_path"):
        if not completed[column].apply(
            lambda value: _is_within(Path(str(value)), quotation_root)
        ).all():
            raise ValueError(f"unsafe or mixed quotation inventory path: {column}")
    raw_hash_ok = completed.apply(
        lambda row: _sha_path(Path(row["raw_path"])) == row["sha256"], axis=1
    )
    normalized_hash_ok = completed.apply(
        lambda row: _sha_path(Path(row["normalized_path"]))
        == row["normalized_sha256"],
        axis=1,
    )
    if not raw_hash_ok.all() or not normalized_hash_ok.all():
        raise ValueError("quotation raw/normalized lineage hash mismatch")
    completed["code_band"] = completed["code_band"].astype(str)
    return completed


def _scoped_admission(admission: pd.DataFrame) -> pd.DataFrame:
    scoped = admission.loc[
        admission["channel"].eq(TARGET_CHANNEL)
        & admission["instrument_class"].eq(TARGET_INSTRUMENT_CLASS)
        & admission["normalization_status"].eq("normalized")
        & admission["in_research_scope"].map(_bool_value)
        & ~admission["mapping_complete"].map(_bool_value)
        & admission["primary_event_id"].ne("")
    ].copy()
    if scoped.empty or not scoped["primary_event_id"].is_unique:
        raise ValueError("bounded unresolved halt denominator is empty or non-unique")
    return scoped


def _join_scoped_events(scoped: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    event_cols = ["event_id", "event_date", "local_code", "issue_name", "event_type"]
    joined = scoped[["report_event_id", "primary_event_id"]].merge(
        events[event_cols],
        left_on="primary_event_id",
        right_on="event_id",
        how="left",
        validate="one_to_one",
    )
    if joined["event_id"].eq("").any() or not joined["event_type"].eq("trading_halt").all():
        raise ValueError("admission/event denominator reconciliation failed")
    return joined


def _build_input_inventory(
    paths: dict[str, Path], v5_manifest: dict[str, Any], quotation_manifest: dict[str, Any]
) -> pd.DataFrame:
    v5_hashes = v5_manifest.get("output_sha256", {})
    expected = {
        "report_event_admission": v5_hashes.get("report_event_admission.csv", ""),
        "halt_events": v5_hashes.get("halt_suspension_events.parquet", ""),
        "g1_reference_symbol_map": v5_hashes.get(
            "inputs/g1_reference_symbol_map.csv", ""
        ),
        "g1_security_master_candidates": v5_hashes.get(
            "inputs/g1_security_master_candidates.csv", ""
        ),
        "g1_lineage": v5_hashes.get("inputs/g1_lineage.json", ""),
        "quotation_request_plan": quotation_manifest.get("request_plan_sha256", ""),
    }
    rows = []
    for input_id, path in paths.items():
        observed = _sha_path(path)
        wanted = expected.get(input_id, observed)
        if input_id == "quotation_request_plan_hash":
            wanted = observed
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


def _validate_input_manifests(
    v5_manifest: dict[str, Any], quotation_manifest: dict[str, Any]
) -> None:
    if (
        v5_manifest.get("run_id") != V5_RUN_ID
        or v5_manifest.get("component") != "halt_suspension_events"
        or v5_manifest.get("g1_lineage_admitted") is not True
    ):
        raise ValueError("V5 halt manifest is not the admitted bounded input")
    if (
        quotation_manifest.get("run_id") != JPX_RUN_ID
        or quotation_manifest.get("official_jpx_source") is not True
        or quotation_manifest.get("status") != "complete_candidate_source"
        or quotation_manifest.get("completed_files")
        != quotation_manifest.get("planned_files")
    ):
        raise ValueError("quotation manifest is not the complete admitted JPX V2 source")


def _source_inventory(decisions: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "source_request_id",
        "source_month",
        "source_known_at_utc",
        "source_url",
        "source_pdf_path",
        "source_pdf_sha256",
        "source_normalized_path",
        "source_normalized_sha256",
    ]
    return (
        decisions.loc[decisions["source_request_id"].ne(""), columns]
        .drop_duplicates()
        .sort_values(["source_month", "source_request_id"])
    )


def _completion_plan(rejected: pd.DataFrame) -> pd.DataFrame:
    if rejected.empty:
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_date",
                "local_code",
                "blocker",
                "required_evidence",
                "next_action",
            ]
        )
    result = rejected[
        ["event_id", "event_date", "local_code", "decision_reason"]
    ].rename(columns={"decision_reason": "blocker"})
    actions = {
        "no_g1_identity_candidate": (
            "admitted exact-code Tokyo ordinary-share identity",
            "acquire and independently admit an exact-code G1 identity candidate",
        ),
        "no_unique_exact_tokyo_ric_candidate": (
            "one exact-code Tokyo ordinary-share RIC",
            "resolve G1 candidate absence without constructing code.T or using a successor",
        ),
        "multiple_exact_tokyo_ric_candidates": (
            "unique exact-code Tokyo ordinary-share RIC",
            "independently disambiguate the competing G1 identities",
        ),
        "g1_map_security_identity_not_stable": (
            "stable ISIN-linked G1 map/security identity",
            "repair and independently admit the G1 identity lineage",
        ),
        "unstable_g1_identity_evidence": (
            "single stable ISIN, PermID, and source lineage",
            "resolve conflicting G1 identity fields and revalidate",
        ),
        "latest_eligible_source_record_missing": (
            "official quotation observation known before the event",
            "locate an admitted pre-event official source; do not extend an older window",
        ),
        "latest_eligible_source_record_not_unique": (
            "one exact source record for the code",
            "resolve duplicate PDF extraction records independently",
        ),
        "foreign_share_excluded": (
            "domestic common-equity evidence",
            "leave excluded unless an official source proves a domestic ordinary share",
        ),
        "domestic_market_continuity_not_proven": (
            "contiguous domestic quotation record",
            "obtain explicit source-backed record continuity",
        ),
        "truncated_ordinary_label_without_continuity": (
            "explicit ordinary-share or contiguous row evidence",
            "obtain explicit source-backed continuity; do not infer the label",
        ),
        "issue_identity_name_mismatch": (
            "matching event, quotation, and G1 identity names",
            "resolve the identity-name conflict with official evidence",
        ),
        "source_known_at_after_event": (
            "official observation known on or before event date",
            "locate an earlier admitted observation; do not backdate known_at",
        ),
        "no_official_observation_known_before_event": (
            "official observation known on or before event date",
            "locate an admitted pre-event official source",
        ),
    }
    result["required_evidence"] = result["blocker"].map(
        lambda value: actions.get(value, ("source-backed evidence", "manual review"))[0]
    )
    result["next_action"] = result["blocker"].map(
        lambda value: actions.get(value, ("source-backed evidence", "manual review"))[1]
    )
    return result


def _producer_checks(
    scoped: pd.DataFrame,
    events: pd.DataFrame,
    decisions: pd.DataFrame,
    sources: pd.DataFrame,
) -> pd.DataFrame:
    accepted = decisions["decision"].eq("accepted")
    checks = [
        ("denominator_exact", len(scoped) == len(events) == len(decisions), str(len(decisions))),
        ("event_ids_unique", decisions["event_id"].is_unique, str(decisions["event_id"].nunique())),
        ("decision_partition", decisions["decision"].isin(["accepted", "rejected"]).all(), "accepted/rejected"),
        ("accepted_exact_ric", decisions.loc[accepted].apply(lambda row: bool(re.match(_EXACT_TOKYO_RIC.format(code=re.escape(str(row["local_code"]))), str(row["candidate_ric"]), re.I)), axis=1).all(), "exact local code .T family"),
        ("accepted_known_at", (pd.to_datetime(decisions.loc[accepted, "source_known_at_utc"], utc=True) <= pd.to_datetime(decisions.loc[accepted, "event_date"], utc=True)).all(), "known_at <= event_date"),
        ("accepted_unique_candidates", decisions.loc[accepted, "candidate_count"].astype(int).eq(1).all(), "candidate_count=1"),
        ("accepted_source_records_hashed", decisions.loc[accepted].apply(lambda row: _sha_text(str(row["source_record_text"])) == row["source_record_sha256"], axis=1).all(), "record text hashes"),
        ("accepted_no_foreign", ~decisions.loc[accepted, "decision_reason"].eq("foreign_share_excluded").any(), "foreign excluded"),
        ("source_inventory_unique", sources["source_request_id"].is_unique, str(len(sources))),
    ]
    return pd.DataFrame(checks, columns=["check", "passed", "detail"])


def _add_source_fields(base: dict[str, Any], source: dict[str, Any]) -> None:
    base.update(
        {
            "source_month": source["observation_month"],
            "source_known_at_utc": source["known_at_utc"],
            "source_request_id": source["request_id"],
            "source_url": source["source_url"],
            "source_pdf_path": source["raw_path"],
            "source_pdf_sha256": source["sha256"],
            "source_normalized_path": source["normalized_path"],
            "source_normalized_sha256": source["normalized_sha256"],
        }
    )


def _add_record_fields(base: dict[str, Any], record: QuotationRecord) -> None:
    base.update(
        {
            "source_page_number": record.page_number,
            "source_record_text": record.record_text,
            "source_record_sha256": record.record_sha256,
            "source_issue_name_japanese": record.issue_name_japanese,
            "source_issue_name_english": record.issue_name_english,
            "continuity_evidence": record.continuity_evidence,
        }
    )


def _reject(base: dict[str, Any], reason: str) -> dict[str, Any]:
    base["decision"] = "rejected"
    base["decision_reason"] = reason
    return base


def _identity_name_matches(
    event_name: str, japanese_name: str, english_name: str, candidate_title: str
) -> bool:
    event = _name_tokens(event_name)
    source = _name_tokens(f"{japanese_name} {english_name}")
    if not source or not event:
        return False
    return bool(
        event & source
        or _normalized_name(english_name) in _normalized_name(event_name)
    )


def _name_tokens(value: str) -> set[str]:
    folded = _normalized_name(value)
    stop = {
        "co", "corp", "corporation", "company", "inc", "ltd", "limited", "holdings",
        "holding", "group", "ordinary", "share", "ord", "shs", "tokyo", "stock", "exchange",
    }
    return {token for token in folded.split() if len(token) >= 3 and token not in stop}


def _normalized_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold().replace("&", " and ")))


def _trim_record_lines(lines: list[str]) -> list[str]:
    result: list[str] = []
    nonblank = 0
    blank_run = 0
    for line in lines:
        if line.strip():
            if nonblank >= 3 and blank_run >= 2:
                break
            blank_run = 0
            nonblank += 1
            result.append(line)
        else:
            blank_run += 1
            result.append(line)
    while result and not result[-1].strip():
        result.pop()
    return result


def _first_english_name_line(lines: list[str]) -> str:
    for line in lines[:8]:
        stripped = line.strip()
        if re.search(r"[A-Za-z]{3}", stripped) and not re.match(r"^(?:JQG|JQS|TPM|Mothers|Prime|Standard|Growth|PRM|STD|GRT|[12M])(?:\s|外)", stripped):
            return line
    return ""


def _first_market_detail_line(lines: list[str]) -> str:
    for line in lines[:10]:
        stripped = line.strip()
        if re.match(r"^(?:JQG|JQS|TPM|Mothers|Prime|Standard|Growth|PRM|STD|GRT|[12M])(?:\s|外)", stripped) and re.search(r"\d|\s-\s", stripped):
            return line
    return ""


def _japanese_name(code_line: str, code: str) -> str:
    value = re.sub(rf"^\s*{re.escape(code)}(?:\s|\u3000)+", "", code_line)
    value = re.split(r"\s{2,}", value, maxsplit=1)[0]
    return re.sub(r"\s*普通(?:株式)?\s*$", "", value).strip()


def _english_name(line: str) -> str:
    return re.split(r"\s{2,}", line.strip(), maxsplit=1)[0].strip()


def _extract_layout_pages(path: Path) -> list[str]:
    return [page.extract_text(extraction_mode="layout") or "" for page in PdfReader(path).pages]


def _inventory_hash(inventory: pd.DataFrame, input_id: str) -> str:
    rows = inventory.loc[inventory["input_id"].eq(input_id), "observed_sha256"]
    return str(rows.iloc[0]) if len(rows) == 1 else ""


def _bool_value(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes"}


def _require_safe_run_id(run_id: str) -> None:
    if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
        raise ValueError("run_id must be a path-safe name")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _artifact_hashes(root: Path, *, exclude: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }
