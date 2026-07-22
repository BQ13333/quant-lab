from __future__ import annotations

import hashlib
import io
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from pypdf import PdfReader


JPX_ROOT = "https://www.jpx.co.jp/"
HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
EXPECTED_DENOMINATOR_ROWS = 366
EXPECTED_HALT_MONTHS = 240
EXPECTED_DELISTING_MONTHS = 126

PageExtractor = Callable[[bytes], list[str]]


@dataclass(frozen=True)
class JapanG4DenominatorValidationArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    manifest: dict[str, Any]


def independently_parse_lifecycle_pages(pages: list[str], expected_month: str) -> dict[str, Any]:
    """Parse an exact-month lifecycle report without using producer parser code."""
    period = pd.Period(expected_month, freq="M")
    lines = [" ".join(line.split()) for page in pages for line in page.splitlines()]
    title_months: set[str] = set()
    title_pattern = re.compile(
        r"Changes\s+in\s+Listed\s+Companies\s+and\s+Issues.*?"
        r"\((?P<name>[A-Za-z]{3,9})\.?\s*(?P<year>20\d{2})\)",
        flags=re.IGNORECASE,
    )
    for line in lines:
        match = title_pattern.search(line)
        if not match:
            continue
        try:
            parsed = pd.to_datetime(f"{match.group('name')} {match.group('year')}")
            title_months.add(parsed.strftime("%Y-%m"))
        except ValueError:
            continue

    sections = {
        DELISTING_CHANNEL: _parse_explicit_section(lines, "Delisting", period),
        HALT_CHANNEL: _parse_explicit_section(lines, "Suspension of Trading", period),
    }
    return {
        "title_months": sorted(title_months),
        "exact_month": title_months == {expected_month},
        "sections": sections,
    }


def validate_japan_g4_halt_denominator_recovery(
    *,
    run_id: str,
    producer_dir: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_denominator_recovery_validation",
    page_extractor: PageExtractor | None = None,
    require_expected_denominator: bool = True,
) -> JapanG4DenominatorValidationArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    producer = Path(producer_dir).resolve()
    root = Path(artifact_root) / run_id
    if root.exists():
        raise FileExistsError(f"write-once validation artifact already exists: {root}")
    root.mkdir(parents=True)
    extract_pages = page_extractor or _extract_pages
    issues: list[dict[str, str]] = []

    producer_manifest = _read_json(producer / "manifest.json", issues, "producer_manifest")
    declared_hashes = _read_json(producer / "artifact_hashes.json", issues, "producer_hash_inventory")
    hash_rows = _validate_producer_hashes(producer, producer_manifest, declared_hashes, issues)

    original, denominator, denominator_checks = _validate_denominator(
        producer,
        producer_manifest,
        require_expected_denominator,
        issues,
    )

    source_inventory = _read_csv(producer / "source_inventory.csv", issues, "source_inventory")
    coverage = _read_csv(producer / "month_channel_coverage.csv", issues, "coverage")
    report_rows, parsed_reports = _validate_reports(
        producer,
        source_inventory,
        extract_pages,
        issues,
    )
    claim_rows = _reconcile_claims(coverage, denominator, parsed_reports, issues)

    pd.DataFrame(hash_rows).to_csv(root / "producer_hash_validation.csv", index=False)
    pd.DataFrame(denominator_checks).to_csv(root / "denominator_validation.csv", index=False)
    pd.DataFrame(report_rows).to_csv(root / "report_validation.csv", index=False)
    claims = pd.DataFrame(claim_rows)
    claims.to_csv(root / "claim_reconciliation.csv", index=False)
    issue_frame = pd.DataFrame(issues, columns=["issue_code", "scope", "detail"])
    issue_frame.to_csv(root / "validation_issues.csv", index=False)

    covered_claims = int(_bools(coverage.get("covered", pd.Series(dtype=str))).sum())
    explicit_proven = int(claims.get("explicit_section_proven", pd.Series(dtype=bool)).sum())
    reconciled = int(claims.get("claim_reconciled", pd.Series(dtype=bool)).sum())
    full_exact_coverage = bool(
        len(coverage) == EXPECTED_DENOMINATOR_ROWS
        and covered_claims == EXPECTED_DENOMINATOR_ROWS
        and explicit_proven == EXPECTED_DENOMINATOR_ROWS
        and reconciled == EXPECTED_DENOMINATOR_ROWS
        and not issues
    )
    status = "failed_validation" if issues else (
        "pass_complete_component" if full_exact_coverage else "pass_partial_component"
    )
    hashes = _artifact_hashes(root, exclude={"manifest.json", "artifact_hashes.json"})
    hashes_path = root / "artifact_hashes.json"
    hashes_path.write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "producer_dir": str(producer),
        "producer_manifest_sha256": _sha_path(producer / "manifest.json") if (producer / "manifest.json").is_file() else "",
        "producer_artifact_hashes_sha256": _sha_path(producer / "artifact_hashes.json") if (producer / "artifact_hashes.json").is_file() else "",
        "source_denominator_rows": int(len(denominator)),
        "original_source_rows": int(len(original)),
        "covered_claims": covered_claims,
        "explicit_section_claims_proven": explicit_proven,
        "fully_reconciled_claims": reconciled,
        "uncovered_month_channels": int(len(coverage) - covered_claims),
        "exact_month_reports_reparsed": int(len(parsed_reports)),
        "validation_issue_count": int(len(issues)),
        "validation_issue_codes": sorted({item["issue_code"] for item in issues}),
        "absence_is_zero": False,
        "refinitiv_used": False,
        "exact_source_coverage_complete": full_exact_coverage,
        "scorecard_eligible": full_exact_coverage,
        "promotion_recommended": full_exact_coverage,
        "event_count_reconciliation_required": True,
        "write_once": True,
        "artifact_hashes_sha256": _sha_path(hashes_path),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return JapanG4DenominatorValidationArtifacts(run_id, root, manifest_path, manifest)


def verify_validation_artifact(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir)
    issues: list[str] = []
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        hashes_path = root / "artifact_hashes.json"
        if _sha_path(hashes_path) != manifest.get("artifact_hashes_sha256"):
            issues.append("hash_inventory_digest_mismatch")
        declared = json.loads(hashes_path.read_text(encoding="utf-8"))
        for relative, digest in declared.items():
            path = root / relative
            if not path.is_file() or _sha_path(path) != digest:
                issues.append(f"artifact_hash_mismatch:{relative}")
    except Exception as exc:
        issues.append(f"validation_artifact_parse_error:{type(exc).__name__}")
    return issues


def _parse_explicit_section(lines: list[str], heading: str, period: pd.Period) -> dict[str, Any]:
    starts = [index for index, line in enumerate(lines) if line.casefold() == heading.casefold()]
    if len(starts) != 1:
        return {
            "explicit": False,
            "heading_count": len(starts),
            "event_rows": 0,
            "explicit_zero": False,
        }
    start = starts[0] + 1
    end = len(lines)
    for index in range(start, len(lines)):
        if _is_next_event_heading(lines[index], heading):
            end = index
            break
    section = lines[start:end]
    date_pattern = re.compile(
        rf"{period.year}\s*[./]\s*0?{period.month}\s*[./]\s*\d{{1,2}}"
    )
    event_rows = sum(len(date_pattern.findall(line)) for line in section)
    explicit_zero = any(line.casefold() in {"none", "nil", "zero", "no events"} for line in section)
    return {
        "explicit": True,
        "heading_count": 1,
        "event_rows": event_rows,
        "explicit_zero": explicit_zero,
    }


def _is_next_event_heading(line: str, current: str) -> bool:
    if not line or line.casefold() == current.casefold():
        return False
    return bool(
        re.match(
            r"^(?:New Listings|Delisting|Suspension of Trading|Assignment to|"
            r"Reassignment|Transfer to|Name Change|Change of|Change in|"
            r"Additional Assignment|Exception from|Designation as|Ex-New\b)",
            line,
            flags=re.IGNORECASE,
        )
        or re.match(r"^\d{1,2}(?:-\d+)?\s+.*(?:\(|（)", line)
    )


def _validate_producer_hashes(
    producer: Path,
    manifest: dict[str, Any],
    declared: dict[str, str],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    inventory_path = producer / "artifact_hashes.json"
    observed_inventory_hash = _sha_path(inventory_path) if inventory_path.is_file() else ""
    expected_inventory_hash = str(manifest.get("artifact_hashes_sha256", ""))
    inventory_match = observed_inventory_hash == expected_inventory_hash and bool(expected_inventory_hash)
    rows.append({"path": "artifact_hashes.json", "expected_sha256": expected_inventory_hash, "observed_sha256": observed_inventory_hash, "hash_match": inventory_match})
    if not inventory_match:
        _issue(issues, "producer_hash_inventory_digest_mismatch", "artifact_hashes.json", observed_inventory_hash)
    for relative, expected in sorted(declared.items()):
        path = producer / relative
        observed = _sha_path(path) if path.is_file() else ""
        match = observed == expected and bool(expected)
        rows.append({"path": relative, "expected_sha256": expected, "observed_sha256": observed, "hash_match": match})
        if not match:
            _issue(issues, "producer_output_hash_mismatch", relative, observed)
    actual = {str(path.relative_to(producer)) for path in producer.rglob("*") if path.is_file()}
    expected_files = set(declared) | {"manifest.json", "artifact_hashes.json"}
    for relative in sorted(actual - expected_files):
        _issue(issues, "undeclared_producer_output", relative, "not present in artifact_hashes.json")
    for relative in sorted(expected_files - actual):
        _issue(issues, "missing_producer_output", relative, "declared/required file missing")
    return rows


def _validate_denominator(
    producer: Path,
    manifest: dict[str, Any],
    require_expected: bool,
    issues: list[dict[str, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    source_path = Path(str(manifest.get("source_input", "")))
    source_hash = _sha_path(source_path) if source_path.is_file() else ""
    source_hash_match = source_hash == manifest.get("source_input_sha256") and bool(source_hash)
    checks.append({"check": "original_source_hash", "passed": source_hash_match, "detail": source_hash})
    if not source_hash_match:
        _issue(issues, "original_source_hash_mismatch", str(source_path), source_hash)
    original_all = _read_csv(source_path, issues, "original_source") if source_path.is_file() else pd.DataFrame()
    if "gap_type" in original_all:
        original = original_all.loc[original_all["gap_type"].eq("source_denominator_uncovered")].copy()
    else:
        original = pd.DataFrame()
    denominator = _read_csv(producer / "input_denominator.csv", issues, "input_denominator")
    expected_rows = EXPECTED_DENOMINATOR_ROWS if require_expected else len(denominator)
    row_count_ok = len(original) == len(denominator) == expected_rows
    checks.append({"check": "denominator_row_count", "passed": row_count_ok, "detail": f"original={len(original)},artifact={len(denominator)},expected={expected_rows}"})
    if not row_count_ok:
        _issue(issues, "denominator_row_count_mismatch", "input_denominator", checks[-1]["detail"])
    identity_columns = [column for column in original.columns if column in denominator.columns]
    exact_rows = bool(identity_columns) and original[identity_columns].reset_index(drop=True).equals(denominator[identity_columns].reset_index(drop=True))
    checks.append({"check": "denominator_exact_rows", "passed": exact_rows, "detail": ",".join(identity_columns)})
    if not exact_rows:
        _issue(issues, "denominator_content_mismatch", "input_denominator", "does not equal original filtered rows")
    gap_ids_ok = "gap_id" in denominator and denominator["gap_id"].nunique() == len(denominator)
    checks.append({"check": "gap_ids_unique", "passed": gap_ids_ok, "detail": str(denominator.get("gap_id", pd.Series(dtype=str)).nunique())})
    if not gap_ids_ok:
        _issue(issues, "denominator_gap_ids_not_unique", "input_denominator", "gap_id uniqueness failed")
    if require_expected and {"scope", "month"}.issubset(denominator.columns):
        halt = denominator.loc[denominator["scope"].eq(HALT_CHANNEL), "month"]
        delist = denominator.loc[denominator["scope"].eq(DELISTING_CHANNEL), "month"]
        halt_ok = set(halt) == {str(p) for p in pd.period_range("2006-07", "2026-06", freq="M")} and len(halt) == EXPECTED_HALT_MONTHS
        delist_ok = set(delist) == {str(p) for p in pd.period_range("2006-07", "2016-12", freq="M")} and len(delist) == EXPECTED_DELISTING_MONTHS
        checks.extend([
            {"check": "halt_month_shape", "passed": halt_ok, "detail": str(len(halt))},
            {"check": "delisting_month_shape", "passed": delist_ok, "detail": str(len(delist))},
        ])
        if not halt_ok:
            _issue(issues, "halt_denominator_shape_invalid", HALT_CHANNEL, str(len(halt)))
        if not delist_ok:
            _issue(issues, "delisting_denominator_shape_invalid", DELISTING_CHANNEL, str(len(delist)))
    return original, denominator, checks


def _validate_reports(
    producer: Path,
    inventory: pd.DataFrame,
    extract_pages: PageExtractor,
    issues: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    parsed: dict[str, dict[str, Any]] = {}
    required = {"source_kind", "source_key", "source_url", "sha256", "validation_status", "local_path"}
    if not required.issubset(inventory.columns):
        _issue(issues, "source_inventory_schema_invalid", "source_inventory", ",".join(sorted(required - set(inventory.columns))))
        return rows, parsed
    claimed = inventory.loc[
        inventory["source_kind"].eq("lifecycle_pdf")
        & inventory["validation_status"].eq("exact_month_valid")
    ]
    for item in claimed.to_dict("records"):
        month = item["source_key"]
        expected_path = (producer / "raw" / "lifecycle" / f"{month}.pdf").resolve()
        recorded_path = Path(item["local_path"]).resolve()
        canonical_path = recorded_path == expected_path and _is_within(recorded_path, producer)
        observed_hash = _sha_path(expected_path) if expected_path.is_file() else ""
        hash_match = observed_hash == item["sha256"] and bool(observed_hash)
        official_url = str(item["source_url"]).startswith(JPX_ROOT)
        parsed_report: dict[str, Any] = {"exact_month": False, "sections": {}}
        parse_error = ""
        if expected_path.is_file() and hash_match:
            try:
                parsed_report = independently_parse_lifecycle_pages(extract_pages(expected_path.read_bytes()), month)
            except Exception as exc:
                parse_error = type(exc).__name__
        exact_month = bool(parsed_report.get("exact_month"))
        row = {
            "month": month,
            "source_url": item["source_url"],
            "expected_sha256": item["sha256"],
            "observed_sha256": observed_hash,
            "hash_match": hash_match,
            "official_jpx_url": official_url,
            "canonical_raw_path": canonical_path,
            "exact_month": exact_month,
            "delisting_explicit": parsed_report.get("sections", {}).get(DELISTING_CHANNEL, {}).get("explicit", False),
            "delisting_event_rows": parsed_report.get("sections", {}).get(DELISTING_CHANNEL, {}).get("event_rows", 0),
            "delisting_explicit_zero": parsed_report.get("sections", {}).get(DELISTING_CHANNEL, {}).get("explicit_zero", False),
            "halt_explicit": parsed_report.get("sections", {}).get(HALT_CHANNEL, {}).get("explicit", False),
            "halt_event_rows": parsed_report.get("sections", {}).get(HALT_CHANNEL, {}).get("event_rows", 0),
            "halt_explicit_zero": parsed_report.get("sections", {}).get(HALT_CHANNEL, {}).get("explicit_zero", False),
            "parse_error": parse_error,
        }
        rows.append(row)
        parsed[month] = {**parsed_report, "source_url": item["source_url"], "sha256": observed_hash}
        for passed, code, detail in [
            (canonical_path, "source_path_not_canonical", str(recorded_path)),
            (hash_match, "claimed_source_hash_mismatch", observed_hash),
            (official_url, "non_official_source_url", str(item["source_url"])),
            (exact_month, "independent_exact_month_parse_failed", parse_error or str(parsed_report.get("title_months", []))),
        ]:
            if not passed:
                _issue(issues, code, month, detail)
    return rows, parsed


def _reconcile_claims(
    coverage: pd.DataFrame,
    denominator: pd.DataFrame,
    parsed_reports: dict[str, dict[str, Any]],
    issues: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {"gap_id", "month", "channel", "covered", "event_rows", "zero_event_asserted", "source_url", "source_sha256"}
    if not required.issubset(coverage.columns):
        _issue(issues, "coverage_schema_invalid", "month_channel_coverage", ",".join(sorted(required - set(coverage.columns))))
        return rows
    if {"gap_id", "month", "scope"}.issubset(denominator.columns):
        expected = denominator[["gap_id", "month", "scope"]].rename(columns={"scope": "channel"})
        observed = coverage[["gap_id", "month", "channel"]]
        if len(expected) != len(observed) or not expected.reset_index(drop=True).equals(observed.reset_index(drop=True)):
            _issue(issues, "coverage_denominator_mismatch", "month_channel_coverage", "rows/order differ from input denominator")
    covered = _bools(coverage["covered"])
    zeros = _bools(coverage["zero_event_asserted"])
    if (covered & coverage["event_rows"].astype(int).eq(0) & ~zeros).any():
        _issue(issues, "producer_inferred_zero_claim", "month_channel_coverage", "covered row lacks events and explicit zero")
    for index, item in coverage.loc[covered].iterrows():
        month = item["month"]
        channel = item["channel"]
        report = parsed_reports.get(month, {})
        section = report.get("sections", {}).get(channel, {})
        explicit = bool(report.get("exact_month") and section.get("explicit"))
        independent_rows = int(section.get("event_rows", 0))
        independent_zero = bool(section.get("explicit_zero"))
        explicit_proof = explicit and (independent_rows > 0 or independent_zero)
        source_match = item["source_url"] == report.get("source_url") and item["source_sha256"] == report.get("sha256")
        row_count_match = int(item["event_rows"]) == independent_rows
        zero_match = bool(zeros.loc[index]) == independent_zero
        reconciled = explicit_proof and source_match and row_count_match and zero_match
        rows.append({
            "gap_id": item["gap_id"],
            "month": month,
            "channel": channel,
            "producer_event_rows": int(item["event_rows"]),
            "independent_event_rows": independent_rows,
            "producer_explicit_zero": bool(zeros.loc[index]),
            "independent_explicit_zero": independent_zero,
            "exact_month": bool(report.get("exact_month")),
            "explicit_section_proven": explicit_proof,
            "source_identity_match": source_match,
            "event_row_count_match": row_count_match,
            "zero_assertion_match": zero_match,
            "claim_reconciled": reconciled,
        })
        if not explicit_proof:
            _issue(issues, "covered_claim_not_explicitly_proven", item["gap_id"], f"{month}:{channel}")
        if not source_match:
            _issue(issues, "covered_claim_source_mismatch", item["gap_id"], f"{month}:{channel}")
        if not row_count_match:
            _issue(issues, "covered_claim_event_count_mismatch", item["gap_id"], f"producer={item['event_rows']},independent={independent_rows}")
        if not zero_match:
            _issue(issues, "covered_claim_zero_assertion_mismatch", item["gap_id"], f"producer={zeros.loc[index]},independent={independent_zero}")
        if explicit and independent_rows == 0 and not independent_zero:
            _issue(issues, "independent_inferred_zero_rejected", item["gap_id"], f"{month}:{channel}")
    return rows


def _read_json(path: Path, issues: list[dict[str, str]], scope: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except Exception as exc:
        _issue(issues, "json_read_failed", scope, type(exc).__name__)
        return {}


def _read_csv(path: Path, issues: list[dict[str, str]], scope: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str).fillna("")
    except Exception as exc:
        _issue(issues, "csv_read_failed", scope, type(exc).__name__)
        return pd.DataFrame()


def _bools(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.casefold().map({"true": True, "false": False}).fillna(False).astype(bool)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _issue(issues: list[dict[str, str]], code: str, scope: str, detail: str) -> None:
    issues.append({"issue_code": code, "scope": scope, "detail": detail})


def _extract_pages(body: bytes) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages]


def _artifact_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha_path(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and str(path.relative_to(root)) not in exclude
    }


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
