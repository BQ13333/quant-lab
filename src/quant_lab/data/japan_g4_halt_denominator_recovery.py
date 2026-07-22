from __future__ import annotations

import hashlib
import io
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
from bs4 import BeautifulSoup
from pypdf import PdfReader


BASE = "https://www.jpx.co.jp"
HALT_CHANNEL = "halt_suspension_resumption"
DELISTING_CHANNEL = "delisting_non_trading"
EXPECTED_GAPS = 366
USER_AGENT = "quant-lab-japan-g4-denominator-recovery/1.0"
ARCHIVE_START_YEAR = 2016
LEGACY_START_YEAR = 2009
LEGACY_END_YEAR = 2011

Fetcher = Callable[[str], tuple[int, bytes, Mapping[str, str]]]
TextExtractor = Callable[[bytes], str]


@dataclass(frozen=True)
class JapanG4DenominatorRecoveryArtifacts:
    run_id: str
    artifact_dir: Path
    manifest_path: Path
    coverage_path: Path
    manifest: dict[str, Any]


def archive_page(year: int, end_year: int = 2026) -> str:
    if year < ARCHIVE_START_YEAR or year > end_year:
        raise ValueError("retained archive year is outside the bounded public selector")
    if year == end_year:
        return f"{BASE}/english/markets/statistics-equities/monthly/index.html"
    return (
        f"{BASE}/english/markets/statistics-equities/monthly/"
        f"00-archives-{end_year - year:02d}.html"
    )


def discover_lifecycle_urls(body: bytes, page_url: str) -> dict[str, str]:
    """Return exact-month lifecycle reports linked by an official JPX archive page."""
    found: dict[str, str] = {}
    soup = BeautifulSoup(body, "html.parser")
    for link in soup.find_all("a", href=True):
        href = str(link["href"]).strip()
        match = re.search(r"/(?:16|17|20)_idou(?P<yy>\d{2})(?P<mm>\d{2})\.pdf$", href)
        if not match:
            continue
        year = 2000 + int(match.group("yy"))
        month = f"{year:04d}-{int(match.group('mm')):02d}"
        found[month] = urllib.parse.urljoin(page_url, href)
    return found


def legacy_report_url(month: str) -> str:
    period = pd.Period(month, freq="M")
    yyyymm = period.strftime("%Y%m")
    yymm = period.strftime("%y%m")
    return f"{BASE}/files/tse/english/market/data/geppo/{yyyymm}/{yymm}.pdf"


def inspect_lifecycle_text(text: str, month: str) -> dict[str, Any]:
    """Fail closed unless the PDF names the month and explicitly prints a channel section."""
    period = pd.Period(month, freq="M")
    month_name = period.start_time.strftime("%b")
    exact_month = bool(
        re.search(
            rf"Changes\s+in\s+Listed\s+Companies\s+and\s+Issues.*?"
            rf"\({month_name}\.?\s*{period.year}\)",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
    )
    sections = {
        DELISTING_CHANNEL: _section_status(text, "Delisting", period),
        HALT_CHANNEL: _section_status(text, "Suspension of Trading", period),
    }
    return {"exact_month": exact_month, "sections": sections}


def _section_status(text: str, heading: str, period: pd.Period) -> dict[str, Any]:
    lines = [" ".join(line.split()) for line in text.splitlines()]
    starts = [index for index, line in enumerate(lines) if line.casefold() == heading.casefold()]
    if len(starts) != 1:
        return {"explicit": False, "event_rows": 0, "explicit_zero": False}
    start = starts[0] + 1
    end = len(lines)
    for index in range(start, len(lines)):
        if _is_next_event_heading(lines[index], heading):
            end = index
            break
    section = lines[start:end]
    date_pattern = rf"{period.year}\s*[./]\s*0?{period.month}\s*[./]\s*\d{{1,2}}"
    event_rows = sum(len(re.findall(date_pattern, line)) for line in section)
    explicit_zero = any(line.casefold() in {"none", "nil", "zero", "no events"} for line in section)
    return {
        "explicit": True,
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


def run_japan_g4_halt_denominator_recovery(
    *,
    run_id: str,
    unresolved_gaps_path: str | Path,
    artifact_root: str | Path = "artifacts/japan_g4_halt_denominator_recovery",
    end_year: int = 2026,
    max_pdfs: int = 1000,
    fetcher: Fetcher | None = None,
    text_extractor: TextExtractor | None = None,
    require_expected_gaps: bool = True,
    oracle_validation_dir: str | Path | None = None,
    require_oracle: bool = True,
    write_once: bool = True,
) -> JapanG4DenominatorRecoveryArtifacts:
    if not run_id or Path(run_id).name != run_id:
        raise ValueError("run_id must be a path-safe name")
    if max_pdfs < 0:
        raise ValueError("max_pdfs must be non-negative")
    source_path = Path(unresolved_gaps_path).resolve()
    all_gaps = pd.read_csv(source_path, dtype=str).fillna("")
    gaps = all_gaps.loc[all_gaps["gap_type"].eq("source_denominator_uncovered")].copy()
    if require_expected_gaps and len(gaps) != EXPECTED_GAPS:
        raise ValueError(f"expected exactly {EXPECTED_GAPS} source denominator gaps, got {len(gaps)}")
    if gaps["gap_id"].duplicated().any():
        raise ValueError("source denominator gap ids must be unique")

    oracle, oracle_meta = _load_validation_oracle(oracle_validation_dir, require_oracle)
    oracle_by_gap = {row["gap_id"]: row for row in oracle.to_dict("records")}

    fetch = fetcher or _fetch
    extract = text_extractor or _extract_text
    root = Path(artifact_root) / run_id
    if write_once and root.exists():
        raise FileExistsError(f"write-once artifact already exists: {root}")
    raw_pages = root / "raw" / "archive_pages"
    raw_pdfs = root / "raw" / "lifecycle"
    raw_pages.mkdir(parents=True, exist_ok=True)
    raw_pdfs.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    lifecycle_urls: dict[str, str] = {}
    for year in range(ARCHIVE_START_YEAR, end_year + 1):
        url = archive_page(year, end_year)
        status, body, _ = fetch(url)
        local = raw_pages / f"{year}.html"
        local.write_bytes(body)
        digest = _sha_bytes(body)
        discovered = discover_lifecycle_urls(body, url) if status == 200 else {}
        lifecycle_urls.update(discovered)
        source_rows.append(
            _source_row("archive_page", str(year), url, status, local, digest, "discovery_only", len(discovered))
        )

    target_months = set(gaps["month"])
    for period in pd.period_range(f"{LEGACY_START_YEAR}-01", f"{LEGACY_END_YEAR}-12", freq="M"):
        month = str(period)
        if month in target_months:
            lifecycle_urls.setdefault(month, legacy_report_url(month))

    inspections: dict[str, dict[str, Any]] = {}
    for month, url in sorted(lifecycle_urls.items())[:max_pdfs]:
        if month not in target_months:
            continue
        status, body, _ = fetch(url)
        local = raw_pdfs / f"{month}.pdf"
        local.write_bytes(body)
        digest = _sha_bytes(body)
        validation = "http_or_pdf_invalid"
        inspected = {"exact_month": False, "sections": {}}
        if status == 200 and body.startswith(b"%PDF"):
            try:
                inspected = inspect_lifecycle_text(extract(body), month)
                validation = "exact_month_valid" if inspected["exact_month"] else "month_heading_invalid"
            except Exception as exc:  # Keep corrupt/unparseable sources visible and uncovered.
                validation = f"parse_error:{type(exc).__name__}"
        if validation == "exact_month_valid":
            inspections[month] = {**inspected, "url": url, "sha256": digest, "local_path": str(local.resolve())}
        source_rows.append(_source_row("lifecycle_pdf", month, url, status, local, digest, validation, 0))

    coverage_rows: list[dict[str, Any]] = []
    for row in gaps.to_dict("records"):
        month = row["month"]
        channel = row["scope"]
        evidence = inspections.get(month)
        section = (evidence or {}).get("sections", {}).get(channel, {})
        source_candidate = bool(
            evidence
            and section.get("explicit")
            and (int(section.get("event_rows", 0)) > 0 or section.get("explicit_zero"))
        )
        oracle_row = oracle_by_gap.get(row["gap_id"])
        oracle_claim_present = oracle_row is not None
        oracle_explicit = bool(oracle_row and _bool_value(oracle_row.get("explicit_section_proven")))
        oracle_rows = int(oracle_row.get("independent_event_rows", 0)) if oracle_row else 0
        oracle_zero = bool(oracle_row and _bool_value(oracle_row.get("independent_explicit_zero")))
        oracle_source_match = bool(oracle_row and _bool_value(oracle_row.get("source_identity_match")))
        oracle_count_match = bool(
            oracle_row
            and int(section.get("event_rows", 0)) == oracle_rows
            and bool(section.get("explicit_zero")) == oracle_zero
        )
        if require_oracle:
            covered = bool(
                source_candidate
                and oracle_claim_present
                and oracle_explicit
                and oracle_source_match
                and oracle_count_match
            )
        else:
            covered = source_candidate

        if covered:
            reason = "explicit_exact_month_channel_section"
        elif source_candidate and require_oracle and not oracle_claim_present:
            reason = "not_covered_by_independent_v1_oracle"
        elif source_candidate and require_oracle and not oracle_explicit:
            reason = "rejected_by_independent_oracle_no_explicit_event_or_zero"
        elif source_candidate and require_oracle and not oracle_source_match:
            reason = "rejected_by_independent_oracle_source_identity_mismatch"
        elif source_candidate and require_oracle and not oracle_count_match:
            reason = "rejected_by_independent_oracle_event_count_mismatch"
        elif evidence and section.get("explicit"):
            reason = "section_has_no_rows_and_no_explicit_zero"
        elif evidence:
            reason = "channel_section_absent_no_zero_inference"
        else:
            reason = "no_admitted_exact_month_lifecycle_source"
        coverage_rows.append(
            {
                "gap_id": row["gap_id"],
                "month": month,
                "channel": channel,
                "covered": covered,
                "coverage_status": "covered_candidate" if covered else "uncovered",
                "event_rows": int(section.get("event_rows", 0)) if covered else 0,
                "zero_event_asserted": bool(section.get("explicit_zero")) if covered else False,
                "source_url": evidence.get("url", "") if covered else "",
                "source_sha256": evidence.get("sha256", "") if covered else "",
                "acceptance_reason": reason,
                "oracle_claim_present": oracle_claim_present,
                "oracle_explicit_section_proven": oracle_explicit,
                "oracle_event_rows": oracle_rows if oracle_claim_present else 0,
                "oracle_explicit_zero": oracle_zero if oracle_claim_present else False,
                "oracle_source_identity_match": oracle_source_match,
                "oracle_event_count_match": oracle_count_match,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    coverage_path = root / "month_channel_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    source_inventory = pd.DataFrame(source_rows)
    source_inventory.to_csv(root / "source_inventory.csv", index=False)
    gaps.to_csv(root / "input_denominator.csv", index=False)

    counts = _coverage_counts(coverage)
    completion = (
        coverage.loc[~coverage["covered"]]
        .groupby(["channel", "acceptance_reason"], as_index=False)
        .agg(uncovered_months=("month", "count"), first_month=("month", "min"), last_month=("month", "max"))
    )
    completion["next_action"] = completion["acceptance_reason"].map(_next_action)
    completion.to_csv(root / "completion_plan.csv", index=False)
    _validation_plan().to_csv(root / "independent_validation_plan.csv", index=False)

    source_input_sha = _sha_path(source_path)
    validation_checks = pd.DataFrame(
        [
            {"check": "source_gap_count", "passed": len(gaps) == (EXPECTED_GAPS if require_expected_gaps else len(gaps)), "detail": str(len(gaps))},
            {"check": "gap_ids_unique", "passed": gaps["gap_id"].is_unique, "detail": str(gaps["gap_id"].nunique())},
            {"check": "no_inferred_zero", "passed": not ((coverage["covered"]) & (coverage["event_rows"].eq(0)) & (~coverage["zero_event_asserted"])).any(), "detail": "covered zero months require explicit zero text"},
            {"check": "official_sources_only", "passed": source_inventory["source_url"].str.startswith(BASE).all(), "detail": BASE},
        ]
    )
    validation_checks.to_csv(root / "validation_checks.csv", index=False)

    hashes = _artifact_hashes(root, exclude={"manifest.json", "artifact_hashes.json"})
    (root / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial_recovery_candidate" if counts["covered"] else "no_recovery_candidate",
        "source_input": str(source_path),
        "source_input_sha256": source_input_sha,
        "source_denominator_rows": len(gaps),
        "coverage": counts,
        "scope_bounds": {"public_archive_start_year": ARCHIVE_START_YEAR, "legacy_deterministic_start_year": LEGACY_START_YEAR, "legacy_deterministic_end_year": LEGACY_END_YEAR, "end_year": end_year},
        "acceptance_policy": "exact month plus explicit channel section and event rows or explicit zero text",
        "absence_is_zero": False,
        "refinitiv_used": False,
        "independent_validation_status": "not_run",
        "admission_oracle": oracle_meta,
        "oracle_gated": require_oracle,
        "oracle_claim_rows": int(len(oracle)),
        "oracle_rejected_claims": int(sum(1 for item in coverage_rows if item["oracle_claim_present"] and not item["covered"])),
        "event_count_reconciliation_required": True,
        "write_once": write_once,
        "scorecard_eligible": False,
        "promotion_recommended": False,
        "artifact_hashes_sha256": _sha_path(root / "artifact_hashes.json"),
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return JapanG4DenominatorRecoveryArtifacts(run_id, root, manifest_path, coverage_path, manifest)


def validate_recovery_artifacts(artifact_dir: str | Path) -> list[str]:
    root = Path(artifact_dir)
    expected = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    issues = []
    for relative, digest in expected.items():
        path = root / relative
        if not path.is_file() or _sha_path(path) != digest:
            issues.append(f"artifact_hash_mismatch:{relative}")
    try:
        coverage = pd.read_csv(root / "month_channel_coverage.csv")
        required = {"covered", "event_rows", "zero_event_asserted"}
        if not required.issubset(coverage.columns):
            issues.append("coverage_schema_invalid")
        else:
            bad_zero = coverage["covered"].astype(bool) & coverage["event_rows"].eq(0) & ~coverage["zero_event_asserted"].astype(bool)
            if bad_zero.any():
                issues.append("inferred_zero_coverage")
    except Exception as exc:
        issues.append(f"coverage_parse_error:{type(exc).__name__}")
    return issues


def _source_row(kind: str, key: str, url: str, status: int, local: Path, digest: str, validation: str, discovered: int) -> dict[str, Any]:
    return {"source_kind": kind, "source_key": key, "source_url": url, "http_status": status, "local_path": str(local.resolve()), "sha256": digest, "validation_status": validation, "discovered_reports": discovered}


def _coverage_counts(coverage: pd.DataFrame) -> dict[str, Any]:
    by_channel: dict[str, Any] = {}
    for channel, group in coverage.groupby("channel"):
        by_channel[channel] = {"total": int(len(group)), "covered": int(group["covered"].sum()), "uncovered": int((~group["covered"]).sum())}
    return {"total": int(len(coverage)), "covered": int(coverage["covered"].sum()), "uncovered": int((~coverage["covered"]).sum()), "by_channel": by_channel}


def _next_action(reason: str) -> str:
    return {
        "no_admitted_exact_month_lifecycle_source": "acquire an official exact-month lifecycle report/export and preserve its hash",
        "channel_section_absent_no_zero_inference": "obtain an official channel-specific zero assertion or complete export; do not infer zero",
        "section_has_no_rows_and_no_explicit_zero": "independently inspect/OCR the section and require explicit zero evidence before coverage",
        "not_covered_by_independent_v1_oracle": "retain uncovered until a new independent validation proves the exact claim",
        "rejected_by_independent_oracle_no_explicit_event_or_zero": "retain rejected; obtain explicit official event or zero evidence",
        "rejected_by_independent_oracle_source_identity_mismatch": "retain rejected; reconcile official URL and source hash",
        "rejected_by_independent_oracle_event_count_mismatch": "retain rejected; reconcile exact section event counts",
    }.get(reason, "independently validate the exact source before admission")


def _load_validation_oracle(
    oracle_dir: str | Path | None,
    required: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if oracle_dir is None:
        if required:
            raise ValueError("oracle_validation_dir is required for fail-closed admission")
        return pd.DataFrame(), {"required": False}
    root = Path(oracle_dir).resolve()
    manifest_path = root / "manifest.json"
    hashes_path = root / "artifact_hashes.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if _sha_path(hashes_path) != manifest.get("artifact_hashes_sha256"):
        raise ValueError("independent validation oracle hash inventory mismatch")
    hashes = json.loads(hashes_path.read_text(encoding="utf-8"))
    for relative, expected in hashes.items():
        path = root / relative
        if not path.is_file() or _sha_path(path) != expected:
            raise ValueError(f"independent validation oracle artifact mismatch: {relative}")
    claims = pd.read_csv(root / "claim_reconciliation.csv", dtype=str).fillna("")
    required_columns = {
        "gap_id",
        "independent_event_rows",
        "independent_explicit_zero",
        "explicit_section_proven",
        "source_identity_match",
    }
    if not required_columns.issubset(claims.columns) or claims["gap_id"].duplicated().any():
        raise ValueError("independent validation oracle claim schema is invalid")
    return claims, {
        "required": required,
        "path": str(root),
        "manifest_sha256": _sha_path(manifest_path),
        "artifact_hashes_sha256": _sha_path(hashes_path),
        "status": manifest.get("status", ""),
        "claim_rows": int(len(claims)),
    }


def _bool_value(value: Any) -> bool:
    return str(value).strip().casefold() == "true"


def _validation_plan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"step": 1, "owner": "independent_validator", "required": True, "action": "re-download every admitted official URL and compare SHA-256"},
            {"step": 2, "owner": "independent_validator", "required": True, "action": "reparse exact month and both channel sections with an independent parser/OCR route"},
            {"step": 3, "owner": "independent_validator", "required": True, "action": "manually inspect all explicit-zero candidates and a deterministic event-month sample"},
            {"step": 4, "owner": "controller_owner", "required": True, "action": "keep promotion disabled until all hashes, counts, and channel semantics pass"},
        ]
    )


def _artifact_hashes(root: Path, exclude: set[str]) -> dict[str, str]:
    return {str(path.relative_to(root)): _sha_path(path) for path in sorted(root.rglob("*")) if path.is_file() and str(path.relative_to(root)) not in exclude}


def _sha_bytes(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _sha_path(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _extract_text(body: bytes) -> str:
    return "\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(body)).pages)


def _fetch(url: str) -> tuple[int, bytes, Mapping[str, str]]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return int(response.status), response.read(), dict(response.headers.items())
    except urllib.error.HTTPError as exc:
        return int(exc.code), exc.read(), dict(exc.headers.items())
