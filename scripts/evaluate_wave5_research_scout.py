#!/usr/bin/env python3
"""Evaluate a research scout against the curated Wave 5 HMM gold set.

The evaluator is intentionally model-agnostic. A scout may be OpenResearcher,
an API-backed agent, or a human-reviewed multi-agent run. Discovery output is a
CSV; this command deterministically measures canonical-paper recall, red-team
recall, provenance quality, metadata completeness, duplicates, and false
positives without granting any discovered item admission to the library.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse


PRIMARY_SOURCE_DOMAINS = {
    "arxiv.org",
    "jmlr.org",
    "www.jmlr.org",
    "proceedings.mlr.press",
    "proceedings.neurips.cc",
    "projecteuclid.org",
    "repository.upenn.edu",
    "orbit.dtu.dk",
    "backend.orbit.dtu.dk",
    "papers.ssrn.com",
}
PEER_REVIEW_ENUM = {
    "published_peer_reviewed",
    "accepted_peer_reviewed",
    "preprint",
    "working_paper",
    "unknown",
}
LICENSE_STATUS_ENUM = {"verified_open", "review_required", "unknown"}
TRUE_VALUES = {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def normalize(value: str) -> str:
    ascii_value = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", ascii_value.lower()).strip()


def normalize_doi(value: str) -> str:
    value = (value or "").strip().lower()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value)
    return value.removeprefix("doi:").strip()


def normalize_arxiv(value: str) -> str:
    value = (value or "").strip().lower()
    match = re.search(r"(?<!\d)(\d{4}\.\d{4,5})(?:v\d+)?(?!\d)", value)
    return match.group(1) if match else value.removeprefix("arxiv:").strip()


def row_keys(row: dict[str, str]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    doi = normalize_doi(row.get("doi", ""))
    arxiv_id = normalize_arxiv(row.get("arxiv_id", ""))
    title = normalize(row.get("title", ""))
    if doi:
        keys.append(("doi", doi))
    if arxiv_id:
        keys.append(("arxiv", arxiv_id))
    if title:
        keys.append(("title", title))
    return keys


def source_domain(row: dict[str, str]) -> str:
    url = row.get("landing_url") or row.get("pdf_url") or row.get("source_url") or ""
    return urlparse(url).netloc.lower()


def peer_review_status(row: dict[str, str]) -> str:
    status = normalize(row.get("peer_review_status", "")).replace(" ", "_")
    return status if status in PEER_REVIEW_ENUM else "invalid"


def license_status(row: dict[str, str]) -> str:
    status = normalize(row.get("license_status", "")).replace(" ", "_")
    return status if status in LICENSE_STATUS_ENUM else "invalid"


def is_true(value: str) -> bool:
    return (value or "").strip().lower() in TRUE_VALUES


def author_surnames(value: str) -> set[str]:
    surnames: set[str] = set()
    for author in re.split(r"[;|]", value or ""):
        tokens = re.findall(r"[a-z0-9]+", normalize(author))
        if tokens:
            surnames.add(tokens[-1])
    return surnames


def metadata_discrepancies(discovered: dict[str, str], gold: dict[str, str]) -> list[str]:
    issues: list[str] = []
    discovered_title = normalize(discovered.get("title", ""))
    gold_title = normalize(gold.get("title", ""))
    if not discovered_title or discovered_title != gold_title:
        issues.append("title")
    for field, normalizer in (("doi", normalize_doi), ("arxiv_id", normalize_arxiv)):
        left = normalizer(discovered.get(field, ""))
        right = normalizer(gold.get(field, ""))
        if left and right and left != right:
            issues.append(field)
    if gold.get("year") and discovered.get("year") != gold.get("year"):
        issues.append("year")
    discovered_authors = author_surnames(discovered.get("authors", ""))
    gold_authors = author_surnames(gold.get("authors", ""))
    if not discovered_authors or not gold_authors or not discovered_authors.intersection(gold_authors):
        issues.append("authors")
    return issues


def percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", required=True, help="Curated Wave 5 candidate_manifest.csv")
    parser.add_argument("--discovered", required=True, help="Research scout discovery CSV")
    parser.add_argument("--roster", required=True, help="Canonical 20-agent roster CSV")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scout-name", default="unspecified_scout")
    parser.add_argument("--trace", help="Optional tool/review trace CSV")
    parser.add_argument("--require-trace", action="store_true")
    args = parser.parse_args()

    gold_path = Path(args.gold).expanduser().resolve()
    discovered_path = Path(args.discovered).expanduser().resolve()
    roster_path = Path(args.roster).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gold = read_csv(gold_path)
    discovered = read_csv(discovered_path)
    roster = read_csv(roster_path)
    trace = read_csv(Path(args.trace).expanduser().resolve()) if args.trace else []
    valid_agents = {row["agent_id"] for row in roster}
    if not gold:
        raise SystemExit("Gold manifest is empty")
    if not discovered:
        raise SystemExit("Discovery CSV is empty")

    gold_index: dict[tuple[str, str], list[int]] = {}
    for index, row in enumerate(gold):
        for key in row_keys(row):
            gold_index.setdefault(key, []).append(index)

    results: list[dict[str, object]] = []
    matched_gold: set[int] = set()
    duplicate_counter: Counter[tuple[str, str]] = Counter()
    unmatched = 0
    ambiguous = 0
    matched_rows = 0
    metadata_issue_rows = 0
    licence_false_assertions = 0
    peer_review_false_assertions = 0
    evidence_missing_rows = 0
    for position, row in enumerate(discovered, 1):
        keys = row_keys(row)
        for key in keys:
            duplicate_counter[key] += 1
        candidates: set[int] = set()
        matched_by: list[str] = []
        for key in keys:
            indexes = gold_index.get(key, [])
            if indexes:
                candidates.update(indexes)
                matched_by.append(key[0])
        if len(candidates) == 1:
            gold_index_value = next(iter(candidates))
            gold_row = gold[gold_index_value]
            matched_gold.add(gold_index_value)
            match_status = "matched"
            gold_id = gold_row.get("candidate_id", "")
            admission_class = gold_row.get("admission_class", "")
            matched_rows += 1
            discrepancies = metadata_discrepancies(row, gold_row)
            metadata_issue_rows += bool(discrepancies)
            discovered_licence = license_status(row)
            gold_licence = license_status(gold_row)
            if discovered_licence == "verified_open" and gold_licence != "verified_open":
                licence_false_assertions += 1
            discovered_peer = peer_review_status(row)
            gold_peer = peer_review_status(gold_row)
            if discovered_peer in {"published_peer_reviewed", "accepted_peer_reviewed"} and gold_peer not in {
                "published_peer_reviewed", "accepted_peer_reviewed"
            }:
                peer_review_false_assertions += 1
            needs_peer_evidence = discovered_peer in {"published_peer_reviewed", "accepted_peer_reviewed"}
            needs_licence_evidence = discovered_licence == "verified_open"
            evidence_missing = (
                (needs_peer_evidence and not row.get("peer_review_evidence_url", "").startswith("https://"))
                or (needs_licence_evidence and not row.get("license_evidence_url", "").startswith("https://"))
                or not is_true(row.get("primary_page_opened", ""))
            )
            evidence_missing_rows += bool(evidence_missing)
        elif len(candidates) > 1:
            match_status = "ambiguous"
            gold_id = ""
            admission_class = ""
            ambiguous += 1
            discrepancies = []
            evidence_missing = True
        else:
            match_status = "unmatched"
            gold_id = ""
            admission_class = ""
            unmatched += 1
            discrepancies = []
            evidence_missing = True
        results.append(
            {
                "discovery_position": position,
                "discovered_title": row.get("title", ""),
                "discovered_doi": row.get("doi", ""),
                "discovered_arxiv_id": row.get("arxiv_id", ""),
                "source_domain": source_domain(row),
                "primary_source": source_domain(row) in PRIMARY_SOURCE_DOMAINS,
                "peer_review_status": peer_review_status(row),
                "license_status": license_status(row),
                "primary_page_opened": is_true(row.get("primary_page_opened", "")),
                "metadata_discrepancies": ";".join(discrepancies),
                "evidence_missing": evidence_missing,
                "match_status": match_status,
                "matched_by": ";".join(sorted(set(matched_by))),
                "gold_candidate_id": gold_id,
                "admission_class": admission_class,
            }
        )

    class_counts = Counter(row.get("admission_class", "") for row in gold)
    matched_class_counts = Counter(gold[index].get("admission_class", "") for index in matched_gold)
    duplicate_keys = {key for key, count in duplicate_counter.items() if count > 1}
    duplicate_rows = sum(
        1 for row in discovered if any(key in duplicate_keys for key in row_keys(row))
    )
    primary_count = sum(bool(row["primary_source"]) for row in results)
    valid_peer_count = sum(row["peer_review_status"] != "invalid" for row in results)
    valid_license_count = sum(row["license_status"] != "invalid" for row in results)

    trace_open_candidates = {
        row.get("candidate_id", "") for row in trace
        if row.get("event_type") in {"open", "find"}
        and row.get("status") == "success"
        and (row.get("url") or "").startswith("https://")
    }
    trace_supported_candidates = {
        row.get("candidate_id", "") for row in trace
        if is_true(row.get("citation_supported", ""))
    }
    matched_candidate_ids = {
        str(row["gold_candidate_id"]) for row in results if row["match_status"] == "matched"
    }
    trace_open_rate = percent(len(matched_candidate_ids & trace_open_candidates), len(matched_candidate_ids))
    trace_support_rate = percent(len(matched_candidate_ids & trace_supported_candidates), len(matched_candidate_ids))
    trace_calls = len(trace)
    trace_tokens = sum(int(row.get("tokens", "0") or 0) for row in trace)
    trace_cost = round(sum(float(row.get("cost_usd", "0") or 0) for row in trace), 6)

    route_errors: list[str] = []
    for row in gold:
        primary = (row.get("primary_specialist") or "").strip()
        secondaries = [
            value.strip() for value in re.split(r"[;,|]", row.get("secondary_specialists", "")) if value.strip()
        ]
        if primary not in valid_agents:
            route_errors.append(f"{row.get('candidate_id')}: unknown primary specialist {primary}")
        for secondary in secondaries:
            if secondary not in valid_agents:
                route_errors.append(f"{row.get('candidate_id')}: unknown secondary specialist {secondary}")

    metrics = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scout_name": args.scout_name,
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "discovered_sha256": hashlib.sha256(discovered_path.read_bytes()).hexdigest(),
        "gold_total": len(gold),
        "discovered_total": len(discovered),
        "matched_unique_gold": len(matched_gold),
        "matched_discovery_rows": matched_rows,
        "overall_recall_pct": percent(len(matched_gold), len(gold)),
        "precision_pct": percent(len(matched_gold), len(discovered)),
        "false_discovery_rate_pct": percent(unmatched + ambiguous + duplicate_rows, len(discovered)),
        "core_recall_pct": percent(matched_class_counts["core"], class_counts["core"]),
        "red_team_recall_pct": percent(matched_class_counts["red_team"], class_counts["red_team"]),
        "watchlist_recall_pct": percent(matched_class_counts["watchlist"], class_counts["watchlist"]),
        "unmatched_discoveries": unmatched,
        "ambiguous_discoveries": ambiguous,
        "duplicate_discovery_rows": duplicate_rows,
        "primary_source_pct": percent(primary_count, len(discovered)),
        "peer_review_enum_valid_pct": percent(valid_peer_count, len(discovered)),
        "license_status_enum_valid_pct": percent(valid_license_count, len(discovered)),
        "metadata_discrepancy_rows": metadata_issue_rows,
        "license_false_assertions": licence_false_assertions,
        "peer_review_false_assertions": peer_review_false_assertions,
        "evidence_missing_rows": evidence_missing_rows,
        "trace_supplied": bool(trace),
        "trace_calls": trace_calls,
        "trace_tokens": trace_tokens,
        "trace_cost_usd": trace_cost,
        "trace_primary_open_rate_pct": trace_open_rate,
        "trace_citation_support_rate_pct": trace_support_rate,
        "route_error_count": len(route_errors),
        "route_errors": route_errors,
    }
    minimums = {
        "core_recall_pct": 90.0,
        "red_team_recall_pct": 75.0,
        "precision_pct": 85.0,
        "primary_source_pct": 90.0,
        "peer_review_enum_valid_pct": 100.0,
        "license_status_enum_valid_pct": 100.0,
    }
    maximums = {
        "false_discovery_rate_pct": 10.0,
        "ambiguous_discoveries": 0,
        "duplicate_discovery_rows": 0,
        "metadata_discrepancy_rows": 0,
        "license_false_assertions": 0,
        "peer_review_false_assertions": 0,
        "evidence_missing_rows": 0,
        "route_error_count": 0,
    }
    if args.require_trace:
        minimums.update({"trace_primary_open_rate_pct": 90.0, "trace_citation_support_rate_pct": 90.0})
    failures = [
        f"{key}={metrics[key]} below {value}"
        for key, value in minimums.items()
        if float(metrics[key]) < float(value)
    ]
    failures.extend(
        f"{key}={metrics[key]} exceeds {value}"
        for key, value in maximums.items()
        if float(metrics[key]) > float(value)
    )
    if args.require_trace and not trace:
        failures.append("trace_supplied=false while --require-trace was set")
    metrics["benchmark_status"] = "pass" if not failures else "fail"
    metrics["benchmark_failures"] = failures

    match_fields = list(results[0])
    write_csv(output_dir / "match_results.csv", match_fields, results)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Wave 5 Research Scout Benchmark",
        "",
        f"- Scout: `{args.scout_name}`",
        f"- Status: **{metrics['benchmark_status']}**",
        f"- Core recall: **{metrics['core_recall_pct']}%**",
        f"- Red-team recall: **{metrics['red_team_recall_pct']}%**",
        f"- Precision: **{metrics['precision_pct']}%**",
        f"- False-discovery rate: **{metrics['false_discovery_rate_pct']}%**",
        f"- Watchlist recall: **{metrics['watchlist_recall_pct']}%**",
        f"- Primary-source rate: **{metrics['primary_source_pct']}%**",
        f"- Licence-status enum validity: **{metrics['license_status_enum_valid_pct']}%**",
        f"- Metadata discrepancy rows: **{metadata_issue_rows}**",
        f"- False licence assertions: **{licence_false_assertions}**",
        f"- Primary-page trace open rate: **{trace_open_rate}%**",
        f"- Citation-support trace rate: **{trace_support_rate}%**",
        f"- Unmatched discoveries: **{unmatched}**",
        f"- Duplicate discovery rows: **{duplicate_rows}**",
        f"- Route errors: **{len(route_errors)}**",
        "",
        "## Gate failures",
        "",
    ]
    report.extend([f"- {failure}" for failure in failures] or ["- None"])
    report.extend(
        [
            "",
            "Passing this benchmark verifies retrieval and metadata discipline only. It does not admit papers,",
            "validate mathematical claims, clear commercial reuse, or certify PDF-to-Markdown fidelity.",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0 if metrics["benchmark_status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
