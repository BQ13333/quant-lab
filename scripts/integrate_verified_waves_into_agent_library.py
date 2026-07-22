#!/usr/bin/env python3
"""Integrate verified acquisition-wave Markdown into the canonical 20-agent library.

Each paper is materialized once under its primary specialist. Secondary agents
receive index routes only. Localized image assets are copied into the primary
specialist package and only Markdown image targets are rewritten; all other
Markdown text remains byte-identical to the audited canonical artifact.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


WAVES = (
    ("wave1", "ACQ-FIRST-LANES-20260712", 36),
    ("wave2", "ACQ-WAVE2-FOUNDATIONS-20260712", 36),
    ("wave3", "ACQ-WAVE3-DEPTH-20260712", 42),
    ("wave4", "ACQ-WAVE4-SYNTHESIS-20260712", 36),
    ("wave5", "ACQ-WAVE5-HMM-REGIME-20260715", 18),
)
EXPECTED_CORPUS_ROWS = 447
EXPECTED_ACTIVE_PAPERS = 446
EXPECTED_QUARANTINED_PAPERS = 1
EXPECTED_ACTIVE_PAGES = 12_053
EXPECTED_SECONDARY_ROUTES = 946
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
PAGE_RE = re.compile(r"<!--\s*OCR_PAGE\s+(\d+)\s+source=[^>]+-->")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_agent(value: str) -> str:
    return re.sub(r"^\d+_", "", (value or "").strip())


def split_agents(value: str) -> list[str]:
    return [normalize_agent(item) for item in re.split(r"[;,|]", value or "") if item.strip()]


def normalize_image_targets(text: str) -> str:
    return IMAGE_RE.sub(lambda match: f"![{match.group(1)}](<LOCAL_IMAGE>)", text)


def check_run_audits(run_dir: Path) -> None:
    acceptance = json.loads((run_dir / "acceptance_audit.json").read_text(encoding="utf-8"))
    localized = json.loads((run_dir / "localized_acceptance_audit.json").read_text(encoding="utf-8"))
    unexpected_warnings = [
        warning for warning in acceptance.get("warnings", [])
        if not str(warning).startswith("equation-tag review lane retained:")
    ]
    formula_retries_clean = (
        acceptance.get("resolved_formula_retry_pages") == acceptance.get("formula_retry_pages")
        and not acceptance.get("tag_body_integrity_failures")
    )
    if (
        acceptance.get("status") != "pass"
        or acceptance.get("error_count")
        or unexpected_warnings
        or not formula_retries_clean
    ):
        raise RuntimeError(f"Corpus acceptance audit is not clean: {run_dir}")
    if localized.get("status") != "pass" or localized.get("errors"):
        raise RuntimeError(f"Localized acceptance audit is not clean: {run_dir}")


def route_markdown(canonical: Path, target: Path, routed_id: str) -> tuple[int, int]:
    source_text = canonical.read_text(encoding="utf-8")
    image_count = 0
    asset_root = target.parent.parent / "assets" / routed_id

    def replace_image(match: re.Match[str]) -> str:
        nonlocal image_count
        alt, raw_target = match.group(1), match.group(2).strip().strip("<>")
        if raw_target.startswith(("http://", "https://", "data:")) or raw_target == "image":
            raise RuntimeError(f"Non-durable image target in {canonical}: {raw_target}")
        source_asset = (canonical.parent / raw_target).resolve()
        if not source_asset.is_file():
            raise RuntimeError(f"Localized source asset missing: {source_asset}")
        digest = sha256_file(source_asset)
        suffix = source_asset.suffix.lower() or ".bin"
        destination = asset_root / f"{digest[:16]}__{source_asset.stem}{suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            shutil.copy2(source_asset, destination)
        if sha256_file(destination) != digest:
            raise RuntimeError(f"Routed asset hash mismatch: {destination}")
        image_count += 1
        relative = os.path.relpath(destination, target.parent).replace(os.sep, "/")
        return f"![{alt}]({relative})"

    routed_text = IMAGE_RE.sub(replace_image, source_text)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(routed_text, encoding="utf-8")
    if normalize_image_targets(source_text) != normalize_image_targets(routed_text):
        raise RuntimeError(f"Non-image Markdown content changed during routing: {target}")
    source_pages = PAGE_RE.findall(source_text)
    routed_pages = PAGE_RE.findall(routed_text)
    if source_pages != routed_pages:
        raise RuntimeError(f"Page markers changed during routing: {target}")
    for match in IMAGE_RE.finditer(routed_text):
        resolved = (target.parent / match.group(2).strip().strip("<>")).resolve()
        if not resolved.is_file():
            raise RuntimeError(f"Routed image target is broken: {resolved}")
    return len(source_pages), image_count


def rebuild_primary_index(folder: Path, title: str, rows: list[dict[str, str]]) -> None:
    lines = [f"# {title} — Primary Papers", ""]
    for row in sorted(rows, key=lambda item: (item.get("year", ""), item.get("title", "")), reverse=True):
        markdown = Path(row["markdown_path"])
        relative = os.path.relpath(markdown, folder).replace(os.sep, "/")
        lines.append(f"- [{row['title']}]({relative}) — {row.get('year') or 'year unknown'}")
    (folder / "INDEX.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", default=".")
    args = parser.parse_args()
    workspace = Path(args.workspace_root).expanduser().resolve()
    library = workspace / "PhD_Math_Agent_Library"
    config = json.loads((workspace / "quant_lab/configs/phd_math_specialists.json").read_text(encoding="utf-8"))
    specialists = config["specialists"]
    if len(specialists) != 20:
        raise RuntimeError(f"Expected 20 specialists, found {len(specialists)}")
    title_by_agent = {item["id"]: item["title"] for item in specialists}
    folders: dict[str, Path] = {}
    for folder in sorted((library / "specialists").glob("[0-9][0-9]_*")):
        folders[folder.name[3:]] = folder
    if set(folders) != set(title_by_agent):
        raise RuntimeError("Specialist folders do not match the canonical 20-agent config")

    corpus_path = library / "indexes/corpus_index.csv"
    corpus_rows = read_csv(corpus_path)
    corpus_fields = list(corpus_rows[0])
    existing_by_id = {row["document_id"]: row for row in corpus_rows}
    new_rows: list[dict[str, str]] = []
    route_rows: list[dict[str, Any]] = []
    wave_ids: set[str] = set()

    for wave_slug, batch_name, expected_documents in WAVES:
        batch = library / "acquisition_staging" / batch_name
        candidate_by_id = {
            row["candidate_id"]: row
            for row in read_csv(batch / "candidate_manifest.csv")
            if row.get("selection_status", "accepted") == "accepted"
        }
        verified_rows = read_csv(batch / "verified_markdown_manifest.csv")
        if len(verified_rows) != expected_documents:
            raise RuntimeError(
                f"Expected {expected_documents} verified rows in {batch_name}, found {len(verified_rows)}"
            )
        if (
            len(candidate_by_id) != expected_documents
            or {row["candidate_id"] for row in verified_rows} != set(candidate_by_id)
        ):
            raise RuntimeError(f"Candidate/verified manifest membership mismatch: {batch_name}")
        run_dirs = {Path(row["localized_markdown_path"]).resolve().parent.parent for row in verified_rows}
        if len(run_dirs) != 1:
            raise RuntimeError(f"Wave maps to multiple OCR run directories: {batch_name}")
        run_dir = next(iter(run_dirs))
        check_run_audits(run_dir)

        for verified in verified_rows:
            candidate = candidate_by_id[verified["candidate_id"]]
            if verified["verification_status"] != "verified_localized_markdown":
                raise RuntimeError(f"Unverified acquisition item: {verified['candidate_id']}")
            canonical = Path(verified["localized_markdown_path"]).resolve()
            if not canonical.is_file() or sha256_file(canonical) != verified["localized_markdown_sha256"]:
                raise RuntimeError(f"Canonical localized Markdown hash mismatch: {canonical}")
            primary = normalize_agent(verified["primary_specialist"])
            secondary = split_agents(verified.get("secondary_specialists", ""))
            if primary not in folders or any(agent not in folders for agent in secondary):
                raise RuntimeError(f"Unknown specialist routing for {verified['candidate_id']}")
            routed_id = f"{wave_slug}-{verified['ocr_document_id']}"
            if routed_id in wave_ids:
                raise RuntimeError(f"Duplicate routed acquisition ID: {routed_id}")
            wave_ids.add(routed_id)
            target = folders[primary] / "papers" / f"{routed_id}.md"
            page_count, image_count = route_markdown(canonical, target, routed_id)
            if page_count != int(verified["page_markers"]):
                raise RuntimeError(f"Routed page-count mismatch: {routed_id}")
            if image_count != int(verified["localized_image_references"]):
                raise RuntimeError(f"Routed image-count mismatch: {routed_id}")
            relative_target = target.relative_to(library).as_posix()
            status = verified["commercial_promotion_status"]
            retrieval = "active_cleared_cc_by" if status.startswith("cleared_") else "active_license_review"
            row = {
                "document_id": routed_id,
                "title": verified["title"],
                "authors": verified["authors"],
                "year": verified["year"],
                "primary_specialist": primary,
                "secondary_specialists": ";".join(secondary),
                "tier": verified["tier"],
                "source": candidate.get("venue") or candidate.get("lane", ""),
                "source_url": candidate.get("landing_url") or candidate.get("pdf_url", ""),
                "license": verified["license"],
                "peer_review_status": candidate.get("peer_review_status", ""),
                "relevance": candidate.get("quant_relevance", ""),
                "page_count": verified["pdf_page_count"],
                "source_pdf_sha256": verified["pdf_sha256"],
                "original_pdf_path": verified["pdf_path"],
                "markdown_path": str(target),
                "markdown_relative_path": relative_target,
                "markdown_sha256": sha256_file(target),
                "verification_status": verified["verification_status"],
                "warning_pages": "",
                "retrieval_status": retrieval,
                "exclusion_reason": "",
            }
            new_rows.append(row)
            route_rows.append(
                {
                    "acquisition_wave": wave_slug,
                    "batch": batch_name,
                    "candidate_id": verified["candidate_id"],
                    "document_id": routed_id,
                    "title": verified["title"],
                    "primary_specialist": primary,
                    "secondary_specialists": ";".join(secondary),
                    "page_count": page_count,
                    "image_count": image_count,
                    "commercial_promotion_status": status,
                    "canonical_markdown_path": str(canonical),
                    "routed_markdown_path": str(target),
                    "routed_markdown_sha256": sha256_file(target),
                }
            )

    # Replace only prior materializations of these acquisition waves; preserve the
    # original 279-paper library and all unrelated user changes.
    merged_corpus = [row for row in corpus_rows if row["document_id"] not in wave_ids] + new_rows
    merged_corpus.sort(key=lambda row: row["document_id"])
    active_corpus = [row for row in merged_corpus if row["retrieval_status"].startswith("active")]
    quarantined_corpus = [row for row in merged_corpus if not row["retrieval_status"].startswith("active")]
    active_pages = sum(int(row["page_count"]) for row in active_corpus)
    expected_cardinality = (
        EXPECTED_CORPUS_ROWS,
        EXPECTED_ACTIVE_PAPERS,
        EXPECTED_QUARANTINED_PAPERS,
        EXPECTED_ACTIVE_PAGES,
    )
    actual_cardinality = (len(merged_corpus), len(active_corpus), len(quarantined_corpus), active_pages)
    if actual_cardinality != expected_cardinality:
        raise RuntimeError(
            "Post-integration corpus cardinality mismatch: "
            f"actual(all, active, quarantined, pages)={actual_cardinality}, "
            f"expected={expected_cardinality}"
        )
    expected_overlap_total = sum(len(split_agents(row["secondary_specialists"])) for row in active_corpus)
    if expected_overlap_total != EXPECTED_SECONDARY_ROUTES:
        raise RuntimeError(
            "Post-integration corpus secondary-route mismatch: "
            f"{expected_overlap_total} != {EXPECTED_SECONDARY_ROUTES}"
        )
    write_csv(corpus_path, corpus_fields, merged_corpus)

    primary_new: dict[str, list[dict[str, str]]] = defaultdict(list)
    secondary_new: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in new_rows:
        primary_new[row["primary_specialist"]].append(row)
        for agent in split_agents(row["secondary_specialists"]):
            secondary_new[agent].append(row)

    for agent, folder in folders.items():
        primary_path = folder / "primary_index.csv"
        secondary_path = folder / "secondary_index.csv"
        primary_existing = [row for row in read_csv(primary_path) if row["document_id"] not in wave_ids]
        secondary_existing = [row for row in read_csv(secondary_path) if row["document_id"] not in wave_ids]
        primary_rows = sorted(primary_existing + primary_new[agent], key=lambda row: row["document_id"])
        secondary_rows = sorted(secondary_existing + secondary_new[agent], key=lambda row: row["document_id"])
        write_csv(primary_path, corpus_fields, primary_rows)
        write_csv(secondary_path, corpus_fields, secondary_rows)
        rebuild_primary_index(folder, title_by_agent[agent], primary_rows)

        profile_path = folder / "SPECIALIST_PROFILE.md"
        profile = profile_path.read_text(encoding="utf-8")
        profile = re.sub(r"(?m)^Active primary papers: \d+$", f"Active primary papers: {len(primary_rows)}", profile)
        profile = re.sub(r"(?m)^Secondary-routed papers: \d+$", f"Secondary-routed papers: {len(secondary_rows)}", profile)
        profile_path.write_text(profile, encoding="utf-8")

        owned = [row for row in route_rows if row["primary_specialist"] == agent]
        challenged = [row for row in route_rows if agent in split_agents(row["secondary_specialists"])]
        lines = [f"# {title_by_agent[agent]} — Acquisition Waves", ""]
        for heading, rows in (("Primary ownership", owned), ("Secondary review routes", challenged)):
            lines.extend([f"## {heading}", ""])
            if not rows:
                lines.append("- No Wave 1–5 assignment; the specialist retains its existing corpus.")
            for item in sorted(rows, key=lambda value: (value["acquisition_wave"], value["title"])):
                routed = Path(item["routed_markdown_path"])
                relative = os.path.relpath(routed, folder).replace(os.sep, "/")
                lines.append(
                    f"- [{item['title']}]({relative}) — {item['acquisition_wave']}, "
                    f"{item['page_count']} pages, {item['commercial_promotion_status']}"
                )
            lines.append("")
        (folder / "WAVE_UPGRADES.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")

    roster_path = library / "indexes/agent_roster.csv"
    roster = read_csv(roster_path)
    roster_fields = list(roster[0])
    for row in roster:
        folder = folders[row["agent_id"]]
        row["primary_papers"] = str(len(read_csv(folder / "primary_index.csv")))
        row["secondary_routes"] = str(len(read_csv(folder / "secondary_index.csv")))
    write_csv(roster_path, roster_fields, roster)

    route_fields = list(route_rows[0])
    write_csv(library / "indexes/acquisition_wave_routes.csv", route_fields, route_rows)
    overlap_total = sum(int(row["secondary_routes"]) for row in roster)
    if overlap_total != EXPECTED_SECONDARY_ROUTES:
        raise RuntimeError(
            f"Post-integration secondary-route mismatch: {overlap_total} != {EXPECTED_SECONDARY_ROUTES}"
        )
    readme = (library / "README.md").read_text(encoding="utf-8")
    readme = re.sub(r"verified \d+-document OCR corpus", f"verified {len(merged_corpus)}-document OCR corpus", readme)
    readme = re.sub(
        r"(?m)^- Active quant/mathematics papers: \d+$",
        f"- Active quant/mathematics papers: {len(active_corpus)}",
        readme,
    )
    readme = re.sub(r"(?m)^- Secondary routing relationships: \d+$", f"- Secondary routing relationships: {overlap_total}", readme)
    readme = readme.replace(
        "Formula bodies remain byte-identical to the verified OCR artifacts; categorized files are hard links.",
        "Formula and prose bodies remain byte-identical to verified OCR artifacts; acquisition-wave copies rewrite only local image paths and keep self-contained assets.",
    )
    (library / "README.md").write_text(readme, encoding="utf-8")

    audit = {
        "status": "pass",
        "specialists": 20,
        "new_primary_papers": len(new_rows),
        "new_pages": sum(int(row["page_count"]) for row in route_rows),
        "new_local_images": sum(int(row["image_count"]) for row in route_rows),
        "new_secondary_routes": sum(len(split_agents(row["secondary_specialists"])) for row in route_rows),
        "corpus_rows_after_integration": len(merged_corpus),
        "active_papers_after_integration": len(active_corpus),
        "active_pages_after_integration": active_pages,
        "quarantined_papers": len(quarantined_corpus),
        "secondary_routes_after_integration": overlap_total,
        "route_index": str(library / "indexes/acquisition_wave_routes.csv"),
    }
    (library / "indexes/acquisition_wave_integration_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
