#!/usr/bin/env python3
"""Download allowlisted open-access acquisition PDFs into an auditable staging batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from pypdf import PdfReader


DOWNLOAD_FIELDS = [
    "pdf_path", "pdf_bytes", "pdf_sha256", "pdf_page_count", "final_url",
    "content_type", "downloaded_at_utc", "download_status", "download_error",
]


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def slug(value: str, limit: int = 100) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return value[:limit].rstrip("-") or "paper"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path, retries: int, timeout: int) -> tuple[str, str]:
    headers = {"User-Agent": "QuantLab-OpenAccess-Acquisition/1.0 (research corpus; contact local operator)"}
    error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, headers=headers, timeout=timeout, stream=True, allow_redirects=True) as response:
                response.raise_for_status()
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if chunk:
                            handle.write(chunk)
                return str(response.url), str(response.headers.get("content-type") or "")
        except Exception as exc:  # noqa: BLE001 - recorded without credentials.
            error = exc
            target.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(min(2**attempt, 10))
    raise RuntimeError(f"Download failed after {retries} attempts: {type(error).__name__}: {error}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--workspace-root", default=".")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args(argv)
    workspace = Path(args.workspace_root).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    fields, rows = read_csv(manifest)
    required = {"candidate_id", "lane", "title", "pdf_url", "selection_status"}
    if not required.issubset(fields):
        raise SystemExit(f"Candidate manifest missing fields: {sorted(required - set(fields))}")
    output_fields = fields + [field for field in DOWNLOAD_FIELDS if field not in fields]
    accepted = [row for row in rows if row.get("selection_status") == "accepted"]
    if not accepted:
        raise SystemExit("No accepted candidates in manifest")

    seen_urls: set[str] = set()
    seen_hashes: dict[str, str] = {}
    for index, row in enumerate(rows, 1):
        if row.get("selection_status") != "accepted":
            row.setdefault("download_status", "not_selected")
            continue
        url = str(row.get("pdf_url") or "").strip()
        recorded_path = str(row.get("pdf_path") or "").strip()
        if row.get("download_status") == "downloaded_verified_pdf" and recorded_path:
            existing_path = workspace / recorded_path
            recorded_hash = str(row.get("pdf_sha256") or "").strip()
            if existing_path.is_file() and recorded_hash and sha256_file(existing_path) == recorded_hash:
                if url in seen_urls:
                    raise SystemExit(f"Duplicate accepted PDF URL in verified rows: {url}")
                if recorded_hash in seen_hashes:
                    raise SystemExit(f"Duplicate verified PDF content: {row['candidate_id']} and {seen_hashes[recorded_hash]}")
                seen_urls.add(url)
                seen_hashes[recorded_hash] = row["candidate_id"]
                print(f"[{index}/{len(rows)}] {row['candidate_id']} -> already_verified_pdf", flush=True)
                continue
        if not url.startswith("https://"):
            row.update({"download_status": "blocked_invalid_url", "download_error": "PDF URL must use HTTPS"})
            continue
        if url in seen_urls:
            row.update({"download_status": "blocked_duplicate_url", "download_error": "Duplicate accepted PDF URL"})
            continue
        seen_urls.add(url)
        lane = slug(row["lane"], 64)
        candidate = slug(row["candidate_id"], 40)
        target = manifest.parent / "pdfs" / lane / f"{candidate}__{slug(row['title'])}.pdf"
        try:
            final_url, content_type = download(url, target, args.retries, args.timeout)
            with target.open("rb") as handle:
                header = handle.read(5)
            if target.stat().st_size < 1024 or header != b"%PDF-":
                raise RuntimeError("Downloaded artifact is not a valid PDF header")
            page_count = len(PdfReader(str(target), strict=False).pages)
            if page_count < 1:
                raise RuntimeError("PDF contains no pages")
            digest = sha256_file(target)
            duplicate = seen_hashes.get(digest)
            if duplicate:
                raise RuntimeError(f"Duplicate PDF content of candidate {duplicate}")
            seen_hashes[digest] = row["candidate_id"]
            row.update(
                {
                    "pdf_path": str(target.relative_to(workspace)),
                    "pdf_bytes": target.stat().st_size,
                    "pdf_sha256": digest,
                    "pdf_page_count": page_count,
                    "final_url": final_url,
                    "content_type": content_type,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "download_status": "downloaded_verified_pdf",
                    "download_error": "",
                }
            )
        except Exception as exc:  # noqa: BLE001 - every failure stays in the manifest.
            target.unlink(missing_ok=True)
            row.update({"download_status": "download_failed", "download_error": f"{type(exc).__name__}: {exc}"})
        write_csv(manifest, output_fields, rows)
        print(f"[{index}/{len(rows)}] {row['candidate_id']} -> {row.get('download_status')}", flush=True)
    write_csv(manifest, output_fields, rows)
    ok = sum(row.get("download_status") == "downloaded_verified_pdf" for row in rows)
    failed = sum(row.get("download_status") == "download_failed" for row in rows)
    print(f"Acquisition download complete: {ok} verified PDFs, {failed} failures")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
