#!/usr/bin/env python3
"""Download remote OCR image references and emit durable localized Markdown copies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image


IMAGE_RE = re.compile(r"!\[([^\]]*)\]\((https?://[^)]+)\)")
FIELDS = [
    "document_id", "image_index", "alt_text", "original_url", "local_path",
    "sha256", "bytes", "width", "height", "format", "status", "error",
]


def extension(url: str, image_format: str) -> str:
    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".tif", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix
    return {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp", "GIF": ".gif", "TIFF": ".tiff"}.get(image_format, ".img")


def fetch(job: dict[str, object], assets_dir: Path, timeout: int) -> dict[str, object]:
    url = str(job["original_url"])
    row = dict(job)
    try:
        response = requests.get(url, timeout=timeout, headers={"User-Agent": "QuantLab-OCR-Image-Archiver/1.0"})
        response.raise_for_status()
        data = response.content
        if len(data) < 32:
            raise RuntimeError("image payload is unexpectedly small")
        digest = hashlib.sha256(data).hexdigest()
        temp = assets_dir / ".tmp" / f"{digest}.image"
        temp.parent.mkdir(parents=True, exist_ok=True)
        temp.write_bytes(data)
        with Image.open(temp) as image:
            image.verify()
        with Image.open(temp) as image:
            width, height = image.size
            image_format = str(image.format or "")
        ext = extension(url, image_format)
        target = assets_dir / str(job["document_id"]) / f"img-{int(job['image_index']):04d}__{digest[:12]}{ext}"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        temp.unlink(missing_ok=True)
        row.update(
            {
                "local_path": str(target.resolve()), "sha256": digest, "bytes": len(data),
                "width": width, "height": height, "format": image_format,
                "status": "downloaded_verified_image", "error": "",
            }
        )
    except Exception as exc:  # noqa: BLE001 - every failure is retained in the manifest.
        row.update({"status": "download_failed", "error": f"{type(exc).__name__}: {exc}"})
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    markdown_files = sorted(input_dir.glob("*.md"))
    jobs: list[dict[str, object]] = []
    source_text: dict[str, str] = {}
    for path in markdown_files:
        text = path.read_text(encoding="utf-8")
        source_text[path.stem] = text
        for index, match in enumerate(IMAGE_RE.finditer(text), 1):
            jobs.append(
                {
                    "document_id": path.stem, "image_index": index,
                    "alt_text": match.group(1), "original_url": match.group(2),
                }
            )
    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(fetch, job, assets_dir, args.timeout) for job in jobs]
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            results.append(row)
            if index % 25 == 0 or row["status"] != "downloaded_verified_image":
                print(f"[{index}/{len(futures)}] {row['document_id']} image {row['image_index']} -> {row['status']}", flush=True)
    results.sort(key=lambda row: (str(row["document_id"]), int(row["image_index"])))
    failures = [row for row in results if row["status"] != "downloaded_verified_image"]
    manifest_path = output_dir.parent / "localized_image_manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in FIELDS} for row in results)
    if failures:
        raise SystemExit(f"Image localization blocked: {len(failures)} of {len(results)} downloads failed; see {manifest_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows_by_doc: dict[str, list[dict[str, object]]] = {}
    for row in results:
        rows_by_doc.setdefault(str(row["document_id"]), []).append(row)
    for document_id, text in source_text.items():
        replacements = iter(rows_by_doc.get(document_id, []))
        def replace(match: re.Match[str]) -> str:
            row = next(replacements)
            relative = Path(str(row["local_path"])).relative_to(output_dir.parent)
            return f"![{match.group(1)}](../{relative.as_posix()})"
        localized = IMAGE_RE.sub(replace, text)
        if IMAGE_RE.search(localized):
            raise RuntimeError(f"Remote image URL remains in {document_id}")
        (output_dir / f"{document_id}.md").write_text(localized, encoding="utf-8")
    print(f"Localized {len(results)} images across {len(markdown_files)} Markdown files; manifest: {manifest_path}")


if __name__ == "__main__":
    main()
