#!/usr/bin/env python3
"""Z.ai GLM-OCR PDF-to-Markdown pipeline.

Default mode is provider-safe transport chunking:

- local work only inventories, deduplicates, and packages PDF page ranges;
- every OCR/extraction call is sent to Z.ai GLM-OCR;
- raw Z.ai JSON and Markdown are retained per chunk;
- successful chunks are assembled into one final Markdown file per book.

Live OCR requires `ZAI_API_KEY` or `Z_AI_API_KEY`.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import tempfile
import unicodedata
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL.*")

import requests
import pdfplumber
from pypdf import PdfReader, PdfWriter


DEFAULT_ENDPOINT = "https://api.z.ai/api/paas/v4/layout_parsing"
DEFAULT_MODEL = "glm-ocr"
DEFAULT_INPUT_DIR = "New_Pdf_folders_for_markdown"
DEFAULT_OUTPUT_ROOT = "quant_lab/artifacts/zai_ocr"
WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
QUALITY_PROFILES = {"balanced", "optimal", "math-precision"}
TRANSPORT_MODES = {"chunked", "full"}
SPLITTERS = {"hybrid", "poppler", "pypdf"}

logging.getLogger("pypdf").setLevel(logging.ERROR)


@dataclass(frozen=True)
class DocumentRecord:
    document_id: str
    input_path: str
    file_name: str
    mime_type: str
    input_sha256: str
    size_bytes: int
    page_count: int | None
    duplicate_of: str | None
    status: str
    error: str | None = None


@dataclass(frozen=True)
class ChunkRecord:
    document_id: str
    chunk_id: str
    input_path: str
    chunk_pdf_path: str
    input_kind: str
    raw_response_path: str
    text_path: str
    start_page: int
    end_page: int
    expected_pages: int
    size_bytes: int
    status: str
    error: str | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_run_id(prefix: str = "ZAI-OCR") -> str:
    return f"{prefix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def slugify(value: str, max_len: int = 96) -> str:
    value = value.strip()
    value = re.sub(r"\s+", " ", value)
    value = re.sub(r"\s+--\s+\([^)]*\)\s*$", "", value)
    value = re.sub(r"\.(pdf|epub|djvu|azw3)$", "", value, flags=re.I)
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    value = re.sub(r"-+", "-", value).strip("-").lower()
    return value[:max_len].strip("-") or "document"


def unique_document_id(filename: str, seen: set[str]) -> str:
    base = slugify(filename)
    candidate = base
    suffix = 2
    while candidate in seen:
        candidate = f"{base}-{suffix}"
        suffix += 1
    seen.add(candidate)
    return candidate


def count_pdf_pages(path: Path) -> tuple[int | None, str | None]:
    try:
        reader = PdfReader(str(path))
        return len(reader.pages), None
    except Exception as exc:  # noqa: BLE001 - preflight should record bad PDFs.
        return None, f"{type(exc).__name__}: {exc}"


def input_paths(
    input_dir: Path,
    *,
    recursive: bool = False,
    input_manifest: Path | None = None,
) -> list[Path]:
    if input_manifest is not None:
        if not input_manifest.exists():
            raise FileNotFoundError(f"Input manifest not found: {input_manifest}")
        paths: list[Path] = []
        seen_paths: set[Path] = set()
        with input_manifest.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames or "pdf_path" not in reader.fieldnames:
                raise ValueError("Input manifest must contain a pdf_path column.")
            for row in reader:
                raw_path = str(row.get("pdf_path") or "").strip()
                if not raw_path:
                    continue
                path = Path(raw_path).expanduser()
                if not path.is_absolute():
                    path = (WORKSPACE_ROOT / path).resolve()
                if path not in seen_paths:
                    seen_paths.add(path)
                    paths.append(path)
        return paths

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    return sorted(
        [path.resolve() for path in iterator if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda path: str(path).lower(),
    )


def build_inventory(
    input_dir: Path,
    *,
    recursive: bool = False,
    input_manifest: Path | None = None,
) -> list[DocumentRecord]:

    seen_ids: set[str] = set()
    sha_to_doc: dict[str, str] = {}
    records: list[DocumentRecord] = []
    for path in input_paths(input_dir, recursive=recursive, input_manifest=input_manifest):
        if not path.exists():
            records.append(
                DocumentRecord(
                    document_id=unique_document_id(path.name, seen_ids),
                    input_path=str(path),
                    file_name=path.name,
                    mime_type="application/pdf",
                    input_sha256="",
                    size_bytes=0,
                    page_count=None,
                    duplicate_of=None,
                    status="missing_file",
                    error="Manifest path does not exist.",
                )
            )
            continue
        digest = sha256_file(path)
        page_count, page_error = count_pdf_pages(path)
        document_id = unique_document_id(path.name, seen_ids)
        duplicate_of = sha_to_doc.get(digest)
        if duplicate_of is None:
            sha_to_doc[digest] = document_id

        if page_error:
            status = "page_count_error"
            error = page_error
        elif duplicate_of:
            status = "duplicate"
            error = None
        else:
            status = "ready"
            error = None

        records.append(
            DocumentRecord(
                document_id=document_id,
                input_path=str(path.resolve()),
                file_name=path.name,
                mime_type="application/pdf",
                input_sha256=digest,
                size_bytes=path.stat().st_size,
                page_count=page_count,
                duplicate_of=duplicate_of,
                status=status,
                error=error,
            )
        )
    return records


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def encode_file(path: Path, data_uri: bool) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    if data_uri:
        return f"data:{mime_type_for_path(path)};base64,{encoded}"
    return encoded


def mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/pdf"


def make_request_id(run_id: str, chunk_id: str) -> str:
    digest = hashlib.sha1(f"{run_id}:{chunk_id}".encode("utf-8")).hexdigest()[:24]
    return f"ocr-{digest}"


def env_api_key() -> str | None:
    return os.environ.get("ZAI_API_KEY") or os.environ.get("Z_AI_API_KEY")


def provider_limit_error(page_count: int | None, size_bytes: int, max_pages: int, max_mb: float) -> str | None:
    max_bytes = int(max_mb * 1024 * 1024)
    errors: list[str] = []
    if page_count is not None and page_count > max_pages:
        errors.append(f"page_count={page_count} exceeds provider max_pages={max_pages}")
    if size_bytes > max_bytes:
        errors.append(f"size_mb={size_bytes / 1024 / 1024:.2f} exceeds provider max_mb={max_mb}")
    return "; ".join(errors) if errors else None


def chunk_page_ranges(page_count: int, pages_per_chunk: int) -> list[tuple[int, int]]:
    return [
        (start, min(page_count, start + pages_per_chunk - 1))
        for start in range(1, page_count + 1, pages_per_chunk)
    ]


def split_pdf_range(source_pdf: Path, destination_pdf: Path, start_page: int, end_page: int) -> None:
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    for page_index in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page_index])
    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    with destination_pdf.open("wb") as fh:
        writer.write(fh)


def split_pdf_range_pypdf_subprocess(
    source_pdf: Path,
    destination_pdf: Path,
    start_page: int,
    end_page: int,
    timeout_seconds: int,
) -> bool:
    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    code = """
import sys
from pathlib import Path
from pypdf import PdfReader, PdfWriter

source_pdf, destination_pdf = sys.argv[1], sys.argv[2]
start_page, end_page = int(sys.argv[3]), int(sys.argv[4])
reader = PdfReader(source_pdf)
writer = PdfWriter()
for page_index in range(start_page - 1, end_page):
    writer.add_page(reader.pages[page_index])
Path(destination_pdf).parent.mkdir(parents=True, exist_ok=True)
with open(destination_pdf, "wb") as fh:
    writer.write(fh)
"""
    try:
        subprocess.run(
            [sys.executable, "-c", code, str(source_pdf), str(destination_pdf), str(start_page), str(end_page)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout_seconds,
        )
        return destination_pdf.exists() and destination_pdf.stat().st_size > 0
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        destination_pdf.unlink(missing_ok=True)
        return False


def split_pdf_range_poppler(
    source_pdf: Path,
    destination_pdf: Path,
    start_page: int,
    end_page: int,
    timeout_seconds: int,
) -> bool:
    pdfseparate = shutil.which("pdfseparate")
    pdfunite = shutil.which("pdfunite")
    if not pdfseparate or not pdfunite:
        return False

    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zai_ocr_pages_") as temp_dir:
        temp_path = Path(temp_dir)
        page_pattern = temp_path / "page-%04d.pdf"
        separate_cmd = [
            pdfseparate,
            "-f",
            str(start_page),
            "-l",
            str(end_page),
            str(source_pdf),
            str(page_pattern),
        ]
        try:
            subprocess.run(separate_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds)
            page_files = sorted(temp_path.glob("page-*.pdf"))
            if not page_files:
                return False
            unite_cmd = [pdfunite, *[str(path) for path in page_files], str(destination_pdf)]
            subprocess.run(unite_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds)
            return destination_pdf.exists() and destination_pdf.stat().st_size > 0
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            destination_pdf.unlink(missing_ok=True)
            return False


def prepare_poppler_page_cache(source_pdf: Path, page_cache_dir: Path, page_count: int, force: bool) -> bool:
    pdfseparate = shutil.which("pdfseparate")
    if not pdfseparate:
        return False
    page_cache_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(page_cache_dir.glob("page-*.pdf"))
    if not force and len(existing) >= page_count:
        return True
    if force and page_cache_dir.exists():
        shutil.rmtree(page_cache_dir)
        page_cache_dir.mkdir(parents=True, exist_ok=True)
    page_pattern = page_cache_dir / "page-%04d.pdf"
    cmd = [
        pdfseparate,
        "-f",
        "1",
        "-l",
        str(page_count),
        str(source_pdf),
        str(page_pattern),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return False
    return len(list(page_cache_dir.glob("page-*.pdf"))) >= page_count


def unite_poppler_page_range(page_cache_dir: Path, destination_pdf: Path, start_page: int, end_page: int) -> bool:
    pdfunite = shutil.which("pdfunite")
    if not pdfunite:
        return False
    page_files = [page_cache_dir / f"page-{page_number:04d}.pdf" for page_number in range(start_page, end_page + 1)]
    if not all(path.exists() for path in page_files):
        return False
    destination_pdf.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [pdfunite, *[str(path) for path in page_files], str(destination_pdf)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return destination_pdf.exists() and destination_pdf.stat().st_size > 0
    except subprocess.CalledProcessError:
        destination_pdf.unlink(missing_ok=True)
        return False


def render_page_image(source_pdf: Path, destination_png: Path, page_number: int, dpi: int) -> bool:
    pdftoppm = shutil.which("pdftoppm")
    if not pdftoppm:
        return False
    destination_png.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="zai_ocr_render_") as temp_dir:
        temp_path = Path(temp_dir)
        output_prefix = temp_path / "page"
        cmd = [
            pdftoppm,
            "-f",
            str(page_number),
            "-l",
            str(page_number),
            "-r",
            str(dpi),
            "-png",
            str(source_pdf),
            str(output_prefix),
        ]
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=180)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
        outputs = sorted(temp_path.glob("page*.png"))
        if not outputs:
            return False
        shutil.move(str(outputs[0]), str(destination_png))
        return destination_png.exists() and destination_png.stat().st_size > 0


def chunk_file_name(start_page: int, end_page: int) -> str:
    return f"p{start_page:04d}-{end_page:04d}"


def prepare_full_chunk(document: DocumentRecord, run_dir: Path, args: argparse.Namespace) -> list[ChunkRecord]:
    error = provider_limit_error(document.page_count, document.size_bytes, args.provider_max_pages, args.provider_max_mb)
    if error and not args.ignore_provider_limits:
        return [
            ChunkRecord(
                document_id=document.document_id,
                chunk_id=f"{document.document_id}__full",
                input_path=document.input_path,
                chunk_pdf_path=document.input_path,
                input_kind="pdf",
                raw_response_path="",
                text_path="",
                start_page=1,
                end_page=document.page_count or 0,
                expected_pages=document.page_count or 0,
                size_bytes=document.size_bytes,
                status="blocked_by_zai_limit",
                error=error,
            )
        ]
    raw_path = run_dir / "raw_responses" / document.document_id / "full.json"
    text_path = run_dir / "chunk_markdown" / document.document_id / "full.md"
    return [
        ChunkRecord(
            document_id=document.document_id,
            chunk_id=f"{document.document_id}__full",
            input_path=document.input_path,
            chunk_pdf_path=document.input_path,
            input_kind="pdf",
            raw_response_path=str(raw_path.resolve()),
            text_path=str(text_path.resolve()),
            start_page=1,
            end_page=document.page_count or 0,
            expected_pages=document.page_count or 0,
            size_bytes=document.size_bytes,
            status="ready",
        )
    ]


def prepare_chunked_document(document: DocumentRecord, run_dir: Path, args: argparse.Namespace) -> list[ChunkRecord]:
    if document.page_count is None or document.page_count < 1:
        return [
            ChunkRecord(
                document_id=document.document_id,
                chunk_id=f"{document.document_id}__unknown-pages",
                input_path=document.input_path,
                chunk_pdf_path="",
                input_kind="pdf",
                raw_response_path="",
                text_path="",
                start_page=0,
                end_page=0,
                expected_pages=0,
                size_bytes=0,
                status="page_count_error",
                error="Cannot transport-chunk a PDF without a page count.",
            )
        ]

    source_pdf = Path(document.input_path)
    chunks: list[ChunkRecord] = []
    max_bytes = int(args.provider_max_mb * 1024 * 1024)
    page_cache_dir = run_dir / "page_cache" / document.document_id
    use_poppler_cache = False
    force_image_transport = False
    if args.splitter == "poppler":
        use_poppler_cache = prepare_poppler_page_cache(source_pdf, page_cache_dir, document.page_count, args.force)
        if not use_poppler_cache:
            return [
                ChunkRecord(
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}__splitter-error",
                    input_path=document.input_path,
                    chunk_pdf_path="",
                    input_kind="pdf",
                    raw_response_path="",
                    text_path="",
                    start_page=0,
                    end_page=0,
                    expected_pages=0,
                    size_bytes=0,
                    status="splitter_error",
                    error="Poppler page-cache preparation failed.",
                )
            ]

    def image_path_for_page(page_number: int) -> Path:
        return run_dir / "chunks_image" / document.document_id / f"p{page_number:04d}.png"

    def existing_image_range(start_page: int, end_page: int) -> bool:
        if args.force or not args.image_fallback:
            return False
        return all(image_path_for_page(page_number).exists() for page_number in range(start_page, end_page + 1))

    def add_image_pages(start_page: int, end_page: int, reason: str) -> None:
        for page_number in range(start_page, end_page + 1):
            name = f"p{page_number:04d}"
            image_path = image_path_for_page(page_number)
            raw_path = run_dir / "raw_responses" / document.document_id / f"{name}.json"
            text_path = run_dir / "chunk_markdown" / document.document_id / f"{name}.md"
            if args.force or not image_path.exists():
                if not render_page_image(source_pdf, image_path, page_number, args.image_dpi):
                    chunks.append(
                        ChunkRecord(
                            document_id=document.document_id,
                            chunk_id=f"{document.document_id}__{name}",
                            input_path=document.input_path,
                            chunk_pdf_path=str(image_path.resolve()),
                            input_kind="image",
                            raw_response_path=str(raw_path.resolve()),
                            text_path=str(text_path.resolve()),
                            start_page=page_number,
                            end_page=page_number,
                            expected_pages=1,
                            size_bytes=0,
                            status="splitter_error",
                            error=f"{reason} Page image rendering failed.",
                        )
                    )
                    continue
            size_bytes = image_path.stat().st_size
            max_image_bytes = int(args.provider_max_image_mb * 1024 * 1024)
            limit_error = ""
            if size_bytes > max_image_bytes:
                limit_error = (
                    f"image_size_mb={size_bytes / 1024 / 1024:.2f} "
                    f"exceeds provider max_image_mb={args.provider_max_image_mb}"
                )
            chunks.append(
                ChunkRecord(
                    document_id=document.document_id,
                    chunk_id=f"{document.document_id}__{name}",
                    input_path=document.input_path,
                    chunk_pdf_path=str(image_path.resolve()),
                    input_kind="image",
                    raw_response_path=str(raw_path.resolve()),
                    text_path=str(text_path.resolve()),
                    start_page=page_number,
                    end_page=page_number,
                    expected_pages=1,
                    size_bytes=size_bytes,
                    status="blocked_by_zai_limit" if limit_error and not args.ignore_provider_limits else "ready",
                    error=limit_error or reason,
                )
            )

    def add_range(start_page: int, end_page: int) -> None:
        nonlocal use_poppler_cache, force_image_transport
        name = chunk_file_name(start_page, end_page)
        chunk_pdf = run_dir / "chunks_pdf" / document.document_id / f"{name}.pdf"
        raw_path = run_dir / "raw_responses" / document.document_id / f"{name}.json"
        text_path = run_dir / "chunk_markdown" / document.document_id / f"{name}.md"

        if force_image_transport:
            add_image_pages(start_page, end_page, "Prior PDF splitting failure for this document; using page-image transport.")
            return
        if existing_image_range(start_page, end_page):
            force_image_transport = True
            add_image_pages(start_page, end_page, "Reusing existing page-image transport cache.")
            return

        if args.force or not chunk_pdf.exists():
            if use_poppler_cache:
                if not unite_poppler_page_range(page_cache_dir, chunk_pdf, start_page, end_page):
                    chunks.append(
                        ChunkRecord(
                            document_id=document.document_id,
                            chunk_id=f"{document.document_id}__{name}",
                            input_path=document.input_path,
                            chunk_pdf_path=str(chunk_pdf.resolve()),
                            input_kind="pdf",
                            raw_response_path=str(raw_path.resolve()),
                            text_path=str(text_path.resolve()),
                            start_page=start_page,
                            end_page=end_page,
                            expected_pages=end_page - start_page + 1,
                            size_bytes=0,
                            status="splitter_error",
                            error="Poppler page-range assembly failed.",
                        )
                    )
                    return
            elif args.splitter == "hybrid":
                if not split_pdf_range_pypdf_subprocess(
                    source_pdf,
                    chunk_pdf,
                    start_page,
                    end_page,
                    args.pypdf_split_timeout_seconds,
                ):
                    if not split_pdf_range_poppler(
                        source_pdf,
                        chunk_pdf,
                        start_page,
                        end_page,
                        args.poppler_range_timeout_seconds,
                    ):
                        if args.image_fallback:
                            force_image_transport = True
                            add_image_pages(start_page, end_page, "Hybrid PDF splitting failed; using page-image transport.")
                            return
                        chunks.append(
                            ChunkRecord(
                                document_id=document.document_id,
                                chunk_id=f"{document.document_id}__{name}",
                                input_path=document.input_path,
                                chunk_pdf_path=str(chunk_pdf.resolve()),
                                input_kind="pdf",
                                raw_response_path=str(raw_path.resolve()),
                                text_path=str(text_path.resolve()),
                                start_page=start_page,
                                end_page=end_page,
                                expected_pages=end_page - start_page + 1,
                                size_bytes=0,
                                status="splitter_error",
                                error="Hybrid splitter failed with both pypdf subprocess and Poppler range fallback.",
                            )
                        )
                        return
            else:
                split_pdf_range(source_pdf, chunk_pdf, start_page, end_page)

        size_bytes = chunk_pdf.stat().st_size
        expected_pages = end_page - start_page + 1
        if size_bytes > max_bytes and expected_pages > 1:
            midpoint = (start_page + end_page) // 2
            if args.force:
                chunk_pdf.unlink(missing_ok=True)
            add_range(start_page, midpoint)
            add_range(midpoint + 1, end_page)
            return

        limit_error = provider_limit_error(expected_pages, size_bytes, args.provider_max_pages, args.provider_max_mb)
        chunks.append(
            ChunkRecord(
                document_id=document.document_id,
                chunk_id=f"{document.document_id}__{name}",
                input_path=document.input_path,
                chunk_pdf_path=str(chunk_pdf.resolve()),
                input_kind="pdf",
                raw_response_path=str(raw_path.resolve()),
                text_path=str(text_path.resolve()),
                start_page=start_page,
                end_page=end_page,
                expected_pages=expected_pages,
                size_bytes=size_bytes,
                status="blocked_by_zai_limit" if limit_error and not args.ignore_provider_limits else "ready",
                error=limit_error,
            )
        )

    for start_page, end_page in chunk_page_ranges(document.page_count, args.pages_per_chunk):
        add_range(start_page, end_page)
    if use_poppler_cache and args.clean_page_cache:
        shutil.rmtree(page_cache_dir, ignore_errors=True)
    return chunks


def estimate_document_plan(document: DocumentRecord, args: argparse.Namespace) -> dict[str, Any]:
    row = asdict(document)
    row["transport_mode"] = args.transport_mode
    row["planned_chunks_estimate"] = ""
    row["planned_status"] = document.status
    row["provider_limit_error"] = ""
    if document.status != "ready":
        return row

    if args.transport_mode == "full":
        limit_error = provider_limit_error(document.page_count, document.size_bytes, args.provider_max_pages, args.provider_max_mb)
        row["planned_chunks_estimate"] = 1
        row["provider_limit_error"] = limit_error or ""
        row["planned_status"] = "ready_for_ocr" if not limit_error or args.ignore_provider_limits else "blocked_by_zai_limit"
        return row

    if document.page_count is None:
        row["planned_status"] = "page_count_error"
        return row
    row["planned_chunks_estimate"] = len(chunk_page_ranges(document.page_count, args.pages_per_chunk))
    row["planned_status"] = "ready_for_chunked_ocr"
    return row


def prepare_chunks(documents: list[DocumentRecord], run_dir: Path, args: argparse.Namespace) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for document in documents:
        if args.transport_mode == "full":
            chunks.extend(prepare_full_chunk(document, run_dir, args))
        else:
            chunks.extend(prepare_chunked_document(document, run_dir, args))
    if args.limit_chunks is not None:
        chunks = chunks[: args.limit_chunks]
    return chunks


def intervals_cover_page_count(intervals: list[tuple[int, int]], page_count: int) -> bool:
    if page_count < 1 or not intervals:
        return False
    merged_end = 0
    for start_page, end_page in sorted(intervals):
        if start_page > merged_end + 1:
            return False
        merged_end = max(merged_end, end_page)
        if merged_end >= page_count:
            return True
    return False


def completed_document_ids_from_artifacts(output_root: Path, documents_by_id: dict[str, DocumentRecord]) -> set[str]:
    good_statuses = {"ok", "cached", "warning"}
    completed: set[str] = set()
    for manifest_path in sorted(output_root.glob("*/ocr_manifest.csv")):
        plan_path = manifest_path.parent / "ocr_plan.csv"
        prior_hashes: dict[str, str] = {}
        if plan_path.exists():
            try:
                with plan_path.open(newline="", encoding="utf-8") as plan_fh:
                    prior_hashes = {
                        str(row.get("document_id") or ""): str(row.get("input_sha256") or "")
                        for row in csv.DictReader(plan_fh)
                    }
            except OSError:
                prior_hashes = {}
        by_doc: dict[str, list[tuple[int, int]]] = {}
        bad_docs: set[str] = set()
        try:
            with manifest_path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    document_id = str(row.get("document_id") or "")
                    if document_id not in documents_by_id:
                        continue
                    status = str(row.get("status") or "")
                    if status not in good_statuses:
                        bad_docs.add(document_id)
                        continue
                    try:
                        start_page = int(row.get("start_page") or 0)
                        end_page = int(row.get("end_page") or 0)
                    except ValueError:
                        bad_docs.add(document_id)
                        continue
                    if start_page > 0 and end_page >= start_page:
                        by_doc.setdefault(document_id, []).append((start_page, end_page))
        except OSError:
            continue

        for document_id, intervals in by_doc.items():
            document = documents_by_id[document_id]
            if document_id in bad_docs or document.page_count is None:
                continue
            if not prior_hashes.get(document_id) or prior_hashes[document_id] != document.input_sha256:
                continue
            if intervals_cover_page_count(intervals, document.page_count):
                completed.add(document_id)
    return completed


def layout_label_counts(response: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for page in response.get("layout_details") or []:
        if not isinstance(page, list):
            continue
        for element in page:
            if not isinstance(element, dict):
                continue
            label = str(element.get("label", "unknown"))
            counts[label] = counts.get(label, 0) + 1
    return counts


MATH_FONT_MARKERS = (
    "math",
    "cmmi",
    "cmsy",
    "cmex",
    "msam",
    "msbm",
    "stix",
    "symbol",
    "lmmi",
    "lmsy",
    "txmi",
    "ntxmi",
    "mtmi",
    "eufm",
    "rsfs",
)
MATH_SYMBOL_PATTERN = re.compile(
    r"[∀-⋿⟀-⟿←-⇿∑∏∫√∞≈≠≤≥⊂⊃∈∉∪∩⊗⊕⊥∂∇]"
)
GREEK_PATTERN = re.compile(r"[Α-Ͽ]")
EQUATION_LABEL_PATTERN = re.compile(r"(?<!\w)\((?:[A-Z]\.)?\d+(?:\.\d+){0,3}[a-z]?\)")
INLINE_MATH_PATTERN = re.compile(r"(?<!\\)\$(?!\$)(.+?)(?<!\\)\$(?!\$)", re.DOTALL)
THEOREM_PATTERN = re.compile(
    r"\b(?:theorem|lemma|proposition|corollary|definition|assumption|proof)\b",
    re.IGNORECASE,
)


def source_page_math_risk(document: DocumentRecord) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with pdfplumber.open(document.input_path) as pdf:
            for page_number, page in enumerate(pdf.pages, 1):
                chars = page.chars or []
                text = page.extract_text() or ""
                total_chars = len(chars)
                math_font_chars = sum(
                    1
                    for char in chars
                    if any(marker in str(char.get("fontname") or "").lower() for marker in MATH_FONT_MARKERS)
                )
                math_symbols = len(MATH_SYMBOL_PATTERN.findall(text)) + len(GREEK_PATTERN.findall(text))
                equation_labels = sorted(set(EQUATION_LABEL_PATTERN.findall(text)))
                display_equation_labels = list(
                    dict.fromkeys(
                        str(word.get("text") or "")
                        for word in (page.extract_words() or [])
                        if EQUATION_LABEL_PATTERN.fullmatch(str(word.get("text") or ""))
                        and float(word.get("x0") or 0) >= float(page.width) * 0.68
                    )
                )
                theorem_headings = len(THEOREM_PATTERN.findall(text))
                math_font_ratio = math_font_chars / max(total_chars, 1)
                score = 0
                if math_font_chars >= 4 or math_font_ratio >= 0.01:
                    score += 2
                if math_symbols >= 3:
                    score += 1
                if display_equation_labels:
                    score += 2
                if theorem_headings:
                    score += 1
                rows.append(
                    {
                        "document_id": document.document_id,
                        "input_path": document.input_path,
                        "page_number": page_number,
                        "source_chars": len(text),
                        "source_glyphs": total_chars,
                        "math_font_chars": math_font_chars,
                        "math_font_ratio": round(math_font_ratio, 5),
                        "math_symbols": math_symbols,
                        "equation_labels": ";".join(equation_labels),
                        "display_equation_labels": ";".join(display_equation_labels),
                        "theorem_headings": theorem_headings,
                        "math_risk_score": score,
                        "formula_expected": score >= 2,
                        "risk_status": "high" if score >= 4 else "medium" if score >= 2 else "low",
                        "error": "",
                    }
                )
    except Exception as exc:  # noqa: BLE001 - source-risk failure is recorded per document.
        rows.append(
            {
                "document_id": document.document_id,
                "input_path": document.input_path,
                "page_number": 0,
                "source_chars": 0,
                "source_glyphs": 0,
                "math_font_chars": 0,
                "math_font_ratio": 0,
                "math_symbols": 0,
                "equation_labels": "",
                "display_equation_labels": "",
                "theorem_headings": 0,
                "math_risk_score": 0,
                "formula_expected": False,
                "risk_status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    return rows


def build_source_math_risk(documents: list[DocumentRecord]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        if document.status == "ready":
            rows.extend(source_page_math_risk(document))
    return rows


def formula_syntax_issues(formula: str) -> list[str]:
    issues: list[str] = []
    if not formula.strip():
        return ["empty_formula"]
    unescaped = re.sub(r"\\[{}$]", "", formula)
    if unescaped.count("{") != unescaped.count("}"):
        issues.append("unbalanced_braces")
    if unescaped.count("$") % 2:
        issues.append("unbalanced_dollars")
    environment_stack: list[str] = []
    environment_error = False
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", formula):
        kind, name = match.groups()
        if kind == "begin":
            environment_stack.append(name)
        elif not environment_stack or environment_stack.pop() != name:
            environment_error = True
            break
    if environment_stack or environment_error:
        issues.append("unbalanced_environments")
    if len(re.findall(r"\\left(?![A-Za-z])", formula)) != len(re.findall(r"\\right(?![A-Za-z])", formula)):
        issues.append("unbalanced_left_right")
    for opening, closing, label in [("\\(", "\\)", "inline_math"), ("\\[", "\\]", "display_math")]:
        if formula.count(opening) != formula.count(closing):
            issues.append(f"unbalanced_{label}_delimiters")
    return issues


def normalized_formula_content(value: str) -> str:
    value = unicodedata.normalize("NFKC", value)
    for delimiter in ("$$", "\\[", "\\]", "\\(", "\\)"):
        value = value.replace(delimiter, "")
    value = value.replace("$", "")
    value = re.sub(r"\s+", "", value)
    return value


def page_layout_rows(chunk: ChunkRecord, response: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pages = response.get("layout_details") or []
    for page_offset in range(chunk.expected_pages):
        elements = pages[page_offset] if page_offset < len(pages) and isinstance(pages[page_offset], list) else []
        counts: dict[str, int] = {}
        formulas: list[str] = []
        content_parts: list[str] = []
        for element in elements:
            if not isinstance(element, dict):
                continue
            label = str(element.get("label") or "unknown")
            native_label = str(element.get("native_label") or "")
            content = str(element.get("content") or "")
            counts[label] = counts.get(label, 0) + 1
            content_parts.append(content)
            if label == "formula" or "formula" in native_label.lower():
                formulas.append(content)
        syntax_issues = sorted({issue for formula in formulas for issue in formula_syntax_issues(formula)})
        page_content = "\n".join(content_parts)
        inline_math_spans = [match.group(1) for match in INLINE_MATH_PATTERN.finditer(page_content)]
        rows.append(
            {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "page_number": chunk.start_page + page_offset,
                "layout_page_present": page_offset < len(pages),
                "layout_elements": len(elements),
                "text_blocks": counts.get("text", 0),
                "formula_blocks": counts.get("formula", 0),
                "detected_formula_blocks": len(formulas),
                "detected_inline_math_spans": len(inline_math_spans),
                "table_blocks": counts.get("table", 0),
                "image_blocks": counts.get("image", 0),
                "layout_content_chars": len(page_content),
                "extracted_equation_labels": ";".join(sorted(set(EQUATION_LABEL_PATTERN.findall(page_content)))),
                "formula_syntax_issues": ";".join(syntax_issues),
                "formula_text": "\n\n".join(formulas),
                "_formula_contents": formulas,
                "formula_hashes": ";".join(
                    hashlib.sha256(normalized_formula_content(formula).encode("utf-8")).hexdigest()
                    for formula in formulas
                    if formula.strip()
                ),
            }
        )
    return rows


def build_math_precision_page_qa(
    run_dir: Path,
    documents: list[DocumentRecord],
    chunk_rows: list[dict[str, Any]],
    source_risk_rows: list[dict[str, Any]],
    recommended_dpi: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source_by_page = {
        (str(row["document_id"]), int(row["page_number"])): row
        for row in source_risk_rows
        if int(row.get("page_number") or 0) > 0
    }
    documents_by_id = {document.document_id: document for document in documents}
    qa_rows: list[dict[str, Any]] = []
    retry_rows: list[dict[str, Any]] = []
    good_statuses = {"ok", "warning", "cached"}
    for chunk_row in chunk_rows:
        if chunk_row.get("status") not in good_statuses or not chunk_row.get("raw_response_path"):
            continue
        raw_path = Path(str(chunk_row["raw_response_path"]))
        if not raw_path.exists():
            continue
        response = json.loads(raw_path.read_text(encoding="utf-8"))
        normalized_markdown = normalized_formula_content(str(response.get("md_results") or ""))
        chunk = ChunkRecord(
            document_id=str(chunk_row["document_id"]),
            chunk_id=str(chunk_row["chunk_id"]),
            input_path=documents_by_id[str(chunk_row["document_id"])].input_path,
            chunk_pdf_path="",
            input_kind=str(chunk_row.get("input_kind") or "pdf"),
            raw_response_path=str(raw_path),
            text_path=str(chunk_row.get("text_path") or ""),
            start_page=int(chunk_row.get("start_page") or 0),
            end_page=int(chunk_row.get("end_page") or 0),
            expected_pages=int(chunk_row.get("expected_pages") or 0),
            size_bytes=int(chunk_row.get("chunk_size_bytes") or 0),
            status="ready",
        )
        for page_row in page_layout_rows(chunk, response):
            source = source_by_page.get((chunk.document_id, int(page_row["page_number"])), {})
            reasons: list[str] = []
            warnings_out: list[str] = []
            if not page_row["layout_page_present"]:
                reasons.append("missing_layout_page")
            has_math_evidence = (
                int(page_row["detected_formula_blocks"]) > 0
                or int(page_row["detected_inline_math_spans"]) > 0
            )
            if source.get("risk_status") == "high" and not has_math_evidence:
                reasons.append("source_math_expected_but_no_math_evidence")
            elif bool(source.get("formula_expected")) and not has_math_evidence:
                reasons.append("source_math_expected_but_no_math_evidence")
            if page_row["formula_syntax_issues"]:
                reasons.append(str(page_row["formula_syntax_issues"]))
            formulas = [formula for formula in page_row.get("_formula_contents", []) if formula.strip()]
            dropped_formulas = [
                formula
                for formula in formulas
                if normalized_formula_content(formula) not in normalized_markdown
            ]
            if dropped_formulas:
                reasons.append(f"formula_dropped_from_markdown={len(dropped_formulas)}")
            source_label_order = [
                split
                for split in str(source.get("display_equation_labels") or "").split(";")
                if split
            ]
            source_labels = set(source_label_order)
            extracted_labels = set(split for split in str(page_row["extracted_equation_labels"] or "").split(";") if split)
            missing_labels = sorted(source_labels - extracted_labels)
            recoverable_equation_labels: list[str] = []
            if missing_labels:
                formula_block_count = int(page_row["detected_formula_blocks"])
                dotted_label_order = [label for label in source_label_order if "." in label]
                repair_order = next(
                    (
                        labels
                        for labels in (source_label_order, dotted_label_order)
                        if len(labels) == formula_block_count
                    ),
                    [],
                )
                if repair_order:
                    recoverable_equation_labels = repair_order
                    warnings_out.append(
                        "recoverable_display_equation_labels=" + "|".join(recoverable_equation_labels)
                    )
                else:
                    warnings_out.append("unmapped_display_equation_labels=" + "|".join(missing_labels))
            if int(page_row["layout_content_chars"]) == 0 and int(source.get("source_chars") or 0) > 20:
                reasons.append("empty_layout_content")
            severity = "fail" if reasons else "warning" if warnings_out else "pass"
            combined = {
                **source,
                **page_row,
                "recoverable_equation_labels": ";".join(recoverable_equation_labels),
                "qa_severity": severity,
                "qa_reasons": ";".join(reasons + warnings_out),
            }
            qa_rows.append(combined)
            if severity == "fail":
                document = documents_by_id[chunk.document_id]
                retry_rows.append(
                    {
                        "document_id": chunk.document_id,
                        "input_path": document.input_path,
                        "page_number": page_row["page_number"],
                        "qa_reasons": combined["qa_reasons"],
                        "recommended_dpi": recommended_dpi,
                        "source_math_risk_score": source.get("math_risk_score", ""),
                        "first_pass_formula_blocks": page_row["detected_formula_blocks"],
                        "first_pass_raw_response_path": str(raw_path),
                        "status": "pending_single_page_reocr",
                    }
                )
    return qa_rows, retry_rows


def post_zai_chunk(
    chunk: ChunkRecord,
    *,
    api_key: str,
    endpoint: str,
    model: str,
    run_id: str,
    timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: float,
    base64_data_uri: bool,
    fallback_data_uri: bool,
    return_crop_images: bool,
    need_layout_visualization: bool,
) -> dict[str, Any]:
    input_path = Path(chunk.chunk_pdf_path)
    file_modes = [base64_data_uri]
    if fallback_data_uri and not base64_data_uri:
        file_modes.append(True)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    last_error: str | None = None
    for mode_index, use_data_uri in enumerate(file_modes, start=1):
        payload: dict[str, Any] = {
            "model": model,
            "file": encode_file(input_path, use_data_uri),
            "request_id": make_request_id(run_id, f"{chunk.chunk_id}-{mode_index}"),
        }
        if return_crop_images:
            payload["return_crop_images"] = True
        if need_layout_visualization:
            payload["need_layout_visualization"] = True

        for attempt in range(1, retries + 1):
            try:
                response = requests.post(endpoint, json=payload, headers=headers, timeout=timeout_seconds)
                response_text = response.text
                if response.status_code == 200:
                    data = response.json()
                    if "md_results" in data:
                        data["_codex_file_transport"] = "data_uri" if use_data_uri else "raw_base64"
                        return data
                    last_error = f"HTTP 200 without md_results: {response_text[:1000]}"
                elif response.status_code in {408, 409, 425, 429, 500, 502, 503, 504}:
                    last_error = f"Retryable HTTP {response.status_code}: {response_text[:1000]}"
                else:
                    last_error = f"HTTP {response.status_code}: {response_text[:2000]}"
                    break
            except Exception as exc:  # noqa: BLE001 - retry and record operational failures.
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < retries:
                time.sleep(retry_sleep_seconds * attempt)

    raise RuntimeError(last_error or "Z.ai OCR request failed without details.")


def summarize_response(chunk: ChunkRecord, response: dict[str, Any], status_override: str | None = None) -> dict[str, Any]:
    markdown = response.get("md_results") or ""
    data_info = response.get("data_info") or {}
    usage = response.get("usage") or {}
    labels = layout_label_counts(response)
    returned_pages = data_info.get("num_pages")

    warnings_out: list[str] = []
    if isinstance(returned_pages, int) and returned_pages != chunk.expected_pages:
        warnings_out.append(f"returned_pages={returned_pages} expected_pages={chunk.expected_pages}")
    if chunk.expected_pages >= 3:
        chars_per_page = len(markdown) / chunk.expected_pages
        if chars_per_page < 250:
            warnings_out.append(f"low_chars_per_page={chars_per_page:.1f}")
    else:
        chars_per_page = len(markdown)
    if not markdown.strip():
        warnings_out.append("empty_markdown")

    return {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "input_kind": chunk.input_kind,
        "start_page": chunk.start_page,
        "end_page": chunk.end_page,
        "expected_pages": chunk.expected_pages,
        "chunk_size_bytes": chunk.size_bytes,
        "status": status_override or ("warning" if warnings_out else "ok"),
        "raw_response_path": chunk.raw_response_path,
        "text_path": chunk.text_path,
        "returned_pages": returned_pages,
        "text_chars": len(markdown),
        "chars_per_page": round(chars_per_page, 2),
        "text_blocks": labels.get("text", 0),
        "formula_blocks": labels.get("formula", 0),
        "table_blocks": labels.get("table", 0),
        "image_blocks": labels.get("image", 0),
        "total_tokens": usage.get("total_tokens"),
        "request_id": response.get("request_id"),
        "response_task_id": response.get("id"),
        "file_transport": response.get("_codex_file_transport", ""),
        "started_at_utc": "",
        "completed_at_utc": utc_now(),
        "error_type": "",
        "error": "; ".join(warnings_out),
        "retry_count": "",
    }


def existing_chunk_result(chunk: ChunkRecord) -> dict[str, Any] | None:
    raw_path = Path(chunk.raw_response_path)
    text_path = Path(chunk.text_path)
    if not raw_path.exists() or not text_path.exists():
        return None
    try:
        response = json.loads(raw_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return summarize_response(chunk, response, status_override="cached")


def process_chunk(
    chunk: ChunkRecord,
    *,
    api_key: str,
    endpoint: str,
    model: str,
    run_id: str,
    timeout_seconds: int,
    retries: int,
    retry_sleep_seconds: float,
    base64_data_uri: bool,
    fallback_data_uri: bool,
    return_crop_images: bool,
    need_layout_visualization: bool,
    force: bool,
) -> dict[str, Any]:
    if chunk.status != "ready":
        return {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "input_kind": chunk.input_kind,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "expected_pages": chunk.expected_pages,
            "chunk_size_bytes": chunk.size_bytes,
            "status": chunk.status,
            "raw_response_path": chunk.raw_response_path,
            "text_path": chunk.text_path,
            "returned_pages": "",
            "text_chars": 0,
            "chars_per_page": "",
            "text_blocks": "",
            "formula_blocks": "",
            "table_blocks": "",
            "image_blocks": "",
            "total_tokens": "",
            "request_id": "",
            "response_task_id": "",
            "file_transport": "",
            "started_at_utc": "",
            "completed_at_utc": utc_now(),
            "error_type": "",
            "error": chunk.error or "",
            "retry_count": "",
        }

    if not force:
        cached = existing_chunk_result(chunk)
        if cached:
            return cached

    started_at = utc_now()
    raw_path = Path(chunk.raw_response_path)
    text_path = Path(chunk.text_path)
    error_path = raw_path.with_suffix(".error.json")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        response = post_zai_chunk(
            chunk,
            api_key=api_key,
            endpoint=endpoint,
            model=model,
            run_id=run_id,
            timeout_seconds=timeout_seconds,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
            base64_data_uri=base64_data_uri,
            fallback_data_uri=fallback_data_uri,
            return_crop_images=return_crop_images,
            need_layout_visualization=need_layout_visualization,
        )
        text_path.write_text(response.get("md_results") or "", encoding="utf-8")
        write_json(raw_path, response)
        row = summarize_response(chunk, response)
        row["started_at_utc"] = started_at
        return row
    except Exception as exc:  # noqa: BLE001 - per-chunk failure must be resumable.
        write_json(
            error_path,
            {
                "chunk": asdict(chunk),
                "started_at_utc": started_at,
                "completed_at_utc": utc_now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        return {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "input_kind": chunk.input_kind,
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "expected_pages": chunk.expected_pages,
            "chunk_size_bytes": chunk.size_bytes,
            "status": "error",
            "raw_response_path": "",
            "text_path": "",
            "returned_pages": "",
            "text_chars": 0,
            "chars_per_page": "",
            "text_blocks": "",
            "formula_blocks": "",
            "table_blocks": "",
            "image_blocks": "",
            "total_tokens": "",
            "request_id": "",
            "response_task_id": "",
            "file_transport": "",
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "retry_count": retries,
        }


def assemble_documents(
    run_dir: Path,
    documents: list[DocumentRecord],
    chunk_rows: list[dict[str, Any]],
    blocked_document_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    blocked_document_ids = blocked_document_ids or set()
    rows_by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in chunk_rows:
        rows_by_doc.setdefault(str(row.get("document_id")), []).append(row)

    assembly_rows: list[dict[str, Any]] = []
    for document in documents:
        if document.status == "duplicate":
            assembly_rows.append(
                {
                    "document_id": document.document_id,
                    "file_name": document.file_name,
                    "status": "skipped_duplicate",
                    "final_markdown_path": "",
                    "chunks_total": 0,
                    "chunks_ok": 0,
                    "chunks_error": 0,
                    "text_chars": 0,
                }
            )
            continue

        doc_rows = sorted(rows_by_doc.get(document.document_id, []), key=lambda row: int(row.get("start_page") or 0))
        ok_rows = [row for row in doc_rows if row.get("status") in {"ok", "warning", "cached"} and row.get("text_path")]
        error_rows = [row for row in doc_rows if row.get("status") in {"error", "blocked_by_zai_limit", "page_count_error"}]
        final_path = run_dir / "markdown" / f"{document.document_id}.md"

        if ok_rows:
            parts = [
                "---",
                f'title: "{document.file_name.replace(chr(34), chr(39))}"',
                f'document_id: "{document.document_id}"',
                f'source_path: "{document.input_path.replace(chr(34), chr(39))}"',
                f"sha256: {document.input_sha256}",
                f"page_count: {document.page_count}",
                f"created_at: {utc_now()}",
                "ocr_provider: zai",
                "ocr_model: glm-ocr",
                "---",
                "",
                f"# {document.file_name}",
                "",
            ]
            for row in ok_rows:
                text_path = Path(str(row["text_path"]))
                parts.append(
                    f"\n\n<!-- OCR_CHUNK {row['chunk_id']} pages {row['start_page']}-{row['end_page']} -->\n\n"
                )
                parts.append(text_path.read_text(encoding="utf-8"))
            final_path.parent.mkdir(parents=True, exist_ok=True)
            final_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
            if document.document_id in blocked_document_ids:
                status = "assembled_qa_blocked"
            else:
                status = "assembled_with_errors" if error_rows else "assembled"
            text_chars = final_path.stat().st_size
        else:
            status = "not_assembled"
            text_chars = 0

        assembly_rows.append(
            {
                "document_id": document.document_id,
                "file_name": document.file_name,
                "status": status,
                "final_markdown_path": str(final_path.resolve()) if ok_rows else "",
                "chunks_total": len(doc_rows),
                "chunks_ok": len(ok_rows),
                "chunks_error": len(error_rows),
                "text_chars": text_chars,
            }
        )
    return assembly_rows


def write_summary(
    run_dir: Path,
    run_id: str,
    documents: list[DocumentRecord],
    chunks: list[ChunkRecord],
    chunk_rows: list[dict[str, Any]],
    assembly_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    chunk_counts: dict[str, int] = {}
    for row in chunk_rows:
        status = str(row.get("status", "unknown"))
        chunk_counts[status] = chunk_counts.get(status, 0) + 1

    ready = [doc for doc in documents if doc.status == "ready"]
    duplicates = [doc for doc in documents if doc.status == "duplicate"]
    total_text_chars = sum(int(row.get("text_chars") or 0) for row in assembly_rows)
    total_tokens = sum(
        int(row.get("total_tokens") or 0)
        for row in chunk_rows
        if str(row.get("total_tokens") or "").isdigit()
    )
    warnings_out = [row for row in chunk_rows if row.get("error")]

    lines = [
        "# Z.ai OCR Summary",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated at UTC: `{utc_now()}`",
        f"- Input root: `{Path(args.input_dir).resolve()}`",
        f"- Artifact dir: `{run_dir.resolve()}`",
        f"- Provider: `zai`",
        f"- OCR model: `{args.model}`",
        f"- Transport mode: `{args.transport_mode}`",
        f"- Quality profile: `{args.quality_profile}`",
        f"- Dry run: `{args.dry_run}`",
        f"- Documents found: `{len(documents)}`",
        f"- Ready documents: `{len(ready)}`",
        f"- Exact duplicates skipped: `{len(duplicates)}`",
        f"- Chunks planned/processed: `{len(chunks)}`",
        f"- Concurrent workers: `{args.workers}`",
        f"- Total assembled Markdown chars: `{total_text_chars}`",
        f"- Total returned tokens: `{total_tokens}`",
        "",
        "## Chunk Status Counts",
        "",
    ]
    for status, count in sorted(chunk_counts.items()):
        lines.append(f"- `{status}`: `{count}`")

    if duplicates:
        lines.extend(["", "## Exact Duplicates Skipped", ""])
        for doc in duplicates:
            lines.append(f"- `{doc.document_id}` duplicates `{doc.duplicate_of}`: {doc.file_name}")

    if assembly_rows:
        lines.extend(["", "## Documents", ""])
        for row in assembly_rows:
            lines.append(
                f"- `{row['document_id']}`: `{row['status']}` "
                f"({row['chunks_ok']}/{row['chunks_total']} chunks ok)"
            )

    if warnings_out:
        lines.extend(["", "## Warnings And Errors", ""])
        for row in warnings_out[:100]:
            lines.append(f"- `{row.get('chunk_id')}`: `{row.get('status')}` - {row.get('error')}")
        if len(warnings_out) > 100:
            lines.append(f"- ... plus `{len(warnings_out) - 100}` more rows in `ocr_manifest.csv`.")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- OCR/extraction is performed by Z.ai GLM-OCR.",
            "- In chunked mode, local PDF page ranges are only transport containers for provider limits.",
            "- The pipeline does not run local OCR.",
            "- Raw Z.ai JSON is retained for layout/formula/table audit.",
        ]
    )
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Z.ai GLM-OCR PDF-to-Markdown pipeline.")
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--input-manifest", default=None, help="CSV containing a pdf_path column. Paths resolve from the workspace root.")
    parser.add_argument("--recursive", action="store_true", help="Recursively discover PDFs under --input-dir.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--transport-mode", choices=sorted(TRANSPORT_MODES), default="chunked")
    parser.add_argument("--splitter", choices=sorted(SPLITTERS), default="hybrid")
    parser.add_argument("--keep-page-cache", action="store_true", help="Keep Poppler single-page cache after request chunks are built.")
    parser.add_argument("--quality-profile", choices=sorted(QUALITY_PROFILES), default="optimal")
    parser.add_argument("--pages-per-chunk", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=None)
    parser.add_argument("--retry-sleep-seconds", type=float, default=10.0)
    parser.add_argument("--provider-max-pages", type=int, default=100)
    parser.add_argument("--provider-max-mb", type=float, default=50.0)
    parser.add_argument("--provider-max-image-mb", type=float, default=10.0)
    parser.add_argument("--image-dpi", type=int, default=None)
    parser.add_argument("--pypdf-split-timeout-seconds", type=int, default=60)
    parser.add_argument("--poppler-range-timeout-seconds", type=int, default=45)
    parser.add_argument("--no-image-fallback", action="store_true", help="Disable page-image transport fallback when PDF splitting fails.")
    parser.add_argument("--ignore-provider-limits", action="store_true")
    parser.add_argument("--book-contains", default=None, help="Only process filenames containing this text.")
    parser.add_argument("--document-offset", type=int, default=0, help="Skip this many ready non-duplicate documents before applying --limit-documents.")
    parser.add_argument("--limit-documents", type=int, default=None, help="Limit ready non-duplicate documents.")
    parser.add_argument("--skip-completed-artifacts", action="store_true", help="Skip documents already fully covered by successful OCR manifests in --output-root.")
    parser.add_argument("--limit-chunks", type=int, default=None, help="Limit chunks, useful for smoke tests.")
    parser.add_argument("--include-duplicates", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.set_defaults(base64_data_uri=True)
    parser.add_argument("--base64-data-uri", action="store_true", help="Send MIME-aware data:...;base64 payloads. This is the default.")
    parser.add_argument("--raw-base64", action="store_false", dest="base64_data_uri", help="Send raw base64 first, with optional data URI fallback.")
    parser.add_argument("--no-data-uri-fallback", action="store_true", help="Disable retrying with data URI transport if raw base64 fails.")
    parser.add_argument("--return-crop-images", action="store_true")
    parser.add_argument("--need-layout-visualization", action="store_true")
    parser.add_argument("--no-layout-artifacts", action="store_true", help="Disable crop and layout visualization requests in optimal mode.")
    return parser.parse_args(argv)


def apply_quality_profile(args: argparse.Namespace) -> None:
    defaults = {
        "balanced": {"pages_per_chunk": 90, "workers": 4, "retries": 5, "image_dpi": 180},
        "optimal": {"pages_per_chunk": 90, "workers": 4, "retries": 5, "image_dpi": 180},
        "math-precision": {"pages_per_chunk": 15, "workers": 2, "retries": 7, "image_dpi": 300},
    }[args.quality_profile]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.quality_profile in {"optimal", "math-precision"} and not args.no_layout_artifacts:
        args.return_crop_images = True
        args.need_layout_visualization = True


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.clean_page_cache = not args.keep_page_cache
    args.image_fallback = not args.no_image_fallback
    apply_quality_profile(args)
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1.")
    if args.pages_per_chunk < 1:
        raise SystemExit("--pages-per-chunk must be >= 1.")
    if args.pages_per_chunk > args.provider_max_pages and not args.ignore_provider_limits:
        raise SystemExit("--pages-per-chunk must be <= --provider-max-pages unless --ignore-provider-limits is set.")

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_root)
    run_id = args.run_id or make_run_id()
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    input_manifest = Path(args.input_manifest).expanduser() if args.input_manifest else None
    if input_manifest is not None and not input_manifest.is_absolute():
        input_manifest = (WORKSPACE_ROOT / input_manifest).resolve()
    all_documents = build_inventory(
        input_dir,
        recursive=args.recursive,
        input_manifest=input_manifest,
    )
    scoped_documents = all_documents
    if args.book_contains:
        needle = args.book_contains.lower()
        scoped_documents = [doc for doc in scoped_documents if needle in doc.file_name.lower()]

    documents = scoped_documents
    if not args.include_duplicates:
        documents = [doc for doc in documents if doc.status != "duplicate"]
    else:
        documents = [
            DocumentRecord(**(asdict(doc) | {"status": "ready", "duplicate_of": None}))
            if doc.status == "duplicate"
            else doc
            for doc in documents
        ]

    ready_documents = [doc for doc in documents if doc.status == "ready"]
    if args.document_offset < 0:
        raise SystemExit("--document-offset must be >= 0.")
    if args.document_offset or args.limit_documents is not None:
        selected_ready_documents = ready_documents[args.document_offset :]
        if args.limit_documents is not None:
            selected_ready_documents = selected_ready_documents[: args.limit_documents]
        allowed = {doc.document_id for doc in selected_ready_documents}
        documents = [doc for doc in documents if doc.document_id in allowed or doc.status != "ready"]
        ready_documents = [doc for doc in documents if doc.status == "ready"]

    skipped_completed_artifacts: set[str] = set()
    if args.skip_completed_artifacts:
        skipped_completed_artifacts = completed_document_ids_from_artifacts(
            output_root,
            {doc.document_id: doc for doc in all_documents},
        )
        documents = [
            doc
            for doc in documents
            if doc.status != "ready" or doc.document_id not in skipped_completed_artifacts
        ]
        ready_documents = [doc for doc in documents if doc.status == "ready"]

    source_risk_rows: list[dict[str, Any]] = []
    if args.quality_profile == "math-precision":
        source_risk_rows = build_source_math_risk(ready_documents)
        source_risk_fields = [
            "document_id", "input_path", "page_number", "source_chars", "source_glyphs",
            "math_font_chars", "math_font_ratio", "math_symbols", "equation_labels",
            "display_equation_labels",
            "theorem_headings", "math_risk_score", "formula_expected", "risk_status", "error",
        ]
        write_csv(
            run_dir / "source_page_math_risk.csv",
            [{field: row.get(field, "") for field in source_risk_fields} for row in source_risk_rows],
            source_risk_fields,
        )

    plan_rows = [estimate_document_plan(doc, args) for doc in scoped_documents]
    plan_fields = list(DocumentRecord.__dataclass_fields__.keys()) + [
        "transport_mode",
        "planned_chunks_estimate",
        "planned_status",
        "provider_limit_error",
    ]
    write_csv(run_dir / "ocr_plan.csv", plan_rows, plan_fields)

    manifest = {
        "run_id": run_id,
        "artifact_dir": str(run_dir.resolve()),
        "provider": "zai",
        "pipeline": "zai_ocr",
        "transport_mode": args.transport_mode,
        "source_root": str(input_dir.resolve()),
        "input_manifest": str(input_manifest.resolve()) if input_manifest else "",
        "recursive_input": args.recursive,
        "document_count": len(scoped_documents),
        "planned_document_count": len(ready_documents),
        "ocr_model": args.model,
        "ocr_endpoint": args.endpoint,
        "ocr_parameters": {
            "workers": args.workers,
            "quality_profile": args.quality_profile,
            "splitter": args.splitter,
            "clean_page_cache": args.clean_page_cache,
            "document_offset": args.document_offset,
            "skip_completed_artifacts": args.skip_completed_artifacts,
            "pages_per_chunk": args.pages_per_chunk,
            "timeout_seconds": args.timeout_seconds,
            "retries": args.retries,
            "provider_max_pages": args.provider_max_pages,
            "provider_max_mb": args.provider_max_mb,
            "provider_max_image_mb": args.provider_max_image_mb,
            "image_dpi": args.image_dpi,
            "pypdf_split_timeout_seconds": args.pypdf_split_timeout_seconds,
            "poppler_range_timeout_seconds": args.poppler_range_timeout_seconds,
            "image_fallback": args.image_fallback,
            "ignore_provider_limits": args.ignore_provider_limits,
            "base64_data_uri": args.base64_data_uri,
            "fallback_data_uri": not args.no_data_uri_fallback,
            "return_crop_images": args.return_crop_images,
            "need_layout_visualization": args.need_layout_visualization,
        },
        "skipped_completed_document_ids": sorted(skipped_completed_artifacts),
        "status": "planned" if args.dry_run else "running",
        "generated_at_utc": utc_now(),
        "artifacts": {
            "ocr_plan": str((run_dir / "ocr_plan.csv").resolve()),
            "chunk_plan": str((run_dir / "chunk_plan.csv").resolve()),
            "ocr_manifest": str((run_dir / "ocr_manifest.csv").resolve()),
            "book_manifest": str((run_dir / "book_manifest.csv").resolve()),
            "ocr_quality": str((run_dir / "ocr_quality.csv").resolve()),
            "source_page_math_risk": str((run_dir / "source_page_math_risk.csv").resolve()),
            "page_formula_qa": str((run_dir / "page_formula_qa.csv").resolve()),
            "formula_retry_queue": str((run_dir / "formula_retry_queue.csv").resolve()),
            "ocr_errors": str((run_dir / "ocr_errors.csv").resolve()),
            "summary": str((run_dir / "summary.md").resolve()),
            "raw_responses": str((run_dir / "raw_responses").resolve()),
            "chunk_markdown": str((run_dir / "chunk_markdown").resolve()),
            "markdown": str((run_dir / "markdown").resolve()),
            "chunks_pdf": str((run_dir / "chunks_pdf").resolve()),
            "chunks_image": str((run_dir / "chunks_image").resolve()),
        },
    }
    write_json(run_dir / "manifest.json", manifest)

    if args.dry_run:
        write_summary(run_dir, run_id, documents, [], [], [], args)
        print(f"Dry run complete: {run_dir}")
        print(f"Documents found: {len(scoped_documents)}; ready after filters: {len(ready_documents)}")
        return 0

    chunks = prepare_chunks(ready_documents, run_dir, args)
    chunk_fields = list(ChunkRecord.__dataclass_fields__.keys())
    write_csv(run_dir / "chunk_plan.csv", [asdict(chunk) for chunk in chunks], chunk_fields)
    if args.prepare_only:
        write_summary(run_dir, run_id, documents, chunks, [], [], args)
        print(f"Prepared chunks only: {run_dir}")
        print(f"Chunks prepared: {len(chunks)}")
        return 0

    api_key = env_api_key()
    if not api_key:
        raise SystemExit("Missing API key. Export ZAI_API_KEY or Z_AI_API_KEY before live OCR.")

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                process_chunk,
                chunk,
                api_key=api_key,
                endpoint=args.endpoint,
                model=args.model,
                run_id=run_id,
                timeout_seconds=args.timeout_seconds,
                retries=args.retries,
                retry_sleep_seconds=args.retry_sleep_seconds,
                base64_data_uri=args.base64_data_uri,
                fallback_data_uri=not args.no_data_uri_fallback,
                return_crop_images=args.return_crop_images,
                need_layout_visualization=args.need_layout_visualization,
                force=args.force,
            )
            for chunk in chunks
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            print(f"[{index}/{len(futures)}] {row['chunk_id']} -> {row['status']}", flush=True)

    manifest_fields = [
        "document_id",
        "chunk_id",
        "input_kind",
        "start_page",
        "end_page",
        "expected_pages",
        "chunk_size_bytes",
        "status",
        "raw_response_path",
        "text_path",
        "returned_pages",
        "text_chars",
        "chars_per_page",
        "text_blocks",
        "formula_blocks",
        "table_blocks",
        "image_blocks",
        "total_tokens",
        "request_id",
        "response_task_id",
        "file_transport",
        "started_at_utc",
        "completed_at_utc",
        "error_type",
        "error",
        "retry_count",
    ]
    normalized_rows = [{field: row.get(field, "") for field in manifest_fields} for row in rows]
    write_csv(run_dir / "ocr_manifest.csv", normalized_rows, manifest_fields)

    quality_fields = [
        "document_id",
        "chunk_id",
        "input_kind",
        "status",
        "start_page",
        "end_page",
        "expected_pages",
        "returned_pages",
        "text_chars",
        "chars_per_page",
        "formula_blocks",
        "table_blocks",
        "error",
    ]
    write_csv(
        run_dir / "ocr_quality.csv",
        [{field: row.get(field, "") for field in quality_fields} for row in normalized_rows],
        quality_fields,
    )

    error_rows = [row for row in normalized_rows if row.get("status") not in {"ok", "warning", "cached"}]
    error_fields = ["document_id", "chunk_id", "error_type", "error"]
    write_csv(run_dir / "ocr_errors.csv", [{field: row.get(field, "") for field in error_fields} for row in error_rows], error_fields)

    page_qa_rows: list[dict[str, Any]] = []
    formula_retry_rows: list[dict[str, Any]] = []
    blocked_document_ids: set[str] = set()
    if args.quality_profile == "math-precision":
        page_qa_rows, formula_retry_rows = build_math_precision_page_qa(
            run_dir,
            documents,
            normalized_rows,
            source_risk_rows,
            args.image_dpi,
        )
        page_qa_fields = [
            "document_id", "input_path", "page_number", "source_chars", "source_glyphs",
            "math_font_chars", "math_font_ratio", "math_symbols", "equation_labels",
            "display_equation_labels",
            "theorem_headings", "math_risk_score", "formula_expected", "risk_status",
            "chunk_id", "layout_page_present", "layout_elements", "text_blocks",
            "formula_blocks", "detected_formula_blocks", "table_blocks", "image_blocks",
            "detected_inline_math_spans",
            "layout_content_chars", "extracted_equation_labels", "formula_syntax_issues",
            "formula_hashes", "recoverable_equation_labels", "qa_severity", "qa_reasons",
        ]
        write_csv(
            run_dir / "page_formula_qa.csv",
            [{field: row.get(field, "") for field in page_qa_fields} for row in page_qa_rows],
            page_qa_fields,
        )
        retry_fields = [
            "document_id", "input_path", "page_number", "qa_reasons", "recommended_dpi",
            "source_math_risk_score", "first_pass_formula_blocks",
            "first_pass_raw_response_path", "status",
        ]
        write_csv(
            run_dir / "formula_retry_queue.csv",
            [{field: row.get(field, "") for field in retry_fields} for row in formula_retry_rows],
            retry_fields,
        )
        blocked_document_ids = {str(row["document_id"]) for row in formula_retry_rows}

    assembly_rows = assemble_documents(run_dir, documents, normalized_rows, blocked_document_ids)
    assembly_fields = [
        "document_id",
        "file_name",
        "status",
        "final_markdown_path",
        "chunks_total",
        "chunks_ok",
        "chunks_error",
        "text_chars",
    ]
    write_csv(run_dir / "book_manifest.csv", assembly_rows, assembly_fields)

    completed_manifest = dict(manifest)
    if error_rows:
        completed_manifest["status"] = "complete_with_errors"
    elif formula_retry_rows:
        completed_manifest["status"] = "formula_review_required"
    else:
        completed_manifest["status"] = "complete"
    completed_manifest["completed_at_utc"] = utc_now()
    completed_manifest["completed_chunks"] = sum(1 for row in normalized_rows if row.get("status") in {"ok", "warning", "cached"})
    completed_manifest["failed_chunks"] = len(error_rows)
    completed_manifest["formula_flagged_pages"] = len(formula_retry_rows)
    completed_manifest["formula_blocked_documents"] = len(blocked_document_ids)
    completed_manifest["completed_documents"] = sum(1 for row in assembly_rows if row.get("status") in {"assembled", "assembled_with_errors"})
    completed_manifest["total_text_chars"] = sum(int(row.get("text_chars") or 0) for row in assembly_rows)
    write_json(run_dir / "manifest.json", completed_manifest)
    write_summary(run_dir, run_id, documents, chunks, normalized_rows, assembly_rows, args)

    print(f"OCR run complete: {run_dir}")
    print(f"Summary: {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
