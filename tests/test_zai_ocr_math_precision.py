from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from pypdf import PdfWriter


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_zai_ocr_pipeline.py"
SPEC = importlib.util.spec_from_file_location("zai_ocr_pipeline", SCRIPT)
assert SPEC and SPEC.loader
ocr = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ocr
SPEC.loader.exec_module(ocr)


def test_math_precision_profile_sets_forensic_defaults() -> None:
    args = argparse.Namespace(
        quality_profile="math-precision",
        pages_per_chunk=None,
        workers=None,
        retries=None,
        image_dpi=None,
        no_layout_artifacts=False,
        return_crop_images=False,
        need_layout_visualization=False,
    )
    ocr.apply_quality_profile(args)
    assert args.pages_per_chunk == 15
    assert args.workers == 2
    assert args.retries == 7
    assert args.image_dpi == 300
    assert args.return_crop_images is True
    assert args.need_layout_visualization is True


def test_manifest_input_preserves_priority_and_deduplicates_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    for path in (first, second):
        writer = PdfWriter()
        writer.add_blank_page(width=612, height=792)
        with path.open("wb") as handle:
            writer.write(handle)
    manifest = tmp_path / "queue.csv"
    manifest.write_text(
        f"pdf_path\n{second}\n{first}\n{second}\n",
        encoding="utf-8",
    )
    paths = ocr.input_paths(tmp_path, input_manifest=manifest)
    assert paths == [second, first]


def test_formula_syntax_scanner_finds_structural_errors() -> None:
    assert ocr.formula_syntax_issues(r"\begin{aligned} x &= y \\ z &= 1 \end{aligned}") == []
    assert ocr.formula_syntax_issues(r"\left[x \rightarrow y\right]") == []
    issues = ocr.formula_syntax_issues(r"\begin{matrix} x_{1 & y \end{align}")
    assert "unbalanced_braces" in issues
    assert "unbalanced_environments" in issues


def test_native_formula_label_is_counted() -> None:
    chunk = ocr.ChunkRecord(
        document_id="doc",
        chunk_id="doc__p0001-0001",
        input_path="source.pdf",
        chunk_pdf_path="chunk.pdf",
        input_kind="pdf",
        raw_response_path="raw.json",
        text_path="page.md",
        start_page=1,
        end_page=1,
        expected_pages=1,
        size_bytes=100,
        status="ready",
    )
    response = {
        "layout_details": [[{"label": "text", "native_label": "display_formula", "content": r"x_{1}=2"}]],
    }
    rows = ocr.page_layout_rows(chunk, response)
    assert rows[0]["detected_formula_blocks"] == 1


def test_inline_math_is_page_level_math_evidence() -> None:
    chunk = ocr.ChunkRecord(
        document_id="doc",
        chunk_id="doc__p0001-0001",
        input_path="source.pdf",
        chunk_pdf_path="chunk.pdf",
        input_kind="pdf",
        raw_response_path="raw.json",
        text_path="page.md",
        start_page=1,
        end_page=1,
        expected_pages=1,
        size_bytes=100,
        status="ready",
    )
    response = {
        "layout_details": [[{
            "label": "text",
            "native_label": "text",
            "content": r"Assume $\\alpha < 1/4$ throughout.",
        }]],
    }
    rows = ocr.page_layout_rows(chunk, response)
    assert rows[0]["detected_formula_blocks"] == 0
    assert rows[0]["detected_inline_math_spans"] == 1


def test_source_math_without_formula_creates_retry(tmp_path: Path) -> None:
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"md_results": "plain text", "layout_details": [[]]}), encoding="utf-8")
    document = ocr.DocumentRecord(
        document_id="doc",
        input_path=str(tmp_path / "source.pdf"),
        file_name="source.pdf",
        mime_type="application/pdf",
        input_sha256="abc",
        size_bytes=100,
        page_count=1,
        duplicate_of=None,
        status="ready",
    )
    chunk_rows = [{
        "document_id": "doc",
        "chunk_id": "doc__p0001-0001",
        "status": "ok",
        "raw_response_path": str(raw),
        "text_path": str(tmp_path / "page.md"),
        "input_kind": "pdf",
        "start_page": 1,
        "end_page": 1,
        "expected_pages": 1,
        "chunk_size_bytes": 100,
    }]
    source_rows = [{
        "document_id": "doc",
        "input_path": document.input_path,
        "page_number": 1,
        "source_chars": 100,
        "math_risk_score": 5,
        "formula_expected": True,
        "equation_labels": "",
    }]
    qa_rows, retry_rows = ocr.build_math_precision_page_qa(
        tmp_path, [document], chunk_rows, source_rows, 300
    )
    assert qa_rows[0]["qa_severity"] == "fail"
    assert retry_rows[0]["status"] == "pending_single_page_reocr"
