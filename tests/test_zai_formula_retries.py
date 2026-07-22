from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_zai_formula_retries.py"
SPEC = importlib.util.spec_from_file_location("zai_formula_retries", SCRIPT)
assert SPEC and SPEC.loader
retry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = retry
SPEC.loader.exec_module(retry)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"stable source pdf bytes")
    first_raw = run_dir / "raw_responses" / "doc" / "p0001-0001.json"
    first_raw.parent.mkdir(parents=True)
    first_raw.write_text(
        json.dumps({"md_results": "first pass prose", "layout_details": [[{"index": 0, "label": "text", "content": "first pass prose"}]]}),
        encoding="utf-8",
    )
    first_markdown = run_dir / "chunk_markdown" / "doc" / "p0001-0001.md"
    first_markdown.parent.mkdir(parents=True)
    first_markdown.write_text("first pass prose", encoding="utf-8")
    write_csv(
        run_dir / "ocr_plan.csv",
        [
            {
                "document_id": "doc",
                "input_path": str(source_pdf),
                "file_name": "source.pdf",
                "mime_type": "application/pdf",
                "input_sha256": sha256(source_pdf),
                "size_bytes": source_pdf.stat().st_size,
                "page_count": 1,
                "duplicate_of": "",
                "status": "ready",
                "error": "",
            }
        ],
    )
    write_csv(
        run_dir / "formula_retry_queue.csv",
        [
            {
                "document_id": "doc",
                "input_path": str(source_pdf),
                "page_number": 1,
                "qa_reasons": "source_math_expected_but_no_formula_block",
                "recommended_dpi": 300,
                "source_math_risk_score": 5,
                "first_pass_formula_blocks": 0,
                "first_pass_raw_response_path": str(first_raw),
                "status": "pending_single_page_reocr",
            }
        ],
    )
    write_csv(
        run_dir / "source_page_math_risk.csv",
        [
            {
                "document_id": "doc",
                "input_path": str(source_pdf),
                "page_number": 1,
                "source_chars": 100,
                "source_glyphs": 100,
                "math_font_chars": 20,
                "math_font_ratio": 0.2,
                "math_symbols": 5,
                "equation_labels": "",
                "theorem_headings": 0,
                "math_risk_score": 5,
                "formula_expected": "True",
                "risk_status": "high",
                "error": "",
            }
        ],
    )
    write_csv(
        run_dir / "page_formula_qa.csv",
        [
            {
                "document_id": "doc",
                "page_number": 1,
                "formula_hashes": "",
                "qa_severity": "fail",
                "qa_reasons": "source_math_expected_but_no_formula_block",
            }
        ],
    )
    write_csv(
        run_dir / "ocr_manifest.csv",
        [
            {
                "document_id": "doc",
                "chunk_id": "doc__p0001-0001",
                "status": "ok",
                "start_page": 1,
                "end_page": 1,
                "expected_pages": 1,
                "raw_response_path": str(first_raw),
                "text_path": str(first_markdown),
            }
        ],
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "TEST-RUN",
                "ocr_endpoint": "https://example.invalid/layout",
                "ocr_model": "glm-ocr",
                "ocr_parameters": {
                    "timeout_seconds": 60,
                    "retries": 1,
                    "provider_max_image_mb": 10,
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir, source_pdf, first_raw


def test_dedupe_queue_preserves_order_and_merges_reasons() -> None:
    rows = retry.dedupe_queue_rows(
        [
            {"document_id": "a", "page_number": "2", "qa_reasons": "missing_formula"},
            {"document_id": "b", "page_number": "1", "qa_reasons": "empty_layout"},
            {"document_id": "a", "page_number": "2", "qa_reasons": "missing_formula;bad_latex"},
        ]
    )
    assert [(row["document_id"], row["page_number"]) for row in rows] == [("a", 2), ("b", 1)]
    assert rows[0]["qa_reasons"] == "missing_formula;bad_latex"


def test_layout_page_markdown_repairs_ordered_equation_tags() -> None:
    elements = [
        {"index": 0, "label": "formula", "native_label": "display_formula", "content": "$$x=1$$"},
        {"index": 1, "label": "formula", "native_label": "display_formula", "content": "$$y=2$$"},
    ]
    markdown = retry.layout_page_markdown(elements, ["(1.1)", "(1.2)"])
    assert r"\tag{1.1}" in markdown
    assert r"\tag{1.2}" in markdown


def test_mocked_retry_is_immutable_resumable_and_page_assembled(tmp_path: Path, monkeypatch) -> None:
    run_dir, _source_pdf, first_raw = make_run(tmp_path)
    original_first_hash = sha256(first_raw)
    first_markdown = run_dir / "chunk_markdown" / "doc" / "p0001-0001.md"
    original_first_markdown_hash = sha256(first_markdown)
    calls = {"render": 0, "process": 0}

    def fake_render(_source: Path, destination: Path, _page: int, _dpi: int) -> bool:
        calls["render"] += 1
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"mock png")
        return True

    def fake_process(chunk, **_kwargs):
        calls["process"] += 1
        response = {
            "md_results": "$$\nx=1\n$$",
            "layout_details": [[{"index": 0, "label": "formula", "native_label": "display_formula", "content": "$$\nx=1\n$$"}]],
            "data_info": {"num_pages": 1},
            "request_id": f"request-{calls['process']}",
            "usage": {"total_tokens": 12},
        }
        raw_path = Path(chunk.raw_response_path)
        markdown_path = Path(chunk.text_path)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(json.dumps(response), encoding="utf-8")
        markdown_path.write_text(response["md_results"], encoding="utf-8")
        return {
            "document_id": chunk.document_id,
            "chunk_id": chunk.chunk_id,
            "input_kind": "image",
            "start_page": chunk.start_page,
            "end_page": chunk.end_page,
            "expected_pages": 1,
            "chunk_size_bytes": chunk.size_bytes,
            "status": "ok",
            "raw_response_path": chunk.raw_response_path,
            "text_path": chunk.text_path,
            "request_id": response["request_id"],
            "total_tokens": 12,
            "started_at_utc": "2026-01-01T00:00:00Z",
            "completed_at_utc": "2026-01-01T00:00:01Z",
            "error_type": "",
            "error": "",
        }

    monkeypatch.setattr(retry.ocr, "render_page_image", fake_render)
    monkeypatch.setattr(retry.ocr, "process_chunk", fake_process)
    monkeypatch.setattr(retry.ocr, "env_api_key", lambda: "test-key-never-sent")

    assert retry.main(["--run-dir", str(run_dir)]) == 0
    assert calls == {"render": 1, "process": 1}
    assert sha256(first_raw) == original_first_hash
    assert sha256(first_markdown) == original_first_markdown_hash
    ledger = retry.read_csv(run_dir / "formula_retry_results.csv")
    assert len(ledger) == 1
    assert ledger[0]["attempt_id"] == "attempt-01"
    assert ledger[0]["resolution_status"] == "resolved"
    assert ledger[0]["first_pass_artifact_unchanged"] == "True"
    assert "/formula_retries/raw/" in ledger[0]["retry_raw_response_path"]
    assert ledger[0]["retry_raw_response_path"] != str(first_raw)
    verified = run_dir / "markdown_verified" / "doc.md"
    assert verified.exists()
    verified_text = verified.read_text(encoding="utf-8")
    assert "<!-- OCR_PAGE 1 source=formula_retry -->" in verified_text
    assert "x=1" in verified_text

    # A normal rerun reuses the completed attempt and needs no API key.
    monkeypatch.setattr(retry.ocr, "env_api_key", lambda: None)
    assert retry.main(["--run-dir", str(run_dir)]) == 0
    assert calls == {"render": 1, "process": 1}
    assert len(retry.read_csv(run_dir / "formula_retry_results.csv")) == 1

    # Force creates a second immutable attempt instead of replacing attempt 01.
    monkeypatch.setattr(retry.ocr, "env_api_key", lambda: "test-key-never-sent")
    assert retry.main(["--run-dir", str(run_dir), "--force-new-attempt"]) == 0
    assert calls == {"render": 2, "process": 2}
    ledger = retry.read_csv(run_dir / "formula_retry_results.csv")
    assert [row["attempt_id"] for row in ledger] == ["attempt-01", "attempt-02"]
    assert Path(ledger[0]["retry_raw_response_path"]).exists()
    assert Path(ledger[1]["retry_raw_response_path"]).exists()
    assert sha256(first_raw) == original_first_hash
    assert sha256(first_markdown) == original_first_markdown_hash


def test_source_hash_change_blocks_retry_without_api(tmp_path: Path, monkeypatch) -> None:
    run_dir, source_pdf, first_raw = make_run(tmp_path)
    original_first_hash = sha256(first_raw)
    source_pdf.write_bytes(b"changed after ocr plan")

    monkeypatch.setattr(retry.ocr, "env_api_key", lambda: (_ for _ in ()).throw(AssertionError("API key must not be requested")))
    monkeypatch.setattr(
        retry.ocr,
        "process_chunk",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )

    assert retry.main(["--run-dir", str(run_dir)]) == 0
    ledger = retry.read_csv(run_dir / "formula_retry_results.csv")
    assert ledger[0]["resolution_status"] == "source_changed"
    assert ledger[0]["transport_status"] == "blocked"
    assert sha256(first_raw) == original_first_hash
    verified_manifest = retry.read_csv(run_dir / "book_manifest_verified.csv")
    assert verified_manifest[0]["status"] == "blocked_unresolved_retry"
    assert not (run_dir / "markdown_verified" / "doc.md").exists()
