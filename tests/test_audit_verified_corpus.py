from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_verified_corpus.py"
SPEC = importlib.util.spec_from_file_location("audit_verified_corpus", SCRIPT)
assert SPEC and SPEC.loader
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)

TAG_FIELDS = [
    "document_id", "page_number", "label", "status", "source_kind", "element_index",
    "source_bbox", "formula_bbox", "body_sha256_before", "body_sha256_after",
    "spatial_cost", "injected", "raw_response_path", "raw_response_sha256",
    "source_pdf_path", "source_pdf_sha256",
]


def write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspace"
    run_dir = workspace / "run"
    sources = workspace / "papers"
    verified = run_dir / "markdown_verified"
    sources.mkdir(parents=True)
    verified.mkdir(parents=True)

    pdf_a = sources / "a.pdf"
    pdf_b = sources / "b.pdf"
    pdf_a.write_bytes(b"mock-pdf-a")
    pdf_b.write_bytes(b"mock-pdf-b")
    sha_a, sha_b = digest(pdf_a), digest(pdf_b)
    queue = workspace / "queue.csv"
    write_csv(
        queue,
        ["pdf_path", "sha256"],
        [
            {"pdf_path": "papers/a.pdf", "sha256": sha_a},
            {"pdf_path": "papers/b.pdf", "sha256": ""},
        ],
    )

    plan_fields = ["document_id", "input_path", "input_sha256", "page_count", "status"]
    write_csv(
        run_dir / "ocr_plan.csv",
        plan_fields,
        [
            {"document_id": "doc-a", "input_path": str(pdf_a), "input_sha256": sha_a, "page_count": 2, "status": "ready"},
            {"document_id": "doc-b", "input_path": str(pdf_b), "input_sha256": sha_b, "page_count": 1, "status": "ready"},
        ],
    )

    raw_a = run_dir / "raw" / "a.json"
    raw_b = run_dir / "raw" / "b.json"
    raw_a.parent.mkdir(parents=True)
    raw_a.write_text(json.dumps({"layout_details": [[], []]}), encoding="utf-8")
    raw_b.write_text(json.dumps({"layout_details": [[]]}), encoding="utf-8")
    chunk_fields = [
        "document_id", "chunk_id", "start_page", "end_page", "expected_pages",
        "returned_pages", "status", "raw_response_path",
    ]
    write_csv(
        run_dir / "ocr_manifest.csv",
        chunk_fields,
        [
            {"document_id": "doc-a", "chunk_id": "a-1", "start_page": 1, "end_page": 2, "expected_pages": 2, "returned_pages": 2, "status": "ok", "raw_response_path": str(raw_a)},
            {"document_id": "doc-b", "chunk_id": "b-1", "start_page": 1, "end_page": 1, "expected_pages": 1, "returned_pages": 1, "status": "cached", "raw_response_path": str(raw_b)},
        ],
    )
    write_csv(run_dir / "ocr_errors.csv", ["document_id", "error"], [])
    write_csv(
        run_dir / "page_formula_qa.csv",
        ["document_id", "page_number", "qa_severity"],
        [
            {"document_id": "doc-a", "page_number": 1, "qa_severity": "pass"},
            {"document_id": "doc-a", "page_number": 2, "qa_severity": "fail"},
            {"document_id": "doc-b", "page_number": 1, "qa_severity": "warning"},
        ],
    )
    write_csv(
        run_dir / "formula_retry_queue.csv",
        ["document_id", "page_number"],
        [{"document_id": "doc-a", "page_number": 2}],
    )
    retry_raw = run_dir / "retry" / "page2.json"
    retry_md = run_dir / "retry" / "page2.md"
    retry_raw.parent.mkdir(parents=True)
    retry_raw.write_text('{"layout_details": [[]]}', encoding="utf-8")
    retry_md.write_text("$x^2$\n", encoding="utf-8")
    retry_fields = [
        "document_id", "page_number", "attempt_number", "completed_at_utc",
        "resolution_status", "transport_status", "source_pdf_sha256",
        "expected_source_sha256", "retry_raw_response_path", "retry_raw_sha256",
        "retry_markdown_path", "retry_markdown_sha256",
    ]
    write_csv(
        run_dir / "formula_retry_results.csv",
        retry_fields,
        [
            {
                "document_id": "doc-a", "page_number": 2, "attempt_number": 1,
                "completed_at_utc": "2026-01-01T00:00:00Z", "resolution_status": "unresolved",
                "transport_status": "ok", "source_pdf_sha256": sha_a,
                "expected_source_sha256": sha_a,
            },
            {
                "document_id": "doc-a", "page_number": 2, "attempt_number": 2,
                "completed_at_utc": "2026-01-01T00:01:00Z", "resolution_status": "resolved",
                "transport_status": "cached", "source_pdf_sha256": sha_a,
                "expected_source_sha256": sha_a, "retry_raw_response_path": str(retry_raw),
                "retry_raw_sha256": digest(retry_raw), "retry_markdown_path": str(retry_md),
                "retry_markdown_sha256": digest(retry_md),
            },
        ],
    )
    write_csv(run_dir / "equation_tag_geometry_qa.csv", TAG_FIELDS, [])

    md_a = verified / "doc-a.md"
    md_b = verified / "doc-b.md"
    md_a.write_text(
        "\n".join(
            [
                "---", 'document_id: "doc-a"', f"source_path: \"{pdf_a}\"", f"sha256: {sha_a}",
                "page_count: 2", "---", "<!-- OCR_PAGE 1 source=first_pass_layout -->", "page one",
                "<!-- OCR_PAGE 2 source=formula_retry -->", "$x^2$", "",
            ]
        ),
        encoding="utf-8",
    )
    md_b.write_text(
        "\n".join(
            [
                "---", 'document_id: "doc-b"', f"source_path: \"{pdf_b}\"", f"sha256: {sha_b}",
                "page_count: 1", "---", "<!-- OCR_PAGE 1 source=first_pass_layout -->", "page one", "",
            ]
        ),
        encoding="utf-8",
    )
    write_csv(
        run_dir / "book_manifest_verified.csv",
        ["document_id", "status", "verified_markdown_path", "page_count", "pages_materialized", "unresolved_pages", "warning_pages"],
        [
            {"document_id": "doc-a", "status": "verified", "verified_markdown_path": str(md_a), "page_count": 2, "pages_materialized": 2, "unresolved_pages": ""},
            {"document_id": "doc-b", "status": "verified", "verified_markdown_path": str(md_b), "page_count": 1, "pages_materialized": 1, "unresolved_pages": ""},
        ],
    )
    return workspace, run_dir, queue


def fixture_page_counter(path: Path) -> int:
    return {"a.pdf": 2, "b.pdf": 1}[path.name]


def test_complete_mock_corpus_passes_and_latest_retry_wins(tmp_path: Path) -> None:
    workspace, run_dir, queue = build_fixture(tmp_path)
    report = audit.audit_corpus(
        run_dir,
        queue,
        workspace,
        expected_documents=2,
        expected_pages=3,
        page_counter=fixture_page_counter,
    )
    assert report["status"] == "pass", report["errors"]
    assert report["resolved_formula_retry_pages"] == 1
    assert report["verified_markdown_files"] == 2
    assert report["verified_page_markers"] == 3
    assert report["formula_warning_pages"] == 1
    assert not list(tmp_path.rglob("*audit*.json"))
    assert not list(tmp_path.rglob("*audit*.csv"))


def test_missing_page_marker_and_changed_source_fail(tmp_path: Path) -> None:
    workspace, run_dir, queue = build_fixture(tmp_path)
    (workspace / "papers" / "b.pdf").write_bytes(b"changed")
    markdown = run_dir / "markdown_verified" / "doc-a.md"
    markdown.write_text(markdown.read_text(encoding="utf-8").replace("<!-- OCR_PAGE 2 source=formula_retry -->", ""), encoding="utf-8")
    report = audit.audit_corpus(
        run_dir,
        queue,
        workspace,
        expected_documents=2,
        expected_pages=3,
        page_counter=fixture_page_counter,
    )
    assert report["status"] == "fail"
    assert any("source SHA mismatch: doc-b" in error for error in report["errors"])
    assert any("page markers incomplete/out of order: doc-a" in error for error in report["errors"])


def test_reports_are_written_only_by_explicit_writer_calls(tmp_path: Path) -> None:
    workspace, run_dir, queue = build_fixture(tmp_path)
    report = audit.audit_corpus(
        run_dir,
        queue,
        workspace,
        expected_documents=2,
        expected_pages=3,
        page_counter=fixture_page_counter,
    )
    json_path = tmp_path / "explicit" / "audit.json"
    csv_path = tmp_path / "explicit" / "audit.csv"
    assert not json_path.exists() and not csv_path.exists()
    audit.write_json_report(json_path, report)
    audit.write_csv_report(csv_path, report["documents"])
    assert json.loads(json_path.read_text(encoding="utf-8"))["status"] == "pass"
    assert len(list(csv.DictReader(csv_path.open(encoding="utf-8")))) == 2


def test_mapped_equation_tag_has_raw_coordinate_and_body_provenance(tmp_path: Path) -> None:
    workspace, run_dir, queue = build_fixture(tmp_path)
    raw_path = run_dir / "raw" / "a.json"
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_formula = {
        "index": 7, "label": "formula", "native_label": "display_formula",
        "content": "$$\nx=1\n$$", "bbox_2d": [100, 200, 700, 260],
        "width": 1000, "height": 1000,
    }
    raw["layout_details"][0] = [raw_formula]
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    markdown_path = run_dir / "markdown_verified" / "doc-a.md"
    markdown = markdown_path.read_text(encoding="utf-8").replace(
        "<!-- OCR_PAGE 1 source=first_pass_layout -->\npage one",
        "<!-- OCR_PAGE 1 source=first_pass_layout -->\n$$\nx=1 \\tag{1}\n$$",
    )
    markdown_path.write_text(markdown, encoding="utf-8")
    source = workspace / "papers" / "a.pdf"
    body_hash = audit.formula_body_hash(raw_formula["content"])
    write_csv(
        run_dir / "equation_tag_geometry_qa.csv",
        TAG_FIELDS,
        [{
            "document_id": "doc-a", "page_number": 1, "label": "(1)", "status": "mapped",
            "source_kind": "first_pass_geometry", "element_index": 7,
            "source_bbox": json.dumps([520, 205, 550, 250]),
            "formula_bbox": json.dumps(raw_formula["bbox_2d"]),
            "body_sha256_before": body_hash, "body_sha256_after": body_hash,
            "spatial_cost": 1.25, "injected": True,
            "raw_response_path": str(raw_path), "raw_response_sha256": digest(raw_path),
            "source_pdf_path": str(source), "source_pdf_sha256": digest(source),
        }],
    )
    report = audit.audit_corpus(
        run_dir, queue, workspace, expected_documents=2, expected_pages=3,
        page_counter=fixture_page_counter,
    )
    assert report["status"] == "pass", report["errors"]
    assert report["equation_tags_mapped"] == 1
    assert report["tag_body_integrity_failures"] == 0
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8").replace("x=1 \\tag{1}", "x=2 \\tag{1}"),
        encoding="utf-8",
    )
    tampered = audit.audit_corpus(
        run_dir, queue, workspace, expected_documents=2, expected_pages=3,
        page_counter=fixture_page_counter,
    )
    assert tampered["status"] == "fail"
    assert tampered["tag_body_integrity_failures"] == 1
    assert any("absent/body changed in final Markdown" in error for error in tampered["errors"])


def test_geometry_warning_page_drift_fails(tmp_path: Path) -> None:
    workspace, run_dir, queue = build_fixture(tmp_path)
    source = workspace / "papers" / "b.pdf"
    write_csv(
        run_dir / "equation_tag_geometry_qa.csv",
        TAG_FIELDS,
        [{
            "document_id": "doc-b", "page_number": 1, "label": "(8)", "status": "ambiguous",
            "source_kind": "first_pass_geometry", "source_bbox": json.dumps([500, 200, 540, 220]),
            "injected": False, "source_pdf_path": str(source), "source_pdf_sha256": digest(source),
        }],
    )
    report = audit.audit_corpus(
        run_dir, queue, workspace, expected_documents=2, expected_pages=3,
        page_counter=fixture_page_counter,
    )
    assert report["status"] == "fail"
    assert report["equation_tag_residual_pages"] == 1
    assert report["equation_tag_residual_labels"] == 1
    assert any("clean verified book omits retained equation-tag warnings: doc-b" in error for error in report["errors"])
