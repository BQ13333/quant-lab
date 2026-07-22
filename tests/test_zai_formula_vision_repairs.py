from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_zai_formula_vision_repairs.py"
SPEC = importlib.util.spec_from_file_location("zai_formula_vision_repairs", SCRIPT)
assert SPEC and SPEC.loader
vision = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vision
SPEC.loader.exec_module(vision)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_run(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"stable source pdf bytes")
    first_raw = run_dir / "raw_responses" / "doc" / "p0001-0001.json"
    first_raw.parent.mkdir(parents=True)
    first_raw.write_text(
        json.dumps(
            {
                "md_results": "first pass",
                "layout_details": [[{"index": 0, "label": "text", "content": "first pass"}]],
            }
        ),
        encoding="utf-8",
    )
    first_markdown = run_dir / "chunk_markdown" / "doc" / "p0001-0001.md"
    first_markdown.parent.mkdir(parents=True)
    first_markdown.write_text("first pass\n", encoding="utf-8")

    rendered = run_dir / "formula_retries" / "rendered" / "doc" / "p0001" / "attempt-01.png"
    rendered.parent.mkdir(parents=True)
    rendered.write_bytes(b"mock png bytes")
    retry_raw = run_dir / "formula_retries" / "raw" / "doc" / "p0001" / "attempt-01.json"
    retry_raw.parent.mkdir(parents=True)
    retry_raw.write_text(json.dumps({"md_results": "bad retry"}), encoding="utf-8")
    retry_markdown = run_dir / "formula_retries" / "markdown" / "doc" / "p0001" / "attempt-01.md"
    retry_markdown.parent.mkdir(parents=True)
    retry_markdown.write_text("$$x_{1$$\n\n$$y=2$$\n", encoding="utf-8")

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
                "qa_reasons": "unbalanced_braces;recoverable_display_equation_labels=(1.1)|(1.2)",
                "recommended_dpi": 300,
                "source_math_risk_score": 6,
                "first_pass_formula_blocks": 2,
                "first_pass_raw_response_path": str(first_raw),
                "status": "pending_single_page_reocr",
            }
        ],
    )
    write_csv(
        run_dir / "page_formula_qa.csv",
        [
            {
                "document_id": "doc",
                "page_number": 1,
                "detected_formula_blocks": 2,
                "formula_hashes": "first-a;first-b",
                "qa_severity": "fail",
                "qa_reasons": "unbalanced_braces;recoverable_display_equation_labels=(1.1)|(1.2)",
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
    ledger_row = {
        "document_id": "doc",
        "input_path": str(source_pdf),
        "page_number": 1,
        "attempt_id": "attempt-01",
        "attempt_number": 1,
        "dpi": 300,
        "model": "glm-ocr",
        "source_pdf_sha256": sha256(source_pdf),
        "expected_source_sha256": sha256(source_pdf),
        "rendered_image_path": str(rendered),
        "rendered_image_sha256": sha256(rendered),
        "first_pass_raw_response_path": str(first_raw),
        "first_pass_raw_sha256": sha256(first_raw),
        "first_pass_artifact_unchanged": True,
        "retry_raw_response_path": str(retry_raw),
        "retry_raw_sha256": sha256(retry_raw),
        "retry_markdown_path": str(retry_markdown),
        "retry_markdown_sha256": sha256(retry_markdown),
        "transport_status": "ok",
        "retry_qa_severity": "fail",
        "original_qa_reasons": "unbalanced_braces;recoverable_display_equation_labels=(1.1)|(1.2)",
        "retry_qa_reasons": "unbalanced_braces;unmapped_display_equation_labels=(1.1)|(1.2)",
        "first_pass_formula_hashes": "first-a;first-b",
        "retry_formula_hashes": "retry-a;retry-b",
        "resolution_status": "unresolved",
        "request_id": "ocr-request",
        "total_tokens": 12,
        "started_at_utc": "2026-01-01T00:00:00Z",
        "completed_at_utc": "2026-01-01T00:00:01Z",
        "error_type": "",
        "error": "",
    }
    write_csv(
        run_dir / "formula_retry_results.csv",
        [{field: ledger_row.get(field, "") for field in vision.retry.LEDGER_FIELDS}],
        vision.retry.LEDGER_FIELDS,
    )
    return run_dir, {
        "source": source_pdf,
        "first_raw": first_raw,
        "first_markdown": first_markdown,
        "rendered": rendered,
        "retry_raw": retry_raw,
        "retry_markdown": retry_markdown,
    }


class FakeResponse:
    status_code = 200
    text = ""

    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def json(self) -> dict[str, object]:
        return self.body


def test_structure_validation_is_formula_aware() -> None:
    assert vision.latex_structure_issues(r"x \rightarrow 0 + \left(a\right)") == []
    assert "unbalanced_left_right" in vision.latex_structure_issues(r"\left. x")
    assert "unbalanced_braces" in vision.latex_structure_issues(r"x_{1")
    assert vision.reason_equation_labels(
        {"qa_reasons": "recoverable_display_equation_labels=(1.1)|(1.2)"},
        {"retry_qa_reasons": "unmapped_display_equation_labels=(1.2)|(1.3)"},
    ) == ["(1.1)", "(1.2)", "(1.3)"]

    good = "$$x=1\\tag{1.1}$$\n\n$$y=2\\tag{1.2}$$"
    result = vision.validate_page_markdown(good, 2, ["(1.1)", "(1.2)"])
    assert result["resolution_status"] == "resolved"
    assert result["display_formula_count"] == 2

    missing = vision.validate_page_markdown("$$x=1\\tag{1.1}$$", 2, ["(1.1)", "(1.2)"])
    assert missing["resolution_status"] == "unresolved"
    assert "missing_equation_labels=(1.2)" in missing["qa_reasons"]

    extra = vision.validate_page_markdown(good + "\n\n$$z=3$$", 2, ["(1.1)", "(1.2)"])
    assert extra["resolution_status"] == "resolved"
    assert extra["qa_reasons"] == ""


def test_markdown_parser_handles_markers_and_fences() -> None:
    marked = f"preface\n{vision.BEGIN_MARKER}\n# Page\n\n$$x=1$$\n{vision.END_MARKER}\nafter"
    assert vision.parse_page_markdown(marked) == "# Page\n\n$$x=1$$"
    assert vision.parse_page_markdown("```markdown\n$$x=1$$\n```") == "$$x=1$$"


def test_mocked_vision_repair_uses_official_shape_and_preserves_artifacts(tmp_path: Path, monkeypatch) -> None:
    run_dir, artifacts = make_run(tmp_path)
    old_hashes = {name: sha256(path) for name, path in artifacts.items()}
    secret = "unit-test-secret-must-never-be-written"
    captured: dict[str, object] = {}
    markdown = """<BEGIN_MARKDOWN>
# Proposition

$$
x_{1}=1 \\tag{1.1}
$$

$$
y^{2}=2 \\tag{1.2}
$$
<END_MARKDOWN>"""
    body = {
        "id": "vision-request-1",
        "choices": [{"message": {"role": "assistant", "content": markdown}}],
        "usage": {"total_tokens": 55},
    }

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return FakeResponse(body)

    monkeypatch.setattr(vision.requests, "post", fake_post)
    monkeypatch.setattr(vision.ocr, "env_api_key", lambda: secret)

    assert vision.main(["--run-dir", str(run_dir), "--retries", "0"]) == 0
    assert captured["url"] == vision.DEFAULT_ENDPOINT
    assert captured["headers"]["Authorization"] == f"Bearer {secret}"
    payload = captured["json"]
    assert payload["model"] == "glm-5v-turbo"
    assert payload["thinking"] == {"type": "enabled"}
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    assert content[1]["type"] == "text"
    assert "Expected display equations: 2" in content[1]["text"]

    for name, path in artifacts.items():
        assert sha256(path) == old_hashes[name], name
    ledger = vision.retry.read_csv(run_dir / "formula_retry_results.csv")
    assert [row["attempt_id"] for row in ledger] == ["attempt-01", "vision-attempt-02"]
    assert ledger[-1]["resolution_status"] == "resolved"
    assert ledger[-1]["retry_qa_severity"] == "pass"
    assert ledger[-1]["model"] == "glm-5v-turbo"
    assert "/formula_vision_repairs/raw/" in ledger[-1]["retry_raw_response_path"]
    assert "/formula_vision_repairs/markdown/" in ledger[-1]["retry_markdown_path"]
    assert Path(ledger[-1]["retry_raw_response_path"]).exists()
    assert Path(ledger[-1]["retry_markdown_path"]).exists()
    assert "\\tag{1.1}" in Path(ledger[-1]["retry_markdown_path"]).read_text(encoding="utf-8")
    assert (run_dir / "markdown_verified" / "doc.md").exists()

    for path in run_dir.rglob("*"):
        if path.is_file() and path.suffix in {".csv", ".json", ".md"}:
            assert secret not in path.read_text(encoding="utf-8")

    # Resolved pages are resumable: a normal rerun must neither request a key nor call the API.
    monkeypatch.setattr(
        vision.ocr,
        "env_api_key",
        lambda: (_ for _ in ()).throw(AssertionError("API key must not be requested")),
    )
    monkeypatch.setattr(
        vision.requests,
        "post",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("API must not be called")),
    )
    assert vision.main(["--run-dir", str(run_dir)]) == 0
    assert len(vision.retry.read_csv(run_dir / "formula_retry_results.csv")) == 2
