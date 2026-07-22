from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "integrations"
    / "cerebras_rag"
    / "rag.py"
)


def _module():
    spec = importlib.util.spec_from_file_location("quant_lab_cerebras_rag", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_split_is_bounded_and_preserves_headings():
    rag = _module()
    text = "# One\n\n" + ("alpha beta gamma\n" * 40) + "\n## Two\n\nfinal"
    chunks = rag.split_markdown(text, max_chars=180, overlap=20)

    assert len(chunks) > 2
    assert all(heading and content for heading, content in chunks)
    assert any(heading.startswith("One") for heading, _ in chunks)
    assert any(heading == "Two" for heading, _ in chunks)


def test_document_discovery_enforces_root_allowlist_and_deny_terms(tmp_path):
    rag = _module()
    public = tmp_path / "specialists" / "public"
    public.mkdir(parents=True)
    (public / "allowed.md").write_text("# Allowed\n\nPublic mathematics.")
    (public / "restricted.md").write_text("# Restricted\n\nContains Reuters data.")

    documents, excluded = rag.discover_documents(
        {
            "source_root": str(tmp_path),
            "allowed_globs": ["specialists/**/*.md"],
            "cloud_allowed_globs": [
                "specialists/public/*.md",
                "specialists/public/**/*.md",
            ],
            "deny_terms": ["Reuters"],
            "max_file_bytes": 4096,
        }
    )

    by_name = {Path(item["rel_path"]).name: item for item in documents}
    assert excluded == []
    assert by_name["allowed.md"]["cloud_allowed"] is True
    assert by_name["restricted.md"]["cloud_allowed"] is False
    assert "restricted term" in by_name["restricted.md"]["cloud_block_reason"]


def test_vector_blob_is_round_trippable():
    rag = _module()
    blob, dimension = rag.vector_blob([0.25, -0.5, 1.0])
    values = rag.array("f")
    values.frombytes(blob)

    assert dimension == 3
    assert list(values) == [0.25, -0.5, 1.0]
