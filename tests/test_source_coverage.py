from __future__ import annotations

import pandas as pd

from quant_lab.research.source_coverage import (
    markdown_file_inventory,
    mine_markdown_headings,
    source_file_coverage,
    source_strategy_spec_coverage,
    write_markdown_source_coverage,
)


def test_mine_markdown_headings_classifies_strategy_and_data_dependency(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    source.joinpath("book.md").write_text(
        "# Quant Book\n"
        "## Strategy: Price Momentum\n"
        "## Earnings Revisions Alpha\n"
        "## Plain Chapter\n",
        encoding="utf-8",
    )

    headings = mine_markdown_headings(source)

    assert len(headings) == 4
    categories = dict(zip(headings["heading"], headings["category"], strict=False))
    assert categories["Strategy: Price Momentum"] == "strategy_or_model"
    assert categories["Earnings Revisions Alpha"] == "strategy_or_model_requires_data"
    assert categories["Plain Chapter"] == "general"


def test_write_markdown_source_coverage_writes_expected_artifacts(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    source.joinpath("book.md").write_text("# Book\n## Strategy: Channel\n", encoding="utf-8")

    artifact_dir = write_markdown_source_coverage(source, artifact_root=tmp_path / "artifacts", run_id="TEST")

    assert artifact_dir.joinpath("markdown_files.csv").exists()
    assert artifact_dir.joinpath("heading_inventory.csv").exists()
    assert artifact_dir.joinpath("strategy_heading_inventory.csv").exists()
    assert artifact_dir.joinpath("source_strategy_spec_coverage.csv").exists()
    assert artifact_dir.joinpath("source_file_coverage.csv").exists()
    assert artifact_dir.joinpath("source_coverage_summary.md").exists()
    assert artifact_dir.joinpath("source_coverage_manifest.json").exists()


def test_markdown_inventory_excludes_quant_lab_by_default(tmp_path):
    source = tmp_path / "sources"
    source.mkdir()
    source.joinpath("book.md").write_text("# Book\n", encoding="utf-8")
    repo = source / "quant_lab"
    repo.mkdir()
    repo.joinpath("docs.md").write_text("# Repo Docs\n", encoding="utf-8")

    files = markdown_file_inventory(source)

    assert list(files["file"]) == ["book.md"]


def test_source_file_coverage_counts_uncovered_headings():
    files = pd.DataFrame(
        [{"file": "book.md", "scope": "root_book", "bytes": 10, "line_count": 2, "heading_count": 2}]
    )
    strategy_headings = pd.DataFrame(
        [
            {"file": "book.md", "heading": "Strategy A"},
            {"file": "book.md", "heading": "Strategy B"},
        ]
    )
    spec_coverage = pd.DataFrame(
        [
            {
                "source_file": "book.md",
                "file_exists": True,
                "coverage_status": "implemented",
            }
        ]
    )

    coverage = source_file_coverage(files, strategy_headings, spec_coverage)

    assert coverage.loc[0, "strategy_heading_count"] == 2
    assert coverage.loc[0, "mapped_spec_count"] == 1
    assert coverage.loc[0, "uncovered_strategy_heading_count"] == 1
