from __future__ import annotations

import pandas as pd
import pytest

from quant_lab.research.artifact_writer import (
    write_cost_sensitivity_md,
    write_failure_modes_md,
    write_index_md,
    write_model_card_md,
    write_next_research_queue_md,
    write_validation_report_md,
)


def test_write_index_md(tmp_path):
    lb = pd.DataFrame({"model_name": ["m1"], "sharpe": [1.0]})
    coverage = pd.DataFrame({"ticker": ["A"], "rows": [100]})
    eligible = pd.DataFrame({"A": [1, 2, 3]})
    paths = {"summary": tmp_path / "summary.csv"}
    path = write_index_md("RUN-1", tmp_path, lb, coverage, eligible, paths, tmp_path)
    assert path.exists()
    text = path.read_text()
    assert "RUN-1" in text
    assert "Best Result" in text
    assert "Caveats" in text
    assert "| model_name |" in text


def test_write_validation_report_md(tmp_path):
    lb = pd.DataFrame({"model_name": ["m1"], "sharpe": [1.0]})
    coverage = pd.DataFrame({"ticker": ["A"], "rows": [100]})
    failures = {"tier1": {"A": "missing data"}}
    path = write_validation_report_md("RUN-1", tmp_path, lb, coverage, failures, tmp_path)
    assert path.exists()
    text = path.read_text()
    assert "Leaderboard" in text
    assert "Coverage Snapshot" in text
    assert "Failures" in text
    assert "Promotion Gate Outcomes" in text
    assert "missing data" in text


def test_write_cost_sensitivity_md(tmp_path):
    cs = pd.DataFrame({"cost_bps": [0, 5], "net_sharpe": [1.0, 0.9]})
    best = pd.DataFrame({"cost_bps": [0], "model_name": ["m1"]})
    path = write_cost_sensitivity_md("RUN-1", cs, best, tmp_path)
    assert path.exists()
    text = path.read_text()
    assert "Cost Sensitivity" in text
    assert "Best By Cost" in text
    assert "Interpretation Prompts" in text
    assert "| cost_bps |" in text


def test_write_model_card_md(tmp_path):
    card = pd.DataFrame({"sharpe": [1.0], "gate_passed": [True]})
    path = write_model_card_md("RUN-1", "m1", card, tmp_path)
    assert path.exists()
    text = path.read_text()
    assert "m1" in text
    assert "Model Statistics" in text
    assert "Caveats" in text
    assert "| sharpe |" in text


def test_write_failure_modes_md(tmp_path):
    path = write_failure_modes_md(
        "RUN-1", ["m1", "m2"], ["low sharpe", "high turnover"], tmp_path
    )
    assert path.exists()
    text = path.read_text()
    assert "Failed Models" in text
    assert "low sharpe" in text
    assert "high turnover" in text


def test_write_failure_modes_md_empty(tmp_path):
    path = write_failure_modes_md("RUN-1", [], [], tmp_path)
    assert path.exists()
    text = path.read_text()
    assert "No models failed" in text


def test_write_next_research_queue_md(tmp_path):
    path = write_next_research_queue_md(
        "RUN-1", ["m1"], ["improve data quality", "add stress tests"], tmp_path
    )
    assert path.exists()
    text = path.read_text()
    assert "Passed Models" in text
    assert "Recommended Next Steps" in text
    assert "improve data quality" in text


def test_write_next_research_queue_md_empty(tmp_path):
    path = write_next_research_queue_md("RUN-1", [], [], tmp_path)
    assert path.exists()
    text = path.read_text()
    assert "No models passed" in text
    assert "No specific gaps" in text
