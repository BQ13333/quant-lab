from __future__ import annotations

import csv
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_wave5_hmm_manifest.py"
WORKSPACE = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(
    not (WORKSPACE / "PhD_Math_Agent_Library/indexes/corpus_index.csv").is_file(),
    reason="private specialist-corpus integration test",
)


def test_builds_frozen_wave5_basket(tmp_path: Path) -> None:
    output = tmp_path / "wave5"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--workspace-root", str(WORKSPACE), "--output-dir", str(output)],
        check=False, capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    with (output / "candidate_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 22
    assert Counter(row["admission_class"] for row in rows if row["selection_status"] == "accepted") == {
        "core": 16,
        "red_team": 2,
    }
    assert sum(row["selection_status"] == "watchlist" for row in rows) == 4
    assert len({row["pdf_url"] for row in rows}) == 22
    assert all(row["commercial_reuse_status"] == "review_required" for row in rows)
