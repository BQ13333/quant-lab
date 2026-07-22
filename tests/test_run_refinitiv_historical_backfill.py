from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


SCRIPT = Path(__file__).parents[1] / "examples" / "run_refinitiv_historical_backfill.py"
SPEC = importlib.util.spec_from_file_location("run_refinitiv_historical_backfill", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_required_plan_row_ledger_accepts_explicit_empty_csv(tmp_path: Path) -> None:
    ledger = tmp_path / "quarantine.csv"
    pd.DataFrame(columns=["plan_row_id"]).to_csv(ledger, index=False)

    assert MODULE._load_plan_row_ids(str(ledger), label="exclude", required=True) == set()


def test_required_plan_row_ledger_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MODULE._load_plan_row_ids(str(tmp_path / "missing.csv"), label="exclude", required=True)


def test_plan_row_ledger_rejects_missing_identifier_column(tmp_path: Path) -> None:
    ledger = tmp_path / "quarantine.csv"
    pd.DataFrame({"wrong_column": []}).to_csv(ledger, index=False)

    with pytest.raises(ValueError, match="missing plan_row_id"):
        MODULE._load_plan_row_ids(str(ledger), label="exclude", required=True)
