from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.ohlcv_row_equality import validate_ohlcv_row_equality


def _write(root: Path, year: int, rows: list[dict]) -> None:
    path = root / f"year_partition={year}"
    path.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(path / "part.parquet", index=False)


def test_row_equality_passes_reordered_files_and_fails_changed_values(tmp_path: Path) -> None:
    source = tmp_path / "source"
    compacted = tmp_path / "compacted"
    rows = [
        {"ticker": "AAA.T", "date": "2020-01-02", "close": 2.0, "batch_id": "B1"},
        {"ticker": "AAA.T", "date": "2020-01-01", "close": 1.0, "batch_id": "B1"},
    ]
    _write(source, 2020, rows)
    _write(compacted, 2020, list(reversed(rows)))

    passed = validate_ohlcv_row_equality(
        source_root=source,
        compacted_root=compacted,
        artifact_root=tmp_path / "artifacts",
        run_id="PASS",
    )
    assert passed.passed is True
    assert passed.manifest["source_rows"] == 2

    changed = tmp_path / "changed"
    changed_rows = [dict(row) for row in rows]
    changed_rows[0]["close"] = 3.0
    _write(changed, 2020, changed_rows)
    failed = validate_ohlcv_row_equality(
        source_root=source,
        compacted_root=changed,
        artifact_root=tmp_path / "artifacts",
        run_id="FAIL",
    )
    assert failed.passed is False
    assert failed.manifest["failed_years"] == ["2020"]

    with pytest.raises(FileExistsError, match="immutable"):
        validate_ohlcv_row_equality(
            source_root=source,
            compacted_root=compacted,
            artifact_root=tmp_path / "artifacts",
            run_id="PASS",
        )
