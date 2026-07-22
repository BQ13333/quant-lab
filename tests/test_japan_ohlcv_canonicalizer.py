from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_ohlcv_canonicalizer import canonicalize_japan_ohlcv


def test_canonicalizer_clips_windows_and_publishes_partial_candidate(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw" / "year_partition=2020" / "month_partition=1"
    raw_root.mkdir(parents=True)
    pd.DataFrame(
        {
            "ticker": ["AAA.T", "AAA.T", "BBB.T"],
            "date": ["2019-12-31", "2020-01-02", "2020-01-02"],
            "close": [9.0, 10.0, 20.0],
            "volume": [100.0, 110.0, 200.0],
            "batch_id": ["ROW-1", "ROW-1", "ROW-1"],
        }
    ).to_parquet(raw_root / "part.parquet", index=False)
    windows_path = tmp_path / "windows.parquet"
    pd.DataFrame(
        {
            "refinitiv_ric": ["AAA.T", "BBB.T"],
            "listing_start": ["2020-01-01", "2020-01-01"],
            "listing_end": ["2020-12-31", "2020-12-31"],
            "right_open": [False, False],
            "pit_usable": [True, True],
        }
    ).to_parquet(windows_path, index=False)
    historical = tmp_path / "historical"
    historical.mkdir()
    pd.DataFrame({"plan_row_id": ["ROW-1"]}).to_csv(historical / "historical_plan.csv", index=False)
    pd.DataFrame({"plan_row_id": ["ROW-1"], "status": ["completed_partial"]}).to_csv(
        historical / "historical_manifest.csv", index=False
    )
    (historical / "historical_run_manifest.json").write_text(
        json.dumps({"run_id": "SOURCE", "planned_rows": 2, "completed_rows": 1, "failed_rows": 0}),
        encoding="utf-8",
    )

    result = canonicalize_japan_ohlcv(
        raw_root=tmp_path / "raw",
        listing_windows_path=windows_path,
        historical_dir=historical,
        artifact_root=tmp_path / "artifacts",
        run_id="CANONICAL",
    )

    assert result.manifest["status"] == "pass_partial_candidate"
    assert result.manifest["promotion_eligible"] is False
    assert result.manifest["source_rows"] == 3
    assert result.manifest["canonical_rows"] == 2
    assert result.manifest["outside_listing_window_rows_removed"] == 1
    assert result.manifest["unknown_ric_rows"] == 0
    output = pd.read_parquet(result.data_root)
    assert sorted(output["ticker"].tolist()) == ["AAA.T", "BBB.T"]
    inventory = json.loads(result.hash_inventory_path.read_text())
    for name, digest in inventory["files"].items():
        path = Path(name)
        if not path.is_absolute():
            path = result.artifact_dir / path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest

    with pytest.raises(FileExistsError, match="immutable"):
        canonicalize_japan_ohlcv(
            raw_root=tmp_path / "raw",
            listing_windows_path=windows_path,
            historical_dir=historical,
            artifact_root=tmp_path / "artifacts",
            run_id="CANONICAL",
        )
