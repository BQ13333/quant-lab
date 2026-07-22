from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.data_lake import wide_ohlcv_to_long, write_ohlcv_long_partitions
from quant_lab.data.lake_ops import (
    compact_long_ohlcv_partitions,
    scan_parquet_file_inventory,
    write_partition_health_report,
)


def _ohlcv_frame() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-01", "2024-02-02"])
    data = {
        ("AAA.KL", "Open"): [10.0, 11.0, 12.0, 13.0],
        ("AAA.KL", "High"): [11.0, 12.0, 13.0, 14.0],
        ("AAA.KL", "Low"): [9.0, 10.0, 11.0, 12.0],
        ("AAA.KL", "Close"): [10.5, 11.5, 12.5, 13.5],
        ("AAA.KL", "Volume"): [1000.0, 1100.0, 1200.0, 1300.0],
    }
    return pd.DataFrame(data, index=dates)


def _listings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"],
            "yahoo_ticker": ["AAA.KL"],
            "country": ["Malaysia"],
            "exchange": ["Bursa Malaysia"],
            "currency": ["MYR"],
        }
    )


def test_partition_health_reports_too_many_monthly_daily_files(tmp_path):
    long = wide_ohlcv_to_long(_ohlcv_frame(), listings=_listings(), provider="mock", run_id="RUN")
    root = write_ohlcv_long_partitions(long, root=tmp_path / "lake", run_id="RUN")

    report = write_partition_health_report(
        root,
        artifact_root=tmp_path / "artifacts",
        run_id="HEALTH",
        min_file_size_bytes=0,
        max_files_per_partition=1,
    )

    assert report.file_inventory_path.exists()
    assert report.partition_health_path.exists()
    assert int(report.manifest["parquet_files"]) == 2
    assert int(report.partition_health.loc[0, "parquet_files"]) == 2
    assert report.partition_health.loc[0, "status"] == "warn"
    assert report.partition_health.loc[0, "blocker"] == "too_many_files_for_partition"


def test_compaction_writes_year_level_output_and_manifest(tmp_path):
    long = wide_ohlcv_to_long(_ohlcv_frame(), listings=_listings(), provider="mock", run_id="RUN")
    root = write_ohlcv_long_partitions(long, root=tmp_path / "lake", run_id="RUN")

    result = compact_long_ohlcv_partitions(
        root,
        artifact_root=tmp_path / "artifacts",
        run_id="COMPACT",
        overwrite=True,
        min_file_size_bytes=0,
        max_files_per_partition=1,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    inventory = scan_parquet_file_inventory(result.output_root)
    stored = pd.read_parquet(result.output_root)

    assert result.compaction_manifest_path.exists()
    assert manifest["source_files"] == 2
    assert manifest["output_files"] == 1
    assert manifest["source_rows"] == 4
    assert manifest["output_rows"] == 4
    assert inventory["relative_path"].str.contains("month_partition").sum() == 0
    assert stored.shape[0] == 4
    assert set(stored["ticker"]) == {"AAA.KL"}
