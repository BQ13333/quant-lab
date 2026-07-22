from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_ohlcv_recovery_merge import merge_japan_ohlcv_recovery


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed(tmp_path: Path):
    stage_artifact = tmp_path / "stage_artifact"
    prod_artifact = tmp_path / "prod_artifact"
    stage_lake = tmp_path / "stage_lake"
    prod_lake = tmp_path / "prod_lake"
    for path in (stage_artifact, prod_artifact, stage_lake, prod_lake):
        path.mkdir()
    row_id = "japan-jpx-0001-001"
    common = {
        "plan_row_id": row_id, "run_id": "RUN", "country": "Japan", "exchange": "JPX",
        "batch_no": "1", "window_no": "1", "start_date": "2006-01-01",
        "end_date": "2006-12-31", "interval": "1d", "tickers": "1000.T",
        "local_symbols": "1000",
    }
    stage_paths = stage_lake / "prices/RUN/batch"
    prod_paths = prod_lake / "prices/RUN/batch"
    stage_paths.mkdir(parents=True)
    long_stage = stage_lake / "prices_long/RUN/ohlcv"
    long_prod = prod_lake / "prices_long/RUN/ohlcv"
    partition = (
        long_stage
        / "provider_partition=refinitiv"
        / "interval_partition=1d"
        / "adjustment_partition=vendor_default"
        / "country_partition=japan"
        / "exchange_partition=jpx"
        / "year_partition=2006"
        / "month_partition=1"
    )
    partition.mkdir(parents=True)
    payload = pd.DataFrame({"date": ["2006-01-04"], "ticker": ["1000.T"], "batch_id": [row_id]})
    payload.to_parquet(partition / "part.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(stage_paths / "prices.parquet", index=False)
    pd.DataFrame({"x": [1]}).to_parquet(stage_paths / "prices.ohlcv.parquet", index=False)
    pd.DataFrame({"ticker": ["1000.T"], "rows": [1], "start": ["2006-01-04"], "end": ["2006-01-04"]}).to_csv(stage_paths / "prices.coverage.csv", index=False)
    pd.DataFrame({"ticker": ["1000.T"], "rows": [1], "start": ["2006-01-04"], "end": ["2006-01-04"]}).to_csv(stage_paths / "prices.canonical_coverage.csv", index=False)
    stage_plan_row = {**common, "output_path": str(stage_paths / "prices.parquet"), "coverage_path": str(stage_paths / "prices.coverage.csv"), "ohlcv_path": str(stage_paths / "prices.ohlcv.parquet"), "long_ohlcv_root": str(long_stage)}
    prod_plan_row = {**common, "output_path": str(prod_paths / "prices.parquet"), "coverage_path": str(prod_paths / "prices.coverage.csv"), "ohlcv_path": str(prod_paths / "prices.ohlcv.parquet"), "long_ohlcv_root": str(long_prod)}
    pd.DataFrame([stage_plan_row]).to_csv(stage_artifact / "historical_plan.csv", index=False)
    pd.DataFrame([prod_plan_row]).to_csv(prod_artifact / "historical_plan.csv", index=False)
    stage_manifest = {**stage_plan_row, "canonical_coverage_path": str(stage_paths / "prices.canonical_coverage.csv"), "status": "completed_partial", "long_ohlcv_rows": 1, "failures": 1}
    prod_manifest = {**prod_plan_row, "canonical_coverage_path": "", "status": "failed_empty", "long_ohlcv_rows": 0, "failures": 1}
    pd.DataFrame([stage_manifest]).to_csv(stage_artifact / "historical_manifest.csv", index=False)
    pd.DataFrame([prod_manifest]).to_csv(prod_artifact / "historical_manifest.csv", index=False)
    pd.DataFrame([{"plan_row_id": row_id, "reason": "provider_failure"}]).to_csv(prod_artifact / "quarantined_plan_rows.csv", index=False)
    prod_paths.mkdir(parents=True)
    pd.DataFrame().to_parquet(prod_paths / "prices.parquet", index=False)
    pd.DataFrame({"ticker": ["1000.T"], "rows": [0], "start": [""], "end": [""]}).to_csv(prod_paths / "prices.coverage.csv", index=False)
    return stage_artifact, stage_lake, prod_artifact, prod_lake, row_id


def test_guarded_partial_recovery_merge(tmp_path: Path) -> None:
    stage_artifact, stage_lake, prod_artifact, prod_lake, row_id = _seed(tmp_path)
    result = merge_japan_ohlcv_recovery(
        run_id="MERGE-V1", plan_row_id=row_id,
        stage_artifact_dir=stage_artifact, stage_lake_root=stage_lake,
        production_artifact_dir=prod_artifact, production_lake_root=prod_lake,
        expected_stage_plan_sha256=_sha(stage_artifact / "historical_plan.csv"),
        expected_production_plan_sha256=_sha(prod_artifact / "historical_plan.csv"),
        artifact_root=tmp_path / "artifacts",
    )
    assert result.manifest["status"] == "merged_and_validated"
    manifest = pd.read_csv(prod_artifact / "historical_manifest.csv")
    assert manifest.iloc[0]["status"] == "completed_partial"
    assert int(manifest.iloc[0]["long_ohlcv_rows"]) == 1
    assert pd.read_csv(prod_artifact / "quarantined_plan_rows.csv").empty
    assert (stage_artifact / "guarded_merge_manifest.json").is_file()


def test_merge_rejects_plan_hash_drift(tmp_path: Path) -> None:
    stage_artifact, stage_lake, prod_artifact, prod_lake, row_id = _seed(tmp_path)
    with pytest.raises(ValueError, match="immutable hash mismatch"):
        merge_japan_ohlcv_recovery(
            run_id="MERGE-V1", plan_row_id=row_id,
            stage_artifact_dir=stage_artifact, stage_lake_root=stage_lake,
            production_artifact_dir=prod_artifact, production_lake_root=prod_lake,
            expected_stage_plan_sha256="0" * 64,
            expected_production_plan_sha256=_sha(prod_artifact / "historical_plan.csv"),
            artifact_root=tmp_path / "artifacts",
        )


def test_guarded_structural_recovery_merge(tmp_path: Path) -> None:
    stage_artifact, stage_lake, prod_artifact, prod_lake, row_id = _seed(tmp_path)
    stage_manifest_path = stage_artifact / "historical_manifest.csv"
    stage_manifest = pd.read_csv(stage_manifest_path, dtype=str, keep_default_na=False)
    stage_manifest.loc[0, "status"] = "completed_prelisting"
    stage_manifest.loc[0, "long_ohlcv_rows"] = "0"
    stage_manifest.loc[0, "structural_reason"] = "all_requested_tickers_first_trade_after_window"
    stage_manifest.loc[0, "ohlcv_path"] = ""
    stage_manifest.loc[0, "canonical_coverage_path"] = ""
    stage_manifest.to_csv(stage_manifest_path, index=False)
    stage_prices = Path(stage_manifest.loc[0, "output_path"])
    stage_coverage = Path(stage_manifest.loc[0, "coverage_path"])
    pd.DataFrame().to_parquet(stage_prices, index=False)
    pd.DataFrame({"ticker": ["1000.T"], "rows": [0], "start": [""], "end": [""]}).to_csv(
        stage_coverage, index=False
    )
    shutil.rmtree(stage_lake / "prices_long")

    result = merge_japan_ohlcv_recovery(
        run_id="MERGE-STRUCTURAL-V1",
        plan_row_id=row_id,
        stage_artifact_dir=stage_artifact,
        stage_lake_root=stage_lake,
        production_artifact_dir=prod_artifact,
        production_lake_root=prod_lake,
        expected_stage_plan_sha256=_sha(stage_artifact / "historical_plan.csv"),
        expected_production_plan_sha256=_sha(prod_artifact / "historical_plan.csv"),
        artifact_root=tmp_path / "artifacts",
    )

    assert result.manifest["recovery_kind"] == "structural_zero_payload"
    manifest = pd.read_csv(prod_artifact / "historical_manifest.csv")
    assert manifest.iloc[0]["status"] == "completed_prelisting"
    assert manifest.iloc[0]["structural_reason"] == "all_requested_tickers_first_trade_after_window"
    assert pd.read_csv(prod_artifact / "quarantined_plan_rows.csv").empty
