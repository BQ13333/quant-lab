from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.agents.usa_data_quality_execution_loop import UsaExecutionLoopConfig, write_usa_execution_loop_artifacts
from quant_lab.agents.usa_data_quality_loop import UsaDataQualityPaths


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _paths(root: Path) -> UsaDataQualityPaths:
    return UsaDataQualityPaths(
        research_universe_dir=root / "research",
        security_master_dir=root / "security_master",
        active_symbol_map_dir=root / "active_symbol_map",
        reference_symbol_map_dir=root / "reference_symbol_map",
        active_historical_dir=root / "active_hist",
        active_depth_dir=root / "active_depth",
        active_compaction_dir=root / "active_compact",
        reference_historical_dir=root / "reference_hist",
        reference_depth_dir=root / "reference_depth",
        reference_compaction_dir=root / "reference_compact",
        pit_index_constituents_dir=root / "pit_index",
        corporate_actions_dir=root / "corpact",
        adjusted_returns_dir=root / "adjusted",
        liquidity_backfill_dir=root / "liquidity",
        fundamentals_ibes_events_dir=root / "fund_ibes",
        fx_calendar_tca_dir=root / "fx_calendar_tca",
        leakage_qa_dir=root / "leakage",
        commands_log=root / "commands.log",
    )


def _seed(root: Path) -> UsaDataQualityPaths:
    paths = _paths(root)
    _write_json(
        paths.research_universe_dir / "manifest.json",
        {"security_master_rows": 323_960, "active_core_rows": 5_214, "reference_core_rows": 46_672},
    )
    _write_csv(paths.active_symbol_map_dir / "symbol_map.csv", [{"country": "United States", "exchange_code": "NYQ", "refinitiv_ric": "IBM.N"}])
    _write_csv(paths.reference_symbol_map_dir / "symbol_map.csv", [{"country": "United States", "exchange_code": "NAS", "refinitiv_ric": "OLD.O"}])
    _write_json(paths.active_historical_dir / "historical_run_manifest.json", {"planned_rows": 3, "completed_rows": 2})
    _write_csv(paths.active_historical_dir / "historical_plan.csv", [{"plan_row_id": "p1"}, {"plan_row_id": "p2"}, {"plan_row_id": "p3"}])
    _write_csv(
        paths.active_historical_dir / "historical_manifest.csv",
        [
            {"plan_row_id": "p1", "status": "completed", "price_columns": 1, "long_ohlcv_rows": 5},
            {
                "plan_row_id": "p2",
                "status": "completed_partial",
                "price_columns": 0,
                "long_ohlcv_rows": 0,
                "coverage_path": str(root / "coverage.csv"),
            },
        ],
    )
    _write_csv(
        paths.active_depth_dir / "overall.csv",
        [{"rows": 5, "tickers": 1, "volume_missing_pct": 0.0, "non_positive_close_rows": 0}],
    )
    _write_json(paths.active_compaction_dir / "manifest.json", {"status": "complete", "source_rows": 5, "output_rows": 5})
    _write_json(paths.reference_historical_dir / "historical_run_manifest.json", {"planned_rows": 2, "completed_rows": 0})
    _write_csv(paths.reference_historical_dir / "historical_plan.csv", [{"plan_row_id": "r1"}, {"plan_row_id": "r2"}])
    _write_json(paths.liquidity_backfill_dir / "manifest.json", {"planned_rows": 1, "completed_rows": 0, "failed_rows": 0, "history_rows": 0})
    paths.commands_log.write_text("uv run example\n", encoding="utf-8")
    return paths


def test_usa_execution_loop_blocks_live_jobs_when_vendor_throttled(tmp_path):
    paths = _seed(tmp_path)
    config = UsaExecutionLoopConfig(
        run_id="USA-EXEC-TEST",
        artifact_root=Path("exec"),
        project_root=tmp_path,
        paths=paths,
        vendor_health_status="throttled",
    )

    artifacts = write_usa_execution_loop_artifacts(config)

    queue = pd.read_csv(artifacts.execution_queue_path)
    quarantine = pd.read_csv(artifacts.zero_output_quarantine_path)
    historical_quarantine = pd.read_csv(artifacts.historical_quarantine_path)
    assert queue.loc[0, "owner_agent"] == "vendor_throttle_guard"
    assert queue.loc[0, "status"] == "blocked"
    assert "429" not in queue.loc[0, "command"]
    assert quarantine["plan_row_id"].tolist() == ["p2"]
    assert historical_quarantine["reason"].tolist() == ["zero_ohlcv_payload"]


def test_usa_execution_loop_prioritizes_large_liquid_snapshot_when_healthy(tmp_path):
    paths = _seed(tmp_path)
    config = UsaExecutionLoopConfig(
        run_id="USA-EXEC-HEALTHY",
        artifact_root=Path("exec"),
        project_root=tmp_path,
        paths=paths,
        vendor_health_status="healthy",
    )

    artifacts = write_usa_execution_loop_artifacts(config)

    queue = pd.read_csv(artifacts.execution_queue_path)
    snapshot = queue[queue["phase"].eq("large_liquid_snapshot")].iloc[0]
    build = queue[queue["phase"].eq("large_liquid_universe_build")].iloc[0]
    assert snapshot["status"] == "ready"
    assert "examples/run_usa_large_liquid_snapshot.py" in snapshot["command"]
    assert build["status"] == "queued_after_snapshot"
    assert "--min-market-cap-usd 1000000000" in build["command"]
    assert bool(snapshot["promotes_alpha"]) is False
