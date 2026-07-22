from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from quant_lab.agents.usa_data_quality_loop import UsaDataQualityPaths, run_usa_data_quality_loop


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
        fundamentals_ibes_events_dir=root / "fund_ibes",
        fx_calendar_tca_dir=root / "fx_calendar_tca",
        leakage_qa_dir=root / "leakage",
        commands_log=root / "commands.log",
    )


def _seed_incomplete_usa(root: Path) -> UsaDataQualityPaths:
    paths = _paths(root)
    _write_json(
        paths.research_universe_dir / "manifest.json",
        {
            "security_master_rows": 323_960,
            "active_core_rows": 5_214,
            "reference_core_rows": 46_672,
        },
    )
    _write_csv(
        paths.active_symbol_map_dir / "symbol_map.csv",
        [
            {"country": "United States", "exchange_code": "NYQ", "refinitiv_ric": "IBM.N"},
            {"country": "United States", "exchange_code": "NSQ", "refinitiv_ric": "AAPL.O"},
        ],
    )
    _write_csv(
        paths.reference_symbol_map_dir / "symbol_map.csv",
        [
            {"country": "United States", "exchange_code": "NYQ", "refinitiv_ric": "IBM.N"},
            {"country": "United States", "exchange_code": "NAS", "refinitiv_ric": "OLD.O"},
        ],
    )
    _write_json(paths.active_historical_dir / "historical_run_manifest.json", {"planned_rows": 10, "completed_rows": 5})
    _write_csv(paths.active_historical_dir / "historical_plan.csv", [{"plan_row_id": f"p{i}"} for i in range(10)])
    _write_csv(
        paths.active_historical_dir / "historical_manifest.csv",
        [
            {"plan_row_id": "p1", "status": "completed", "price_columns": 1, "long_ohlcv_rows": 10},
            {"plan_row_id": "p2", "status": "completed_partial", "price_columns": 0, "long_ohlcv_rows": 0},
        ],
    )
    _write_csv(
        paths.active_depth_dir / "overall.csv",
        [
            {
                "rows": 59_559,
                "tickers": 20,
                "volume_missing_pct": 0.0,
                "non_positive_close_rows": 0,
            }
        ],
    )
    _write_json(
        paths.active_compaction_dir / "manifest.json",
        {"status": "complete", "source_rows": 59_559, "output_rows": 59_559},
    )
    _write_csv(paths.reference_historical_dir / "historical_plan.csv", [{"plan_row_id": f"r{i}"} for i in range(30)])
    paths.commands_log.write_text("uv run example\n", encoding="utf-8")
    return paths


def test_usa_data_quality_loop_blocks_until_all_five_gates_pass(tmp_path):
    paths = _seed_incomplete_usa(tmp_path)

    result = run_usa_data_quality_loop(paths=paths, artifact_root=tmp_path / "scorecards", run_id="USA-DQ-TEST")

    assert result.accepted is False
    assert result.status == "continue_usa_data_quality_loop"
    assert result.summary_path.exists()
    assert result.lane_scorecard_path.exists()
    assert result.next_queue_path.exists()

    lanes = pd.read_csv(result.lane_scorecard_path).set_index("lane_id")
    assert lanes.loc["g1_security_master_universe", "status"] == "pass"
    assert lanes.loc["g2_prices_actions_liquidity", "status"] == "partial"
    assert lanes.loc["g3_pit_index_fundamentals_ibes_events", "status"] == "fail"

    next_queue = pd.read_csv(result.next_queue_path)
    assert "official PIT index constituent artifact missing" in set(next_queue["blocker"])
    assert "active-core OHLCV incomplete: 1/10" in set(next_queue["blocker"])
    assert "active-core OHLCV has completed rows with zero output: 1" in set(next_queue["blocker"])


def test_usa_data_quality_loop_rejects_placeholder_manifests(tmp_path):
    paths = _seed_incomplete_usa(tmp_path)
    _write_json(
        paths.pit_index_constituents_dir / "manifest.json",
        {"status": "incomplete", "scorecard_eligible": False},
    )
    _write_json(
        paths.fx_calendar_tca_dir / "manifest.json",
        {"status": "incomplete", "scorecard_eligible": False, "production_ready": False},
    )
    _write_json(paths.corporate_actions_dir / "manifest.json", {"status": "complete"})
    _write_json(paths.adjusted_returns_dir / "manifest.json", {"status": "pass"})
    _write_json(paths.fundamentals_ibes_events_dir / "manifest.json", {"planned_rows": 1, "completed_rows": 0})
    _write_csv(
        paths.fundamentals_ibes_events_dir / "historical_quality.csv",
        [{"failed_rows": 0, "history_rows": 0, "min_coverage": 0}],
    )
    _write_json(
        paths.leakage_qa_dir / "manifest.json",
        {
            "status": "pass",
            "total_leakage_violations": 0,
            "decision_rows": 0,
            "decision_entities": 0,
            "features": [],
        },
    )

    result = run_usa_data_quality_loop(paths=paths, artifact_root=tmp_path / "scorecards", run_id="USA-DQ-PLACEHOLDER")
    blockers = set(pd.read_csv(result.next_queue_path)["blocker"])

    assert result.accepted is False
    assert "official PIT index constituent artifact missing" in blockers
    assert "USA exchange calendar/halts/suspensions/TCA artifact missing" in blockers
    assert "official USA corporate actions route/backfill missing" in blockers
    assert "USA adjusted/raw price reconciliation missing" in blockers
    assert "USA PIT fundamentals/IBES/events backfill missing" in blockers
    assert "USA PIT feature coverage/quality evidence missing" in blockers
    assert "USA leakage QA did not pass cleanly" in blockers
