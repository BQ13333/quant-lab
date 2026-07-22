from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.bloomberg_malaysia_bulk import (
    BloombergBulkFileSpec,
    _build_candidate_returns,
    _frozen_pit_gate_projection,
    load_source_config,
    normalize_and_curate_source_file,
)


def _write_source(path: Path) -> None:
    pd.DataFrame(
        [
            {
                "DATE": "2022-01-03",
                "ID_BBG": "NEW MK Equity",
                "ID": "B2",
                "NAME": "New Bhd",
                "MARKET_STATUS": "ACTV",
                "SECURITY_TYP": "Common Stock",
                "PX_LAST": "",
                "PX_VOLUME": "0",
            },
            {
                "DATE": "2022-01-04",
                "ID_BBG": "NEW MK Equity",
                "ID": "B2",
                "NAME": "New Bhd",
                "MARKET_STATUS": "ACTV",
                "SECURITY_TYP": "Common Stock",
                "PX_LAST": "",
                "PX_VOLUME": "0",
            },
            {
                "DATE": "2022-01-04",
                "ID_ISIN": "MYL0001OO001",
                "ID_BBG": "NEW MK Equity",
                "ID": "B1",
                "NAME": "New Bhd",
                "MARKET_STATUS": "ACTV",
                "SECURITY_TYP": "Common Stock",
                "PX_LAST": "1.00",
                "PX_OPEN": "0.95",
                "PX_HIGH": "1.05",
                "PX_LOW": "0.94",
                "PX_LAST_RAW": "1.00",
                "PX_VOLUME": "1000",
                "CASH_DIVS": "0",
                "CUR_MKT_CAP": "1000000",
            },
            {
                "DATE": "2022-01-05",
                "ID_ISIN": "MYL0001OO001",
                "ID_BBG": "NEW MK Equity",
                "ID": "B1",
                "NAME": "New Bhd",
                "MARKET_STATUS": "ACTV",
                "SECURITY_TYP": "Common Stock",
                "PX_LAST": "1.10",
                "PX_OPEN": "1.00",
                "PX_HIGH": "1.10",
                "PX_LOW": "0.99",
                "PX_LAST_RAW": "1.10",
                "PX_VOLUME": "1200",
                "CASH_DIVS": "0.01",
                "CUR_MKT_CAP": "1100000",
            },
        ]
    ).to_csv(path, index=False)


def test_normalization_preserves_raw_rows_and_curates_one_security_date(tmp_path: Path) -> None:
    raw = tmp_path / "source.csv"
    _write_source(raw)
    spec = BloombergBulkFileSpec("2020-2022", raw.name, "file-id", raw.stat().st_size)
    normalized = tmp_path / "normalized" / "normalized.parquet"
    market = tmp_path / "curated" / "market.parquet"

    profile, summary, identifiers = normalize_and_curate_source_file(
        spec,
        raw,
        normalized,
        market,
        chunk_rows=2,
    )

    assert profile["raw_rows"] == 4
    assert profile["deduplicated_rows"] == 3
    assert profile["research_market_rows"] == 2
    assert profile["duplicate_security_date_rows"] == 2
    assert profile["duplicate_security_date_keys"] == 1
    assert profile["pre_price_placeholder_rows"] == 1
    curated = pd.read_parquet(market).sort_values("date")
    assert curated["date"].dt.strftime("%Y-%m-%d").tolist() == ["2022-01-04", "2022-01-05"]
    assert curated["adj_close"].tolist() == [1.0, 1.1]
    assert not curated.duplicated(["date", "security_key"]).any()
    assert summary.iloc[0]["first_valid_price_date"] == pd.Timestamp("2022-01-04")
    assert summary.iloc[0]["security_key"] == "bbg_ticker:NEW MK Equity"
    assert len(summary) == 1
    assert set(identifiers["id_type"]) == {"isin", "bbg_ticker", "bbg_id"}


def test_candidate_returns_carry_prior_close_across_period_files(tmp_path: Path) -> None:
    columns = [
        "source_period",
        "source_row_number",
        "date",
        "security_key",
        "ticker",
        "adj_close",
        "cash_dividend_adjusted",
    ]
    first = pd.DataFrame(
        [["2020-2022", 1, pd.Timestamp("2022-12-30"), "S1", "AAA", 10.0, 0.0]],
        columns=columns,
    )
    second = pd.DataFrame(
        [["2023-2025", 1, pd.Timestamp("2023-01-03"), "S1", "AAA", 11.0, 0.0]],
        columns=columns,
    )
    p1 = tmp_path / "market_1.parquet"
    p2 = tmp_path / "market_2.parquet"
    first.to_parquet(p1, index=False)
    second.to_parquet(p2, index=False)

    paths, profile = _build_candidate_returns((p1, p2), tmp_path / "returns")

    carried = pd.read_parquet(paths[1])
    assert carried.iloc[0]["price_return_local_candidate"] == pytest.approx(0.10)
    assert profile["production_adjusted_return_ready"] is False


def test_source_config_cannot_authorize_execution(tmp_path: Path) -> None:
    config = {
        "country": "Malaysia",
        "country_code": "MYS",
        "research_controls": {"strategy_execution_authorized": True},
        "files": [
            {
                "period": str(index),
                "filename": f"f{index}.csv",
                "drive_file_id": f"id{index}",
                "expected_bytes": 1,
            }
            for index in range(8)
        ],
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="must not authorize"):
        load_source_config(path)


def test_frozen_gate_projection_fails_closed_on_partial_dead_name_coverage() -> None:
    security_master = pd.DataFrame(
        {
            "security_key": ["A", "B", "C"],
            "active_candidate": [True, True, False],
        }
    )
    institutional = {
        "historical_u2_rule_frozen": False,
        "terminal_delisting_economics_reconciled": False,
        "production_adjusted_returns_ready": False,
        "fundamentals_point_in_time_proven": False,
        "analyst_estimate_revision_timestamps_proven": False,
        "minimum_95pct_dead_name_coverage": False,
    }

    projection = _frozen_pit_gate_projection(
        security_master=security_master,
        dead_coverage={
            "strategy_relevant_inactive_entities": 10,
            "strategy_relevant_high_confidence_matches": 7,
        },
        institutional_gates=institutional,
    )

    assert projection["admissible"] is False
    assert projection["inputs"]["candidate_signal_and_return_coverage"] == pytest.approx(0.7)
    assert "historical_u2_membership_reconstructable" in projection["failed_gate_ids"]
    assert "minimum_dead_name_return_coverage" in projection["failed_gate_ids"]
    assert projection["strategy_trials_reserved"] is False
