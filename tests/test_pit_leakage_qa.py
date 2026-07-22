from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.pit_leakage_qa import (
    build_decision_lagged_pit_panel,
    load_l2_backtest_decision_calendar,
    summarize_pit_leakage,
    write_pit_leakage_qa_artifacts,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "asof_date": "2024-01-31",
                "requested_ric": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "sid": "MY:AAA",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "dataset_id": "liquidity_market_cap_free_float",
                "field_role": "market_cap",
                "value": 100.0,
                "inside_requested_window": True,
            },
            {
                "asof_date": "2024-01-31",
                "requested_ric": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "sid": "MY:AAA",
                "ticker": "AAA.KL",
                "country": "Malaysia",
                "dataset_id": "fundamentals_pit",
                "field_role": "revenue",
                "value": 9.0,
                "inside_requested_window": True,
            },
        ]
    )


def test_build_decision_lagged_pit_panel_uses_external_calendar():
    decisions = pd.DataFrame(
        {
            "requested_ric": ["AAA.KL", "AAA.KL", "BBB.KL"],
            "decision_date": ["2024-02-29", "2024-04-30", "2024-04-30"],
            "decision_country": ["Malaysia", "Malaysia", "Malaysia"],
        }
    )

    panel = build_decision_lagged_pit_panel(_history(), decisions)

    assert len(panel) == 3
    assert panel["requested_ric"].nunique() == 2
    assert panel.loc[panel["decision_date"].eq(pd.Timestamp("2024-02-29")), "liquidity_market_cap_free_float__market_cap"].iloc[0] == 100.0
    assert pd.isna(panel.loc[panel["decision_date"].eq(pd.Timestamp("2024-02-29")), "fundamentals_pit__revenue"].iloc[0])
    assert panel.loc[panel["decision_date"].eq(pd.Timestamp("2024-04-30")), "fundamentals_pit__revenue"].notna().sum() == 1


def test_summarize_pit_leakage_flags_available_after_decision():
    panel = pd.DataFrame(
        {
            "requested_ric": ["AAA.KL"],
            "decision_date": [pd.Timestamp("2024-02-29")],
            "fundamentals_pit__revenue": [9.0],
            "fundamentals_pit__revenue__source_asof_date": [pd.Timestamp("2024-01-31")],
            "fundamentals_pit__revenue__source_available_date": [pd.Timestamp("2024-03-01")],
        }
    )

    summary, violations = summarize_pit_leakage(panel)

    assert int(summary["leakage_violations"].sum()) == 1
    assert len(violations) == 1
    assert violations["feature"].iloc[0] == "fundamentals_pit__revenue"


def test_write_pit_leakage_qa_artifacts(tmp_path):
    decisions = pd.DataFrame(
        {
            "requested_ric": ["AAA.KL", "AAA.KL"],
            "decision_date": ["2024-02-29", "2024-04-30"],
        }
    )

    artifacts = write_pit_leakage_qa_artifacts(
        history=_history(),
        decisions=decisions,
        artifact_root=tmp_path,
        run_id="TEST-LEAKAGE-QA",
    )

    assert artifacts.decision_panel_path.exists()
    assert artifacts.feature_quality_path.exists()
    assert artifacts.leakage_summary_path.exists()
    assert artifacts.summary_path.exists()
    assert int(artifacts.leakage_summary["leakage_violations"].sum()) == 0


def test_load_l2_backtest_decision_calendar_rebuilds_u2_matrix(tmp_path):
    l2_dir = tmp_path / "l2"
    l2_dir.mkdir()
    long_path = tmp_path / "ohlcv.parquet"
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    rows = []
    for ticker in ["AAA.KL", "BBB.KL", "CCC.KL"]:
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "sid": f"MY:{ticker}",
                    "local_symbol": ticker.split(".")[0],
                    "name": ticker,
                    "country": "Malaysia",
                    "exchange": "Bursa Malaysia",
                    "currency": "MYR",
                    "adj_close": 1.0,
                }
            )
    pd.DataFrame(rows).to_parquet(long_path, index=False)
    tiers = pd.DataFrame(
        {
            "ticker": ["AAA.KL", "BBB.KL", "CCC.KL"],
            "country": ["Malaysia", "Malaysia", "Malaysia"],
            "u2_backtest_eligible": [True, True, False],
            "u3_liquid_proxy": [True, False, False],
        }
    )
    tiers.to_csv(l2_dir / "eligible_universe_tiers.csv", index=False)
    (l2_dir / "manifest.json").write_text(
        json.dumps(
            {
                "long_ohlcv_root": str(long_path),
                "min_rows": 3,
                "max_missing_ratio": 0.5,
            }
        ),
        encoding="utf-8",
    )

    decisions = load_l2_backtest_decision_calendar(l2_dir, universe_tier="u2")

    assert set(decisions["requested_ric"]) == {"AAA.KL", "BBB.KL"}
    assert len(decisions) == 10
    assert decisions["decision_date"].min() == pd.Timestamp("2024-01-01")
