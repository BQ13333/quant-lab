from __future__ import annotations

import json

import pandas as pd

from quant_lab.data.security_master_pit import write_security_master_pit_seed_artifacts


def _selected_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "sid": "Malaysia:AAA",
                "local_symbol": "AAA",
                "name": "AAA Bhd",
                "country": "Malaysia",
                "exchange": "Bursa",
                "currency": "MYR",
                "selected_for_u2": True,
                "selected_for_u3": True,
            },
            {
                "ticker": "BBB.SI",
                "refinitiv_ric": "BBB.SI",
                "sid": "Singapore:BBB",
                "local_symbol": "BBB",
                "name": "BBB Ltd",
                "country": "Singapore",
                "exchange": "SGX",
                "currency": "SGD",
                "selected_for_u2": True,
                "selected_for_u3": False,
            },
        ]
    )


def _current_snapshot() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "requested_ric": "AAA.KL",
                "refinitiv_ric": "AAA.KL",
                "asof_date": "2026-07-08",
                "First Trade Date": "2001-01-01",
                "ISIN": "MYAAA",
                "SEDOL": "111111",
                "Organization PermID": "1001",
            },
            {
                "requested_ric": "BBB.SI",
                "refinitiv_ric": "BBB.SI",
                "asof_date": "2026-07-08",
                "First Trade Date": "2002-01-01",
                "ISIN": "SGBBB",
                "SEDOL": "222222",
                "Organization PermID": "1002",
            },
        ]
    )


def test_security_master_pit_seed_writes_required_shape_without_promotion(tmp_path):
    artifacts = write_security_master_pit_seed_artifacts(
        selected_universe=_selected_universe(),
        current_snapshot=_current_snapshot(),
        artifact_root=tmp_path,
        run_id="SECURITY-SEED",
    )

    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert artifacts.entities_path.exists()
    assert artifacts.identifier_history_path.exists()
    assert artifacts.listing_windows_path.exists()
    assert artifacts.universe_membership_path.exists()
    assert artifacts.quality_path.exists()
    assert artifacts.promotion_gates_path.exists()
    assert manifest["status"] == "current_survivor_seed_only"
    assert manifest["model_denominator_ready"] is False
    assert manifest["dead_name_rows"] == 0
    assert not artifacts.listing_windows["eligible_for_model_denominator"].any()
    assert not artifacts.universe_membership["eligible_flag"].any()


def test_security_master_pit_seed_quality_keeps_dead_name_gate_red(tmp_path):
    artifacts = write_security_master_pit_seed_artifacts(
        selected_universe=_selected_universe(),
        current_snapshot=_current_snapshot(),
        artifact_root=tmp_path,
        run_id="SECURITY-SEED",
    )

    gates = artifacts.promotion_gates.set_index("gate_id")
    quality = artifacts.quality.set_index("country")

    assert gates.loc["current_snapshot_not_promoted", "status"] == "pass"
    assert gates.loc["no_current_survivor_only_universe", "status"] == "fail"
    assert gates.loc["has_delisted_names_or_policy", "status"] == "fail"
    assert quality.loc["Malaysia", "delisted_count"] == 0
    assert quality.loc["Singapore", "dead_name_coverage_rate"] == 0.0
