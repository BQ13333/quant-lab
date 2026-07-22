from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g1_quote_chain_discovery import (
    quote_chain_filter,
    write_japan_g1_quote_chain_plan,
)


def test_quote_chain_plan_is_identity_anchored_and_fail_closed(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    pd.DataFrame({"local_code": ["1805", "1805", "2336"]}).to_csv(missing, index=False)
    master = tmp_path / "security_master.csv"
    pd.DataFrame(
        [
            {"ric": "1805.IXJ^I12", "ticker_symbol": "1805", "instrument_permid": "8590928464", "isin": "JP3629810007", "exchange_code": "IXJ"},
            {"ric": "1805.KAB^K11", "ticker_symbol": "1805", "instrument_permid": "8590928464", "isin": "JP3629810007", "exchange_code": "KAB"},
            {"ric": "9999.T", "ticker_symbol": "9999", "instrument_permid": "111", "isin": "JPX", "exchange_code": "TYO"},
        ]
    ).to_csv(master, index=False)
    result = write_japan_g1_quote_chain_plan(
        run_id="PLAN-V1",
        missing_plan_path=missing,
        security_master_paths=[master],
        artifact_root=tmp_path / "artifacts",
    )
    plan = pd.read_csv(result.identity_plan_path, dtype=str)
    unanchored = pd.read_csv(result.unanchored_codes_path, dtype=str)
    assert plan[["local_code", "instrument_permid"]].to_dict("records") == [
        {"local_code": "1805", "instrument_permid": "8590928464"}
    ]
    assert plan.loc[0, "evidence_rics"] == "1805.IXJ^I12|1805.KAB^K11"
    assert unanchored["local_code"].tolist() == ["2336"]
    assert result.manifest["counts"] == {"unresolved_codes": 2, "anchored_codes": 1, "identity_rows": 1, "unanchored_codes": 1}
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["output_sha256"]["identity_plan.csv"] == hashlib.sha256(result.identity_plan_path.read_bytes()).hexdigest()
    assert json.loads(result.manifest_path.read_text())["promotion_policy"]["live_trading"] is False


def test_quote_chain_filter_is_deterministic_and_rejects_invalid_values() -> None:
    assert quote_chain_filter(["22", "11", "22"]) == "InstrumentPermID eq '11' or InstrumentPermID eq '22'"
    with pytest.raises(ValueError):
        quote_chain_filter([])
    with pytest.raises(ValueError):
        quote_chain_filter(["not-a-permid"])
