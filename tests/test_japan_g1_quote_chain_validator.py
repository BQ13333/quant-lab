from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g1_quote_chain_materializer import write_japan_g1_quote_chain_materializer
from quant_lab.data.japan_g1_quote_chain_validator import write_japan_g1_quote_chain_validation
from quant_lab.data.japan_g1_monthly_materializer import _validated_quote_chain_blockers


_HELPER_PATH = Path(__file__).with_name("test_japan_g1_quote_chain_materializer.py")
_SPEC = importlib.util.spec_from_file_location("quote_chain_materializer_test_helpers", _HELPER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)
_plan = _HELPERS._plan
_shard = _HELPERS._shard


def test_validator_admits_only_same_code_direct_route(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    shard = _shard(tmp_path, plan)
    materializer = write_japan_g1_quote_chain_materializer(run_id="MATERIALIZER-V1", identity_plan_dir=plan, shard_dirs=[shard], artifact_root=tmp_path / "materializer")
    result = write_japan_g1_quote_chain_validation(run_id="VALIDATOR-V1", materializer_dir=materializer.artifact_dir, identity_plan_dir=plan, shard_dirs=[shard], artifact_root=tmp_path / "validator")
    admitted = pd.read_csv(result.security_master_path, dtype=str).fillna("")
    transitions = pd.read_csv(result.transition_plan_path, dtype=str).fillna("")
    assert admitted["ric"].tolist() == ["1773.T"]
    assert admitted["quote_chain_admitted"].tolist() == ["True"]
    assert transitions[["local_code", "candidate_successor_ric"]].to_dict("records") == [{"local_code": "1805", "candidate_successor_ric": "256A.T"}]
    assert result.manifest["status"] == "pass_partial_component"
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["output_sha256"]["security_master"] == hashlib.sha256(result.security_master_path.read_bytes()).hexdigest()
    assert _validated_quote_chain_blockers(
        result.artifact_dir, result.manifest, result.security_master_path
    ) == []
