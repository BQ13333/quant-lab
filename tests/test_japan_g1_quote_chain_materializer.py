from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g1_quote_chain_materializer import (
    write_japan_g1_quote_chain_materializer,
)


def test_materializer_selects_direct_and_successor_candidates_without_admission(tmp_path: Path) -> None:
    plan_dir = _plan(tmp_path)
    shard = _shard(tmp_path, plan_dir)
    result = write_japan_g1_quote_chain_materializer(
        run_id="MATERIALIZER-V1",
        identity_plan_dir=plan_dir,
        shard_dirs=[shard],
        artifact_root=tmp_path / "materializer",
    )
    candidates = pd.read_csv(result.candidates_path, dtype=str).fillna("")
    assert candidates.set_index("local_code")["mapping_route"].to_dict() == {
        "1773": "direct_tyo_candidate",
        "1805": "identity_successor_candidate",
    }
    selected = pd.read_csv(result.selected_security_master_path, dtype=str).fillna("")
    assert selected.set_index("mapping_local_code_override")["ric"].to_dict() == {
        "1773": "1773.T",
        "1805": "256A.T",
    }
    assert selected["quote_chain_admitted"].eq("False").all()
    assert result.manifest["scorecard_eligible"] is False
    assert result.manifest["critical_blockers"] == ["independent_official_event_validation_required"]


def test_materializer_rejects_tampered_shard(tmp_path: Path) -> None:
    plan_dir = _plan(tmp_path)
    shard = _shard(tmp_path, plan_dir)
    with (shard / "security_master.csv").open("a") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="shard contract invalid"):
        write_japan_g1_quote_chain_materializer(
            run_id="MATERIALIZER-V1",
            identity_plan_dir=plan_dir,
            shard_dirs=[shard],
            artifact_root=tmp_path / "materializer",
        )


def _plan(tmp_path: Path) -> Path:
    root = tmp_path / "plan"
    root.mkdir()
    plan = pd.DataFrame(
        [
            {"identity_row_id": "A", "local_code": "1773", "instrument_permid": "11", "isins": "MY1", "evidence_rics": "1773.X", "evidence_exchange_codes": "IXJ", "source_paths": "a"},
            {"identity_row_id": "B", "local_code": "1805", "instrument_permid": "22", "isins": "JP2", "evidence_rics": "1805.X", "evidence_exchange_codes": "IXJ", "source_paths": "a"},
        ]
    )
    path = root / "identity_plan.csv"
    plan.to_csv(path, index=False)
    manifest = {"artifact_type": "japan_g1_quote_chain_identity_plan", "output_sha256": {"identity_plan.csv": _hash(path)}}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def _shard(tmp_path: Path, plan_dir: Path) -> Path:
    root = tmp_path / "shard"
    root.mkdir()
    output = root / "security_master.csv"
    pd.DataFrame(
        [
            {"ric": "1773.T", "ticker_symbol": "1773", "instrument_permid": "11", "isin": "MY1", "exchange_code": "TYO", "is_primary_ric": False, "is_primary_issue_ric": False},
            {"ric": "256A.T", "ticker_symbol": "256A", "instrument_permid": "22", "isin": "JP2", "exchange_code": "TYO", "is_primary_ric": True, "is_primary_issue_ric": True},
        ]
    ).to_csv(output, index=False)
    plan_hash = _hash(plan_dir / "identity_plan.csv")
    payload = {"run_id": "SHARD-01", "identity_row_ids": ["A", "B"], "local_codes": ["1773", "1805"], "instrument_permids": ["11", "22"], "extra_filter": "InstrumentPermID eq '11' or InstrumentPermID eq '22'", "identity_plan_sha256": plan_hash, "country_codes": ["JPN"], "state_buckets": ["active", "inactive"], "top": 1000, "max_pages": 1}
    payload["request_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    payload["artifact_dir"] = str(root.resolve())
    manifest = {"output_sha256": {"security_master": _hash(output)}}
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    request = {**payload, "status": "complete", "manifest_sha256": _hash(manifest_path)}
    (root / "quote_chain_request.json").write_text(json.dumps(request))
    return root


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
