"""Test-only signing keys and patched admin anchor; no production trust changes."""
from types import SimpleNamespace
from test_control_plane import signed_record
from quant_lab.lifecycle.approval import Authorization, digest


def authorize(state,evidence,tmp_path,monkeypatch,approval_id='APR-001'):
    manifest=SimpleNamespace(run_id='LIFECYCLE-TEST',identity_hash=lambda:digest(state))
    bundle=SimpleNamespace(candidate_id='SLEEVE-TEST',content_hash=lambda:digest(evidence))
    record,store=signed_record(manifest,bundle,trust_path=tmp_path/(approval_id+'.json'),monkeypatch=monkeypatch,approval_id=approval_id)
    return Authorization(record,store)
