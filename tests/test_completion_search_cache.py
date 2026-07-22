from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path

from quant_lab.research.knowledge_completion import CompletionInputs


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "complete_knowledge_hygiene.py"
SPEC = importlib.util.spec_from_file_location("complete_knowledge_hygiene_script", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


def test_search_cache_binds_package_stats_and_trust_policy_identity(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    output = tmp_path / "knowledge"
    package = output / "completion" / "steps-9-11-v1-fixture"
    package.mkdir(parents=True)
    (package / "completion-manifest.json").write_text(
        json.dumps({"completion_id": package.name}), encoding="utf-8"
    )
    database = package / "artifact-index.sqlite3"
    database.write_bytes(b"initial-package-bytes")
    configured = CompletionInputs(
        workspace_root=workspace,
        output_root=output,
        overlay_pointer=tmp_path / "overlay.json",
        source_findings_database=tmp_path / "findings.sqlite3",
    )

    CACHE._write_search_cache(configured, package)
    assert CACHE._search_cache_is_current(configured, package)

    database.write_bytes(b"changed-package-bytes")
    assert not CACHE._search_cache_is_current(configured, package)

    CACHE._write_search_cache(configured, package)
    changed_trust = replace(
        configured,
        data_admission_trust_stores={"a" * 64: object()},
    )
    assert not CACHE._search_cache_is_current(changed_trust, package)
