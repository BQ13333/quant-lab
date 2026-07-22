from __future__ import annotations

import json

import pytest

from quant_lab.data.immutable_hash_inventory import (
    verify_immutable_hash_inventory,
    write_immutable_hash_inventory,
)


def test_inventory_is_complete_verified_and_write_once(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (source / "b.json").write_text('{"ok": true}\n', encoding="utf-8")
    output = tmp_path / "inventory.json"

    first = write_immutable_hash_inventory(
        input_paths=[source], output_path=output, inventory_id="TEST"
    )
    assert first.file_count == 2
    assert verify_immutable_hash_inventory(output) == (True, [])
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["immutable"] is True
    assert payload["coverage_complete"] is True

    second = write_immutable_hash_inventory(
        input_paths=[source], output_path=output, inventory_id="TEST"
    )
    assert second.files == first.files

    (source / "a.csv").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable hash inventory drift"):
        write_immutable_hash_inventory(
            input_paths=[source], output_path=output, inventory_id="TEST"
        )
    passed, issues = verify_immutable_hash_inventory(output)
    assert passed is False
    assert any(issue.startswith("digest_mismatch:") for issue in issues)


def test_inventory_rejects_missing_or_empty_inputs(tmp_path):
    with pytest.raises(FileNotFoundError):
        write_immutable_hash_inventory(
            input_paths=[tmp_path / "missing"],
            output_path=tmp_path / "inventory.json",
            inventory_id="MISSING",
        )
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no input files"):
        write_immutable_hash_inventory(
            input_paths=[empty],
            output_path=tmp_path / "inventory.json",
            inventory_id="EMPTY",
        )
