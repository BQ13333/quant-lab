from __future__ import annotations

import json
from pathlib import Path

from quant_lab.agents.japan_data_quality_loop import default_japan_data_quality_paths
from quant_lab.data.japan_g5_preflight import write_japan_g5_preflight_artifacts


def test_preflight_hashes_available_inputs_and_fails_closed(tmp_path: Path) -> None:
    paths = default_japan_data_quality_paths(tmp_path)
    paths.security_master_dir.mkdir(parents=True)
    (paths.security_master_dir / "manifest.json").write_text(
        json.dumps({"status": "failed_validation", "scorecard_eligible": False}),
        encoding="utf-8",
    )
    assert paths.commands_log is not None
    paths.commands_log.write_text("test command\n", encoding="utf-8")
    qa = tmp_path / "qa" / "manifest.json"
    qa.parent.mkdir(parents=True)
    qa.write_text(
        json.dumps(
            {
                "status": "completed_partial",
                "evidence_scope": "qa",
                "scorecard_eligible": False,
                "history_rows": 10,
                "known_at_coverage": 1.0,
                "pit_timestamp_violations": 0,
            }
        ),
        encoding="utf-8",
    )

    result = write_japan_g5_preflight_artifacts(paths, "G5", [qa])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    inventory = json.loads(result.immutable_hashes_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "blocked_upstream"
    assert manifest["scorecard_eligible"] is False
    assert manifest["predecision_timestamp_pass"] is True
    assert manifest["decision_rows"] == 0
    assert inventory["status"] == "incomplete"
    assert inventory["coverage_complete"] is False
    assert inventory["files"][str(paths.commands_log.resolve())]
    assert result.missing_inputs_path.exists()


def test_preflight_treats_missing_and_nan_row_counts_as_zero(tmp_path: Path) -> None:
    paths = default_japan_data_quality_paths(tmp_path)
    assert paths.commands_log is not None
    paths.commands_log.write_text("test command\n", encoding="utf-8")
    qa = tmp_path / "qa" / "manifest.json"
    qa.parent.mkdir(parents=True)
    qa.write_text(
        json.dumps(
            {
                "status": "completed_partial",
                "history_rows": "nan",
                "known_at_coverage": 1.0,
                "pit_timestamp_violations": 0,
            }
        ),
        encoding="utf-8",
    )

    result = write_japan_g5_preflight_artifacts(paths, "G5-NAN", [qa])

    assert result.timestamp_audit.iloc[0]["history_rows"] == 0
    assert not bool(result.timestamp_audit.iloc[0]["predecision_timestamp_pass"])
