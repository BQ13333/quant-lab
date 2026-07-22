from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "revalidate_jpx_exceptional_price_limit_notices.py"
SPEC = importlib.util.spec_from_file_location("revalidate_jpx_exceptional_price_limits", SCRIPT)
assert SPEC and SPEC.loader
revalidation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(revalidation)


def test_writer_publishes_separate_atomic_write_once_artifact(tmp_path: Path) -> None:
    source = tmp_path / "retained" / "V1"
    source.mkdir(parents=True)
    output_root = tmp_path / "revalidation"

    destination = revalidation.write_revalidation_artifact(
        {"counts": {"strict_validation_quarantine_after": 13}},
        source_artifact_dir=source,
        output_root=output_root,
        run_id="REVALIDATION-V1",
    )

    report = json.loads((destination / "strict_revalidation_report.json").read_text())
    manifest = json.loads((destination / "manifest.json").read_text())
    assert report["counts"]["strict_validation_quarantine_after"] == 13
    assert manifest["artifact_type"] == "jpx_exceptional_price_limit_revalidation"
    assert manifest["status"] == "complete"
    assert manifest["scorecard_eligible"] is False
    assert not list(output_root.glob(".REVALIDATION-V1.*"))

    with pytest.raises(FileExistsError):
        revalidation.write_revalidation_artifact(
            report,
            source_artifact_dir=source,
            output_root=output_root,
            run_id="REVALIDATION-V1",
        )


def test_writer_rejects_output_inside_retained_corpus(tmp_path: Path) -> None:
    source = tmp_path / "retained" / "V1"
    source.mkdir(parents=True)
    with pytest.raises(ValueError, match="separate"):
        revalidation.write_revalidation_artifact(
            {},
            source_artifact_dir=source,
            output_root=source / "revalidation",
            run_id="BAD-V1",
        )
