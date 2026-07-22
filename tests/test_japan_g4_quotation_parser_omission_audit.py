from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_quotation_parser_omission_audit import (
    _exact_code_count,
    produce_japan_g4_quotation_parser_omission_audit,
)
from quant_lab.data.japan_g4_quotation_parser_omission_audit_validator import (
    validate_japan_g4_quotation_parser_omission_audit,
)


ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "artifacts/japan_g4_halt_mapping_recovery/JPN-G4-HALT-MAPPING-RECOVERY-20260712-V3"
V16 = ROOT / "artifacts/japan_g4_halt_history/JPN-G4-HALTS-200607-202606-V16"

pytestmark = pytest.mark.skipif(
    not (V3.is_dir() and V16.is_dir()),
    reason="private JPX artifact integration tests",
)


def _produce(tmp_path: Path):
    return produce_japan_g4_quotation_parser_omission_audit(
        run_id="QUOTATION-OMISSION-AUDIT-TEST-V1",
        mapping_v3_dir=V3,
        halt_v16_dir=V16,
        artifact_root=tmp_path / "producer",
    )


def test_real_artifacts_fail_closed_without_eleven_row_denominator(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    counts = result.manifest["counts"]
    assert result.manifest["status"] == "blocked_unsealed_eleven_row_denominator"
    assert counts["already_admitted_raw_pdf_mappings"] == 23
    assert counts["baseline_normalized_absent_in_admitted_population"] == 22
    assert counts["source_record_missing_rows"] == 8
    assert counts["verified_parser_omissions_in_missing_rows"] == 0
    assert counts["code_6174_rows"] == 1
    assert counts["accepted_parser_repairs"] == 0


def test_independent_validator_confirms_blocker(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    _, validation = validate_japan_g4_quotation_parser_omission_audit(
        producer_dir=result.artifact_dir,
        run_id="QUOTATION-OMISSION-AUDIT-VALIDATION-TEST-V1",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "pass_blocker_confirmed"
    assert validation["counts"]["validation_issues"] == 0
    assert validation["promotion_eligible"] is False


def test_validator_rejects_output_tampering(tmp_path: Path) -> None:
    result = _produce(tmp_path)
    path = result.artifact_dir / "population_b_source_record_missing.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "verified_parser_omission"] = True
    frame.to_csv(path, index=False)
    _, validation = validate_japan_g4_quotation_parser_omission_audit(
        producer_dir=result.artifact_dir,
        run_id="QUOTATION-OMISSION-AUDIT-TAMPER-TEST-V1",
        artifact_root=tmp_path / "validation",
    )
    assert validation["status"] == "fail"
    assert validation["counts"]["validation_issues"] > 0


def test_exact_code_count_does_not_match_substrings() -> None:
    assert _exact_code_count(" 6174 ordinary  X6174  6174A ", "6174") == 1
