from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_price_limit_completion_plan import (
    CompletionPlanInputError,
    PROMOTION_POLICY,
    REQUIRED_COMPLETION_BLOCKERS,
    write_japan_price_limit_completion_plan,
)


def test_writer_enumerates_exact_fail_closed_bounded_plan(tmp_path: Path) -> None:
    producer, reconciliation, validation = _write_upstream_bundle(tmp_path)
    result = write_japan_price_limit_completion_plan(
        run_id="PLAN-TEST",
        producer_dir=producer,
        reconciliation_dir=reconciliation,
        validation_dir=validation,
        artifact_root=tmp_path / "plans",
        return_batch_size=1,
        max_return_sessions=8,
        max_return_lookahead_days=12,
    )

    source_queue = pd.read_csv(result.artifact_dir / "source_queue.csv", dtype={"code": str})
    requirements = pd.read_csv(
        result.artifact_dir / "event_evidence_requirements.csv", dtype={"code": str}
    )
    extraction = pd.read_csv(result.artifact_dir / "extraction_queue.csv")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    archive = source_queue.loc[source_queue["lane"].eq("official_archive")]
    returns = source_queue.loc[source_queue["lane"].eq("return_normal_evidence")]
    assert len(archive) == 86
    assert archive.iloc[0]["interval_start"] == "2006-07-01"
    assert archive.iloc[-1]["interval_end"] == "2013-08-31"
    assert len(requirements) == 2
    assert len(returns) == 2
    assert requirements["code"].tolist() == ["1111", "1111"]
    assert requirements["continuation_count"].tolist() == [1, 0]
    assert requirements.iloc[0]["source_event_keys"] == (
        "2021-01-04:BROADENING:1111|2021-01-05:CONTINUATION:1111"
    )
    assert requirements.iloc[0]["bounded_evidence_end_date"] == "2021-01-17"
    assert requirements["inference_status"].eq(
        "not_attempted_pending_admitted_evidence"
    ).all()
    assert not source_queue["execution_enabled"].any()
    assert not extraction["execution_enabled"].any()
    assert extraction["requirement_count"].max() <= 1
    assert not extraction["promotion_allowed"].any()
    assert manifest["status"] == "planning_complete_blockers_unresolved"
    assert manifest["remaining_completion_blockers"] == list(REQUIRED_COMPLETION_BLOCKERS)
    assert manifest["blockers_solved"] == []
    assert manifest["completion_claimed"] is False
    assert manifest["inference_performed"] is False
    assert manifest["promotion_allowed"] is False
    assert manifest["promotion_policy"] == PROMOTION_POLICY
    assert manifest["counts"]["archive_month_rows"] == 86
    assert manifest["counts"]["return_normal_event_cases"] == 2
    assert len(manifest["input_sha256"]) == 12
    for filename, digest in manifest["output_sha256"].items():
        assert digest == _digest(result.artifact_dir / filename)


def test_writer_rejects_tampered_input_before_creating_output(tmp_path: Path) -> None:
    producer, reconciliation, validation = _write_upstream_bundle(tmp_path)
    with (producer / "price_limit_events.csv").open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")

    with pytest.raises(CompletionPlanInputError, match="producer output hash mismatch"):
        write_japan_price_limit_completion_plan(
            run_id="TAMPERED",
            producer_dir=producer,
            reconciliation_dir=reconciliation,
            validation_dir=validation,
            artifact_root=tmp_path / "plans",
        )
    assert not (tmp_path / "plans" / "TAMPERED").exists()


def test_writer_never_overwrites_an_immutable_run(tmp_path: Path) -> None:
    producer, reconciliation, validation = _write_upstream_bundle(tmp_path)
    kwargs = {
        "run_id": "IMMUTABLE",
        "producer_dir": producer,
        "reconciliation_dir": reconciliation,
        "validation_dir": validation,
        "artifact_root": tmp_path / "plans",
    }
    result = write_japan_price_limit_completion_plan(**kwargs)
    original = result.manifest_path.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        write_japan_price_limit_completion_plan(**kwargs)
    assert result.manifest_path.read_bytes() == original


def test_writer_rejects_any_attempt_to_drop_a_retained_blocker(tmp_path: Path) -> None:
    producer, reconciliation, validation = _write_upstream_bundle(tmp_path)
    manifest_path = validation / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["remaining_completion_blockers"] = [REQUIRED_COMPLETION_BLOCKERS[0]]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(CompletionPlanInputError, match="exactly the two blockers"):
        write_japan_price_limit_completion_plan(
            run_id="DROPPED-BLOCKER",
            producer_dir=producer,
            reconciliation_dir=reconciliation,
            validation_dir=validation,
            artifact_root=tmp_path / "plans",
        )


def _write_upstream_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    producer = tmp_path / "producer"
    reconciliation = tmp_path / "reconciliation"
    validation = tmp_path / "validation"
    producer.mkdir()
    reconciliation.mkdir()
    validation.mkdir()

    events = pd.DataFrame(
        [
            _event("2021-01-04", "2021-01-05", "BROADENING", "1111", 1000, 1200, 300),
            _event("2021-01-05", "2021-01-06", "CONTINUATION", "1111", 2200, 2000, 500),
            _event("2021-03-01", "2021-03-02", "BROADENING", "1111", 700, 600, 150),
        ]
    )
    events.to_csv(producer / "price_limit_events.csv", index=False)
    pd.DataFrame({"plan_row_id": ["JPL-00001"]}).to_csv(
        producer / "notice_plan.csv", index=False
    )
    pd.DataFrame({"plan_row_id": ["JPL-00001"], "terminal_status": ["success"]}).to_csv(
        producer / "page_state.csv", index=False
    )
    pd.DataFrame(
        {
            "publication_date": events["publication_date"],
            "event_type": events["event_type"],
            "code": events["code"],
            "expected_multiplier": 4.0,
            "observed_multiplier": 4.0,
            "regime_multiplier_valid": True,
        }
    ).to_csv(producer / "event_quality.csv", index=False)
    producer_hashes = _hashes(
        producer,
        ("notice_plan.csv", "page_state.csv", "price_limit_events.csv", "event_quality.csv"),
    )
    producer_manifest = {
        "run_id": "PRODUCER",
        "artifact_type": "japan_exceptional_price_limit_event_history_candidate",
        "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "critical_blockers": ["corrected_by_overlay", *REQUIRED_COMPLETION_BLOCKERS],
        "promotion_policy": PROMOTION_POLICY,
        "output_sha256": producer_hashes,
    }
    (producer / "manifest.json").write_text(json.dumps(producer_manifest), encoding="utf-8")

    for filename in (
        "counterpart_fetch_state.csv",
        "reconciliation.csv",
        "corrected_candidate_events.csv",
    ):
        pd.DataFrame({"row": []}).to_csv(reconciliation / filename, index=False)
    reconciliation_hashes = _hashes(
        reconciliation,
        ("counterpart_fetch_state.csv", "reconciliation.csv", "corrected_candidate_events.csv"),
    )
    reconciliation_manifest = {
        "run_id": "RECONCILIATION",
        "artifact_type": "japan_exceptional_price_limit_bilingual_reconciliation_candidate",
        "status": "partial_candidate",
        "reconciliation_blockers": [],
        "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "critical_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "promotion_allowed": False,
        "promotion_policy": PROMOTION_POLICY,
        "source_manifest_sha256": _digest(producer / "manifest.json"),
        "source_output_sha256": producer_hashes,
        "output_sha256": reconciliation_hashes,
    }
    (reconciliation / "manifest.json").write_text(
        json.dumps(reconciliation_manifest), encoding="utf-8"
    )

    detail = pd.DataFrame(
        [
            {
                "publication_date": row.publication_date,
                "event_type": row.event_type,
                "code": row.code,
                "source_url": row.source_url,
                "resolved_side": "UPPER",
                "validated_base_price_yen": row.base_price_yen,
                "validated_expanded_range_yen": row.expanded_range_yen,
                "validated_normal_range_yen": row.normal_range_yen,
                "validated_observed_multiplier": 4.0,
                "post_overlay_numeric_discrepancy": False,
                "post_overlay_regime_discrepancy": False,
                "expected_multiplier": 4.0,
            }
            for row in events.itertuples(index=False)
        ]
    )
    detail.to_csv(validation / "event_validation_detail.csv", index=False)
    pd.DataFrame({"check": ["fixture"], "passed": [True]}).to_csv(
        validation / "validation_summary.csv", index=False
    )
    validation_hashes = _hashes(
        validation, ("validation_summary.csv", "event_validation_detail.csv")
    )
    validation_manifest = {
        "run_id": "VALIDATION",
        "artifact_type": "japan_exceptional_price_limit_bilingual_reconciliation_independent_validation",
        "status": "pass_partial_component",
        "independent_validation": True,
        "validation_issues": [],
        "remaining_completion_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "critical_blockers": list(REQUIRED_COMPLETION_BLOCKERS),
        "promotion_allowed": False,
        "promotion_policy": PROMOTION_POLICY,
        "source_manifest_sha256": _digest(producer / "manifest.json"),
        "reconciliation_manifest_sha256": _digest(reconciliation / "manifest.json"),
        "output_sha256": validation_hashes,
    }
    (validation / "manifest.json").write_text(
        json.dumps(validation_manifest), encoding="utf-8"
    )
    return producer, reconciliation, validation


def _event(
    published: str,
    effective: str,
    event_type: str,
    code: str,
    base: int,
    expanded: int,
    normal: int,
) -> dict[str, object]:
    return {
        "publication_date": published,
        "archive_publication_date": published,
        "effective_date": effective,
        "event_type": event_type,
        "code": code,
        "issue_name": "Alpha",
        "side": "UPPER",
        "base_price_yen": base,
        "expanded_range_yen": expanded,
        "normal_range_yen": normal,
        "observed_multiplier": 4.0,
        "source_url": f"https://www.jpx.co.jp/english/news/1030/{published.replace('-', '')}-01.html",
        "title": event_type,
    }


def _hashes(root: Path, filenames: tuple[str, ...]) -> dict[str, str]:
    return {filename: _digest(root / filename) for filename in filenames}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
