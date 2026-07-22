from __future__ import annotations

import hashlib
import json

import pytest

from quant_lab.research.qforge20_workflow import (
    build_qforge20_malaysia_index,
    build_qforge20_v4_review_report,
    persist_qforge20_index,
    persist_qforge20_review_report,
)
from quant_lab.research.contracts import ModelCallRecord, SpecialistReview
from quant_lab.research.trial_ledger import TrialLedger


def test_closed_cohort_is_registered_without_relabelling_prior_ollama_reviews(tmp_path):
    manifest = {
        "cohort_version": "MALAYSIA-ATOMIC-COHORT-V1",
        "strategies": [
            {
                "strategy_id": "MYS-S-ONE",
                "sequence": 1,
                "family": "momentum",
                "hypothesis_id": "MYS-H-ONE",
                "economic_claim": "Lagged momentum predicts returns.",
                "parameters": {"lookback": 126},
                "source_file": "docs/source.md",
                "source_section": "Momentum",
            }
        ],
    }
    report = {
        "cohort_version": "MALAYSIA-ATOMIC-COHORT-V1",
        "cohort_closed": True,
        "atomic_metrics": {
            "MYS-S-ONE": {
                "net_sharpe": 1.0,
                "max_drawdown": -0.2,
                "mean_daily_return": 0.0001,
                "hac_one_sided_pvalue": 0.01,
                "holm_adjusted_pvalue": 0.01,
            }
        },
        "returns_matrix_hash": "a" * 64,
        "pbo": 0.1,
        "correlation_matrix": {"MYS-S-ONE": {"MYS-S-ONE": 1.0}},
        "predeclared_combination_metrics": {},
    }
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report_path.write_text(json.dumps(report), encoding="utf-8")
    index = build_qforge20_malaysia_index(
        cohort_manifest_path=manifest_path,
        cohort_report_path=report_path,
        lifetime_counts={"lifetime_attempts": 3347},
        v4_review_status="pending_rotated_key",
    )
    assert index["workflow_name"] == "QForge-20"
    assert index["strategy_count"] == 1
    assert index["v4_reviewed_strategy_count"] == 0
    assert index["provenance"]["prior_results_are_imported_not_relabelled"] is True
    assert index["strategies"][0]["qforge20_v4_review_status"] == "pending_rotated_key"
    assert index["provenance"]["cohort_report_sha256"] == hashlib.sha256(
        report_path.read_bytes()
    ).hexdigest()
    target = persist_qforge20_index(index, tmp_path / "qforge" / "index.json")
    assert json.loads(target.read_text(encoding="utf-8")) == index


def test_open_or_mismatched_cohort_cannot_enter_qforge_index(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    report_path = tmp_path / "report.json"
    manifest_path.write_text(
        json.dumps({"cohort_version": "V1", "strategies": []}), encoding="utf-8"
    )
    report_path.write_text(
        json.dumps({"cohort_version": "V2", "cohort_closed": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="closed, matching"):
        build_qforge20_malaysia_index(
            cohort_manifest_path=manifest_path,
            cohort_report_path=report_path,
            lifetime_counts={},
            v4_review_status="pending_rotated_key",
        )


def test_completed_review_report_verifies_evidence_and_marks_unpriced_failures(tmp_path):
    index_path = tmp_path / "index.json"
    index = {
        "workflow_name": "QForge-20",
        "workflow_version": "QFORGE20-MALAYSIA-RESEARCH-V1",
        "v4_review_status": "review_complete",
        "strategy_count": 1,
        "terminal_evidence_available": False,
        "promotion_status": "research_only_blocked_from_dry_run",
        "strategies": [
            {
                "sequence": 1,
                "strategy_id": "MYS-S-ONE",
                "family": "momentum",
                "research_period_result": {"net_sharpe": 1.0},
            }
        ],
    }
    persist_qforge20_index(index, index_path)
    ledger = TrialLedger(tmp_path / "ledger.sqlite", tmp_path / "evidence")
    reservation = ledger.reserve_trial(
        idempotency_key="qforge-test",
        campaign="qforge20_malaysia_review",
        family="momentum",
        hypothesis_id="MYS-H-ONE",
        candidate_id="MYS-C-MYS-S-ONE",
        code_hash="a" * 64,
        data_hash="b" * 64,
        config_hash="c" * 64,
        prompt_hash="d" * 64,
        model_hash="e" * 64,
    )
    ledger.record_specialist_failure(
        reservation.trial_id,
        task_id="TASK-FAILED",
        specialist_id="selection_bias_validation_specialist",
        failure_code="MALFORMED_MODEL_OUTPUT",
        request_fingerprint="f" * 64,
    )
    for number, role in enumerate(
        (
            "selection_bias_validation_specialist",
            "measure_probability_martingale_specialist",
            "microstructure_hawkes_execution_specialist",
        ),
        start=1,
    ):
        task_id = f"TASK-{number}"
        review = SpecialistReview(
            review_id=f"REVIEW-{number}",
            task_id=task_id,
            specialist_id=role,
            verdict="conditional",
            claims=("Research-period effect is not terminal evidence.",),
            assumptions=("Point-in-time inputs are correct.",),
            counterexamples=(),
            falsification_tests=("Run the frozen rule on genuinely future data.",),
            veto=False,
            confidence=0.5,
            provenance_hashes={"candidate": "1" * 64},
        )
        record = ModelCallRecord(
            call_id=f"CALL-{number}",
            provider="deepseek",
            model="deepseek-v4-pro",
            endpoint="https://api.deepseek.com",
            started_at="start",
            completed_at="end",
            status="success",
            prompt_hash="2" * 64,
            schema_hash="3" * 64,
            config_hash="4" * 64,
            response_hash="5" * 64,
            input_tokens=100,
            output_tokens=200,
            cost_usd=0.001,
            stop_reason="stop",
            redactions=("api_key", "prompt_body", "response_body"),
        )
        ledger.record_specialist_review(reservation.trial_id, review, record)
    ledger.record_panel_disposition(
        reservation.trial_id,
        {
            "backtest_allowed": True,
            "progression_allowed": False,
            "vetoed": False,
        },
    )
    ledger.record_outcome(
        reservation.trial_id,
        status="success",
        payload={
            "review_completed": True,
            "progression_allowed": False,
            "vetoed": False,
        },
    )
    report = build_qforge20_v4_review_report(
        index_path=index_path,
        ledger_path=ledger.db_path,
        evidence_root=ledger.evidence_root,
        known_unledgered_call_count=1,
        known_unledgered_cost_usd=0.002,
    )
    assert report["completion"]["successful_structured_review_count"] == 3
    assert report["completion"]["verdict_counts"] == {"conditional": 3}
    assert report["cost_accounting"]["tracked_successful_call_cost_usd"] == 0.003
    assert report["cost_accounting"]["tracked_failed_call_cost_usd"] == 0.0
    assert report["cost_accounting"]["tracked_total_call_cost_usd"] == 0.003
    assert report["cost_accounting"]["known_unledgered_call_count"] == 1
    assert report["cost_accounting"]["known_unledgered_cost_usd"] == 0.002
    assert report["cost_accounting"]["known_cost_lower_bound_usd"] == 0.005
    assert report["cost_accounting"]["unreceipted_failure_count"] == 1
    assert report["cost_accounting"]["unpriced_malformed_call_count"] == 1
    assert report["cost_accounting"]["cost_completeness"] == "lower_bound_only"
    target = persist_qforge20_review_report(report, tmp_path / "report.json")
    assert json.loads(target.read_text(encoding="utf-8"))["conclusion"] == "no_candidate_promoted"


def test_review_report_rejects_negative_unledgered_cost_reconciliation(tmp_path):
    with pytest.raises(ValueError, match="cannot be negative"):
        build_qforge20_v4_review_report(
            index_path=tmp_path / "missing-index.json",
            ledger_path=tmp_path / "missing-ledger.sqlite",
            evidence_root=tmp_path / "missing-evidence",
            known_unledgered_call_count=-1,
            known_unledgered_cost_usd=0.0,
        )
