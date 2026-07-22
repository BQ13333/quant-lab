from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g1_reconciliation_qa import (
    ACTION_CATEGORIES,
    RECONCILIATION_FILE,
    STABLE_INSTRUMENTS_FILE,
    UNMATCHED_FILE,
    classify_japan_g1_denominator_differences,
    write_japan_g1_reconciliation_qa_artifacts,
)


NOW = datetime(2026, 7, 11, 16, 0, tzinfo=timezone.utc)


def _unmatched_rows() -> pd.DataFrame:
    rows = [
        ("public_jpx", "12345", "", "", "public_in_scope_unmatched", "Preferred"),
        ("public_jpx", "2000", "", "", "public_in_scope_unmatched", "Primary gap"),
        ("public_jpx", "3000", "", "", "public_in_scope_unmatched", "Retired gap"),
        ("security_master", "4000", "id:dup-a", "4000.T", "master_active_unmatched", ""),
        ("security_master", "40A", "id:alpha", "40A.T", "master_active_unmatched", ""),
        ("security_master", "5000", "id:reit", "5000.T", "master_active_unmatched", ""),
        ("security_master", "6000", "id:stale", "6000.T", "master_active_unmatched", ""),
        ("security_master", "7000", "id:open", "7000.T", "master_active_unmatched", ""),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "source_side",
            "local_code",
            "stable_instrument_id",
            "ric",
            "reconciliation_status",
            "jpx_name",
        ],
    ).assign(
        jpx_source_row=lambda frame: range(1, len(frame) + 1),
        jpx_asof_date="2026-07-11",
        jpx_known_at_utc="",
        jpx_market_category_raw="",
        jpx_market_scope="",
        in_primary_ordinary_share_scope="True",
        stable_identifier_type="isin",
        price_currency="JPY",
        unmatched_classification="source_gap",
    )


def _stable_rows() -> pd.DataFrame:
    rows = [
        ("1234", "id:base", "1234.T", True, True, "", "Issuer ordinary", 1),
        ("2000", "id:no-primary", "2000.T", True, False, "", "Primary gap", 1),
        ("3000", "id:retired", "3000.T^H25", False, False, "2025-01-01", "Retired", 1),
        ("4000", "id:dup-a", "4000.T", True, True, "", "Duplicate A", 1),
        ("4000", "id:dup-b", "4000.T^H20", False, False, "2020-01-01", "Duplicate B", 1),
        ("40A", "id:alpha", "40A.T", True, True, "", "Alpha company", 1),
        ("5000", "id:reit", "5000.T", True, True, "", "Example REIT ORD", 1),
        ("6000", "id:stale", "6000.T", True, True, "2026-01-01", "Stale active", 1),
        ("7000", "id:open", "7000.T", True, True, "", "Unresolved company", 1),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "local_code",
            "stable_instrument_id",
            "ric",
            "is_active",
            "is_primary_ric",
            "retire_date",
            "common_name",
            "jpx_quote_row_count",
        ],
    ).assign(
        is_primary_issue_ric=True,
        first_trade_date="2020-01-01",
        jpx_last_retire_date="",
        quote_row_count=1,
        isin=lambda frame: "JP" + frame.index.astype(str).str.zfill(10),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_packet(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    unmatched = _unmatched_rows()
    unmatched.to_csv(source / UNMATCHED_FILE, index=False)
    unmatched.to_csv(source / RECONCILIATION_FILE, index=False)
    _stable_rows().to_csv(source / STABLE_INSTRUMENTS_FILE, index=False)
    artifact_sha256 = {
        "unmatched": _sha256(source / UNMATCHED_FILE),
        "reconciliation": _sha256(source / RECONCILIATION_FILE),
        "stable_instruments": _sha256(source / STABLE_INSTRUMENTS_FILE),
    }
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at_utc": NOW.isoformat(),
                "artifact_sha256": artifact_sha256,
            }
        ),
        encoding="utf-8",
    )
    return source


def test_classifier_emits_every_action_category_and_preserves_evidence() -> None:
    unmatched = _unmatched_rows()
    classified = classify_japan_g1_denominator_differences(
        unmatched,
        unmatched,
        _stable_rows(),
        as_of_date="2026-07-11",
    )
    by_code = classified.set_index("local_code")
    assert set(classified["action_category"]) == set(ACTION_CATEGORIES)
    assert by_code.loc["12345", "ticker_normalization_candidate"] == "1234"
    assert by_code.loc["12345", "action_category"] == "ticker_normalization"
    assert by_code.loc["2000", "action_category"] == "missing_primary_ric"
    assert by_code.loc["3000", "action_category"] == "future_retired_status"
    assert by_code.loc["4000", "action_category"] == "duplicate_quote_listing"
    assert by_code.loc["40A", "action_category"] == "alpha_codes"
    assert by_code.loc["5000", "action_category"] == "out_of_scope_market"
    assert by_code.loc["6000", "action_category"] == "stale_active_flag"
    assert by_code.loc["7000", "action_category"] == "truly_unresolved"
    assert classified["requires_source_backed_resolution"].astype(bool).all()
    assert not classified["auto_waived"].astype(bool).any()
    assert not classified["point_in_time_claimed"].astype(bool).any()
    duplicate_evidence = json.loads(by_code.loc["4000", "stable_row_evidence_json"])
    assert len(duplicate_evidence) == 2
    assert {row["stable_instrument_id"] for row in duplicate_evidence} == {
        "id:dup-a",
        "id:dup-b",
    }


def test_writer_is_atomic_immutable_checksummed_and_queue_ordered(tmp_path: Path) -> None:
    source = _source_packet(tmp_path)
    result = write_japan_g1_reconciliation_qa_artifacts(
        source_dir=source,
        artifact_root=tmp_path / "out",
        run_id="QA",
        clock=lambda: NOW,
    )
    assert result.manifest["immutable"] is True
    assert result.manifest["atomic_publish"] is True
    assert result.manifest["does_not_assert_point_in_time"] is True
    assert result.manifest["historical_pit_ready"] is False
    assert result.manifest["counts"]["classified_rows"] == 8
    assert result.manifest["counts"]["auto_waived_rows"] == 0
    assert result.category_counts["action_category"].tolist() == list(ACTION_CATEGORIES)
    assert result.agent_next_queue["resolvability_rank"].is_monotonic_increasing
    assert result.agent_next_queue["agent_priority"].tolist() == list(range(1, 9))
    for line in result.checksums_path.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        assert _sha256(result.artifact_dir / name) == expected
    assert result.source_checksums_path.exists()
    assert not list((tmp_path / "out").glob(".QA.tmp-*"))
    assert not (tmp_path / "out" / ".QA.lock").exists()

    with pytest.raises(FileExistsError, match="immutable artifact directory"):
        write_japan_g1_reconciliation_qa_artifacts(
            source_dir=source,
            artifact_root=tmp_path / "out",
            run_id="QA",
        )


def test_manifest_checksum_failure_never_publishes_partial_directory(tmp_path: Path) -> None:
    source = _source_packet(tmp_path)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["artifact_sha256"]["unmatched"] = "0" * 64
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="manifest checksum mismatch"):
        write_japan_g1_reconciliation_qa_artifacts(
            source_dir=source,
            artifact_root=tmp_path / "out",
            run_id="BROKEN",
        )
    assert not (tmp_path / "out" / "BROKEN").exists()
    assert not list((tmp_path / "out").glob(".BROKEN.tmp-*"))


def test_real_jpx_packet_has_exact_requested_denominator_counts(tmp_path: Path) -> None:
    source = Path(
        "artifacts/research_universes/JPX-UNIVERSE-RECONCILIATION-V2-20260711"
    )
    if not source.is_dir():
        pytest.skip("workspace JPX reconciliation packet is unavailable")
    result = write_japan_g1_reconciliation_qa_artifacts(
        source_dir=source,
        artifact_root=tmp_path,
        run_id="REAL",
        clock=lambda: NOW,
    )
    assert result.manifest["counts"]["public_in_scope_unmatched_rows"] == 9
    assert result.manifest["counts"]["active_master_only_rows"] == 255
    assert result.manifest["counts"]["classified_rows"] == 264
    assert result.manifest["counts"]["excluded_preclassified_rows"] == 5
    assert sum(result.manifest["category_counts"].values()) == 264
    assert result.manifest["no_unresolved_rows_auto_waived"] is True


def test_runner_defaults_and_run_id_override() -> None:
    path = Path(__file__).resolve().parents[1] / "examples" / "run_japan_g1_reconciliation_qa.py"
    spec = importlib.util.spec_from_file_location("run_japan_g1_reconciliation_qa", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = module.parse_args(["--run-id", "CUSTOM"])
    assert args.run_id == "CUSTOM"
    assert args.source_dir.name == "JPX-UNIVERSE-RECONCILIATION-V2-20260711"
