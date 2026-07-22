from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.japan_g4_halt_history import (
    DELISTING_CHANNEL,
    EVENT_COLUMNS,
    HALT_CHANNEL,
    PIT_MAPPING_STATUSES,
    PIT_SAFE_STATUSES,
    SOURCE_CATALOG_COLUMNS,
    _copy_validated_tse_wayback_lifecycle,
    _copy_validated_tse_wayback_lifecycles,
    _attach_pit_mapping,
    _build_coverage,
    _build_completion_outputs,
    _resolve_pit_timestamps,
    _validated_halt_mapping_recovery_counts,
    _merge_report_event_packets,
    _normalize_tse_wayback_lifecycle_pairs,
    _normalize_halt_mapping_recovery_pairs,
    build_halt_acquisition_plan,
    produce_japan_g4_halt_history,
)
from quant_lab.data.japan_g4_halt_report_admission import (
    normalize_halt_report_events,
    update_halt_report_admission,
)
from quant_lab.data.japan_g4_halt_history_validator import (
    validate_japan_g4_halt_history,
)


HALTS = b"""
<table><tr><th>Header</th></tr>
<tr><td>Alpha Co.</td><td>1234</td><td>May 05, 2026</td><td>10:00</td><td>10:15</td><td>10:16</td><td>Merger</td></tr>
</table><table><tr><th>Header</th></tr></table>
"""
DELISTINGS = b"""
<table><tr><th>Date</th><th>Name</th><th>Code</th><th>Market</th><th>Reason</th></tr>
<tr><td>May 20, 2026</td><td>Beta Co.</td><td>5678</td><td>Prime</td><td>Merger</td></tr>
</table>
"""

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TSE_WAYBACK_V2 = (
    PROJECT_ROOT
    / "artifacts/japan_g4_tse_wayback_lifecycle"
    / "JPN-G4-TSE-WAYBACK-LIFECYCLE-2012-PILOT-20260712-V2"
)
TSE_WAYBACK_VALIDATION_V2 = (
    PROJECT_ROOT
    / "artifacts/japan_g4_tse_wayback_lifecycle_validation"
    / "JPN-G4-TSE-WAYBACK-LIFECYCLE-2012-HARDENED-INDEPENDENT-VALIDATION-20260712-V4"
)
TSE_WAYBACK_2013_V2 = (
    PROJECT_ROOT
    / "artifacts/japan_g4_tse_wayback_lifecycle"
    / "JPN-G4-TSE-WAYBACK-LIFECYCLE-2013-PILOT-20260712-V2"
)
TSE_WAYBACK_2013_VALIDATION_V3 = (
    PROJECT_ROOT
    / "artifacts/japan_g4_tse_wayback_lifecycle_validation"
    / "JPN-G4-TSE-WAYBACK-LIFECYCLE-2013-PILOT-INDEPENDENT-VALIDATION-20260712-V3"
)

pytestmark = pytest.mark.skipif(
    not all(
        path.is_dir()
        for path in (
            TSE_WAYBACK_V2,
            TSE_WAYBACK_VALIDATION_V2,
            TSE_WAYBACK_2013_V2,
            TSE_WAYBACK_2013_VALIDATION_V3,
        )
    ),
    reason="private JPX artifact integration tests",
)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def test_halt_mapping_recovery_count_contract_accepts_dynamic_exact_counts() -> None:
    producer = {
        "denominator_rows": 73,
        "accepted_rows": 13,
        "rejected_rows": 60,
    }
    validation = {
        "denominator_rows": 73,
        "accepted_rows": 13,
        "rejected_rows": 60,
        "accepted_rows_validated": 13,
    }

    assert _validated_halt_mapping_recovery_counts(producer, validation) == (
        73,
        13,
        60,
    )


def test_halt_mapping_recovery_count_contract_rejects_count_drift() -> None:
    producer = {
        "denominator_rows": 73,
        "accepted_rows": 13,
        "rejected_rows": 60,
    }
    validation = {
        "denominator_rows": 73,
        "accepted_rows": 12,
        "rejected_rows": 61,
        "accepted_rows_validated": 12,
    }

    with pytest.raises(ValueError, match="count contract failed"):
        _validated_halt_mapping_recovery_counts(producer, validation)


def test_halt_mapping_recovery_pair_cardinality_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="both halt mapping recovery"):
        _normalize_halt_mapping_recovery_pairs("producer", None, None)
    with pytest.raises(ValueError, match="either the legacy"):
        _normalize_halt_mapping_recovery_pairs(
            "producer", "validation", [("producer-2", "validation-2")]
        )


def _catalog(halt_path: Path, delist_path: Path) -> pd.DataFrame:
    rows = [
        {
            "source_id": "halt-page",
            "source_kind": "jpx_halt_html",
            "channel": HALT_CHANNEL,
            "official_source": True,
            "source_url": "https://www.jpx.co.jp/english/markets/equities/suspended/index.html",
            "local_path": str(halt_path),
            "expected_sha256": _sha(HALTS),
            "observed_at_utc": "2026-07-01T00:00:00+00:00",
            "coverage_start": "",
            "coverage_end": "",
            "denominator_complete": False,
            "no_event_capable": False,
            "pit_timestamp_status": "snapshot_observation_only",
        },
        {
            "source_id": "delisting-2026",
            "source_kind": "jpx_delisting_html",
            "channel": DELISTING_CHANNEL,
            "official_source": True,
            "source_url": "https://www.jpx.co.jp/english/listing/stocks/delisted/index.html",
            "local_path": str(delist_path),
            "expected_sha256": _sha(DELISTINGS),
            "observed_at_utc": "2026-07-01T00:00:00+00:00",
            "coverage_start": "2026-01-01",
            "coverage_end": "2026-12-31",
            "denominator_complete": True,
            "no_event_capable": True,
            "pit_timestamp_status": "event_date_only",
        },
    ]
    return pd.DataFrame(rows, columns=SOURCE_CATALOG_COLUMNS)


def _fixture(tmp_path: Path):
    halt = tmp_path / "halts.html"
    delist = tmp_path / "delist.html"
    halt.write_bytes(HALTS)
    delist.write_bytes(DELISTINGS)
    symbol_map = tmp_path / "symbol_map.csv"
    pd.DataFrame(
        [
            {"stable_listing_id": "TSE:1234", "local_code": "1234", "refinitiv_ric": "1234.T", "listing_start": "2000-01-01", "listing_end": "", "known_at": "2025-12-31", "pit_usable": True},
            {"stable_listing_id": "TSE:5678", "local_code": "5678", "refinitiv_ric": "5678.T", "listing_start": "2000-01-01", "listing_end": "2026-05-20", "known_at": "2025-12-31", "pit_usable": True},
        ]
    ).to_csv(symbol_map, index=False)
    return _catalog(halt, delist), symbol_map


def _empty_lifecycle_fixture(tmp_path: Path) -> tuple[pd.DataFrame, Path]:
    symbol_map = tmp_path / "empty_symbol_map.csv"
    pd.DataFrame(columns=["local_code", "refinitiv_ric"]).to_csv(
        symbol_map, index=False
    )
    inventory = tmp_path / "inventory-only.txt"
    inventory.write_text("offline lifecycle packet\n", encoding="utf-8")
    catalog = pd.DataFrame(
        [
            {
                "source_id": "inventory-only",
                "source_kind": "inventory_only",
                "channel": HALT_CHANNEL,
                "official_source": True,
                "source_url": "",
                "local_path": str(inventory),
                "expected_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
                "observed_at_utc": "2026-07-12T00:00:00+00:00",
                "coverage_start": "",
                "coverage_end": "",
                "denominator_complete": False,
                "no_event_capable": False,
                "pit_timestamp_status": "inventory_only",
            }
        ],
        columns=SOURCE_CATALOG_COLUMNS,
    )
    return catalog, symbol_map


def _rewrite_hash_inventory_manifest(packet: Path, inventory: dict[str, str]) -> None:
    inventory_path = packet / "artifact_hashes.json"
    inventory_path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_path = packet / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_hashes_sha256"] = hashlib.sha256(
        inventory_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_acquisition_plan_has_exact_g4_month_channel_denominator() -> None:
    plan = build_halt_acquisition_plan()
    assert len(plan) == 480
    assert plan["month"].nunique() == 240
    assert set(plan["channel"]) == {HALT_CHANNEL, DELISTING_CHANNEL}
    assert plan["plan_id"].nunique() == 480
    assert plan["coverage_status"].eq("uncovered").all()


def test_producer_preserves_event_only_and_official_no_event_evidence(tmp_path: Path) -> None:
    catalog, symbol_map = _fixture(tmp_path)
    result = produce_japan_g4_halt_history(
        run_id="PRODUCER",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        archive_roots=[tmp_path / "halts.html", tmp_path / "delist.html"],
        artifact_root=tmp_path / "artifacts",
        start_date="2026-05-01",
        end_date="2026-06-30",
        clock=lambda: datetime(2026, 7, 2, tzinfo=timezone.utc),
    )
    events = pd.read_parquet(result.events_path)
    coverage = pd.read_csv(result.coverage_path)
    no_event = pd.read_csv(result.artifact_dir / "no_event_evidence.csv")
    assert len(events) == 2
    assert set(events["event_type"]) == {"trading_halt", "delisting_non_trading_interval"}
    assert events["mapping_status"].eq("pit_exact").all()
    assert coverage.loc[(coverage["month"].eq("2026-05")) & (coverage["channel"].eq(HALT_CHANNEL)), "coverage_status"].iloc[0] == "event_evidence_only"
    assert coverage.loc[(coverage["month"].eq("2026-06")) & (coverage["channel"].eq(DELISTING_CHANNEL)), "no_event_evidence"].iloc[0]
    assert no_event[["month", "channel"]].to_records(index=False).tolist() == [("2026-06", DELISTING_CHANNEL)]
    assert result.manifest["status"] == "partial_candidate"
    assert result.manifest["scorecard_eligible"] is False


def test_remote_source_is_checkpointed_and_resume_skips_fetch(tmp_path: Path) -> None:
    symbol_map = tmp_path / "symbol_map.csv"
    pd.DataFrame(columns=["local_code", "refinitiv_ric"]).to_csv(symbol_map, index=False)
    catalog = pd.DataFrame(
        [
            {
                "source_id": "remote-halt",
                "source_kind": "jpx_halt_html",
                "channel": HALT_CHANNEL,
                "official_source": True,
                "source_url": "https://example.invalid/halts",
                "local_path": "",
                "expected_sha256": _sha(HALTS),
                "observed_at_utc": "",
                "coverage_start": "",
                "coverage_end": "",
                "denominator_complete": False,
                "no_event_capable": False,
                "pit_timestamp_status": "snapshot_observation_only",
            }
        ],
        columns=SOURCE_CATALOG_COLUMNS,
    )
    calls = []

    def fetch(url: str, timeout: float) -> bytes:
        calls.append((url, timeout))
        return HALTS

    kwargs = dict(
        run_id="RESUME",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        artifact_root=tmp_path / "artifacts",
        start_date="2026-05-01",
        end_date="2026-05-31",
        fetcher=fetch,
        clock=lambda: datetime(2026, 7, 2, tzinfo=timezone.utc),
    )
    produce_japan_g4_halt_history(**kwargs)
    produce_japan_g4_halt_history(**kwargs)
    assert len(calls) == 1


def test_independent_validator_reparses_raw_sources_and_passes_partial(tmp_path: Path) -> None:
    catalog, symbol_map = _fixture(tmp_path)
    producer = produce_japan_g4_halt_history(
        run_id="PRODUCER",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        artifact_root=tmp_path / "producer",
        start_date="2026-05-01",
        end_date="2026-06-30",
    )
    validated = validate_japan_g4_halt_history(
        run_id="VALIDATOR",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validator",
    )
    assert validated.manifest["status"] == "pass_partial_component"
    assert validated.manifest["validation_issues"] == []
    assert validated.manifest["component_scorecard_eligible"] is False


def test_validated_timestamp_recovery_is_copied_applied_and_revalidated(
    tmp_path: Path,
) -> None:
    catalog, symbol_map = _fixture(tmp_path)
    baseline = produce_japan_g4_halt_history(
        run_id="BASELINE",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        artifact_root=tmp_path / "baseline",
        start_date="2026-05-01",
        end_date="2026-06-30",
    )
    events = pd.read_parquet(baseline.events_path)
    event_id = events.loc[
        events["event_type"].eq("delisting_non_trading_interval"), "event_id"
    ].iloc[0]

    recovery = tmp_path / "recovery"
    recovery.mkdir()
    recovery_row = {
        "event_id": event_id,
        "recovery_status": "recovered",
        "known_at_utc": "2026-05-01T15:00:00+00:00",
        "publication_date": "2026-05-01",
        "timestamp_rule": "first_instant_after_official_publication_date_in_JST",
        "notice_source_path": str(tmp_path / "delist.html"),
        "notice_source_sha256": _sha(DELISTINGS),
    }
    pd.DataFrame([recovery_row]).to_csv(recovery / "timestamp_recovery.csv", index=False)
    producer_manifest = {
        "run_id": "RECOVERY",
        "status": "recovery_partial",
        "output_sha256": {
            "timestamp_recovery.csv": hashlib.sha256(
                (recovery / "timestamp_recovery.csv").read_bytes()
            ).hexdigest()
        },
    }
    (recovery / "manifest.json").write_text(json.dumps(producer_manifest))
    validation_dir = recovery / "validation"
    validation_dir.mkdir()
    validation_manifest = {
        "status": "pass",
        "independent_validation_passed": True,
        "producer_manifest_sha256": hashlib.sha256(
            (recovery / "manifest.json").read_bytes()
        ).hexdigest(),
        "validation_issues": [],
        "output_sha256": {},
    }
    (validation_dir / "manifest.json").write_text(json.dumps(validation_manifest))

    recovered = produce_japan_g4_halt_history(
        run_id="RECOVERED",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        timestamp_recovery_dir=recovery,
        artifact_root=tmp_path / "recovered",
        start_date="2026-05-01",
        end_date="2026-06-30",
    )
    recovered_events = pd.read_parquet(recovered.events_path)
    recovered_event = recovered_events.loc[recovered_events["event_id"].eq(event_id)].iloc[0]
    assert recovered_event["known_at_status"] == "official_jpx_publication_date_recovery"
    assert recovered.manifest["timestamp_recovery_admitted"] is True
    assert recovered.manifest["counts"]["historical_pit_timestamp_gap_rows"] == 0

    validation = validate_japan_g4_halt_history(
        run_id="RECOVERED-VALIDATION",
        producer_dir=recovered.artifact_dir,
        artifact_root=tmp_path / "recovered_validation",
    )
    assert validation.manifest["status"] == "pass_partial_component"
    assert validation.manifest["validation_issues"] == []


def test_independent_validator_detects_raw_source_tampering(tmp_path: Path) -> None:
    catalog, symbol_map = _fixture(tmp_path)
    producer = produce_japan_g4_halt_history(
        run_id="PRODUCER",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        artifact_root=tmp_path / "producer",
        start_date="2026-05-01",
        end_date="2026-06-30",
    )
    raw = producer.artifact_dir / "raw" / "halt-page.html"
    raw.write_bytes(b"tampered")
    validated = validate_japan_g4_halt_history(
        run_id="VALIDATOR",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validator",
    )
    assert validated.manifest["status"] == "failed_validation"
    assert "raw_source_hash_mismatch:halt-page" in validated.manifest["validation_issues"]


def _mapping_event(
    *, event_id: str, event_type: str, event_date: str, code: str
) -> dict[str, str]:
    row = {column: "" for column in EVENT_COLUMNS}
    row.update(
        {
            "event_id": event_id,
            "event_type": event_type,
            "exchange": "TSE",
            "local_code": code,
            "issue_name": code,
            "event_date": event_date,
            "event_start_jst": f"{event_date}T10:00:00+09:00",
            "source_id": "source",
            "source_sha256": "a" * 64,
            "source_record_id": event_id,
            "mapping_status": "unmapped",
        }
    )
    return row


def test_g1_mapping_routes_respect_open_windows_and_reject_successor_rics() -> None:
    events = pd.DataFrame(
        [
            _mapping_event(event_id="halt", event_type="trading_halt", event_date="2026-06-30", code="1234"),
            _mapping_event(event_id="bridge", event_type="delisting_non_trading_interval", event_date="2020-01-02", code="5678"),
            _mapping_event(event_id="foreign", event_type="delisting_non_trading_interval", event_date="2020-02-03", code="8648"),
            _mapping_event(event_id="successor", event_type="delisting_non_trading_interval", event_date="2020-03-04", code="3313"),
        ],
        columns=EVENT_COLUMNS,
    )
    symbol_map = pd.DataFrame(
        [
            {"stable_listing_id": "JPX:1234", "local_code": "1234", "refinitiv_ric": "1234.T", "listing_start": "2010-01-01", "listing_end": "2026-06-30", "known_at": "2010-02-20T15:00:00+00:00", "window_end_known_at": "", "right_open": True, "pit_usable": True},
            {"stable_listing_id": "JPX:5678", "local_code": "5678", "refinitiv_ric": "5678.T^A20", "listing_start": "2010-01-01", "listing_end": "2018-12-31", "known_at": "2010-02-20T15:00:00+00:00", "window_end_known_at": "2019-02-20T15:00:00+00:00", "right_open": False, "pit_usable": True},
            {"stable_listing_id": "JPX:3313", "local_code": "3313", "refinitiv_ric": "9278.T", "listing_start": "2010-01-01", "listing_end": "2018-01-01", "known_at": "2010-02-20T15:00:00+00:00", "window_end_known_at": "2018-03-20T15:00:00+00:00", "right_open": False, "pit_usable": True},
        ]
    ).fillna("")
    security = pd.DataFrame(
        [
            {"local_code": "5678", "candidate_ric": "5678.T^A20", "retire_date": "2020-01-02", "exchange_code": "TYO", "admitted_source_sha256": "b" * 64, "source_record_id": "retire-5678", "source_loaded_at_utc": "2026-07-01T00:00:00+00:00"},
            {"local_code": "8648", "candidate_ric": "8648.T^B20", "retire_date": "2020-02-03", "exchange_code": "TYO", "admitted_source_sha256": "c" * 64, "source_record_id": "retire-8648", "source_loaded_at_utc": "2026-07-01T00:00:00+00:00"},
            {"local_code": "3313", "candidate_ric": "9278.T", "retire_date": "2020-03-04", "exchange_code": "TYO", "admitted_source_sha256": "d" * 64, "source_record_id": "successor-ric", "source_loaded_at_utc": "2026-07-01T00:00:00+00:00"},
        ]
    ).fillna("")
    mapped = _attach_pit_mapping(
        events,
        symbol_map,
        security_candidates=security,
        symbol_map_sha256="e" * 64,
    ).set_index("event_id")
    assert mapped.loc["halt", "mapping_status"] == "pit_exact_g1_monthly"
    assert mapped.loc["bridge", "mapping_status"] == "pit_exact_g1_identity_retire_bridge"
    assert mapped.loc["foreign", "mapping_status"] == "deterministic_reference_post_event"
    assert mapped.loc["successor", "mapping_status"] == "missing_admitted_mapping"
    assert mapped.loc["successor", "refinitiv_ric"] == ""


def test_conservative_timestamps_and_completion_plan_quantify_remaining_gaps() -> None:
    events = pd.DataFrame(
        [
            _mapping_event(event_id="halt", event_type="trading_halt", event_date="2020-01-02", code="1234"),
            _mapping_event(event_id="delist", event_type="delisting_non_trading_interval", event_date="2020-02-03", code="5678"),
            _mapping_event(event_id="gap", event_type="delisting_non_trading_interval", event_date="2020-03-04", code="6697"),
        ],
        columns=EVENT_COLUMNS,
    )
    events.loc[events.event_id.eq("delist"), ["stable_listing_id", "mapping_status", "refinitiv_ric"]] = ["JPX:5678", "pit_exact_g1_monthly", "5678.T"]
    events.loc[events.event_id.eq("gap"), "mapping_status"] = "missing_admitted_mapping"
    symbol_map = pd.DataFrame(
        [
            {"stable_listing_id": "JPX:5678", "local_code": "5678", "refinitiv_ric": "5678.T", "listing_start": "2010-01-01", "listing_end": "2020-02-03", "known_at": "2010-02-20T15:00:00+00:00", "window_end_known_at": "2020-03-20T15:00:00+00:00", "right_open": False, "pit_usable": True},
        ]
    )
    resolved = _resolve_pit_timestamps(events, symbol_map, "f" * 64)
    assert set(resolved.loc[resolved.event_id.isin(["halt", "delist"]), "known_at_status"]).issubset(PIT_SAFE_STATUSES)
    assert resolved.loc[resolved.event_id.eq("gap"), "known_at_utc"].iloc[0] == ""
    coverage = pd.DataFrame(
        [
            {"plan_id": "2020-01-halt", "month": "2020-01", "channel": HALT_CHANNEL, "source_covered": False, "event_rows": 1},
            {"plan_id": "2020-02-delist", "month": "2020-02", "channel": DELISTING_CHANNEL, "source_covered": True, "event_rows": 1},
        ]
    )
    gaps, completion = _build_completion_outputs(coverage, resolved)
    assert gaps.groupby("gap_type").size().to_dict() == {
        "historical_pit_timestamp_unresolved": 1,
        "ric_mapping_unresolved": 2,
        "source_denominator_uncovered": 1,
    }
    assert completion["irreducible_rows"].sum() == len(gaps)
    assert resolved.loc[resolved.event_id.eq("delist"), "mapping_status"].iloc[0] in PIT_MAPPING_STATUSES


def test_report_admission_uses_report_month_and_quarantines_malformed_rows() -> None:
    base = {
        "report_month": "2024-06",
        "channel": HALT_CHANNEL,
        "event_type": "trading_halt_report_row",
        "raw_code": "1234",
        "normalized_local_code": "1234",
        "instrument_class": "equity",
        "issue_name": "Alpha",
        "halt_start_printed": "10:00",
        "halt_end_printed": "10:30",
        "halt_start_jst": "2022-03-17T10:00:00+09:00",
        "halt_end_jst": "2022-03-17T10:30:00+09:00",
        "source_url": "https://www.jpx.co.jp/report.pdf",
        "source_path": "/sealed/report.pdf",
        "source_sha256": "a" * 64,
        "source_page": 1,
        "source_row_ordinal": 1,
        "source_record_text": "row",
        "known_at": "",
        "ric": "",
        "mapping_status": "pending",
        "timestamp_status": "pending",
        "admissible_to_shared_halt_history": False,
    }
    rows = []
    for event_id, record_sha, changes in (
        ("recurring", "b" * 64, {"event_date": "2022-03-17"}),
        ("future", "c" * 64, {"event_date": "2025-06-01"}),
        ("missing-start", "d" * 64, {"event_date": "2024-06-03", "halt_start_jst": ""}),
        ("bond", "e" * 64, {"event_date": "2024-06-04", "instrument_class": "straight_bond"}),
    ):
        row = {**base, "event_id": event_id, "source_record_sha256": record_sha, **changes}
        rows.append(row)
    normalized, admission = normalize_halt_report_events(
        pd.DataFrame(rows), event_columns=EVENT_COLUMNS
    )
    assert normalized["event_id"].tolist() == ["jpx-report-recurring"]
    statuses = admission.set_index("report_event_id")["normalization_status"].to_dict()
    assert statuses == {
        "recurring": "normalized",
        "future": "future_dated_source_record",
        "missing-start": "missing_halt_start",
        "bond": "out_of_research_universe",
    }
    normalized.loc[:, "mapping_status"] = "pit_exact"
    normalized.loc[:, "known_at_status"] = "conservative_halt_start_market_state"
    updated = update_halt_report_admission(
        admission,
        normalized,
        pit_mapping_statuses=PIT_MAPPING_STATUSES,
        pit_timestamp_statuses=PIT_SAFE_STATUSES,
    )
    coverage, _ = _build_coverage(
        start=pd.Timestamp("2024-06-01").date(),
        end=pd.Timestamp("2024-06-30").date(),
        catalog=pd.DataFrame(columns=SOURCE_CATALOG_COLUMNS),
        requests=pd.DataFrame(columns=["source_id", "status"]),
        events=normalized,
        report_admission=updated,
    )
    halt = coverage.loc[coverage["channel"].eq(HALT_CHANNEL)].iloc[0]
    assert halt["source_covered"]
    assert halt["event_rows"] == 1
    assert halt["coverage_status"] == "source_complete_pit_or_mapping_incomplete"
    assert "report_event_normalization_incomplete" in halt["blockers"]


def test_tse_wayback_lifecycle_clean_packet_is_admitted_and_revalidated(
    tmp_path: Path,
) -> None:
    catalog, symbol_map = _empty_lifecycle_fixture(tmp_path)
    produced = produce_japan_g4_halt_history(
        run_id="TSE-WAYBACK-CLEAN",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        tse_wayback_lifecycle_dir=TSE_WAYBACK_V2,
        tse_wayback_lifecycle_validation_dir=TSE_WAYBACK_VALIDATION_V2,
        artifact_root=tmp_path / "producer",
        start_date="2012-01-01",
        end_date="2012-12-31",
    )
    admission = pd.read_csv(
        produced.artifact_dir / "report_event_admission.csv", dtype=str
    ).fillna("")
    lineage = pd.read_csv(
        produced.artifact_dir / "report_event_lineage.csv", dtype=str
    ).fillna("")
    events = pd.read_parquet(produced.events_path).fillna("")
    no_event = pd.read_csv(produced.artifact_dir / "no_event_evidence.csv")

    assert len(admission) == 240
    assert admission["in_research_scope"].str.lower().eq("true").sum() == 185
    assert admission["normalization_status"].eq("normalized").sum() == 181
    assert admission["normalization_status"].eq("out_of_research_universe").sum() == 55
    assert admission["normalization_status"].eq("invalid_halt_interval").sum() == 1
    assert len(events) == 181
    assert len(lineage) == 240
    assert lineage["disposition"].eq("admitted").all()
    assert no_event.empty
    packet_summary = produced.manifest["tse_wayback_lifecycle_packet"]
    assert packet_summary["packet_count"] == 1
    assert packet_summary["month_count"] == 12
    assert packet_summary["source_rows"] == 240
    assert packet_summary["covered_claims"] == 24
    assert packet_summary["common_equity_rows"] == 185
    assert packet_summary["preserved_non_equity_rows"] == 55
    assert packet_summary["zero_event_claims"] == 0
    assert len(packet_summary["packets"]) == 1
    copied_claims = pd.read_csv(
        produced.artifact_dir
        / "inputs/tse_wayback_lifecycle/packets/packet-0001/producer/exact_nonempty_claims.csv",
        dtype=str,
    ).fillna("")
    assert copied_claims["event_count"].astype(int).gt(0).all()
    assert copied_claims["zero_event_asserted"].str.lower().eq("false").all()

    validated = validate_japan_g4_halt_history(
        run_id="TSE-WAYBACK-CLEAN-VALIDATION",
        producer_dir=produced.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validated.manifest["status"] == "pass_partial_component"
    assert validated.manifest["validation_issues"] == []
    assert validated.manifest["tse_wayback_lifecycle_reparsed"] is True
    assert validated.manifest["counts"]["tse_wayback_lifecycle_claims"] == 24


def test_tse_wayback_lifecycle_directories_are_a_required_pair(tmp_path: Path) -> None:
    catalog, symbol_map = _empty_lifecycle_fixture(tmp_path)
    with pytest.raises(ValueError, match="both TSE Wayback lifecycle"):
        produce_japan_g4_halt_history(
            run_id="UNPAIRED",
            source_catalog=catalog,
            symbol_map_path=symbol_map,
            tse_wayback_lifecycle_dir=TSE_WAYBACK_V2,
            artifact_root=tmp_path / "producer",
            start_date="2012-01-01",
            end_date="2012-12-31",
        )


def test_tse_wayback_lifecycle_pair_cardinality_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="must contain producer and validation"):
        _normalize_tse_wayback_lifecycle_pairs(
            None,
            None,
            [(TSE_WAYBACK_V2,)],  # type: ignore[list-item]
        )
    with pytest.raises(ValueError, match="either the legacy"):
        _normalize_tse_wayback_lifecycle_pairs(
            TSE_WAYBACK_V2,
            TSE_WAYBACK_VALIDATION_V2,
            [(TSE_WAYBACK_2013_V2, TSE_WAYBACK_2013_VALIDATION_V3)],
        )


def test_two_month_disjoint_lifecycle_packets_are_admitted_and_revalidated(
    tmp_path: Path,
) -> None:
    catalog, symbol_map = _empty_lifecycle_fixture(tmp_path)
    produced = produce_japan_g4_halt_history(
        run_id="TSE-WAYBACK-TWO-PACKETS",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        tse_wayback_lifecycle_pairs=[
            (TSE_WAYBACK_V2, TSE_WAYBACK_VALIDATION_V2),
            (TSE_WAYBACK_2013_V2, TSE_WAYBACK_2013_VALIDATION_V3),
        ],
        artifact_root=tmp_path / "producer",
        start_date="2012-01-01",
        end_date="2013-12-31",
    )
    summary = produced.manifest["tse_wayback_lifecycle_packet"]
    admission = pd.read_csv(
        produced.artifact_dir / "report_event_admission.csv", dtype=str
    ).fillna("")
    assert summary["packet_count"] == 2
    assert summary["month_count"] == 24
    assert summary["covered_claims"] == 48
    assert summary["source_rows"] == 520
    assert summary["common_equity_rows"] + summary["preserved_non_equity_rows"] == 520
    assert len(summary["packets"]) == 2
    assert len(admission) == 520
    assert set(admission["report_month"].str[:4]) == {"2012", "2013"}

    validated = validate_japan_g4_halt_history(
        run_id="TSE-WAYBACK-TWO-PACKETS-VALIDATION",
        producer_dir=produced.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validated.manifest["status"] == "pass_partial_component"
    assert validated.manifest["validation_issues"] == []
    assert validated.manifest["counts"]["tse_wayback_lifecycle_packets"] == 2
    assert validated.manifest["counts"]["tse_wayback_lifecycle_months"] == 24
    assert validated.manifest["counts"]["tse_wayback_lifecycle_claims"] == 48
    assert validated.manifest["counts"]["tse_wayback_lifecycle_source_rows"] == 520


def test_lifecycle_packets_reject_overlapping_month_channel_claims(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="overlapping TSE Wayback lifecycle"):
        _copy_validated_tse_wayback_lifecycles(
            [
                (TSE_WAYBACK_V2, TSE_WAYBACK_VALIDATION_V2),
                (TSE_WAYBACK_V2, TSE_WAYBACK_VALIDATION_V2),
            ],
            tmp_path / "copied",
        )


def test_independent_validator_rejects_copied_lifecycle_packet_tamper(
    tmp_path: Path,
) -> None:
    catalog, symbol_map = _empty_lifecycle_fixture(tmp_path)
    produced = produce_japan_g4_halt_history(
        run_id="TSE-WAYBACK-TAMPER",
        source_catalog=catalog,
        symbol_map_path=symbol_map,
        tse_wayback_lifecycle_pairs=[
            (TSE_WAYBACK_V2, TSE_WAYBACK_VALIDATION_V2),
            (TSE_WAYBACK_2013_V2, TSE_WAYBACK_2013_VALIDATION_V3),
        ],
        artifact_root=tmp_path / "producer",
        start_date="2012-01-01",
        end_date="2013-12-31",
    )
    copied_pdf = (
        produced.artifact_dir
        / "inputs/tse_wayback_lifecycle/packets/packet-0002/producer/raw/pdfs/2013-01.pdf"
    )
    copied_pdf.write_bytes(copied_pdf.read_bytes() + b"tampered")

    validated = validate_japan_g4_halt_history(
        run_id="TSE-WAYBACK-TAMPER-VALIDATION",
        producer_dir=produced.artifact_dir,
        artifact_root=tmp_path / "validation",
    )
    assert validated.manifest["status"] == "failed_validation"
    assert any(
        "packet-0002" in issue or "2013-01" in issue
        for issue in validated.manifest["validation_issues"]
    )


@pytest.mark.parametrize("failure", ["tamper", "traversal", "lineage"])
def test_tse_wayback_lifecycle_packet_failures_are_rejected(
    tmp_path: Path, failure: str
) -> None:
    producer = tmp_path / "producer_packet"
    validation = tmp_path / "validation_packet"
    shutil.copytree(TSE_WAYBACK_V2, producer)
    shutil.copytree(TSE_WAYBACK_VALIDATION_V2, validation)
    if failure == "tamper":
        path = producer / "exact_events.csv"
        path.write_bytes(path.read_bytes() + b"tampered")
    elif failure == "traversal":
        inventory = json.loads(
            (producer / "artifact_hashes.json").read_text(encoding="utf-8")
        )
        inventory["../escape.csv"] = "0" * 64
        _rewrite_hash_inventory_manifest(producer, inventory)
        validation_manifest_path = validation / "manifest.json"
        validation_manifest = json.loads(
            validation_manifest_path.read_text(encoding="utf-8")
        )
        validation_manifest["producer_manifest_sha256"] = hashlib.sha256(
            (producer / "manifest.json").read_bytes()
        ).hexdigest()
        validation_manifest_path.write_text(
            json.dumps(validation_manifest), encoding="utf-8"
        )
    else:
        manifest_path = validation / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["producer_manifest_sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError):
        _copy_validated_tse_wayback_lifecycle(
            producer, validation, tmp_path / "copied"
        )


def test_report_packet_overlap_is_deduplicated_with_one_to_one_lineage() -> None:
    lifecycle = pd.read_csv(TSE_WAYBACK_V2 / "exact_events.csv", dtype=str).fillna("")
    selected = lifecycle.loc[
        lifecycle["report_month"].eq("2012-01")
        & lifecycle["channel"].eq(HALT_CHANNEL)
    ].copy()
    prior = selected.copy()
    prior["event_id"] = prior["event_id"].map(lambda value: f"prior-{value}")
    prior["source_record_sha256"] = prior["source_record_sha256"].map(
        lambda value: hashlib.sha256(f"prior-{value}".encode()).hexdigest()
    )

    merged, lineage = _merge_report_event_packets(
        [("halt_report_events", prior), ("tse_wayback_lifecycle", selected)]
    )
    assert len(merged) == len(selected)
    assert len(lineage) == len(selected) * 2
    assert lineage["disposition"].value_counts().to_dict() == {
        "duplicate_equivalent_superseded": len(selected),
        "admitted": len(selected),
    }
    superseded = lineage.loc[
        lineage["disposition"].eq("duplicate_equivalent_superseded")
    ]
    assert superseded["selected_source_record_sha256"].nunique() == len(selected)

    mismatched = prior.copy()
    mismatched.loc[mismatched.index[0], "issue_name"] = "different issue"
    with pytest.raises(ValueError, match="exact row sets disagree"):
        _merge_report_event_packets(
            [("halt_report_events", mismatched), ("tse_wayback_lifecycle", selected)]
        )
