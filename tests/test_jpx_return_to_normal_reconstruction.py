from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from quant_lab.data.jpx_return_to_normal_reconstruction import (
    build_cli_parser,
    produce_jpx_return_to_normal,
)
from quant_lab.data.jpx_return_to_normal_reconstruction_validator import (
    build_cli_parser as build_validator_cli_parser,
)
from quant_lab.data.jpx_return_to_normal_reconstruction_validator import (
    validate_jpx_return_to_normal,
)


def test_producer_is_fail_closed_atomic_hashed_and_partial(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    result = produce_jpx_return_to_normal(
        run_id="RETURN-PRODUCER",
        artifact_root=tmp_path / "producer_artifacts",
        additional_ohlcv_dirs=[inputs["ohlcv_dir"]],
        max_search_sessions=3,
        **inputs,
    )

    outcomes = pd.read_csv(result.artifact_dir / "event_outcomes.csv", dtype={"security_code": str})
    states = pd.read_csv(result.artifact_dir / "state_chains.csv", dtype={"security_code": str})
    unresolved = pd.read_csv(result.artifact_dir / "unresolved_cases.csv", dtype={"security_code": str})
    extraction = pd.read_csv(
        result.artifact_dir / "missing_price_extraction_plan.csv",
        dtype={"security_code": str},
    )
    routing = pd.read_csv(
        result.artifact_dir / "local_ohlcv_routing_audit.csv",
        dtype={"security_code": str},
    )

    assert result.manifest["status"] == "partial_reconstruction"
    assert result.manifest["immutable"] is True
    assert result.manifest["atomic_publish"] is True
    assert result.manifest["ohlcv_coverage_complete"] is False
    assert result.manifest["full_component_coverage_claimed"] is False
    assert result.manifest["scorecard_eligible"] is False
    assert "partial" in result.manifest["coverage_warning"].lower()
    assert result.manifest["counts"]["activation_cases"] == 5
    assert result.manifest["counts"]["resolved_cases"] == 2
    assert result.manifest["counts"]["unresolved_cases"] == 3
    assert result.manifest["counts"]["canonical_ohlcv_artifacts_audited"] == 2
    assert result.manifest["counts"]["cases_added_by_additional_canonical_routing"] == 0

    by_code = outcomes.set_index("security_code")
    assert by_code.loc["1111", "return_normal_date"] == "2021-01-06"
    assert by_code.loc["1111", "trigger_evidence"] == "positive_volume_non_limit_ohlcv"
    assert by_code.loc["3333", "return_normal_date"] == "2021-01-06"
    assert by_code.loc["3333", "trigger_evidence"] == "admitted_closing_trade_evidence"
    assert by_code.loc["2222", "unresolved_reason"] == "missing_ohlcv_without_closing_evidence"
    assert by_code.loc["4444", "unresolved_reason"] == "zero_volume_without_closing_evidence"
    assert by_code.loc["5555", "unresolved_reason"] == (
        "positive_volume_boundary_only_without_closing_evidence"
    )
    assert set(unresolved["security_code"]) == {"2222", "4444", "5555"}
    assert set(states.loc[states["state"].eq("return_normal_proven"), "security_code"]) == {
        "1111",
        "3333",
    }
    historical_request = extraction.loc[extraction["security_code"].eq("2222")].iloc[0]
    assert historical_request["refinitiv_ric"] == "2222.T^A22"
    assert historical_request["execution_enabled"] in {False, "False"}
    assert historical_request["writer_owner"] == "single_vendor_writer"
    assert set(routing["audit_decision"]) == {"unresolved_no_exact_local_ohlcv"}
    assert not list((tmp_path / "producer_artifacts").glob(".*.tmp-*"))
    for filename, digest in result.manifest["output_sha256"].items():
        assert _sha(result.artifact_dir / filename) == digest

    with pytest.raises(FileExistsError, match="immutable reconstruction"):
        produce_jpx_return_to_normal(
            run_id="RETURN-PRODUCER",
            artifact_root=tmp_path / "producer_artifacts",
            additional_ohlcv_dirs=[inputs["ohlcv_dir"]],
            max_search_sessions=3,
            **inputs,
        )


def test_independent_validator_passes_valid_partial_lane_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    inputs = _write_inputs(tmp_path)
    producer = produce_jpx_return_to_normal(
        run_id="RETURN-PRODUCER",
        artifact_root=tmp_path / "producer_artifacts",
        max_search_sessions=3,
        **inputs,
    )
    validation = validate_jpx_return_to_normal(
        run_id="RETURN-VALIDATOR",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validation_artifacts",
    )
    assert validation.manifest["status"] == "pass_partial_lane"
    assert validation.manifest["validation_issues"] == []
    assert validation.manifest["full_component_coverage_claimed"] is False
    detail = pd.read_csv(validation.artifact_dir / "case_validation.csv")
    assert detail["validation_pass"].all()

    outcomes_path = producer.artifact_dir / "event_outcomes.csv"
    outcomes = pd.read_csv(outcomes_path, dtype={"security_code": str})
    outcomes.loc[outcomes["security_code"].eq("1111"), "return_normal_date"] = "2021-01-07"
    outcomes.to_csv(outcomes_path, index=False)
    failed = validate_jpx_return_to_normal(
        run_id="RETURN-VALIDATOR-TAMPERED",
        producer_dir=producer.artifact_dir,
        artifact_root=tmp_path / "validation_artifacts",
    )
    assert failed.manifest["status"] == "failed_validation"
    assert any("producer_output_hash_mismatch:event_outcomes.csv" in issue for issue in failed.manifest["validation_issues"])


def test_producer_rejects_mutated_upstream_before_publish(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    events_path = inputs["event_dir"] / "events.csv"
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write("tampered\n")
    with pytest.raises(ValueError, match="declared output hash mismatch"):
        produce_jpx_return_to_normal(
            run_id="MUTATED",
            artifact_root=tmp_path / "producer_artifacts",
            **inputs,
        )
    assert not (tmp_path / "producer_artifacts" / "MUTATED").exists()


def test_lineage_admitted_raw_exact_historical_ric_can_resolve_case(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    raw_root, historical = _write_raw_audit_bundle(tmp_path)
    result = produce_jpx_return_to_normal(
        run_id="RAW-ROUTED",
        artifact_root=tmp_path / "producer_artifacts",
        raw_ohlcv_audit_root=raw_root,
        raw_historical_dir=historical,
        max_search_sessions=3,
        **inputs,
    )

    outcomes = pd.read_csv(result.artifact_dir / "event_outcomes.csv", dtype={"security_code": str})
    routed = outcomes.loc[outcomes["security_code"].eq("2222")].iloc[0]
    assert routed["outcome_status"] == "resolved"
    assert routed["return_normal_date"] == "2021-01-06"
    assert result.manifest["counts"]["raw_exact_rows_admitted"] == 1
    assert result.manifest["counts"]["cases_added_by_raw_routing"] == 1
    audit = pd.read_csv(result.artifact_dir / "local_ohlcv_routing_audit.csv")
    audit_row = audit.loc[audit["security_code"].astype(str).eq("2222")].iloc[0]
    assert audit_row["raw_routing_resolved"] in {True, "True"}
    assert audit_row["raw_same_code_wrong_ric_rows"] == 0
    plan = pd.read_csv(
        result.artifact_dir / "missing_price_extraction_plan.csv", dtype={"security_code": str}
    )
    assert "2222" not in set(plan["security_code"])

    validation = validate_jpx_return_to_normal(
        run_id="RAW-ROUTED-VALIDATION",
        producer_dir=result.artifact_dir,
        artifact_root=tmp_path / "validation_artifacts",
    )
    assert validation.manifest["status"] == "pass_partial_lane"


def test_owned_cli_parsers_expose_required_inputs() -> None:
    args = build_cli_parser().parse_args(
        [
            "--run-id", "RUN", "--event-dir", "event", "--event-validation-dir", "validation",
            "--listing-dir", "listing", "--calendar-validation-dir", "calendar",
            "--ohlcv-dir", "ohlcv",
        ]
    )
    assert args.run_id == "RUN"
    assert args.max_search_sessions == 20
    validator_args = build_validator_cli_parser().parse_args(
        ["--run-id", "VALIDATE", "--producer-dir", "producer"]
    )
    assert validator_args.producer_dir == "producer"


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    event = tmp_path / "event"
    event_validation = tmp_path / "event_validation"
    listing = tmp_path / "listing"
    calendar = tmp_path / "calendar"
    ohlcv = tmp_path / "ohlcv"
    closing = tmp_path / "closing"
    for path in (event, event_validation, listing, calendar, ohlcv, closing):
        path.mkdir(parents=True)

    event_rows = []
    quality_rows = []
    detail_rows = []
    boundaries = {
        "1111": 120.0,
        "2222": 220.0,
        "3333": 320.0,
        "4444": 420.0,
        "5555": 520.0,
    }
    for index, (code, boundary) in enumerate(boundaries.items(), 1):
        source_url = f"https://www.jpx.co.jp/english/news/1030/20210104-{index:02d}.html"
        source_sha = hashlib.sha256(source_url.encode()).hexdigest()
        event_rows.append(
            {
                "notice_id": f"JPX-EPL-{index:05d}",
                "archive_publication_date": "2021-01-04",
                "publication_date": "2021-01-04",
                "effective_date": "2021-01-05",
                "event_type": "activate",
                "security_code": code,
                "security_name": f"Issue {code}",
                "side": "upper",
                "base_price_yen": boundary - 40,
                "expanded_range_yen": 40,
                "normal_range_yen": 10,
                "upper_limit_price_yen": boundary,
                "lower_limit_price_yen": boundary - 50,
                "observed_multiplier": 4.0,
                "publication_date_consistent": True,
                "source_url": source_url,
                "source_sha256": source_sha,
                "title": "Daily Price Limits to be Broadened",
            }
        )
        quality_rows.append(
            {
                "publication_date": "2021-01-04",
                "event_type": "activate",
                "security_code": code,
                "source_url": source_url,
                "validation_pass": True,
                "validation_issues": "",
            }
        )
        detail_rows.append(
            {
                "publication_date": "2021-01-04",
                "event_type": "BROADENING",
                "code": code,
                "source_url": source_url,
                "resolved_side": "UPPER",
                "validated_base_price_yen": boundary - 40,
                "validated_expanded_range_yen": 40,
                "validated_normal_range_yen": 10,
                "validated_observed_multiplier": 4.0,
                "overlay_applied": False,
                "post_overlay_numeric_discrepancy": False,
                "post_overlay_regime_discrepancy": False,
                "expected_multiplier": 4.0,
            }
        )
    pd.DataFrame(event_rows).to_csv(event / "events.csv", index=False)
    pd.DataFrame(quality_rows).to_csv(event / "event_quality.csv", index=False)
    event_hashes = {
        name: _sha(event / name) for name in ("events.csv", "event_quality.csv")
    }
    _write_manifest(
        event,
        {
            "run_id": "EVENT",
            "artifact_type": "jpx_english_exceptional_price_limit_notice_events",
            "status": "complete_denominator_partial_component",
            "immutable": True,
            "official_jpx_source": True,
            "event_type_counts": {"activate": 5},
            "output_sha256": event_hashes,
        },
    )

    pd.DataFrame(detail_rows).to_csv(
        event_validation / "event_validation_detail.csv", index=False
    )
    _write_manifest(
        event_validation,
        {
            "run_id": "EVENT-VALIDATION",
            "artifact_type": (
                "japan_exceptional_price_limit_bilingual_reconciliation_independent_validation"
            ),
            "status": "pass_partial_component",
            "independent_validation": True,
            "validation_issues": [],
            "output_sha256": {
                "event_validation_detail.csv": _sha(
                    event_validation / "event_validation_detail.csv"
                )
            },
        },
    )

    pd.DataFrame(
        [
            {
                "stable_listing_id": f"JPX:{code}:2020-01:1",
                "stable_instrument_id": f"isin:JP{code}",
                "refinitiv_ric": "2222.T^A22" if code == "2222" else f"{code}.T",
                "local_code": code,
                "exchange_code": "TYO",
                "listing_start": "2020-01-01",
                "listing_end": "2021-12-31",
                "right_open": True,
                "pit_usable": True,
            }
            for code in boundaries
        ]
    ).to_parquet(listing / "listing_windows.parquet", index=False)
    _write_manifest(
        listing,
        {
            "run_id": "LISTING",
            "artifact_type": "japan_g1_pit_materializer_producer",
            "status": "complete",
            "historical_pit_ready": True,
            "official_jpx_source": True,
            "critical_blockers": [],
            "output_sha256": {
                "listing_windows.parquet": _sha(listing / "listing_windows.parquet")
            },
        },
    )

    pd.DataFrame(
        {"session_date": ["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07"]}
    ).to_parquet(calendar / "validated_exchange_calendar.parquet", index=False)
    _write_manifest(
        calendar,
        {
            "run_id": "CALENDAR",
            "artifact_type": "japan_exchange_calendar_independent_validation",
            "status": "pass_component",
            "independent_validation": True,
            "official_jpx_calendar": True,
            "critical_blockers": [],
            "output_sha256": {
                "validated_exchange_calendar.parquet": _sha(
                    calendar / "validated_exchange_calendar.parquet"
                )
            },
        },
    )

    partition = (
        ohlcv
        / "ohlcv/provider_partition=fixture/interval_partition=1d/adjustment_partition=unadjusted"
        / "country_partition=japan/exchange_partition=jpx/year_partition=2021/month_partition=1"
    )
    partition.mkdir(parents=True)
    price_path = partition / "part-00000.parquet"
    pd.DataFrame(
        [
            _price("1111.T", 120, 120, 110, 115, 100),
            _price("3333.T", 320, 320, 320, 320, 0),
            _price("4444.T", 420, 420, 420, 420, 0),
            _price("5555.T", 520, 520, 520, 520, 100),
        ]
    ).to_parquet(price_path, index=False)
    relative = str(price_path.relative_to(ohlcv))
    (ohlcv / "hash_inventory.json").write_text(
        json.dumps(
            {
                "algorithm": "sha256",
                "immutable": True,
                "coverage_complete": False,
                "files": {relative: _sha(price_path)},
            }
        ),
        encoding="utf-8",
    )
    _write_manifest(
        ohlcv,
        {
            "run_id": "OHLCV",
            "artifact_type": "japan_listing_window_canonical_ohlcv",
            "status": "pass_partial_candidate",
            "evidence_scope": "partial_candidate",
            "quality_passed": True,
            "source_extraction_complete": False,
            "promotion_eligible": False,
            "artifacts": {"ohlcv_root": "ohlcv", "hash_inventory": "hash_inventory.json"},
        },
    )

    raw_closing = closing / "raw_order_book_3333.bin"
    raw_closing.write_bytes(b"independently retained closing trade evidence")
    closing_columns = {
        "code": "3333",
        "session_date": "2021-01-05",
        "evidence_kind": "closing_auction",
        "executed_price": 310,
        "executed_volume": 10,
        "session_volume": 10,
        "pre_close_volume": 0,
        "closing_auction_price": 310,
        "closing_auction_volume": 10,
        "closing_auction_executed": True,
        "boundary_orders_remaining": False,
        "best_bid_price": 309,
        "best_bid_size": 5,
        "best_ask_price": 310,
        "best_ask_size": 5,
        "source_path": str(raw_closing.resolve()),
        "source_sha256": _sha(raw_closing),
    }
    pd.DataFrame([closing_columns]).to_csv(closing / "closing_evidence.csv", index=False)
    _write_manifest(
        closing,
        {
            "run_id": "CLOSING",
            "artifact_type": "jpx_closing_quote_order_book_evidence",
            "status": "complete",
            "immutable": True,
            "independent_validation": True,
            "validation_issues": [],
            "output_sha256": {
                "closing_evidence.csv": _sha(closing / "closing_evidence.csv")
            },
        },
    )
    return {
        "event_dir": event,
        "event_validation_dir": event_validation,
        "listing_dir": listing,
        "calendar_validation_dir": calendar,
        "ohlcv_dir": ohlcv,
        "closing_evidence_dir": closing,
    }


def _price(ticker: str, open_: float, high: float, low: float, close: float, volume: float) -> dict[str, object]:
    return {
        "date": "2021-01-05",
        "ticker": ticker,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


def _write_raw_audit_bundle(tmp_path: Path) -> tuple[Path, Path]:
    raw_root = (
        tmp_path
        / "raw_active/ohlcv/provider_partition=fixture/interval_partition=1d"
        / "adjustment_partition=unadjusted/country_partition=japan/exchange_partition=jpx"
        / "year_partition=2021/month_partition=1"
    )
    raw_root.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                **_price("2222.T^A22", 220, 220, 210, 215, 100),
                "local_symbol": "2222",
                "batch_id": "RAW-001",
            }
        ]
    ).to_parquet(raw_root / "part-00000.parquet", index=False)
    historical = tmp_path / "raw_historical"
    historical.mkdir()
    pd.DataFrame(
        {
            "plan_row_id": ["RAW-001"],
            "start_date": ["2021-01-01"],
            "end_date": ["2021-01-31"],
            "tickers": ["2222.T^A22"],
        }
    ).to_csv(historical / "historical_plan.csv", index=False)
    pd.DataFrame(
        {
            "plan_row_id": ["RAW-001"],
            "status": ["completed"],
            "start_date": ["2021-01-01"],
            "end_date": ["2021-01-31"],
            "tickers": ["2222.T^A22"],
        }
    ).to_csv(historical / "historical_manifest.csv", index=False)
    (historical / "historical_run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": "RAW-HISTORICAL",
                "long_ohlcv_root": str(raw_root.parents[6].resolve()),
                "planned_rows": 1,
                "completed_rows": 1,
                "failed_rows": 0,
            }
        ),
        encoding="utf-8",
    )
    return raw_root.parents[6], historical


def _write_manifest(root: Path, payload: dict[str, object]) -> None:
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
