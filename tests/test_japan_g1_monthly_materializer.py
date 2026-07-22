from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from quant_lab.data.japan_g1_monthly_materializer import (
    _map_segments,
    _normalize_security_master,
    _preferred_exchanges,
    _targeted_shard_blockers,
    write_japan_g1_monthly_materializer,
)
from quant_lab.data.jpx_monthly_quotations import conservative_quotation_known_at


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _monthly_source(root: Path) -> Path:
    source = root / "monthly"
    source.mkdir()
    periods = pd.period_range("2006-07", "2026-06", freq="M")
    membership = pd.DataFrame(
        {
            "observation_month": [str(period) for period in periods],
            "membership_effective_month_end": [period.end_time.date().isoformat() for period in periods],
            "known_at_utc": [conservative_quotation_known_at(period) for period in periods],
            "known_at_policy": ["conservative"] * len(periods),
            "code": ["1001"] * len(periods),
            "issue_name_japanese": ["One"] * len(periods),
            "source_families": ["TSE"] * len(periods),
            "source_artifacts": ["fixture"] * len(periods),
            "source_sha256": ["a" * 64] * len(periods),
            "evidence_rows": [1] * len(periods),
            "currency": ["JPY"] * len(periods),
            "membership_observed": [True] * len(periods),
            "pit_usable": [True] * len(periods),
        }
    )
    files = {
        "monthly_membership.parquet": source / "monthly_membership.parquet",
        "monthly_coverage.csv": source / "monthly_coverage.csv",
        "monthly_churn.csv": source / "monthly_churn.csv",
        "anomalies.csv": source / "anomalies.csv",
    }
    membership.to_parquet(files["monthly_membership.parquet"], index=False)
    for name in ("monthly_coverage.csv", "monthly_churn.csv", "anomalies.csv"):
        files[name].write_text("value\n", encoding="utf-8")
    manifest = {
        "status": "pass_candidate_source",
        "independent_validation": True,
        "planned_months": 240,
        "passed_months": 240,
        "anomalies": 0,
        "evidence_scope": "full_20_year_monthly_tse_ose_union",
        "artifacts": {
            "monthly_membership": str(files["monthly_membership.parquet"]),
            "monthly_coverage": str(files["monthly_coverage.csv"]),
            "monthly_churn": str(files["monthly_churn.csv"]),
            "anomalies": str(files["anomalies.csv"]),
        },
        "output_sha256": {name: _sha(path) for name, path in files.items()},
    }
    (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return source


def _master(root: Path, exchange: str = "TYO") -> Path:
    source = root / f"master-{exchange}"
    source.mkdir()
    frame = pd.DataFrame(
        {
            "ric": ["1001.T"],
            "ticker_symbol": ["1001"],
            "exchange_code": [exchange],
            "rcs_asset_category_leaf": ["Ordinary Share"],
            "first_trade_date": ["2000-01-01"],
            "retire_date": [None],
            "isin": ["JP0000001001"],
            "sedol": ["0001001"],
            "perm_id": ["1"],
            "instrument_permid": ["2"],
            "is_primary_ric": [True],
            "is_primary_issue_ric": [True],
        }
    )
    frame.to_csv(source / "security_master.csv", index=False)
    (source / "manifest.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")
    return source


def test_monthly_materializer_builds_bitemporal_candidate(tmp_path: Path) -> None:
    result = write_japan_g1_monthly_materializer(
        run_id="PRODUCER",
        monthly_qa_dir=_monthly_source(tmp_path),
        security_master_dirs=[_master(tmp_path)],
        artifact_root=tmp_path / "out",
    )
    assert result.manifest["status"] == "complete"
    assert result.manifest["historical_pit_ready"] is True
    assert result.manifest["counts"]["mapped_monthly_rows"] == 240
    reference = pd.read_csv(result.artifact_dir / "reference_symbol_map_candidate.csv")
    assert reference.loc[0, "right_open"]
    assert pd.isna(reference.loc[0, "window_end_known_at"])


def test_normalizer_preserves_disjoint_local_code_aliases_for_one_chained_ric() -> None:
    base = {
        "ric": "256A.T",
        "ticker_symbol": "256A",
        "exchange_code": "TYO",
        "rcs_asset_category_leaf": "Ordinary Share",
        "instrument_permid": "8590928464",
        "isin": "JP3629810007",
    }
    normalized = _normalize_security_master(
        pd.DataFrame([base, {**base, "mapping_local_code_override": "1805"}])
    )
    assert set(normalized["mapping_local_code"]) == {"1805", "256A"}


def test_normalizer_preserves_numeric_mapping_override_as_identifier_text() -> None:
    normalized = _normalize_security_master(
        pd.DataFrame(
            [
                {
                    "ric": "256A.T",
                    "ticker_symbol": "256A",
                    "exchange_code": "TYO",
                    "rcs_asset_category_leaf": "Ordinary Share",
                    "instrument_permid": "8590928464",
                    "mapping_local_code_override": 1805,
                }
            ]
        )
    )
    assert normalized.loc[0, "mapping_local_code"] == "1805"


def test_wrong_exchange_emits_targeted_missing_plan(tmp_path: Path) -> None:
    result = write_japan_g1_monthly_materializer(
        run_id="BLOCKED",
        monthly_qa_dir=_monthly_source(tmp_path),
        security_master_dirs=[_master(tmp_path, exchange="OSA")],
        artifact_root=tmp_path / "out",
    )
    assert result.manifest["status"] == "complete_with_blockers"
    assert result.manifest["counts"]["matched_segments"] == 0
    missing = pd.read_csv(result.missing_plan_path)
    assert missing.loc[0, "mapping_status"] == "unresolved_no_candidate"


def test_closed_window_prefers_candidate_retiring_near_window_end() -> None:
    segments = pd.DataFrame(
        {
            "stable_listing_id": ["JPX:7940:2006-07:1"],
            "local_code": ["7940"],
            "listing_start": ["2006-07-01"],
            "listing_end": ["2009-07-31"],
            "right_open": [False],
            "source_families": ["TSE"],
        }
    )
    master = pd.DataFrame(
        {
            "ric": ["7940.T^G09", "7940.T^G26"],
            "mapping_local_code": ["7940", "7940"],
            "exchange_code": ["TYO", "TYO"],
            "first_trade_date": pd.to_datetime([None, None]),
            "retire_date": pd.to_datetime(["2009-07-20", None]),
            "is_primary_ric": ["False", "False"],
            "is_primary_issue_ric": ["True", "True"],
            "isin": ["", "JP3687800007"],
            "sedol": ["", ""],
            "perm_id": ["", ""],
            "instrument_permid": ["8590011304", "21608946027"],
        }
    )

    reconciled, audit = _map_segments(segments, master)

    assert reconciled.loc[0, "mapping_status"] == "matched"
    assert reconciled.loc[0, "refinitiv_ric"] == "7940.T^G09"
    assert audit.set_index("candidate_ric").loc["7940.T^G09", "lifecycle_rank"] == 0
    assert audit.set_index("candidate_ric").loc["7940.T^G26", "lifecycle_rank"] == 1


def test_dual_venue_preference_and_exact_vendor_window_clipping() -> None:
    assert _preferred_exchanges("OSE_MAIN|TSE") == ["TYO", "OSA", "JSD"]
    assert _preferred_exchanges("JASDAQ|TSE") == ["TYO", "OSA", "JSD"]
    segments = pd.DataFrame(
        [{
            "stable_listing_id": "JPX:9797:2006-07:1",
            "local_code": "9797",
            "listing_start": "2006-07-01",
            "listing_end": "2021-07-31",
            "right_open": False,
            "source_families": "TSE",
        }]
    )
    master = pd.DataFrame(
        [{
            "ric": "7377.T",
            "mapping_local_code": "9797",
            "exchange_code": "TYO",
            "first_trade_date": pd.NaT,
            "retire_date": pd.Timestamp("2021-07-12"),
            "is_primary_ric": "True",
            "is_primary_issue_ric": "True",
            "isin": "",
            "sedol": "",
            "perm_id": "",
            "instrument_permid": "1",
        }]
    )
    reconciled, _ = _map_segments(segments, master)
    assert reconciled.loc[0, "listing_end"] == "2021-07-12"
    assert reconciled.loc[0, "monthly_observed_listing_end"] == "2021-07-31"


def test_exact_code_secondary_venue_beats_successor_alias_on_dual_venue_segment() -> None:
    segments = pd.DataFrame(
        [{
            "stable_listing_id": "JPX:5912:2006-07:1",
            "local_code": "5912",
            "listing_start": "2006-07-01",
            "listing_end": "2010-09-30",
            "right_open": False,
            "source_families": "OSE_MAIN|TSE",
        }]
    )
    master = pd.DataFrame(
        [
            {
                "ric": "5912.OS^G13",
                "mapping_local_code": "5912",
                "exchange_code": "OSA",
                "first_trade_date": pd.NaT,
                "retire_date": pd.Timestamp("2013-07-13"),
                "is_primary_ric": "True",
                "is_primary_issue_ric": "True",
                "isin": "",
                "sedol": "",
                "perm_id": "",
                "instrument_permid": "1",
            },
            {
                "ric": "1786.T",
                "mapping_local_code": "5912",
                "exchange_code": "TYO",
                "first_trade_date": pd.Timestamp("2013-07-14"),
                "retire_date": pd.Timestamp("2021-03-30"),
                "is_primary_ric": "True",
                "is_primary_issue_ric": "True",
                "isin": "",
                "sedol": "",
                "perm_id": "",
                "instrument_permid": "2",
            },
        ]
    )
    reconciled, _ = _map_segments(segments, master)
    assert reconciled.loc[0, "refinitiv_ric"] == "5912.OS^G13"


def test_targeted_shard_requires_completed_hash_verified_request(tmp_path: Path) -> None:
    shard = tmp_path / "SECMASTER-DISCOVERY-JPN-G1-TARGETED-MISSING-TEST-SHARD-01"
    shard.mkdir()
    security = shard / "security_master.csv"
    security.write_text("ric,ticker_symbol\n1001.T,1001\n", encoding="utf-8")
    manifest = {
        "run_id": shard.name,
        "requested_country_codes": ["JPN"],
        "extra_filter": "TickerSymbol eq '1001'",
        "artifacts": {"security_master": str(security.resolve())},
        "output_sha256": {"security_master": _sha(security)},
    }
    manifest_path = shard / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    payload = {
        "run_id": shard.name,
        "codes": ["1001"],
        "country_codes": ["JPN"],
        "state_buckets": ["active", "inactive"],
        "extra_filter": "TickerSymbol eq '1001'",
        "top": 1000,
        "max_pages": 1,
        "input_plan_sha256": "a" * 64,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    request = {
        **payload,
        "request_sha256": digest,
        "status": "complete",
        "manifest_sha256": _sha(manifest_path),
    }
    (shard / "targeted_request.json").write_text(json.dumps(request), encoding="utf-8")

    assert _targeted_shard_blockers(shard, manifest) == []

    security.write_text("tampered\n", encoding="utf-8")
    assert "targeted_security_master_hash_mismatch" in _targeted_shard_blockers(shard, manifest)
