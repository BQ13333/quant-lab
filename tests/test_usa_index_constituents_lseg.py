from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from quant_lab.data.usa_index_constituents_lseg import (
    LSEG_JOINER_LEAVER_FIELDS,
    LSEG_VERIFIED_2020_EVENT_COUNTS,
    build_lseg_usa_index_constituent_plan,
    execute_lseg_usa_index_constituents,
    lseg_constituents_as_of,
    reconstruct_lseg_constituent_intervals,
)
from quant_lab.data.usa_index_constituents_pit import (
    USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
    default_usa_index_catalog,
)


class _FakeLSEG:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get_data(self, universe, fields, parameters=None):
        self.calls.append({"universe": universe, "fields": fields, "parameters": parameters})
        if universe == "0#.SPX":
            return pd.DataFrame({"Instrument": ["A.N", "C.O"] , "Price Close": [10.0, 20.0]})
        if universe == ".SPX":
            return pd.DataFrame(
                {
                    "Constituent Change Date": ["2020-03-01", "2020-06-01"],
                    "Constituent RIC": ["B.N", "C.O"],
                    "Constituent Name": ["B", "C"],
                    "Constituent Change": ["Leaver", "Joiner"],
                }
            )
        raise AssertionError(f"unexpected live request for {universe}")


class _SameDayRoundTripFakeLSEG:
    def get_data(self, universe, fields, parameters=None):
        if universe == "0#.SPX":
            return pd.DataFrame({"Instrument": ["A.N"], "Price Close": [10.0]})
        if universe == ".SPX":
            return pd.DataFrame(
                {
                    "Constituent Change Date": ["2020-06-01", "2020-06-01"],
                    "Constituent RIC": ["B.N", "B.N"],
                    "Constituent Name": ["B", "B"],
                    "Constituent Change": ["Joiner", "Leaver"],
                }
            )
        raise AssertionError(f"unexpected live request for {universe}")


class _EmptyJoinerLeaverFakeLSEG:
    def get_data(self, universe, fields, parameters=None):
        if universe == "0#.SPX":
            return pd.DataFrame({"Instrument": ["A.N"], "Price Close": [10.0]})
        if universe == ".SPX":
            return pd.DataFrame(
                {
                    "Instrument": [".SPX"],
                    "Date": [pd.NaT],
                    "Constituent RIC": [""],
                    "Constituent Name": [""],
                    "Change": [""],
                }
            )
        raise AssertionError(f"unexpected live request for {universe}")


class _CompleteBacktestScopeFakeLSEG:
    def get_data(self, universe, fields, parameters=None):
        anchors = {
            "0#.SPX": "SPX_MEMBER.N",
            "0#.NDX": "NDX_MEMBER.O",
            "0#.DJI": "DJI_MEMBER.N",
        }
        if universe in anchors:
            return pd.DataFrame({"Instrument": [anchors[universe]], "Price Close": [10.0]})
        if universe in {".SPX", ".NDX", ".DJI"}:
            return pd.DataFrame(
                {
                    "Instrument": [universe],
                    "Date": [pd.NaT],
                    "Constituent RIC": [""],
                    "Constituent Name": [""],
                    "Change": [""],
                }
            )
        raise AssertionError(f"unexpected live request for {universe}")


class _DJIFakeLSEG:
    def __init__(
        self,
        anchor_members: list[str],
        events: pd.DataFrame,
        identity_map: dict[str, str] | None = None,
    ) -> None:
        self.anchor_members = anchor_members
        self.events = events
        self.identity_map = identity_map or {}
        self.calls: list[dict[str, object]] = []

    def get_data(self, universe, fields, parameters=None):
        self.calls.append({"universe": universe, "fields": fields, "parameters": parameters})
        if universe == "0#.DJI":
            return pd.DataFrame({"Instrument": self.anchor_members, "Price Close": [1.0] * len(self.anchor_members)})
        if universe == ".DJI":
            return self.events.copy()
        if isinstance(universe, list):
            resolved = [identifier for identifier in universe if identifier in self.identity_map]
            return pd.DataFrame(
                {
                    "Instrument": resolved,
                    "RIC": [self.identity_map[identifier] for identifier in resolved],
                    "Common Name": [f"Resolved {identifier}" for identifier in resolved],
                    "SEDOL": resolved,
                    "Primary RIC": [self.identity_map[identifier] for identifier in resolved],
                }
            )
        raise AssertionError(f"unexpected live request for {universe}")


def _spx_and_russell_catalog() -> pd.DataFrame:
    catalog = default_usa_index_catalog()
    return catalog[catalog["candidate_index_ric"].isin([".SPX", ".RUI"])].reset_index(drop=True)


def _dji_catalog() -> pd.DataFrame:
    catalog = default_usa_index_catalog()
    return catalog[catalog["candidate_index_ric"].eq(".DJI")].reset_index(drop=True)


def _dji_events(dates: list[str], members: list[str], changes: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Constituent Change Date": dates,
            "Constituent RIC": members,
            "Constituent Name": members,
            "Constituent Change": changes,
        }
    )


def test_plan_uses_current_chain_and_exact_official_jl_route():
    plan = build_lseg_usa_index_constituent_plan(
        "2020-01-01",
        "2020-12-31",
        index_catalog=_spx_and_russell_catalog(),
        anchor_date="2020-12-31",
    )
    spx = plan[plan["index_ric"].eq(".SPX")]
    snapshot = spx[spx["request_kind"].eq("current_snapshot")].iloc[0]
    jl = spx[spx["request_kind"].eq("joiner_leaver")].iloc[0]
    russell = plan[plan["index_ric"].eq(".RUI")]

    assert snapshot["universe"] == "0#.SPX"
    assert "(" not in snapshot["universe"]
    assert json.loads(jl["fields_json"]) == list(LSEG_JOINER_LEAVER_FIELDS)
    assert json.loads(jl["parameters_json"]) == {"EDATE": "2020-12-31", "IC": "B", "SDATE": "2020-01-01"}
    assert set(LSEG_VERIFIED_2020_EVENT_COUNTS.items()) == {(".SPX", 34), (".NDX", 25), (".DJI", 6)}
    assert not russell["live_request_allowed"].any()
    assert set(russell["request_status"]) == {"blocked_missing_entitlement"}


def test_backtest_no_russell_plan_selects_exactly_three_entitled_indexes():
    plan = build_lseg_usa_index_constituent_plan(
        "2020-01-01",
        "2020-12-31",
        anchor_date="2020-12-31",
        scope_profile=USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
    )

    assert plan["index_id"].drop_duplicates().tolist() == [
        "sp_500",
        "nasdaq_100",
        "dow_jones_industrial_average",
    ]
    assert plan["live_request_allowed"].all()
    assert not plan["index_family"].eq("Russell").any()


def test_cli_dry_run_selects_backtest_no_russell_scope(tmp_path):
    run_id = "CLI-BACKTEST-NO-RUSSELL"
    result = subprocess.run(
        [
            sys.executable,
            "examples/run_usa_lseg_index_constituents.py",
            "--dry-run",
            "--run-id",
            run_id,
            "--artifact-root",
            str(tmp_path),
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-12-31",
            "--scope-profile",
            "backtest-no-russell",
        ],
        cwd=Path(__file__).parents[1],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = pd.read_csv(tmp_path / run_id / "request_plan.csv")

    assert "Scope profile: backtest_no_russell" in result.stdout
    assert set(plan["index_id"]) == {"sp_500", "nasdaq_100", "dow_jones_industrial_average"}
    assert plan["live_request_allowed"].all()


def test_same_day_joiner_leaver_round_trip_is_neutralized():
    rows = pd.DataFrame(
        [
            {
                "index_id": "sp_500",
                "index_ric": ".SPX",
                "member_ric": "A.N",
                "member_name": "A",
                "membership_action": "snapshot_member",
                "membership_as_of_date": "2020-12-31",
                "effective_date": pd.NA,
                "known_at_utc": "2020-12-31T22:00:00Z",
                "source_record_id": "anchor-a",
                "source_request_id": "snapshot",
                "raw_payload_sha256": "a" * 64,
            },
            {
                "index_id": "sp_500",
                "index_ric": ".SPX",
                "member_ric": "B.N",
                "member_name": "B",
                "membership_action": "added",
                "membership_as_of_date": pd.NA,
                "effective_date": "2020-06-01",
                "known_at_utc": "2020-06-01T00:00:00Z",
                "source_record_id": "round-trip-add",
                "source_request_id": "jl",
                "raw_payload_sha256": "b" * 64,
            },
            {
                "index_id": "sp_500",
                "index_ric": ".SPX",
                "member_ric": "B.N",
                "member_name": "B",
                "membership_action": "removed",
                "membership_as_of_date": pd.NA,
                "effective_date": "2020-06-01",
                "known_at_utc": "2020-06-01T00:00:00Z",
                "source_record_id": "round-trip-remove",
                "source_request_id": "jl",
                "raw_payload_sha256": "c" * 64,
            },
        ]
    )

    result = reconstruct_lseg_constituent_intervals(rows, start_date="2020-01-01", end_date="2020-12-31")

    assert result.issues.empty
    assert set(result.intervals["member_ric"]) == {"A.N"}


def test_executor_completes_request_with_same_day_joiner_leaver_round_trip(tmp_path):
    catalog = default_usa_index_catalog()
    catalog = catalog[catalog["candidate_index_ric"].eq(".SPX")].reset_index(drop=True)

    artifacts = execute_lseg_usa_index_constituents(
        client=_SameDayRoundTripFakeLSEG(),
        artifact_root=tmp_path,
        run_id="LSEG-USA-INDEX-ROUND-TRIP-TEST",
        start_date="2020-01-01",
        end_date="2020-12-31",
        index_catalog=catalog,
        observed_at_utc="2020-12-31T22:00:00Z",
        sleep_seconds=0,
        max_requests=10,
    )

    joiner_leaver = artifacts.request_manifest[artifacts.request_manifest["request_kind"].eq("joiner_leaver")]
    assert joiner_leaver["status"].tolist() == ["completed"]
    assert artifacts.reconstruction.issues.empty
    assert set(artifacts.reconstruction.intervals["member_ric"]) == {"A.N"}


def test_executor_treats_blank_joiner_leaver_sentinel_as_completed_empty(tmp_path):
    catalog = default_usa_index_catalog()
    catalog = catalog[catalog["candidate_index_ric"].eq(".SPX")].reset_index(drop=True)

    artifacts = execute_lseg_usa_index_constituents(
        client=_EmptyJoinerLeaverFakeLSEG(),
        artifact_root=tmp_path,
        run_id="LSEG-USA-INDEX-EMPTY-JL-TEST",
        start_date="2020-01-01",
        end_date="2020-12-31",
        index_catalog=catalog,
        observed_at_utc="2020-12-31T22:00:00Z",
        sleep_seconds=0,
        max_requests=10,
    )

    joiner_leaver = artifacts.request_manifest[artifacts.request_manifest["request_kind"].eq("joiner_leaver")]
    assert joiner_leaver["status"].tolist() == ["completed_empty"]
    assert artifacts.reconstruction.issues.empty
    assert set(artifacts.reconstruction.intervals["member_ric"]) == {"A.N"}


def test_backtest_no_russell_manifest_is_complete_only_for_exact_clean_scope(tmp_path):
    artifacts = execute_lseg_usa_index_constituents(
        client=_CompleteBacktestScopeFakeLSEG(),
        artifact_root=tmp_path,
        run_id="LSEG-USA-INDEX-BACKTEST-NO-RUSSELL-COMPLETE",
        start_date="2020-01-01",
        end_date="2020-12-31",
        observed_at_utc="2020-12-31T22:00:00Z",
        sleep_seconds=0,
        max_requests=10,
        scope_profile=USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert manifest["scope_profile"] == "backtest_no_russell"
    assert manifest["required_index_ids"] == [
        "sp_500",
        "nasdaq_100",
        "dow_jones_industrial_average",
    ]
    assert manifest["excluded_index_ids"] == ["russell_1000", "russell_2000", "russell_3000"]
    assert "deliberately excluded" in manifest["excluded_index_reason"]
    assert manifest["scope_match"] is True
    assert manifest["completion_gates"] == {
        "all_required_indexes_complete": True,
        "catalog_exactly_matches_scope_profile": True,
        "zero_reconstruction_issues": True,
    }
    assert manifest["reconstruction_issue_count"] == 0
    assert manifest["status"] == "complete"
    assert manifest["scorecard_eligible"] is True
    assert set(artifacts.coverage["status"]) == {"complete"}


def test_backtest_no_russell_scope_mismatch_cannot_claim_completion(tmp_path):
    catalog = default_usa_index_catalog()
    catalog = catalog[catalog["index_id"].eq("sp_500")].reset_index(drop=True)
    artifacts = execute_lseg_usa_index_constituents(
        client=_CompleteBacktestScopeFakeLSEG(),
        artifact_root=tmp_path,
        run_id="LSEG-USA-INDEX-BACKTEST-NO-RUSSELL-PARTIAL",
        start_date="2020-01-01",
        end_date="2020-12-31",
        index_catalog=catalog,
        observed_at_utc="2020-12-31T22:00:00Z",
        sleep_seconds=0,
        max_requests=10,
        scope_profile=USA_INDEX_SCOPE_PROFILE_BACKTEST_NO_RUSSELL,
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))

    assert artifacts.coverage["status"].tolist() == ["complete"]
    assert manifest["scope_match"] is False
    assert manifest["incomplete_required_index_ids"] == ["nasdaq_100", "dow_jones_industrial_average"]
    assert manifest["completion_gates"]["all_required_indexes_complete"] is False
    assert manifest["status"] == "incomplete"
    assert manifest["scorecard_eligible"] is False
    assert "selected index catalog does not exactly match the required scope profile" in manifest["blockers"]


def test_reverse_reconstruction_persists_hashes_checkpoints_and_resumes(tmp_path):
    client = _FakeLSEG()
    kwargs = {
        "client": client,
        "artifact_root": tmp_path,
        "run_id": "LSEG-USA-INDEX-TEST",
        "start_date": "2020-01-01",
        "end_date": "2020-12-31",
        "index_catalog": _spx_and_russell_catalog(),
        "observed_at_utc": "2020-12-31T22:00:00Z",
        "sleep_seconds": 0,
    }

    partial = execute_lseg_usa_index_constituents(max_requests=1, **kwargs)
    assert len(client.calls) == 1
    assert client.calls[0]["universe"] == "0#.SPX"
    assert partial.coverage.set_index("index_id").loc["sp_500", "status"] == "incomplete_request_coverage"

    complete = execute_lseg_usa_index_constituents(max_requests=10, **kwargs)
    assert len(client.calls) == 2
    assert client.calls[1] == {
        "universe": ".SPX",
        "fields": list(LSEG_JOINER_LEAVER_FIELDS),
        "parameters": {"EDATE": "2020-12-31", "IC": "B", "SDATE": "2020-01-01"},
    }
    manifest = json.loads(complete.manifest_path.read_text(encoding="utf-8"))
    coverage = complete.coverage.set_index("index_id")
    intervals = complete.reconstruction.intervals

    assert manifest["status"] == "incomplete"
    assert manifest["scorecard_eligible"] is False
    assert manifest["dated_chain_historical_anchor_accepted"] is False
    assert manifest["missing_russell_coverage"] == ["russell_1000"]
    assert coverage.loc["sp_500", "status"] == "complete"
    assert coverage.loc["russell_1000", "status"] == "incomplete_missing_constituent_entitlement"
    assert complete.reconstruction.issues.empty
    assert set(intervals.loc[intervals["valid_from"].eq("2020-01-01"), "member_ric"]) == {"A.N", "B.N"}
    assert intervals.loc[intervals["member_ric"].eq("B.N"), "valid_to"].item() == "2020-02-29"
    assert intervals.loc[intervals["member_ric"].eq("C.O"), "valid_from"].item() == "2020-06-01"
    assert set(intervals["current_anchor_chain"]) == {"0#.SPX"}

    known_in_january = lseg_constituents_as_of(
        intervals,
        as_of_date="2020-01-15",
        known_at_utc="2020-01-15T23:59:59Z",
    )
    assert known_in_january.empty

    delayed_rows = complete.pit_rows.copy()
    delayed_rows.loc[
        delayed_rows["member_ric"].eq("C.O") & delayed_rows["membership_action"].eq("added"),
        "known_at_utc",
    ] = "2020-07-01T00:00:00Z"
    delayed = reconstruct_lseg_constituent_intervals(
        delayed_rows,
        start_date="2020-01-01",
        end_date="2020-12-31",
    )
    assert delayed.intervals.loc[delayed.intervals["member_ric"].eq("C.O"), "valid_from"].item() == "2020-07-01"

    completed_rows = complete.request_manifest[complete.request_manifest["status"].isin(["completed", "completed_empty"])]
    for row in completed_rows.itertuples(index=False):
        raw_path = Path(row.raw_response_path)
        assert raw_path.exists()
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == row.raw_response_sha256
    checkpoints = pd.read_csv(complete.checkpoint_manifest_path)
    assert set(checkpoints["checkpoint_date"]) == {"2020-03-01", "2020-06-01", "2020-12-31"}
    assert checkpoints["state_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()

    execute_lseg_usa_index_constituents(max_requests=10, **kwargs)
    assert len(client.calls) == 2

    snapshot_manifest = complete.request_manifest[
        complete.request_manifest["request_kind"].eq("current_snapshot")
        & complete.request_manifest["status"].eq("completed")
    ].iloc[0]
    Path(snapshot_manifest["raw_response_path"]).write_text("tampered", encoding="utf-8")
    execute_lseg_usa_index_constituents(max_requests=1, **kwargs)
    assert len(client.calls) == 3


def test_dji_identifier_resolution_maps_before_reconstruction_and_resumes(tmp_path):
    client = _DJIFakeLSEG(
        anchor_members=["WMT.N"],
        events=_dji_events(["2025-03-24"], ["2206657"], ["Joiner"]),
        identity_map={"2206657": "WMT.N"},
    )
    kwargs = {
        "client": client,
        "artifact_root": tmp_path,
        "run_id": "LSEG-DJI-IDENTITY-MAP-TEST",
        "start_date": "2024-01-01",
        "end_date": "2025-12-31",
        "index_catalog": _dji_catalog(),
        "chunk_years": 10,
        "observed_at_utc": "2025-12-31T22:00:00Z",
        "sleep_seconds": 0,
        "max_requests": 10,
    }

    artifacts = execute_lseg_usa_index_constituents(**kwargs)

    assert artifacts.reconstruction.issues.empty
    assert artifacts.identity_map[["raw_identifier", "resolved_ric", "resolution_status"]].to_dict("records") == [
        {"raw_identifier": "2206657", "resolved_ric": "WMT.N", "resolution_status": "resolved"}
    ]
    assert set(artifacts.pit_rows["member_ric"]) == {"WMT.N"}
    assert not artifacts.reconstruction.issues["issue_code"].eq("reverse_joiner_not_in_state").any()
    identity_call = next(call for call in client.calls if isinstance(call["universe"], list))
    assert identity_call == {
        "universe": ["2206657"],
        "fields": ["TR.RIC", "TR.CommonName", "TR.SEDOL", "TR.PrimaryRIC"],
        "parameters": None,
    }
    identity_request = artifacts.identity_resolution_manifest.iloc[0]
    raw_path = Path(identity_request["raw_response_path"])
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == identity_request["raw_response_sha256"]

    call_count = len(client.calls)
    execute_lseg_usa_index_constituents(**kwargs)
    assert len(client.calls) == call_count

    raw_path.write_text("tampered", encoding="utf-8")
    execute_lseg_usa_index_constituents(**kwargs)
    assert len(client.calls) == call_count + 1


def test_dji_snapshot_like_all_added_reset_is_strictly_quarantined(tmp_path):
    anchor_members = [f"DOW{number:02d}.N" for number in range(30)]
    raw_identifiers = [f"{2_200_000 + number:07d}" for number in range(29)]
    client = _DJIFakeLSEG(
        anchor_members=anchor_members,
        events=_dji_events(["2025-03-25"] * 29, raw_identifiers, ["Joiner"] * 29),
        identity_map=dict(zip(raw_identifiers, anchor_members[:29], strict=True)),
    )

    artifacts = execute_lseg_usa_index_constituents(
        client=client,
        artifact_root=tmp_path,
        run_id="LSEG-DJI-RESET-SENTINEL-TEST",
        start_date="2024-01-01",
        end_date="2026-12-31",
        index_catalog=_dji_catalog(),
        chunk_years=10,
        observed_at_utc="2026-12-31T22:00:00Z",
        sleep_seconds=0,
        max_requests=10,
    )

    assert artifacts.reconstruction.issues.empty
    assert len(artifacts.quarantined_rows) == 29
    assert set(artifacts.quarantined_rows["quarantine_reason"]) == {"snapshot_like_all_added_reset_sentinel"}
    assert set(artifacts.quarantined_rows["current_anchor_member_count"]) == {30}
    assert set(artifacts.quarantined_rows["anchor_overlap_count"]) == {29}
    assert set(artifacts.quarantined_rows["anchor_overlap_fraction"]) == {29 / 30}
    assert set(artifacts.reconstruction.intervals["member_ric"]) == set(anchor_members)
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["quarantined_joiner_leaver_rows"] == 29
    assert hashlib.sha256(artifacts.quarantine_path.read_bytes()).hexdigest() == manifest["quarantine_sha256"]


def test_dji_balanced_events_are_not_quarantined(tmp_path):
    client = _DJIFakeLSEG(
        anchor_members=["NEW.N", "STAY.N"],
        events=_dji_events(
            ["2024-11-08", "2024-11-08"],
            ["NEW.N", "OLD.N"],
            ["Joiner", "Leaver"],
        ),
    )

    artifacts = execute_lseg_usa_index_constituents(
        client=client,
        artifact_root=tmp_path,
        run_id="LSEG-DJI-BALANCED-EVENT-TEST",
        start_date="2024-01-01",
        end_date="2025-12-31",
        index_catalog=_dji_catalog(),
        chunk_years=10,
        observed_at_utc="2025-12-31T22:00:00Z",
        sleep_seconds=0,
        max_requests=10,
    )

    assert artifacts.quarantined_rows.empty
    assert artifacts.reconstruction.issues.empty
    intervals = artifacts.reconstruction.intervals.set_index("member_ric")
    assert intervals.loc["OLD.N", "valid_to"] == "2024-11-07"
    assert intervals.loc["NEW.N", "valid_from"] == "2024-11-08"
    assert artifacts.pit_rows["membership_action"].isin(["added", "removed"]).sum() == 2


def test_dji_unresolved_identifier_still_blocks_reconstruction(tmp_path):
    client = _DJIFakeLSEG(
        anchor_members=["WMT.N"],
        events=_dji_events(["2025-03-24"], ["BYVY8G0"], ["Joiner"]),
    )

    artifacts = execute_lseg_usa_index_constituents(
        client=client,
        artifact_root=tmp_path,
        run_id="LSEG-DJI-UNRESOLVED-IDENTITY-TEST",
        start_date="2024-01-01",
        end_date="2025-12-31",
        index_catalog=_dji_catalog(),
        chunk_years=10,
        observed_at_utc="2025-12-31T22:00:00Z",
        sleep_seconds=0,
        max_requests=10,
    )

    assert artifacts.identity_map.loc[0, "resolution_status"] == "unresolved_no_matching_response"
    assert "unresolved_member_identifier" in set(artifacts.reconstruction.issues["issue_code"])
    assert "reverse_joiner_not_in_state" in set(artifacts.reconstruction.issues["issue_code"])
    assert (
        artifacts.coverage.set_index("index_id").loc["dow_jones_industrial_average", "status"]
        == "incomplete_reconstruction_validation"
    )
    manifest = json.loads(artifacts.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    assert "joiner/leaver reconstruction validation failed" in manifest["blockers"]
