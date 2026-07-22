from __future__ import annotations

import json
from collections import deque
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from quant_lab.data.jquants_listed_shares import (
    JQuantsHttpResponse,
    JQuantsProV2Client,
    JQuantsRequestError,
    build_jquants_listed_shares_plan,
    known_at_from_updated_at,
    normalize_jquants_shares,
    normalize_jquants_shares_change,
    run_jquants_listed_shares_backfill,
)


UTC = timezone.utc
FIXED_NOW = datetime(2026, 7, 12, 1, 2, 3, tzinfo=UTC)


def _response(status: int, body: object, headers: dict[str, str] | None = None) -> JQuantsHttpResponse:
    return JQuantsHttpResponse(
        status_code=status,
        content=json.dumps(body, separators=(",", ":")).encode(),
        headers=headers or {},
    )


class QueueRequester:
    def __init__(self, *responses: JQuantsHttpResponse):
        self.responses = deque(responses)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("unexpected HTTP call")
        return self.responses.popleft()


def _run(tmp_path, requester, **overrides):
    kwargs = {
        "artifact_root": tmp_path,
        "run_id": "JQ-TEST",
        "start_date": "2024-01-05",
        "end_date": "2024-01-05",
        "max_dates": 1,
        "max_codes": 1,
        "requester": requester,
        "environ": {"JQUANTS_ID_TOKEN": "top-secret-id-token"},
        "clock": lambda: FIXED_NOW,
        "sleeper": lambda _seconds: None,
        "pacing_seconds": 0,
        "base_retry_seconds": 0,
        "max_retry_seconds": 0,
    }
    kwargs.update(overrides)
    return run_jquants_listed_shares_backfill(**kwargs)


def test_plan_is_deterministic_daily_and_crosses_explicit_codes():
    first = build_jquants_listed_shares_plan(
        "2008-05-07", "2008-05-08", codes=["7203", "13010", "7203"]
    )
    second = build_jquants_listed_shares_plan(
        "2008-05-07", "2008-05-08", codes=["13010", "7203"]
    )

    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 8
    assert set(first["endpoint"]) == {"shares", "shares_change"}
    assert set(first["code"]) == {"7203", "13010"}
    assert all("updated_at" in json.loads(value) for value in first["conditions_json"])
    with pytest.raises(ValueError, match="cannot precede"):
        build_jquants_listed_shares_plan("2008-05-06", "2008-05-07")

    weekend = build_jquants_listed_shares_plan("2008-05-09", "2008-05-12")
    assert set(weekend["updated_at"]) == {"2008-05-09", "2008-05-12"}


def test_client_preserves_every_original_condition_across_pagination():
    requester = QueueRequester(
        _response(200, {"shares": [{"Code": "72030"}], "pagination_key": "next-1"}),
        _response(200, {"shares": []}),
    )
    client = JQuantsProV2Client(
        "secret",
        requester=requester,
        sleeper=lambda _seconds: None,
        clock=lambda: FIXED_NOW,
        retries=0,
        pacing_seconds=0,
    )
    conditions = {"code": "7203", "date": "2024-01-05", "updated_at": "2024-01-06"}

    page1 = client.fetch_page(
        url="https://api.jquants-pro.com/v2/listed/shares",
        response_root="shares",
        conditions=conditions,
    )
    client.fetch_page(
        url="https://api.jquants-pro.com/v2/listed/shares",
        response_root="shares",
        conditions=conditions,
        pagination_key=page1.next_pagination_key,
    )

    assert conditions == {"code": "7203", "date": "2024-01-05", "updated_at": "2024-01-06"}
    assert requester.calls[0]["params"] == conditions
    assert requester.calls[1]["params"] == {**conditions, "pagination_key": "next-1"}


def test_known_at_uses_next_valid_xtks_day_at_0900_jst():
    # 2024-01-08 was Coming of Age Day, so Friday's update is known Tuesday.
    known = known_at_from_updated_at("2024-01-05")

    assert known is not None
    assert known.isoformat() == "2024-01-09T09:00:00+09:00"
    assert known.astimezone(UTC).isoformat() == "2024-01-09T00:00:00+00:00"
    assert known_at_from_updated_at("9999-99-99") is None


def test_normalizers_include_every_official_field_and_fail_closed_updated_at():
    shares = normalize_jquants_shares(
        [
            {
                "Date": "2024-02-01",
                "EffectiveDate": "2024-02-01",
                "EffectiveEndDate": "9999-99-99",
                "Code": "72030",
                "OutstandingShares": 100,
                "ListedShares": 90,
                "IndexShares": 80,
                "UpdatedAt": None,
            }
        ],
        observed_at_utc=FIXED_NOW,
        response_sha256="abc",
    )
    changes = normalize_jquants_shares_change(
        [
            {
                "EffectiveDate": "2027-01-01",
                "SequentialNumber": 2,
                "Code": "72030",
                "FactorType": "101",
                "CABaseDate": "2026-12-30",
                "CASubstantialBaseDate": "9999-99-99",
                "CAEffectiveDate": "2027-01-01",
                "Changes": 10,
                "Denominator": 1,
                "Numerator": 2,
                "UpdatedAt": "2024-01-05",
            }
        ],
        observed_at_utc=FIXED_NOW,
        response_sha256="def",
    )

    assert shares.loc[0, "effective_end_date"] == "9999-99-99"
    assert not shares.loc[0, "pit_usable"]
    assert shares.loc[0, "unusable_reason"] == "updated_at_unresolved"
    assert changes.loc[0, "effective_date"] == "2027-01-01"
    assert changes.loc[0, "known_at_jst"] == "2024-01-09T09:00:00+09:00"
    assert changes.loc[0, "pit_usable"]


def test_missing_credentials_publishes_blocked_plan_without_any_call(tmp_path):
    requester = QueueRequester()
    artifacts = _run(tmp_path, requester, environ={})

    assert requester.calls == []
    assert artifacts.manifest["status"] == "blocked_credentials"
    assert artifacts.manifest["promotion"] is False
    assert artifacts.plan_path.is_file()
    assert artifacts.share_ledger.empty
    assert artifacts.share_change_ledger.empty
    assert artifacts.manifest["claims"] == {
        "comprehensive_dividends": False,
        "comprehensive_corporate_actions": False,
        "g2_complete": False,
    }


def test_backfill_retains_raw_pages_hashes_future_events_and_revisions(tmp_path):
    requester = QueueRequester(
        _response(
            200,
            {
                "shares": [
                    {
                        "Date": "2024-02-01",
                        "Code": "72030",
                        "OutstandingShares": 100,
                        "ListedShares": 90,
                        "IndexShares": 80,
                        "UpdatedAt": "2024-01-05",
                    }
                ],
                "pagination_key": "page-2",
            },
        ),
        _response(
            200,
            {
                "shares": [
                    {
                        "Date": "2024-02-01",
                        "Code": "72030",
                        "OutstandingShares": 101,
                        "ListedShares": 91,
                        "IndexShares": 81,
                        "UpdatedAt": "2024-01-05",
                    }
                ]
            },
        ),
        _response(
            200,
            {
                "shares_change": [
                    {
                        "EffectiveDate": "2027-01-01",
                        "SequentialNumber": 1,
                        "Code": "72030",
                        "FactorType": "101",
                        "CABaseDate": "9999-99-99",
                        "CASubstantialBaseDate": "9999-99-99",
                        "CAEffectiveDate": "2027-01-01",
                        "Changes": 1,
                        "Denominator": None,
                        "Numerator": None,
                        "UpdatedAt": "2024-01-05",
                    }
                ]
            },
        ),
    )

    artifacts = _run(tmp_path, requester)

    assert artifacts.manifest["status"] == "complete"
    assert len(artifacts.page_manifest) == 3
    assert len(artifacts.share_ledger) == 2  # distinct corrections are not collapsed
    assert len(artifacts.share_change_ledger) == 1
    assert artifacts.share_change_ledger.loc[0, "effective_date"] == "2027-01-01"
    assert artifacts.share_change_ledger.loc[0, "known_at_jst"] == "2024-01-09T09:00:00+09:00"
    assert requester.calls[1]["params"] == {
        "updated_at": "2024-01-05",
        "pagination_key": "page-2",
    }
    for row in artifacts.page_manifest.to_dict("records"):
        assert len(row["response_sha256"]) == 64
    all_artifacts = b"".join(path.read_bytes() for path in artifacts.artifact_dir.rglob("*") if path.is_file())
    assert b"top-secret-id-token" not in all_artifacts


def test_page_bound_checkpoints_and_resumes_without_repeating_completed_route(tmp_path):
    first_requester = QueueRequester(
        _response(200, {"shares": [], "pagination_key": "resume-key"}),
        _response(200, {"shares_change": []}),
    )
    first = _run(tmp_path, first_requester, max_pages_per_plan_row=1)
    assert set(first.checkpoints["status"]) == {"in_progress", "completed_empty"}

    second_requester = QueueRequester(_response(200, {"shares": [{"Code": "72030", "UpdatedAt": "2024-01-05"}]}))
    second = _run(tmp_path, second_requester, max_pages_per_plan_row=1)

    assert second.manifest["status"] == "complete"
    assert len(second_requester.calls) == 1
    assert second_requester.calls[0]["params"] == {
        "updated_at": "2024-01-05",
        "pagination_key": "resume-key",
    }
    assert len(second.page_manifest) == 3


def test_429_and_5xx_retries_are_bounded():
    sleeps: list[float] = []
    requester = QueueRequester(
        _response(429, {"message": "limited"}, {"Retry-After": "7"}),
        _response(503, {"message": "later"}),
        _response(200, {"shares": []}),
    )
    client = JQuantsProV2Client(
        "secret",
        requester=requester,
        sleeper=sleeps.append,
        clock=lambda: FIXED_NOW,
        retries=2,
        pacing_seconds=0,
        base_retry_seconds=2,
        max_retry_seconds=10,
    )

    page = client.fetch_page(
        url="https://api.jquants-pro.com/v2/listed/shares",
        response_root="shares",
        conditions={"updated_at": "2024-01-05"},
    )

    assert page.attempts == 3
    assert sleeps == [7.0, 4]
    assert len(requester.calls) == 3


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [(401, "blocked_credentials"), (403, "blocked_entitlement"), (413, "blocked_narrowing_required")],
)
def test_terminal_http_states_stop_the_invocation(tmp_path, status_code, expected):
    requester = QueueRequester(_response(status_code, {"message": "contains no useful detail"}))
    artifacts = _run(tmp_path, requester, retries=9)

    assert artifacts.manifest["status"] == expected
    assert len(requester.calls) == 1
    assert artifacts.checkpoints.loc[0, "status"] == expected
    assert artifacts.checkpoints.loc[0, "attempts"] == 1


def test_refresh_token_is_exchanged_in_memory_and_never_persisted(tmp_path):
    requester = QueueRequester(
        _response(200, {"idToken": "ephemeral-id-token"}),
        _response(200, {"shares": []}),
        _response(200, {"shares_change": []}),
    )
    artifacts = _run(
        tmp_path,
        requester,
        environ={"JQUANTS_REFRESH_TOKEN": "top-secret-refresh-token"},
    )

    assert requester.calls[0]["json_body"] == {"refreshtoken": "top-secret-refresh-token"}
    assert requester.calls[1]["headers"]["Authorization"] == "Bearer ephemeral-id-token"
    assert artifacts.manifest["auth_source"] == "JQUANTS_REFRESH_TOKEN"
    persisted = b"".join(path.read_bytes() for path in artifacts.artifact_dir.rglob("*") if path.is_file())
    assert b"top-secret-refresh-token" not in persisted
    assert b"ephemeral-id-token" not in persisted


def test_existing_plan_cannot_be_mutated_on_resume(tmp_path):
    first = QueueRequester(_response(200, {"shares": []}), _response(200, {"shares_change": []}))
    _run(tmp_path, first)

    with pytest.raises(ValueError, match="immutable request plan"):
        _run(tmp_path, QueueRequester(), codes=["7203"])


def test_client_does_not_include_token_in_classified_exception():
    client = JQuantsProV2Client(
        "highly-secret",
        requester=QueueRequester(_response(401, {"message": "highly-secret"})),
        retries=0,
        pacing_seconds=0,
    )

    with pytest.raises(JQuantsRequestError) as captured:
        client.fetch_page(
            url="https://api.jquants-pro.com/v2/listed/shares",
            response_root="shares",
            conditions={"code": "7203"},
        )
    assert "highly-secret" not in str(captured.value)
