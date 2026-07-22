from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from quant_lab.data.jquants_listed_info import (
    API_VERSION,
    AUTH_REFRESH_ENDPOINT,
    HISTORY_START_DATE,
    LISTED_INFO_ENDPOINT,
    NORMALIZED_COLUMNS,
    HttpResponse,
    _run_lock,
    availability_lower_bound_utc,
    build_listed_info_date_plan,
    extract_jquants_listed_info,
    normalize_listed_info_response,
)


FIXED_CLOCK = lambda: datetime(2024, 2, 12, 10, 0, tzinfo=timezone.utc)


def _listed_payload(day: str, code: str = "86970") -> bytes:
    return json.dumps(
        {
            "info": [
                {
                    "Date": day,
                    "Code": code,
                    "CompanyName": "Japan Exchange Group",
                    "CompanyNameEnglish": "Japan Exchange Group,Inc.",
                    "Sector17Code": "16",
                    "Sector17CodeName": "Finance",
                    "Sector33Code": "7200",
                    "Sector33CodeName": "Other Financing Business",
                    "ScaleCategory": "TOPIX Large70",
                    "MarketCode": "0111",
                    "MarketCodeName": "Prime",
                    "MarginCode": "2",
                    "MarginCodeName": "Loan",
                    "BasePrice": 1234.0,
                }
            ]
        },
        ensure_ascii=False,
    ).encode("utf-8")


def test_date_plan_is_inclusive_from_official_history_start_and_skips_weekends():
    plan = build_listed_info_date_plan("2008-05-11")

    assert HISTORY_START_DATE.isoformat() == "2008-05-07"
    assert plan["request_date"].tolist() == [
        "2008-05-07",
        "2008-05-08",
        "2008-05-09",
    ]
    assert plan["endpoint"].eq(LISTED_INFO_ENDPOINT).all()
    assert plan["api_version"].eq(API_VERSION).all()

    with pytest.raises(ValueError, match="history start"):
        build_listed_info_date_plan("2008-05-06")


def test_normalization_preserves_source_dimensions_and_never_infers_known_at():
    payload = _listed_payload("2024-02-09")
    digest = hashlib.sha256(payload).hexdigest()

    normalized = normalize_listed_info_response(
        payload,
        request_date="2024-02-09",
        observed_at_utc="2024-02-12T10:00:00Z",
        raw_content_sha256=digest,
    )

    assert tuple(normalized.columns) == NORMALIZED_COLUMNS
    row = normalized.iloc[0]
    assert row["effective_date"] == "2024-02-09"
    assert row["code"] == "86970"
    assert row["company_name"] == "Japan Exchange Group"
    assert row["sector_17_code"] == "16"
    assert row["sector_33_code"] == "7200"
    assert row["market_code"] == "0111"
    assert row["margin_code"] == "2"
    assert row["base_price"] == 1234.0
    assert pd.isna(row["known_at_utc"])
    assert pd.isna(row["known_at_source_field"])
    assert pd.isna(row["availability_lower_bound_utc"])
    assert row["observed_at_utc"] == "2024-02-12T10:00:00.000000+00:00"

    sourced = normalize_listed_info_response(
        payload,
        request_date="2024-02-09",
        observed_at_utc="2024-02-12T10:00:00Z",
        raw_content_sha256=digest,
        source_publication_date_jst="2024-02-08",
    )
    assert pd.isna(sourced.iloc[0]["known_at_utc"])
    assert (
        sourced.iloc[0]["availability_lower_bound_utc"]
        == "2024-02-08T08:30:00.000000+00:00"
    )
    assert availability_lower_bound_utc("2024-02-08").endswith("08:30:00.000000+00:00")


def test_normalization_accepts_holiday_next_business_date_but_rejects_stale_date():
    next_business_payload = _listed_payload("2024-02-12")
    normalized = normalize_listed_info_response(
        next_business_payload,
        request_date="2024-02-10",
        observed_at_utc="2024-02-12T10:00:00Z",
        raw_content_sha256=hashlib.sha256(next_business_payload).hexdigest(),
    )
    assert normalized.iloc[0]["effective_date"] == "2024-02-12"

    stale_payload = _listed_payload("2024-02-09")
    with pytest.raises(ValueError, match="precedes request date"):
        normalize_listed_info_response(
            stale_payload,
            request_date="2024-02-10",
            observed_at_utc="2024-02-12T10:00:00Z",
            raw_content_sha256=hashlib.sha256(stale_payload).hexdigest(),
        )


def test_absent_credentials_write_blocked_plan_manifest_and_make_no_call(tmp_path: Path):
    calls = 0

    def forbidden_transport(*_args) -> HttpResponse:
        nonlocal calls
        calls += 1
        raise AssertionError("HTTP must not run without credentials")

    artifacts = extract_jquants_listed_info(
        artifact_root=tmp_path,
        run_id="BLOCKED-CREDENTIALS",
        end_date="2008-05-09",
        max_dates=2,
        environment={},
        transport=forbidden_transport,
        rate_limit_per_minute=None,
        clock=FIXED_CLOCK,
    )

    assert calls == 0
    assert artifacts.manifest["status"] == "blocked_credentials"
    assert artifacts.request_plan["request_date"].tolist() == [
        "2008-05-07",
        "2008-05-08",
        "2008-05-09",
    ]
    assert artifacts.request_manifest.empty
    assert artifacts.listed_info.empty
    assert artifacts.manifest["credential_policy"]["tokens_persisted"] is False
    assert all(
        value is False for value in artifacts.manifest["promotion_flags"].values()
    )
    assert artifacts.checksums_path.is_file()
    assert artifacts.quality_metrics["planned_dates"] == 3
    assert artifacts.quality_metrics["pending_dates"] == 3


def test_refresh_token_bounded_resume_raw_provenance_checksums_and_no_token_leakage(
    tmp_path: Path,
):
    refresh_token = "fixture-refresh-token-never-persist"
    id_token = "fixture-id-token-never-persist"
    requested_dates: list[str] = []
    auth_calls = 0

    def transport(
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        _timeout: float,
    ) -> HttpResponse:
        nonlocal auth_calls
        if url == AUTH_REFRESH_ENDPOINT:
            auth_calls += 1
            assert method == "POST"
            assert json.loads(body or b"{}") == {"refreshtoken": refresh_token}
            return HttpResponse(
                200,
                json.dumps({"idToken": id_token}).encode("utf-8"),
                {},
            )
        assert method == "GET"
        assert headers["Authorization"] == f"Bearer {id_token}"
        request_date = parse_qs(urlparse(url).query)["date"][0]
        requested_dates.append(request_date)
        return HttpResponse(200, _listed_payload(request_date, code="086970"), {})

    kwargs = {
        "artifact_root": tmp_path,
        "run_id": "BOUNDED-RESUME",
        "end_date": "2008-05-08",
        "max_dates": 1,
        "environment": {"JQUANTS_REFRESH_TOKEN": refresh_token},
        "transport": transport,
        "rate_limit_per_minute": None,
        "clock": FIXED_CLOCK,
    }
    first = extract_jquants_listed_info(**kwargs)

    assert first.manifest["status"] == "incomplete"
    assert first.manifest["request_policy"]["attempted_dates_this_invocation"] == 1
    assert requested_dates == ["2008-05-07"]
    assert first.listed_info["code"].tolist() == ["086970"]
    assert first.listed_info["known_at_utc"].isna().all()

    second = extract_jquants_listed_info(**kwargs)

    assert auth_calls == 2
    assert requested_dates == ["2008-05-07", "2008-05-08"]
    assert second.manifest["status"] == "complete"
    assert second.request_manifest["status"].eq("completed").all()
    assert len(second.listed_info) == 2
    assert second.quality_metrics["date_coverage_ratio"] == 1.0
    assert second.quality_metrics["known_at_non_null_rows"] == 0

    first_request = second.request_manifest.iloc[0]
    raw_path = Path(first_request["raw_response_path"])
    metadata_path = Path(first_request["raw_metadata_path"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["request_date"] == "2008-05-07"
    assert metadata["endpoint"] == LISTED_INFO_ENDPOINT
    assert metadata["api_version"] == API_VERSION
    assert metadata["http_status"] == 200
    assert metadata["observed_at_utc"] == "2024-02-12T10:00:00.000000+00:00"
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == metadata["content_sha256"]

    checksum_lines = second.checksums_path.read_text(encoding="utf-8").splitlines()
    assert any("request_plan.csv" in line for line in checksum_lines)
    assert any("raw/2008-05-07.response.json" in line for line in checksum_lines)
    for path in second.artifact_dir.rglob("*"):
        if path.is_file():
            payload = path.read_bytes()
            assert refresh_token.encode("utf-8") not in payload
            assert id_token.encode("utf-8") not in payload


def test_429_and_5xx_use_bounded_exponential_retry_and_checkpoint_attempts(
    tmp_path: Path,
):
    responses = [
        HttpResponse(429, b'{"message":"slow down"}', {"Retry-After": "3"}),
        HttpResponse(503, b'{"message":"unavailable"}', {}),
        HttpResponse(200, _listed_payload("2008-05-07"), {}),
    ]
    sleeps: list[float] = []

    def transport(*_args) -> HttpResponse:
        return responses.pop(0)

    artifacts = extract_jquants_listed_info(
        artifact_root=tmp_path,
        run_id="RETRY",
        end_date="2008-05-07",
        max_dates=1,
        retries=2,
        backoff_seconds=1.0,
        max_backoff_seconds=10.0,
        environment={"JQUANTS_ID_TOKEN": "id-token"},
        transport=transport,
        sleeper=sleeps.append,
        rate_limit_per_minute=None,
        clock=FIXED_CLOCK,
    )

    assert sleeps == [3.0, 2.0]
    assert responses == []
    assert artifacts.request_manifest.iloc[0]["attempts"] == 3
    assert artifacts.request_manifest.iloc[0]["status"] == "completed"
    assert artifacts.manifest["status"] == "complete"


def test_401_stops_without_requesting_later_dates_and_retains_safe_error_raw(
    tmp_path: Path,
):
    token = "authorization-token"
    calls = 0

    def transport(*_args) -> HttpResponse:
        nonlocal calls
        calls += 1
        return HttpResponse(
            401,
            json.dumps({"message": f"invalid {token}"}).encode("utf-8"),
            {},
        )

    artifacts = extract_jquants_listed_info(
        artifact_root=tmp_path,
        run_id="AUTH-STOP",
        end_date="2008-05-09",
        max_dates=3,
        retries=4,
        environment={"JQUANTS_ID_TOKEN": token},
        transport=transport,
        rate_limit_per_minute=None,
        clock=FIXED_CLOCK,
    )

    assert calls == 1
    assert artifacts.manifest["status"] == "blocked_authorization"
    assert artifacts.request_manifest["request_date"].tolist() == ["2008-05-07"]
    assert artifacts.request_manifest.iloc[0]["status"] == "blocked_authorization"
    assert artifacts.request_manifest.iloc[0]["attempts"] == 1
    raw_path = Path(artifacts.request_manifest.iloc[0]["raw_response_path"])
    assert b"[REDACTED]" in raw_path.read_bytes()
    assert token.encode("utf-8") not in raw_path.read_bytes()


def test_corrupt_success_raw_is_retried_on_resume(tmp_path: Path):
    calls = 0

    def transport(
        _method: str,
        url: str,
        _headers: dict[str, str],
        _body: bytes | None,
        _timeout: float,
    ) -> HttpResponse:
        nonlocal calls
        calls += 1
        request_date = parse_qs(urlparse(url).query)["date"][0]
        return HttpResponse(200, _listed_payload(request_date), {})

    kwargs = {
        "artifact_root": tmp_path,
        "run_id": "RAW-RECOVERY",
        "end_date": "2008-05-07",
        "max_dates": 1,
        "environment": {"JQUANTS_ID_TOKEN": "id-token"},
        "transport": transport,
        "rate_limit_per_minute": None,
        "clock": FIXED_CLOCK,
    }
    first = extract_jquants_listed_info(**kwargs)
    Path(first.request_manifest.iloc[0]["raw_response_path"]).write_bytes(b"{}")

    resumed = extract_jquants_listed_info(**kwargs)

    assert calls == 2
    assert resumed.request_manifest.iloc[0]["status"] == "completed"
    assert resumed.request_manifest.iloc[0]["attempts"] == 2
    assert len(resumed.listed_info) == 1


def test_plan_is_immutable_and_single_writer_lock_is_nonblocking(tmp_path: Path):
    kwargs = {
        "artifact_root": tmp_path,
        "run_id": "IMMUTABLE",
        "end_date": "2008-05-07",
        "max_dates": 1,
        "environment": {},
        "rate_limit_per_minute": None,
        "clock": FIXED_CLOCK,
    }
    extract_jquants_listed_info(**kwargs)
    with pytest.raises(ValueError, match="immutable date plan"):
        extract_jquants_listed_info(**{**kwargs, "end_date": "2008-05-08"})

    artifact_dir = tmp_path / "LOCKED"
    artifact_dir.mkdir()
    with _run_lock(artifact_dir):
        with pytest.raises(RuntimeError, match="another extractor"):
            with _run_lock(artifact_dir):
                pytest.fail("second writer unexpectedly acquired lock")


def test_written_normalized_csv_has_stable_schema_for_empty_and_nonempty_runs(
    tmp_path: Path,
):
    blocked = extract_jquants_listed_info(
        artifact_root=tmp_path,
        run_id="EMPTY-SCHEMA",
        end_date="2008-05-07",
        max_dates=1,
        environment={},
        rate_limit_per_minute=None,
        clock=FIXED_CLOCK,
    )
    empty = pd.read_csv(blocked.normalized_path)
    assert empty.empty
    assert tuple(empty.columns) == NORMALIZED_COLUMNS
