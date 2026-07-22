from __future__ import annotations

import fcntl
import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd


DATASET_ID = "jquants_pro_v2_listed_info"
API_VERSION = "v2"
HISTORY_START_DATE = date(2008, 5, 7)
LISTED_INFO_ENDPOINT = "https://api.jquants-pro.com/v2/listed/info"
AUTH_REFRESH_ENDPOINT = "https://api.jquants-pro.com/v2/token/auth_refresh"
LISTED_INFO_DOCUMENTATION = "https://jpx.gitbook.io/j-quants-pro/api-reference/listed_info"
AUTH_REFRESH_DOCUMENTATION = "https://jpx.gitbook.io/j-quants-pro/api-reference/idtoken"
JST = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
SUCCESS_STATUSES = frozenset({"completed", "completed_empty"})

PLAN_COLUMNS = ("request_date", "endpoint", "api_version")
REQUEST_MANIFEST_COLUMNS = (
    "request_date",
    "request_url",
    "status",
    "attempts",
    "http_status",
    "observed_at_utc",
    "completed_at_utc",
    "source_rows",
    "raw_response_path",
    "raw_metadata_path",
    "content_sha256",
    "response_sanitized",
    "error_type",
    "error",
)
NORMALIZED_COLUMNS = (
    "effective_date",
    "code",
    "company_name",
    "company_name_english",
    "sector_17_code",
    "sector_17_name",
    "sector_33_code",
    "sector_33_name",
    "scale_category",
    "market_code",
    "market_name",
    "margin_code",
    "margin_name",
    "base_price",
    "request_date",
    "known_at_utc",
    "known_at_source_field",
    "availability_lower_bound_utc",
    "availability_basis",
    "observed_at_utc",
    "source_endpoint",
    "source_api_version",
    "raw_content_sha256",
)


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


HttpTransport = Callable[
    [str, str, Mapping[str, str], bytes | None, float], HttpResponse
]
Sleeper = Callable[[float], None]
Clock = Callable[[], datetime]


@dataclass(frozen=True)
class JQuantsListedInfoArtifacts:
    run_id: str
    artifact_dir: Path
    request_plan_path: Path
    request_manifest_path: Path
    raw_dir: Path
    normalized_path: Path
    quality_metrics_path: Path
    checksums_path: Path
    manifest_path: Path
    request_plan: pd.DataFrame
    request_manifest: pd.DataFrame
    listed_info: pd.DataFrame
    quality_metrics: dict[str, object]
    manifest: dict[str, object]


@dataclass(frozen=True)
class _RequestOutcome:
    response: HttpResponse | None
    attempts: int
    observed_at_utc: str
    error: Exception | None


def build_listed_info_date_plan(end_date: str | date) -> pd.DataFrame:
    """Build the immutable inclusive weekday plan for J-Quants history.

    JPX is closed on Saturdays and Sundays, and the official endpoint maps
    non-business dates to the next business day's snapshot. Excluding weekends
    removes guaranteed duplicate requests without relying on a third-party holiday
    calendar. Japanese weekday holidays remain in the plan and are reconciled from
    the effective date returned by the official endpoint.
    """

    end = _parse_iso_date(end_date, "end_date")
    if end < HISTORY_START_DATE:
        raise ValueError(
            f"end_date cannot be before J-Quants history start {HISTORY_START_DATE.isoformat()}"
        )
    rows: list[dict[str, str]] = []
    current = HISTORY_START_DATE
    while current <= end:
        if current.weekday() < 5:
            rows.append(
                {
                    "request_date": current.isoformat(),
                    "endpoint": LISTED_INFO_ENDPOINT,
                    "api_version": API_VERSION,
                }
            )
        current += timedelta(days=1)
    return pd.DataFrame(rows, columns=PLAN_COLUMNS)


def availability_lower_bound_utc(
    source_publication_date_jst: str | date,
) -> str:
    """Return the documented 17:30 JST not-before bound for a sourced publish day.

    The publication date must come from independent source evidence. It must not
    be inferred from the listed-info ``Date`` field.
    """

    publication_date = _parse_iso_date(
        source_publication_date_jst, "source_publication_date_jst"
    )
    local = datetime.combine(
        publication_date, datetime_time(hour=17, minute=30), tzinfo=JST
    )
    return local.astimezone(UTC).isoformat(timespec="microseconds")


def normalize_listed_info_response(
    payload: bytes | Mapping[str, object],
    *,
    request_date: str | date,
    observed_at_utc: str | datetime,
    raw_content_sha256: str,
    source_publication_date_jst: str | date | None = None,
) -> pd.DataFrame:
    """Validate and normalize one official listed-info response.

    ``known_at_utc`` remains null by design. A deterministic availability lower
    bound is emitted only when the caller supplies an independently sourced JPX
    publication date; the effective ``Date`` never supplies that evidence.
    """

    requested = _parse_iso_date(request_date, "request_date")
    observed = _as_utc_iso(observed_at_utc, "observed_at_utc")
    if not _is_sha256(raw_content_sha256):
        raise ValueError("raw_content_sha256 must be a lowercase SHA-256 digest")
    document = _decode_json_document(payload)
    info = document.get("info")
    if not isinstance(info, list):
        raise ValueError("listed-info response must contain an 'info' array")

    lower_bound: str | None = None
    availability_basis: str | None = None
    if source_publication_date_jst is not None:
        lower_bound = availability_lower_bound_utc(source_publication_date_jst)
        availability_basis = (
            "official_next_business_day_release_not_before_17_30_jst_on_"
            "independently_sourced_publication_date"
        )

    rows: list[dict[str, object]] = []
    for index, source_row in enumerate(info):
        if not isinstance(source_row, Mapping):
            raise ValueError(f"info[{index}] must be a JSON object")
        effective = _parse_iso_date(source_row.get("Date"), f"info[{index}].Date")
        if effective < requested:
            raise ValueError(
                f"info[{index}].Date {effective.isoformat()} precedes request date "
                f"{requested.isoformat()}"
            )
        code = _required_string(source_row.get("Code"), f"info[{index}].Code")
        rows.append(
            {
                "effective_date": effective.isoformat(),
                "code": code,
                "company_name": _optional_string(source_row.get("CompanyName")),
                "company_name_english": _optional_string(
                    source_row.get("CompanyNameEnglish")
                ),
                "sector_17_code": _optional_string(source_row.get("Sector17Code")),
                "sector_17_name": _optional_string(
                    source_row.get("Sector17CodeName")
                ),
                "sector_33_code": _optional_string(source_row.get("Sector33Code")),
                "sector_33_name": _optional_string(
                    source_row.get("Sector33CodeName")
                ),
                "scale_category": _optional_string(source_row.get("ScaleCategory")),
                "market_code": _optional_string(source_row.get("MarketCode")),
                "market_name": _optional_string(source_row.get("MarketCodeName")),
                "margin_code": _optional_string(source_row.get("MarginCode")),
                "margin_name": _optional_string(source_row.get("MarginCodeName")),
                "base_price": _optional_number(source_row.get("BasePrice"), index),
                "request_date": requested.isoformat(),
                "known_at_utc": None,
                "known_at_source_field": None,
                "availability_lower_bound_utc": lower_bound,
                "availability_basis": availability_basis,
                "observed_at_utc": observed,
                "source_endpoint": LISTED_INFO_ENDPOINT,
                "source_api_version": API_VERSION,
                "raw_content_sha256": raw_content_sha256,
            }
        )
    return pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)


def stdlib_http_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: float,
) -> HttpResponse:
    """Execute one bounded HTTP request with the Python standard library."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers=dict(headers),
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            response_body = _read_bounded(response)
            return HttpResponse(
                status=int(response.status),
                body=response_body,
                headers={str(key): str(value) for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        response_body = _read_bounded(exc)
        return HttpResponse(
            status=int(exc.code),
            body=response_body,
            headers={str(key): str(value) for key, value in exc.headers.items()},
        )


def extract_jquants_listed_info(
    *,
    artifact_root: str | Path,
    run_id: str,
    end_date: str | date,
    max_dates: int,
    timeout_seconds: float = 30.0,
    retries: int = 3,
    backoff_seconds: float = 1.0,
    max_backoff_seconds: float = 60.0,
    sleep_seconds: float = 0.0,
    rate_limit_per_minute: float | None = 5.0,
    resume: bool = True,
    transport: HttpTransport = stdlib_http_transport,
    sleeper: Sleeper = time.sleep,
    clock: Clock | None = None,
    environment: Mapping[str, str] | None = None,
) -> JQuantsListedInfoArtifacts:
    """Run a credential-safe, bounded, resumable listed-info backfill."""

    _validate_run_parameters(
        run_id=run_id,
        max_dates=max_dates,
        timeout_seconds=timeout_seconds,
        retries=retries,
        backoff_seconds=backoff_seconds,
        max_backoff_seconds=max_backoff_seconds,
        sleep_seconds=sleep_seconds,
        rate_limit_per_minute=rate_limit_per_minute,
    )
    now = clock or (lambda: datetime.now(UTC))
    env = os.environ if environment is None else environment
    plan = build_listed_info_date_plan(end_date)
    artifact_dir = Path(artifact_root) / run_id
    raw_dir = artifact_dir / "raw"
    plan_path = artifact_dir / "request_plan.csv"
    request_manifest_path = artifact_dir / "request_manifest.csv"
    normalized_path = artifact_dir / "listed_info.csv"
    quality_path = artifact_dir / "quality_metrics.json"
    checksums_path = artifact_dir / "checksums.sha256"
    manifest_path = artifact_dir / "manifest.json"

    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not resume:
        raise FileExistsError(f"artifact directory is not empty: {artifact_dir}")
    raw_dir.mkdir(parents=True, exist_ok=True)

    with _run_lock(artifact_dir):
        if plan_path.exists():
            _validate_existing_plan(plan_path, plan)
        else:
            _atomic_write_csv(plan, plan_path)

        request_rows = _load_request_rows(request_manifest_path) if resume else []
        _validate_request_scope(plan, request_rows)
        request_rows = _mark_invalid_raw_successes(request_rows)
        _atomic_write_csv(_request_manifest_frame(request_rows), request_manifest_path)

        id_token = str(env.get("JQUANTS_ID_TOKEN", "")).strip()
        refresh_token = str(env.get("JQUANTS_REFRESH_TOKEN", "")).strip()
        credentials_mode = (
            "id_token_env"
            if id_token
            else "refresh_token_env"
            if refresh_token
            else "none"
        )
        run_status_override = ""
        attempted_this_invocation = 0

        if not id_token and refresh_token:
            auth_body = json.dumps(
                {"refreshtoken": refresh_token}, separators=(",", ":")
            ).encode("utf-8")
            auth_outcome = _request_with_retries(
                method="POST",
                url=AUTH_REFRESH_ENDPOINT,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": "quant-lab-jquants-listed-info/1.0",
                },
                body=auth_body,
                timeout_seconds=timeout_seconds,
                retries=retries,
                backoff_seconds=backoff_seconds,
                max_backoff_seconds=max_backoff_seconds,
                transport=transport,
                sleeper=sleeper,
                clock=now,
            )
            id_token, run_status_override = _id_token_from_refresh_outcome(
                auth_outcome, secrets=(refresh_token,)
            )

        if not id_token and credentials_mode == "none":
            run_status_override = "blocked_credentials"

        if id_token:
            successful_dates = {
                str(row["request_date"])
                for row in request_rows
                if row.get("status") in SUCCESS_STATUSES
            }
            pending = plan.loc[
                ~plan["request_date"].isin(successful_dates), "request_date"
            ].head(max_dates)
            selected_dates = pending.astype(str).tolist()
            inter_request_sleep = _effective_request_sleep(
                sleep_seconds=sleep_seconds,
                rate_limit_per_minute=rate_limit_per_minute,
            )
            for position, request_date in enumerate(selected_dates):
                attempted_this_invocation += 1
                request_url = LISTED_INFO_ENDPOINT + "?" + urllib.parse.urlencode(
                    {"date": request_date}
                )
                outcome = _request_with_retries(
                    method="GET",
                    url=request_url,
                    headers={
                        "Accept": "application/json",
                        "Authorization": f"Bearer {id_token}",
                        "User-Agent": "quant-lab-jquants-listed-info/1.0",
                    },
                    body=None,
                    timeout_seconds=timeout_seconds,
                    retries=retries,
                    backoff_seconds=backoff_seconds,
                    max_backoff_seconds=max_backoff_seconds,
                    transport=transport,
                    sleeper=sleeper,
                    clock=now,
                )
                previous = next(
                    (
                        row
                        for row in request_rows
                        if str(row.get("request_date")) == request_date
                    ),
                    None,
                )
                previous_attempts = _as_int(previous.get("attempts") if previous else 0)
                manifest_row, stop_authorization = _persist_request_outcome(
                    request_date=request_date,
                    request_url=request_url,
                    outcome=outcome,
                    previous_attempts=previous_attempts,
                    raw_dir=raw_dir,
                    clock=now,
                    secrets=(id_token, refresh_token),
                )
                request_rows = _replace_request_row(request_rows, manifest_row)
                _atomic_write_csv(
                    _request_manifest_frame(request_rows), request_manifest_path
                )
                if stop_authorization:
                    run_status_override = "blocked_authorization"
                    break
                if position + 1 < len(selected_dates) and inter_request_sleep > 0:
                    sleeper(inter_request_sleep)

        request_manifest = _request_manifest_frame(request_rows)
        listed_info = _normalized_from_verified_raw(request_rows)
        quality = _quality_metrics(plan, request_manifest, listed_info)
        _atomic_write_csv(request_manifest, request_manifest_path)
        _atomic_write_csv(listed_info, normalized_path)
        _atomic_write_json(quality, quality_path)

        checksum_entries = _artifact_checksums(
            artifact_dir,
            paths=(plan_path, request_manifest_path, normalized_path, quality_path),
            raw_dir=raw_dir,
        )
        _atomic_write_text(
            checksums_path,
            "".join(f"{digest}  {relative}\n" for relative, digest in checksum_entries),
        )
        completed_dates = int(
            request_manifest["status"].isin(SUCCESS_STATUSES).sum()
            if not request_manifest.empty
            else 0
        )
        status = run_status_override or (
            "complete" if completed_dates == len(plan) else "incomplete"
        )
        manifest = _build_manifest(
            run_id=run_id,
            artifact_dir=artifact_dir,
            end_date=_parse_iso_date(end_date, "end_date"),
            status=status,
            credentials_mode=credentials_mode,
            attempted_this_invocation=attempted_this_invocation,
            max_dates=max_dates,
            timeout_seconds=timeout_seconds,
            retries=retries,
            backoff_seconds=backoff_seconds,
            max_backoff_seconds=max_backoff_seconds,
            sleep_seconds=sleep_seconds,
            rate_limit_per_minute=rate_limit_per_minute,
            resume=resume,
            quality=quality,
            paths={
                "request_plan": plan_path,
                "request_manifest": request_manifest_path,
                "raw_directory": raw_dir,
                "listed_info": normalized_path,
                "quality_metrics": quality_path,
                "checksums": checksums_path,
            },
            checksum_entries=checksum_entries,
            checksums_path=checksums_path,
            generated_at_utc=_as_utc_iso(now(), "clock result"),
        )
        _atomic_write_json(manifest, manifest_path)

    return JQuantsListedInfoArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        request_plan_path=plan_path,
        request_manifest_path=request_manifest_path,
        raw_dir=raw_dir,
        normalized_path=normalized_path,
        quality_metrics_path=quality_path,
        checksums_path=checksums_path,
        manifest_path=manifest_path,
        request_plan=plan,
        request_manifest=request_manifest,
        listed_info=listed_info,
        quality_metrics=quality,
        manifest=manifest,
    )


def _persist_request_outcome(
    *,
    request_date: str,
    request_url: str,
    outcome: _RequestOutcome,
    previous_attempts: int,
    raw_dir: Path,
    clock: Clock,
    secrets: Sequence[str],
) -> tuple[dict[str, object], bool]:
    completed_at = _as_utc_iso(clock(), "clock result")
    base: dict[str, object] = {
        "request_date": request_date,
        "request_url": request_url,
        "status": "failed_transport",
        "attempts": previous_attempts + outcome.attempts,
        "http_status": "",
        "observed_at_utc": outcome.observed_at_utc,
        "completed_at_utc": completed_at,
        "source_rows": 0,
        "raw_response_path": "",
        "raw_metadata_path": "",
        "content_sha256": "",
        "response_sanitized": False,
        "error_type": "",
        "error": "",
    }
    if outcome.response is None:
        assert outcome.error is not None
        base["error_type"] = type(outcome.error).__name__
        base["error"] = _redact_text(str(outcome.error), secrets)
        return base, False

    response = outcome.response
    retained_body, sanitized = _redact_payload(response.body, secrets)
    content_sha256 = hashlib.sha256(retained_body).hexdigest()
    response_path = raw_dir / f"{request_date}.response.json"
    metadata_path = raw_dir / f"{request_date}.metadata.json"
    _atomic_write_bytes(response_path, retained_body)
    metadata = {
        "schema_version": "1.0",
        "request_date": request_date,
        "observed_at_utc": outcome.observed_at_utc,
        "endpoint": LISTED_INFO_ENDPOINT,
        "api_version": API_VERSION,
        "http_status": response.status,
        "content_sha256": content_sha256,
        "response_sanitized_for_credential_safety": sanitized,
    }
    _atomic_write_json(metadata, metadata_path)
    base.update(
        {
            "http_status": response.status,
            "raw_response_path": str(response_path.resolve()),
            "raw_metadata_path": str(metadata_path.resolve()),
            "content_sha256": content_sha256,
            "response_sanitized": sanitized,
        }
    )

    if response.status in {401, 403}:
        base.update(
            {
                "status": "blocked_authorization",
                "error_type": "AuthorizationError",
                "error": f"J-Quants returned HTTP {response.status}; extraction stopped",
            }
        )
        return base, True
    if response.status != 200:
        base.update(
            {
                "status": (
                    "failed_transient"
                    if response.status == 429 or 500 <= response.status <= 599
                    else "failed_http"
                ),
                "error_type": "HttpStatusError",
                "error": f"J-Quants returned HTTP {response.status}",
            }
        )
        return base, False

    try:
        normalized = normalize_listed_info_response(
            retained_body,
            request_date=request_date,
            observed_at_utc=outcome.observed_at_utc,
            raw_content_sha256=content_sha256,
        )
    except Exception as exc:
        base.update(
            {
                "status": "failed_parse",
                "error_type": type(exc).__name__,
                "error": _redact_text(str(exc), secrets),
            }
        )
        return base, False
    base.update(
        {
            "status": "completed_empty" if normalized.empty else "completed",
            "source_rows": len(normalized),
        }
    )
    return base, False


def _request_with_retries(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes | None,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    transport: HttpTransport,
    sleeper: Sleeper,
    clock: Clock,
) -> _RequestOutcome:
    last_error: Exception | None = None
    last_observed = ""
    for attempt in range(1, retries + 2):
        try:
            response = transport(method, url, headers, body, timeout_seconds)
            if not isinstance(response, HttpResponse):
                raise TypeError("transport must return HttpResponse")
            if len(response.body) > MAX_RESPONSE_BYTES:
                raise ValueError(
                    f"response exceeds {MAX_RESPONSE_BYTES} byte safety limit"
                )
            last_observed = _as_utc_iso(clock(), "clock result")
            transient = response.status == 429 or 500 <= response.status <= 599
            if transient and attempt <= retries:
                sleeper(
                    _retry_delay(
                        response=response,
                        attempt=attempt,
                        backoff_seconds=backoff_seconds,
                        max_backoff_seconds=max_backoff_seconds,
                    )
                )
                continue
            return _RequestOutcome(response, attempt, last_observed, None)
        except Exception as exc:
            last_error = exc
            last_observed = _as_utc_iso(clock(), "clock result")
            if attempt <= retries:
                sleeper(
                    min(
                        max_backoff_seconds,
                        backoff_seconds * (2 ** (attempt - 1)),
                    )
                )
                continue
            return _RequestOutcome(None, attempt, last_observed, exc)
    assert last_error is not None
    return _RequestOutcome(None, retries + 1, last_observed, last_error)


def _id_token_from_refresh_outcome(
    outcome: _RequestOutcome, *, secrets: Sequence[str]
) -> tuple[str, str]:
    if outcome.response is None:
        return "", "incomplete"
    if outcome.response.status in {401, 403}:
        return "", "blocked_authorization"
    if outcome.response.status != 200:
        return "", "incomplete"
    try:
        document = _decode_json_document(outcome.response.body)
        token = _required_string(document.get("idToken"), "auth_refresh.idToken")
    except Exception:
        return "", "incomplete"
    if any(secret and secret == token for secret in secrets):
        return token, ""
    return token, ""


def _normalized_from_verified_raw(
    request_rows: list[dict[str, object]],
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for row in sorted(request_rows, key=lambda item: str(item.get("request_date", ""))):
        if row.get("status") not in SUCCESS_STATUSES:
            continue
        request_date = str(row["request_date"])
        response_path = Path(str(row["raw_response_path"]))
        metadata_path = Path(str(row["raw_metadata_path"]))
        expected_hash = str(row["content_sha256"])
        if not response_path.is_file() or not metadata_path.is_file():
            raise ValueError(f"successful request is missing raw artifacts for {request_date}")
        payload = response_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected_hash:
            raise ValueError(f"raw response checksum mismatch for {request_date}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_metadata = {
            "request_date": request_date,
            "observed_at_utc": str(row["observed_at_utc"]),
            "endpoint": LISTED_INFO_ENDPOINT,
            "api_version": API_VERSION,
            "http_status": _as_int(row["http_status"]),
            "content_sha256": expected_hash,
        }
        for field, expected in expected_metadata.items():
            if metadata.get(field) != expected:
                raise ValueError(
                    f"raw metadata mismatch for {request_date}: field {field}"
                )
        normalized = normalize_listed_info_response(
            payload,
            request_date=request_date,
            observed_at_utc=str(row["observed_at_utc"]),
            raw_content_sha256=expected_hash,
        )
        if len(normalized) != _as_int(row["source_rows"]):
            raise ValueError(f"raw/manifest row-count mismatch for {request_date}")
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=NORMALIZED_COLUMNS)
    return pd.concat(frames, ignore_index=True).reindex(columns=NORMALIZED_COLUMNS)


def _quality_metrics(
    plan: pd.DataFrame,
    request_manifest: pd.DataFrame,
    listed_info: pd.DataFrame,
) -> dict[str, object]:
    statuses = (
        request_manifest["status"].astype(str)
        if not request_manifest.empty
        else pd.Series(dtype="object")
    )
    completed_dates = int(statuses.isin(SUCCESS_STATUSES).sum())
    duplicate_rows = int(
        listed_info.duplicated(subset=["request_date", "effective_date", "code"]).sum()
        if not listed_info.empty
        else 0
    )
    effective_before_request = int(
        (
            pd.to_datetime(listed_info["effective_date"], errors="coerce")
            < pd.to_datetime(listed_info["request_date"], errors="coerce")
        ).sum()
        if not listed_info.empty
        else 0
    )
    null_counts = {
        column: int(
            listed_info[column].isna().sum()
            + listed_info[column].astype(str).str.strip().eq("").sum()
        )
        for column in (
            "company_name",
            "sector_17_code",
            "sector_33_code",
            "market_code",
            "margin_code",
            "base_price",
            "known_at_utc",
            "availability_lower_bound_utc",
        )
    }
    return {
        "schema_version": "1.0",
        "planned_dates": int(len(plan)),
        "attempted_dates": int(len(request_manifest)),
        "completed_dates": completed_dates,
        "completed_empty_dates": int(statuses.eq("completed_empty").sum()),
        "failed_dates": int(statuses.str.startswith("failed").sum()),
        "blocked_authorization_dates": int(statuses.eq("blocked_authorization").sum()),
        "pending_dates": int(len(plan) - completed_dates),
        "date_coverage_ratio": completed_dates / len(plan),
        "normalized_rows": int(len(listed_info)),
        "unique_codes": int(listed_info["code"].nunique()) if not listed_info.empty else 0,
        "duplicate_request_effective_code_rows": duplicate_rows,
        "effective_date_before_request_rows": effective_before_request,
        "known_at_non_null_rows": int(_nonblank_mask(listed_info["known_at_utc"]).sum())
        if not listed_info.empty
        else 0,
        "availability_lower_bound_non_null_rows": int(
            _nonblank_mask(listed_info["availability_lower_bound_utc"]).sum()
        )
        if not listed_info.empty
        else 0,
        "null_counts": null_counts,
        "raw_integrity_rebuilt": True,
        "plan_validated_immutable": True,
        "promotion_flags": {
            "research_ready": False,
            "backtest_ready": False,
            "scorecard_eligible": False,
            "production_ready": False,
            "live_execution_ready": False,
        },
    }


def _build_manifest(
    *,
    run_id: str,
    artifact_dir: Path,
    end_date: date,
    status: str,
    credentials_mode: str,
    attempted_this_invocation: int,
    max_dates: int,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    sleep_seconds: float,
    rate_limit_per_minute: float | None,
    resume: bool,
    quality: dict[str, object],
    paths: Mapping[str, Path],
    checksum_entries: Sequence[tuple[str, str]],
    checksums_path: Path,
    generated_at_utc: str,
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "dataset_id": DATASET_ID,
        "run_id": run_id,
        "status": status,
        "research_only": True,
        "artifact_dir": str(artifact_dir.resolve()),
        "source": {
            "publisher": "JPX Market Innovation & Research, Inc.",
            "service": "J-Quants Pro",
            "endpoint": LISTED_INFO_ENDPOINT,
            "api_version": API_VERSION,
            "documentation": LISTED_INFO_DOCUMENTATION,
            "history_start_date": HISTORY_START_DATE.isoformat(),
            "documented_update_time_jst": "17:30:00",
            "coverage_note": (
                "Historical, current, and next-business-day listed information; "
                "next-business-day data is available after 17:30 JST."
            ),
        },
        "date_plan": {
            "start_date": HISTORY_START_DATE.isoformat(),
            "end_date": end_date.isoformat(),
            "date_basis": "inclusive_weekdays_with_official_effective_date_reconciliation",
            "planned_dates": quality["planned_dates"],
            "immutable": True,
        },
        "credential_policy": {
            "source": credentials_mode,
            "supported_environment_variables": [
                "JQUANTS_ID_TOKEN",
                "JQUANTS_REFRESH_TOKEN",
            ],
            "tokens_persisted": False,
            "tokens_logged": False,
            "auth_refresh_endpoint": AUTH_REFRESH_ENDPOINT,
            "auth_refresh_documentation": AUTH_REFRESH_DOCUMENTATION,
        },
        "known_at_policy": {
            "known_at_utc_default": None,
            "effective_date_used_as_known_at": False,
            "observed_at_used_as_historical_known_at": False,
            "policy": (
                "The API payload has no exact historical publication timestamp. "
                "Date is an effective date and is never promoted to known_at_utc."
            ),
            "availability_lower_bound_policy": (
                "Populate only when an independent source publication date is supplied; "
                "then use the documented not-before time of 17:30 JST. Historical "
                "backfill requests do not supply that evidence, so the field remains null."
            ),
        },
        "request_policy": {
            "max_dates_this_invocation": max_dates,
            "attempted_dates_this_invocation": attempted_this_invocation,
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "backoff_seconds": backoff_seconds,
            "max_backoff_seconds": max_backoff_seconds,
            "sleep_seconds": sleep_seconds,
            "rate_limit_per_minute": rate_limit_per_minute,
            "stop_on_http_status": [401, 403],
            "retry_http_status": [429, "5xx"],
            "single_writer_lock": True,
            "atomic_checkpoints": True,
            "resume": resume,
            "max_response_bytes": MAX_RESPONSE_BYTES,
        },
        "quality_metrics": quality,
        "promotion_flags": {
            "research_ready": False,
            "backtest_ready": False,
            "scorecard_eligible": False,
            "production_ready": False,
            "live_execution_ready": False,
        },
        "artifacts": {
            key: str(path.resolve()) for key, path in paths.items()
        },
        "sha256": {relative: digest for relative, digest in checksum_entries},
        "checksums_file_sha256": _sha256(checksums_path),
        "generated_at_utc": generated_at_utc,
    }


def _validate_run_parameters(
    *,
    run_id: str,
    max_dates: int,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
    sleep_seconds: float,
    rate_limit_per_minute: float | None,
) -> None:
    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    if max_dates <= 0:
        raise ValueError("max_dates must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if retries < 0:
        raise ValueError("retries cannot be negative")
    if backoff_seconds < 0 or max_backoff_seconds < 0:
        raise ValueError("backoff values cannot be negative")
    if max_backoff_seconds < backoff_seconds:
        raise ValueError("max_backoff_seconds cannot be below backoff_seconds")
    if sleep_seconds < 0:
        raise ValueError("sleep_seconds cannot be negative")
    if rate_limit_per_minute is not None and rate_limit_per_minute <= 0:
        raise ValueError("rate_limit_per_minute must be positive or None")


def _effective_request_sleep(
    *, sleep_seconds: float, rate_limit_per_minute: float | None
) -> float:
    rate_sleep = (
        60.0 / rate_limit_per_minute if rate_limit_per_minute is not None else 0.0
    )
    return max(sleep_seconds, rate_sleep)


def _retry_delay(
    *,
    response: HttpResponse,
    attempt: int,
    backoff_seconds: float,
    max_backoff_seconds: float,
) -> float:
    exponential = min(max_backoff_seconds, backoff_seconds * (2 ** (attempt - 1)))
    retry_after = _header(response.headers, "Retry-After")
    if retry_after:
        try:
            return min(max_backoff_seconds, max(exponential, float(retry_after)))
        except ValueError:
            pass
    return exponential


def _header(headers: Mapping[str, str], name: str) -> str:
    lowered = name.lower()
    return next(
        (str(value) for key, value in headers.items() if str(key).lower() == lowered),
        "",
    )


def _read_bounded(response: object) -> bytes:
    payload = response.read(MAX_RESPONSE_BYTES + 1)  # type: ignore[attr-defined]
    if len(payload) > MAX_RESPONSE_BYTES:
        raise ValueError(f"response exceeds {MAX_RESPONSE_BYTES} byte safety limit")
    return payload


def _decode_json_document(payload: bytes | Mapping[str, object]) -> dict[str, object]:
    if isinstance(payload, Mapping):
        return dict(payload)
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes or a mapping")
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"response is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise ValueError("response JSON root must be an object")
    return decoded


def _optional_number(value: object, row_index: int) -> float | str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        raise ValueError(f"info[{row_index}].BasePrice must be numeric or null")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"info[{row_index}].BasePrice must be numeric or null"
        ) from exc


def _nonblank_mask(series: pd.Series) -> pd.Series:
    return series.notna() & series.astype(str).str.strip().ne("")


def _required_string(value: object, field: str) -> str:
    result = _optional_string(value)
    if not result:
        raise ValueError(f"{field} must be a non-empty string")
    return result


def _optional_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _parse_iso_date(value: object, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD): {value!r}") from exc


def _as_utc_iso(value: str | datetime, field: str) -> str:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} is not a valid timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError(f"{field} must be a datetime or ISO timestamp")
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _as_int(value: object) -> int:
    if value in (None, ""):
        return 0
    return int(float(value))


def _load_request_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(REQUEST_MANIFEST_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"request manifest missing columns: {sorted(missing)}")
    return frame.loc[:, REQUEST_MANIFEST_COLUMNS].to_dict("records")


def _request_manifest_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=REQUEST_MANIFEST_COLUMNS)
    frame = pd.DataFrame(rows).reindex(columns=REQUEST_MANIFEST_COLUMNS)
    return frame.sort_values("request_date", kind="stable").reset_index(drop=True)


def _validate_request_scope(
    plan: pd.DataFrame, request_rows: list[dict[str, object]]
) -> None:
    planned = set(plan["request_date"].astype(str))
    seen: set[str] = set()
    for row in request_rows:
        request_date = str(row.get("request_date", ""))
        if request_date not in planned:
            raise ValueError(f"request manifest date is outside immutable plan: {request_date}")
        if request_date in seen:
            raise ValueError(f"request manifest contains duplicate date: {request_date}")
        seen.add(request_date)


def _mark_invalid_raw_successes(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for source in rows:
        row = dict(source)
        if row.get("status") in SUCCESS_STATUSES:
            response_path = Path(str(row.get("raw_response_path", "")))
            metadata_path = Path(str(row.get("raw_metadata_path", "")))
            expected = str(row.get("content_sha256", ""))
            valid = (
                response_path.is_file()
                and metadata_path.is_file()
                and _is_sha256(expected)
                and _sha256(response_path) == expected
            )
            if not valid:
                row.update(
                    {
                        "status": "failed_integrity",
                        "error_type": "RawIntegrityError",
                        "error": "successful checkpoint raw artifacts failed verification",
                    }
                )
        output.append(row)
    return output


def _replace_request_row(
    rows: list[dict[str, object]], replacement: dict[str, object]
) -> list[dict[str, object]]:
    request_date = str(replacement["request_date"])
    output = [
        row for row in rows if str(row.get("request_date", "")) != request_date
    ]
    output.append(replacement)
    return output


def _validate_existing_plan(path: Path, expected: pd.DataFrame) -> None:
    existing = pd.read_csv(path, dtype=str, keep_default_na=False)
    if tuple(existing.columns) != PLAN_COLUMNS:
        raise ValueError("existing request plan schema differs from immutable plan")
    if existing.to_dict("records") != expected.to_dict("records"):
        raise ValueError("existing request plan differs from requested immutable date plan")


def _redact_payload(payload: bytes, secrets: Sequence[str]) -> tuple[bytes, bool]:
    redacted = payload
    changed = False
    for secret in secrets:
        if not secret:
            continue
        encoded = secret.encode("utf-8")
        if encoded in redacted:
            redacted = redacted.replace(encoded, b"[REDACTED]")
            changed = True
    return redacted, changed


def _redact_text(value: str, secrets: Sequence[str]) -> str:
    redacted = value
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _artifact_checksums(
    artifact_dir: Path, *, paths: Sequence[Path], raw_dir: Path
) -> list[tuple[str, str]]:
    files = [path for path in paths if path.is_file()]
    files.extend(sorted(path for path in raw_dir.glob("*.json") if path.is_file()))
    return [
        (path.relative_to(artifact_dir).as_posix(), _sha256(path))
        for path in sorted(files)
    ]


@contextmanager
def _run_lock(artifact_dir: Path):
    lock_path = artifact_dir / ".extract.lock"
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another extractor owns the run lock: {lock_path}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        handle.close()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _atomic_write_json(data: Mapping[str, object], path: Path) -> None:
    _atomic_write_text(path, json.dumps(data, indent=2, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
