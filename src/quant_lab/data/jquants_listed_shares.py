from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests


API_BASE_URL = "https://api.jquants-pro.com/v2"
AUTH_REFRESH_URL = f"{API_BASE_URL}/token/auth_refresh"
HISTORY_START = date(2008, 5, 7)
TOKYO = ZoneInfo("Asia/Tokyo")
UTC = timezone.utc
MAX_RESPONSE_BYTES = 64 * 1024 * 1024
ALLOWED_QUERY_PARAMETERS = frozenset({"code", "date", "updated_at"})
SUCCESS_STATUSES = frozenset({"completed", "completed_empty"})
TERMINAL_PLAN_STATUSES = frozenset({"blocked_narrowing_required"})

ENDPOINTS = (
    ("shares", f"{API_BASE_URL}/listed/shares", "shares"),
    ("shares_change", f"{API_BASE_URL}/listed/shares_change", "shares_change"),
)

SHARE_FIELDS = (
    "Date",
    "EffectiveDate",
    "EffectiveEndDate",
    "Code",
    "OutstandingShares",
    "ListedShares",
    "IndexShares",
    "UpdatedAt",
)
SHARE_CHANGE_FIELDS = (
    "EffectiveDate",
    "SequentialNumber",
    "Code",
    "FactorType",
    "CABaseDate",
    "CASubstantialBaseDate",
    "CAEffectiveDate",
    "Changes",
    "Denominator",
    "Numerator",
    "UpdatedAt",
)

PROVENANCE_COLUMNS = (
    "known_at_jst",
    "known_at_utc",
    "updated_at_resolved",
    "pit_usable",
    "unusable_reason",
    "observed_at_utc",
    "source_endpoint",
    "source_plan_row_id",
    "source_page_number",
    "source_raw_path",
    "source_response_sha256",
    "source_record_sha256",
    "source_record_json",
)

SHARE_LEDGER_COLUMNS = (
    "date",
    "effective_date",
    "effective_end_date",
    "code",
    "outstanding_shares",
    "listed_shares",
    "index_shares",
    "updated_at",
    *PROVENANCE_COLUMNS,
)
SHARE_CHANGE_LEDGER_COLUMNS = (
    "effective_date",
    "sequential_number",
    "code",
    "factor_type",
    "ca_base_date",
    "ca_substantial_base_date",
    "ca_effective_date",
    "changes",
    "denominator",
    "numerator",
    "updated_at",
    *PROVENANCE_COLUMNS,
)

CHECKPOINT_COLUMNS = (
    "plan_row_id",
    "endpoint",
    "response_root",
    "updated_at",
    "code",
    "conditions_json",
    "status",
    "attempts",
    "pages",
    "rows",
    "next_pagination_key",
    "last_observed_at_utc",
    "completed_at_utc",
    "error_category",
    "error",
)
PAGE_MANIFEST_COLUMNS = (
    "plan_row_id",
    "endpoint",
    "page_number",
    "request_pagination_key",
    "next_pagination_key",
    "rows",
    "observed_at_utc",
    "raw_path",
    "response_sha256",
)

Clock = Callable[[], datetime]
Sleeper = Callable[[float], None]
Requester = Callable[..., object]
NextSession = Callable[[date], date | None]


@dataclass(frozen=True)
class JQuantsHttpResponse:
    status_code: int
    content: bytes
    headers: Mapping[str, str]


@dataclass(frozen=True)
class JQuantsJsonPage:
    payload: bytes
    records: tuple[dict[str, object], ...]
    next_pagination_key: str | None
    observed_at_utc: str
    attempts: int


@dataclass(frozen=True)
class JQuantsListedSharesArtifacts:
    run_id: str
    artifact_dir: Path
    plan_path: Path
    checkpoint_path: Path
    page_manifest_path: Path
    raw_dir: Path
    share_ledger_path: Path
    share_change_ledger_path: Path
    quality_path: Path
    checksums_path: Path
    manifest_path: Path
    plan: pd.DataFrame
    checkpoints: pd.DataFrame
    page_manifest: pd.DataFrame
    share_ledger: pd.DataFrame
    share_change_ledger: pd.DataFrame
    quality: pd.DataFrame
    manifest: dict[str, object]


class JQuantsRequestError(RuntimeError):
    """Credential-safe request failure carrying only classified state."""

    def __init__(self, category: str, status_code: int | None, attempts: int):
        self.category = category
        self.status_code = status_code
        self.attempts = attempts
        suffix = f" (HTTP {status_code})" if status_code is not None else ""
        super().__init__(f"J-Quants request failed: {category}{suffix}")


class JQuantsProV2Client:
    """Small credential-safe client for the official J-Quants Pro v2 routes."""

    __slots__ = (
        "_id_token",
        "_requester",
        "_sleeper",
        "_clock",
        "_timeout_seconds",
        "_retries",
        "_pacing_seconds",
        "_base_retry_seconds",
        "_max_retry_seconds",
        "_made_request",
    )

    def __init__(
        self,
        id_token: str,
        *,
        requester: Requester | None = None,
        sleeper: Sleeper = time.sleep,
        clock: Clock | None = None,
        timeout_seconds: float = 30.0,
        retries: int = 3,
        pacing_seconds: float = 1.0,
        base_retry_seconds: float = 2.0,
        max_retry_seconds: float = 60.0,
    ) -> None:
        if not id_token.strip():
            raise ValueError("id_token cannot be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if retries < 0:
            raise ValueError("retries cannot be negative")
        if min(pacing_seconds, base_retry_seconds, max_retry_seconds) < 0:
            raise ValueError("request delays cannot be negative")
        self._id_token = id_token
        self._requester = requester or request_jquants_http
        self._sleeper = sleeper
        self._clock = clock or (lambda: datetime.now(UTC))
        self._timeout_seconds = timeout_seconds
        self._retries = retries
        self._pacing_seconds = pacing_seconds
        self._base_retry_seconds = base_retry_seconds
        self._max_retry_seconds = max_retry_seconds
        self._made_request = False

    def __repr__(self) -> str:
        return "JQuantsProV2Client(id_token=<redacted>)"

    def fetch_page(
        self,
        *,
        url: str,
        response_root: str,
        conditions: Mapping[str, str],
        pagination_key: str | None = None,
    ) -> JQuantsJsonPage:
        original = {str(key): str(value) for key, value in conditions.items() if str(value)}
        invalid = set(original) - ALLOWED_QUERY_PARAMETERS
        if invalid:
            raise ValueError(f"unsupported query conditions: {sorted(invalid)}")
        if not original:
            raise ValueError("at least one of code, date, or updated_at is required")
        params = dict(original)
        if pagination_key:
            params["pagination_key"] = pagination_key

        for attempt in range(1, self._retries + 2):
            if self._made_request and self._pacing_seconds:
                self._sleeper(self._pacing_seconds)
            self._made_request = True
            try:
                raw_response = self._requester(
                    method="GET",
                    url=url,
                    headers={"Authorization": f"Bearer {self._id_token}", "Accept": "application/json"},
                    params=params,
                    json_body=None,
                    timeout_seconds=self._timeout_seconds,
                )
                response = _coerce_response(raw_response)
            except Exception:
                if attempt <= self._retries:
                    self._sleep_for_retry(attempt, None)
                    continue
                raise JQuantsRequestError("transport_exhausted", None, attempt) from None

            status = response.status_code
            if status == 200:
                return _parse_json_page(response.content, response_root, self._clock(), attempt)
            if status == 401:
                raise JQuantsRequestError("blocked_credentials", status, attempt)
            if status == 403:
                raise JQuantsRequestError("blocked_entitlement", status, attempt)
            if status == 413:
                raise JQuantsRequestError("blocked_narrowing_required", status, attempt)
            if status == 429 or 500 <= status <= 599:
                if attempt <= self._retries:
                    self._sleep_for_retry(attempt, response.headers.get("Retry-After"))
                    continue
                category = "rate_limit_exhausted" if status == 429 else "server_retries_exhausted"
                raise JQuantsRequestError(category, status, attempt)
            raise JQuantsRequestError("terminal_http_error", status, attempt)
        raise AssertionError("unreachable")

    def _sleep_for_retry(self, attempt: int, retry_after: str | None) -> None:
        delay = min(self._base_retry_seconds * (2 ** (attempt - 1)), self._max_retry_seconds)
        if retry_after:
            try:
                delay = min(max(delay, float(retry_after)), self._max_retry_seconds)
            except ValueError:
                pass
        if delay:
            self._sleeper(delay)


def request_jquants_http(
    *,
    method: str,
    url: str,
    headers: Mapping[str, str],
    params: Mapping[str, str] | None,
    json_body: Mapping[str, object] | None,
    timeout_seconds: float,
) -> JQuantsHttpResponse:
    """Issue one bounded HTTP request without including secrets in raised messages."""
    try:
        with requests.request(
            method,
            url,
            headers=dict(headers),
            params=dict(params or {}),
            json=dict(json_body) if json_body is not None else None,
            timeout=timeout_seconds,
            stream=True,
        ) as response:
            chunks: list[bytes] = []
            size = 0
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                size += len(chunk)
                if size > MAX_RESPONSE_BYTES:
                    raise ValueError("response exceeds configured byte limit")
                chunks.append(chunk)
            return JQuantsHttpResponse(
                status_code=int(response.status_code),
                content=b"".join(chunks),
                headers={str(key): str(value) for key, value in response.headers.items()},
            )
    except Exception as exc:
        raise RuntimeError(f"HTTP transport failed ({type(exc).__name__})") from None


def build_jquants_listed_shares_plan(
    start_date: str | date,
    end_date: str | date,
    codes: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build a deterministic revision-capture plan using one updated_at day per condition."""
    start = _parse_date(start_date, "start_date")
    end = _parse_date(end_date, "end_date")
    if start < HISTORY_START:
        raise ValueError(f"start_date cannot precede {HISTORY_START.isoformat()}")
    if end < start:
        raise ValueError("end_date cannot be before start_date")
    normalized_codes = _normalize_codes(codes)
    plan_codes = normalized_codes or ("",)

    rows: list[dict[str, str]] = []
    current = start
    while current <= end:
        # JPX is closed on weekends. Excluding those guaranteed no-update days
        # avoids duplicate official requests without assuming a holiday calendar.
        if current.weekday() < 5:
            updated_at = current.isoformat()
            for code in plan_codes:
                for endpoint, source_url, response_root in ENDPOINTS:
                    conditions = {"updated_at": updated_at}
                    if code:
                        conditions["code"] = code
                    conditions_json = _canonical_json(conditions)
                    scope = f"{endpoint}|{conditions_json}"
                    suffix = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:12]
                    rows.append(
                        {
                            "plan_row_id": f"{updated_at}-{endpoint}-{suffix}",
                            "endpoint": endpoint,
                            "source_url": source_url,
                            "response_root": response_root,
                            "updated_at": updated_at,
                            "code": code,
                            "conditions_json": conditions_json,
                        }
                    )
        current += timedelta(days=1)
    return pd.DataFrame(rows, columns=(
        "plan_row_id",
        "endpoint",
        "source_url",
        "response_root",
        "updated_at",
        "code",
        "conditions_json",
    ))


def known_at_from_updated_at(
    updated_at: object,
    *,
    next_session: NextSession | None = None,
) -> datetime | None:
    """Resolve availability at 09:00 JST on the JPX session after UpdatedAt."""
    if not isinstance(updated_at, str) or not updated_at.strip():
        return None
    try:
        updated_day = date.fromisoformat(updated_at.strip())
    except ValueError:
        return None
    resolver = next_session or _next_xtks_session
    try:
        delivery_day = resolver(updated_day)
    except Exception:
        return None
    if delivery_day is None or delivery_day <= updated_day:
        return None
    return datetime.combine(delivery_day, datetime_time(9, 0), tzinfo=TOKYO)


def normalize_jquants_shares(
    records: Sequence[Mapping[str, object]],
    *,
    observed_at_utc: str | datetime,
    response_sha256: str,
    source_plan_row_id: str = "",
    source_page_number: int = 1,
    source_raw_path: str = "",
    next_session: NextSession | None = None,
) -> pd.DataFrame:
    mapping = {
        "Date": "date",
        "EffectiveDate": "effective_date",
        "EffectiveEndDate": "effective_end_date",
        "Code": "code",
        "OutstandingShares": "outstanding_shares",
        "ListedShares": "listed_shares",
        "IndexShares": "index_shares",
        "UpdatedAt": "updated_at",
    }
    return _normalize_records(
        records,
        mapping=mapping,
        columns=SHARE_LEDGER_COLUMNS,
        endpoint="shares",
        observed_at_utc=observed_at_utc,
        response_sha256=response_sha256,
        source_plan_row_id=source_plan_row_id,
        source_page_number=source_page_number,
        source_raw_path=source_raw_path,
        next_session=next_session,
    )


def normalize_jquants_shares_change(
    records: Sequence[Mapping[str, object]],
    *,
    observed_at_utc: str | datetime,
    response_sha256: str,
    source_plan_row_id: str = "",
    source_page_number: int = 1,
    source_raw_path: str = "",
    next_session: NextSession | None = None,
) -> pd.DataFrame:
    mapping = {
        "EffectiveDate": "effective_date",
        "SequentialNumber": "sequential_number",
        "Code": "code",
        "FactorType": "factor_type",
        "CABaseDate": "ca_base_date",
        "CASubstantialBaseDate": "ca_substantial_base_date",
        "CAEffectiveDate": "ca_effective_date",
        "Changes": "changes",
        "Denominator": "denominator",
        "Numerator": "numerator",
        "UpdatedAt": "updated_at",
    }
    return _normalize_records(
        records,
        mapping=mapping,
        columns=SHARE_CHANGE_LEDGER_COLUMNS,
        endpoint="shares_change",
        observed_at_utc=observed_at_utc,
        response_sha256=response_sha256,
        source_plan_row_id=source_plan_row_id,
        source_page_number=source_page_number,
        source_raw_path=source_raw_path,
        next_session=next_session,
    )


def run_jquants_listed_shares_backfill(
    *,
    artifact_root: str | Path,
    run_id: str,
    start_date: str | date,
    end_date: str | date,
    max_dates: int,
    max_codes: int,
    codes: Sequence[str] | None = None,
    timeout_seconds: float = 30.0,
    retries: int = 3,
    pacing_seconds: float = 1.0,
    base_retry_seconds: float = 2.0,
    max_retry_seconds: float = 60.0,
    max_pages_per_plan_row: int = 100,
    resume: bool = True,
    requester: Requester | None = None,
    sleeper: Sleeper = time.sleep,
    clock: Clock | None = None,
    environ: Mapping[str, str] | None = None,
    next_session: NextSession | None = None,
) -> JQuantsListedSharesArtifacts:
    """Execute a bounded, single-writer, revision-preserving official backfill."""
    _validate_run_inputs(
        run_id=run_id,
        max_dates=max_dates,
        max_codes=max_codes,
        timeout_seconds=timeout_seconds,
        retries=retries,
        pacing_seconds=pacing_seconds,
        base_retry_seconds=base_retry_seconds,
        max_retry_seconds=max_retry_seconds,
        max_pages_per_plan_row=max_pages_per_plan_row,
    )
    now = clock or (lambda: datetime.now(UTC))
    env = os.environ if environ is None else environ
    plan = build_jquants_listed_shares_plan(start_date, end_date, codes)
    artifact_dir = Path(artifact_root) / run_id
    if artifact_dir.exists() and any(artifact_dir.iterdir()) and not resume:
        raise FileExistsError(f"artifact directory is not empty: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    run_lock = _acquire_run_lock(artifact_dir)

    plan_path = artifact_dir / "request_plan.csv"
    checkpoint_path = artifact_dir / "checkpoint.csv"
    page_manifest_path = artifact_dir / "page_manifest.csv"
    raw_dir = artifact_dir / "raw"
    share_ledger_path = artifact_dir / "share_ledger.csv"
    share_change_ledger_path = artifact_dir / "share_change_ledger.csv"
    quality_path = artifact_dir / "quality_metrics.csv"
    checksums_path = artifact_dir / "checksums.json"
    manifest_path = artifact_dir / "manifest.json"
    raw_dir.mkdir(parents=True, exist_ok=True)

    _write_or_validate_immutable_plan(plan, plan_path)
    checkpoints = _read_csv(checkpoint_path, CHECKPOINT_COLUMNS)
    page_manifest = _read_csv(page_manifest_path, PAGE_MANIFEST_COLUMNS)
    _validate_checkpoint_scope(plan, checkpoints)
    _validate_page_evidence(page_manifest)

    selected = _select_bounded_plan(plan, checkpoints, max_dates=max_dates, max_codes=max_codes)
    invocation_state = "running"
    auth_source = "none"
    calls_attempted = 0
    id_token: str | None = None

    if not selected.empty:
        try:
            id_token, auth_source, auth_attempts = _resolve_id_token(
                env,
                requester=requester or request_jquants_http,
                sleeper=sleeper,
                clock=now,
                timeout_seconds=timeout_seconds,
                retries=retries,
                base_retry_seconds=base_retry_seconds,
                max_retry_seconds=max_retry_seconds,
            )
            calls_attempted += auth_attempts
        except JQuantsRequestError as exc:
            invocation_state = exc.category
        if id_token is None and invocation_state == "running":
            invocation_state = "blocked_credentials"

    if id_token is not None and invocation_state == "running":
        client = JQuantsProV2Client(
            id_token,
            requester=requester,
            sleeper=sleeper,
            clock=now,
            timeout_seconds=timeout_seconds,
            retries=retries,
            pacing_seconds=pacing_seconds,
            base_retry_seconds=base_retry_seconds,
            max_retry_seconds=max_retry_seconds,
        )
        checkpoint_rows = checkpoints.to_dict("records")
        page_rows = page_manifest.to_dict("records")
        for plan_row in selected.to_dict("records"):
            existing = _row_by_id(checkpoint_rows, str(plan_row["plan_row_id"]))
            page_number = _as_int(existing.get("pages") if existing else 0) + 1
            pagination_key = _clean(existing.get("next_pagination_key") if existing else "") or None
            total_rows = _as_int(existing.get("rows") if existing else 0)
            total_attempts = _as_int(existing.get("attempts") if existing else 0)
            pages_this_invocation = 0
            seen_keys = {
                _clean(row.get("request_pagination_key"))
                for row in page_rows
                if str(row.get("plan_row_id")) == str(plan_row["plan_row_id"])
            }
            while pages_this_invocation < max_pages_per_plan_row:
                conditions = json.loads(str(plan_row["conditions_json"]))
                try:
                    page = client.fetch_page(
                        url=str(plan_row["source_url"]),
                        response_root=str(plan_row["response_root"]),
                        conditions=conditions,
                        pagination_key=pagination_key,
                    )
                    calls_attempted += page.attempts
                    total_attempts += page.attempts
                except JQuantsRequestError as exc:
                    calls_attempted += exc.attempts
                    total_attempts += exc.attempts
                    status = _checkpoint_status_for_error(exc.category)
                    checkpoint = _checkpoint_row(
                        plan_row,
                        status=status,
                        attempts=total_attempts,
                        pages=page_number - 1,
                        rows=total_rows,
                        next_pagination_key=pagination_key or "",
                        observed_at="",
                        completed_at=_as_utc_iso(now()),
                        error_category=exc.category,
                    )
                    checkpoint_rows = _replace_by_key(checkpoint_rows, checkpoint, ("plan_row_id",))
                    _atomic_write_csv(_frame(checkpoint_rows, CHECKPOINT_COLUMNS), checkpoint_path)
                    if status in {"blocked_credentials", "blocked_entitlement", "blocked_narrowing_required"}:
                        invocation_state = status
                    break

                request_key = pagination_key or ""
                next_key = page.next_pagination_key or ""
                if next_key and (next_key == request_key or next_key in seen_keys):
                    checkpoint = _checkpoint_row(
                        plan_row,
                        status="failed_response",
                        attempts=total_attempts,
                        pages=page_number - 1,
                        rows=total_rows,
                        next_pagination_key=request_key,
                        observed_at=page.observed_at_utc,
                        completed_at=_as_utc_iso(now()),
                        error_category="pagination_cycle",
                    )
                    checkpoint_rows = _replace_by_key(checkpoint_rows, checkpoint, ("plan_row_id",))
                    _atomic_write_csv(_frame(checkpoint_rows, CHECKPOINT_COLUMNS), checkpoint_path)
                    break

                raw_path = raw_dir / str(plan_row["endpoint"]) / str(plan_row["plan_row_id"]) / f"page-{page_number:04d}.json"
                _atomic_write_bytes(raw_path, page.payload)
                response_sha = _sha256(raw_path)
                page_row = {
                    "plan_row_id": plan_row["plan_row_id"],
                    "endpoint": plan_row["endpoint"],
                    "page_number": page_number,
                    "request_pagination_key": request_key,
                    "next_pagination_key": next_key,
                    "rows": len(page.records),
                    "observed_at_utc": page.observed_at_utc,
                    "raw_path": str(raw_path.resolve()),
                    "response_sha256": response_sha,
                }
                page_rows = _replace_by_key(page_rows, page_row, ("plan_row_id", "page_number"))
                _atomic_write_csv(_frame(page_rows, PAGE_MANIFEST_COLUMNS), page_manifest_path)
                total_rows += len(page.records)
                pages_this_invocation += 1
                status = "in_progress" if next_key else ("completed_empty" if total_rows == 0 else "completed")
                checkpoint = _checkpoint_row(
                    plan_row,
                    status=status,
                    attempts=total_attempts,
                    pages=page_number,
                    rows=total_rows,
                    next_pagination_key=next_key,
                    observed_at=page.observed_at_utc,
                    completed_at=_as_utc_iso(now()) if not next_key else "",
                    error_category="",
                )
                checkpoint_rows = _replace_by_key(checkpoint_rows, checkpoint, ("plan_row_id",))
                _atomic_write_csv(_frame(checkpoint_rows, CHECKPOINT_COLUMNS), checkpoint_path)
                seen_keys.add(request_key)
                if not next_key:
                    break
                pagination_key = next_key
                page_number += 1
            if invocation_state in {"blocked_credentials", "blocked_entitlement", "blocked_narrowing_required"}:
                break
        checkpoints = _frame(checkpoint_rows, CHECKPOINT_COLUMNS)
        page_manifest = _frame(page_rows, PAGE_MANIFEST_COLUMNS)

    _atomic_write_csv(checkpoints, checkpoint_path)
    _atomic_write_csv(page_manifest, page_manifest_path)
    _validate_page_evidence(page_manifest)
    share_ledger, share_change_ledger = _rebuild_ledgers(
        plan,
        checkpoints,
        page_manifest,
        next_session=next_session,
    )
    _atomic_write_csv(share_ledger, share_ledger_path)
    _atomic_write_csv(share_change_ledger, share_change_ledger_path)

    quality = _build_quality(plan, checkpoints, page_manifest, share_ledger, share_change_ledger)
    _atomic_write_csv(quality, quality_path)
    checksums = _build_checksums(
        plan_path=plan_path,
        checkpoint_path=checkpoint_path,
        page_manifest_path=page_manifest_path,
        share_ledger_path=share_ledger_path,
        share_change_ledger_path=share_change_ledger_path,
        quality_path=quality_path,
        page_manifest=page_manifest,
    )
    _atomic_write_text(checksums_path, json.dumps(checksums, indent=2, sort_keys=True) + "\n")

    completed = set(checkpoints.loc[checkpoints["status"].isin(SUCCESS_STATUSES), "plan_row_id"].astype(str))
    narrowing = int(checkpoints["status"].eq("blocked_narrowing_required").sum())
    if invocation_state == "running":
        if len(completed) == len(plan):
            invocation_state = "complete"
        elif narrowing:
            invocation_state = "blocked_narrowing_required"
        else:
            invocation_state = "incomplete"
    manifest: dict[str, object] = {
        "schema_version": "1.0",
        "run_id": run_id,
        "scope": "official_jquants_pro_v2_listed_shares_and_shares_change",
        "status": invocation_state,
        "promotion": False,
        "artifact_dir": str(artifact_dir.resolve()),
        "history_start": _parse_date(start_date, "start_date").isoformat(),
        "history_end": _parse_date(end_date, "end_date").isoformat(),
        "plan_rows": int(len(plan)),
        "selected_plan_rows_this_invocation": int(len(selected)),
        "completed_plan_rows": int(len(completed)),
        "pending_plan_rows": int(len(plan) - len(completed)),
        "max_dates_this_invocation": max_dates,
        "max_codes_this_invocation": max_codes,
        "http_attempts_this_invocation": calls_attempted,
        "auth_source": auth_source,
        "credential_values_persisted": False,
        "share_ledger_rows": int(len(share_ledger)),
        "share_change_ledger_rows": int(len(share_change_ledger)),
        "unusable_rows": int(
            share_ledger["pit_usable"].eq(False).sum()
            + share_change_ledger["pit_usable"].eq(False).sum()
        ),
        "known_at_policy": "09:00 Asia/Tokyo on the next valid XTKS session after UpdatedAt only",
        "revision_policy": "retain distinct source revisions and future-effective records; remove exact source-record duplicates only",
        "claims": {
            "comprehensive_dividends": False,
            "comprehensive_corporate_actions": False,
            "g2_complete": False,
        },
        "request_policy": {
            "timeout_seconds": timeout_seconds,
            "retries": retries,
            "pacing_seconds": pacing_seconds,
            "base_retry_seconds": base_retry_seconds,
            "max_retry_seconds": max_retry_seconds,
            "max_pages_per_plan_row": max_pages_per_plan_row,
            "single_writer_lock": True,
            "atomic_checkpoints": True,
            "resume": resume,
            "http_413_policy": "fail_closed_and_require_narrower_code/date/updated_at_conditions",
            "http_401_403_policy": "terminal_for_current_invocation",
        },
        "generated_at_utc": _as_utc_iso(now()),
        "artifacts": {
            "request_plan": str(plan_path.resolve()),
            "checkpoint": str(checkpoint_path.resolve()),
            "page_manifest": str(page_manifest_path.resolve()),
            "raw_directory": str(raw_dir.resolve()),
            "share_ledger": str(share_ledger_path.resolve()),
            "share_change_ledger": str(share_change_ledger_path.resolve()),
            "quality_metrics": str(quality_path.resolve()),
            "checksums": str(checksums_path.resolve()),
        },
        "checksums_sha256": _sha256(checksums_path),
    }
    _atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    run_lock.close()
    return JQuantsListedSharesArtifacts(
        run_id=run_id,
        artifact_dir=artifact_dir,
        plan_path=plan_path,
        checkpoint_path=checkpoint_path,
        page_manifest_path=page_manifest_path,
        raw_dir=raw_dir,
        share_ledger_path=share_ledger_path,
        share_change_ledger_path=share_change_ledger_path,
        quality_path=quality_path,
        checksums_path=checksums_path,
        manifest_path=manifest_path,
        plan=plan,
        checkpoints=checkpoints,
        page_manifest=page_manifest,
        share_ledger=share_ledger,
        share_change_ledger=share_change_ledger,
        quality=quality,
        manifest=manifest,
    )


def _resolve_id_token(
    environ: Mapping[str, str],
    *,
    requester: Requester,
    sleeper: Sleeper,
    clock: Clock,
    timeout_seconds: float,
    retries: int,
    base_retry_seconds: float,
    max_retry_seconds: float,
) -> tuple[str | None, str, int]:
    id_token = environ.get("JQUANTS_ID_TOKEN", "").strip()
    if id_token:
        return id_token, "JQUANTS_ID_TOKEN", 0
    refresh_token = environ.get("JQUANTS_REFRESH_TOKEN", "").strip()
    if not refresh_token:
        return None, "none", 0
    del clock
    for attempt in range(1, retries + 2):
        try:
            response = _coerce_response(
                requester(
                    method="POST",
                    url=AUTH_REFRESH_URL,
                    headers={"Accept": "application/json", "Content-Type": "application/json"},
                    params=None,
                    json_body={"refreshtoken": refresh_token},
                    timeout_seconds=timeout_seconds,
                )
            )
        except Exception:
            if attempt <= retries:
                _sleep_retry(sleeper, attempt, None, base_retry_seconds, max_retry_seconds)
                continue
            raise JQuantsRequestError("transport_exhausted", None, attempt) from None
        if response.status_code == 200:
            try:
                body = json.loads(response.content)
                token = body["idToken"]
            except (json.JSONDecodeError, KeyError, TypeError):
                raise JQuantsRequestError("invalid_refresh_response", 200, attempt) from None
            if not isinstance(token, str) or not token.strip():
                raise JQuantsRequestError("invalid_refresh_response", 200, attempt)
            return token, "JQUANTS_REFRESH_TOKEN", attempt
        if response.status_code in (401, 403):
            category = "blocked_credentials" if response.status_code == 401 else "blocked_entitlement"
            raise JQuantsRequestError(category, response.status_code, attempt)
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            if attempt <= retries:
                _sleep_retry(
                    sleeper,
                    attempt,
                    response.headers.get("Retry-After"),
                    base_retry_seconds,
                    max_retry_seconds,
                )
                continue
            category = "rate_limit_exhausted" if response.status_code == 429 else "server_retries_exhausted"
            raise JQuantsRequestError(category, response.status_code, attempt)
        raise JQuantsRequestError("terminal_http_error", response.status_code, attempt)
    raise AssertionError("unreachable")


def _normalize_records(
    records: Sequence[Mapping[str, object]],
    *,
    mapping: Mapping[str, str],
    columns: Sequence[str],
    endpoint: str,
    observed_at_utc: str | datetime,
    response_sha256: str,
    source_plan_row_id: str,
    source_page_number: int,
    source_raw_path: str,
    next_session: NextSession | None,
) -> pd.DataFrame:
    observed = _as_utc_iso(observed_at_utc)
    rows: list[dict[str, object]] = []
    for source in records:
        source_dict = dict(source)
        canonical = _canonical_json(source_dict)
        updated_at = source_dict.get("UpdatedAt")
        known_at = known_at_from_updated_at(updated_at, next_session=next_session)
        row = {normalized: source_dict.get(official) for official, normalized in mapping.items()}
        row.update(
            {
                "known_at_jst": known_at.isoformat() if known_at else "",
                "known_at_utc": known_at.astimezone(UTC).isoformat() if known_at else "",
                "updated_at_resolved": known_at is not None,
                "pit_usable": known_at is not None,
                "unusable_reason": "" if known_at else "updated_at_unresolved",
                "observed_at_utc": observed,
                "source_endpoint": endpoint,
                "source_plan_row_id": source_plan_row_id,
                "source_page_number": source_page_number,
                "source_raw_path": source_raw_path,
                "source_response_sha256": response_sha256,
                "source_record_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "source_record_json": canonical,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _rebuild_ledgers(
    plan: pd.DataFrame,
    checkpoints: pd.DataFrame,
    page_manifest: pd.DataFrame,
    *,
    next_session: NextSession | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    completed = set(checkpoints.loc[checkpoints["status"].isin(SUCCESS_STATUSES), "plan_row_id"].astype(str))
    plan_by_id = plan.set_index("plan_row_id", drop=False).to_dict("index")
    share_frames: list[pd.DataFrame] = []
    change_frames: list[pd.DataFrame] = []
    ordered_pages = page_manifest.sort_values(["plan_row_id", "page_number"], kind="stable")
    for evidence in ordered_pages.to_dict("records"):
        plan_row_id = str(evidence["plan_row_id"])
        if plan_row_id not in completed:
            continue
        plan_row = plan_by_id[plan_row_id]
        raw_path = Path(str(evidence["raw_path"]))
        payload = raw_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != str(evidence["response_sha256"]):
            raise ValueError(f"raw response hash mismatch for {raw_path}")
        page = _parse_json_page(
            payload,
            str(plan_row["response_root"]),
            str(evidence["observed_at_utc"]),
            attempts=1,
        )
        common = {
            "observed_at_utc": evidence["observed_at_utc"],
            "response_sha256": evidence["response_sha256"],
            "source_plan_row_id": plan_row_id,
            "source_page_number": _as_int(evidence["page_number"]),
            "source_raw_path": str(raw_path),
            "next_session": next_session,
        }
        if plan_row["endpoint"] == "shares":
            share_frames.append(normalize_jquants_shares(page.records, **common))
        else:
            change_frames.append(normalize_jquants_shares_change(page.records, **common))
    shares = _concat_and_dedupe(share_frames, SHARE_LEDGER_COLUMNS)
    changes = _concat_and_dedupe(change_frames, SHARE_CHANGE_LEDGER_COLUMNS)
    return shares, changes


def _concat_and_dedupe(frames: list[pd.DataFrame], columns: Sequence[str]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=columns)
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.drop_duplicates("source_record_sha256", keep="first")
    sort_columns = [column for column in ("code", "effective_date", "date", "updated_at", "sequential_number") if column in frame]
    return frame.sort_values(sort_columns, kind="stable", na_position="last").reset_index(drop=True)


def _build_quality(
    plan: pd.DataFrame,
    checkpoints: pd.DataFrame,
    page_manifest: pd.DataFrame,
    shares: pd.DataFrame,
    changes: pd.DataFrame,
) -> pd.DataFrame:
    completed = int(checkpoints["status"].isin(SUCCESS_STATUSES).sum())
    unusable = int(
        shares["pit_usable"].eq(False).sum()
        + changes["pit_usable"].eq(False).sum()
    )
    future_events = 0
    if not changes.empty:
        effective = pd.to_datetime(changes["effective_date"], errors="coerce")
        updated = pd.to_datetime(changes["updated_at"], errors="coerce")
        future_events = int((effective > updated).sum())
    metrics = [
        ("planned_rows", len(plan), "informational"),
        ("completed_rows", completed, "pass" if completed == len(plan) else "incomplete"),
        ("pending_rows", len(plan) - completed, "pass" if completed == len(plan) else "incomplete"),
        ("raw_pages", len(page_manifest), "informational"),
        ("share_ledger_rows", len(shares), "informational"),
        ("share_change_ledger_rows", len(changes), "informational"),
        ("pit_unusable_updated_at_rows", unusable, "pass" if unusable == 0 else "blocked_rows_retained"),
        ("future_share_change_rows_retained", future_events, "informational"),
        ("promotion", False, "fail_closed"),
        ("comprehensive_dividends_claim", False, "not_claimed"),
        ("comprehensive_corporate_actions_claim", False, "not_claimed"),
        ("g2_complete_claim", False, "not_claimed"),
    ]
    return pd.DataFrame(metrics, columns=("metric", "value", "status"))


def _build_checksums(**kwargs: object) -> dict[str, object]:
    page_manifest = kwargs.pop("page_manifest")
    assert isinstance(page_manifest, pd.DataFrame)
    artifacts = {
        key.removesuffix("_path"): _sha256(value)
        for key, value in kwargs.items()
        if isinstance(value, Path) and value.is_file()
    }
    raw_entries = [
        f"{row['raw_path']} {row['response_sha256']}"
        for row in page_manifest.sort_values(["plan_row_id", "page_number"]).to_dict("records")
    ]
    aggregate = hashlib.sha256(("\n".join(raw_entries) + ("\n" if raw_entries else "")).encode("utf-8")).hexdigest()
    return {"algorithm": "sha256", "artifacts": artifacts, "raw_pages_aggregate": aggregate, "raw_page_count": len(raw_entries)}


def _parse_json_page(payload: bytes, response_root: str, observed_at: str | datetime, attempts: int) -> JQuantsJsonPage:
    if len(payload) > MAX_RESPONSE_BYTES:
        raise JQuantsRequestError("response_too_large", 200, attempts)
    try:
        body = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise JQuantsRequestError("invalid_json_response", 200, attempts) from None
    if not isinstance(body, dict) or not isinstance(body.get(response_root), list):
        raise JQuantsRequestError("invalid_response_root", 200, attempts)
    records = body[response_root]
    if not all(isinstance(record, dict) for record in records):
        raise JQuantsRequestError("invalid_response_records", 200, attempts)
    next_key = body.get("pagination_key")
    if next_key is not None and (not isinstance(next_key, str) or not next_key.strip()):
        raise JQuantsRequestError("invalid_pagination_key", 200, attempts)
    return JQuantsJsonPage(
        payload=payload,
        records=tuple(dict(record) for record in records),
        next_pagination_key=next_key,
        observed_at_utc=_as_utc_iso(observed_at),
        attempts=attempts,
    )


def _select_bounded_plan(plan: pd.DataFrame, checkpoints: pd.DataFrame, *, max_dates: int, max_codes: int) -> pd.DataFrame:
    statuses = dict(zip(checkpoints["plan_row_id"].astype(str), checkpoints["status"].astype(str), strict=False))
    pending = plan[
        ~plan["plan_row_id"].astype(str).map(statuses).isin(SUCCESS_STATUSES | TERMINAL_PLAN_STATUSES)
    ].copy()
    dates = pending["updated_at"].drop_duplicates().head(max_dates)
    pending = pending[pending["updated_at"].isin(dates)]
    codes = pending["code"].fillna("").astype(str).drop_duplicates().head(max_codes)
    return pending[pending["code"].fillna("").astype(str).isin(codes)].reset_index(drop=True)


def _checkpoint_row(
    plan_row: Mapping[str, object],
    *,
    status: str,
    attempts: int,
    pages: int,
    rows: int,
    next_pagination_key: str,
    observed_at: str,
    completed_at: str,
    error_category: str,
) -> dict[str, object]:
    return {
        "plan_row_id": plan_row["plan_row_id"],
        "endpoint": plan_row["endpoint"],
        "response_root": plan_row["response_root"],
        "updated_at": plan_row["updated_at"],
        "code": plan_row["code"],
        "conditions_json": plan_row["conditions_json"],
        "status": status,
        "attempts": attempts,
        "pages": pages,
        "rows": rows,
        "next_pagination_key": next_pagination_key,
        "last_observed_at_utc": observed_at,
        "completed_at_utc": completed_at,
        "error_category": error_category,
        "error": _public_error_message(error_category),
    }


def _checkpoint_status_for_error(category: str) -> str:
    if category in {"blocked_credentials", "blocked_entitlement", "blocked_narrowing_required"}:
        return category
    if category in {"rate_limit_exhausted", "server_retries_exhausted", "transport_exhausted"}:
        return "failed_transient"
    return "failed_response" if category.startswith("invalid_") or category == "response_too_large" else "failed_request"


def _public_error_message(category: str) -> str:
    messages = {
        "": "",
        "blocked_credentials": "credentials were rejected; token value was not retained",
        "blocked_entitlement": "dataset entitlement was rejected; token value was not retained",
        "blocked_narrowing_required": "HTTP 413: create a narrower immutable plan using code/date/updated_at conditions",
        "rate_limit_exhausted": "bounded HTTP 429 retries exhausted",
        "server_retries_exhausted": "bounded HTTP 5xx retries exhausted",
        "transport_exhausted": "bounded transport retries exhausted",
        "pagination_cycle": "pagination key repeated; extraction stopped fail-closed",
        "invalid_json_response": "response was not valid JSON",
        "invalid_response_root": "response did not contain the required official root",
        "invalid_response_records": "response root contained a non-object record",
        "invalid_pagination_key": "response pagination key was invalid",
        "response_too_large": "response exceeded the local byte safety bound",
        "terminal_http_error": "non-retryable HTTP response",
    }
    return messages.get(category, "classified request failure; sensitive response detail was not retained")


def _validate_checkpoint_scope(plan: pd.DataFrame, checkpoints: pd.DataFrame) -> None:
    if checkpoints.empty:
        return
    plan_by_id = plan.set_index("plan_row_id")
    if checkpoints["plan_row_id"].astype(str).duplicated().any():
        raise ValueError("checkpoint contains duplicate plan_row_id values")
    for row in checkpoints.to_dict("records"):
        row_id = str(row["plan_row_id"])
        if row_id not in plan_by_id.index:
            raise ValueError(f"checkpoint row is outside immutable plan: {row_id}")
        expected = plan_by_id.loc[row_id]
        for column in ("endpoint", "response_root", "updated_at", "code", "conditions_json"):
            if _clean(row.get(column)) != _clean(expected[column]):
                raise ValueError(f"checkpoint scope mismatch for {row_id}: {column}")


def _validate_page_evidence(page_manifest: pd.DataFrame) -> None:
    if page_manifest.empty:
        return
    if page_manifest.duplicated(["plan_row_id", "page_number"]).any():
        raise ValueError("page manifest contains duplicate page evidence")
    for row in page_manifest.to_dict("records"):
        raw_path = Path(str(row["raw_path"]))
        if not raw_path.is_file():
            raise ValueError(f"raw page is missing: {raw_path}")
        if _sha256(raw_path) != str(row["response_sha256"]):
            raise ValueError(f"raw page hash mismatch: {raw_path}")


def _write_or_validate_immutable_plan(plan: pd.DataFrame, path: Path) -> None:
    content = plan.to_csv(index=False)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing immutable request plan does not match requested scope: {path}")
        return
    _atomic_write_text(path, content)


def _validate_run_inputs(**values: object) -> None:
    run_id = str(values["run_id"])
    if not run_id.strip() or Path(run_id).name != run_id:
        raise ValueError("run_id must be a non-empty path-safe name")
    for field in ("max_dates", "max_codes", "max_pages_per_plan_row"):
        if int(values[field]) <= 0:
            raise ValueError(f"{field} must be positive")
    if float(values["timeout_seconds"]) <= 0:
        raise ValueError("timeout_seconds must be positive")
    if int(values["retries"]) < 0:
        raise ValueError("retries cannot be negative")
    for field in ("pacing_seconds", "base_retry_seconds", "max_retry_seconds"):
        if float(values[field]) < 0:
            raise ValueError(f"{field} cannot be negative")


def _normalize_codes(codes: Sequence[str] | None) -> tuple[str, ...]:
    if not codes:
        return ()
    normalized: set[str] = set()
    for value in codes:
        code = str(value).strip().upper()
        if not re.fullmatch(r"[0-9A-Z]{4,5}", code):
            raise ValueError(f"invalid J-Quants code: {value!r}")
        normalized.add(code)
    return tuple(sorted(normalized))


def _next_xtks_session(updated_day: date) -> date | None:
    try:
        import exchange_calendars as xcals

        calendar = xcals.get_calendar("XTKS")
        candidate = pd.Timestamp(updated_day + timedelta(days=1))
        return calendar.date_to_session(candidate, direction="next").date()
    except (ImportError, ValueError, OverflowError):
        return None


def _coerce_response(response: object) -> JQuantsHttpResponse:
    if isinstance(response, JQuantsHttpResponse):
        return response
    status = int(getattr(response, "status_code"))
    content = getattr(response, "content")
    if isinstance(content, str):
        content = content.encode("utf-8")
    if not isinstance(content, bytes):
        raise TypeError("HTTP response content must be bytes")
    headers = getattr(response, "headers", {})
    return JQuantsHttpResponse(status, content, {str(key): str(value) for key, value in dict(headers).items()})


def _sleep_retry(
    sleeper: Sleeper,
    attempt: int,
    retry_after: str | None,
    base_retry_seconds: float,
    max_retry_seconds: float,
) -> None:
    delay = min(base_retry_seconds * (2 ** (attempt - 1)), max_retry_seconds)
    if retry_after:
        try:
            delay = min(max(delay, float(retry_after)), max_retry_seconds)
        except ValueError:
            pass
    if delay:
        sleeper(delay)


def _parse_date(value: str | date, field: str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as exc:
        raise ValueError(f"{field} must be YYYY-MM-DD") from exc


def _as_utc_iso(value: str | datetime) -> str:
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"invalid observed timestamp: {value!r}") from exc
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TypeError("timestamp must be a datetime or ISO string")
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed.astimezone(UTC).isoformat(timespec="microseconds")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _clean(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _as_int(value: object) -> int:
    return int(float(_clean(value) or 0))


def _row_by_id(rows: list[dict[str, object]], row_id: str) -> dict[str, object] | None:
    return next((row for row in rows if str(row.get("plan_row_id")) == row_id), None)


def _replace_by_key(
    rows: list[dict[str, object]],
    replacement: dict[str, object],
    keys: tuple[str, ...],
) -> list[dict[str, object]]:
    wanted = tuple(_clean(replacement.get(key)) for key in keys)
    kept = [row for row in rows if tuple(_clean(row.get(key)) for key in keys) != wanted]
    kept.append(replacement)
    return kept


def _frame(rows: list[dict[str, object]], columns: Sequence[str]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows).reindex(columns=columns)
    for column in ("attempts", "pages", "rows", "page_number"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype("Int64")
    sort_columns = [column for column in ("updated_at", "code", "endpoint", "plan_row_id", "page_number") if column in frame]
    return frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _read_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"checkpoint artifact is missing columns: {sorted(missing)}")
    return frame.reindex(columns=columns)


def _acquire_run_lock(artifact_dir: Path):
    path = artifact_dir / ".extract.lock"
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError(f"another extractor owns the run lock: {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()}\n")
    handle.flush()
    return handle


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_write_text(path: Path, content: str) -> None:
    _atomic_write_bytes(path, content.encode("utf-8"))


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write_text(path, frame.to_csv(index=False))


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "API_BASE_URL",
    "AUTH_REFRESH_URL",
    "HISTORY_START",
    "JQuantsHttpResponse",
    "JQuantsJsonPage",
    "JQuantsListedSharesArtifacts",
    "JQuantsProV2Client",
    "JQuantsRequestError",
    "SHARE_CHANGE_LEDGER_COLUMNS",
    "SHARE_LEDGER_COLUMNS",
    "build_jquants_listed_shares_plan",
    "known_at_from_updated_at",
    "normalize_jquants_shares",
    "normalize_jquants_shares_change",
    "request_jquants_http",
    "run_jquants_listed_shares_backfill",
]
