from __future__ import annotations

import hashlib
import threading
from datetime import datetime, timezone
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from quant_lab.data.nasdaq_trade_halts import (
    MAX_RESPONSE_BYTES,
    _acquire_run_lock,
    build_business_date_plan,
    extract_nasdaq_trade_halts,
    fetch_nasdaq_trade_halt_xml,
    parse_trade_halt_xml,
    source_url_for_date,
)
from quant_lab.data.usa_exchange_calendar_tca import (
    HALT_COLUMNS,
    load_halt_suspension_evidence,
)


@pytest.fixture
def feed_factory():
    def build(items: list[dict[str, str]], *, declared_items: int | None = None) -> bytes:
        item_xml = []
        defaults = {
            "IssueName": "Fixture Incorporated",
            "PauseThresholdPrice": "",
            "ReasonCode": "LUDP",
            "ResumptionDate": "",
            "ResumptionQuoteTime": "",
            "ResumptionTradeTime": "",
        }
        for supplied in items:
            values = {**defaults, **supplied}
            fields = "\n".join(
                f"<ndaq:{name}>{value}</ndaq:{name}>"
                for name, value in values.items()
            )
            item_xml.append(f"<item>{fields}</item>")
        count = len(items) if declared_items is None else declared_items
        return (
            "<?xml version='1.0' encoding='utf-8'?>"
            "<rss version='2.0' xmlns:ndaq='http://www.nasdaqtrader.com/'>"
            "<channel><title>NASDAQTrader.com</title>"
            f"<ndaq:numItems>{count}</ndaq:numItems>"
            f"{''.join(item_xml)}"
            "</channel></rss>"
        ).encode("utf-8")

    return build


def _item(
    day: str,
    *,
    symbol: str = "TEST",
    market: str = "Q",
    halt_time: str = "09:30:00",
    reason: str = "LUDP",
    resume_day: str = "",
    resume_time: str = "",
) -> dict[str, str]:
    return {
        "IssueSymbol": symbol,
        "Mkt": market,
        "ReasonCode": reason,
        "HaltDate": day,
        "HaltTime": halt_time,
        "ResumptionDate": resume_day,
        "ResumptionQuoteTime": resume_time,
        "ResumptionTradeTime": resume_time,
    }


def test_business_date_plan_is_inclusive_and_skips_weekends():
    plan = build_business_date_plan("2024-06-07", "2024-06-11")

    assert plan["request_date"].tolist() == ["2024-06-07", "2024-06-10", "2024-06-11"]
    assert plan.iloc[1]["source_url"].endswith("haltdate=06102024")


def test_parse_ndaq_fields_normalizes_dst_and_preserves_source_codes(feed_factory):
    source_url = source_url_for_date("2024-06-03")
    payload = feed_factory(
        [
            _item(
                "06/03/2024",
                symbol="DSTQ",
                market="Q",
                halt_time="09:30:00                      .125",
                reason="T1",
                resume_day="06/03/2024",
                resume_time="10:00:00",
            ),
            _item("06/03/2024", symbol="NYN", market="N", halt_time="10:15:00"),
            _item("06/03/2024", symbol="ARCA", market="P", halt_time="11:00:00"),
        ]
    )

    parsed = parse_trade_halt_xml(
        payload,
        requested_date="2024-06-03",
        source_url=source_url,
        observed_at_utc="2024-06-04T01:02:03Z",
    )

    assert tuple(parsed.events.columns) == HALT_COLUMNS
    assert parsed.declared_items == 3
    assert parsed.parsed_items == 3
    assert parsed.unsupported_market_events == 0
    assert parsed.market_codes == ("N", "P", "Q")
    assert set(parsed.events["exchange"]) == {"NASDAQ", "NYSE", "NYSE_ARCA"}
    nasdaq = parsed.events.set_index("symbol").loc["DSTQ"]
    assert nasdaq["halt_start_utc"] == "2024-06-03T13:30:00.125000+00:00"
    assert nasdaq["halt_end_utc"] == "2024-06-03T14:00:00.000000+00:00"
    assert nasdaq["reason_code"] == "T1"
    assert nasdaq["source_name"] == source_url
    assert "mkt=Q" in nasdaq["source_record_id"]
    assert "reason=T1" in nasdaq["source_record_id"]
    assert nasdaq["observed_at_utc"] == "2024-06-04T01:02:03.000000+00:00"

    winter = parse_trade_halt_xml(
        feed_factory([_item("01/03/2024", halt_time="09:30:00")]),
        requested_date="2024-01-03",
        source_url=source_url_for_date("2024-01-03"),
        observed_at_utc="2024-01-04T00:00:00Z",
    )
    assert winter.events.iloc[0]["halt_start_utc"] == "2024-01-03T14:30:00.000000+00:00"


def test_parser_rejects_wrong_date_and_count_mismatch_but_deduplicates_exact_records(
    feed_factory,
):
    with pytest.raises(ValueError, match="date coverage mismatch"):
        parse_trade_halt_xml(
            feed_factory([_item("06/04/2024")]),
            requested_date="2024-06-03",
            source_url=source_url_for_date("2024-06-03"),
            observed_at_utc="2024-06-04T00:00:00Z",
        )

    with pytest.raises(ValueError, match="item count mismatch"):
        parse_trade_halt_xml(
            feed_factory([], declared_items=1),
            requested_date="2024-06-03",
            source_url=source_url_for_date("2024-06-03"),
            observed_at_utc="2024-06-04T00:00:00Z",
        )

    duplicate = _item("06/03/2024")
    parsed = parse_trade_halt_xml(
        feed_factory([duplicate, duplicate]),
        requested_date="2024-06-03",
        source_url=source_url_for_date("2024-06-03"),
        observed_at_utc="2024-06-04T00:00:00Z",
    )
    assert parsed.parsed_items == 2
    assert len(parsed.events) == 1
    assert parsed.duplicate_source_items == 1


def test_parser_preserves_halt_and_blanks_impossible_resumption(feed_factory):
    parsed = parse_trade_halt_xml(
        feed_factory(
            [
                _item(
                    "11/03/2009",
                    symbol="FKFS",
                    halt_time="18:08:01",
                    resume_day="11/03/2009",
                    resume_time="16:50:00",
                )
            ]
        ),
        requested_date="2009-11-03",
        source_url=source_url_for_date("2009-11-03"),
        observed_at_utc="2024-06-04T00:00:00Z",
    )
    assert len(parsed.events) == 1
    assert parsed.events.iloc[0]["halt_end_utc"] == ""
    assert parsed.invalid_resumption_events == 1


def test_bounded_run_persists_raw_coverage_and_resumes_without_empty_events(
    tmp_path, feed_factory
):
    payloads = {
        "01022024": feed_factory(
            [
                _item(
                    "01/02/2024",
                    symbol="ONE",
                    market="Q",
                    halt_time="09:31:00",
                    resume_day="01/02/2024",
                    resume_time="09:36:00",
                )
            ]
        ),
        "01032024": feed_factory([]),
        "01042024": feed_factory(
            [_item("01/04/2024", symbol="TWO", market="N", halt_time="10:00:00")]
        ),
    }
    calls: list[tuple[str, float]] = []

    def fake_fetch(url: str, timeout: float) -> bytes:
        calls.append((url, timeout))
        haltdate = parse_qs(urlparse(url).query)["haltdate"][0]
        return payloads[haltdate]

    fixed_clock = lambda: datetime(2024, 1, 5, 12, 0, tzinfo=timezone.utc)
    first = extract_nasdaq_trade_halts(
        artifact_root=tmp_path,
        run_id="NASDAQ-HALTS-TEST",
        start_date="2024-01-02",
        end_date="2024-01-04",
        max_dates=2,
        timeout_seconds=7.5,
        retries=0,
        sleep_seconds=0,
        fetcher=fake_fetch,
        clock=fixed_clock,
    )

    assert len(calls) == 2
    assert all(timeout == 7.5 for _, timeout in calls)
    assert first.manifest["status"] == "incomplete"
    assert first.manifest["covered_dates"] == 2
    assert first.coverage["status"].tolist() == ["completed", "completed_empty", "pending"]
    assert first.halt_events["symbol"].tolist() == ["ONE"]
    assert tuple(first.halt_events.columns) == HALT_COLUMNS
    assert len(list(first.raw_dir.glob("*.xml"))) == 2
    for row in first.request_manifest.to_dict("records"):
        raw = first.raw_dir / f"{row['request_date']}.xml"
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == row["raw_sha256"]

    loaded = load_halt_suspension_evidence(first.halt_events_path)
    assert loaded["symbol"].tolist() == ["ONE"]

    second = extract_nasdaq_trade_halts(
        artifact_root=tmp_path,
        run_id="NASDAQ-HALTS-TEST",
        start_date="2024-01-02",
        end_date="2024-01-04",
        max_dates=1,
        timeout_seconds=7.5,
        retries=0,
        sleep_seconds=0,
        fetcher=fake_fetch,
        clock=fixed_clock,
    )

    assert len(calls) == 3
    assert "haltdate=01042024" in calls[-1][0]
    assert second.manifest["status"] == "complete"
    assert second.manifest["covered_dates"] == 3
    assert second.manifest["completed_empty_dates"] == 1
    assert set(second.halt_events["symbol"]) == {"ONE", "TWO"}
    assert not second.halt_events["symbol"].eq("").any()
    assert second.manifest_path.exists()
    assert second.coverage_path.exists()
    assert second.summary_path.exists()


def test_failed_request_is_retried_and_remains_resumable(tmp_path, feed_factory):
    calls = 0
    sleeps: list[float] = []

    def failing_fetch(_url: str, _timeout: float) -> bytes:
        nonlocal calls
        calls += 1
        raise TimeoutError("fixture timeout")

    failed = extract_nasdaq_trade_halts(
        artifact_root=tmp_path,
        run_id="NASDAQ-HALTS-RETRY",
        start_date="2024-01-02",
        end_date="2024-01-02",
        max_dates=1,
        retries=1,
        sleep_seconds=0.25,
        fetcher=failing_fetch,
        sleeper=sleeps.append,
        clock=lambda: datetime(2024, 1, 3, tzinfo=timezone.utc),
    )

    assert calls == 2
    assert sleeps == [0.25]
    assert failed.request_manifest.iloc[0]["status"] == "failed"
    assert failed.request_manifest.iloc[0]["attempts"] == 2
    assert failed.manifest["covered_dates"] == 0

    resumed_calls = 0

    def successful_fetch(_url: str, _timeout: float) -> bytes:
        nonlocal resumed_calls
        resumed_calls += 1
        return feed_factory([])

    resumed = extract_nasdaq_trade_halts(
        artifact_root=tmp_path,
        run_id="NASDAQ-HALTS-RETRY",
        start_date="2024-01-02",
        end_date="2024-01-02",
        max_dates=1,
        retries=0,
        sleep_seconds=0,
        fetcher=successful_fetch,
        clock=lambda: datetime(2024, 1, 3, tzinfo=timezone.utc),
    )

    assert resumed_calls == 1
    assert resumed.request_manifest.iloc[0]["status"] == "completed_empty"
    assert resumed.request_manifest.iloc[0]["attempts"] == 3
    assert resumed.halt_events.empty
    assert tuple(resumed.halt_events.columns) == HALT_COLUMNS
    written = pd.read_csv(resumed.halt_events_path)
    assert written.empty
    assert tuple(written.columns) == HALT_COLUMNS


def test_parallel_fetch_is_bounded_and_keeps_one_complete_manifest(tmp_path, feed_factory):
    barrier = threading.Barrier(2)

    def concurrent_fetch(_url: str, _timeout: float) -> bytes:
        barrier.wait(timeout=2)
        return feed_factory([])

    artifacts = extract_nasdaq_trade_halts(
        artifact_root=tmp_path,
        run_id="NASDAQ-HALTS-PARALLEL",
        start_date="2024-01-02",
        end_date="2024-01-03",
        max_dates=2,
        retries=0,
        workers=2,
        sleep_seconds=0,
        fetcher=concurrent_fetch,
        clock=lambda: datetime(2024, 1, 4, tzinfo=timezone.utc),
    )

    assert artifacts.manifest["status"] == "complete"
    assert artifacts.manifest["request_policy"]["workers"] == 2
    assert len(artifacts.request_manifest) == 2
    assert artifacts.request_manifest["status"].eq("completed_empty").all()


def test_http_response_byte_cap_and_single_writer_lock(tmp_path, monkeypatch):
    class OversizedResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, limit: int):
            assert limit == MAX_RESPONSE_BYTES + 1
            return b"x" * limit

    monkeypatch.setattr(
        "quant_lab.data.nasdaq_trade_halts.urllib.request.urlopen",
        lambda *_args, **_kwargs: OversizedResponse(),
    )
    with pytest.raises(ValueError, match="byte safety limit"):
        fetch_nasdaq_trade_halt_xml("https://example.test/feed", 1.0)

    tmp_path.mkdir(exist_ok=True)
    first_lock = _acquire_run_lock(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="another extractor owns"):
            _acquire_run_lock(tmp_path)
    finally:
        first_lock.close()


def test_resume_rejects_a_changed_request_plan(tmp_path, feed_factory):
    kwargs = {
        "artifact_root": tmp_path,
        "run_id": "NASDAQ-HALTS-SCOPE",
        "start_date": "2024-01-02",
        "end_date": "2024-01-02",
        "max_dates": 1,
        "retries": 0,
        "sleep_seconds": 0,
        "fetcher": lambda _url, _timeout: feed_factory([]),
        "clock": lambda: datetime(2024, 1, 3, tzinfo=timezone.utc),
    }
    extract_nasdaq_trade_halts(**kwargs)

    with pytest.raises(ValueError, match="request plan does not match"):
        extract_nasdaq_trade_halts(**{**kwargs, "end_date": "2024-01-03"})
