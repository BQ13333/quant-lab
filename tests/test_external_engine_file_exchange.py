from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from quant_lab.engines.file_exchange import (
    CONTRACT_VERSION,
    LICENCE_ACKNOWLEDGEMENT,
    DailyBar,
    FileExchangeError,
    load_request,
    read_result,
    write_request,
)

ROOT = Path(__file__).resolve().parents[1]


def _bar(symbol: str = "1155.KL", session: str = "2024-01-02", price: float = 8.5) -> DailyBar:
    return DailyBar(symbol, session, price, price, price, price, 1_000)


def _request(tmp_path: Path, **overrides):
    payload = {
        "directory": tmp_path / "request",
        "market": "MY",
        "decision_session": "2024-01-02",
        "next_session": "2024-01-03",
        "costs_bps_one_way": 20.0,
        "bars": (_bar(session="2023-12-29", price=8.4), _bar()),
    }
    payload.update(overrides)
    return write_request(**payload)


def _fill(**overrides) -> dict[str, object]:
    payload: dict[str, object] = {
        "symbol": "1155.KL",
        "decision_session": "2024-01-02",
        "fill_session": "2024-01-03",
        "side": "BUY",
        "quantity": 100,
        "fill_price": 8.6,
        "cost_bps": 20.0,
        "cost_applied": True,
    }
    payload.update(overrides)
    return payload


def _write_result(directory: Path, request, **overrides) -> None:
    payload: dict[str, object] = {
        "contract_version": CONTRACT_VERSION,
        "purpose": "research_only",
        "licence_acknowledgement": LICENCE_ACKNOWLEDGEMENT,
        "live_trading": False,
        "request_sha256": request.request_sha256,
        "external_engine_name": "ziplime",
        "lookahead_detected": False,
        "fills": [_fill()],
    }
    payload.update(overrides)
    directory.mkdir()
    (directory / "result.json").write_text(json.dumps(payload), encoding="utf-8")


def test_round_trip_accepts_next_bar_fill_with_cost(tmp_path: Path):
    request = _request(tmp_path)
    reloaded = load_request(request.directory)
    assert reloaded.request_sha256 == request.request_sha256
    assert reloaded.bars_sha256 == request.bars_sha256
    assert reloaded.symbols == ("1155.KL",)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request)
    result = read_result(result_dir, reloaded)
    assert result.external_engine_name == "ziplime"
    assert result.fills[0].fill_session == "2024-01-03"
    assert result.fills[0].cost_applied is True
    text = (request.directory / "request.json").read_text(encoding="utf-8")
    assert "live_trading" in text and "false" in text
    assert LICENCE_ACKNOWLEDGEMENT in text


@pytest.mark.parametrize(
    ("market", "symbol"),
    [("CN", "600519.SS"), ("CN", "000001.SZ"), ("HK", "0700.HK")],
)
def test_china_and_hong_kong_symbols_are_admitted(tmp_path: Path, market: str, symbol: str):
    request = _request(
        tmp_path,
        market=market,
        bars=(_bar(symbol, "2024-01-02", 10.0),),
    )
    assert request.market == market
    assert request.symbols == (symbol,)


@pytest.mark.parametrize("market", ["US", "JP", "SG", "CRYPTO", "CNH"])
def test_out_of_scope_markets_are_rejected(tmp_path: Path, market: str):
    with pytest.raises(FileExchangeError, match="MY, CN, or HK"):
        _request(tmp_path, market=market)


@pytest.mark.parametrize("symbol", ["AAPL", "1155.US", "BTC-USD", "1155.KLUSDT", "ETH1155.KL"])
def test_crypto_and_us_symbols_are_rejected(tmp_path: Path, symbol: str):
    with pytest.raises(FileExchangeError):
        _request(tmp_path, bars=(_bar(symbol),))


def test_intraday_timestamp_is_rejected(tmp_path: Path):
    with pytest.raises(FileExchangeError, match="daily session"):
        _request(tmp_path, bars=(_bar(session="2024-01-02T09:30:00"),))


def test_bar_after_the_decision_is_look_ahead(tmp_path: Path):
    with pytest.raises(FileExchangeError, match="after the decision"):
        _request(tmp_path, bars=(_bar(), _bar(session="2024-01-03")))


def test_same_session_fill_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, fills=[_fill(fill_session="2024-01-02")])
    with pytest.raises(FileExchangeError, match="next bar"):
        read_result(result_dir, request)


def test_skipped_session_is_not_the_next_bar(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, fills=[_fill(fill_session="2024-01-04")])
    with pytest.raises(FileExchangeError, match="next bar"):
        read_result(result_dir, request)


@pytest.mark.parametrize(
    "overrides",
    [
        {"cost_applied": False},
        {"cost_bps": 0},
        {"cost_bps": -5},
    ],
)
def test_missing_or_zero_cost_is_rejected(tmp_path: Path, overrides: dict[str, object]):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, fills=[_fill(**overrides)])
    with pytest.raises(FileExchangeError):
        read_result(result_dir, request)


def test_live_trading_flag_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, live_trading=True)
    with pytest.raises(FileExchangeError, match="live trading"):
        read_result(result_dir, request)


def test_broker_field_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, broker="ibkr")
    with pytest.raises(FileExchangeError, match="schema"):
        read_result(result_dir, request)


def test_lookahead_report_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, lookahead_detected=True)
    with pytest.raises(FileExchangeError, match="look-ahead"):
        read_result(result_dir, request)


def test_altered_licence_acknowledgement_is_rejected(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, licence_acknowledgement="fine for production")
    with pytest.raises(FileExchangeError, match="licence"):
        read_result(result_dir, request)


def test_broker_engine_name_is_rejected(tmp_path: Path):
    with pytest.raises(FileExchangeError, match="broker or live"):
        _request(tmp_path, external_engine_name="ibkr-paper")


def test_tampered_bars_fail_closed(tmp_path: Path):
    request = _request(tmp_path)
    bars = request.directory / "bars.csv"
    bars.write_text(bars.read_text(encoding="utf-8").replace("8.5", "9.5"), encoding="utf-8")
    with pytest.raises(FileExchangeError, match="bars.csv"):
        load_request(request.directory)


def test_empty_fill_list_is_a_valid_research_result(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, fills=[])
    assert read_result(result_dir, request).fills == ()


def test_fractional_shares_are_rejected(tmp_path: Path):
    request = _request(tmp_path)
    result_dir = tmp_path / "result"
    _write_result(result_dir, request, fills=[_fill(quantity=0.5)])
    with pytest.raises(FileExchangeError, match="whole number"):
        read_result(result_dir, request)


def test_ziplime_is_not_imported_or_declared():
    import quant_lab.engines.file_exchange as module

    assert "ziplime" not in sys.modules
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "import ziplime" not in source
    project = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "ziplime" not in project
