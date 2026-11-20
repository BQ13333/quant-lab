from pathlib import Path

from quant_lab.engines.file_exchange import load_request

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "ziplime_bursa_1155"


def test_one_name_bursa_request_is_on_disk_and_has_no_engine_result() -> None:
    request = load_request(FIXTURE)
    assert request.market == "MY"
    assert request.symbols == ("1155.KL",)
    assert request.decision_session == "2024-01-02"
    assert request.next_session == "2024-01-03"
    assert request.external_engine_name == "ziplime"
    assert request.costs_bps_one_way == 42.0
    assert not (FIXTURE / "result.json").exists()
    package = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "ziplime" not in package
    assert "lime-trader" not in package
