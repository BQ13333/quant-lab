from __future__ import annotations

from pathlib import Path

from quant_lab.data.pit_coverage import (
    COLUMNS,
    CoverageRow,
    assess_row,
    load_sheet,
    on_disk_identities,
    write_sheet,
)

ROOT = Path(__file__).resolve().parents[1]
SHEET = ROOT / "configs" / "pit_coverage_on_disk.csv"


def _row(**overrides: str) -> CoverageRow:
    base = {
        "sid": "1155.KL",
        "market": "Malaysia",
        "exchange": "Bursa Malaysia",
        "symbol": "1155.KL",
        "status": "active",
        "listing_date": "2000-01-01",
        "delist_date": "",
        "corporate_actions": "explicitly_absent",
        "fundamentals_source": "bursa_announcements",
        "evidence_pointer": "https://www.bursamalaysia.com/market_information/announcements/company_announcement",
        "where_found": "tests/test_pit_coverage.py",
    }
    base.update(overrides)
    return CoverageRow(**base)


def test_on_disk_sheet_blocks_every_committed_name() -> None:
    rows = on_disk_identities()
    assert [row.symbol for row in rows if row.market == "Malaysia"] == sorted(
        row.symbol for row in rows if row.market == "Malaysia"
    )
    assert len([row for row in rows if row.market == "Malaysia"]) == 31
    assert {row.symbol for row in rows if row.market == "Hong Kong"} == {"0005.HK", "0700.HK"}
    assert {row.symbol for row in rows if row.exchange == "SSE"} == {"600000.SS", "600004.SS", "600519.SS"}
    assert {row.symbol for row in rows if row.exchange == "SZSE"} == {"000001.SZ"}
    assert all(row.exchange != "BSE" for row in rows)
    assert all(row.fundamentals_source == "absent" for row in rows)
    assert all(not assess_row(row).admitted for row in rows)
    assert "listing_date_missing" in assess_row(rows[0]).reasons
    assert load_sheet(SHEET) == rows


def test_complete_official_row_is_admitted_and_gaps_fail_closed(tmp_path: Path) -> None:
    admitted = _row()
    inactive = _row(status="inactive", delist_date="")
    paid = _row(fundamentals_source="refinitiv", evidence_pointer="vendor")
    bse = _row(sid="830000.BJ", symbol="830000.BJ", market="China", exchange="BSE")
    frozen = _row(sid="D05.SI", symbol="D05.SI", market="Singapore", exchange="SGX")
    assert assess_row(admitted).admitted is True
    assert "inactive_without_delist_date" in assess_row(inactive).reasons
    assert "fundamentals_source_not_official" in assess_row(paid).reasons
    assert "bse_out" in assess_row(bse).reasons
    assert "market_not_active" in assess_row(frozen).reasons
    target = tmp_path / "sheet.csv"
    write_sheet(target, [admitted])
    assert load_sheet(target) == (admitted,)
    text = target.read_text(encoding="utf-8")
    assert text.splitlines()[0] == ",".join(COLUMNS)
