from __future__ import annotations

from quant_lab.data.japan_g4_halt_mapping_recovery_validator import (
    independently_reparse_plain_pages,
)


def test_independent_plain_reparse_finds_contiguous_domestic_record() -> None:
    page = """年月 (Year/Month) :2016/04
Stock Quotations (4000 - 4999)
4281　デジタル・アドバタイジング・コン
D.A.Consortium Inc.
サービス業
Services
JQS 貸借 100 819.00 (1) 855.00 (21)
4282　ＥＰＳホールディングス　普通株式
EPS Holdings,Inc.
"""
    result = independently_reparse_plain_pages(
        [page], observation_month="2016-04", code="4281"
    )
    assert result["record_count"] == 1
    assert result["unique_record"]["domestic_continuity"]
    assert not result["unique_record"]["foreign_marker"]


def test_independent_plain_reparse_rejects_foreign_marker() -> None:
    page = """年月 (Year/Month) :2024/02
Stock Quotations (9000 - 9999)
9399　ビート・ホールディングス・リミテッド <外国株>
Beat Holdings Limited <Foreign>
S外 信用 1 100.00
"""
    result = independently_reparse_plain_pages(
        [page], observation_month="2024-02", code="9399"
    )
    assert result["record_count"] == 1
    assert result["unique_record"]["foreign_marker"]
    assert not result["unique_record"]["domestic_continuity"]
