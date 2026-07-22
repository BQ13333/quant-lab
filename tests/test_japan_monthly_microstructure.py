from __future__ import annotations

from quant_lab.data.japan_monthly_microstructure import (
    _preferred_unit_source,
    parse_ose_lot_size_text,
    parse_tse_lot_size_text,
)


def test_tse_legacy_parser_reads_compact_and_spaced_units() -> None:
    text = """
8001 伊藤忠商事 普通株式
ITOCHU Corporation
Wholesale Trade
1貸借1,000 975(1) 996(6)
8005 ムトウ 普通株式
Mutow Co.,Ltd.
Retail Trade
1貸借 100 544(1) 560(6)
"""
    result = parse_tse_lot_size_text(text).set_index("code")
    assert result.loc["8001", "trading_unit"] == 1000
    assert result.loc["8005", "trading_unit"] == 100


def test_tse_combined_parser_reads_decimal_and_untraded_rows() -> None:
    text = """
2020/03 1301 極洋 普通株式 KYOKUYO Fishery 1 貸借 100 2,562.00 2 2,757.00 21
2020/03 1432 動力 普通株式 DORYOKU Construction TPM 100 － － －
2024/09 143A イシン 普通株式 Ishin Services G 信用 10 0 855.00 2 1,916.00 19
2024/09 161A 三葉 普通株式 Mitsuba Services TPM 1 00 － － －
2024/09 1451 ＫＨＣ 普通株式 KHC Construction S 確 信用 100 1, 124.00 2 1,127.00 19
2025/09 3317 フライングガーデン 普通株式 FLYING GARDEN S 信用 100 2,823 .00 1 3,025.00 18
2025/09 3317 フライングガーデン 普通株式 FLYING GARDEN S 信用 100 1,501 .00 29 1,531.00 2
"""
    result = parse_tse_lot_size_text(text).set_index("code")
    assert result.loc["1301", "trading_unit"] == 100
    assert result.loc["1432", "trading_unit"] == 100
    assert result.loc["143A", "trading_unit"] == 100
    assert result.loc["161A", "trading_unit"] == 100
    assert result.loc["1451", "trading_unit"] == 100
    assert result.loc["3317", "trading_unit"] == 100
    assert result.reset_index().loc[lambda frame: frame.code.eq("3317"), "trading_unit"].unique().tolist() == [100]


def test_ose_parser_reads_jasdaq_and_ose_layouts() -> None:
    text = """
Closing Price and Trading Volume of Issues
100 5 1376 カネコ種苗
第一部 1332 日本水産 普通株式 水産・農林業
NIPPON SUISAN Fishery
* 100 08 280 08 280
スタンダード 1380 秋川牧園 普通株式 水産・農林業
AKIKAWA FOODS & FARMS
# 1000 06 387 13 394
第一部 8591 オリックス 普通株式 その他金融業
ORIX Other Financing Business
*1 0 04 29070 18 24800
第二部 9636 きんえい 普通株式 サービス業
Kin-Ei Services
# 2000 03 394 06 401
スタンダード 3264 アスコット 普通株式 不動産業
ASCOT Real Estate
# 100 □ 02 86 16 109
"""
    result = parse_ose_lot_size_text(text).set_index("code")
    assert result.loc["1376", "trading_unit"] == 100
    assert result.loc["1332", "trading_unit"] == 100
    assert result.loc["1380", "trading_unit"] == 1000
    assert result.loc["8591", "trading_unit"] == 10
    assert result.loc["9636", "trading_unit"] == 2000
    assert result.loc["3264", "trading_unit"] == 100


def test_post_integration_ose_does_not_apply_legacy_row_parser() -> None:
    text = """
第一部 1980 ダイダン 普通株式 建設業
DAI-DAN Construction
* 1000 01 457 07 467
100 03 1980 unrelated numeric row
"""
    result = parse_ose_lot_size_text(text)
    assert result.loc[result.code.eq("1980"), "trading_unit"].tolist() == [1000]


def test_tse_legacy_parser_supports_prestandardization_units() -> None:
    text = """
9704 東海観光 普通株式
TOKAI KANKO
Services
1信用3,000 42(3) 46(3)
9708 帝国ホテル 普通株式
Imperial Hotel
Services
2 50 5,040(3) 5,190(11)
4589 アキュセラ・インク 普通株式
Acucela Inc.
Pharmaceutical
M外 信用 100 2,269 (3) 2,271 (5)
4875 メディシノバ・インク 普通株式
MediciNova,Inc.
Pharmaceutical
JQS外 信用 100 1,206.00 (1) 1,215.00 (16)
"""
    result = parse_tse_lot_size_text(text).set_index("code")
    assert result.loc["9704", "trading_unit"] == 3000
    assert result.loc["9708", "trading_unit"] == 50
    assert result.loc["4589", "trading_unit"] == 100
    assert result.loc["4875", "trading_unit"] == 100


def test_membership_source_selects_venue_specific_unit() -> None:
    assert _preferred_unit_source("OSE_COMBINED|TSE") == "TSE"
    assert _preferred_unit_source("JASDAQ") == "JASDAQ_OSE"
    assert _preferred_unit_source("OSE_MAIN") == "OSE_MAIN"
