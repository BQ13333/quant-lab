from __future__ import annotations

from io import BytesIO

import pandas as pd
import pytest

from quant_lab.data.universe_mining import (
    classify_us_security,
    download_mined_universe_prices,
    mine_asia_listings_resilient,
    mine_us_listings_from_nasdaq_trader,
    parse_hkex_securities,
    parse_jpx_listed,
    parse_krx_kind_listed,
    parse_nasdaq_listed,
    parse_other_listed,
    parse_hnx_listed,
    parse_pse_company_directory,
    parse_set_listed,
    parse_sgx_securities,
    parse_sgx_stock_screener,
    parse_sse_listed,
    parse_szse_listed,
    parse_tpex_listed,
    parse_twse_listed,
)


NASDAQ_FIXTURE = """Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares
AAPL|Apple Inc. - Common Stock|Q|N|N|100|N|N
QQQ|Invesco QQQ Trust|G|N|N|100|Y|N
ZZTEST|Test Company|S|Y|N|100|N|N
File Creation Time: 0705202612:00
"""


OTHER_FIXTURE = """ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol
IBM|International Business Machines Corporation Common Stock|N|IBM|N|100|N|IBM
SPY|SPDR S&P 500 ETF Trust|P|SPY|Y|100|N|SPY
BRK.B|Berkshire Hathaway Inc. Class B|N|BRK.B|N|100|N|BRK.B
"""


def test_parse_nasdaq_listed_fixture():
    frame = parse_nasdaq_listed(NASDAQ_FIXTURE)

    assert frame.shape[0] == 3
    assert frame.loc[frame["ticker"] == "AAPL", "exchange"].iloc[0] == "NASDAQ Global Select"
    assert bool(frame.loc[frame["ticker"] == "QQQ", "is_etf"].iloc[0]) is True
    assert frame.loc[frame["ticker"] == "QQQ", "security_type"].iloc[0] == "ETF"
    assert bool(frame.loc[frame["ticker"] == "ZZTEST", "is_test_issue"].iloc[0]) is True


def test_parse_other_listed_fixture_yahoo_symbol_normalization():
    frame = parse_other_listed(OTHER_FIXTURE)

    assert frame.shape[0] == 3
    assert frame.loc[frame["ticker"] == "IBM", "exchange"].iloc[0] == "NYSE"
    assert frame.loc[frame["ticker"] == "BRK.B", "yahoo_ticker"].iloc[0] == "BRK-B"


def test_mine_us_listings_filters_etfs_and_test_issues():
    frame = mine_us_listings_from_nasdaq_trader(
        include_etfs=False,
        include_test_issues=False,
        nasdaq_text=NASDAQ_FIXTURE,
        other_text=OTHER_FIXTURE,
    )

    assert set(frame["ticker"]) == {"AAPL", "BRK.B", "IBM"}
    assert not frame["is_etf"].any()
    assert not frame["is_test_issue"].any()
    assert frame["is_stock_like"].all()


def test_security_classifier_excludes_non_stock_like_instruments():
    assert classify_us_security("Example Corp. Common Stock", False) == "Common Stock"
    assert classify_us_security("Example plc American Depositary Shares", False) == "ADR"
    assert classify_us_security("Example Acquisition Corp. - Warrant", False) == "Warrant"
    assert classify_us_security("Example Acquisition Corp. - Units", False) == "Unit"
    assert classify_us_security("Example Volatility ETN", False) == "ETN"
    assert classify_us_security("Example ETF", True) == "ETF"


def test_download_mined_universe_prices_rejects_empty_listing(tmp_path):
    with pytest.raises(ValueError):
        download_mined_universe_prices(pd.DataFrame(), tmp_path / "prices.parquet")


def _xlsx_bytes(frame: pd.DataFrame, startrow: int = 0) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, startrow=startrow)
    return buffer.getvalue()


def test_parse_twse_and_tpex_fixtures():
    twse = '[{"公司代號":"2330","公司名稱":"台積電","公司簡稱":"台積電","產業別":"24","上市日期":"19940905"}]'
    tpex = '[{"SecuritiesCompanyCode":"6488","CompanyName":"環球晶圓","CompanyAbbreviation":"環球晶","SecuritiesIndustryCode":"24","DateOfListing":"20150925"}]'

    twse_frame = parse_twse_listed(twse)
    tpex_frame = parse_tpex_listed(tpex)

    assert twse_frame.loc[0, "yahoo_ticker"] == "2330.TW"
    assert tpex_frame.loc[0, "yahoo_ticker"] == "6488.TWO"
    assert twse_frame.loc[0, "country"] == "Taiwan"


def test_parse_krx_kind_fixture():
    html = """
    <table>
      <tr><th>회사명</th><th>시장구분</th><th>종목코드</th><th>업종</th><th>상장일</th></tr>
      <tr><td>삼성전자</td><td>유가증권</td><td>005930</td><td>전자</td><td>1975-06-11</td></tr>
      <tr><td>현대차</td><td>KRX 유가</td><td>005380</td><td>운수장비</td><td>1974-06-28</td></tr>
      <tr><td>카카오게임즈</td><td>코스닥</td><td>293490</td><td>소프트웨어</td><td>2020-09-10</td></tr>
      <tr><td>툴젠</td><td>코넥스</td><td>199800</td><td>바이오</td><td>2014-06-25</td></tr>
    </table>
    """
    frame = parse_krx_kind_listed(html)

    assert set(frame["yahoo_ticker"].dropna()) == {"005930.KS", "005380.KS", "293490.KQ"}
    assert frame.loc[frame["ticker"] == "199800", "yahoo_ticker"].isna().all()
    assert frame["is_stock_like"].all()


def test_parse_hkex_fixture():
    data = pd.DataFrame(
        [
            {"Stock Code": 1, "Name of Securities": "CKH HOLDINGS", "Category": "Equity", "Sub-Category": "Equity Securities (Main Board)", "Trading Currency": "HKD", "ISIN": "KYG217651051"},
            {"Stock Code": 823, "Name of Securities": "LINK REIT", "Category": "Real Estate Investment Trusts", "Sub-Category": "REIT", "Trading Currency": "HKD", "ISIN": "HK0823032773"},
            {"Stock Code": 2800, "Name of Securities": "TRACKER FUND", "Category": "Exchange Traded Products", "Sub-Category": "Exchange Traded Funds", "Trading Currency": "HKD", "ISIN": "HK2820000007"},
        ]
    )
    frame = parse_hkex_securities(_xlsx_bytes(data, startrow=2))

    assert set(frame["yahoo_ticker"]) == {"0001.HK", "0823.HK"}
    assert "2800.HK" not in set(frame["yahoo_ticker"])


def test_parse_jpx_and_szse_fixtures():
    jpx = pd.DataFrame(
        [
            {"コード": 7203, "銘柄名": "トヨタ自動車", "市場・商品区分": "プライム（内国株式）", "33業種区分": "輸送用機器"},
            {"コード": 1306, "銘柄名": "TOPIX ETF", "市場・商品区分": "ETF・ETN", "33業種区分": "-"},
        ]
    )
    szse = pd.DataFrame(
        [
            {"A股代码": 1, "A股简称": "平安银行", "公司全称": "平安银行股份有限公司", "板块": "主板", "所属行业": "J 金融业", "A股上市日期": "1991-04-03"}
        ]
    )

    jpx_frame = parse_jpx_listed(_xlsx_bytes(jpx))
    szse_frame = parse_szse_listed(_xlsx_bytes(szse))

    assert jpx_frame["yahoo_ticker"].tolist() == ["7203.T"]
    assert szse_frame["yahoo_ticker"].tolist() == ["000001.SZ"]


def test_parse_sse_fixture():
    payload = {
        "pageHelp": {
            "data": [
                {"SECURITY_CODE_A": "600000", "SECURITY_ABBR_A": "浦发银行", "COMPANY_ABBR": "浦发银行", "LISTING_BOARD": "0", "LISTING_DATE": "1999-11-10"}
            ]
        }
    }
    frame = parse_sse_listed(f"jsonpCallback({__import__('json').dumps(payload)})")

    assert frame.loc[0, "yahoo_ticker"] == "600000.SS"
    assert frame.loc[0, "exchange"] == "SSE"


def test_mine_asia_listings_resilient_logs_failed_source(monkeypatch):
    import quant_lab.data.universe_mining as universe_mining

    def fake_mine_asia_listings(sources):
        source = sources[0]
        if source == "sse":
            raise TimeoutError("slow endpoint")
        return (
            pd.DataFrame(
                {
                    "ticker": ["7203"],
                    "yahoo_ticker": ["7203.T"],
                    "name": ["Toyota"],
                    "exchange": ["JPX"],
                    "country": ["Japan"],
                    "currency": ["JPY"],
                }
            ),
            {"jpx_listed.xls": b"raw"},
        )

    monkeypatch.setattr(universe_mining, "mine_asia_listings", fake_mine_asia_listings)

    listings, raw_files, errors = mine_asia_listings_resilient(("jpx", "sse"))

    assert listings["yahoo_ticker"].tolist() == ["7203.T"]
    assert raw_files == {"jpx_listed.xls": b"raw"}
    assert errors.loc[0, "source"] == "sse"
    assert errors.loc[0, "error_type"] == "TimeoutError"


def test_parse_sgx_stock_screener_fixture():
    payload = {
        "meta": {"totalItems": 2},
        "data": [
            {"exchange": "SES", "exchangeCountryCode": "SGP", "companyName": "DBS Group Holdings Ltd", "stockCode": "D05", "sector": "Financial Services", "priceCurrCode": "SGD", "marketCapitalization": "189680000000"},
            {"exchange": "SES", "exchangeCountryCode": "SGP", "companyName": "Mapletree Logistics Trust", "stockCode": "M44U", "sector": "Real Estate", "priceCurrCode": "SGD", "marketCapitalization": "7000000000"},
        ],
    }

    frame = parse_sgx_stock_screener(__import__("json").dumps(payload))

    assert set(frame["yahoo_ticker"]) == {"D05.SI", "M44U.SI"}
    assert frame["exchange"].eq("SGX").all()
    assert frame.loc[frame["ticker"] == "D05", "market_cap"].iloc[0] == 189680000000
    assert frame.loc[frame["ticker"] == "M44U", "security_type"].iloc[0] == "Common Stock"
    assert frame["country"].eq("Singapore").all()


def test_parse_sgx_securities_fixture():
    payload = {
        "meta": {"code": "200"},
        "data": {
            "prices": [
                {"nc": "D05", "n": "DBS", "type": "stocks", "ls": "P", "m": "MAINBOARD", "sc": "F", "bl": "100", "cur": "SGD"},
                {"nc": "CLR", "n": "Lion-OCBC Securities China Leaders ETF", "type": "etfs", "ls": "P", "m": "ETF", "sc": "", "bl": "1", "cur": "SGD"},
                {"nc": None, "n": None, "type": "listedcertificates"},
            ]
        },
    }

    frame = parse_sgx_securities(__import__("json").dumps(payload))

    assert set(frame["yahoo_ticker"]) == {"D05.SI", "CLR.SI"}
    assert frame.loc[frame["ticker"] == "D05", "market_segment"].iloc[0] == "MAINBOARD"
    assert frame.loc[frame["ticker"] == "CLR", "security_type"].iloc[0] == "ETF"
    assert bool(frame.loc[frame["ticker"] == "D05", "is_stock_like"].iloc[0]) is True


def test_parse_set_listed_fixture():
    data = pd.DataFrame(
        [
            {"Symbol": "PTT", "Company Name": "PTT Public Company Limited", "Market": "SET", "Sector": "Energy & Utilities", "Listing Date": "2001-12-06"},
            {"Symbol": "AOT", "Company Name": "Airports of Thailand Public Company Limited", "Market": "SET", "Sector": "Transportation & Logistics", "Listing Date": "2004-03-11"},
        ]
    )

    frame = parse_set_listed(_xlsx_bytes(data))

    assert frame["yahoo_ticker"].tolist() == ["AOT.BK", "PTT.BK"]
    assert frame["country"].eq("Thailand").all()


def test_parse_pse_company_directory_fixture():
    html = """
    <span class="count">[1 / 1] [Total 2]</span>
    <table class="list">
      <thead><tr><th>Company Name</th><th>Stock Symbol</th><th>Sector</th><th>Subsector</th><th>Listing Date</th></tr></thead>
      <tbody>
        <tr><td>Ayala Corporation</td><td>AC</td><td>Holding Firms</td><td>Holding Firms</td><td>Nov 08, 1976</td></tr>
        <tr><td>First Metro Philippine Equity Exchange Traded Fund, Inc.</td><td>FMETF</td><td>ETF</td><td>ETF-Equity</td><td>Dec 02, 2013</td></tr>
      </tbody>
    </table>
    """

    frame = parse_pse_company_directory(html)

    assert set(frame["yahoo_ticker"]) == {"AC.PS", "FMETF.PS"}
    assert frame.loc[frame["ticker"] == "FMETF", "security_type"].iloc[0] == "ETF"
    assert bool(frame.loc[frame["ticker"] == "AC", "is_stock_like"].iloc[0]) is True


def test_parse_hnx_listed_fixture():
    content = """
    <table>
      <thead><tr><th>STT</th><th>Mã CK</th><th>Tên tổ chức phát hành</th><th>Ngành</th><th>Ngày GD đầu tiên</th><th>Khối lượng NY</th><th>KLLH (Cổ phiếu)</th></tr></thead>
      <tbody>
        <tr><td>1</td><td>ADC</td><td>Công ty cổ phần Mĩ thuật và Truyền thông</td><td>Thông tin</td><td>24/12/2010</td><td>3.977.936</td><td>3.977.936</td></tr>
      </tbody>
    </table>
    """
    payload = {"Content": content, "Total": "Total 1 Record(s)"}

    frame = parse_hnx_listed(__import__("json").dumps(payload), exchange="HNX")

    assert frame.loc[0, "ticker"] == "ADC"
    assert frame.loc[0, "yahoo_ticker"] == "ADC.VN"
    assert frame.loc[0, "listed_shares"] == 3977936
