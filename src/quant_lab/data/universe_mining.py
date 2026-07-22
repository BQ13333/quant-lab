from __future__ import annotations

import json
import re
import ssl
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from http.client import IncompleteRead
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

import pandas as pd

from quant_lab.data.yfinance_loader import DownloadResult, download_price_history


ListingSource = Literal["nasdaq_trader", "asean_seed"]


NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
NASDAQ_OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"
TWSE_LISTED_URL = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
TPEX_LISTED_URL = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
KRX_KIND_LISTED_URL = "https://kind.krx.co.kr/corpgeneral/corpList.do?method=download&searchType=13"
HKEX_SECURITIES_URL = "https://www.hkex.com.hk/eng/services/trading/securities/securitieslists/ListOfSecurities.xlsx"
JPX_LISTED_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
SZSE_LISTED_URL = "https://www.szse.cn/api/report/ShowReport?SHOWTYPE=xlsx&CATALOGID=1110&TABKEY=tab1&random=0.713"
SSE_LISTED_URL = (
    "https://query.sse.com.cn/security/stock/getStockListData2.do"
    "?jsonCallBack=jsonpCallback&isPagination=true&stockCode=&csrcCode=&areaName=&stockType=1"
    "&pageHelp.cacheSize=1&pageHelp.beginPage=1&pageHelp.pageSize=5000&pageHelp.pageNo=1&_=1670000000000"
)
SGX_STOCK_SCREENER_URL = "https://api.sgx.com/stockscreener/v2.0/all"
SGX_SECURITIES_URL = "https://api.sgx.com/securities/v1.1"
SGX_SECURITIES_PARAMS = "nc,n,type,ls,m,sc,bl,sip,ex,ej,clo,cr,cur,el,r,i,cc,ig,lf"
SGX_STOCK_SCREENER_PARAMS = (
    "exchange,exchangeCountryCode,companyName,stockCode,marketCapitalization,salesTTM,"
    "priceToEarningsRatio,dividendYield,fourWeekPricePercentChange,thirteenWeekPricePercentChange,"
    "twentySixWeekPricePercentChange,fiftyTwoWeekPricePercentChange,netProfitMargin,"
    "returnOnAvgCommonEquity,priceToCashFlowPerShareRatio,totalDebtToTotalEquityRatio,"
    "salesPercentageChange,sector,priceToBookRatio,priceCurrCode"
)
SET_LISTED_COMPANIES_URL = "https://www.set.or.th/dat/eod/listedcompany/static/listedCompanies_en_US.xls"
PSE_COMPANY_DIRECTORY_URL = "https://edge.pse.com.ph/companyDirectory/search.ax"
HNX_LISTED_SEARCH_URL = "https://hnx.vn/ModuleIssuer/List/ListSearch_Datas"
UPCOM_LISTED_SEARCH_URL = "https://hnx.vn/ModuleIssuer/UC_Issuer/ListSearch_Datas"

US_EXCHANGE_CODES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Q": "NASDAQ Global Select",
    "G": "NASDAQ Global",
    "S": "NASDAQ Capital",
    "V": "IEX",
    "Z": "Cboe BZX",
}


@dataclass(frozen=True)
class MinedUniverseResult:
    run_id: str
    artifact_dir: Path
    listings: pd.DataFrame
    source_files: dict[str, Path]


def _now_run_id(prefix: str = "UNIVERSE") -> str:
    return datetime.now(timezone.utc).strftime(f"{prefix}-%Y%m%d-%H%M%S")


_HTTPS_CONTEXT: ssl.SSLContext | None = None
TLS_VERIFY_FALLBACK_HOSTS = {"hnx.vn", "www.hnx.vn"}


def _https_context() -> ssl.SSLContext | None:
    global _HTTPS_CONTEXT
    if _HTTPS_CONTEXT is not None:
        return _HTTPS_CONTEXT
    try:
        import certifi

        _HTTPS_CONTEXT = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        _HTTPS_CONTEXT = None
    return _HTTPS_CONTEXT


def _urlopen(request: Request, timeout: int):
    context = _https_context() if request.full_url.startswith("https://") else None
    try:
        if context is not None:
            return urlopen(request, timeout=timeout, context=context)
        return urlopen(request, timeout=timeout)
    except URLError as exc:
        host = urlparse(request.full_url).hostname or ""
        reason = getattr(exc, "reason", exc)
        if host in TLS_VERIFY_FALLBACK_HOSTS and isinstance(reason, ssl.SSLCertVerificationError):
            return urlopen(request, timeout=timeout, context=ssl._create_unverified_context())
        raise


def _read_response_bytes(response, max_read_seconds: float = 45.0, chunk_size: int = 1_048_576) -> bytes:
    deadline = time.monotonic() + max_read_seconds
    chunks: list[bytes] = []
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"response read exceeded {max_read_seconds:.1f}s")
        chunk = response.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _fetch_text(url: str, timeout: int = 30, max_read_seconds: float = 45.0) -> str:
    request = Request(url, headers={"User-Agent": "quant-lab-universe-miner/0.1"})
    with _urlopen(request, timeout=timeout) as response:
        return _read_response_bytes(response, max_read_seconds=max_read_seconds).decode("utf-8", errors="replace")


def _fetch_bytes(
    url: str,
    timeout: int = 30,
    referer: str | None = None,
    retries: int = 3,
    max_read_seconds: float = 45.0,
) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 quant-lab-universe-miner/0.1",
        "Accept": "*/*",
    }
    if referer is not None:
        headers["Referer"] = referer
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers=headers)
            with _urlopen(request, timeout=timeout) as response:
                return _read_response_bytes(response, max_read_seconds=max_read_seconds)
        except IncompleteRead as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        time.sleep(0.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to fetch {url}")


def _fetch_bytes_with_params(
    url: str,
    params: dict[str, object],
    timeout: int = 30,
    referer: str | None = None,
) -> bytes:
    query = urlencode(params)
    separator = "&" if "?" in url else "?"
    return _fetch_bytes(f"{url}{separator}{query}", timeout=timeout, referer=referer)


def _post_form_bytes(
    url: str,
    form: dict[str, object],
    timeout: int = 30,
    referer: str | None = None,
    retries: int = 3,
    max_read_seconds: float = 45.0,
) -> bytes:
    headers = {
        "User-Agent": "Mozilla/5.0 quant-lab-universe-miner/0.1",
        "Accept": "*/*",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        "X-Requested-With": "XMLHttpRequest",
    }
    if referer is not None:
        headers["Referer"] = referer
    data = urlencode(form).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, data=data, headers=headers, method="POST")
            with _urlopen(request, timeout=timeout) as response:
                return _read_response_bytes(response, max_read_seconds=max_read_seconds)
        except IncompleteRead as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
        time.sleep(0.5 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"failed to post {url}")


def _read_pipe_table(text: str) -> pd.DataFrame:
    lines = [line for line in text.splitlines() if line and not line.startswith("File Creation Time")]
    if not lines:
        return pd.DataFrame()
    return pd.read_csv(StringIO("\n".join(lines)), sep="|")


def _read_html_tables(html_text: str) -> list[pd.DataFrame]:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    frames: list[pd.DataFrame] = []
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        if not trs:
            continue
        header_row = next((tr for tr in trs if tr.find_all("th")), trs[0])
        headers = [cell.get_text(" ", strip=True) for cell in header_row.find_all(["th", "td"])]
        if not headers:
            continue
        rows = []
        for tr in trs[trs.index(header_row) + 1:]:
            cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["td", "th"])]
            if not cells or len(cells) != len(headers):
                continue
            rows.append(cells)
        frames.append(pd.DataFrame(rows, columns=headers))
    return frames


def _normalize_us_symbol(symbol: str) -> str:
    """Convert Nasdaq Trader symbols into Yahoo-compatible tickers when possible."""

    return str(symbol).strip().replace("/", "-").replace(".", "-")


def _clean_code(value: object, width: int | None = None) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    if width is not None and text.isdigit():
        text = text.zfill(width)
    return text


def _clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unescape(str(value))).strip()


def _base_listing_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["mined_at_utc"] = datetime.now(timezone.utc).isoformat()
    return frame.sort_values(["country", "exchange", "ticker"]).reset_index(drop=True)


def _first_existing(row: pd.Series, names: tuple[str, ...]) -> object:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return None


def classify_us_security(name: str, is_etf: bool) -> str:
    """Classify a US listing from its security-name text."""

    text = str(name).lower()
    if is_etf:
        return "ETF"
    if " etn" in text or "exchange traded note" in text:
        return "ETN"
    if "warrant" in text:
        return "Warrant"
    if " right" in text or "rights" in text:
        return "Right"
    if " unit" in text or "units" in text:
        return "Unit"
    if "preferred" in text or "preference" in text:
        return "Preferred"
    if "bond" in text or "note" in text or "debenture" in text:
        return "Debt"
    if "closed end fund" in text or "closed-end fund" in text or "fund" in text:
        return "Fund"
    if "depositary" in text or "american depositary" in text or "american depository" in text:
        return "ADR"
    if "ordinary share" in text or "ordinary shares" in text:
        return "Ordinary Share"
    if "common stock" in text or "common shares" in text or "class a common" in text or "class b common" in text:
        return "Common Stock"
    if "reit" in text:
        return "REIT"
    return "Other Equity"


def _is_stock_like(security_type: str) -> bool:
    return security_type in {"Common Stock", "Ordinary Share", "ADR", "REIT", "Other Equity"}


def _with_security_classification(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["security_type"] = [
        classify_us_security(name, is_etf)
        for name, is_etf in zip(frame["name"], frame["is_etf"])
    ]
    frame["asset_type"] = frame["security_type"].map(
        lambda security_type: "Fund" if security_type in {"ETF", "Fund"} else (
            "Structured Product" if security_type in {"ETN", "Warrant", "Right", "Unit", "Debt"} else "Equity"
        )
    )
    frame["is_stock_like"] = frame["security_type"].map(_is_stock_like)
    return frame


def parse_nasdaq_listed(text: str) -> pd.DataFrame:
    """Parse Nasdaq-listed securities from Nasdaq Trader's symbol file."""

    raw = _read_pipe_table(text)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.rename(
        columns={
            "Symbol": "ticker",
            "Security Name": "name",
            "Market Category": "exchange_code",
            "Test Issue": "test_issue",
            "ETF": "is_etf",
            "Round Lot Size": "round_lot_size",
        }
    )
    frame = pd.DataFrame(
        {
            "ticker": raw["ticker"].astype(str),
            "yahoo_ticker": raw["ticker"].map(_normalize_us_symbol),
            "name": raw["name"].astype(str),
            "exchange": raw["exchange_code"].map(US_EXCHANGE_CODES).fillna("NASDAQ"),
            "exchange_code": raw["exchange_code"],
            "country": "United States",
            "currency": "USD",
            "is_etf": raw["is_etf"].eq("Y"),
            "is_test_issue": raw["test_issue"].eq("Y"),
            "round_lot_size": pd.to_numeric(raw.get("round_lot_size"), errors="coerce"),
            "source": "nasdaq_trader",
        }
    )
    return _with_security_classification(frame)


def parse_other_listed(text: str) -> pd.DataFrame:
    """Parse NYSE/NYSE Arca/Cboe/IEX securities from Nasdaq Trader's other-listed file."""

    raw = _read_pipe_table(text)
    if raw.empty:
        return pd.DataFrame()
    raw = raw.rename(
        columns={
            "ACT Symbol": "ticker",
            "Security Name": "name",
            "Exchange": "exchange_code",
            "Test Issue": "test_issue",
            "ETF": "is_etf",
            "Round Lot Size": "round_lot_size",
        }
    )
    frame = pd.DataFrame(
        {
            "ticker": raw["ticker"].astype(str),
            "yahoo_ticker": raw["ticker"].map(_normalize_us_symbol),
            "name": raw["name"].astype(str),
            "exchange": raw["exchange_code"].map(US_EXCHANGE_CODES).fillna(raw["exchange_code"]),
            "exchange_code": raw["exchange_code"],
            "country": "United States",
            "currency": "USD",
            "is_etf": raw["is_etf"].eq("Y"),
            "is_test_issue": raw["test_issue"].eq("Y"),
            "round_lot_size": pd.to_numeric(raw.get("round_lot_size"), errors="coerce"),
            "source": "nasdaq_trader",
        }
    )
    return _with_security_classification(frame)


def mine_us_listings_from_nasdaq_trader(
    include_etfs: bool = False,
    stock_like_only: bool = True,
    include_test_issues: bool = False,
    nasdaq_text: str | None = None,
    other_text: str | None = None,
) -> pd.DataFrame:
    """Mine current US listed securities from Nasdaq Trader symbol files."""

    nasdaq_text = nasdaq_text if nasdaq_text is not None else _fetch_text(NASDAQ_LISTED_URL)
    other_text = other_text if other_text is not None else _fetch_text(NASDAQ_OTHER_LISTED_URL)
    frames = [parse_nasdaq_listed(nasdaq_text), parse_other_listed(other_text)]
    listings = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True)
    if listings.empty:
        return listings

    listings = listings.drop_duplicates(subset=["ticker", "exchange"], keep="first")
    if not include_etfs:
        listings = listings[~listings["is_etf"]]
    if stock_like_only:
        listings = listings[listings["is_stock_like"]]
    if not include_test_issues:
        listings = listings[~listings["is_test_issue"]]

    listings = listings.sort_values(["exchange", "ticker"]).reset_index(drop=True)
    listings["mined_at_utc"] = datetime.now(timezone.utc).isoformat()
    return listings


def save_mined_us_universe(
    artifact_root: str | Path = "artifacts/universes",
    include_etfs: bool = False,
    stock_like_only: bool = True,
    include_test_issues: bool = False,
    run_id: str | None = None,
) -> MinedUniverseResult:
    """Fetch and save a current US listed-security universe snapshot."""

    run_id = run_id or _now_run_id("US-LISTINGS")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    nasdaq_text = _fetch_text(NASDAQ_LISTED_URL)
    other_text = _fetch_text(NASDAQ_OTHER_LISTED_URL)
    source_files = {
        "nasdaqlisted": artifact_dir / "nasdaqlisted.txt",
        "otherlisted": artifact_dir / "otherlisted.txt",
    }
    source_files["nasdaqlisted"].write_text(nasdaq_text, encoding="utf-8")
    source_files["otherlisted"].write_text(other_text, encoding="utf-8")

    listings = mine_us_listings_from_nasdaq_trader(
        include_etfs=include_etfs,
        stock_like_only=stock_like_only,
        include_test_issues=include_test_issues,
        nasdaq_text=nasdaq_text,
        other_text=other_text,
    )
    listings.to_csv(artifact_dir / "listings.csv", index=False)

    manifest = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "source": "nasdaq_trader",
                "nasdaq_url": NASDAQ_LISTED_URL,
                "other_url": NASDAQ_OTHER_LISTED_URL,
                "include_etfs": include_etfs,
                "stock_like_only": stock_like_only,
                "include_test_issues": include_test_issues,
                "rows": len(listings),
                "mined_at_utc": datetime.now(timezone.utc).isoformat(),
                "caveat": "Current listed securities only; not a point-in-time or delisted-security database.",
            }
        ]
    )
    manifest.to_csv(artifact_dir / "manifest.csv", index=False)

    return MinedUniverseResult(
        run_id=run_id,
        artifact_dir=artifact_dir,
        listings=listings,
        source_files=source_files,
    )


def download_mined_universe_prices(
    listings: pd.DataFrame,
    output_path: str | Path,
    period: str = "5y",
    interval: str = "1d",
    max_tickers: int | None = None,
    fallback_periods: list[str] | None = None,
) -> DownloadResult:
    """Download price history for a mined listing table using Yahoo tickers."""

    if listings.empty:
        raise ValueError("listings must not be empty")
    ticker_col = "yahoo_ticker" if "yahoo_ticker" in listings.columns else "ticker"
    tickers = [ticker for ticker in listings[ticker_col].dropna().astype(str).tolist() if ticker]
    if max_tickers is not None:
        tickers = tickers[:max_tickers]
    if not tickers:
        raise ValueError("no tickers available for price download")

    return download_price_history(
        tickers=tickers,
        period=period,
        interval=interval,
        output_path=output_path,
        fallback_periods=fallback_periods or ["3y", "1y", "6mo"],
        ohlcv=True,
    )


# ---------------------------------------------------------------------------
# Asia / global exchange adapters
# ---------------------------------------------------------------------------


def parse_twse_listed(json_text: str) -> pd.DataFrame:
    data = json.loads(json_text)
    rows = []
    for item in data:
        code = _clean_code(item.get("公司代號"), width=4)
        if not code:
            continue
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.TW",
                "name": item.get("公司名稱") or item.get("公司簡稱") or code,
                "short_name": item.get("公司簡稱"),
                "exchange": "TWSE",
                "country": "Taiwan",
                "currency": "TWD",
                "security_type": "Common Stock",
                "asset_type": "Equity",
                "is_stock_like": True,
                "industry_code": item.get("產業別"),
                "listing_date": item.get("上市日期"),
                "source": "twse_openapi",
            }
        )
    return _base_listing_frame(rows)


def parse_tpex_listed(json_text: str) -> pd.DataFrame:
    data = json.loads(json_text)
    rows = []
    for item in data:
        code = _clean_code(item.get("SecuritiesCompanyCode"), width=4)
        if not code:
            continue
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.TWO",
                "name": item.get("CompanyName") or item.get("CompanyAbbreviation") or code,
                "short_name": item.get("CompanyAbbreviation"),
                "exchange": "TPEx",
                "country": "Taiwan",
                "currency": "TWD",
                "security_type": "Common Stock",
                "asset_type": "Equity",
                "is_stock_like": True,
                "industry_code": item.get("SecuritiesIndustryCode"),
                "listing_date": item.get("DateOfListing"),
                "source": "tpex_openapi",
            }
        )
    return _base_listing_frame(rows)


def parse_krx_kind_listed(html_text: str) -> pd.DataFrame:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    table = soup.find("table")
    if table is None:
        return pd.DataFrame()
    rows = []
    trs = table.find_all("tr")
    if not trs:
        return pd.DataFrame()
    headers = [cell.get_text(strip=True) for cell in trs[0].find_all(["th", "td"])]
    for tr in trs[1:]:
        values = [cell.get_text(strip=True) for cell in tr.find_all(["td", "th"])]
        if len(values) != len(headers):
            continue
        item = dict(zip(headers, values))
        code = _clean_code(item.get("종목코드"), width=6)
        if not code:
            continue
        market = item.get("시장구분", "")
        suffix = _krx_yahoo_suffix(market)
        name = item.get("회사명", code)
        security_type = "SPAC" if "스팩" in name else "Common Stock"
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}{suffix}" if suffix else pd.NA,
                "name": name,
                "short_name": name,
                "exchange": f"KRX {market}".strip(),
                "country": "South Korea",
                "currency": "KRW",
                "security_type": security_type,
                "asset_type": "Equity",
                "is_stock_like": security_type == "Common Stock",
                "industry": item.get("업종"),
                "listing_date": item.get("상장일"),
                "source": "krx_kind",
            }
        )
    return _base_listing_frame(rows)


def _krx_yahoo_suffix(market: object) -> str:
    market_text = _clean_text(market)
    market_upper = market_text.upper()
    if "유가" in market_text or "KOSPI" in market_upper:
        return ".KS"
    if "코스닥" in market_text or "KOSDAQ" in market_upper:
        return ".KQ"
    if "코넥스" in market_text or "KONEX" in market_upper:
        return ""
    return ""


def parse_hkex_securities(xlsx_bytes: bytes, include_reits: bool = True) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(xlsx_bytes), header=2)
    rows = []
    for _, item in raw.iterrows():
        category = str(item.get("Category", "")).strip()
        sub_category = str(item.get("Sub-Category", "")).strip()
        if category == "Equity":
            security_type = "Common Stock"
        elif include_reits and category == "Real Estate Investment Trusts":
            security_type = "REIT"
        else:
            continue
        code = _clean_code(item.get("Stock Code"))
        if not code or not code.isdigit():
            continue
        yahoo_code = str(int(code)).zfill(4) if int(code) < 10000 else str(int(code))
        rows.append(
            {
                "ticker": code.zfill(5),
                "yahoo_ticker": f"{yahoo_code}.HK",
                "name": item.get("Name of Securities") or code,
                "short_name": item.get("Name of Securities"),
                "exchange": "HKEX",
                "country": "Hong Kong",
                "currency": item.get("Trading Currency") or "HKD",
                "security_type": security_type,
                "asset_type": "Equity",
                "is_stock_like": True,
                "category": category,
                "sub_category": sub_category,
                "isin": item.get("ISIN"),
                "source": "hkex_securities_list",
            }
        )
    return _base_listing_frame(rows)


def parse_jpx_listed(excel_bytes: bytes, include_reits: bool = False) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(excel_bytes))
    rows = []
    for _, item in raw.iterrows():
        market_type = str(item.get("市場・商品区分", "")).strip()
        is_stock = "株式" in market_type
        is_reit = "REIT" in market_type
        if not is_stock and not (include_reits and is_reit):
            continue
        code = _clean_code(item.get("コード"), width=4)
        if not code:
            continue
        security_type = "REIT" if is_reit else "Common Stock"
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.T",
                "name": item.get("銘柄名") or code,
                "short_name": item.get("銘柄名"),
                "exchange": "JPX",
                "country": "Japan",
                "currency": "JPY",
                "security_type": security_type,
                "asset_type": "Equity",
                "is_stock_like": True,
                "market_segment": market_type,
                "industry": item.get("33業種区分"),
                "source": "jpx_listed_issues",
            }
        )
    return _base_listing_frame(rows)


def parse_szse_listed(xlsx_bytes: bytes) -> pd.DataFrame:
    raw = pd.read_excel(BytesIO(xlsx_bytes))
    rows = []
    for _, item in raw.iterrows():
        code = _clean_code(item.get("A股代码"), width=6)
        if not code or code == "000000":
            continue
        name = item.get("A股简称") or item.get("公司全称") or code
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.SZ",
                "name": item.get("公司全称") or name,
                "short_name": name,
                "exchange": "SZSE",
                "country": "China",
                "currency": "CNY",
                "security_type": "Common Stock",
                "asset_type": "Equity",
                "is_stock_like": True,
                "market_segment": item.get("板块"),
                "industry": item.get("所属行业"),
                "listing_date": item.get("A股上市日期"),
                "source": "szse_report",
            }
        )
    return _base_listing_frame(rows)


def parse_sse_listed(jsonp_text: str) -> pd.DataFrame:
    match = re.search(r"jsonpCallback\((.*)\)\s*$", jsonp_text, flags=re.S)
    payload = match.group(1) if match else jsonp_text
    data = json.loads(payload)
    rows = []
    for item in data.get("pageHelp", {}).get("data", []):
        code = _clean_code(item.get("SECURITY_CODE_A"), width=6)
        if not code or code == "-":
            continue
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.SS",
                "name": item.get("COMPANY_ABBR") or item.get("SECURITY_ABBR_A") or code,
                "short_name": item.get("SECURITY_ABBR_A") or item.get("COMPANY_ABBR"),
                "exchange": "SSE",
                "country": "China",
                "currency": "CNY",
                "security_type": "Common Stock",
                "asset_type": "Equity",
                "is_stock_like": True,
                "market_segment": item.get("LISTING_BOARD"),
                "listing_date": item.get("LISTING_DATE"),
                "source": "sse_query",
            }
        )
    return _base_listing_frame(rows)


# ---------------------------------------------------------------------------
# ASEAN exchange adapters
# ---------------------------------------------------------------------------


def _sgx_security_type(name: str, sector: object) -> str:
    text = f"{name} {sector or ''}".lower()
    if "reit" in text or "real estate investment trust" in text:
        return "REIT"
    if "business trust" in text or "stapled" in text:
        return "Business Trust"
    if "etf" in text or "exchange traded fund" in text:
        return "ETF"
    return "Common Stock"


def _sgx_type_to_security_type(raw_type: object, name: str = "") -> str:
    text = f"{raw_type or ''} {name}".lower()
    if "reit" in text:
        return "REIT"
    if "trust" in text:
        return "Business Trust"
    if "etf" in text or "fund" in text:
        return "ETF"
    if "warrant" in text:
        return "Warrant"
    if "certificate" in text:
        return "Certificate"
    if "stock" in text or "share" in text or not text.strip():
        return "Common Stock"
    return str(raw_type or "Common Stock")


def parse_sgx_securities(json_text: str) -> pd.DataFrame:
    """Parse Singapore Exchange securities-master JSON."""

    payload = json.loads(json_text)
    data = payload.get("data", payload if isinstance(payload, list) else [])
    if isinstance(data, dict):
        data = data.get("prices", [])
    rows = []
    for item in data:
        code = _clean_code(item.get("nc"))
        name = _clean_text(item.get("n"))
        if not code or not name:
            continue
        raw_type = item.get("type")
        security_type = _sgx_type_to_security_type(raw_type, name=name)
        asset_type = "Fund" if security_type == "ETF" else (
            "Structured Product" if security_type in {"Warrant", "Certificate"} else "Equity"
        )
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.SI",
                "name": name,
                "short_name": name,
                "exchange": "SGX",
                "exchange_code": item.get("ex") or item.get("m"),
                "country": "Singapore",
                "currency": item.get("cur") or "SGD",
                "security_type": security_type,
                "asset_type": asset_type,
                "is_stock_like": security_type in {"Common Stock", "REIT", "Business Trust"},
                "market_segment": item.get("m"),
                "listing_status": item.get("ls"),
                "instrument_type": raw_type,
                "sector_code": item.get("sc"),
                "board_lot": pd.to_numeric(item.get("bl"), errors="coerce"),
                "source": "sgx_securities",
            }
        )
    return _base_listing_frame(rows)


def parse_sgx_stock_screener(json_text: str) -> pd.DataFrame:
    """Parse Singapore Exchange stock screener JSON."""

    payload = json.loads(json_text)
    data = payload.get("data", payload if isinstance(payload, list) else [])
    rows = []
    for item in data:
        code = _clean_code(item.get("stockCode"))
        name = _clean_text(item.get("companyName"))
        if not code or not name:
            continue
        sector = _clean_text(item.get("sector"))
        security_type = _sgx_security_type(name, sector)
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.SI",
                "name": name,
                "short_name": name,
                "exchange": "SGX",
                "exchange_code": item.get("exchange"),
                "country": "Singapore",
                "currency": item.get("priceCurrCode") or "SGD",
                "security_type": security_type,
                "asset_type": "Fund" if security_type == "ETF" else "Equity",
                "is_stock_like": security_type != "ETF",
                "sector": sector,
                "market_cap": pd.to_numeric(item.get("marketCapitalization"), errors="coerce"),
                "source": "sgx_stock_screener",
            }
        )
    return _base_listing_frame(rows)


def parse_set_listed(excel_bytes: bytes) -> pd.DataFrame:
    """Parse SET listed-company Excel exports when the official file is available."""

    raw = pd.read_excel(BytesIO(excel_bytes))
    raw = raw.dropna(how="all")
    rows = []
    for _, item in raw.iterrows():
        code = _clean_code(
            _first_existing(
                item,
                (
                    "Symbol",
                    "Security Symbol",
                    "หลักทรัพย์",
                    "ชื่อย่อหลักทรัพย์",
                    "Ticker",
                ),
            )
        )
        name = _clean_text(
            _first_existing(
                item,
                (
                    "Company Name",
                    "Company",
                    "Name",
                    "บริษัท",
                    "ชื่อบริษัท",
                    "Securities Name",
                ),
            )
        )
        if not code or not name:
            continue
        market = _clean_text(_first_existing(item, ("Market", "Exchange", "ตลาด"))) or "SET"
        sector = _clean_text(_first_existing(item, ("Sector", "Industry", "หมวดธุรกิจ")))
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.BK",
                "name": name,
                "short_name": name,
                "exchange": market,
                "country": "Thailand",
                "currency": "THB",
                "security_type": "Common Stock",
                "asset_type": "Equity",
                "is_stock_like": True,
                "sector": sector,
                "listing_date": _first_existing(item, ("Listing Date", "Listed Date", "วันที่เข้าซื้อขาย")),
                "source": "set_listed_companies_excel",
            }
        )
    return _base_listing_frame(rows)


def _parse_pse_company_ids(value: object) -> tuple[str, str]:
    match = re.search(r"cmDetail\('([^']+)','([^']+)'\)", str(value))
    if not match:
        return "", ""
    return match.group(1), match.group(2)


def parse_pse_company_directory(html_texts: str | list[str]) -> pd.DataFrame:
    """Parse PSE EDGE company-directory HTML pages."""

    if isinstance(html_texts, str):
        html_texts = [html_texts]
    rows = []
    for html_text in html_texts:
        tables = _read_html_tables(html_text)
        table = next((candidate for candidate in tables if "Stock Symbol" in candidate.columns), pd.DataFrame())
        if table.empty:
            continue
        for _, item in table.iterrows():
            code = _clean_code(item.get("Stock Symbol"))
            name = _clean_text(item.get("Company Name"))
            if not code or not name:
                continue
            company_id, security_id = _parse_pse_company_ids(item.get("Company Name", ""))
            sector = _clean_text(item.get("Sector"))
            security_type = "ETF" if sector.upper() == "ETF" else "Common Stock"
            rows.append(
                {
                    "ticker": code,
                    "yahoo_ticker": f"{code}.PS",
                    "name": name,
                    "short_name": name,
                    "exchange": "PSE",
                    "country": "Philippines",
                    "currency": "PHP",
                    "security_type": security_type,
                    "asset_type": "Fund" if security_type == "ETF" else "Equity",
                    "is_stock_like": security_type != "ETF",
                    "sector": sector,
                    "subsector": _clean_text(item.get("Subsector")),
                    "listing_date": item.get("Listing Date"),
                    "company_id": company_id,
                    "security_id": security_id,
                    "source": "pse_edge_company_directory",
                }
            )
    return _base_listing_frame(rows)


def _pse_total_pages(html_text: str) -> int:
    match = re.search(r"\[\s*\d+\s*/\s*(\d+)\s*\]", html_text)
    return int(match.group(1)) if match else 1


def _hnx_table_from_payload(json_text: str) -> pd.DataFrame:
    payload = json.loads(json_text)
    content = payload.get("Content", "")
    if not content:
        return pd.DataFrame()
    tables = _read_html_tables(content)
    return tables[0] if tables else pd.DataFrame()


def _parse_hnx_number(value: object) -> float:
    text = _clean_text(value).replace(".", "").replace(",", "")
    return pd.to_numeric(text, errors="coerce")


def parse_hnx_listed(json_text: str, exchange: str = "HNX") -> pd.DataFrame:
    """Parse HNX listed-stock or UPCoM AJAX payloads."""

    raw = _hnx_table_from_payload(json_text)
    if raw.empty:
        return pd.DataFrame()
    rows = []
    for _, item in raw.iterrows():
        code = _clean_code(_first_existing(item, ("Mã CK", "Listed code", "UPCoM code", "Stock code")))
        name = _clean_text(_first_existing(item, ("Tên tổ chức phát hành", "Issuers", "Issuer")))
        if not code or not name:
            continue
        rows.append(
            {
                "ticker": code,
                "yahoo_ticker": f"{code}.VN",
                "name": name,
                "short_name": name,
                "exchange": exchange,
                "country": "Vietnam",
                "currency": "VND",
                "security_type": "Common Stock",
                "asset_type": "Equity",
                "is_stock_like": True,
                "industry": _clean_text(_first_existing(item, ("Ngành", "Industry"))),
                "listing_date": _first_existing(item, ("Ngày GD đầu tiên", "Listing date")),
                "listed_shares": _parse_hnx_number(
                    _first_existing(item, ("Khối lượng NY", "Khối lượng ĐKGD", "Listed volume"))
                ),
                "outstanding_shares": _parse_hnx_number(
                    _first_existing(item, ("KLLH (Cổ phiếu)", "Outstanding volume (Share)"))
                ),
                "source": "hnx_ajax_listed" if exchange == "HNX" else "hnx_ajax_upcom",
            }
        )
    return _base_listing_frame(rows)


def mine_asean_listings(
    sources: tuple[str, ...] = ("sgx", "pse", "hnx", "upcom"),
) -> tuple[pd.DataFrame, dict[str, bytes]]:
    """Mine current ASEAN listings from public exchange endpoints that are reachable."""

    raw_files: dict[str, bytes] = {}
    frames: list[pd.DataFrame] = []

    if "sgx" in sources:
        sgx_added = False
        try:
            raw = _fetch_bytes_with_params(
                SGX_STOCK_SCREENER_URL,
                {"params": SGX_STOCK_SCREENER_PARAMS},
                referer="https://investors.sgx.com/stock-screener",
            )
            raw_files["sgx_stock_screener.json"] = raw
            screener = parse_sgx_stock_screener(raw.decode("utf-8", errors="replace"))
            if not screener.empty:
                frames.append(screener)
                sgx_added = True
        except Exception as exc:
            raw_files["sgx_stock_screener.error.txt"] = f"{type(exc).__name__}: {exc}".encode()
        try:
            raw = _fetch_bytes_with_params(
                SGX_SECURITIES_URL,
                {"excludetypes": "bonds", "params": SGX_SECURITIES_PARAMS},
                referer="https://investors.sgx.com/stock-screener",
            )
            raw_files["sgx_securities.json"] = raw
            securities = parse_sgx_securities(raw.decode("utf-8", errors="replace"))
            if not securities.empty:
                frames.append(securities)
                sgx_added = True
        except Exception as exc:
            raw_files["sgx_securities.error.txt"] = f"{type(exc).__name__}: {exc}".encode()
        if not sgx_added:
            raise RuntimeError("SGX source requested but no SGX listing adapter returned rows.")
    if "set" in sources:
        raw = _fetch_bytes(SET_LISTED_COMPANIES_URL, referer="https://www.set.or.th/en/market/information/securities-list/main")
        raw_files["set_listed_companies.xls"] = raw
        frames.append(parse_set_listed(raw))
    if "pse" in sources:
        page_raw = _post_form_bytes(
            PSE_COMPANY_DIRECTORY_URL,
            {
                "pageNo": 1,
                "sortType": "",
                "cmpySortType": "",
                "symbolSortType": "ASC",
                "sector": "ALL",
                "subsector": "ALL",
            },
            referer="https://edge.pse.com.ph/companyDirectory/form.do",
        )
        page_text = page_raw.decode("utf-8", errors="replace")
        raw_files["pse_company_directory_page_001.html"] = page_raw
        pages = [page_text]
        for page_no in range(2, _pse_total_pages(page_text) + 1):
            raw = _post_form_bytes(
                PSE_COMPANY_DIRECTORY_URL,
                {
                    "pageNo": page_no,
                    "sortType": "",
                    "cmpySortType": "",
                    "symbolSortType": "ASC",
                    "sector": "ALL",
                    "subsector": "ALL",
                },
                referer="https://edge.pse.com.ph/companyDirectory/form.do",
            )
            raw_files[f"pse_company_directory_page_{page_no:03d}.html"] = raw
            pages.append(raw.decode("utf-8", errors="replace"))
        frames.append(parse_pse_company_directory(pages))
    if "hnx" in sources:
        raw = _post_form_bytes(
            HNX_LISTED_SEARCH_URL,
            {
                "p_issearch": 0,
                "p_keysearch": "",
                "p_market_code": "",
                "p_orderby": "STOCK_CODE",
                "p_ordertype": "ASC",
                "p_currentpage": 1,
                "p_record_on_page": 1000,
            },
            referer="https://hnx.vn/en-gb/cophieu-etfs/chung-khoan-ny.html",
        )
        raw_files["hnx_listed.json"] = raw
        frames.append(parse_hnx_listed(raw.decode("utf-8", errors="replace"), exchange="HNX"))
    if "upcom" in sources:
        raw = _post_form_bytes(
            UPCOM_LISTED_SEARCH_URL,
            {
                "p_issearch": 0,
                "p_keysearch": "",
                "p_market_code": "UC",
                "p_orderby": "STOCK_CODE",
                "p_ordertype": "ASC",
                "p_currentpage": 1,
                "p_record_on_page": 2000,
            },
            referer="https://hnx.vn/en-gb/cophieu-etfs/chung-khoan-uc.html",
            timeout=60,
        )
        raw_files["upcom_listed.json"] = raw
        frames.append(parse_hnx_listed(raw.decode("utf-8", errors="replace"), exchange="UPCoM"))

    listings = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if not listings.empty:
        listings = listings.drop_duplicates(subset=["yahoo_ticker", "exchange"], keep="first")
        listings = listings.sort_values(["country", "exchange", "ticker"]).reset_index(drop=True)
    return listings, raw_files


def save_mined_asean_universe(
    artifact_root: str | Path = "artifacts/universes",
    sources: tuple[str, ...] = ("sgx",),
    run_id: str | None = None,
) -> MinedUniverseResult:
    """Fetch and save a current ASEAN listed-security universe snapshot."""

    run_id = run_id or _now_run_id("ASEAN-LISTINGS")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    listings, raw_files = mine_asean_listings(sources=sources)
    source_files: dict[str, Path] = {}
    for filename, data in raw_files.items():
        path = artifact_dir / filename
        path.write_bytes(data)
        source_files[filename] = path

    listings.to_csv(artifact_dir / "listings.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "sources": ",".join(sources),
                "rows": len(listings),
                "countries": ",".join(sorted(listings["country"].dropna().unique())) if not listings.empty else "",
                "mined_at_utc": datetime.now(timezone.utc).isoformat(),
                "caveat": (
                    "Current listed securities only; not point-in-time. Default public mining is SGX-only; "
                    "Malaysia and Indonesia use vendor/Refinitiv routes for the active three-country universe."
                ),
            }
        ]
    )
    manifest.to_csv(artifact_dir / "manifest.csv", index=False)
    return MinedUniverseResult(run_id=run_id, artifact_dir=artifact_dir, listings=listings, source_files=source_files)


def mine_asia_listings(
    sources: tuple[str, ...] = ("twse", "tpex", "krx", "hkex", "jpx", "sse", "szse"),
) -> tuple[pd.DataFrame, dict[str, bytes]]:
    """Mine current listings from public Asia exchange sources."""

    raw_files: dict[str, bytes] = {}
    frames: list[pd.DataFrame] = []

    if "twse" in sources:
        raw = _fetch_bytes(TWSE_LISTED_URL)
        raw_files["twse.json"] = raw
        frames.append(parse_twse_listed(raw.decode("utf-8", errors="replace")))
    if "tpex" in sources:
        raw = _fetch_bytes(TPEX_LISTED_URL)
        raw_files["tpex.json"] = raw
        frames.append(parse_tpex_listed(raw.decode("utf-8", errors="replace")))
    if "krx" in sources:
        raw = _fetch_bytes(KRX_KIND_LISTED_URL)
        raw_files["krx_kind.xls"] = raw
        frames.append(parse_krx_kind_listed(raw.decode("euc-kr", errors="replace")))
    if "hkex" in sources:
        raw = _fetch_bytes(HKEX_SECURITIES_URL)
        raw_files["hkex_securities.xlsx"] = raw
        frames.append(parse_hkex_securities(raw))
    if "jpx" in sources:
        raw = _fetch_bytes(JPX_LISTED_URL)
        raw_files["jpx_listed.xls"] = raw
        frames.append(parse_jpx_listed(raw))
    if "sse" in sources:
        raw = _fetch_bytes(SSE_LISTED_URL, referer="https://www.sse.com.cn/assortment/stock/list/share/")
        raw_files["sse_listed.jsonp"] = raw
        frames.append(parse_sse_listed(raw.decode("utf-8", errors="replace")))
    if "szse" in sources:
        raw = _fetch_bytes(SZSE_LISTED_URL, referer="https://www.szse.cn/market/product/stock/list/index.html")
        raw_files["szse_listed.xlsx"] = raw
        frames.append(parse_szse_listed(raw))

    listings = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if not listings.empty:
        listings = listings.drop_duplicates(subset=["yahoo_ticker", "exchange"], keep="first")
        listings = listings.sort_values(["country", "exchange", "ticker"]).reset_index(drop=True)
    return listings, raw_files


def mine_asia_listings_resilient(
    sources: tuple[str, ...] = ("twse", "tpex", "krx", "hkex", "jpx", "sse", "szse"),
) -> tuple[pd.DataFrame, dict[str, bytes], pd.DataFrame]:
    """Mine Asia listings source-by-source so one blocked endpoint cannot stall the whole run."""

    raw_files: dict[str, bytes] = {}
    frames: list[pd.DataFrame] = []
    errors: list[dict[str, object]] = []
    for source in sources:
        try:
            source_listings, source_raw_files = mine_asia_listings(sources=(source,))
        except Exception as exc:
            errors.append(
                {
                    "source": source,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "failed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
            )
            continue
        if not source_listings.empty:
            frames.append(source_listings)
        raw_files.update(source_raw_files)
    listings = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not listings.empty:
        listings = listings.drop_duplicates(subset=["yahoo_ticker", "exchange"], keep="first")
        listings = listings.sort_values(["country", "exchange", "ticker"]).reset_index(drop=True)
    return listings, raw_files, pd.DataFrame(errors)


def save_mined_asia_universe(
    artifact_root: str | Path = "artifacts/universes",
    sources: tuple[str, ...] = ("twse", "tpex", "krx", "hkex", "jpx", "sse", "szse"),
    run_id: str | None = None,
    continue_on_error: bool = False,
) -> MinedUniverseResult:
    """Fetch and save a broad current Asia listed-security universe snapshot."""

    run_id = run_id or _now_run_id("ASIA-LISTINGS")
    artifact_dir = Path(artifact_root) / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    errors = pd.DataFrame()
    if continue_on_error:
        listings, raw_files, errors = mine_asia_listings_resilient(sources=sources)
    else:
        listings, raw_files = mine_asia_listings(sources=sources)
    source_files: dict[str, Path] = {}
    for filename, data in raw_files.items():
        path = artifact_dir / filename
        path.write_bytes(data)
        source_files[filename] = path
    if not errors.empty:
        errors.to_csv(artifact_dir / "source_errors.csv", index=False)

    listings.to_csv(artifact_dir / "listings.csv", index=False)
    manifest = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "sources": ",".join(sources),
                "rows": len(listings),
                "countries": ",".join(sorted(listings["country"].dropna().unique())) if not listings.empty else "",
                "failed_sources": ",".join(errors["source"].astype(str).tolist()) if not errors.empty else "",
                "continue_on_error": continue_on_error,
                "mined_at_utc": datetime.now(timezone.utc).isoformat(),
                "caveat": "Current listed securities only; not a point-in-time or delisted-security database.",
            }
        ]
    )
    manifest.to_csv(artifact_dir / "manifest.csv", index=False)
    return MinedUniverseResult(run_id=run_id, artifact_dir=artifact_dir, listings=listings, source_files=source_files)
