import pandas as pd
from pathlib import Path
import json
from datetime import datetime

DATA_ROOT = Path("artifacts/data_lake/prices_long/REFINITIV-ASEAN-L2-HISTORICAL-20Y-FULL-20260707/ohlcv/provider_partition=refinitiv/interval_partition=1d/adjustment_partition=vendor_default/country_partition=vietnam")
OUTPUT_DIR = Path("artifacts/kimi_swarm/ASEAN_L2_MEGA_RESEARCH_20260707/agent_findings")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"[{datetime.now()}] Data_Steward_Vietnam starting analysis...")

vn_files = list(DATA_ROOT.rglob("*.parquet"))
print(f"Total Vietnam parquet files: {len(vn_files)}")

print(f"[{datetime.now()}] Reading all parquet files...")
frames = []
for i, fpath in enumerate(vn_files):
    df = pd.read_parquet(fpath)
    if not df.empty:
        frames.append(df)
    if i % 300 == 0:
        print(f"  Processed {i}/{len(vn_files)} files")

vn = pd.concat(frames, ignore_index=True)
print(f"[{datetime.now()}] Concatenated {len(vn):,} rows")

print(f"\n=== VIETNAM DATA STEWARDSHIP REPORT ===")
print(f"Date range: {vn['date'].min()} to {vn['date'].max()}")
print(f"Unique tickers (ticker): {vn['ticker'].nunique()}")
print(f"Unique SIDs: {vn['sid'].nunique()}")
print(f"Unique local_symbols: {vn['local_symbol'].nunique()}")

exch_stats = vn.groupby('exchange').agg(
    rows=('date', 'count'),
    tickers=('ticker', 'nunique'),
    sids=('sid', 'nunique'),
    local_symbols=('local_symbol', 'nunique'),
    start_date=('date', 'min'),
    end_date=('date', 'max'),
).sort_values('rows', ascending=False)
print(f"\n--- Exchange breakdown ---")
print(exch_stats.to_string())

year_stats = vn.groupby('year').agg(
    rows=('date', 'count'),
    tickers=('ticker', 'nunique'),
    start_date=('date', 'min'),
    end_date=('date', 'max'),
).sort_values('year')
print(f"\n--- Year breakdown ---")
print(year_stats.to_string())

key_cols = ['open', 'high', 'low', 'close', 'adj_close', 'volume']
print(f"\n--- Missing data in key columns ---")
missing_map = {}
for col in key_cols:
    missing = int(vn[col].isna().sum())
    missing_pct = missing / len(vn) * 100
    zero_count = int((vn[col] == 0).sum()) if pd.api.types.is_numeric_dtype(vn[col]) else 'N/A'
    zero_pct = (zero_count / len(vn) * 100) if isinstance(zero_count, int) else 'N/A'
    print(f"  {col}: missing={missing:,} ({missing_pct:.3f}%), zero={zero_count} ({zero_pct:.3f}%)")
    missing_map[col] = {"missing": missing, "missing_pct": missing_pct, "zero": zero_count}

invalid_ohlc = vn[
    (vn['high'] < vn['low']) |
    (vn['high'] < vn['open']) |
    (vn['high'] < vn['close']) |
    (vn['low'] > vn['open']) |
    (vn['low'] > vn['close'])
].shape[0]
print(f"\nInvalid OHLC rows: {invalid_ohlc:,} ({invalid_ohlc/len(vn)*100:.4f}%)")

neg_prices = vn[(vn['close'] < 0) | (vn['open'] < 0) | (vn['high'] < 0) | (vn['low'] < 0)].shape[0]
print(f"Negative price rows: {neg_prices:,}")

dup_counts = vn.groupby(['ticker', 'date']).size().reset_index(name='count')
dups = dup_counts[dup_counts['count'] > 1]
print(f"Duplicate (ticker, date) pairs: {len(dups):,}")

print(f"\n--- Stale price analysis (top 10 tickers by rows) ---")
top_tickers = vn['ticker'].value_counts().head(10).index.tolist()
stale_summary = []
for tkr in top_tickers:
    tdf = vn[vn['ticker'] == tkr].sort_values('date')
    tdf['close_change'] = tdf['close'].diff()
    tdf['stale_run'] = (tdf['close_change'] == 0).astype(int).groupby((tdf['close_change'] != 0).cumsum()).cumsum()
    max_stale = tdf['stale_run'].max()
    stale_rows = (tdf['stale_run'] >= 5).sum()
    stale_summary.append({"ticker": tkr, "max_consecutive_same_close": int(max_stale), "rows_with_5plus_stale": int(stale_rows)})

for s in stale_summary:
    print(f"  {s['ticker']}: max_stale_run={s['max_consecutive_same_close']}, rows_5+_stale={s['rows_with_5plus_stale']}")

currencies = vn['currency'].unique().tolist()
print(f"\nCurrencies found: {currencies}")

tiers = vn['data_tier'].value_counts().to_dict()
print(f"Data tier distribution: {tiers}")

ticker_counts = vn['ticker'].value_counts().reset_index()
ticker_counts.columns = ['ticker', 'row_count']
print(f"\nTop 20 tickers by row count:")
print(ticker_counts.head(20).to_string(index=False))

all_tickers = sorted(vn['ticker'].unique().tolist())
print(f"\nTotal tickers: {len(all_tickers)}")

expected_vn = [
    "VCB.VN", "BID.VN", "CTG.VN", "TCB.VN", "VPB.VN", "MBB.VN", "STB.VN", "ACB.VN",
    "VIC.VN", "VHM.VN", "VRE.VN", "HPG.VN", "FPT.VN", "MWG.VN", "GAS.VN", "PLX.VN",
    "VNM.VN", "SAB.VN", "MSN.VN", "SSI.VN", "VND.VN", "HDB.VN", "TPB.VN", "GVR.VN",
    "POW.VN", "BCM.VN", "BVH.VN", "SHB.VN", "VJC.VN", "VIB.VN", "LPB.VN"
]
expected_in_data = [t for t in expected_vn if t in all_tickers]
missing_from_data = [t for t in expected_vn if t not in all_tickers]
unexpected_in_data = [t for t in all_tickers if t not in expected_vn]

print(f"\n--- Universe alignment check ---")
print(f"Expected tickers present: {len(expected_in_data)}/{len(expected_vn)}")
print(f"Missing expected tickers: {missing_from_data}")
print(f"Unexpected tickers (not in universe): {len(unexpected_in_data)}")

findings = {
    "agent": "Data_Steward_Vietnam",
    "data_source": str(DATA_ROOT),
    "analysis_timestamp": datetime.now().isoformat(),
    "total_files": len(vn_files),
    "total_rows": int(len(vn)),
    "date_min": str(vn['date'].min()),
    "date_max": str(vn['date'].max()),
    "unique_tickers": int(vn['ticker'].nunique()),
    "unique_sids": int(vn['sid'].nunique()),
    "unique_local_symbols": int(vn['local_symbol'].nunique()),
    "exchange_breakdown": exch_stats.to_dict(orient='index'),
    "year_breakdown": year_stats.to_dict(orient='index'),
    "missing_data": missing_map,
    "invalid_ohlc": int(invalid_ohlc),
    "negative_price_rows": int(neg_prices),
    "duplicate_date_pairs": int(len(dups)),
    "currencies": currencies,
    "data_tiers": tiers,
    "stale_price_summary": stale_summary,
    "universe_alignment": {
        "expected_tickers": len(expected_vn),
        "present": len(expected_in_data),
        "missing": missing_from_data,
        "unexpected": len(unexpected_in_data),
        "unexpected_tickers_sample": unexpected_in_data[:50] if len(unexpected_in_data) > 50 else unexpected_in_data,
    },
    "all_tickers": all_tickers,
    "top_tickers_by_rows": ticker_counts.head(20).to_dict(orient='records'),
}

json_path = OUTPUT_DIR / "vietnam_data_stewardship.json"
with open(json_path, 'w') as f:
    json.dump(findings, f, indent=2, default=str)
print(f"\nSaved JSON findings to: {json_path}")

print(f"[{datetime.now()}] Analysis complete.")
