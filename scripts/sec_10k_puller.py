"""
SEC EDGAR 10-K Bulk Puller
"""

import requests
import pandas as pd
import json
import time
import zipfile
import os
from datetime import datetime

USE_RUSSELL_3000 = False
OUTPUT_CSV       = "sec_10k_latest.csv"
OUTPUT_ZIP       = "sec_10k_latest.zip"
RATE_LIMIT_DELAY = 0.12

HEADERS = {
    "User-Agent": "10K-Research-Pipeline research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

def get_ticker_cik_map():
    print("Fetching ticker → CIK map from EDGAR...")
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    mapping = {}
    for entry in data.values():
        ticker = entry["ticker"].upper()
        cik    = str(entry["cik_str"]).zfill(10)
        name   = entry["title"]
        mapping[ticker] = {"cik": cik, "name": name}
    print(f"  → {len(mapping)} tickers in EDGAR map")
    return mapping

def get_sp500_tickers():
    print("Fetching S&P 500 tickers from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia blocks default Python user-agent, so we mimic a browser
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    from io import StringIO
    tables = pd.read_html(StringIO(r.text))
    df = tables[0]
    tickers = df["Symbol"].str.upper().str.replace(".", "-", regex=False).tolist()
    print(f"  → {len(tickers)} S&P 500 tickers found")
    return tickers

def get_russell3000_tickers():
    print("Fetching Russell 3000 tickers via iShares IWV holdings...")
    url = "https://www.ishares.com/us/products/239714/ishares-russell-3000-etf/1467271812596.ajax?fileType=csv&fileName=IWV_holdings&dataType=fund"
    r = requests.get(url, headers=HEADERS, timeout=30)
    lines = r.text.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith("Ticker,"))
    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines[start:])))
    tickers = df["Ticker"].dropna().str.upper().tolist()
    tickers = [t for t in tickers if t.isalpha() or "-" in t]
    print(f"  → {len(tickers)} Russell 3000 tickers found")
    return tickers

def get_latest_10k(cik, ticker, company_name):
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return {"ticker": ticker, "company_name": company_name, "cik": cik, "error": str(e)}

    recent       = data.get("filings", {}).get("recent", {})
    forms        = recent.get("form", [])
    dates        = recent.get("filingDate", [])
    accessions   = recent.get("accessionNumber", [])
    periods      = recent.get("reportDate", [])
    descriptions = recent.get("primaryDocument", [])

    for form, date, accession, period, doc in zip(forms, dates, accessions, periods, descriptions):
        if form == "10-K":
            acc_clean  = accession.replace("-", "")
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_clean}/{doc}"
            index_url  = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&dateb=&owner=include&count=1"
            return {
                "ticker": ticker, "company_name": company_name, "cik": cik,
                "filing_date": date, "report_period": period,
                "accession_number": accession, "primary_document": doc,
                "filing_url": filing_url, "edgar_index_url": index_url, "error": None,
            }

    return {"ticker": ticker, "company_name": company_name, "cik": cik, "error": "No 10-K found"}

def main():
    start_time     = datetime.now()
    ticker_cik_map = get_ticker_cik_map()
    tickers        = get_russell3000_tickers() if USE_RUSSELL_3000 else get_sp500_tickers()

    print(f"\nPulling 10-K data for {len(tickers)} companies...")
    results, not_found = [], []

    for i, ticker in enumerate(tickers):
        if ticker not in ticker_cik_map:
            not_found.append(ticker)
            continue
        info = ticker_cik_map[ticker]
        results.append(get_latest_10k(info["cik"], ticker, info["name"]))
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(tickers)} processed...")
        time.sleep(RATE_LIMIT_DELAY)

    df      = pd.DataFrame(results)
    cols    = ["ticker","company_name","cik","filing_date","report_period","accession_number","primary_document","filing_url","edgar_index_url","error"]
    df      = df.reindex(columns=cols)
    success = df["error"].isna().sum()
    failed  = df["error"].notna().sum()

    print(f"\n✓ Done: {success} 10-Ks found, {failed} errors, {len(not_found)} tickers not in EDGAR map")

    df.to_csv(OUTPUT_CSV, index=False)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(OUTPUT_CSV)
    os.remove(OUTPUT_CSV)

    elapsed = (datetime.now() - start_time).seconds
    print(f"\nOutput: {OUTPUT_ZIP}  ({elapsed}s elapsed) | Rows: {len(df)}")
    print(df[df["error"].isna()].head(3).to_string(index=False))

if __name__ == "__main__":
    main()