#!/usr/bin/env python3
"""
Task 4 - SEC Exhibit 21 subsidiary extraction

Converts a company list into:
1. company list with matched SEC CIKs
2. Exhibit-21 availability log
3. extracted parent-subsidiary relationships
4. merged master company-subsidiary dataset

Example:
    python subsidiary_extraction.py \
        --company-file data/raw/unique_companies.csv \
        --output-dir outputs/subsidiary_extraction
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz, process
from tqdm.auto import tqdm


SEC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "Chrome/120.0 DavoAcevedo/1.0 (davoacevedo@arizona.edu)"
    )
}

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

PARENT_SUB_FIELDS = [
    "parent_name",
    "parent_cik",
    "subsidiary_name_raw",
    "subsidiary_name_clean",
    "source_type",
    "source_url",
    "confidence",
    "extraction_notes",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_name(name: str) -> str:
    """Normalize company names for fuzzy matching."""
    return (
        str(name)
        .lower()
        .replace("&", "and")
        .replace(",", "")
        .replace(".", "")
        .strip()
    )


def normalize_subsidiary_name(name: object) -> Optional[str]:
    """Basic normalization for subsidiary names."""
    if not isinstance(name, str):
        return None

    value = name.lower().strip()
    value = re.sub(
        r"\b(inc|inc\.|llc|l\.l\.c\.|corp|corporation|ltd|limited|company|co)\b",
        "",
        value,
    )
    value = re.sub(r"[.,]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def empty_parent_df(parent_name: str, parent_cik: str) -> pd.DataFrame:
    """Return an empty standardized parent-subsidiary DataFrame."""
    _ = parent_name, parent_cik
    return pd.DataFrame(columns=PARENT_SUB_FIELDS)


def load_sec_company_index() -> pd.DataFrame:
    """Load SEC company_tickers.json into a normalized DataFrame."""
    response = requests.get(SEC_TICKERS_URL, headers=SEC_HEADERS, timeout=60)
    response.raise_for_status()

    data = response.json()
    sec_df = pd.DataFrame.from_dict(data, orient="index")
    sec_df["title_clean"] = (
        sec_df["title"]
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace("&", "and")
        .str.strip()
    )
    sec_df["cik_str"] = sec_df["cik_str"].astype(str).str.zfill(10)
    return sec_df


def get_cik(company_name: str, sec_df: pd.DataFrame, threshold: int = 85) -> Optional[str]:
    """Fuzzy match a company name to the SEC company list and return the CIK."""
    choices = sec_df["title_clean"].tolist()
    match = process.extractOne(clean_name(company_name), choices, scorer=fuzz.WRatio)
    if not match:
        return None

    _, score, idx = match
    if score >= threshold:
        return str(sec_df.iloc[idx]["cik_str"])
    return None


def add_ciks_to_company_file(
    company_df: pd.DataFrame,
    sec_df: pd.DataFrame,
    checkpoint_path: Path,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    """Append CIKs to a company list with checkpoint support."""
    df = company_df.copy()

    if checkpoint_path.exists():
        df = pd.read_csv(checkpoint_path, dtype={"cik": str})

    if "cik" not in df.columns:
        df["cik"] = None

    total = len(df)
    start_index = df[df["cik"].isna()].index.min()
    if pd.isna(start_index):
        start_index = 0

    for i in tqdm(range(int(start_index), total), desc="Matching CIKs"):
        if pd.notna(df.at[i, "cik"]):
            continue

        company_name = df.at[i, "company_name"]
        df.at[i, "cik"] = get_cik(company_name, sec_df)

        if i > 0 and i % 100 == 0:
            df.to_csv(checkpoint_path, index=False)

        time.sleep(sleep_seconds)

    df.to_csv(checkpoint_path, index=False)
    return df


def get_latest_10k_metadata(cik: str) -> Optional[dict]:
    """Return accession metadata for the latest 10-K filing."""
    url = f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json"
    response = requests.get(url, headers=SEC_HEADERS, timeout=60)
    if not response.ok:
        return None

    data = response.json()
    forms = data.get("filings", {}).get("recent", {})
    for form, accession_number, primary_doc in zip(
        forms.get("form", []),
        forms.get("accessionNumber", []),
        forms.get("primaryDocument", []),
    ):
        if form == "10-K":
            return {
                "accession": accession_number.replace("-", ""),
                "primary_doc": primary_doc,
                "cik": cik,
            }
    return None


def find_exhibit_21(cik: str, accession: str) -> Optional[str]:
    """Locate Exhibit 21 inside an SEC filing directory."""
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/"
    index_url = base + "index.json"

    response = requests.get(index_url, headers=SEC_HEADERS, timeout=60)
    if not response.ok:
        return None

    items = response.json().get("directory", {}).get("item", [])
    for file_info in items:
        name = file_info["name"].lower()
        if "ex21" in name or "ex-21" in name:
            return base + file_info["name"]
    return None


def parse_exhibit_21(url: str) -> list[str]:
    """Extract subsidiary names from an Exhibit 21 HTML page."""
    response = requests.get(url, headers=SEC_HEADERS, timeout=60)
    if not response.ok:
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    subsidiaries: list[str] = []

    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = row.find_all(["td", "th"])
            if not cells:
                continue

            text = cells[0].get_text(" ", strip=True)
            if text.lower() in {"name", "subsidiary", "entity", "company"}:
                continue
            if len(text) < 3:
                continue
            if not re.search(r"\b(inc|llc|ltd|corp|co|company|bank|group)\b", text, re.I):
                continue

            subsidiaries.append(text)

    if subsidiaries:
        return subsidiaries

    text = soup.get_text("\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    pattern = re.compile(r"\b(inc|llc|ltd|corp|company|co|bank|group)\b", re.I)
    return [line for line in lines if pattern.search(line)]


def extract_exhibit21_subsidiaries(parent_name: str, cik: str) -> pd.DataFrame:
    """Extract subsidiaries for one company using its latest 10-K Exhibit 21."""
    metadata = get_latest_10k_metadata(cik)
    if not metadata:
        return empty_parent_df(parent_name, cik)

    ex21_url = find_exhibit_21(cik, metadata["accession"])
    if not ex21_url:
        return empty_parent_df(parent_name, cik)

    raw_list = parse_exhibit_21(ex21_url)
    if not raw_list:
        return empty_parent_df(parent_name, cik)

    rows = []
    for raw in raw_list:
        rows.append(
            {
                "parent_name": parent_name,
                "parent_cik": cik,
                "subsidiary_name_raw": raw,
                "subsidiary_name_clean": normalize_subsidiary_name(raw),
                "source_type": "exhibit_21",
                "source_url": ex21_url,
                "confidence": 1.0,
                "extraction_notes": "",
            }
        )
    return pd.DataFrame(rows)


def extract_all_ex21(
    companies_df: pd.DataFrame,
    checkpoint_file: Path,
    final_file: Path,
    error_file: Path,
    checkpoint_every: int = 250,
    sleep_seconds: float = 0.25,
) -> pd.DataFrame:
    """Extract subsidiaries for all companies with checkpointing and error logging."""
    results: list[pd.DataFrame] = []
    errors: list[dict] = []
    processed: set[str] = set()

    if checkpoint_file.exists():
        checkpoint_df = pd.read_csv(checkpoint_file, dtype=str)
        if not checkpoint_df.empty:
            results.append(checkpoint_df)
            processed.update(checkpoint_df["parent_cik"].astype(str).unique())

    if error_file.exists():
        error_df = pd.read_csv(error_file, dtype=str)
        errors = error_df.to_dict("records")

    counter = 0
    total = len(companies_df)

    for row in tqdm(companies_df.itertuples(index=False), total=total, desc="Exhibit-21 Extraction"):
        parent = row.company_name
        cik = str(row.cik).replace(".0", "").strip()

        if not cik or cik.lower() == "nan" or cik in processed:
            continue

        try:
            df_subs = extract_exhibit21_subsidiaries(parent, cik)
            if not df_subs.empty:
                results.append(df_subs)
        except Exception as exc:
            errors.append(
                {
                    "parent_name": parent,
                    "parent_cik": cik,
                    "error": repr(exc),
                }
            )

        processed.add(cik)
        counter += 1

        if counter % checkpoint_every == 0:
            if results:
                pd.concat(results, ignore_index=True).to_csv(checkpoint_file, index=False)
            if errors:
                pd.DataFrame(errors).to_csv(error_file, index=False)

        time.sleep(sleep_seconds)

    full_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=PARENT_SUB_FIELDS)
    full_df.to_csv(final_file, index=False)

    if errors:
        pd.DataFrame(errors).to_csv(error_file, index=False)

    return full_df


def check_ex21_availability(companies_df: pd.DataFrame, sleep_seconds: float = 0.25) -> pd.DataFrame:
    """Create a log showing whether each company has an Exhibit 21 and how many subsidiaries were found."""
    results: list[dict] = []

    for row in tqdm(companies_df.itertuples(index=False), total=len(companies_df), desc="Checking Exhibit 21"):
        parent = row.company_name
        cik = str(row.cik).replace(".0", "").strip()

        if not cik or cik.lower() == "nan":
            results.append(
                {
                    "company_name": parent,
                    "cik": cik,
                    "has_exhibit21": False,
                    "exhibit21_url": None,
                    "subsidiary_count": 0,
                    "reason": "Missing CIK",
                }
            )
            continue

        try:
            metadata = get_latest_10k_metadata(cik)
            if not metadata:
                results.append(
                    {
                        "company_name": parent,
                        "cik": cik,
                        "has_exhibit21": False,
                        "exhibit21_url": None,
                        "subsidiary_count": 0,
                        "reason": "No 10-K found",
                    }
                )
                continue

            ex21_url = find_exhibit_21(cik, metadata["accession"])
            if ex21_url is None:
                results.append(
                    {
                        "company_name": parent,
                        "cik": cik,
                        "has_exhibit21": False,
                        "exhibit21_url": None,
                        "subsidiary_count": 0,
                        "reason": "10-K found, but Exhibit-21 missing",
                    }
                )
                continue

            raw_list = parse_exhibit_21(ex21_url)
            results.append(
                {
                    "company_name": parent,
                    "cik": cik,
                    "has_exhibit21": True,
                    "exhibit21_url": ex21_url,
                    "subsidiary_count": len(raw_list),
                    "reason": "Exhibit-21 found",
                }
            )
        except Exception as exc:
            results.append(
                {
                    "company_name": parent,
                    "cik": cik,
                    "has_exhibit21": False,
                    "exhibit21_url": None,
                    "subsidiary_count": 0,
                    "reason": f"Error: {repr(exc)}",
                }
            )

        time.sleep(sleep_seconds)

    return pd.DataFrame(results)


def build_master_company_subsidiary_dataset(
    ex21_log: pd.DataFrame,
    subsidiaries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the Exhibit 21 availability log with extracted subsidiaries."""
    subs = subsidiaries_df.rename(columns={"parent_name": "company_name", "parent_cik": "cik"}).copy()

    master = ex21_log.merge(subs, on=["company_name", "cik"], how="left")
    if "subsidiary_name_raw" in master.columns:
        master["subsidiary_name_raw"] = master["subsidiary_name_raw"].fillna("")
    if "subsidiary_name_clean" in master.columns:
        master["subsidiary_name_clean"] = master["subsidiary_name_clean"].fillna("")

    return master.sort_values(["has_exhibit21", "company_name"], ascending=[False, True])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract parent-subsidiary relationships from SEC Exhibit 21 filings.")
    parser.add_argument("--company-file", required=True, help="CSV with at least a company_name column.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for all generated CSV outputs.",
    )
    parser.add_argument(
        "--skip-cik-match",
        action="store_true",
        help="Use an existing company file with a cik column instead of matching CIKs from SEC.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    company_file = Path(args.company_file)
    companies_df = pd.read_csv(company_file, dtype=str)

    if "company_name" not in companies_df.columns:
        raise ValueError("Input CSV must contain a 'company_name' column.")

    unique_with_cik_path = output_dir / "unique_companies_with_cik.csv"
    ex21_log_path = output_dir / "companies_exhibit21_log.csv"
    checkpoint_file = output_dir / "ex21_checkpoint.csv"
    final_file = output_dir / "ex21_full_output.csv"
    error_file = output_dir / "ex21_errors.csv"
    master_file = output_dir / "master_company_subsidiary_dataset.csv"

    if args.skip_cik_match:
        if "cik" not in companies_df.columns:
            raise ValueError("--skip-cik-match was used, but input CSV has no 'cik' column.")
        companies_with_cik = companies_df.copy()
        companies_with_cik["cik"] = companies_with_cik["cik"].astype(str).str.replace(".0", "", regex=False).str.strip()
        companies_with_cik.to_csv(unique_with_cik_path, index=False)
    else:
        sec_df = load_sec_company_index()
        companies_with_cik = add_ciks_to_company_file(companies_df, sec_df, unique_with_cik_path)

    ex21_log = check_ex21_availability(companies_with_cik)
    ex21_log.to_csv(ex21_log_path, index=False)

    subs_all = extract_all_ex21(
        companies_with_cik,
        checkpoint_file=checkpoint_file,
        final_file=final_file,
        error_file=error_file,
    )

    master = build_master_company_subsidiary_dataset(ex21_log, subs_all)
    master.to_csv(master_file, index=False)

    print(f"Saved companies with CIKs: {unique_with_cik_path}")
    print(f"Saved Exhibit 21 log: {ex21_log_path}")
    print(f"Saved extracted subsidiaries: {final_file}")
    print(f"Saved master dataset: {master_file}")
    if error_file.exists():
        print(f"Saved errors: {error_file}")


if __name__ == "__main__":
    main()
