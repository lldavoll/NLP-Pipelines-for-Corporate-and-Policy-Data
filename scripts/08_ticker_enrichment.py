#!/usr/bin/env python3
"""
Task 9 — Executive ticker enrichment

Converts the Task 9 notebook into a reusable command-line script.

Pipeline:
1. Load executive dataset and company-ticker crosswalk
2. Compute baseline exact-name ticker match rate
3. Normalize company names for better matching
4. Build ticker candidate lists per normalized company
5. Select a deterministic primary ticker and a preferred common ticker
6. Write enriched executive dataset
7. Write unmatched and ambiguous company QA logs
8. Optionally enrich executive names for later donation matching
9. Optionally triage unmatched companies for likely ETF/fund/trust entities
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import pandas as pd


def norm_company(value: object) -> str:
    """Normalize company names for crosswalk matching."""
    if pd.isna(value):
        return ""

    s = str(value).upper().strip()
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("&", " AND ")

    # Remove SEC-style jurisdiction tags such as /DE/ or /MD/
    s = re.sub(r"/[A-Z]{2,3}/", " ", s)

    # Remove punctuation and collapse whitespace
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # Strip common company suffixes iteratively
    suffixes = [
        " INCORPORATED", " INC",
        " CORPORATION", " CORP",
        " COMPANY", " CO",
        " LIMITED", " LTD",
        " L L C", " LLC",
        " L P", " LP",
        " P L C", " PLC",
        " HOLDINGS", " HOLDING",
        " GROUP",
        " THE",
    ]

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if s.endswith(suffix):
                s = s[: -len(suffix)].strip()
                changed = True

    return s


def choose_common_ticker(candidates: object) -> object:
    """
    Choose a deterministic common-share ticker from a list of candidates.

    Excludes common preferred-share and SPAC-like suffixes when possible.
    """
    if not isinstance(candidates, list) or len(candidates) == 0:
        return pd.NA

    cands = [str(x).strip().upper() for x in candidates if pd.notna(x)]

    def is_common(ticker: str) -> bool:
        if re.search(r"(\.PR|\.PRA|\.PRB|\.PRC|\.PRD|\.PRE|\.PRF|\.PRG|\.PRH|\.PRI|\.PRJ|\.PRK|\.PRL)\b", ticker):
            return False
        if re.search(r"(\-PR[A-Z]?)\b", ticker):
            return False
        if re.search(r"\b(W|WS|WT|R|RT|U)\b$", ticker):
            return False
        return True

    commons = [ticker for ticker in cands if is_common(ticker)]
    return sorted(commons)[0] if commons else sorted(cands)[0]


def norm_person_name(value: object) -> str:
    """Light normalization for executive names."""
    if pd.isna(value):
        return ""
    s = str(value).strip()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\s+", " ", s)
    return s


def build_ticker_map(xwalk: pd.DataFrame) -> pd.DataFrame:
    """Build normalized-company to ticker-candidate mapping."""
    return (
        xwalk.groupby("company_norm")["TICKER"]
        .apply(lambda s: sorted(set(s.dropna().astype(str))))
        .reset_index()
        .rename(columns={"TICKER": "ticker_candidates"})
    )


def compute_baseline(exec_df: pd.DataFrame, xwalk: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Compute baseline exact uppercased company-name merge and missing rate."""
    exec_base = exec_df.copy()
    xwalk_base = xwalk.copy()

    exec_base["company_clean_basic"] = exec_base["company"].astype(str).str.upper().str.strip()
    xwalk_base["company_clean_basic"] = xwalk_base["COMPANY_NAME"].astype(str).str.upper().str.strip()

    baseline = exec_base.merge(
        xwalk_base[["company_clean_basic", "TICKER"]],
        on="company_clean_basic",
        how="left",
    )
    return baseline, baseline["TICKER"].isna().mean()


def enrich_with_tickers(exec_df: pd.DataFrame, xwalk: pd.DataFrame) -> pd.DataFrame:
    """Attach ticker candidate lists and selected ticker fields to executive records."""
    df = exec_df.copy()
    crosswalk = xwalk.copy()

    df["company_norm"] = df["company"].map(norm_company)
    crosswalk["company_norm"] = crosswalk["COMPANY_NAME"].map(norm_company)

    ticker_map = build_ticker_map(crosswalk)

    enriched = df.merge(ticker_map, on="company_norm", how="left")
    enriched["TICKER"] = enriched["ticker_candidates"].apply(
        lambda lst: lst[0] if isinstance(lst, list) and len(lst) else pd.NA
    )
    enriched["ticker_ambiguous"] = enriched["ticker_candidates"].apply(
        lambda lst: isinstance(lst, list) and len(lst) > 1
    )
    enriched["TICKER_COMMON"] = enriched["ticker_candidates"].apply(choose_common_ticker)
    return enriched


def build_unmatched_log(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.loc[df["TICKER"].isna(), ["company", "company_norm"]]
        .drop_duplicates()
        .sort_values(["company_norm", "company"])
    )


def build_ambiguous_log(df: pd.DataFrame, include_common: bool = False) -> pd.DataFrame:
    ambiguous = df.loc[
        df["ticker_ambiguous"],
        ["company", "company_norm", "ticker_candidates"],
    ].copy()

    ambiguous["ticker_candidates_str"] = ambiguous["ticker_candidates"].apply(
        lambda x: "|".join(map(str, x)) if isinstance(x, list) else str(x)
    )

    out_cols = ["company", "company_norm", "ticker_candidates_str"]

    if include_common:
        ambiguous["TICKER_COMMON"] = ambiguous["ticker_candidates"].apply(choose_common_ticker)
        out_cols.append("TICKER_COMMON")

    return (
        ambiguous[out_cols]
        .drop_duplicates()
        .sort_values(["company_norm", "company"])
    )


def build_unmatched_triage(unmatched: pd.DataFrame) -> pd.DataFrame:
    triage = unmatched.copy()
    pattern = r"\b(ETF|TRUST|FUND|PORTFOLIO|ETN)\b"
    triage["looks_like_fund_or_etf"] = triage["company_norm"].str.contains(pattern, regex=True, na=False)
    return triage


def add_exec_name_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Add conservative first/last name fields for later donation matching."""
    out = df.copy()
    out["exec_name_raw"] = out["executive_name"].map(norm_person_name)
    out["exec_name_up"] = out["exec_name_raw"].str.upper()
    out["exec_name_up"] = (
        out["exec_name_up"]
        .str.replace(",", " ", regex=False)
        .str.replace(r"\b(JR|SR|II|III|IV|V)\b\.?", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    parts = out["exec_name_up"].str.split(" ")
    out["exec_first_name"] = parts.str[0]
    out["exec_last_name"] = parts.str[-1]
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def validate_columns(df: pd.DataFrame, required: Iterable[str], df_name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Enrich executive records with stock tickers.")
    parser.add_argument("--executives-csv", required=True, help="Path to executive_final.csv")
    parser.add_argument("--crosswalk-csv", required=True, help="Path to company_ticker_crosswalk.csv")
    parser.add_argument("--output-dir", required=True, help="Directory for enriched output and logs")
    parser.add_argument(
        "--add-exec-name-fields",
        action="store_true",
        help="Add normalized executive first/last name fields for downstream donation matching.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    logs_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    exec_df = pd.read_csv(args.executives_csv)
    xwalk = pd.read_csv(args.crosswalk_csv)

    validate_columns(exec_df, ["company", "executive_name"], "Executive dataset")
    validate_columns(xwalk, ["COMPANY_NAME", "TICKER"], "Crosswalk dataset")

    _, baseline_missing_rate = compute_baseline(exec_df, xwalk)
    print(f"Baseline missing ticker rate: {baseline_missing_rate:.4f}")

    enriched = enrich_with_tickers(exec_df, xwalk)

    missing_rate = enriched["TICKER"].isna().mean()
    ambiguous_rate = enriched["ticker_ambiguous"].mean()
    print(f"After normalization missing ticker rate: {missing_rate:.4f}")
    print(f"Ambiguous ticker rate: {ambiguous_rate:.4f}")

    if args.add_exec_name_fields:
        enriched = add_exec_name_fields(enriched)

    unmatched = build_unmatched_log(enriched)
    ambiguous = build_ambiguous_log(enriched, include_common=False)
    ambiguous_with_common = build_ambiguous_log(enriched, include_common=True)
    unmatched_triage = build_unmatched_triage(unmatched)

    out_exec_path = output_dir / "task9_executives_with_ticker_enriched.csv"
    unmatched_path = logs_dir / "task9_unmatched_companies_after_norm.csv"
    ambiguous_path = logs_dir / "task9_ambiguous_company_to_ticker.csv"
    ambiguous_common_path = logs_dir / "task9_ambiguous_company_to_ticker_with_common.csv"
    triage_path = logs_dir / "task9_unmatched_companies_triage.csv"

    write_csv(enriched, out_exec_path)
    write_csv(unmatched, unmatched_path)
    write_csv(ambiguous, ambiguous_path)
    write_csv(ambiguous_with_common, ambiguous_common_path)
    write_csv(unmatched_triage, triage_path)

    print(f"Saved: {out_exec_path}")
    print(f"Wrote: {unmatched_path}")
    print(f"Wrote: {ambiguous_path}")
    print(f"Wrote: {ambiguous_common_path}")
    print(f"Wrote: {triage_path}")

    print(f"Unmatched total: {len(unmatched_triage)}")
    print(f"Likely ETF/fund/trust share: {unmatched_triage['looks_like_fund_or_etf'].mean():.4f}")

    if enriched["ticker_ambiguous"].any():
        ambig_has_common = enriched.loc[enriched["ticker_ambiguous"], "TICKER_COMMON"].notna().mean()
        print(f"Among ambiguous companies, share with TICKER_COMMON: {ambig_has_common:.4f}")


if __name__ == "__main__":
    main()
