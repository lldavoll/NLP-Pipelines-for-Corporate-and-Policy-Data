
#!/usr/bin/env python3
"""
task7_pac_superset.py

Clean Python script for GUU Task 7:
Build a company-level PAC activity superset by combining:
1) FEC committee master data
2) Snowflake PAC activity data
3) Company universe + ticker crosswalk

This script consolidates the logic from:
- GUU_Task_7_I.ipynb
- GUU_Task_7_II.ipynb
- GUU_Task_7_III.ipynb

Outputs:
- fec_committee_master.csv
- fec_pac_master.csv
- fec_pacs_with_donation_flag.csv
- fec_pacs_no_donations.csv
- company_pac_donation_status.csv
- company_pac_superset_U.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Optional

import pandas as pd


FEC_COLUMNS = [
    "CMTE_ID",
    "CMTE_NM",
    "TRES_NM",
    "CMTE_ST1",
    "CMTE_ST2",
    "CMTE_CITY",
    "CMTE_ST",
    "CMTE_ZIP",
    "CMTE_DSGN",
    "CMTE_TP",
    "CMTE_PTY_AFFILIATION",
    "CMTE_FILING_FREQ",
    "ORG_TP",
    "CONNECTED_ORG_NM",
    "CAND_ID",
]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def norm_committee_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(pac|political|action|committee|fund|the|and|of|for|employees)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_company_name(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(inc|incorporated|corp|corporation|co|company|llc|ltd|plc|sa|ag|nv)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def looks_like_pac(s: Optional[str]) -> bool:
    s = "" if s is None else str(s).lower()
    return bool(re.search(r"\b(pac|political action committee)\b", s))


def truthy_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y", "t"])
    )


def load_fec_committee_master(fec_txt_path: Path) -> pd.DataFrame:
    fec = pd.read_csv(
        fec_txt_path,
        sep="|",
        header=None,
        dtype=str,
        encoding="latin1",
    )
    fec.columns = FEC_COLUMNS
    return fec


def build_fec_pac_master(fec: pd.DataFrame) -> pd.DataFrame:
    return fec[
        fec["CONNECTED_ORG_NM"].notna()
        & (fec["CMTE_TP"] == "Q")
        & (fec["ORG_TP"] == "C")
    ][["CMTE_ID", "CMTE_NM", "CONNECTED_ORG_NM", "CMTE_TP", "ORG_TP"]].copy()


def merge_fec_with_snowflake(fec_pac_master: pd.DataFrame, snowflake_df: pd.DataFrame) -> pd.DataFrame:
    fec_p = fec_pac_master.copy()
    sf = snowflake_df.copy()

    fec_p["committee_name_clean"] = fec_p["CMTE_NM"].map(norm_committee_name)
    sf["committee_name_clean"] = sf["COMMITTEE_NAME"].map(norm_committee_name)

    sf["TOTAL_AMOUNT"] = pd.to_numeric(sf["TOTAL_AMOUNT"], errors="coerce").fillna(0)

    merged = fec_p.merge(
        sf[["committee_name_clean", "TOTAL_AMOUNT", "HAS_DONATED", "FIRST_CYCLE", "LAST_CYCLE"]],
        on="committee_name_clean",
        how="left",
        suffixes=("_fec", "_sf"),
    )

    merged["TOTAL_AMOUNT"] = pd.to_numeric(merged["TOTAL_AMOUNT"], errors="coerce").fillna(0)
    # preserve notebook logic: anything with positive amount counts as donated
    merged["HAS_DONATED"] = merged["TOTAL_AMOUNT"] > 0
    return merged


def build_company_pac_donation_status(fec_with_flags: pd.DataFrame) -> pd.DataFrame:
    df = fec_with_flags.copy()
    df["TOTAL_AMOUNT"] = pd.to_numeric(df["TOTAL_AMOUNT"], errors="coerce").fillna(0)
    df["HAS_DONATED"] = df["TOTAL_AMOUNT"] > 0

    company_rollup = (
        df.groupby("CONNECTED_ORG_NM", dropna=True)
        .agg(
            pac_count=("CMTE_ID", "nunique"),
            active_pac_count=("HAS_DONATED", "sum"),
            has_any_pac_donated=("HAS_DONATED", "any"),
            total_pac_amount=("TOTAL_AMOUNT", "sum"),
        )
        .reset_index()
    )

    company_rollup["has_pac"] = True
    company_rollup["has_pac_donations"] = company_rollup["has_any_pac_donated"]

    company_rollup_clean = company_rollup[
        ~company_rollup["CONNECTED_ORG_NM"].map(looks_like_pac)
    ].copy()

    return company_rollup_clean


def build_company_pac_superset(
    company_universe: pd.DataFrame,
    ticker_crosswalk: pd.DataFrame,
    company_pac_status: pd.DataFrame,
) -> pd.DataFrame:
    U = company_universe.copy()
    xwalk = ticker_crosswalk.copy()
    pac = company_pac_status.copy()

    xwalk["company_name_clean"] = xwalk["COMPANY_NAME"].map(norm_company_name)
    pac["company_name_clean"] = pac["CONNECTED_ORG_NM"].map(norm_company_name)

    pac["has_pac"] = truthy_series(pac["has_pac"])
    pac["has_pac_donations"] = truthy_series(pac["has_pac_donations"])
    pac["pac_count"] = pd.to_numeric(pac.get("pac_count"), errors="coerce")
    pac["active_pac_count"] = pd.to_numeric(pac.get("active_pac_count"), errors="coerce")
    pac["total_pac_amount"] = pd.to_numeric(pac.get("total_pac_amount"), errors="coerce")

    pac_name_level = (
        pac.groupby("company_name_clean", as_index=False)
        .agg(
            has_pac=("has_pac", "any"),
            has_pac_donations=("has_pac_donations", "any"),
            pac_count=("pac_count", "max"),
            active_pac_count=("active_pac_count", "max"),
            total_pac_amount=("total_pac_amount", "max"),
        )
    )

    pac_to_ticker = xwalk.merge(pac_name_level, on="company_name_clean", how="inner")

    pac_ticker_level = (
        pac_to_ticker.groupby("TICKER", as_index=False)
        .agg(
            has_pac=("has_pac", "any"),
            has_pac_donations=("has_pac_donations", "any"),
            pac_count=("pac_count", "max"),
            active_pac_count=("active_pac_count", "max"),
            total_pac_amount=("total_pac_amount", "max"),
        )
    )

    final = U.merge(
        pac_ticker_level[
            [
                "TICKER",
                "has_pac",
                "has_pac_donations",
                "pac_count",
                "active_pac_count",
                "total_pac_amount",
            ]
        ],
        on="TICKER",
        how="left",
    )

    final["has_pac"] = final["has_pac"].fillna(False)
    final["has_pac_donations"] = final["has_pac_donations"].fillna(False)

    def status(row: pd.Series) -> str:
        if not row["has_pac"]:
            return "NO_PAC"
        if row["has_pac_donations"]:
            return "PAC_DONATED"
        return "PAC_NO_DONATIONS"

    final["pac_status"] = final.apply(status, axis=1)
    return final


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the GUU Task 7 company PAC superset from FEC, Snowflake, and company universe data."
    )
    parser.add_argument("--fec-txt", required=True, help="Path to fec_committee_master.txt")
    parser.add_argument("--snowflake-pac", required=True, help="Path to snowflake_pac_activity.csv")
    parser.add_argument("--company-universe", required=True, help="Path to company_universe.csv")
    parser.add_argument("--ticker-crosswalk", required=True, help="Path to company_ticker_crosswalk.csv")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be written")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    fec = load_fec_committee_master(Path(args.fec_txt))
    fec.to_csv(output_dir / "fec_committee_master.csv", index=False)

    fec_pac_master = build_fec_pac_master(fec)
    fec_pac_master.to_csv(output_dir / "fec_pac_master.csv", index=False)

    snowflake_df = pd.read_csv(args.snowflake_pac, dtype=str)
    fec_with_flags = merge_fec_with_snowflake(fec_pac_master, snowflake_df)
    fec_with_flags.to_csv(output_dir / "fec_pacs_with_donation_flag.csv", index=False)

    no_donations = fec_with_flags[fec_with_flags["TOTAL_AMOUNT"] == 0].copy()
    no_donations.to_csv(output_dir / "fec_pacs_no_donations.csv", index=False)

    company_pac_status = build_company_pac_donation_status(fec_with_flags)
    company_pac_status.to_csv(output_dir / "company_pac_donation_status.csv", index=False)

    company_universe = pd.read_csv(args.company_universe, dtype=str)
    ticker_crosswalk = pd.read_csv(args.ticker_crosswalk, dtype=str)
    final = build_company_pac_superset(company_universe, ticker_crosswalk, company_pac_status)
    final.to_csv(output_dir / "company_pac_superset_U.csv", index=False)

    print("=== Task 7 complete ===")
    print(f"FEC corporate PACs: {len(fec_with_flags):,}")
    print(f"Matched PACs with donations: {(fec_with_flags['TOTAL_AMOUNT'] > 0).sum():,}")
    print(f"PACs with no donation activity: {(fec_with_flags['TOTAL_AMOUNT'] == 0).sum():,}")
    print()
    print(f"Company PAC status rows: {len(company_pac_status):,}")
    print(f"Universe tickers: {len(final):,}")
    print("PAC status counts:")
    print(final["pac_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
