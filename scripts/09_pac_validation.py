#!/usr/bin/env python3
"""Task 8 PAC validation pipeline.

Validates Snowflake PAC committee names against a unified FEC committee master
using exact and fuzzy matching.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Optional, Tuple

import pandas as pd
from rapidfuzz import fuzz, process

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

STOPWORDS = {
    "THE", "AND", "OF", "FOR", "TO", "A", "AN",
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LLC", "LTD", "LIMITED", "LP", "LLP", "HOLDING", "HOLDINGS", "GROUP",
}

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


def normalize_name(value: str) -> str:
    if pd.isna(value):
        return ""
    value = str(value).upper()
    value = re.sub(r"[^A-Z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    tokens = [tok for tok in value.split() if tok not in STOPWORDS]
    return " ".join(tokens)


def build_fec_master(input_dir: str, years: list[str], output_csv: Optional[str] = None) -> pd.DataFrame:
    all_fec = []
    for year in years:
        path = os.path.join(input_dir, f"{year}.txt")
        df = pd.read_csv(path, sep="|", header=None, dtype=str, encoding="latin1")
        df.columns = FEC_COLUMNS
        df["ELECTION_CYCLE"] = year
        all_fec.append(df)

    fec_master = pd.concat(all_fec, ignore_index=True)
    fec_master = fec_master.drop_duplicates(subset=["CMTE_ID"])

    if output_csv:
        fec_master.to_csv(output_csv, index=False)

    return fec_master


def load_inputs(snowflake_csv: str, fec_master_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sf = pd.read_csv(snowflake_csv, dtype=str)
    fec = pd.read_csv(fec_master_csv, dtype=str)
    return sf, fec


def filter_cycles(sf: pd.DataFrame, election_col: str, valid_cycles: list[int]) -> pd.DataFrame:
    sf = sf.copy()
    sf[election_col] = pd.to_numeric(sf[election_col], errors="coerce")
    return sf[sf[election_col].isin(valid_cycles)].copy()


def prepare_names(sf: pd.DataFrame, fec: pd.DataFrame, committee_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sf = sf.copy()
    fec = fec.copy()

    sf["cmte_norm"] = sf[committee_col].map(normalize_name)
    fec["cmte_norm"] = fec["CMTE_NM"].map(normalize_name)

    sf = sf[sf["cmte_norm"] != ""].copy()
    fec = fec[fec["cmte_norm"] != ""].copy()
    return sf, fec


def exact_match(sf: pd.DataFrame, fec: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    fec_key = fec.drop_duplicates("cmte_norm").set_index("cmte_norm")
    out = sf.join(
        fec_key[["CMTE_ID", "CMTE_NM", "CONNECTED_ORG_NM", "CMTE_TP", "ORG_TP"]],
        on="cmte_norm",
        how="left",
        rsuffix="_fec",
    )
    out["match_method"] = out["CMTE_ID"].notna().map(lambda ok: "NAME_EXACT" if ok else "NO_MATCH")
    out["match_score"] = out["CMTE_ID"].notna().map(lambda ok: 100 if ok else None)
    return out, fec_key


def _best_fuzzy(query: str, choices: list[str]) -> Tuple[Optional[str], Optional[float]]:
    hit = process.extractOne(query, choices, scorer=fuzz.token_set_ratio)
    if not hit:
        return None, None
    return hit[0], float(hit[1])


def fuzzy_match_residuals(
    out: pd.DataFrame,
    fec_key: pd.DataFrame,
    auto_threshold: int = 95,
    review_threshold: int = 90,
) -> pd.DataFrame:
    unmatched = out[out["CMTE_ID"].isna()].copy()
    if unmatched.empty:
        return unmatched

    choices = fec_key.reset_index()[
        ["cmte_norm", "CMTE_ID", "CMTE_NM", "CONNECTED_ORG_NM", "CMTE_TP", "ORG_TP"]
    ].copy()
    choice_list = choices["cmte_norm"].tolist()
    choice_map = choices.set_index("cmte_norm").to_dict(orient="index")

    unmatched[["best_norm", "fuzzy_score"]] = unmatched["cmte_norm"].apply(
        lambda s: pd.Series(_best_fuzzy(s, choice_list))
    )

    def bucket(score: float) -> str:
        if pd.isna(score):
            return "NO_MATCH"
        if score >= auto_threshold:
            return "FUZZY_HIGH"
        if score >= review_threshold:
            return "FUZZY_REVIEW"
        return "NO_MATCH"

    unmatched["match_method"] = unmatched["fuzzy_score"].apply(bucket)

    def attach(row: pd.Series) -> pd.Series:
        if row["match_method"] in {"FUZZY_HIGH", "FUZZY_REVIEW"}:
            info = choice_map.get(row["best_norm"], {})
            row["CMTE_ID"] = info.get("CMTE_ID")
            row["CMTE_NM"] = info.get("CMTE_NM")
            row["CONNECTED_ORG_NM"] = info.get("CONNECTED_ORG_NM")
            row["CMTE_TP"] = info.get("CMTE_TP")
            row["ORG_TP"] = info.get("ORG_TP")
        return row

    return unmatched.apply(attach, axis=1)


def consolidate_results(out: pd.DataFrame, unmatched: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    matched_exact = out[out["CMTE_ID"].notna()].copy()
    final = pd.concat([matched_exact, unmatched], ignore_index=True)
    summary = (
        final.groupby("match_method")
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
    )
    total_rows = len(final)
    matched_rows = final["CMTE_ID"].notna().sum()
    match_rate = matched_rows / total_rows if total_rows else 0.0
    return final, summary, match_rate


def top_unmatched(final: pd.DataFrame, committee_col: str, n: int = 50) -> pd.DataFrame:
    return (
        final[final["CMTE_ID"].isna()]
        .groupby(committee_col)
        .size()
        .reset_index(name="rows")
        .sort_values("rows", ascending=False)
        .head(n)
    )


def save_outputs(
    final: pd.DataFrame,
    summary: pd.DataFrame,
    review: pd.DataFrame,
    unmatched_top: pd.DataFrame,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    final.to_csv(os.path.join(output_dir, "task8_private_pac_matches.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "task8_private_pac_match_summary.csv"), index=False)
    review.to_csv(os.path.join(output_dir, "task8_private_pac_fuzzy_review.csv"), index=False)
    unmatched_top.to_csv(os.path.join(output_dir, "task8_private_pac_top_unmatched_committees.csv"), index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Snowflake PACs against FEC committee master.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-fec-master", help="Build unified FEC master from cycle text files.")
    build.add_argument("--input-dir", required=True, help="Directory containing 2020/2022/2024/2026 FEC .txt files.")
    build.add_argument("--years", nargs="+", default=["2020", "2022", "2024", "2026"])
    build.add_argument("--output-csv", required=True)

    run = subparsers.add_parser("run", help="Run Task 8 PAC validation.")
    run.add_argument("--snowflake-csv", required=True)
    run.add_argument("--fec-master-csv", required=True)
    run.add_argument("--output-dir", required=True)
    run.add_argument("--committee-col", default="COMMITTEE_NAME")
    run.add_argument("--election-col", default="ELECTION_CYCLE")
    run.add_argument("--valid-cycles", nargs="+", type=int, default=[2020, 2022, 2024, 2026])
    run.add_argument("--auto-threshold", type=int, default=95)
    run.add_argument("--review-threshold", type=int, default=90)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "build-fec-master":
        fec_master = build_fec_master(args.input_dir, args.years, args.output_csv)
        print(f"Built unified FEC master with {len(fec_master):,} rows -> {args.output_csv}")
        return

    sf, fec = load_inputs(args.snowflake_csv, args.fec_master_csv)
    print(f"Snowflake rows: {len(sf):,}")
    print(f"FEC rows: {len(fec):,}")

    sf = filter_cycles(sf, args.election_col, args.valid_cycles)
    print(f"Snowflake rows after cycle filter: {len(sf):,}")

    sf, fec = prepare_names(sf, fec, args.committee_col)
    print(f"Snowflake rows after name prep: {len(sf):,}")
    print(f"FEC rows after name prep: {len(fec):,}")

    out, fec_key = exact_match(sf, fec)
    print(f"Exact matches: {out['CMTE_ID'].notna().sum():,}")

    residuals = fuzzy_match_residuals(out, fec_key, args.auto_threshold, args.review_threshold)
    final, summary, match_rate = consolidate_results(out, residuals)
    review = final[final["match_method"] == "FUZZY_REVIEW"].copy()
    unmatched_top = top_unmatched(final, args.committee_col)

    save_outputs(final, summary, review, unmatched_top, args.output_dir)

    print("\nMatch summary:")
    print(summary.to_string(index=False))
    print(f"\nTotal rows: {len(final):,}")
    print(f"Matched rows: {final['CMTE_ID'].notna().sum():,}")
    print(f"Match rate: {match_rate:.4f}")
    print(f"\nSaved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
