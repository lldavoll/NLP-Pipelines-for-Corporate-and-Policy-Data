#!/usr/bin/env python3
"""
Title normalization and executive title summary pipeline.

Converted from the Task 2 notebook into a reusable Python script.

What it does:
- loads the executive dataset
- cleans raw executive titles
- normalizes titles into standard buckets
- assigns broader categories
- writes cleaned data and summary tables
- optionally saves visualizations

Example:
    python title_normalization.py --input executive_final.csv --output-dir outputs/title_normalization
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TITLE_COLUMN = "executive_title"


def clean_title_series(series: pd.Series) -> pd.Series:
    """Basic title cleaning and standardization."""
    cleaned = (
        series.astype(str)
        .str.strip()
        .str.title()
        .str.replace(r"[^a-zA-Z\s]", "", regex=True)
    )
    pattern = r"^(Unknown\s*)$"
    cleaned = cleaned.mask(cleaned.str.match(pattern, case=False, na=True), "Unknown")
    return cleaned


def normalize_title(title: str) -> str:
    """
    Normalize and classify executive titles by hierarchical importance.
    """
    t = str(title).lower().strip()

    if t in ["unknown", "na", "n/a", "none", ""]:
        return "Other Executive"

    if re.search(r"\b(ceo|chief executive)\b", t):
        return "Chief Executive Officer"
    elif re.search(r"\b(vice|svp|evp)\s+president\b", t):
        return "Vice President"
    elif re.search(r"(?<!vice\s)(?<!svp\s)(?<!evp\s)\bpresident\b", t):
        return "President"
    elif re.search(r"\b(coo|chief operating)\b", t):
        return "Chief Operating Officer"
    elif re.search(r"\b(cfo|chief financial)\b", t):
        return "Chief Financial Officer"
    elif re.search(r"\b(cto|cio|chief technology|chief information)\b", t):
        return "Chief Technology Officer"
    elif re.search(r"\b(cmo|chief marketing)\b", t):
        return "Chief Marketing Officer"
    elif re.search(r"\b(general counsel|chief legal|chief counsel|clo)\b", t):
        return "General Counsel"
    elif re.search(r"\bdirector\b", t):
        return "Director"
    elif re.search(r"\b(treasurer|controller|finance)\b", t):
        return "Treasurer / Controller"
    elif re.search(r"\b(secretary)\b", t):
        return "Corporate Secretary"
    else:
        return "Other Executive"


def categorize_standard(title_standard: str) -> str:
    """Assign each standardized title to a broader executive category."""
    t = str(title_standard)

    if "Executive Officer" in t:
        return "C-Suite: CEO"
    elif "Financial Officer" in t:
        return "C-Suite: CFO"
    elif "Operating Officer" in t:
        return "C-Suite: COO"
    elif "Technology Officer" in t:
        return "C-Suite: CTO"
    elif "Marketing Officer" in t:
        return "C-Suite: CMO"
    elif "General Counsel" in t:
        return "C-Suite: Legal"
    elif re.search(r"\bVice President\b", t, re.IGNORECASE):
        return "VP/EVP/SVP"
    elif re.search(r"\bPresident\b", t, re.IGNORECASE):
        return "President"
    elif "Director" in t:
        return "Director"
    elif "Treasurer" in t or "Controller" in t:
        return "Finance Officer"
    elif "Secretary" in t:
        return "Corporate Secretary"
    else:
        return "Other Executive"


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate standardized titles and categories into a summary table."""
    return (
        df.groupby(["title_standard", "category"])
        .size()
        .reset_index(name="count")
        .sort_values(by="count", ascending=False)
    )


def create_visualizations(summary: pd.DataFrame, output_dir: Path) -> None:
    """Save bar chart, pie chart, and histogram from the title summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Bar chart
    plt.figure(figsize=(12, 7))
    sorted_summary = summary.sort_values(by="count", ascending=True)
    bars = plt.barh(sorted_summary["title_standard"], sorted_summary["count"])

    for bar in bars:
        width = bar.get_width()
        plt.text(width + 30, bar.get_y() + bar.get_height() / 2, f"{int(width)}", va="center", fontsize=9)

    plt.title("Executive Titles")
    plt.xlabel("Count")
    plt.ylabel("Executive Title")
    plt.tight_layout()
    plt.savefig(output_dir / "bar_chart.png", dpi=300)
    plt.close()

    # Pie chart
    plt.figure(figsize=(10, 8))
    cat_summary = (
        summary.groupby("category")["count"]
        .sum()
        .reset_index()
        .sort_values(by="count", ascending=False)
    )

    wedges, texts, autotexts = plt.pie(
        cat_summary["count"],
        labels=None,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.8,
    )

    plt.legend(
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        labels=[f"{c} ({v:,})" for c, v in zip(cat_summary["category"], cat_summary["count"])],
        title="Executive Categories",
        title_fontsize=11,
        fontsize=10,
    )
    plt.setp(autotexts, size=9, weight="bold", color="white")
    plt.title("Distribution of Executive Title Categories", fontsize=14, weight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / "category_pie.png", dpi=300)
    plt.close()

    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(summary["count"], bins=15, edgecolor="black")
    plt.title("Distribution of Executive Title Frequencies")
    plt.xlabel("Occurrences per Title")
    plt.ylabel("Number of Titles")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "histogram_frequencies.png", dpi=300)
    plt.close()


def run_pipeline(
    input_file: Path,
    output_dir: Path,
    title_column: str = TITLE_COLUMN,
    save_plots: bool = True,
) -> None:
    """Run the title normalization pipeline end to end."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading file: {input_file}")
    df = pd.read_csv(input_file)

    if title_column not in df.columns:
        raise ValueError(f"Column '{title_column}' not found in input file. Available columns: {list(df.columns)}")

    df["title_clean"] = clean_title_series(df[title_column])
    df["title_standard"] = df["title_clean"].apply(normalize_title)
    df["category"] = df["title_standard"].apply(categorize_standard)

    summary = build_summary(df)

    cleaned_output = output_dir / "executive_titles_cleaned.csv"
    summary_output = output_dir / "executive_titles_category.csv"

    df.to_csv(cleaned_output, index=False)
    summary.to_csv(summary_output, index=False)

    print(f"Saved cleaned dataset: {cleaned_output}")
    print(f"Saved summary dataset: {summary_output}")

    if save_plots:
        create_visualizations(summary, output_dir)
        print(f"Saved plots to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize executive titles and generate summary outputs.")
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument("--output-dir", required=True, help="Directory where outputs will be saved.")
    parser.add_argument("--title-column", default=TITLE_COLUMN, help="Name of the title column to normalize.")
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip creation of bar chart, pie chart, and histogram.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        input_file=Path(args.input),
        output_dir=Path(args.output_dir),
        title_column=args.title_column,
        save_plots=not args.no_plots,
    )


if __name__ == "__main__":
    main()
