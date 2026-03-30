"""
SEC executive extraction pipeline.

Converts raw 8-K filing text into a cleaned executive dataset by:
1. extracting executive names/titles with regex + spaCy
2. cleaning low-confidence names
3. re-running spaCy PERSON extraction on cleaned names
4. normalizing executive titles
5. saving a final dataset and summary report

Example:
    python sec_executive_extraction.py \
        --input 8k_filings_raw_text_2024.csv \
        --output-dir outputs/sec_execs
"""

from __future__ import annotations

import argparse
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import spacy


@dataclass(frozen=True)
class PipelinePaths:
    raw_output: Path
    cleaned_low_output: Path
    cleaned_spacy_output: Path
    final_output: Path
    summary_output: Path


class ExecutiveExtractor:
    """Extract executive information from SEC filing text."""

    def __init__(self, model_name: str = "en_core_web_sm", nlp_max_length: int = 2_000_000) -> None:
        self.nlp = spacy.load(model_name)
        self.nlp.max_length = nlp_max_length

    def extract_executives(self, text: str) -> List[Tuple[str, str, str]]:
        regex_ans = self.regex_extract(text)
        spacy_ans = self.spacy_extract(text)
        return self.deduplicate_executives(regex_ans + spacy_ans)

    def regex_extract(self, text: str) -> List[Tuple[str, str, str]]:
        executives: List[Tuple[str, str, str]] = []

        regex1 = re.compile(
            r"By:\s*/s/\s*([A-Z][a-zA-Z.\s]+?)\s*\n\s*"
            r"(?:Name:\s*)?([A-Z][a-zA-Z.\s]+?)\s*\n\s*"
            r"(?:Title:\s*)?([^\n]+?)(?=\n\s*Date:|\n\n|\Z)",
            re.MULTILINE,
        )
        for match in regex1.finditer(text):
            name = match.group(2).strip()
            title = re.sub(r"\s*Date:.*", "", match.group(3)).strip()
            if len(name.split()) >= 2 and self.executive_title(title):
                executives.append((name, title, "high"))

        regex2 = re.compile(r"By:\s*/s/\s*([A-Z][a-zA-Z.\s]+?)", re.MULTILINE)
        for match in regex2.finditer(text):
            name = match.group(1).strip()
            next_text = text[match.end() : match.end() + 200]
            title_match = re.search(
                r"([A-Z][a-zA-Z\s,&]+(?:Officer|President|Director|Counsel|Secretary|Chairman|Treasurer))",
                next_text,
            )
            if title_match and len(name.split()) >= 2:
                title = title_match.group(1).strip()
                executives.append((name, title, "medium"))

        regex3 = re.compile(
            r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+),\s*"
            r"(Chief\s+\w+\s+Officer|President|Vice\s+President|General\s+Counsel|Director)",
            re.IGNORECASE,
        )
        for match in regex3.finditer(text):
            name = match.group(1).strip()
            title = match.group(2).strip()
            if len(name.split()) >= 2 and self.executive_title(title):
                executives.append((name, title, "low"))

        regex4 = re.compile(
            r"By:\s*/s/\s*([A-Z][a-zA-Z.\s]+?)\s*Name:\s*([A-Z][a-zA-Z.\s]+?)\s*Title:\s*([A-Za-z\s,&]+)",
            re.MULTILINE,
        )
        for match in regex4.finditer(text):
            by_name = match.group(1).strip()
            true_name = match.group(2).strip()
            title = match.group(3).strip()
            name = true_name if len(true_name.split()) >= 2 else by_name
            if len(name.split()) >= 2 and self.executive_title(title):
                executives.append((name, title, "high"))

        return executives

    def spacy_extract(self, text: str) -> List[Tuple[str, str, str]]:
        text = text.replace("\xa0", " ")
        if len(text) <= self.nlp.max_length:
            doc = self.nlp(text)
            return self._extract_from_doc(doc, text)

        executives: List[Tuple[str, str, str]] = []
        step = max(100_000, self.nlp.max_length - 50_000)
        for start in range(0, len(text), step):
            part = text[start : start + step]
            doc = self.nlp(part)
            executives.extend(self._extract_from_doc(doc, part))
        return executives

    def _extract_from_doc(self, doc, text: str) -> List[Tuple[str, str, str]]:
        executives: List[Tuple[str, str, str]] = []
        exec_keywords = {
            "chief",
            "officer",
            "president",
            "counsel",
            "secretary",
            "chairman",
            "treasurer",
            "director",
            "ceo",
            "cfo",
            "coo",
            "cto",
        }

        for ent in doc.ents:
            if ent.label_ != "PERSON":
                continue

            name_text = ent.text.strip()
            if len(name_text.split()) < 2:
                continue
            if any(w in name_text.lower() for w in ["item", "section", "exhibit", "entry"]):
                continue

            context = text[max(0, ent.start_char - 150) : ent.end_char + 150].lower()
            if not any(kw in context for kw in exec_keywords):
                continue

            title_match = re.search(
                r"(ceo|chief\s+[a-z]+\s+officer|cfo|coo|cto|president|vice\s+president|"
                r"general\s+counsel|corporate\s+secretary|chairman|director|treasurer)",
                context,
                re.IGNORECASE,
            )
            title = title_match.group(1).title() if title_match else "Unknown"
            executives.append((name_text, title.strip(), "spacy"))

        return executives

    @staticmethod
    def executive_title(title: str) -> bool:
        keywords = [
            "chief",
            "officer",
            "president",
            "counsel",
            "secretary",
            "chairman",
            "director",
            "treasurer",
        ]
        return any(k in title.lower() for k in keywords)

    @staticmethod
    def deduplicate_executives(executives: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        if not executives:
            return []

        confidence_order = {"high": 0, "medium": 1, "spacy": 2, "low": 3}
        name_dict = {}
        for name, title, conf in executives:
            key = name.lower().strip()
            if key not in name_dict or confidence_order[conf] < confidence_order[name_dict[key][2]]:
                name_dict[key] = (name, title, conf)
        return list(name_dict.values())


ROLE_TERMS = set(
    """
officer officers president vice-president vicepresident vice president chair chairman chairperson
director directors partner partners trustee trustees member members stockholder stockholders shareholder shareholders
advisor advisers adviser consultants consultant manager managers employee employees employer
counsel secretary treasurer controller comptroller attorney
affiliates affiliate subsidiary subsidiaries principal principals owner owners
board committee corporation corp inc llc ltd plc co company companies association
executive operating financial legal accounting compliance general corporate administrative
svp evp avp vp cfo ceo coo cto clo cso cio cro cmo cao gc
""".split()
)

PARTICLES = {"da", "de", "del", "de la", "di", "du", "la", "le", "van", "von", "der", "den", "dos", "das", "mac", "mc", "bin", "ibn", "al"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

NAME_PATTERN = re.compile(
    r"(?:[A-Z][a-z]+|[A-Z]\.?|[A-Z][a-z]+[-'][A-Z][a-z]+|O['’][A-Z][a-z]+)"
    r"(?:\s+(?:[A-Z][a-z]+|[A-Z]\.?|[A-Z][a-z]+[-'][A-Z][a-z]+|O['’][A-Z][a-z]+|"
    r"(?:da|de|del|de la|di|du|la|le|van|von|der|den|dos|das|mac|mc|bin|ibn|al)))+"
)

FALSE_POSITIVE_KEYWORDS = [
    "Prepared Remarks", "Good Standing", "Qualified Transferee", "Diligence", "Milestones",
    "Retirement", "Witness", "Indemnifying", "Securities", "Transfer Restricted", "Baton Rouge",
    "Ganado Advocates", "Due Diligence", "Mutual Acknowledgment", "Hasche Sigle",
    "Dykema Gossett", "Gunderson Dettmer", "Advocates", "Consulting", "LLP", "LLC", "Group",
    "Corp", "Holdings", "Capital", "Incorporated", "Partners", "PLC", "Advisors", "Associates",
    "Inc", "Company", "Enterprises", "Legal", "Strategy", "Bank", "Investments", "Management",
    "Services",
]
FALSE_POSITIVE_PATTERN = re.compile("|".join(re.escape(k) for k in FALSE_POSITIVE_KEYWORDS), re.IGNORECASE)


def token_is_name_like(tok: str) -> bool:
    tok = tok.replace("’", "'")
    plain = re.sub(r"[^A-Za-z\-']", "", tok)
    if not plain:
        return False
    low = plain.lower()
    if re.fullmatch(r"[A-Z]\.?", tok):
        return True
    if re.fullmatch(r"[A-Z][a-z]+(?:[-'][A-Z][a-z]+)+", tok):
        return True
    if re.fullmatch(r"[A-Z][a-z]+", tok) and low not in ROLE_TERMS:
        return True
    if low in PARTICLES or low.strip(".") in SUFFIXES:
        return True
    return False


def extract_best_name(text: object) -> Optional[str]:
    if pd.isna(text):
        return None
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).replace("’", "'")
    candidates: List[str] = []

    for match in NAME_PATTERN.finditer(text):
        span = match.group(0).strip()
        toks = span.split()
        if sum(token_is_name_like(t) for t in toks) >= max(2, len(toks) - 1):
            if any(t.lower().strip(".") in ROLE_TERMS for t in toks):
                continue
            candidates.append(span)

    if not candidates:
        return None

    candidates.sort(key=lambda s: (abs(len(s.split()) - 2), len(s)))
    return candidates[0]


def looks_like_name(name: object) -> bool:
    if pd.isna(name):
        return False
    toks = str(name).strip().split()
    if len(toks) < 2:
        return False
    return all(token_is_name_like(t) for t in toks)


def extract_person_name(text: object, nlp) -> Optional[str]:
    if pd.isna(text):
        return None
    doc = nlp(str(text))
    names = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
    return names[0] if names else None


def is_valid_name(name: object) -> bool:
    if pd.isna(name):
        return False
    name = str(name).strip()
    if not name:
        return False
    if FALSE_POSITIVE_PATTERN.search(name):
        return False
    if len(name.split()) > 6 or any(char.isdigit() for char in name):
        return False
    if name.isupper() and len(name) > 2:
        return False
    return True


def clean_title(title: object) -> Optional[str]:
    if pd.isna(title):
        return None

    cleaned = re.sub(r"<.*?>", " ", str(title))
    cleaned = re.sub(r"[^A-Za-z\s&\-./]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    lower = cleaned.lower()

    if lower in {"na", "n a", "none", "null", "—", "-", "", "0"}:
        return None

    replacements = {
        r"\bceo\b|chief exec(utive)? officer": "Chief Executive Officer",
        r"\bcoo\b|chief operating officer": "Chief Operating Officer",
        r"\bcfo\b|chief financial officer": "Chief Financial Officer",
        r"\bcio\b|chief information officer": "Chief Information Officer",
        r"\bcto\b|chief technology officer": "Chief Technology Officer",
        r"\bchro\b|chief human resources officer": "Chief Human Resources Officer",
        r"\bcmo\b|chief marketing officer": "Chief Marketing Officer",
        r"\bcso\b|chief strategy officer": "Chief Strategy Officer",
        r"\bcro\b|chief risk officer": "Chief Risk Officer",
        r"\bvp\b|vice pres(ident)?": "Vice President",
        r"\bpresident\b": "President",
        r"\bchair(man|woman)?\b": "Chair",
        r"\bmanaging dir(ector)?\b": "Managing Director",
        r"\bexec(utive)? dir(ector)?\b": "Executive Director",
        r"\bboard member\b|director\b": "Director",
    }

    for pattern, replacement in replacements.items():
        if re.search(pattern, lower):
            return replacement

    return cleaned.title()


def build_output_paths(output_dir: Path) -> PipelinePaths:
    output_dir.mkdir(parents=True, exist_ok=True)
    return PipelinePaths(
        raw_output=output_dir / "executive_raw.csv",
        cleaned_low_output=output_dir / "executive_cleaned_low.csv",
        cleaned_spacy_output=output_dir / "executive_cleaned_spacy.csv",
        final_output=output_dir / "executive_final.csv",
        summary_output=output_dir / "executive_dataset_summary.txt",
    )


def extract_raw_executives(
    input_path: Path,
    output_path: Path,
    extractor: ExecutiveExtractor,
    chunk_size: int = 1_500_000,
    resume: bool = True,
) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    existing_rows: List[pd.DataFrame] = []
    completed = set()

    if resume and output_path.exists():
        existing = pd.read_csv(output_path)
        existing_rows.append(existing)
        completed = set(existing["company"].astype(str).unique())

    results = []
    new_executives = 0

    for _, row in df.iterrows():
        company = str(row["company_name"])
        filing = str(row["sec_filing_type"])
        text = str(row["raw_text"])

        if company in completed:
            continue

        parts = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)] if len(text) > extractor.nlp.max_length else [text]

        for part in parts:
            start_time = time.time()
            try:
                execs = extractor.extract_executives(part)
            except Exception as exc:
                print(f"Error processing {company}: {exc}")
                continue

            elapsed = time.time() - start_time
            print(f"Processed {company} chunk in {elapsed:.2f}s")

            for name, title, confidence in execs:
                results.append(
                    {
                        "company": company,
                        "filing_type": filing,
                        "executive_name": name,
                        "executive_title": title,
                        "confidence": confidence,
                    }
                )
            new_executives += len(execs)

        completed.add(company)

    new_df = pd.DataFrame(results)
    final_df = pd.concat(existing_rows + [new_df], ignore_index=True) if existing_rows else new_df
    final_df.to_csv(output_path, index=False)
    print(f"Saved raw extraction to {output_path} ({len(final_df):,} rows, {new_executives:,} new executives)")
    return final_df


def clean_low_confidence_names(raw_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = raw_df.copy()
    mask_low = df["confidence"].astype(str).str.lower().eq("low")

    df_low = df.loc[mask_low].copy()
    df_low["executive_name"] = df_low["executive_name"].apply(extract_best_name)
    df_low = df_low[df_low["executive_name"].notna()]

    df_keep = df.loc[~mask_low].copy()
    df_final = pd.concat([df_keep, df_low], ignore_index=True)
    df_final = df_final[df_final["executive_name"].apply(looks_like_name)]

    df_final.to_csv(output_path, index=False)
    print(f"Saved low-confidence cleaned file to {output_path} ({len(df_final):,} rows)")
    return df_final


def spacy_clean_names(cleaned_df: pd.DataFrame, output_path: Path, model_name: str = "en_core_web_sm") -> pd.DataFrame:
    nlp = spacy.load(model_name)

    df = cleaned_df.copy()
    df["executive_name"] = df["executive_name"].apply(lambda x: extract_person_name(x, nlp))
    df["executive_name"] = (
        df["executive_name"]
        .astype(str)
        .str.replace(r"\b(Email|Phone)\b", "", regex=True)
        .str.strip()
    )

    df = df[df["executive_name"].apply(is_valid_name)]
    df = df.drop_duplicates(subset=["executive_name"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved spaCy-cleaned file to {output_path} ({len(df):,} rows)")
    return df


def normalize_titles(cleaned_df: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    df = cleaned_df.copy()
    df["executive_title_clean"] = df["executive_title"].apply(clean_title)
    df = df.drop_duplicates(subset=["executive_name", "executive_title_clean"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"Saved final normalized file to {output_path} ({len(df):,} rows)")
    return df


def write_summary(df: pd.DataFrame, output_path: Path) -> None:
    name_col = next(c for c in df.columns if "name" in c.lower())
    title_col = next(c for c in df.columns if "title" in c.lower())
    conf_col = next(c for c in df.columns if "conf" in c.lower())
    company_cols = [c for c in df.columns if "company" in c.lower()]

    summary = {
        "total_rows": len(df),
        "unique_executives": df[name_col].nunique(),
        "unique_titles": df[title_col].nunique(),
    }
    if company_cols:
        summary["unique_companies"] = df[company_cols[0]].nunique()
    summary["confidence_breakdown"] = df[conf_col].value_counts().to_dict()

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("Executive Dataset Summary\n")
        fh.write("File: executive_final.csv\n\n")
        for key, value in summary.items():
            fh.write(f"{key}: {value}\n")
        fh.write("\nMain columns:\n")
        for col in [name_col, title_col, conf_col] + company_cols:
            fh.write(f" - {col}\n")

    print(f"Saved summary to {output_path}")


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    model_name: str = "en_core_web_sm",
    nlp_max_length: int = 2_000_000,
    chunk_size: int = 1_500_000,
    resume: bool = True,
) -> PipelinePaths:
    paths = build_output_paths(output_dir)
    extractor = ExecutiveExtractor(model_name=model_name, nlp_max_length=nlp_max_length)

    raw_df = extract_raw_executives(
        input_path=input_path,
        output_path=paths.raw_output,
        extractor=extractor,
        chunk_size=chunk_size,
        resume=resume,
    )
    cleaned_low_df = clean_low_confidence_names(raw_df, paths.cleaned_low_output)
    cleaned_spacy_df = spacy_clean_names(cleaned_low_df, paths.cleaned_spacy_output, model_name=model_name)
    final_df = normalize_titles(cleaned_spacy_df, paths.final_output)
    write_summary(final_df, paths.summary_output)
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract executives from SEC filing text.")
    parser.add_argument("--input", required=True, help="Path to input CSV with company_name, sec_filing_type, and raw_text columns.")
    parser.add_argument("--output-dir", default="outputs/sec_execs", help="Directory where outputs will be written.")
    parser.add_argument("--model", default="en_core_web_sm", help="spaCy model name.")
    parser.add_argument("--nlp-max-length", type=int, default=2_000_000, help="Max characters per spaCy call.")
    parser.add_argument("--chunk-size", type=int, default=1_500_000, help="Chunk size for very long filings.")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse an existing raw output file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_pipeline(
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        nlp_max_length=args.nlp_max_length,
        chunk_size=args.chunk_size,
        resume=not args.no_resume,
    )
    print("\nPipeline complete.")
    print(f"Final dataset: {paths.final_output}")


if __name__ == "__main__":
    main()
