
#!/usr/bin/env python3
"""
Task 3 - Contact Enrichment

Build a company-contact dataset from an executive dataset by:
1. extracting unique companies
2. normalizing company names for search
3. discovering official company domains
4. extracting investor-relations and customer-service pages/emails
5. optionally enriching with social-media links
6. optionally retrying rows with missing values

Example:
    python contact_enrichment.py \
        --input executive_final.csv \
        --output-dir outputs/contact_enrichment \
        --run-social \
        --fix-missing
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
import tldextract
from bs4 import BeautifulSoup
from ddgs import DDGS
from tqdm.auto import tqdm

EMAIL_RE = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
HEADERS = {"User-Agent": "Mozilla/5.0"}

NOISE_DOMAINS = [
    "wikipedia.org",
    "yahoo.com",
    "news.yahoo.com",
    "bloomberg.com",
    "reuters.com",
    "marketwatch.com",
    "techcrunch.com",
    "crunchbase.com",
    "linkedin.com",
    "dwinnex.com",
    "wordpress.com",
    "blogspot.com",
    "medium.com",
    "zoominfo.com",
    "herokuapp.com",
    "sourceforge.net",
    "sec.gov",
    "bsky.app",
    "facebook.com",
    "twitter.com",
    "x.com",
    "duckduckgo.com/y.js",
    "bing.com/aclick",
    "ad_domain=",
    "ad_provider=",
    "utm_",
    "doubleclick.net",
]

LEGAL_SUFFIXES = {
    "INC", "CORP", "CO", "LTD", "LLC", "PLC",
    "GROUP", "HOLDINGS", "HLDGS", "COM", "COMPANY",
}

BAD_BASE_DOMAINS = {
    "microsoft.com",
    "amazon.com",
    "google.com",
    "bing.com",
    "duckduckgo.com",
    "yahoo.com",
    "reddit.com",
    "facebook.com",
    "instagram.com",
    "walmart.com",
    "bestbuy.com",
    "newegg.com",
    "lenovo.com",
}

ddg_cache: Dict[Tuple[str, int], List[str]] = {}
domain_cache: Dict[str, Optional[str]] = {}
html_cache: Dict[str, Optional[str]] = {}


def is_acronym(word: str) -> bool:
    return (
        word.isupper()
        and word.isalpha()
        and 2 <= len(word) <= 4
        and word not in LEGAL_SUFFIXES
    )


def is_mix_acronym(word: str) -> bool:
    return bool(re.match(r"^\d[A-Z]$", word))


def process_simple(part: str) -> str:
    if not part:
        return part

    if re.match(r"^\d+[A-Za-z]\d+$", part):
        return part.lower()

    if is_acronym(part):
        return part

    if is_mix_acronym(part):
        return part

    upper = part.upper()
    if upper in LEGAL_SUFFIXES:
        return upper if upper in {"LLC", "PLC"} else upper.capitalize()

    if part.isupper():
        return part.capitalize()

    return part.capitalize()


DOMAIN_RE = re.compile(
    r"^[A-Za-z0-9\-]+\.(com|org|net|io|co|ai|gov|edu|biz|info)$",
    flags=re.IGNORECASE,
)


def normalize_company_name(name: str) -> str:
    """Normalize company names for web search while preserving acronyms."""
    if not isinstance(name, str):
        return ""

    name = name.strip()
    if not name:
        return ""

    if DOMAIN_RE.match(name):
        return name.lower()

    pieces = []
    for token in name.split():
        if "." in token and DOMAIN_RE.match(token):
            pieces.append(token.lower())
            continue

        if "-" in token and not token.startswith("-") and not token.endswith("-"):
            parts = token.split("-")
            pieces.append("-".join(process_simple(p) for p in parts))
        else:
            pieces.append(process_simple(token))

    return " ".join(pieces)


def ddg_urls(query: str, max_results: int = 8) -> List[str]:
    cache_key = (query, max_results)
    if cache_key in ddg_cache:
        return ddg_cache[cache_key]

    urls: List[str] = []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            for result in results:
                url = result.get("href", "")
                if not url:
                    continue
                if "duckduckgo.com/y.js" in url or "aclick?" in url:
                    continue
                urls.append(url)
    except Exception:
        urls = []

    ddg_cache[cache_key] = urls
    return urls


def is_noise(url: str) -> bool:
    url = url.lower()
    if any(noise in url for noise in NOISE_DOMAINS):
        return True

    ad_patterns = [
        "duckduckgo.com/y.js",
        "bing.com/aclick",
        "ad_domain=",
        "ad_provider=",
        "utm_",
        "doubleclick.net",
        "clickserve",
        "tracking",
        "redirect",
    ]
    return any(pattern in url for pattern in ad_patterns)


def extract_domain(url: str) -> Optional[str]:
    if not url:
        return None
    try:
        ext = tldextract.extract(url)
        if not ext.domain or not ext.suffix:
            return None
        return f"{ext.domain}.{ext.suffix}".lower()
    except Exception:
        return None


def fetch_html(url: str) -> Optional[str]:
    if not url:
        return None
    if url in html_cache:
        return html_cache[url]

    for _ in range(2):
        try:
            response = requests.get(url, headers=HEADERS, timeout=8)
            if response.status_code == 200:
                html_cache[url] = response.text
                return response.text
        except Exception:
            time.sleep(1)

    html_cache[url] = None
    return None


def extract_emails_from_html(html: Optional[str]) -> List[str]:
    if not html:
        return []
    return sorted(set(re.findall(EMAIL_RE, html)))


def discover_domain(company: str) -> Optional[str]:
    """
    Discover the most likely official company domain using DDG search results
    and token-based brand matching.
    """
    if company in domain_cache:
        return domain_cache[company]

    clean = company.lower().strip()
    tokens = [t for t in re.split(r"\W+", clean) if len(t) > 2]
    if not tokens:
        domain_cache[company] = None
        return None

    generic_terms = {
        "inc", "corp", "company", "group", "ltd", "co", "holdings", "plc", "sa", "nv"
    }
    brand_tokens = [t for t in tokens if t not in generic_terms] or tokens

    queries = [
        f"{company} official website",
        f"{company} corporate site",
        f"{company} homepage",
        f"{company} investor relations",
    ]

    best_domain: Optional[str] = None
    best_score = 0

    for query in queries:
        urls = ddg_urls(query, max_results=10)
        for url in urls:
            if is_noise(url):
                continue

            domain = extract_domain(url)
            if not domain or domain in BAD_BASE_DOMAINS:
                continue

            score = sum(1 for token in brand_tokens if token in domain)

            initials = "".join(token[0] for token in brand_tokens if token)
            if initials and initials.lower() in domain:
                score += 1

            if domain.endswith(".com"):
                score += 1

            if score > best_score:
                best_score = score
                best_domain = domain

    if best_domain:
        domain_cache[company] = best_domain
        return best_domain

    fallback_domains = []
    root = tokens[0]
    fallback_domains.extend([
        f"{root}.com",
        f"{root}inc.com",
        f"{root}corp.com",
        f"{root}co.com",
        f"{root}group.com",
    ])
    fallback_domains.extend(f"{token}.com" for token in brand_tokens)

    for fallback in fallback_domains:
        urls = ddg_urls(fallback, max_results=3)
        for url in urls:
            if extract_domain(url) == fallback:
                domain_cache[company] = fallback
                return fallback

    domain_cache[company] = None
    return None


def find_page_on_domain(company: str, domain: str, keyword: str) -> Optional[str]:
    if not domain:
        return None

    query = f"{company} {keyword} site:{domain}"
    urls = ddg_urls(query, max_results=10)
    for url in urls:
        if domain in url.lower() and not is_noise(url):
            return url
    return None


def get_investor_info(company: str, domain: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not domain:
        return None, None

    page = find_page_on_domain(company, domain, "investor relations")
    if not page:
        for keyword in ["investor", "ir"]:
            page = find_page_on_domain(company, domain, keyword)
            if page:
                break
    if not page:
        return None, None

    html = fetch_html(page)
    emails = extract_emails_from_html(html)
    for email in emails:
        if any(tag in email.lower() for tag in ["ir@", "investor@", "investors@", "shareholder"]):
            return email, page
    return (emails[0], page) if emails else (None, page)


def get_customer_service_info(company: str, domain: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not domain:
        return None, None

    page = find_page_on_domain(company, domain, "customer service")
    if not page:
        for keyword in ["contact", "support", "help"]:
            page = find_page_on_domain(company, domain, keyword)
            if page:
                break
    if not page:
        return None, None

    html = fetch_html(page)
    emails = extract_emails_from_html(html)
    for email in emails:
        if any(tag in email.lower() for tag in ["support@", "service@", "help@", "info@", "care@"]):
            return email, page
    return (emails[0], page) if emails else (None, page)


def find_twitter(company: str) -> Optional[str]:
    urls = ddg_urls(f"{company} official twitter x", max_results=8)
    for url in urls:
        lower = url.lower()
        if ("twitter.com/" in lower or "x.com/" in lower) and not any(
            bad in lower for bad in ["/status/", "/intent/", "/share", "/hashtag", "/search"]
        ):
            return url
    return None


def find_facebook(company: str) -> Optional[str]:
    urls = ddg_urls(f"{company} official facebook page", max_results=8)
    for url in urls:
        lower = url.lower()
        if "facebook.com/" in lower and not any(
            bad in lower for bad in ["sharer", "php", "story.php", "l.php"]
        ):
            return url
    return None


def find_bluesky(company: str) -> Optional[str]:
    urls = ddg_urls(f"{company} official bluesky", max_results=8)
    for url in urls:
        if "bsky.app/profile/" in url.lower():
            return url
    return None


def prepare_unique_companies(input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    if "company" not in df.columns:
        raise ValueError("Input file must contain a 'company' column.")

    companies = df[["company"]].drop_duplicates().sort_values("company").reset_index(drop=True)
    companies["company_clean"] = companies["company"].map(normalize_company_name)

    if output_file:
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        companies.to_csv(output_file, index=False)

    return companies


def extract_for_company(row: pd.Series) -> Dict[str, Optional[str]]:
    company_raw = row["company"]
    company_clean = row["company_clean"]

    domain = discover_domain(company_clean)
    ir_email, ir_page = (None, None)
    cs_email, cs_page = (None, None)

    if domain:
        ir_email, ir_page = get_investor_info(company_clean, domain)
        cs_email, cs_page = get_customer_service_info(company_clean, domain)

    return {
        "company": company_raw,
        "company_clean": company_clean,
        "domain": domain,
        "ir_page": ir_page,
        "ir_email": ir_email,
        "cs_page": cs_page,
        "cs_email": cs_email,
        "error": None,
    }


def run_contact_extraction(
    companies: pd.DataFrame,
    out_file: str,
    max_companies: Optional[int] = None,
) -> pd.DataFrame:
    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    columns = [
        "company", "company_clean", "domain",
        "ir_page", "ir_email",
        "cs_page", "cs_email",
        "error",
    ]

    if out_path.exists():
        previous = pd.read_csv(out_path)
        completed = len(previous)
        print(f"Resuming extraction from row {completed}")
    else:
        pd.DataFrame(columns=columns).to_csv(out_path, index=False)
        completed = 0
        print("Starting fresh extraction")

    df_iter = companies.iloc[:max_companies].reset_index(drop=True) if max_companies else companies.reset_index(drop=True)

    for idx in tqdm(range(completed, len(df_iter)), desc="Extracting company contacts"):
        row = df_iter.iloc[idx]
        try:
            info = extract_for_company(row)
        except Exception as exc:
            info = {
                "company": row["company"],
                "company_clean": row["company_clean"],
                "domain": None,
                "ir_page": None,
                "ir_email": None,
                "cs_page": None,
                "cs_email": None,
                "error": str(exc),
            }
        pd.DataFrame([info]).to_csv(out_path, mode="a", header=False, index=False)

    return pd.read_csv(out_path)


def run_social_extraction(
    in_file: str,
    out_file: str,
    max_companies: Optional[int] = None,
) -> pd.DataFrame:
    df = pd.read_csv(in_file)

    for col in ["twitter_url", "facebook_url", "bluesky_url"]:
        if col not in df.columns:
            df[col] = None

    total = len(df) if max_companies is None else min(max_companies, len(df))
    completed = df[["twitter_url", "facebook_url", "bluesky_url"]].notna().all(axis=1).sum()
    print(f"Resuming social extraction from row {completed}")

    for idx in tqdm(range(completed, total), desc="Extracting social links"):
        company = df.iloc[idx]["company_clean"]
        df.at[idx, "twitter_url"] = find_twitter(company)
        df.at[idx, "facebook_url"] = find_facebook(company)
        df.at[idx, "bluesky_url"] = find_bluesky(company)
        df.to_csv(out_file, index=False)

    return df


def extract_missing_info(row: pd.Series, idx: int) -> Dict[str, Optional[str]]:
    company_raw = row["company"]
    company_clean = row["company_clean"]

    domain = row.get("domain", None)
    ir_page = row.get("ir_page", None)
    ir_email = row.get("ir_email", None)
    cs_page = row.get("cs_page", None)
    cs_email = row.get("cs_email", None)
    twitter = row.get("twitter_url", None)
    facebook = row.get("facebook_url", None)
    bluesky = row.get("bluesky_url", None)

    if pd.isna(domain) or len(str(domain)) < 3:
        domain = discover_domain(company_clean)

    if (pd.isna(ir_page) or str(ir_page).strip() == "") and domain:
        new_ir_email, new_ir_page = get_investor_info(company_clean, domain)
        if new_ir_page:
            ir_page = new_ir_page
        if new_ir_email:
            ir_email = new_ir_email

    if (pd.isna(cs_page) or str(cs_page).strip() == "") and domain:
        new_cs_email, new_cs_page = get_customer_service_info(company_clean, domain)
        if new_cs_page:
            cs_page = new_cs_page
        if new_cs_email:
            cs_email = new_cs_email

    if pd.isna(twitter) or str(twitter).strip() == "":
        twitter = find_twitter(company_clean)
    if pd.isna(facebook) or str(facebook).strip() == "":
        facebook = find_facebook(company_clean)
    if pd.isna(bluesky) or str(bluesky).strip() == "":
        bluesky = find_bluesky(company_clean)

    return {
        "idx": idx,
        "company": company_raw,
        "company_clean": company_clean,
        "domain": domain,
        "ir_page": ir_page,
        "ir_email": ir_email,
        "cs_page": cs_page,
        "cs_email": cs_email,
        "twitter_url": twitter,
        "facebook_url": facebook,
        "bluesky_url": bluesky,
    }


def run_missing_extraction(
    in_file: str,
    out_file: str,
    checkpoint_every: int = 50,
) -> pd.DataFrame:
    full = pd.read_csv(in_file)

    for col in ["twitter_url", "facebook_url", "bluesky_url"]:
        if col not in full.columns:
            full[col] = None

    missing = full[
        (full["domain"].isna()) |
        (full["domain"].astype(str).str.len() < 3) |
        (full["ir_page"].isna()) |
        (full["cs_page"].isna()) |
        (full["twitter_url"].isna()) |
        (full["facebook_url"].isna()) |
        (full["bluesky_url"].isna())
    ].copy()

    print(f"Total rows needing fixes: {len(missing)}")
    full_fixed = full.copy()

    updated_batch: List[Dict[str, Optional[str]]] = []

    for i, (idx, row) in enumerate(tqdm(missing.iterrows(), total=len(missing), desc="Fixing rows")):
        updated_batch.append(extract_missing_info(row, idx))

        if (i + 1) % checkpoint_every == 0:
            _apply_batch(full_fixed, updated_batch)
            full_fixed.to_csv(out_file, index=False)
            print(f"Checkpoint saved after {i + 1} missing rows")
            updated_batch = []

    if updated_batch:
        _apply_batch(full_fixed, updated_batch)
        full_fixed.to_csv(out_file, index=False)
        print("Final checkpoint saved")

    return full_fixed


def _apply_batch(full_df: pd.DataFrame, batch: List[Dict[str, Optional[str]]]) -> None:
    batch_df = pd.DataFrame(batch)
    for _, row in batch_df.iterrows():
        idx = row["idx"]
        full_df.loc[idx, [
            "company", "company_clean", "domain",
            "ir_page", "ir_email",
            "cs_page", "cs_email",
            "twitter_url", "facebook_url", "bluesky_url"
        ]] = [
            row["company"], row["company_clean"], row["domain"],
            row["ir_page"], row["ir_email"],
            row["cs_page"], row["cs_email"],
            row["twitter_url"], row["facebook_url"], row["bluesky_url"],
        ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Task 3 contact enrichment pipeline")
    parser.add_argument("--input", required=True, help="Input CSV with at least a 'company' column")
    parser.add_argument("--output-dir", required=True, help="Directory for Task 3 outputs")
    parser.add_argument("--max-companies", type=int, default=None, help="Optional cap for testing")
    parser.add_argument("--run-social", action="store_true", help="Also extract Twitter/Facebook/Bluesky links")
    parser.add_argument("--fix-missing", action="store_true", help="Retry rows with missing fields after extraction")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_companies_file = out_dir / "unique_companies.csv"
    contacts_file = out_dir / "company_contacts_full.csv"

    companies = prepare_unique_companies(args.input, str(unique_companies_file))
    print(f"Prepared {len(companies)} unique companies")

    contacts_df = run_contact_extraction(
        companies=companies,
        out_file=str(contacts_file),
        max_companies=args.max_companies,
    )
    print(f"Contact extraction complete: {len(contacts_df)} rows")

    if args.run_social:
        contacts_df = run_social_extraction(
            in_file=str(contacts_file),
            out_file=str(contacts_file),
            max_companies=args.max_companies,
        )
        print("Social enrichment complete")

    if args.fix_missing:
        contacts_df = run_missing_extraction(
            in_file=str(contacts_file),
            out_file=str(contacts_file),
            checkpoint_every=50,
        )
        print("Missing-row retry complete")

    print(f"Final output: {contacts_file}")


if __name__ == "__main__":
    main()
