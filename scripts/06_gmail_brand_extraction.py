
from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from ftfy import fix_text
from tqdm.auto import tqdm

try:
    from rapidfuzz import fuzz, process
    HAVE_RAPIDFUZZ = True
except Exception:
    HAVE_RAPIDFUZZ = False


# ============================================================================
# Stage 1: Gmail extraction
# ============================================================================

@dataclass
class ExtractConfig:
    gmail_query: str = '(subject:"New Company Request via iOS" OR subject:"New Company Request via Android") before:2025/01/01'
    max_messages: Optional[int] = None
    checkpoint_path: str = "brand_email_checkpoint.jsonl"
    extracted_csv: str = "brand_request_extracted.csv"
    counts_csv: str = "brand_request_counts.csv"
    alias_review_csv: str = "alias_review.csv"
    auto_merge_threshold: int = 90
    review_threshold_low: int = 80


def build_gmail_service(
    credentials_json_path: str = "credentials.json",
    token_path: str = "token.json",
):
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_json_path, scopes)
            creds = flow.run_local_server(port=0)
        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def list_message_ids(service, user_id: str, query: str, max_messages: Optional[int]):
    ids: list[str] = []
    page_token = None

    while True:
        resp = (
            service.users()
            .messages()
            .list(userId=user_id, q=query, pageToken=page_token, maxResults=500)
            .execute()
        )

        for message in resp.get("messages", []) or []:
            ids.append(message["id"])
            if max_messages and len(ids) >= max_messages:
                return ids

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return ids


def get_message(service, user_id: str, message_id: str):
    return service.users().messages().get(userId=user_id, id=message_id, format="full").execute()


def get_header(headers, name: str) -> str:
    for h in headers:
        if h.get("name", "").lower() == name.lower():
            return h.get("value", "")
    return ""


def decode_b64url(data: str) -> str:
    if not data:
        return ""
    pad = len(data) % 4
    if pad:
        data += "=" * (4 - pad)
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def extract_plaintext(payload: dict) -> str:
    def walk(part):
        parts = [part]
        for child in part.get("parts", []) or []:
            parts.extend(walk(child))
        return parts

    for part in walk(payload):
        if part.get("mimeType") == "text/plain":
            return decode_b64url(part.get("body", {}).get("data", ""))

    return decode_b64url(payload.get("body", {}).get("data", ""))


BRAND_BLOCK_RE = re.compile(
    r"i['’]d\s+like\s+to\s+know\s+more\s+details\s+about:\s*(.*?)\s*thanks",
    flags=re.IGNORECASE | re.DOTALL,
)
SUBJECT_FALLBACK_RE = re.compile(
    r"details\s+about:\s*(.*?)\s*(?:thanks|\Z)",
    flags=re.IGNORECASE | re.DOTALL,
)
TRAILING_JUNK_RE = re.compile(
    r"""
    (.*?)
    (?:
        (?:\.\s*based\s+in\b.*)$
      | (?:\bbased\s+in\b.*)$
      | (?:\blocated\s+in\b.*)$
      | (?:\bbased\s+out\s+of\b.*)$
    )
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
PAREN_LIST_RE = re.compile(r"^\s*(.*?)\s*\((.*?)\)\s*$", flags=re.DOTALL)


def extract_brand_block(body: str, subject: str) -> Tuple[Optional[str], str]:
    if body:
        match = BRAND_BLOCK_RE.search(body)
        if match:
            raw = match.group(1).strip()
            return (raw if raw else None), "body"

    if subject:
        match = SUBJECT_FALLBACK_RE.search(subject)
        if match:
            raw = match.group(1).strip()
            return (raw if raw else None), "subject"

    return None, "none"


def _split_commas_outside_parens(text: str) -> List[str]:
    out, buf, depth = [], [], 0
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)

        if ch == "," and depth == 0:
            piece = "".join(buf).strip()
            if piece:
                out.append(piece)
            buf = []
        else:
            buf.append(ch)

    last = "".join(buf).strip()
    if last:
        out.append(last)
    return out


def split_brands(raw: str) -> List[str]:
    raw = (raw or "").strip()
    if not raw:
        return []

    junk_match = TRAILING_JUNK_RE.match(raw)
    if junk_match:
        raw = (junk_match.group(1) or "").strip()

    parts = []
    for chunk in raw.split("/"):
        chunk = re.sub(r"\s+", " ", chunk).strip()
        if chunk:
            parts.append(chunk)

    tmp = []
    for part in parts:
        tmp.extend(_split_commas_outside_parens(part))
    parts = [re.sub(r"\s+", " ", p).strip() for p in tmp if p.strip()]

    expanded = []
    for part in parts:
        paren_match = PAREN_LIST_RE.match(part)
        if paren_match:
            main = paren_match.group(1).strip()
            inside = paren_match.group(2).strip()
            if main:
                expanded.append(main)
            inside_items = [x.strip() for x in inside.split(",") if x.strip()]
            expanded.extend(inside_items)
        else:
            expanded.append(part)

    exceptions = {
        "wine and spirits",
        "oil and gas",
        "research and development",
    }
    generic_tails = {"spirits", "vodka", "tequila", "store", "stores", "company"}

    final = []
    for part in expanded:
        part = re.sub(r"\s+", " ", part).strip()
        lower_part = part.lower()

        if any(exc in lower_part for exc in exceptions):
            final.append(part)
            continue

        if re.search(r"\s+and\s+", part, flags=re.IGNORECASE):
            and_parts = re.split(r"\s+and\s+", part, maxsplit=1, flags=re.IGNORECASE)
            if len(and_parts) == 2:
                left, right = and_parts[0].strip(), and_parts[1].strip()
                if right.lower() in generic_tails:
                    final.append(part)
                else:
                    if left:
                        final.append(left)
                    if right:
                        final.append(right)
            else:
                final.append(part)
        else:
            final.append(part)

    return [x for x in final if x.strip()]


QUESTION_PREFIX_RE = re.compile(r"^\s*(do|does|is|are|can|could|would|should)\b", re.I)
COMMENTARY_RE = re.compile(r"\b(im assuming|i think|republican|democrat|democrats|politics|support)\b", re.I)
WEBSITE_RE = re.compile(r"\b(website\s+is|site\s+is|www\.|https?://|\.com\b|\.net\b|\.org\b)\b", re.I)

ALIAS_MAP = {
    "Cosco": "Costco",
    "Humanna": "Humana",
    "Safeways": "Safeway",
    "Sketchers": "Skechers",
    "Dominoes": "Domino's",
    "Tommy'S Car Wash": "Tommy's Car Wash",
    "Pendelton": "Pendleton",
    "Khols": "Kohls",
    "Sketcher": "Skechers",
    "Jersy Mikes": "Jersey Mike's",
    "Mcdonalds": "McDonald's",
    "Anytimefitness": "Anytime Fitness",
    "Hungry Root": "Hungryroot",
    "Hyvee": "Hy-Vee",
    "Shoprite": "Shop Rite",
}


def apply_alias(clean: str) -> str:
    clean = clean.replace("’", "'")
    return ALIAS_MAP.get(clean, clean)


def clean_candidate_brand(raw: str) -> Optional[str]:
    if not raw:
        return None

    value = re.sub(r"\s+", " ", raw).strip()
    value_low = value.lower()

    if WEBSITE_RE.search(value):
        return None

    if "?" in value:
        tail = value.split("?")[-1].strip()
        if 2 <= len(tail) <= 60:
            value = tail
            value_low = value.lower()
        else:
            return None

    if QUESTION_PREFIX_RE.search(value_low):
        return None

    if COMMENTARY_RE.search(value_low):
        if "." in value:
            tail = value.split(".")[-1].strip()
            if 2 <= len(tail) <= 60 and not COMMENTARY_RE.search(tail.lower()):
                value = tail
            else:
                return None
        else:
            return None

    generic = {
        "gas", "bank", "paper towel", "paper towels", "paper products",
        "toilet paper", "hair accessories", "a yarn store",
    }
    if value_low in generic:
        return None

    if len(value) > 80:
        return None

    if len(value.split()) > 8:
        return None

    return value


def normalize_brand(value: str) -> str:
    value = fix_text(value)
    value = re.sub(r"[.!?;:\-]+$", "", value.strip())
    value = re.sub(r"\s+", " ", value)

    if re.fullmatch(r"[A-Z0-9&]{2,}", value):
        return value
    return value.title()


def fuzzy_canonicalize(brand: str, canon: Iterable[str], cfg: ExtractConfig):
    canon = list(canon)
    if not HAVE_RAPIDFUZZ or not canon:
        return brand, None

    match = process.extractOne(brand, canon, scorer=fuzz.token_set_ratio)
    if not match:
        return brand, None

    best, score, _ = match

    if score >= cfg.auto_merge_threshold:
        return best, None

    if cfg.review_threshold_low <= score < cfg.auto_merge_threshold:
        return brand, {
            "brand_raw": brand,
            "suggested_canonical": best,
            "score": score,
        }

    return brand, None


def load_processed_ids(path: str) -> set[str]:
    seen: set[str] = set()
    if not os.path.exists(path):
        return seen

    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(json.loads(line)["message_id"])
            except Exception:
                pass
    return seen


def append_checkpoint(path: str, message_id: str):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"message_id": message_id}) + "\n")


def write_csv(path: str, rows: list[dict], fields: list[str]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_extraction(cfg: ExtractConfig, credentials_json: str, token_json: str):
    service = build_gmail_service(credentials_json, token_json)
    processed = load_processed_ids(cfg.checkpoint_path)

    msg_ids = list_message_ids(service, "me", cfg.gmail_query, cfg.max_messages)
    to_process = [mid for mid in msg_ids if mid not in processed]

    print(f"Found {len(msg_ids)} messages.")
    print(f"Skipping {len(msg_ids) - len(to_process)} already processed.")
    print(f"Processing {len(to_process)} messages.\n")

    extracted = []
    review = []
    counts: Counter[str] = Counter()
    first_seen: dict[str, dt.datetime] = {}
    last_seen: dict[str, dt.datetime] = {}

    pbar = tqdm(
        to_process,
        total=len(to_process),
        desc="Processing brand request emails",
        unit="email",
        dynamic_ncols=True,
    )

    for mid in pbar:
        msg = get_message(service, "me", mid)
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        subject = get_header(headers, "Subject")
        sender = get_header(headers, "From")
        date_raw = get_header(headers, "Date")

        try:
            msg_dt = parsedate_to_datetime(date_raw)
        except Exception:
            internal_ms = int(msg.get("internalDate", "0"))
            msg_dt = dt.datetime.fromtimestamp(internal_ms / 1000.0, tz=dt.timezone.utc)

        body = extract_plaintext(payload)
        block, source = extract_brand_block(body, subject)

        if not block:
            append_checkpoint(cfg.checkpoint_path, mid)
            continue

        canon_set = set(counts.keys())

        for raw in split_brands(block):
            cleaned_raw = clean_candidate_brand(raw)
            if not cleaned_raw:
                continue

            clean = normalize_brand(cleaned_raw)
            clean = apply_alias(clean)

            canonical = None
            for existing in canon_set:
                if existing.lower() == clean.lower():
                    canonical = existing
                    break

            rev = None
            if canonical is None:
                canonical, rev = fuzzy_canonicalize(clean, canon_set, cfg)
                if rev:
                    review.append(rev)

            counts[canonical] += 1
            canon_set.add(canonical)
            first_seen[canonical] = min(first_seen.get(canonical, msg_dt), msg_dt)
            last_seen[canonical] = max(last_seen.get(canonical, msg_dt), msg_dt)

            extracted.append(
                {
                    "message_id": mid,
                    "date": msg_dt.isoformat(),
                    "from": sender,
                    "subject": subject,
                    "brand_raw": raw,
                    "brand_clean": clean,
                    "brand_canonical": canonical,
                    "source": source,
                }
            )

        append_checkpoint(cfg.checkpoint_path, mid)
        pbar.set_postfix(rows=len(extracted), brands=len(counts))

    write_csv(
        cfg.extracted_csv,
        extracted,
        [
            "message_id",
            "date",
            "from",
            "subject",
            "brand_raw",
            "brand_clean",
            "brand_canonical",
            "source",
        ],
    )

    write_csv(
        cfg.counts_csv,
        [
            {
                "brand_canonical": brand,
                "count": cnt,
                "first_seen": first_seen[brand].isoformat(),
                "last_seen": last_seen[brand].isoformat(),
            }
            for brand, cnt in counts.most_common()
        ],
        ["brand_canonical", "count", "first_seen", "last_seen"],
    )

    write_csv(
        cfg.alias_review_csv,
        review,
        ["brand_raw", "suggested_canonical", "score"],
    )

    print(f"\nWrote extracted rows: {cfg.extracted_csv}")
    print(f"Wrote brand counts:   {cfg.counts_csv}")
    print(f"Wrote alias review:   {cfg.alias_review_csv}")


# ============================================================================
# Stage 2: Post-cleaning
# ============================================================================

STATE_HINT = r"(alabama|alaska|arizona|arkansas|california|colorado|connecticut|delaware|florida|georgia|hawaii|idaho|illinois|indiana|iowa|kansas|kentucky|louisiana|maine|maryland|massachusetts|michigan|minnesota|mississippi|missouri|montana|nebraska|nevada|new hampshire|new jersey|new mexico|new york|north carolina|north dakota|ohio|oklahoma|oregon|pennsylvania|rhode island|south carolina|south dakota|tennessee|texas|utah|vermont|virginia|washington|west virginia|wisconsin|wyoming)"
TRIM_PATTERNS = [
    r"\s*(>\s*>)+\s*",
    r"\b(sent from|thanks|thank you)\b.*$",
    r"\bdo they\b.*$",
    r"\b(does it|is it)\b.*$",
    r"\b(support|democrat|republican|progressive)\b.*$",
    r"\bi think\b.*$|\bi assume\b.*$|\bim assuming\b.*$",
    r"\bwebsite\b.*$|\bwww\..*$|\bhttps?://.*$",
    r"\bbased in\b.*$|\blocated\b.*$",
    r"\bin\s+" + STATE_HINT + r"\b.*$",
    r"\bnear me\b.*$",
    r"\bthey are\b.*$|\bparent company\b.*$|\band parent company\b.*$",
    r"\baka\b.*$|\ba\.k\.a\.\b.*$",
    r"\bwhere you can\b.*$",
    r"\b(an?|the)\s+parent\s+company\b.*$",
]

URL_RE = re.compile(r"(https?://|www\.|\.(com|net|org|io|co)\b)", re.I)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", re.I)
ALLOWED_CHARS_RE = re.compile(r"^[A-Za-z0-9&'’\-\.\s]+$")
COMMENTARY_RE_V2 = re.compile(
    r"\b("
    r"do they|does it|support|democrat|democrats|republican|republicans|progressive|"
    r"i think|i assume|im assuming|i'm assuming|"
    r"parent company|and parent company|they are the parent company|"
    r"website is|website|sent from|thanks|thank you|"
    r"where you can|based in|located|near me"
    r")\b",
    re.I,
)
DROP_EXACT = {
    "", "Stop", "Out", "Wings", "Cooling", "Etc",
    "A Meal Delivery Company",
    "Cell Phone Services Provider",
    "Fast Foods & Burgers Drive-Thru",
    "They Are The Parent Company",
}


def fix_mojibake(value: str) -> str:
    if not isinstance(value, str):
        return ""
    return (
        value.replace("â€™", "’")
        .replace("â€˜", "‘")
        .replace("â€œ", "“")
        .replace("â€\u009d", "”")
        .replace("â€“", "–")
        .replace("â€”", "—")
        .replace("Ã©", "é")
        .replace("Ã¨", "è")
        .replace("Ã¡", "á")
        .replace("Ã³", "ó")
        .replace("Ã±", "ñ")
        .replace("Ã¼", "ü")
    )


def normalize_text(value: str) -> str:
    value = fix_mojibake(value)
    value = re.sub(r"\s+", " ", value).strip()
    value = value.strip(" -–—|•\t\r\n")
    return value


def trim_commentary(value: str) -> str:
    out = normalize_text(value)
    for pattern in TRIM_PATTERNS:
        new = re.sub(pattern, " ", out, flags=re.I).strip()
        if new != out:
            out = re.sub(r"\s+", " ", new).strip()
    if ":" in out:
        out = out.split(":", 1)[0].strip()
    if "(" in out and len(out) > 12:
        out = out.split("(", 1)[0].strip()
    return out


def looks_like_brand_v2(value: str) -> tuple[str, str]:
    value = (value or "").strip()
    if not value:
        return "drop", "empty"
    if value in DROP_EXACT:
        return "drop", "drop_exact"
    if URL_RE.search(value) or EMAIL_RE.search(value):
        return "drop", "url_or_email"
    if not ALLOWED_CHARS_RE.match(value):
        return "drop", "weird_chars"
    if COMMENTARY_RE_V2.search(value):
        if len(value) > 60 or len(value.split()) > 10:
            return "drop", "commentary_long"
        return "review", "commentary"
    if len(value) > 55 or len(value.split()) > 8:
        return "review", "too_long"
    return "keep", "ok"


def run_post_cleaning(
    input_counts: str,
    clean_counts_out: str,
    review_out: str,
    dropped_out: str,
):
    df = pd.read_csv(input_counts)
    df["brand_canonical"] = df["brand_canonical"].astype(str)
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    work = df.copy()
    work["candidate"] = (
        work["brand_canonical"]
        .map(trim_commentary)
        .map(normalize_text)
        .map(apply_alias)
        .map(lambda x: re.sub(r"\s+", " ", x).strip())
    )

    labels = work["candidate"].map(looks_like_brand_v2)
    work["label"] = labels.map(lambda t: t[0])
    work["reason"] = labels.map(lambda t: t[1])

    kept = work[work["label"] == "keep"].copy()
    review = work[work["label"] == "review"].copy()
    dropped = work[work["label"] == "drop"].copy()

    clean_counts = (
        kept.groupby("candidate", as_index=False)
        .agg(
            count=("count", "sum"),
            first_seen=("first_seen", "min"),
            last_seen=("last_seen", "max"),
        )
        .sort_values("count", ascending=False)
        .rename(columns={"candidate": "brand_canonical_clean"})
    )

    clean_counts.to_csv(clean_counts_out, index=False)
    review[["brand_canonical", "candidate", "count", "reason"]].sort_values(
        "count", ascending=False
    ).to_csv(review_out, index=False)
    dropped[["brand_canonical", "candidate", "count", "reason"]].sort_values(
        "count", ascending=False
    ).to_csv(dropped_out, index=False)

    print("Input rows:", len(df))
    print("Kept rows:", len(kept), "=> unique kept:", len(clean_counts))
    print("Review rows:", len(review))
    print("Dropped rows:", len(dropped))
    print(f"Wrote cleaned counts: {clean_counts_out}")
    print(f"Wrote review file:    {review_out}")
    print(f"Wrote dropped file:   {dropped_out}")


# ============================================================================
# CLI
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description="Task 6 pipeline: extract brand requests from Gmail and post-clean the aggregated counts."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Run Gmail extraction from raw messages.")
    extract_parser.add_argument("--credentials-json", default="credentials.json")
    extract_parser.add_argument("--token-json", default="token.json")
    extract_parser.add_argument("--gmail-query", default=ExtractConfig.gmail_query)
    extract_parser.add_argument("--max-messages", type=int, default=None)
    extract_parser.add_argument("--checkpoint-path", default="brand_email_checkpoint.jsonl")
    extract_parser.add_argument("--extracted-csv", default="brand_request_extracted.csv")
    extract_parser.add_argument("--counts-csv", default="brand_request_counts.csv")
    extract_parser.add_argument("--alias-review-csv", default="alias_review.csv")
    extract_parser.add_argument("--auto-merge-threshold", type=int, default=90)
    extract_parser.add_argument("--review-threshold-low", type=int, default=80)

    clean_parser = subparsers.add_parser("clean", help="Post-clean aggregated brand counts.")
    clean_parser.add_argument("--input-counts", default="brand_request_counts.csv")
    clean_parser.add_argument("--clean-counts-out", default="brand_request_counts_clean.csv")
    clean_parser.add_argument("--review-out", default="brand_request_counts_review.csv")
    clean_parser.add_argument("--dropped-out", default="brand_request_counts_dropped.csv")

    all_parser = subparsers.add_parser("all", help="Run extraction and then post-cleaning.")
    all_parser.add_argument("--credentials-json", default="credentials.json")
    all_parser.add_argument("--token-json", default="token.json")
    all_parser.add_argument("--gmail-query", default=ExtractConfig.gmail_query)
    all_parser.add_argument("--max-messages", type=int, default=None)
    all_parser.add_argument("--checkpoint-path", default="brand_email_checkpoint.jsonl")
    all_parser.add_argument("--extracted-csv", default="brand_request_extracted.csv")
    all_parser.add_argument("--counts-csv", default="brand_request_counts.csv")
    all_parser.add_argument("--alias-review-csv", default="alias_review.csv")
    all_parser.add_argument("--auto-merge-threshold", type=int, default=90)
    all_parser.add_argument("--review-threshold-low", type=int, default=80)
    all_parser.add_argument("--clean-counts-out", default="brand_request_counts_clean.csv")
    all_parser.add_argument("--review-out", default="brand_request_counts_review.csv")
    all_parser.add_argument("--dropped-out", default="brand_request_counts_dropped.csv")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "extract":
        cfg = ExtractConfig(
            gmail_query=args.gmail_query,
            max_messages=args.max_messages,
            checkpoint_path=args.checkpoint_path,
            extracted_csv=args.extracted_csv,
            counts_csv=args.counts_csv,
            alias_review_csv=args.alias_review_csv,
            auto_merge_threshold=args.auto_merge_threshold,
            review_threshold_low=args.review_threshold_low,
        )
        run_extraction(cfg, args.credentials_json, args.token_json)

    elif args.command == "clean":
        run_post_cleaning(
            input_counts=args.input_counts,
            clean_counts_out=args.clean_counts_out,
            review_out=args.review_out,
            dropped_out=args.dropped_out,
        )

    elif args.command == "all":
        cfg = ExtractConfig(
            gmail_query=args.gmail_query,
            max_messages=args.max_messages,
            checkpoint_path=args.checkpoint_path,
            extracted_csv=args.extracted_csv,
            counts_csv=args.counts_csv,
            alias_review_csv=args.alias_review_csv,
            auto_merge_threshold=args.auto_merge_threshold,
            review_threshold_low=args.review_threshold_low,
        )
        run_extraction(cfg, args.credentials_json, args.token_json)
        run_post_cleaning(
            input_counts=args.counts_csv,
            clean_counts_out=args.clean_counts_out,
            review_out=args.review_out,
            dropped_out=args.dropped_out,
        )


if __name__ == "__main__":
    main()
