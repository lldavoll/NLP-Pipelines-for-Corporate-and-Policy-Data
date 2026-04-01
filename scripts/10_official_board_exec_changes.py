from __future__ import annotations

import argparse
import base64
import csv
import datetime as dt
import json
import os
import re
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Dict, List, Optional, Tuple

from tqdm.auto import tqdm


@dataclass
class Config:
    gmail_query: str = "from:alert@news.theofficialboard.com"
    max_messages: Optional[int] = None
    credentials_json_path: str = "credentials.json"
    token_path: str = "token.json"
    checkpoint_path: str = "official_board_checkpoint.jsonl"
    extracted_csv: str = "official_board_exec_changes.csv"
    failures_csv: str = "official_board_parse_failures.csv"


EXTRACTED_FIELDS = [
    "message_id",
    "email_date",
    "email_subject",
    "company",
    "industry",
    "subsidiary_of",
    "section_type",
    "event_type",
    "executive_name",
    "old_position",
    "new_position",
    "effective_date",
    "posted_date",
    "raw_event_text",
]


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
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_json_path, scopes
            )
            creds = flow.run_local_server(port=0)

        with open(token_path, "w", encoding="utf-8") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


def list_message_ids(service, user_id: str, query: str, max_messages: Optional[int]) -> List[str]:
    ids: List[str] = []
    page_token = None

    while True:
        resp = service.users().messages().list(
            userId=user_id,
            q=query,
            pageToken=page_token,
            maxResults=500,
        ).execute()

        for message in resp.get("messages", []) or []:
            ids.append(message["id"])
            if max_messages and len(ids) >= max_messages:
                return ids

        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    return ids


def get_message(service, user_id: str, message_id: str) -> dict:
    return service.users().messages().get(
        userId=user_id,
        id=message_id,
        format="full",
    ).execute()


def get_header(headers: List[dict], name: str) -> str:
    for header in headers:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", "")
    return ""


def decode_b64url(data: str) -> str:
    if not data:
        return ""
    pad = len(data) % 4
    if pad:
        data += "=" * (4 - pad)
    return base64.urlsafe_b64decode(data).decode("utf-8", errors="replace")


def extract_plaintext(payload: dict) -> str:
    def walk(part: dict) -> List[dict]:
        parts = [part]
        for child in part.get("parts", []) or []:
            parts.extend(walk(child))
        return parts

    for part in walk(payload):
        if part.get("mimeType") == "text/plain":
            return decode_b64url(part.get("body", {}).get("data", ""))

    return decode_b64url(payload.get("body", {}).get("data", ""))


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


def append_checkpoint(path: str, message_id: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"message_id": message_id}) + "\n")


def write_csv(path: str, rows: List[dict], fields: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


INDUSTRY_TERMS = [
    "Aerospace",
    "Airlines",
    "Banking",
    "Biotechnology",
    "Broadcasting",
    "Business Services",
    "Casinos",
    "Communication & Sales",
    "Construction",
    "Consumer Electronics",
    "Financial Services",
    "Fund",
    "Furniture",
    "Holding",
    "Hotels",
    "Industrial Conglomerates",
    "Insurance",
    "Machinery",
    "Materials",
    "Pharmaceuticals",
    "Real Estate",
    "Recruiting",
    "Reinsurance",
    "Retail",
    "Semiconductors",
    "Software",
    "Telecommunications",
    "Video Games",
]

INDUSTRY_PATTERN = "|".join(
    sorted(map(re.escape, INDUSTRY_TERMS), key=len, reverse=True)
)

FOOTER_PATTERNS = [
    r"Your have reached the maximum level of information available.*",
    r"Click to create your signals on.*",
    r"Manage your existing alerts.*",
    r"Essentials — Monthly tips.*",
    r"Get Essentials.*",
    r"Org trends — Leadership changes.*",
    r"Get Org Trends.*",
    r"This alert is sent once a month by The Official Board\..*",
    r"Unsubscribe to this alert.*",
]

MONTH_LINE_RE = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}",
    flags=re.IGNORECASE,
)

POSTED_DATE_RE = re.compile(r"\(posted on ([^)]+)\)", flags=re.IGNORECASE)
EFFECTIVE_DATE_RE = re.compile(r"\bon ([A-Z][a-z]{2,}\.? \d{1,2}, \d{4})\b")


def clean_official_board_text(body: str) -> str:
    text = body or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n?\[https?://[^\]]+\]\s*", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"THE OFFICIAL BOARD\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"/\s*MY ALERTS\s*", "\n", text, flags=re.IGNORECASE)

    text = re.sub(r"(Appointments)", r"\n\1\n", text)
    text = re.sub(r"(Changes)", r"\n\1\n", text)
    text = re.sub(r"(Subsidiary of [^\n]+)", r"\n\1\n", text)
    text = re.sub(r"(View the new org chart of [^\n]+)", r"\n\1\n", text)
    text = re.sub(r"(/\s*Report an error)", r"\n\1\n", text)

    text = re.sub(
        r"((January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})",
        r"\1\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        rf"([A-Za-z0-9&.,'’/\- )]+?)({INDUSTRY_PATTERN})\b",
        r"\1\n\2",
        text,
    )

    text = re.sub(r"(?<!\n)([A-Z][A-Za-z0-9&.'’\- ]+\s+has joined the company as\b)", r"\n\1", text)
    text = re.sub(r"(?<!\n)([A-Z][A-Za-z0-9&.'’\- ]+\s+will join the company as\b)", r"\n\1", text)
    text = re.sub(r"(?<!\n)([A-Z][A-Za-z0-9&.'’\- ]+,\s+who was\b)", r"\n\1", text)
    text = re.sub(r"(?<!\n)([A-Z][A-Za-z0-9&.'’\- ]+\s+who was\b)", r"\n\1", text)
    text = re.sub(r"(?<!\n)([A-Z][A-Za-z0-9&.'’\- ]+,\s+who is\b)", r"\n\1", text)
    text = re.sub(r"(?<!\n)([A-Z][A-Za-z0-9&.'’\- ]+\s+who is\b)", r"\n\1", text)

    text = re.sub(r"\(posted on [^)]+\)\.?", lambda m: m.group(0) + "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+,", ",", text)
    text = re.sub(r"\s+\.", ".", text)
    text = re.sub(r"[ \t]+", " ", text)

    for pattern in FOOTER_PATTERNS:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"View\s+the\s+new\s+org\s+chart\s+of\s+", "\nView the new org chart of ", text, flags=re.IGNORECASE)
    text = re.sub(r"/\s*Report\s+an\s+error", "\n/ Report an error\n", text, flags=re.IGNORECASE)

    return text.strip()


def split_company_blocks(text: str) -> List[Dict[str, str]]:
    blocks: List[Dict[str, str]] = []

    text = re.sub(r"View\s+the\s+new\s+org\s+chart\s+of\s+", "View the new org chart of ", text, flags=re.IGNORECASE)
    text = re.sub(r"/\s*Report\s+an\s+error", "/ Report an error", text, flags=re.IGNORECASE)

    raw_blocks = re.split(r"\nView the new org chart of .*?(?=\n|$)", text)

    for raw in raw_blocks:
        raw = raw.strip()
        if not raw:
            continue

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        if len(lines) < 3:
            continue

        while lines and (
            lines[0].upper().startswith("THE OFFICIAL BOARD")
            or lines[0].lower().startswith("forwarded message")
            or lines[0].lower().startswith("from:")
            or lines[0].lower().startswith("date:")
            or lines[0].lower().startswith("subject:")
            or lines[0].lower().startswith("to:")
            or MONTH_LINE_RE.match(lines[0])
            or lines[0] == "/ Report an error"
        ):
            lines = lines[1:]

        while lines and lines[-1] == "/ Report an error":
            lines = lines[:-1]

        if len(lines) < 3:
            continue

        company = lines[0]
        industry = lines[1]
        if industry not in INDUSTRY_TERMS:
            continue

        idx = 2
        subsidiary_of = ""
        if idx < len(lines) and lines[idx].startswith("Subsidiary of "):
            subsidiary_of = lines[idx].replace("Subsidiary of ", "").strip()
            idx += 1

        block_body = "\n".join(lines[idx:]).strip()
        if "Appointments" not in block_body and "Changes" not in block_body:
            continue

        blocks.append(
            {
                "company": company,
                "industry": industry,
                "subsidiary_of": subsidiary_of,
                "block_body": block_body,
            }
        )

    return blocks


def split_sections(block_body: str) -> List[Tuple[str, List[str]]]:
    text = block_body.strip()
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(\(posted on [^)]+\)\.)([A-Z])", r"\1\n\2", text, flags=re.IGNORECASE)
    text = re.sub(r"(\(posted on [^)]+\))([A-Z])", r"\1\n\2", text, flags=re.IGNORECASE)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    sections: List[Tuple[str, List[str]]] = []
    current_section: Optional[str] = None
    current_lines: List[str] = []

    def flush_section() -> None:
        nonlocal current_section, current_lines, sections
        if current_section is None:
            return

        section_text = "\n".join(current_lines).strip()
        section_text = re.sub(
            r"(\(posted on [^)]+\)\.?)",
            r"\1<<<EVENT_SPLIT>>>",
            section_text,
            flags=re.IGNORECASE,
        )

        raw_events = section_text.split("<<<EVENT_SPLIT>>>")
        events: List[str] = []
        for event in raw_events:
            event = event.strip()
            if not event:
                continue
            event = re.sub(r"\s+", " ", event).strip()
            events.append(event)

        sections.append((current_section, events))
        current_lines = []

    for line in lines:
        if line in {"Appointments", "Changes"}:
            flush_section()
            current_section = line
            current_lines = []
        else:
            if current_section is not None:
                current_lines.append(line)

    flush_section()
    return sections


def extract_posted_date(text: str) -> str:
    match = POSTED_DATE_RE.search(text)
    return match.group(1).strip() if match else ""


def extract_effective_date(text: str) -> str:
    match = EFFECTIVE_DATE_RE.search(text)
    return match.group(1).strip() if match else ""


def clean_event_text(line: str) -> str:
    line = line.strip()
    line = re.sub(r"\s+", " ", line)
    line = re.sub(r"\s+,", ",", line)
    line = re.sub(r"\s+\.", ".", line)
    line = re.sub(r"\(\s*posted on", "(posted on", line, flags=re.IGNORECASE)
    return line


def remove_posted_suffix(line: str) -> str:
    return re.sub(r"\s*\(posted on [^)]+\)\.?$", "", line, flags=re.IGNORECASE).strip()


def parse_event_line(line: str) -> Dict[str, str]:
    raw_event_text = clean_event_text(line)
    posted_date = extract_posted_date(raw_event_text)
    effective_date = extract_effective_date(raw_event_text)
    core = remove_posted_suffix(raw_event_text)

    result = {
        "event_type": "",
        "executive_name": "",
        "old_position": "",
        "new_position": "",
        "effective_date": effective_date,
        "posted_date": posted_date,
        "raw_event_text": raw_event_text,
    }

    patterns = [
        (r"^(.*?)(?: has joined the company as )(.+?)\.?$", "joined", ("executive_name", "new_position")),
        (r"^(.*?)(?: will join the company as )(.+?)(?: on .+?)?\.?$", "will_join", ("executive_name", "new_position")),
        (r"^(.*?), who was (.*?), is promoted to (.+?)\.?$", "promoted", ("executive_name", "old_position", "new_position")),
        (r"^(.*?)(?: who was )(.*?)(?:, becomes )(.+?)\.?$", "became", ("executive_name", "old_position", "new_position")),
        (r"^(.*?)(?: who is )(.*?)(?:, will be promoted to )(.+?)(?: on .+?)?\.?$", "will_be_promoted", ("executive_name", "old_position", "new_position")),
        (r"^(.*?)(?: who is )(.*?)(?:, will become )(.+?)(?: on .+?)?\.?$", "will_become", ("executive_name", "old_position", "new_position")),
        (r"^(.*?), (.*?), has left the company\.?$", "left", ("executive_name", "old_position")),
        (r"^(.*?), (.*?), will leave the company(?: on .+?)?\.?$", "will_leave", ("executive_name", "old_position")),
        (r"^(.*?), (.*?), will retire(?: on .+?)?\.?$", "will_retire", ("executive_name", "old_position")),
        (r"^(.*?), who is (.*?)(?: will soon retire)\.?$", "will_retire", ("executive_name", "old_position")),
    ]

    for pattern, event_type, fields in patterns:
        match = re.match(pattern, core, flags=re.IGNORECASE)
        if not match:
            continue
        result["event_type"] = event_type
        groups = [g.strip(" ,.") for g in match.groups()]
        for key, value in zip(fields, groups):
            result[key] = value
        return result

    result["event_type"] = "unparsed"
    return result


def parse_company_block(
    block: Dict[str, str],
    message_id: str,
    email_date: str,
    email_subject: str,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    extracted_rows: List[Dict[str, str]] = []
    failure_rows: List[Dict[str, str]] = []

    company = block["company"]
    industry = block["industry"]
    subsidiary_of = block["subsidiary_of"]
    block_body = block["block_body"]

    sections = split_sections(block_body)

    for section_type, lines in sections:
        for line in lines:
            parsed = parse_event_line(line)
            row = {
                "message_id": message_id,
                "email_date": email_date,
                "email_subject": email_subject,
                "company": company,
                "industry": industry,
                "subsidiary_of": subsidiary_of,
                "section_type": section_type,
                "event_type": parsed["event_type"],
                "executive_name": parsed["executive_name"],
                "old_position": parsed["old_position"],
                "new_position": parsed["new_position"],
                "effective_date": parsed["effective_date"],
                "posted_date": parsed["posted_date"],
                "raw_event_text": parsed["raw_event_text"],
            }
            extracted_rows.append(row)
            if parsed["event_type"] == "unparsed":
                failure_rows.append(row.copy())

    return extracted_rows, failure_rows


def run(cfg: Config) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    service = build_gmail_service(
        credentials_json_path=cfg.credentials_json_path,
        token_path=cfg.token_path,
    )

    processed = load_processed_ids(cfg.checkpoint_path)
    message_ids = list_message_ids(service, "me", cfg.gmail_query, cfg.max_messages)
    to_process = [message_id for message_id in message_ids if message_id not in processed]

    print(f"Found {len(message_ids)} messages.")
    print(f"Skipping {len(message_ids) - len(to_process)} already processed.")
    print(f"Processing {len(to_process)} messages.\n")

    extracted: List[Dict[str, str]] = []
    failures: List[Dict[str, str]] = []

    progress = tqdm(
        to_process,
        total=len(to_process),
        desc="Processing Official Board emails",
        unit="email",
        dynamic_ncols=True,
    )

    for message_id in progress:
        msg = get_message(service, "me", message_id)
        payload = msg.get("payload", {})
        headers = payload.get("headers", [])

        subject = get_header(headers, "Subject")
        date_raw = get_header(headers, "Date")

        try:
            msg_dt = parsedate_to_datetime(date_raw)
        except Exception:
            internal_ms = int(msg.get("internalDate", "0"))
            msg_dt = dt.datetime.fromtimestamp(internal_ms / 1000.0, tz=dt.timezone.utc)

        body = extract_plaintext(payload)
        body = clean_official_board_text(body)
        blocks = split_company_blocks(body)

        for block in blocks:
            rows, bad_rows = parse_company_block(
                block=block,
                message_id=message_id,
                email_date=msg_dt.isoformat(),
                email_subject=subject,
            )
            extracted.extend(rows)
            failures.extend(bad_rows)

        append_checkpoint(cfg.checkpoint_path, message_id)
        progress.set_postfix(rows=len(extracted), failures=len(failures))

    write_csv(cfg.extracted_csv, extracted, EXTRACTED_FIELDS)
    write_csv(cfg.failures_csv, failures, EXTRACTED_FIELDS)

    print("\nRun complete.")
    print(f"Total extracted rows: {len(extracted)}")
    print(f"Total parse failures: {len(failures)}")

    return extracted, failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract structured executive change events from Official Board Gmail alerts."
    )
    parser.add_argument("--credentials-json", default="credentials.json", help="Path to Gmail OAuth credentials JSON.")
    parser.add_argument("--token-json", default="token.json", help="Path to stored Gmail OAuth token JSON.")
    parser.add_argument("--gmail-query", default="from:alert@news.theofficialboard.com", help="Gmail search query.")
    parser.add_argument("--max-messages", type=int, default=None, help="Maximum number of Gmail messages to process.")
    parser.add_argument("--checkpoint-path", default="official_board_checkpoint.jsonl", help="Checkpoint JSONL path.")
    parser.add_argument("--extracted-csv", default="official_board_exec_changes.csv", help="Output CSV for extracted rows.")
    parser.add_argument("--failures-csv", default="official_board_parse_failures.csv", help="Output CSV for unparsed rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        gmail_query=args.gmail_query,
        max_messages=args.max_messages,
        credentials_json_path=args.credentials_json,
        token_path=args.token_json,
        checkpoint_path=args.checkpoint_path,
        extracted_csv=args.extracted_csv,
        failures_csv=args.failures_csv,
    )
    run(cfg)


if __name__ == "__main__":
    main()
