
#!/usr/bin/env python3
"""
firebase_integration.py

Best-effort production script based on the Task 5 notebook, which invoked an
`uploader.py` script to push company contacts and subsidiary data to Firebase.

What this script does:
- Loads contacts and subsidiary CSV files
- Optionally filters to a single company
- Prepares a per-company payload
- Supports dry-run preview mode
- Uploads data to Firestore when dry-run is disabled

Expected behavior inferred from the notebook:
    python uploader.py \
      --contacts-csv company_contacts_full.csv \
      --subsidiary-csv company_subsidiary.csv \
      --firebase-credentials ./firebase-credentials.json \
      --firebase-project guu-task-5 \
      --dry-run

and

    python uploader.py \
      --contacts-csv company_contacts_full.csv \
      --subsidiary-csv company_subsidiary.csv \
      --firebase-credentials ./firebase-credentials.json \
      --firebase-project guu-task-5 \
      --single-company "3M Co" \
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class CompanyBundle:
    company_name: str
    contacts: List[Dict[str, Any]]
    subsidiaries: List[Dict[str, Any]]

    def summary(self) -> Dict[str, Any]:
        return {
            "company_name": self.company_name,
            "contacts_count": len(self.contacts),
            "subsidiaries_count": len(self.subsidiaries),
        }


def safe_slug(text: str) -> str:
    """Normalize text into a Firestore-safe document id."""
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "unknown-company"


def normalize_company_column(df: pd.DataFrame, label: str) -> str:
    """
    Detect the likely company-name column.
    """
    candidates = [
        "company",
        "company_name",
        "issuer_name",
        "parent_company",
        "name",
    ]
    lowered = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    raise ValueError(
        f"Could not find a company column in {label}. "
        f"Available columns: {list(df.columns)}"
    )


def row_to_clean_dict(row: pd.Series) -> Dict[str, Any]:
    """
    Convert a pandas row to a JSON-serializable dict without NaN values.
    """
    out: Dict[str, Any] = {}
    for key, value in row.items():
        if pd.isna(value):
            continue
        if hasattr(value, "item"):
            try:
                value = value.item()
            except Exception:
                pass
        out[str(key)] = value
    return out


def load_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return pd.read_csv(path)


def build_company_bundles(
    contacts_df: pd.DataFrame,
    subsidiary_df: pd.DataFrame,
    single_company: Optional[str] = None,
) -> List[CompanyBundle]:
    contacts_company_col = normalize_company_column(contacts_df, "contacts CSV")
    subsidiary_company_col = normalize_company_column(subsidiary_df, "subsidiary CSV")

    contacts_df = contacts_df.copy()
    subsidiary_df = subsidiary_df.copy()

    contacts_df["_company_norm"] = contacts_df[contacts_company_col].astype(str).str.strip()
    subsidiary_df["_company_norm"] = subsidiary_df[subsidiary_company_col].astype(str).str.strip()

    if single_company:
        target = single_company.strip().lower()
        contacts_df = contacts_df[
            contacts_df["_company_norm"].str.lower() == target
        ]
        subsidiary_df = subsidiary_df[
            subsidiary_df["_company_norm"].str.lower() == target
        ]

    company_names = sorted(
        set(contacts_df["_company_norm"].dropna().tolist())
        | set(subsidiary_df["_company_norm"].dropna().tolist())
    )

    bundles: List[CompanyBundle] = []
    for company in company_names:
        company_contacts = contacts_df.loc[
            contacts_df["_company_norm"] == company
        ].drop(columns=["_company_norm"], errors="ignore")
        company_subs = subsidiary_df.loc[
            subsidiary_df["_company_norm"] == company
        ].drop(columns=["_company_norm"], errors="ignore")

        bundles.append(
            CompanyBundle(
                company_name=company,
                contacts=[row_to_clean_dict(row) for _, row in company_contacts.iterrows()],
                subsidiaries=[row_to_clean_dict(row) for _, row in company_subs.iterrows()],
            )
        )

    return bundles


def print_dry_run_preview(bundles: Iterable[CompanyBundle], limit: int = 5) -> None:
    bundles = list(bundles)
    print(f"Prepared {len(bundles)} company bundle(s).")
    for bundle in bundles[:limit]:
        print(json.dumps(bundle.summary(), indent=2, default=str))
    if len(bundles) > limit:
        print(f"... {len(bundles) - limit} additional bundle(s) not shown")


def upload_to_firestore(
    bundles: Iterable[CompanyBundle],
    firebase_credentials: str | Path,
    firebase_project: Optional[str] = None,
    collection_name: str = "companies",
) -> None:
    """
    Upload bundles to Firestore.

    Schema:
      companies/{company_slug}
        - company_name
        - contacts_count
        - subsidiaries_count

      companies/{company_slug}/contacts/{index}
      companies/{company_slug}/subsidiaries/{index}
    """
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
    except ImportError as exc:
        raise ImportError(
            "firebase_admin is required for live uploads. "
            "Install it with: pip install firebase-admin"
        ) from exc

    cred = credentials.Certificate(str(firebase_credentials))
    app_kwargs: Dict[str, Any] = {}
    if firebase_project:
        app_kwargs["projectId"] = firebase_project

    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(cred, app_kwargs)

    db = firestore.client()

    for bundle in bundles:
        doc_id = safe_slug(bundle.company_name)
        company_ref = db.collection(collection_name).document(doc_id)

        company_ref.set(bundle.summary(), merge=True)

        for idx, contact in enumerate(bundle.contacts):
            company_ref.collection("contacts").document(f"{idx:05d}").set(contact)

        for idx, sub in enumerate(bundle.subsidiaries):
            company_ref.collection("subsidiaries").document(f"{idx:05d}").set(sub)

    print("Upload complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload contacts + subsidiary data to Firebase/Firestore."
    )
    parser.add_argument(
        "--contacts-csv",
        required=True,
        help="Path to the company contacts CSV.",
    )
    parser.add_argument(
        "--subsidiary-csv",
        required=True,
        help="Path to the company subsidiary CSV.",
    )
    parser.add_argument(
        "--firebase-credentials",
        help="Path to a Firebase service-account JSON file.",
    )
    parser.add_argument(
        "--firebase-project",
        help="Firebase / Google Cloud project id.",
    )
    parser.add_argument(
        "--single-company",
        help="Optional company name filter for testing one company at a time.",
    )
    parser.add_argument(
        "--collection-name",
        default="companies",
        help="Top-level Firestore collection name (default: companies).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the upload payload without writing to Firestore.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    contacts_df = load_csv(args.contacts_csv, "Contacts CSV")
    subsidiary_df = load_csv(args.subsidiary_csv, "Subsidiary CSV")

    bundles = build_company_bundles(
        contacts_df=contacts_df,
        subsidiary_df=subsidiary_df,
        single_company=args.single_company,
    )

    if args.dry_run:
        print_dry_run_preview(bundles)
        return

    if not args.firebase_credentials:
        raise ValueError(
            "--firebase-credentials is required unless --dry-run is used."
        )

    upload_to_firestore(
        bundles=bundles,
        firebase_credentials=args.firebase_credentials,
        firebase_project=args.firebase_project,
        collection_name=args.collection_name,
    )


if __name__ == "__main__":
    main()
