#!/usr/bin/env python3
"""
clean_corpus.py — Pre-ingestion data cleaning for LEC Trade Intelligence Agent corpus.
Run this BEFORE ingestion. Cleans the raw files in data/raw/ in-place.
"""

import csv
import re
import os
import shutil

RAW_DIR = "data/raw"
BACKUP_DIR = "data/raw/backup"
os.makedirs(BACKUP_DIR, exist_ok=True)


def backup(path):
    backup_path = os.path.join(BACKUP_DIR, os.path.basename(path))
    shutil.copy2(path, backup_path)
    print(f"  ↳ backed up to {backup_path}")


def clean_sanctions_csv(path):
    """Remove the metadata header row (row 1) from UK Sanctions List CSV."""
    backup(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Row 0 is "Report Date: 16-Apr-2026,..." — skip it
    # Row 1 is actual headers: Last Updated, Unique ID, OFSI Group ID...
    # Data starts at row 2
    cleaned = [rows[1]] + rows[2:]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(cleaned)
    print(f"  ↳ removed metadata header row, {len(cleaned)-1} data rows remain")


def clean_tariff_csv(path):
    """Rename double-underscore columns to simple names."""
    backup(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        original_headers = reader.fieldnames
        rows = list(reader)

    rename = {
        "commodity__code": "commodity_code",
        "commodity__description": "description",
        "measure__duty_expression": "duty_rate",
        "measure__type__description": "measure_type",
        "measure__geographical_area__description": "origin_rules",
        "measure__regulation__id": "regulation_id",
        "measure__effective_start_date": "effective_start_date",
        "measure__effective_end_date": "effective_end_date",
    }

    renamed_headers = [rename.get(h, h) for h in original_headers]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=renamed_headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({rename.get(k, k): v for k, v in row.items()})

    print(f"  ↳ renamed {len(rows)} rows")


def parse_html_fragment(html_str):
    """Extract clean text from an HTML fragment cell."""
    from bs4 import BeautifulSoup
    if not html_str or not html_str.strip():
        return ""
    soup = BeautifulSoup(html_str, "lxml")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_lec_csv(path, label=""):
    """
    Parse LEC website CSV — each cell contains HTML fragments.
    Extract clean text, concatenate all non-empty cells per row,
    produce one clean text chunk per row.
    """
    backup(path)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    cleaned_rows = []
    for row in rows:
        texts = []
        for cell in row:
            text = parse_html_fragment(cell)
            if text:
                texts.append(text)
        if texts:
            combined = " ".join(texts)
            cleaned_rows.append([combined])

    out_path = path
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        writer.writerows(cleaned_rows)

    print(f"  ↳ {label} extracted {len(cleaned_rows)} clean text chunks")


def main():
    print("\n=== LEC Corpus Cleaning Script ===\n")

    sanctions_path = os.path.join(RAW_DIR, "UK-Sanctions-List.csv")
    if os.path.exists(sanctions_path):
        print("Cleaning UK Sanctions List...")
        clean_sanctions_csv(sanctions_path)
    else:
        print(f"  ⚠️  File not found: {sanctions_path}")

    tariff_path = os.path.join(RAW_DIR, "uk-tariff-2021-01-01--v4.0.1477--measures-as-defined.csv")
    if os.path.exists(tariff_path):
        print("Cleaning UK Tariff CSV...")
        clean_tariff_csv(tariff_path)
        with open(tariff_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
        expected = ["commodity_code", "description", "duty_rate"]
        missing = [c for c in expected if c not in headers]
        if missing:
            print(f"  ⚠️  Missing expected columns: {missing}")
        else:
            print(f"  ✓  All expected columns present: {', '.join(expected)}")
    else:
        print(f"  ⚠️  File not found: {tariff_path}")

    about_path = os.path.join(RAW_DIR, "londonexportcorp (1).csv")
    if os.path.exists(about_path):
        print("Cleaning LEC About page (1953 story)...")
        clean_lec_csv(about_path, label="About page: ")
    else:
        print(f"  ⚠️  File not found: {about_path}")

    pastworks_path = os.path.join(RAW_DIR, "londonexportcorp.csv")
    if os.path.exists(pastworks_path):
        print("Cleaning LEC Past Works page...")
        clean_lec_csv(pastworks_path, label="Past works: ")
    else:
        print(f"  ⚠️  File not found: {pastworks_path}")

    tsingtao_path = os.path.join(RAW_DIR, "青岛啤酒2024年年报-20250423.pdf")
    if os.path.exists(tsingtao_path):
        size_mb = os.path.getsize(tsingtao_path) / 1e6
        print(f"\nTsingtao PDF: ✓ present ({size_mb:.1f} MB)")
    else:
        print(f"\n⚠️  Tsingtao PDF not found: {tsingtao_path}")

    hmrc_path = os.path.join(RAW_DIR, "Force_of_law_guidance_for_Alcohol_Duty.pdf")
    if os.path.exists(hmrc_path):
        size_mb = os.path.getsize(hmrc_path) / 1e6
        print(f"HMRC PDF: ✓ present ({size_mb:.1f} MB)")
    else:
        print(f"⚠️  HMRC PDF not found: {hmrc_path}")

    print("\n=== Cleaning complete ===")
    print(f"Original files backed up to: {BACKUP_DIR}/")


if __name__ == "__main__":
    main()
