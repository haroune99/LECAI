import sqlite3
import os
import csv
import re
import time
import json
from typing import Literal, Optional
from dataclasses import dataclass
from src.tools.base import ToolResult


KB_DIR = "data/processed"
os.makedirs(KB_DIR, exist_ok=True)
DB_PATH = os.path.join(KB_DIR, "trade_regulations.db")


def init_kb():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS tariff_codes (
            commodity_code TEXT PRIMARY KEY,
            description TEXT,
            category TEXT,
            uk_duty_rate REAL,
            vat_rate REAL,
            origin_rules TEXT,
            restrictions TEXT,
            last_updated DATE
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS sanctions_entities (
            id INTEGER PRIMARY KEY,
            entity_name TEXT,
            entity_type TEXT,
            jurisdiction TEXT,
            listing_date DATE,
            reason TEXT,
            source TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS regulatory_requirements (
            id INTEGER PRIMARY KEY,
            category TEXT,
            requirement TEXT,
            regulatory_body TEXT,
            applies_to TEXT,
            notes TEXT
        )
    """)
    c.execute("CREATE INDEX IF NOT EXISTS idx_tariff_category ON tariff_codes(category)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_sanctions_name ON sanctions_entities(entity_name)")
    conn.commit()
    conn.close()


def ingest_tariff_csv(csv_path: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            commodity_code = row.get("commodity_code", "").strip()
            if not commodity_code:
                continue
            duty_str = row.get("duty_rate", "0").replace("%", "").strip()
            try:
                duty_rate = float(duty_str)
            except (ValueError, TypeError):
                duty_rate = 0.0
            c.execute("""
                INSERT OR IGNORE INTO tariff_codes
                (commodity_code, description, category, uk_duty_rate, vat_rate, origin_rules, restrictions, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                commodity_code,
                row.get("description", ""),
                row.get("measure_type", ""),
                duty_rate,
                20.0,
                row.get("origin_rules", ""),
                "",
                row.get("effective_start_date", ""),
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def ingest_sanctions_csv(csv_path: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count = 0
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entity_name = row.get("Name 6", "").strip()
            if not entity_name:
                continue
            c.execute("""
                INSERT OR IGNORE INTO sanctions_entities
                (entity_name, entity_type, jurisdiction, listing_date, reason, source)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                entity_name,
                row.get("Name type", ""),
                row.get("Address Country", ""),
                row.get("Date Designated", ""),
                row.get("UK Statement of Reasons", ""),
                "OFSI",
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def ingest_regulatory_pdf(pdf_path: str) -> int:
    from src.retrieval.chunker import HybridChunker
    from pypdf import PdfReader

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    count = 0

    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text = page.extract_text()
        if not text:
            continue
        sections = text.split("\n\n")
        for section in sections:
            section = section.strip()
            if len(section) < 50:
                continue
            c.execute("""
                INSERT OR IGNORE INTO regulatory_requirements
                (category, requirement, regulatory_body, applies_to, notes)
                VALUES (?, ?, ?, ?, ?)
            """, (
                "import",
                section[:500],
                "HMRC",
                "beverages",
                f"Source: {os.path.basename(pdf_path)}",
            ))
            count += 1
    conn.commit()
    conn.close()
    return count


def trade_regulations_lookup(
    query_type: Literal["tariff", "sanctions_check", "regulatory_requirements", "sanctions"],
    commodity_code: Optional[str] = None,
    entity_name: Optional[str] = None,
    category: Optional[str] = None,
) -> ToolResult:
    if query_type == "sanctions":
        query_type = "sanctions_check"

    query_type_map = {
        "duty": "tariff",
        "dutyrate": "tariff",
        "duty_rate": "tariff",
        "tariff_rate": "tariff",
        "tariffcode": "tariff",
        "tariff_code": "tariff",
        "tariff_codes": "tariff",
        "import_duty": "tariff",
        "importduty": "tariff",
        "sanctions": "sanctions_check",
        "sanction": "sanctions_check",
        "ofsi": "sanctions_check",
        "regulatory": "regulatory_requirements",
        "regulations": "regulatory_requirements",
    }
    normalized = query_type_map.get(query_type.lower(), query_type.lower())
    if normalized in ("tariff", "sanctions_check", "regulatory_requirements"):
        query_type = normalized

    start = time.time()
    init_kb()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        if query_type == "tariff":
            if not commodity_code and not category:
                return ToolResult(
                    call_id="local",
                    tool_name="trade_regulations_lookup",
                    status="error",
                    content={"error": "Must provide commodity_code or category"},
                    latency_ms=int((time.time() - start) * 1000),
                    error_message="Missing required parameter",
                )

            if commodity_code:
                c.execute("SELECT * FROM tariff_codes WHERE commodity_code = ?", (commodity_code,))
                rows = c.fetchall()
                if not rows and len(commodity_code) >= 4:
                    c.execute("SELECT * FROM tariff_codes WHERE commodity_code LIKE ?", (commodity_code + "%",))
            else:
                c.execute("SELECT * FROM tariff_codes WHERE category = ? LIMIT 20", (category,))

            rows = c.fetchall()
            if not rows:
                return ToolResult(
                    call_id="local",
                    tool_name="trade_regulations_lookup",
                    status="success",
                    content={"results": [], "message": f"No tariff data found for {commodity_code or category}"},
                    latency_ms=int((time.time() - start) * 1000),
                )

            results = []
            for row in rows:
                results.append({
                    "commodity_code": row[0],
                    "description": row[1],
                    "category": row[2],
                    "uk_duty_rate": row[3],
                    "vat_rate": row[4],
                    "origin_rules": row[5],
                    "restrictions": row[6],
                    "last_updated": row[7],
                })
            return ToolResult(
                call_id="local",
                tool_name="trade_regulations_lookup",
                status="success",
                content={"results": results, "query_type": "tariff"},
                latency_ms=int((time.time() - start) * 1000),
            )

        elif query_type == "sanctions_check":
            if not entity_name:
                return ToolResult(
                    call_id="local",
                    tool_name="trade_regulations_lookup",
                    status="error",
                    content={"error": "Must provide entity_name for sanctions check"},
                    latency_ms=int((time.time() - start) * 1000),
                    error_message="Missing required parameter",
                )

            pattern = f"%{entity_name}%"
            c.execute("SELECT * FROM sanctions_entities WHERE entity_name LIKE ?", (pattern,))
            rows = c.fetchall()

            if not rows:
                return ToolResult(
                    call_id="local",
                    tool_name="trade_regulations_lookup",
                    status="success",
                    content={
                        "entity_name": entity_name,
                        "sanctioned": False,
                        "message": f"Entity '{entity_name}' not found in sanctions list — clean status",
                        "results": [],
                    },
                    latency_ms=int((time.time() - start) * 1000),
                )

            results = []
            for row in rows:
                results.append({
                    "entity_name": row[1],
                    "entity_type": row[2],
                    "jurisdiction": row[3],
                    "listing_date": row[4],
                    "reason": row[5],
                    "source": row[6],
                })
            return ToolResult(
                call_id="local",
                tool_name="trade_regulations_lookup",
                status="success",
                content={
                    "entity_name": entity_name,
                    "sanctioned": True,
                    "results": results,
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        elif query_type == "regulatory_requirements":
            c.execute("SELECT * FROM regulatory_requirements WHERE applies_to = ? LIMIT 20", (category or "beverages",))
            rows = c.fetchall()
            results = []
            for row in rows:
                results.append({
                    "category": row[1],
                    "requirement": row[2],
                    "regulatory_body": row[3],
                    "applies_to": row[4],
                    "notes": row[5],
                })
            return ToolResult(
                call_id="local",
                tool_name="trade_regulations_lookup",
                status="success",
                content={"results": results, "query_type": "regulatory_requirements"},
                latency_ms=int((time.time() - start) * 1000),
            )

    finally:
        conn.close()

    return ToolResult(
        call_id="local",
        tool_name="trade_regulations_lookup",
        status="error",
        content={"error": f"Unknown query_type: {query_type}. Valid types: tariff, sanctions_check, regulatory_requirements"},
        latency_ms=int((time.time() - start) * 1000),
        error_message=f"Unknown query_type: {query_type}",
    )
