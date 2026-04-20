import os
import time
import json
import sqlite3
from typing import Literal
from src.tools.base import ToolResult


KB_DIR = "data/processed"
os.makedirs(KB_DIR, exist_ok=True)
KB_PATH = os.path.join(KB_DIR, "partnership_kb.db")


KNOWN_ENTITIES = [
    {"name": "Tsingtao Brewery", "type": "beverage_company", "sector": "beverages", "description": "Chinese brewery, LEC's exclusive UK distributor", "uk_presence": True},
    {"name": "Meituan Dianping", "type": "tech_company", "sector": "food_delivery", "description": "Chinese tech platform for food delivery and local services", "uk_presence": False},
    {"name": "Longi Green Energy", "type": "energy_company", "sector": "renewable_energy", "description": "Leading Chinese solar panel manufacturer", "uk_presence": False},
    {"name": "Huawei", "type": "tech_company", "sector": "telecommunications", "description": "Chinese telecommunications equipment company", "uk_presence": True},
    {"name": "BYD", "type": "automotive_company", "sector": "electric_vehicles", "description": "Chinese electric vehicle manufacturer", "uk_presence": False},
    {"name": "Alibaba", "type": "tech_company", "sector": "e_commerce", "description": "Chinese e-commerce and cloud computing company", "uk_presence": False},
    {"name": "ByteDance", "type": "tech_company", "sector": "media", "description": "Chinese social media and AI company (TikTok parent)", "uk_presence": False},
    {"name": "Tencent", "type": "tech_company", "sector": "technology", "description": "Chinese internet services conglomerate", "uk_presence": False},
    {"name": "SMIC", "type": "tech_company", "sector": "semiconductors", "description": "China's largest semiconductor foundry", "uk_presence": False},
    {"name": "ZTE", "type": "tech_company", "sector": "telecommunications", "description": "Chinese telecommunications equipment company", "uk_presence": False},
]


def init_partnership_kb():
    conn = sqlite3.connect(KB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY,
            name TEXT UNIQUE,
            entity_type TEXT,
            sector TEXT,
            description TEXT,
            uk_presence INTEGER,
            strategic_fit_score REAL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY,
            entity_name TEXT,
            assessment_type TEXT,
            assessment_content TEXT
        )
    """)
    for entity in KNOWN_ENTITIES:
        c.execute("""
            INSERT OR IGNORE INTO entities
            (name, entity_type, sector, description, uk_presence, strategic_fit_score)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            entity["name"],
            entity["type"],
            entity["sector"],
            entity["description"],
            1 if entity["uk_presence"] else 0,
            0.8 if entity["sector"] in ["beverages", "renewable_energy", "electric_vehicles"] else 0.5,
        ))
    conn.commit()
    conn.close()


def partnership_profiler(
    entity_name: str,
    entity_type: Literal["chinese_company", "uk_company"] = "chinese_company",
    analysis_type: Literal["profile", "strategic_fit", "risk_assessment"] = "profile",
) -> ToolResult:
    start = time.time()
    init_partnership_kb()

    conn = None
    try:
        conn = sqlite3.connect(KB_PATH)
        c = conn.cursor()

        c.execute("SELECT * FROM entities WHERE name LIKE ?", (f"%{entity_name}%",))
        row = c.fetchone()

        if not row:
            return ToolResult(
                call_id="local",
                tool_name="partnership_profiler",
                status="success",
                content={
                    "entity_name": entity_name,
                    "found": False,
                    "message": f"Entity '{entity_name}' not in our known database. Consider: verify via Companies House and OFSI sanctions check.",
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        entity = {
            "id": row[0],
            "name": row[1],
            "type": row[2],
            "sector": row[3],
            "description": row[4],
            "uk_presence": bool(row[5]),
            "strategic_fit_score": row[6],
        }

        if analysis_type == "profile":
            return ToolResult(
                call_id="local",
                tool_name="partnership_profiler",
                status="success",
                content={
                    "entity": entity,
                    "analysis_type": "profile",
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        elif analysis_type == "strategic_fit":
            lec_sectors = ["beverages", "renewable_energy", "robotics", "healthcare", "electric_vehicles"]
            fit = entity["sector"] in lec_sectors
            score = entity["strategic_fit_score"]

            return ToolResult(
                call_id="local",
                tool_name="partnership_profiler",
                status="success",
                content={
                    "entity": entity,
                    "analysis_type": "strategic_fit",
                    "fits_lec_sectors": fit,
                    "fit_score": score,
                    "rationale": f"{entity['name']} operates in {entity['sector']} — {'strategically relevant to LEC' if fit else 'not a primary LEC sector'}",
                },
                latency_ms=int((time.time() - start) * 1000),
            )

        elif analysis_type == "risk_assessment":
            from src.tools.trade_regulations import trade_regulations_lookup

            sanctions_result = trade_regulations_lookup(
                query_type="sanctions_check",
                entity_name=entity_name,
            )

            sanctioned = sanctions_result.content.get("sanctioned", False)

            return ToolResult(
                call_id="local",
                tool_name="partnership_profiler",
                status="success",
                content={
                    "entity": entity,
                    "analysis_type": "risk_assessment",
                    "sanctioned": sanctioned,
                    "sanctions_details": sanctions_result.content.get("results", []),
                    "risk_factors": [
                        "OFSI sanctions check required before any engagement"
                    ] if sanctioned else [
                        "No known sanctions flags",
                        "UK presence verified" if entity["uk_presence"] else "No known UK presence",
                        "Standard due diligence apply",
                    ],
                },
                latency_ms=int((time.time() - start) * 1000),
            )

    finally:
        if conn:
            conn.close()
