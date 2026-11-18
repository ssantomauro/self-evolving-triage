"""SQLite storage: downloaded issues/labels, and a cache of Sage responses (so reruns are free)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "issues.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS issues (
    repo TEXT, number INTEGER, title TEXT, body TEXT, created_at TEXT, closed_at TEXT,
    state TEXT, state_reason TEXT, author_association TEXT, labels TEXT,
    PRIMARY KEY (repo, number)
);
CREATE TABLE IF NOT EXISTS labels (
    repo TEXT, name TEXT, description TEXT, created_at TEXT, PRIMARY KEY (repo, name)
);
CREATE TABLE IF NOT EXISTS sage_cache (
    key TEXT PRIMARY KEY, response TEXT, input_tokens INTEGER, output_tokens INTEGER
);
CREATE TABLE IF NOT EXISTS llm_cache (key TEXT PRIMARY KEY, response TEXT);
"""


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn
