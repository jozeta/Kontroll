"""SQLite connection handling and schema creation for the scans/settings tables."""

import sqlite3
from pathlib import Path

# All runtime-generated state (database, captured images, logs) lives under data/,
# which is excluded from git and, unlike the app database itself, receives no
# application-level encryption -- disk-at-rest protection for this file is expected
# to come from OS-level full-disk encryption (BitLocker) on the deployment machine.
# See README.md "Security posture" for the full accepted-risk statement.
DB_PATH: Path = Path(__file__).resolve().parent.parent / "data" / "app.db"


def get_connection() -> sqlite3.Connection:
    """Open (and implicitly create) the SQLite database file, with row access by column name enabled."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create the scans and settings tables if they do not already exist."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            operator TEXT,
            face_image_path TEXT,
            doc_image_path TEXT,
            face_embedding BLOB,
            match_score REAL,
            match_result TEXT,
            doc_type TEXT,
            doc_country TEXT,
            doc_fields_json TEXT,
            validation_status TEXT,
            validation_detail_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS settings (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )
    conn.commit()
