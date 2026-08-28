"""SQLite connection handling and schema creation for the scans/settings tables."""

import sqlite3

from config.app_paths import get_db_dir

# The database directory is resolved via config.app_paths rather than a path relative to this
# file's own location -- once packaged as a standalone EXE, a __file__-relative path resolves
# inside an ephemeral or not-reliably-writable location. See config/app_paths.py for the full
# rationale and how the operator-configurable location works. This receives no application-level
# encryption -- disk-at-rest protection for this file is expected to come from OS-level full-disk
# encryption (BitLocker) on the deployment machine. See README.md "Security posture" for the full
# accepted-risk statement.
DB_PATH = get_db_dir() / "app.db"


def get_connection() -> sqlite3.Connection:
    """Open (and implicitly create) the SQLite database file, with row access by column name enabled."""
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_schema(conn: sqlite3.Connection) -> None:
    """Create the scans, settings, and flagged_persons tables if they do not already exist, and apply any column migrations needed on top of an older existing scans table."""
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
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS flagged_persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            name TEXT,
            personal_number TEXT,
            image_path TEXT,
            notes TEXT
        )
        """
    )
    _add_column_if_missing(conn, "scans", "operator_comment", "TEXT")
    # list_scans/list_scans_older_than both order/filter on created_at -- without this, every
    # history load and every retention purge does a full-table scan-and-sort instead of an
    # indexed lookup, which only gets worse as scans accumulate (there's no retention limit by
    # default, so this table is meant to grow indefinitely).
    conn.execute("CREATE INDEX IF NOT EXISTS idx_scans_created_at ON scans (created_at)")
    conn.commit()


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, column_type: str) -> None:
    """Add a column to an already-existing table if it isn't there yet.

    CREATE TABLE IF NOT EXISTS above is a no-op against a table an existing install already
    created before this column existed, so a column added after v1's initial release needs this
    explicit ALTER TABLE path instead -- sqlite has no "ADD COLUMN IF NOT EXISTS" on the version
    this app's requirements.txt pins, so the existing-columns check is done by hand via
    PRAGMA table_info.
    """
    existing_columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing_columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
