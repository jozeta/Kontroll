"""Load/save app settings: retention period, face-match threshold, and camera device selections."""

import sqlite3

DEFAULT_RETENTION_DAYS: int = 90

# Starting point for buffalo_l (ArcFace-family) cosine similarity on L2-normalized embeddings.
# Published guidance for this model typically lands in the 0.30-0.45 range for same-domain
# (photo-vs-photo) verification, but this app compares a live webcam capture against a
# printed/laminated ID photo -- a cross-domain comparison that tends to push genuine-pair
# scores lower. Treat this default as unvalidated until recalibrated against real captured
# pairs during pilot use; this is why the UI always shows the raw score and an operator
# override alongside the automated pass/fail badge, rather than only a hard gate.
DEFAULT_FACE_MATCH_THRESHOLD: float = 0.38

DEFAULT_FACE_CAMERA_INDEX: int = 0
DEFAULT_DOC_CAMERA_INDEX: int = 1

DEFAULT_SETTINGS: dict[str, str] = {
    "retention_days": str(DEFAULT_RETENTION_DAYS),
    "face_match_threshold": str(DEFAULT_FACE_MATCH_THRESHOLD),
    "face_camera_index": str(DEFAULT_FACE_CAMERA_INDEX),
    "doc_camera_index": str(DEFAULT_DOC_CAMERA_INDEX),
}


def ensure_default_settings(conn: sqlite3.Connection) -> None:
    """Insert any missing settings keys with their default values, leaving existing values untouched."""
    for key, value in DEFAULT_SETTINGS.items():
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
    conn.commit()


def get_setting(conn: sqlite3.Connection, key: str) -> str:
    """Return the raw string value for a settings key."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"]


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update a settings key's value."""
    conn.execute(
        """
        INSERT INTO settings (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


def get_retention_days(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the retention_days setting."""
    return int(get_setting(conn, "retention_days"))


def get_face_match_threshold(conn: sqlite3.Connection) -> float:
    """Typed convenience accessor for the face_match_threshold setting."""
    return float(get_setting(conn, "face_match_threshold"))


def get_face_camera_index(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the face_camera_index setting."""
    return int(get_setting(conn, "face_camera_index"))


def get_doc_camera_index(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the doc_camera_index setting."""
    return int(get_setting(conn, "doc_camera_index"))
