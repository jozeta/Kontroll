"""Load/save app settings: retention period, face-match threshold, and camera device selections."""

import sqlite3

DEFAULT_RETENTION_DAYS: int = 90

# Stored in the same retention_days setting as an ordinary value (rather than a separate
# enabled/disabled flag) so there's only ever one source of truth for the startup purge job to
# read -- 0 days of retention is otherwise meaningless (nothing would ever be old enough to
# survive a purge), so it's free to reuse as the "auto-purge disabled" sentinel.
RETENTION_DAYS_NEVER: int = 0

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

DEFAULT_CAPTURE_SHORTCUT: str = "F12"

# Free-space floor below which the status bar shows a low-disk-space warning. 2GB is a rough
# starting point (a handful of scans at most before actually running out), not a calibrated
# value -- same unvalidated-default caveat as DEFAULT_FACE_MATCH_THRESHOLD above.
DEFAULT_DISK_SPACE_WARNING_MB: int = 2000

DEFAULT_SETTINGS: dict[str, str] = {
    "retention_days": str(DEFAULT_RETENTION_DAYS),
    "face_match_threshold": str(DEFAULT_FACE_MATCH_THRESHOLD),
    "face_camera_index": str(DEFAULT_FACE_CAMERA_INDEX),
    "doc_camera_index": str(DEFAULT_DOC_CAMERA_INDEX),
    "capture_shortcut": DEFAULT_CAPTURE_SHORTCUT,
    "disk_space_warning_mb": str(DEFAULT_DISK_SPACE_WARNING_MB),
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


def get_setting_or_none(conn: sqlite3.Connection, key: str) -> str | None:
    """Return the raw string value for a settings key that may not exist yet, or None if it's never been set.

    Unlike get_setting, this doesn't assume the key was pre-populated by ensure_default_settings --
    for settings like saved window geometry, there is no sensible hardcoded default to seed, only
    "nothing saved yet".
    """
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else None


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


def get_capture_shortcut(conn: sqlite3.Connection) -> str:
    """Typed convenience accessor for the capture_shortcut setting (a Qt key-sequence string, e.g. 'F12')."""
    return get_setting(conn, "capture_shortcut")


def get_disk_space_warning_mb(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the disk_space_warning_mb setting."""
    return int(get_setting(conn, "disk_space_warning_mb"))
