"""Startup job that purges scan records and associated image files older than the configured retention period."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import get_retention_days
from storage.models import delete_scan, list_scans_older_than


def purge_expired_scans(conn: sqlite3.Connection) -> int:
    """Delete every scan record (and its associated image files) older than the configured retention period.

    Intended to run once at app startup, not on a running background schedule -- matches the v1
    requirement for a simple startup purge job rather than a scheduler/thread. Image files are
    removed before the database row, so a crash mid-purge can at worst leave an orphaned row
    pointing at an already-deleted file (harmless, cleaned up on the next startup run) rather than
    an orphaned file with no row to ever reference it again.
    """
    retention_days = get_retention_days(conn)
    cutoff = datetime.now() - timedelta(days=retention_days)
    expired_scans = list_scans_older_than(conn, cutoff)

    for scan in expired_scans:
        _delete_image_file_if_present(scan.face_image_path)
        _delete_image_file_if_present(scan.doc_image_path)
        delete_scan(conn, scan.id)

    return len(expired_scans)


def _delete_image_file_if_present(image_path: str | None) -> None:
    """Remove a captured image file from disk if a path was recorded and the file still exists."""
    if image_path is None:
        return
    path = Path(image_path)
    if path.exists():
        path.unlink()
