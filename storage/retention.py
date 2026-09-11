"""Startup job that purges scan records and associated image files older than the configured retention period."""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

from config.settings import RETENTION_DAYS_NEVER, get_retention_days
from storage.models import list_scans_older_than, thumbnail_cache_path


def purge_expired_scans(conn: sqlite3.Connection) -> int:
    """Delete every scan record (and its associated image files) older than the configured retention period.

    Intended to run once at app startup, not on a running background schedule -- matches the v1
    requirement for a simple startup purge job rather than a scheduler/thread. If the operator has
    set retention to "Never" (RETENTION_DAYS_NEVER), the job is skipped entirely rather than
    calling purge_scans_older_than_days(conn, 0), which would instead delete every scan.
    """
    retention_days = get_retention_days(conn)
    if retention_days == RETENTION_DAYS_NEVER:
        return 0
    return purge_scans_older_than_days(conn, retention_days)


def purge_scans_older_than_days(conn: sqlite3.Connection, days: int) -> int:
    """Delete every scan record (and its associated image files) older than the given number of days.

    Shared by the startup retention job (using the configured retention period) and the operator-
    triggered manual purge in the settings window (using whatever cutoff the operator picked).
    Image files are removed before the database row, so a crash mid-purge can at worst leave an
    orphaned row pointing at an already-deleted file (harmless, cleaned up on the next run) rather
    than an orphaned file with no row to ever reference it again.
    """
    cutoff = datetime.now() - timedelta(days=days)
    expired_scans = list_scans_older_than(conn, cutoff)

    for scan in expired_scans:
        _delete_image_file_if_present(scan.face_image_path)
        _delete_image_file_if_present(scan.doc_image_path)

    # One statement + one commit for every expired row, rather than storage.models.delete_scan's
    # own commit-per-call N times over -- N separate fsync'd transactions at startup whenever a
    # backlog of scans has expired (e.g. after time away, or a newly shortened retention period)
    # is a real, easily-avoided startup delay, since this purge is already one atomic unit of work.
    conn.execute("DELETE FROM scans WHERE created_at < ?", (cutoff.isoformat(),))
    conn.commit()

    return len(expired_scans)


def _delete_image_file_if_present(image_path: str | None) -> None:
    """Remove a captured image file from disk, and its cached thumbnail (see storage.models.thumbnail_cache_path), if a path was recorded and either still exists."""
    if image_path is None:
        return
    for path in (Path(image_path), thumbnail_cache_path(image_path)):
        if path.exists():
            path.unlink()
