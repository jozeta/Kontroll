"""Creates a single zip archive containing a consistent snapshot of the database plus every saved
face/document image, for a manual operator-triggered backup (e.g. to a USB drive or network
share). This is a supplement to BitLocker (see README "Security posture"), not a replacement --
BitLocker protects data at rest on this machine, but does nothing against drive failure or the
machine itself being lost; a backup archive is the only protection against that.
"""

import sqlite3
import tempfile
import zipfile
from pathlib import Path

from storage.db import DB_PATH


def create_backup_archive(conn: sqlite3.Connection, faces_dir: Path, documents_dir: Path, destination_zip_path: Path) -> None:
    """Write a zip archive to destination_zip_path containing a consistent DB snapshot plus every file under faces_dir/documents_dir.

    The database is captured via sqlite3's own backup API (Connection.backup) into a temp file
    first, rather than copying app.db's bytes directly off disk -- a raw file copy risks landing
    mid-write and capturing a half-written page, while the backup API guarantees a consistent
    snapshot regardless of what else the live connection is doing.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        db_snapshot_path = Path(temp_dir) / DB_PATH.name
        snapshot_conn = sqlite3.connect(db_snapshot_path)
        try:
            conn.backup(snapshot_conn)
        finally:
            snapshot_conn.close()

        with zipfile.ZipFile(destination_zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(db_snapshot_path, arcname=f"database/{DB_PATH.name}")
            for label, directory in (("faces", faces_dir), ("documents", documents_dir)):
                if not directory.exists():
                    continue
                for file_path in directory.rglob("*"):
                    if file_path.is_file():
                        archive.write(file_path, arcname=f"{label}/{file_path.name}")
