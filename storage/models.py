"""Typed insert/query functions over the scans and settings tables."""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


def thumbnail_cache_path(image_path: str) -> Path:
    """The path of the small cached thumbnail file for a saved scan image, next to the full-
    resolution image itself -- same naming-convention approach as
    ui/image_viewer.py::ImageEditDialog's own "<name>.original<ext>" pre-edit backup, so a fresh
    schema column/migration isn't needed for either. Pure pathlib -- no cv2/Qt dependency -- so
    both ui/ (which generates and reads the cache: see ui/frame_display.py::write_thumbnail_cache
    and ui/main_window.py::_load_thumbnail) and storage/retention.py (which only ever needs the
    path, to delete it alongside the full image) can share one definition of the convention without
    storage/ having to depend on ui/.
    """
    path = Path(image_path)
    return path.with_name(f"{path.stem}.thumb.jpg")


@dataclass
class ScanRecord:
    """One row of the scans table, as read back from the database."""

    id: int
    created_at: str
    operator: str | None
    face_image_path: str | None
    doc_image_path: str | None
    face_embedding: bytes | None
    match_score: float | None
    match_result: str | None
    doc_type: str | None
    doc_country: str | None
    doc_fields_json: str | None
    validation_status: str | None
    validation_detail_json: str | None
    operator_comment: str | None


def _row_to_scan_record(row: sqlite3.Row) -> ScanRecord:
    """Convert a raw sqlite3.Row from the scans table into a typed ScanRecord."""
    return ScanRecord(
        id=row["id"],
        created_at=row["created_at"],
        operator=row["operator"],
        face_image_path=row["face_image_path"],
        doc_image_path=row["doc_image_path"],
        face_embedding=row["face_embedding"],
        match_score=row["match_score"],
        match_result=row["match_result"],
        doc_type=row["doc_type"],
        doc_country=row["doc_country"],
        doc_fields_json=row["doc_fields_json"],
        validation_status=row["validation_status"],
        validation_detail_json=row["validation_detail_json"],
        # Added after v1's initial release (see storage.db._add_column_if_missing) -- a row
        # written before this column existed still returns None for it via sqlite3.Row's normal
        # missing-value behavior once the ALTER TABLE has run, so no extra .get()-style guard is
        # needed here the way validation_detail_json's *contents* need one.
        operator_comment=row["operator_comment"],
    )


def insert_scan(
    conn: sqlite3.Connection,
    operator: str | None,
    face_image_path: str | None,
    doc_image_path: str | None,
    face_embedding: bytes | None,
    match_score: float | None,
    match_result: str | None,
    doc_type: str | None,
    doc_country: str | None,
    doc_fields_json: str | None,
    validation_status: str | None,
    validation_detail_json: str | None,
) -> int:
    """Insert one completed scan session and return its new row id."""
    cursor = conn.execute(
        """
        INSERT INTO scans (
            created_at, operator, face_image_path, doc_image_path, face_embedding,
            match_score, match_result, doc_type, doc_country, doc_fields_json,
            validation_status, validation_detail_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            datetime.now().isoformat(),
            operator,
            face_image_path,
            doc_image_path,
            face_embedding,
            match_score,
            match_result,
            doc_type,
            doc_country,
            doc_fields_json,
            validation_status,
            validation_detail_json,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def get_scan_by_id(conn: sqlite3.Connection, scan_id: int) -> ScanRecord | None:
    """Fetch a single scan record by primary key."""
    row = conn.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        return None
    return _row_to_scan_record(row)


def list_scans(conn: sqlite3.Connection, limit: int = 100, offset: int = 0) -> list[ScanRecord]:
    """Fetch scan records ordered newest-first, for a history/review view."""
    rows = conn.execute(
        "SELECT * FROM scans ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    return [_row_to_scan_record(row) for row in rows]


def list_scans_older_than(conn: sqlite3.Connection, cutoff: datetime) -> list[ScanRecord]:
    """Fetch scan records created before the given cutoff, for use by the retention purge job."""
    rows = conn.execute(
        "SELECT * FROM scans WHERE created_at < ?",
        (cutoff.isoformat(),),
    ).fetchall()
    return [_row_to_scan_record(row) for row in rows]


def delete_scan(conn: sqlite3.Connection, scan_id: int) -> None:
    """Delete a single scan row by id. Does not touch image files on disk -- callers remove those separately."""
    conn.execute("DELETE FROM scans WHERE id = ?", (scan_id,))
    conn.commit()


def _parse_comments(operator_comment_raw: str | None) -> list[dict]:
    """Parse a scan's operator_comment column into a list of {"id", "created_at", "text"} comment dicts.

    Tolerates the legacy plain-string format the column held before comments became a list (a
    single free-text note with no id/timestamp of its own) by wrapping it as one comment, so an
    older row's existing note isn't lost the first time this reads it back.
    """
    if not operator_comment_raw:
        return []
    try:
        parsed = json.loads(operator_comment_raw)
    except (json.JSONDecodeError, TypeError):
        return [{"id": 1, "created_at": None, "text": operator_comment_raw}]
    if isinstance(parsed, list):
        return parsed
    return [{"id": 1, "created_at": None, "text": operator_comment_raw}]


def list_scan_comments(conn: sqlite3.Connection, scan_id: int) -> list[dict]:
    """Return every comment attached to a scan, oldest first."""
    row = conn.execute("SELECT operator_comment FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        return []
    return _parse_comments(row["operator_comment"])


def add_scan_comment(conn: sqlite3.Connection, scan_id: int, text: str) -> dict:
    """Append one comment to a scan's comment list and return it.

    Each comment gets an id unique within that scan's own list (not globally) -- simple
    incrementing integers are enough since comments are only ever addressed relative to one scan
    at a time (see delete_scan_comment), and this stays a single-column JSON list rather than its
    own table, consistent with how validation_detail_json stores other per-scan structured data.
    """
    comments = list_scan_comments(conn, scan_id)
    next_id = max((comment.get("id", 0) for comment in comments), default=0) + 1
    new_comment = {"id": next_id, "created_at": datetime.now().isoformat(), "text": text}
    comments.append(new_comment)
    conn.execute("UPDATE scans SET operator_comment = ? WHERE id = ?", (json.dumps(comments), scan_id))
    conn.commit()
    return new_comment


def delete_scan_comment(conn: sqlite3.Connection, scan_id: int, comment_id: int) -> None:
    """Remove one comment (by its per-scan id, from add_scan_comment's return value) from a scan's comment list."""
    remaining = [comment for comment in list_scan_comments(conn, scan_id) if comment.get("id") != comment_id]
    conn.execute("UPDATE scans SET operator_comment = ? WHERE id = ?", (json.dumps(remaining) if remaining else None, scan_id))
    conn.commit()


def update_match_result(conn: sqlite3.Connection, scan_id: int, match_result: str) -> None:
    """Update a previously saved scan's match_result, e.g. when an operator applies a manual override from the history list."""
    conn.execute("UPDATE scans SET match_result = ? WHERE id = ?", (match_result, scan_id))
    conn.commit()


def update_scan_doc_fields(conn: sqlite3.Connection, scan_id: int, doc_fields_json: str) -> None:
    """Overwrite a scan's extracted doc_fields_json, e.g. after an operator manually edits its name/personal-number fields from the history list's detail section (see ui/main_window.py's edit-fields dialog)."""
    conn.execute("UPDATE scans SET doc_fields_json = ? WHERE id = ?", (doc_fields_json, scan_id))
    conn.commit()


def update_scan_analysis(
    conn: sqlite3.Connection,
    scan_id: int,
    face_embedding: bytes | None,
    match_score: float | None,
    match_result: str,
    doc_type: str | None,
    doc_country: str | None,
    doc_fields_json: str,
    validation_status: str | None,
    validation_detail_json: str,
) -> None:
    """Overwrite every automatically-derived field on an already-saved scan, after HistoryRowWidget's "Re-analyze" action re-runs the face-match + document pipeline against its saved images (see ui/main_window.py::_analyze_capture).

    Leaves created_at/operator/image paths/operator_comment untouched -- re-analysis only refreshes
    analysis *outputs*, not the scan's identity or an operator's own notes.
    """
    conn.execute(
        """
        UPDATE scans SET
            face_embedding = ?, match_score = ?, match_result = ?, doc_type = ?, doc_country = ?,
            doc_fields_json = ?, validation_status = ?, validation_detail_json = ?
        WHERE id = ?
        """,
        (
            face_embedding, match_score, match_result, doc_type, doc_country,
            doc_fields_json, validation_status, validation_detail_json, scan_id,
        ),
    )
    conn.commit()


def rewrite_image_directory_paths(conn: sqlite3.Connection, old_dir: Path, new_dir: Path) -> int:
    """Repoint every stored face_image_path/doc_image_path that lives under old_dir to the equivalent path under new_dir, preserving each file's name.

    Called once after config.app_paths completes a faces/documents directory relocation (which
    moves the files themselves but has no DB access of its own to fix up rows that were written
    before the move) -- see config/app_paths.py::pop_completed_relocation. Done row-by-row in
    Python rather than a single SQL UPDATE ... LIKE, since a path can contain '%'/'_' characters
    that LIKE would otherwise interpret as wildcards.
    """
    old_dir_str = str(old_dir)
    rows = conn.execute("SELECT id, face_image_path, doc_image_path FROM scans").fetchall()
    updated_count = 0
    for row in rows:
        new_face_path = _rebase_path(row["face_image_path"], old_dir_str, new_dir)
        new_doc_path = _rebase_path(row["doc_image_path"], old_dir_str, new_dir)
        if new_face_path is None and new_doc_path is None:
            continue
        conn.execute(
            "UPDATE scans SET face_image_path = ?, doc_image_path = ? WHERE id = ?",
            (
                new_face_path if new_face_path is not None else row["face_image_path"],
                new_doc_path if new_doc_path is not None else row["doc_image_path"],
                row["id"],
            ),
        )
        updated_count += 1
    conn.commit()
    return updated_count


def _rebase_path(stored_path: str | None, old_dir_str: str, new_dir: Path) -> str | None:
    """Return stored_path re-rooted under new_dir if it currently lives under old_dir_str, else None (no change needed)."""
    if stored_path is None or not stored_path.startswith(old_dir_str):
        return None
    return str(new_dir / Path(stored_path).name)


def count_scans(conn: sqlite3.Connection) -> int:
    """Return the total number of saved scan records, for display in the status bar."""
    return conn.execute("SELECT COUNT(*) AS count FROM scans").fetchone()["count"]


def calculate_scan_storage_bytes(*directories: Path) -> int:
    """Sum the on-disk size of every file under the given directories (the saved face/document image folders).

    Walks the filesystem directly rather than summing individual scans' recorded image paths, so
    the figure reflects actual disk usage even if it includes an orphaned file left behind by a
    prior crash mid-capture or mid-delete.
    """
    total_bytes = 0
    for directory in directories:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if path.is_file():
                total_bytes += path.stat().st_size
    return total_bytes


@dataclass
class FlaggedPerson:
    """One entry on the operator-maintained watch list: a name and/or personal number (personnummer) to warn on, plus an optional reference image."""

    id: int
    created_at: str
    name: str | None
    personal_number: str | None
    image_path: str | None
    notes: str | None


def _row_to_flagged_person(row: sqlite3.Row) -> FlaggedPerson:
    """Convert a raw sqlite3.Row from the flagged_persons table into a typed FlaggedPerson."""
    return FlaggedPerson(
        id=row["id"],
        created_at=row["created_at"],
        name=row["name"],
        personal_number=row["personal_number"],
        image_path=row["image_path"],
        notes=row["notes"],
    )


def insert_flagged_person(
    conn: sqlite3.Connection,
    name: str | None,
    personal_number: str | None,
    image_path: str | None,
    notes: str | None,
) -> int:
    """Add one entry to the flagged-persons watch list and return its new row id."""
    cursor = conn.execute(
        "INSERT INTO flagged_persons (created_at, name, personal_number, image_path, notes) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), name or None, personal_number or None, image_path or None, notes or None),
    )
    conn.commit()
    return cursor.lastrowid


def update_flagged_person(
    conn: sqlite3.Connection,
    flagged_person_id: int,
    name: str | None,
    personal_number: str | None,
    image_path: str | None,
    notes: str | None,
) -> None:
    """Update every editable field of an existing flagged-persons watch-list entry in place -- created_at and id are untouched.

    Changing name/personal_number here changes what document.flag_matcher.find_matching_flagged_person
    considers a match against this entry, which can both stop matching scans currently flagged
    against it and start matching ones that weren't before -- see
    ui/flag_retroactive.py::apply_flagged_person_edit, the caller responsible for reconciling
    already-flagged scans against whatever this update just changed. This function itself only
    writes the row; it doesn't touch any scan's own flagged_match.
    """
    conn.execute(
        "UPDATE flagged_persons SET name = ?, personal_number = ?, image_path = ?, notes = ? WHERE id = ?",
        (name or None, personal_number or None, image_path or None, notes or None, flagged_person_id),
    )
    conn.commit()


def list_flagged_persons(conn: sqlite3.Connection) -> list[FlaggedPerson]:
    """Fetch every flagged-persons entry, newest first -- small, fixed-size operator-maintained list, so no pagination."""
    rows = conn.execute("SELECT * FROM flagged_persons ORDER BY created_at DESC").fetchall()
    return [_row_to_flagged_person(row) for row in rows]


def get_flagged_person_by_id(conn: sqlite3.Connection, flagged_person_id: int) -> FlaggedPerson | None:
    """Fetch a single flagged-persons entry by id, or None if it no longer exists (e.g. removed from Settings between a scan being flagged and an operator later acting on that flag)."""
    row = conn.execute("SELECT * FROM flagged_persons WHERE id = ?", (flagged_person_id,)).fetchone()
    return _row_to_flagged_person(row) if row is not None else None


def delete_flagged_person(conn: sqlite3.Connection, flagged_person_id: int) -> None:
    """Remove one entry from the flagged-persons watch list by id. Does not touch its reference image file on disk."""
    conn.execute("DELETE FROM flagged_persons WHERE id = ?", (flagged_person_id,))
    conn.commit()


def set_scan_flagged_match(conn: sqlite3.Connection, scan_id: int, flagged_match: dict | None) -> None:
    """Set (or, if None, clear) the "flagged_match" key inside one scan's validation_detail_json blob, leaving every other key untouched.

    Used to retroactively flag a scan against a flagged-persons entry added after that scan was
    captured (see ui/flag_retroactive.py), and to clear a flag from a single scan when an operator
    removes it from the scan's own detail view (see ui/main_window.py::HistoryRowWidget).
    """
    row = conn.execute("SELECT validation_detail_json FROM scans WHERE id = ?", (scan_id,)).fetchone()
    if row is None:
        return
    detail = json.loads(row["validation_detail_json"]) if row["validation_detail_json"] else {}
    if flagged_match is None:
        detail.pop("flagged_match", None)
    else:
        detail["flagged_match"] = flagged_match
    conn.execute("UPDATE scans SET validation_detail_json = ? WHERE id = ?", (json.dumps(detail), scan_id))
    conn.commit()


def clear_flagged_match_for_person(conn: sqlite3.Connection, flagged_person_id: int) -> int:
    """Clear the "flagged_match" key from every scan currently pointing at the given flagged-persons id, and return how many were changed.

    Called when that flagged-persons entry is deleted (from Settings, or via a scan's own
    "Remove Flag" detail button -- both cascade the same way) so removing someone from the watch
    list also removes their red-flag treatment everywhere it was previously applied, rather than
    leaving a dangling reference to a now-deleted entry.
    """
    rows = conn.execute(
        "SELECT id, validation_detail_json FROM scans WHERE validation_detail_json IS NOT NULL"
    ).fetchall()
    cleared_count = 0
    for row in rows:
        detail = json.loads(row["validation_detail_json"])
        flagged_match = detail.get("flagged_match")
        if not flagged_match or flagged_match.get("flagged_person_id") != flagged_person_id:
            continue
        detail.pop("flagged_match")
        conn.execute("UPDATE scans SET validation_detail_json = ? WHERE id = ?", (json.dumps(detail), row["id"]))
        cleared_count += 1
    if cleared_count:
        conn.commit()
    return cleared_count
