"""Typed insert/query functions over the scans and settings tables."""

import sqlite3
from dataclasses import dataclass
from datetime import datetime


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
