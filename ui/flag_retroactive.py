"""Shared "check earlier scans against a newly added flagged-persons entry" flow.

Run once right after a flagged-persons watch-list entry is inserted -- from the manual add form
in Settings (ui/settings_window.py::SettingsWindow._on_add_flagged_person_clicked) and from a
scan's own "Add to Flagged Users..." detail button (ui/main_window.py::HistoryRowWidget.
_on_flag_person_clicked). A brand-new entry has no effect on scans saved before it existed --
document.flag_matcher.find_matching_flagged_person is only ever consulted at capture time -- so
this walks every saved scan once, offers the operator a checklist of which now-matching scans
(including, commonly, the very scan the entry was just created from) to retroactively mark, and
applies flagged_match only to the ones the operator confirms.
"""

import json
import logging
import sqlite3

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from document.flag_matcher import find_matching_flagged_person, matched_on_display
from storage.models import FlaggedPerson, ScanRecord, list_scans, set_scan_flagged_match, update_flagged_person
from ui.window_geometry import restore_geometry, save_geometry

_RETROACTIVE_MATCH_DIALOG_GEOMETRY_KEY = "retroactive_match_dialog_geometry"

logger = logging.getLogger(__name__)

_THUMBNAIL_SIZE = (48, 48)


def _already_flagged_with(scan: ScanRecord, flagged_person_id: int) -> bool:
    if not scan.validation_detail_json:
        return False
    existing = json.loads(scan.validation_detail_json).get("flagged_match")
    return bool(existing and existing.get("flagged_person_id") == flagged_person_id)


def run_retroactive_flag_check(conn: sqlite3.Connection, flagged_person: FlaggedPerson, parent: QWidget) -> int:
    """Offer to retroactively mark any saved scan that matches `flagged_person`, and apply the operator's choices. Returns the number of scans marked."""
    candidates: list[tuple[ScanRecord, str]] = []
    for scan in list_scans(conn, limit=100_000):
        if _already_flagged_with(scan, flagged_person.id):
            continue
        doc_fields = json.loads(scan.doc_fields_json) if scan.doc_fields_json else {}
        match = find_matching_flagged_person(doc_fields, [flagged_person])
        if match is not None:
            candidates.append((scan, match.matched_on))

    if not candidates:
        return 0

    dialog = _RetroactiveMatchDialog(candidates, conn, parent=parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return 0

    selected = dialog.selected_candidates()
    for scan, matched_on in selected:
        set_scan_flagged_match(
            conn,
            scan.id,
            {
                "flagged_person_id": flagged_person.id,
                "flagged_person_name": flagged_person.name,
                "matched_on": matched_on,
                "notes": flagged_person.notes,
            },
        )
        logger.info("Retroactively flagged scan %d against flagged person id=%d", scan.id, flagged_person.id)

    if selected:
        QMessageBox.information(parent, "Flagged Users", f"Marked {len(selected)} scan(s) as flagged.")
    return len(selected)


def scans_flagged_for_person(conn: sqlite3.Connection, flagged_person_id: int) -> list[ScanRecord]:
    """Every currently-saved scan whose stored flagged_match currently points at flagged_person_id.

    A read-only counterpart to storage.models.clear_flagged_match_for_person's own query (that one
    clears as it goes; this one only reads) -- used by apply_flagged_person_edit below to preview
    and then reconcile the impact of editing a watch-list entry's own matching fields.
    """
    return [scan for scan in list_scans(conn, limit=100_000) if _already_flagged_with(scan, flagged_person_id)]


def apply_flagged_person_edit(
    conn: sqlite3.Connection,
    original: FlaggedPerson,
    new_name: str | None,
    new_personal_number: str | None,
    new_notes: str | None,
    new_image_path: str | None,
    parent: QWidget,
) -> bool:
    """Apply an edit to an existing flagged-persons watch-list entry, reconciling every scan
    currently flagged against it with whatever changed, and offering to flag any newly-matching
    scan the same way a brand new entry would. Returns True once applied, False if the operator
    canceled after being warned the edit would unflag some scans (nothing is written in that case).

    Shared by both places an operator can correct a flagged person's details: the Edit button on
    ui/settings_window.py's Flagged Users tab, and ui/main_window.py::HistoryRowWidget.
    _on_edit_fields_clicked (correcting a flagged scan's own name propagates back to the master
    watch-list entry it's linked to) -- one shared code path means the two can never silently
    diverge on the warn-before-unflagging behavior.
    """
    updated_person = FlaggedPerson(
        id=original.id,
        created_at=original.created_at,
        name=new_name or None,
        personal_number=new_personal_number or None,
        image_path=new_image_path,
        notes=new_notes or None,
    )

    currently_flagged = scans_flagged_for_person(conn, original.id)
    no_longer_matching: list[ScanRecord] = []
    still_matching: list[tuple[ScanRecord, str]] = []
    for scan in currently_flagged:
        doc_fields = json.loads(scan.doc_fields_json) if scan.doc_fields_json else {}
        match = find_matching_flagged_person(doc_fields, [updated_person])
        if match is None:
            no_longer_matching.append(scan)
        else:
            still_matching.append((scan, match.matched_on))

    if no_longer_matching:
        confirm = QMessageBox.question(
            parent,
            "Edit flagged person",
            f"Changing this entry's name/personal number will remove the flag from "
            f"{len(no_longer_matching)} scan(s) that no longer match the new details. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return False

    update_flagged_person(conn, original.id, new_name, new_personal_number, new_image_path, new_notes)
    logger.info("Operator edited flagged person id=%d", original.id)

    for scan in no_longer_matching:
        set_scan_flagged_match(conn, scan.id, None)
        logger.info("Cleared flag from scan %d: no longer matches edited flagged person id=%d", scan.id, original.id)

    for scan, matched_on in still_matching:
        set_scan_flagged_match(
            conn,
            scan.id,
            {
                "flagged_person_id": original.id,
                "flagged_person_name": updated_person.name,
                "matched_on": matched_on,
                "notes": updated_person.notes,
            },
        )

    run_retroactive_flag_check(conn, updated_person, parent)
    return True


class _RetroactiveMatchDialog(QDialog):
    """A checklist of previously saved scans that now match a flagged-persons entry, each shown with its visitor image, name, and personal number."""

    def __init__(self, candidates: list[tuple[ScanRecord, str]], conn: sqlite3.Connection, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Matching Previous Scans Found")
        self.resize(480, 420)
        restore_geometry(conn, self, _RETROACTIVE_MATCH_DIALOG_GEOMETRY_KEY)
        self.finished.connect(lambda _result: save_geometry(conn, self, _RETROACTIVE_MATCH_DIALOG_GEOMETRY_KEY))
        self._candidates = candidates
        self._checkboxes: list[QCheckBox] = []

        outer_layout = QVBoxLayout(self)
        outer_layout.addWidget(
            QLabel(f"{len(candidates)} previous scan(s) match this flagged person. Select which to mark as flagged:")
        )

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        list_container = QWidget()
        list_layout = QVBoxLayout(list_container)
        for scan, matched_on in candidates:
            list_layout.addWidget(self._build_candidate_row(scan, matched_on))
        list_layout.addStretch(1)
        scroll_area.setWidget(list_container)
        outer_layout.addWidget(scroll_area, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        outer_layout.addWidget(buttons)

    def _build_candidate_row(self, scan: ScanRecord, matched_on: str) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)

        checkbox = QCheckBox()
        checkbox.setChecked(True)
        self._checkboxes.append(checkbox)
        row_layout.addWidget(checkbox)

        thumbnail_label = QLabel()
        thumbnail_label.setFixedSize(*_THUMBNAIL_SIZE)
        pixmap = QPixmap(scan.face_image_path) if scan.face_image_path else None
        if pixmap is not None and not pixmap.isNull():
            thumbnail_label.setPixmap(
                pixmap.scaled(*_THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            thumbnail_label.setText("No\nimage")
            thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail_label.setStyleSheet("color: gray; font-size: 8pt; border: 1px solid gray;")
        row_layout.addWidget(thumbnail_label)

        doc_fields = json.loads(scan.doc_fields_json) if scan.doc_fields_json else {}
        name_display = f"{doc_fields.get('name', '')} {doc_fields.get('surname', '')}".strip() or "(no name)"
        personal_number_display = doc_fields.get("personal_number") or "-"
        info_column = QVBoxLayout()
        info_column.addWidget(QLabel(f"<b>{name_display}</b>  (matched on {matched_on_display(matched_on)})"))
        info_column.addWidget(QLabel(f"Personnummer: {personal_number_display}   Scanned: {scan.created_at}"))
        row_layout.addLayout(info_column, stretch=1)

        return row_widget

    def selected_candidates(self) -> list[tuple[ScanRecord, str]]:
        return [candidate for candidate, checkbox in zip(self._candidates, self._checkboxes) if checkbox.isChecked()]
