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

from document.flag_matcher import find_matching_flagged_person
from storage.models import FlaggedPerson, ScanRecord, list_scans, set_scan_flagged_match

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

    dialog = _RetroactiveMatchDialog(candidates, parent=parent)
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


class _RetroactiveMatchDialog(QDialog):
    """A checklist of previously saved scans that now match a flagged-persons entry, each shown with its visitor image, name, and personal number."""

    def __init__(self, candidates: list[tuple[ScanRecord, str]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Matching Previous Scans Found")
        self.resize(480, 420)
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
        info_column.addWidget(QLabel(f"<b>{name_display}</b>  (matched on {matched_on})"))
        info_column.addWidget(QLabel(f"Personnummer: {personal_number_display}   Scanned: {scan.created_at}"))
        row_layout.addLayout(info_column, stretch=1)

        return row_widget

    def selected_candidates(self) -> list[tuple[ScanRecord, str]]:
        return [candidate for candidate, checkbox in zip(self._candidates, self._checkboxes) if checkbox.isChecked()]
