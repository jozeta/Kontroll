"""Single-screen PyQt6 window: two live camera previews, one capture button, and a scan history list."""

import base64
import json
import logging
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PyQt6.QtCore import QByteArray, QDate, QSize, Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QColor, QImage, QKeySequence, QMouseEvent, QPixmap, QResizeEvent, QShortcut
from PyQt6.QtWidgets import (
    QCalendarWidget,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import CameraManager
from capture.frame_capture import capture_still_frame
from config.app_paths import get_documents_dir, get_faces_dir
from config.settings import (
    get_capture_shortcut,
    get_disk_space_warning_mb,
    get_face_match_threshold,
    get_setting_or_none,
    set_setting,
)
from document.barcode_parser import decode_pdf417_barcodes
from document.country_flags import alpha2_for_country_code
from document.dates import calculate_age, parse_mrz_style_date
from document.dl_field_parser import parse_driving_licence_fields
from document.doc_classifier import classify_document
from document.field_crosscheck import check_person_flags, determine_validation_status
from document.image_quality import assess_image_quality
from document.mrz_parser import parse_and_validate_mrz
from document.ocr import extract_text
from face.detector import detect_faces
from face.embedder import get_best_face_embedding
from face.matcher import match_faces
from storage.models import (
    ScanRecord,
    calculate_scan_storage_bytes,
    count_scans,
    delete_scan,
    get_scan_by_id,
    insert_scan,
    list_scans,
    update_match_result,
)
from ui.flag_icons import render_flag_icon
from ui.scan_export import build_scan_report_html
from ui.settings_window import SettingsWindow

# Each independently resolved (and auto-created if missing) via config.app_paths rather than a
# path relative to this file's own location -- see that module for why (packaged-EXE safety) and
# how each is independently operator-configurable.
FACE_IMAGE_DIR = get_faces_dir()
DOC_IMAGE_DIR = get_documents_dir()

logger = logging.getLogger(__name__)

_PREVIEW_INTERVAL_MS = 33  # ~30fps live preview polling
_PREVIEW_SIZE = (320, 240)
_THUMBNAIL_SIZE = (110, 82)

# Consecutive failed preview-tick reads (at _PREVIEW_INTERVAL_MS apart) before a camera's frozen
# last-good preview frame is replaced with an explicit "signal lost" message -- see
# MainWindow._maybe_show_signal_lost. ~1 second's worth, so a single dropped tick (which happens
# occasionally and harmlessly) doesn't flicker the warning on and off.
_SIGNAL_LOST_TICK_THRESHOLD = round(1000 / _PREVIEW_INTERVAL_MS)

_OVERRIDE_CHOICES = ["Automatic", "Override: Match", "Override: No Match"]

# Opacity applied to the date-filter dropdown while "Any date" is checked -- dims it to signal
# it's inactive without actually disabling it, since a disabled widget can't receive the click
# that opens its calendar popup (see MainWindow._on_date_calendar_shown).
_DATE_FILTER_DIMMED_OPACITY = 0.4

# How many rows the history list is sized to show, without scrolling, the first time the app is
# opened. It's a floor (via QScrollArea.setMinimumSize), not a cap -- the list's own stretch
# factor lets it grow past this whenever the window is resized bigger.
_DEFAULT_VISIBLE_HISTORY_ROWS = 5

# settings-table keys the window's size/position and the top pane's height are persisted under
# (see MainWindow._restore_window_state / _save_window_state). The geometry is base64 text
# wrapping the raw QByteArray that Qt's own saveGeometry() produces; the pane height is stored as
# a plain pixel integer instead of via QSplitter's saveState()/restoreState() -- see
# _restore_window_state's docstring for why.
_WINDOW_GEOMETRY_SETTING_KEY = "window_geometry"
_TOP_PANE_HEIGHT_SETTING_KEY = "top_pane_height"


def _format_timestamp(created_at: str) -> str:
    """Format a scan's stored ISO-format created_at timestamp as 'YYYY-MM-DD HH:MM:SS' for display."""
    return datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M:%S")


def _bgr_frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame into a QPixmap suitable for display in a QLabel."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    # .copy() detaches the QImage from rgb_frame's buffer, which is a stack-local that would
    # otherwise be freed once this function returns.
    qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def _load_thumbnail(image_path: str | None) -> QPixmap | None:
    """Load a saved scan image from disk and scale it down to thumbnail size for the history list."""
    if not image_path or not Path(image_path).exists():
        return None
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    pixmap = _bgr_frame_to_pixmap(frame)
    return pixmap.scaled(
        _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


def _match_icon_text(match_result: str | None) -> str:
    """Return a short, visually distinct glyph+label summarizing the face-match result for the history list."""
    if match_result in ("match", "override_match"):
        return "✅ MATCH"
    if match_result in ("no_match", "override_no_match"):
        return "❌ NO MATCH"
    return "➖ N/A"


@dataclass
class DisplayFields:
    """Name/birth-date/age fields derived from a scan's stored MRZ data, for display in the history list."""

    first_name: str
    last_name: str
    birth_date_display: str
    age_display: str


def _derive_display_fields(doc_fields_json: str | None) -> DisplayFields:
    """Extract first name, last name, birth date, and computed age from a scan's stored structured fields, if any.

    doc_fields_json holds whichever structured fields were available for this scan: MRZ fields for
    passports/ID cards, or the OCR-heuristic-derived fields from document/dl_field_parser.py for
    driving licences (see ui/main_window.py's capture handler for which one wins). Documents with
    neither show '-' placeholders here rather than a guessed value, consistent with how
    validation_status handles the same gap.
    """
    if not doc_fields_json:
        return DisplayFields("-", "-", "-", "-")

    fields = json.loads(doc_fields_json)
    first_name = fields.get("name", "").strip() or "-"
    last_name = fields.get("surname", "").strip() or "-"

    birth_date = parse_mrz_style_date(fields.get("birth_date", ""), assume_past=True)
    if birth_date is None:
        return DisplayFields(first_name, last_name, "-", "-")

    return DisplayFields(first_name, last_name, birth_date.isoformat(), str(calculate_age(birth_date)))


# Never persisted or displayed -- used only to measure a representative row's natural size for
# picking the history list's default minimum size (see MainWindow._measure_default_history_size).
# The name/status text lengths are chosen to be reasonably long, similar to real MRZ data, so the
# measured width isn't optimistically narrow.
_SAMPLE_SCAN_FOR_SIZING = ScanRecord(
    id=0,
    created_at=datetime.now().isoformat(),
    operator=None,
    face_image_path=None,
    doc_image_path=None,
    face_embedding=None,
    match_score=0.9876,
    match_result="match",
    doc_type="passport",
    doc_country="SWE",
    doc_fields_json=json.dumps({"name": "MARGARETHA", "surname": "LINDQVIST-JOHANSSON", "birth_date": "850615"}),
    validation_status="valid",
    validation_detail_json=json.dumps({"notes": [], "field_comparisons": []}),
)


def _final_match_result_label(automatic_is_match: bool | None, override_choice: str) -> str:
    """Compute the match_result string to persist, combining the automated result with any operator override choice."""
    if override_choice == "Override: Match":
        return "override_match"
    if override_choice == "Override: No Match":
        return "override_no_match"
    if automatic_is_match:
        return "match"
    return "no_match"


class ImageViewerWindow(QWidget):
    """A standalone, freely-resizable window showing one scan image at a much larger size.

    Keeps the original full-resolution QPixmap and rescales it to fit the window on every
    resize, rather than stretching a pre-scaled thumbnail, so the image stays sharp at any
    window size.
    """

    def __init__(self, image_path: str, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._original_pixmap = QPixmap(image_path)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout = QVBoxLayout(self)
        layout.addWidget(self.image_label)

        if self._original_pixmap.isNull():
            self.image_label.setText("Image not found")
        self.resize(900, 700)
        self._update_scaled_pixmap()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._original_pixmap.isNull():
            return
        self.image_label.setPixmap(
            self._original_pixmap.scaled(
                self.image_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )


class _ClickableThumbnailLabel(QLabel):
    """A thumbnail QLabel that opens the underlying image full-size in a resizable ImageViewerWindow on double-click."""

    def __init__(self, image_path: str | None, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._title = title
        # Kept alive by this reference for as long as the thumbnail itself lives (i.e. for the
        # app's lifetime, since history rows are never removed) -- a top-level PyQt widget with no
        # Python reference held anywhere is garbage-collected out from under its own open window.
        self._viewer: ImageViewerWindow | None = None

    def mouseDoubleClickEvent(self, event) -> None:
        if self._image_path and Path(self._image_path).exists():
            self._viewer = ImageViewerWindow(self._image_path, self._title)
            self._viewer.show()
        super().mouseDoubleClickEvent(event)


class HistoryRowWidget(QFrame):
    """One row in the scan history list: both thumbnails, extracted identity fields, the match icon, and an expandable detail/override section."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        scan: ScanRecord,
        on_delete: Callable[["HistoryRowWidget"], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.conn = conn
        self.scan = scan
        self._on_delete = on_delete
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._build_layout()

    def _build_layout(self) -> None:
        """Build the always-visible summary row and the hidden-by-default detail section beneath it."""
        outer_layout = QVBoxLayout(self)

        summary_row = QHBoxLayout()
        summary_row.addWidget(self._build_thumbnail(self.scan.face_image_path, "Live Face Capture"))
        summary_row.addWidget(self._build_doc_thumbnail_column())

        display_fields = _derive_display_fields(self.scan.doc_fields_json)
        identity_column = QVBoxLayout()
        identity_column.addWidget(QLabel(f"{display_fields.first_name} {display_fields.last_name}"))
        identity_column.addWidget(QLabel(f"DOB: {display_fields.birth_date_display}   Age: {display_fields.age_display}"))
        identity_column.addWidget(self._build_country_row())
        identity_column.addWidget(QLabel(_format_timestamp(self.scan.created_at)))
        summary_row.addLayout(identity_column, stretch=1)
        summary_row.addSpacing(32)

        self.match_icon_label = QLabel(_match_icon_text(self.scan.match_result))
        self.match_icon_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        summary_row.addWidget(self.match_icon_label)

        self.status_label = QLabel(f"Doc: {(self.scan.validation_status or '-').upper()}")
        summary_row.addWidget(self.status_label)

        person_flags_text = self._person_flags_badge_text()
        if person_flags_text:
            self.person_flags_label = QLabel(f"⚠ {person_flags_text}")
            self.person_flags_label.setStyleSheet(
                "color: white; background-color: #C62828; font-weight: bold; padding: 2px 6px; border-radius: 3px;"
            )
            summary_row.addWidget(self.person_flags_label)

        self.detail_toggle = QPushButton("Details ▾")
        self.detail_toggle.setCheckable(True)
        self.detail_toggle.toggled.connect(self._on_detail_toggle)
        summary_row.addWidget(self.detail_toggle)

        outer_layout.addLayout(summary_row)
        outer_layout.addWidget(self._build_detail_section())

    def _build_country_row(self) -> QWidget:
        """Build the row showing the identified document country: a small drawn flag icon (see ui/flag_icons.py) plus the country code as plain text."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        alpha2_code = alpha2_for_country_code(self.scan.doc_country)
        if alpha2_code:
            icon_label = QLabel()
            icon_label.setPixmap(render_flag_icon(alpha2_code))
            row_layout.addWidget(icon_label)

        text_label = QLabel(self.scan.doc_country or "-")
        text_label.setTextFormat(Qt.TextFormat.PlainText)
        row_layout.addWidget(text_label)
        row_layout.addStretch(1)

        return row_widget

    _QUALITY_BANNER_HEIGHT = 14

    def _build_doc_thumbnail_column(self) -> QWidget:
        """Build the document thumbnail, with a passive low-quality banner overlaid flush across its top edge when the capture was flagged blurry/glare-heavy.

        The banner is purely informational -- it was decided at capture time (see
        MainWindow._on_capture_and_process_clicked) and stored in validation_detail_json's
        "image_quality" key, never re-evaluated here, and never blocks or prompts anything.
        It's positioned with explicit geometry (rather than added to a layout) so it sits exactly
        flush against the image's top edge at full thumbnail width, instead of a layout-managed
        widget shrinking to its own size hint and floating disconnected from the image. This also
        keeps the container's own size fixed to the thumbnail size regardless of whether a banner
        is present, so the document thumbnail never shifts relative to the face thumbnail beside
        it (which never gets a banner).
        """
        container = QWidget()
        container.setFixedSize(*_THUMBNAIL_SIZE)

        thumbnail = self._build_thumbnail(self.scan.doc_image_path, "Document Capture")
        thumbnail.setParent(container)
        thumbnail.move(0, 0)

        quality_warning_text = self._doc_image_quality_warning_text()
        if quality_warning_text:
            banner = QLabel(f"⚠ {quality_warning_text}", container)
            banner.setStyleSheet(
                "color: white; background-color: rgba(255, 152, 0, 130); font-weight: bold; font-size: 7pt;"
            )
            banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
            banner.setGeometry(0, 0, _THUMBNAIL_SIZE[0], self._QUALITY_BANNER_HEIGHT)
            banner.raise_()

        return container

    def _person_flags_badge_text(self) -> str | None:
        """Return a short banner label for this scan if it was flagged as under 18 and/or expired, else None.

        Reads the "person_flags" key stored in validation_detail_json at capture time (see
        MainWindow._on_capture_and_process_clicked / document.field_crosscheck.check_person_flags)
        -- never re-evaluated here.
        """
        if not self.scan.validation_detail_json:
            return None
        detail = json.loads(self.scan.validation_detail_json)
        flags = detail.get("person_flags")
        if not flags:
            return None
        labels = []
        if flags.get("is_minor"):
            labels.append("UNDER 18")
        if flags.get("is_expired"):
            labels.append("EXPIRED")
        return " / ".join(labels) if labels else None

    def _doc_image_quality_warning_text(self) -> str | None:
        """Return the stored image-quality warning text for this scan's document capture, or None if it was fine (or never assessed)."""
        if not self.scan.validation_detail_json:
            return None
        detail = json.loads(self.scan.validation_detail_json)
        quality = detail.get("image_quality")
        if not quality or not quality.get("has_warning"):
            return None
        return f"Low quality ({quality.get('text', 'unknown')})"

    def _build_thumbnail(self, image_path: str | None, title: str) -> QLabel:
        """Build a fixed-size thumbnail label for one of the row's two saved images. Double-clicking it opens the full image in a resizable viewer window."""
        label = _ClickableThumbnailLabel(image_path, title)
        label.setFixedSize(*_THUMBNAIL_SIZE)
        pixmap = _load_thumbnail(image_path)
        if pixmap is not None:
            label.setPixmap(pixmap)
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.setToolTip("Double-click to view full size")
        else:
            label.setText("No image")
        return label

    def _build_detail_section(self) -> QWidget:
        """Build the collapsible detail section: score, document classification, cross-check notes, and the override control."""
        self.detail_widget = QWidget()
        self.detail_widget.setVisible(False)
        layout = QVBoxLayout(self.detail_widget)

        score_text = f"{self.scan.match_score:.4f}" if self.scan.match_score is not None else "-"
        layout.addWidget(QLabel(f"Raw similarity score: {score_text}"))
        layout.addWidget(QLabel(f"Document type: {self.scan.doc_type or '-'}   Country: {self.scan.doc_country or '-'}"))

        if self.scan.validation_detail_json:
            detail = json.loads(self.scan.validation_detail_json)
            for note in detail.get("notes", []):
                layout.addWidget(QLabel(f"• {note}"))
            for comparison in detail.get("field_comparisons", []):
                status = "OK" if comparison.get("found_in_ocr") else "MISMATCH"
                layout.addWidget(QLabel(f"[{status}] {comparison.get('field_name')}: {comparison.get('machine_read_value')}"))

        override_row = QHBoxLayout()
        override_row.addWidget(QLabel("Operator override:"))
        self.override_combo = QComboBox()
        self.override_combo.addItems(_OVERRIDE_CHOICES)
        self.override_combo.setCurrentText(self._initial_override_choice())
        override_row.addWidget(self.override_combo)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._on_apply_override_clicked)
        override_row.addWidget(apply_button)

        export_button = QPushButton("Export report")
        export_button.clicked.connect(self._on_export_clicked)
        override_row.addWidget(export_button)

        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet("color: #FF1744;")
        delete_button.setMaximumWidth(70)
        delete_button.clicked.connect(self._on_delete_clicked)
        override_row.addWidget(delete_button)

        layout.addLayout(override_row)

        return self.detail_widget

    def _initial_override_choice(self) -> str:
        """Pre-select the override combo to reflect this scan's currently stored match_result, so a prior override isn't lost from view."""
        if self.scan.match_result == "override_match":
            return "Override: Match"
        if self.scan.match_result == "override_no_match":
            return "Override: No Match"
        return "Automatic"

    def _on_detail_toggle(self, checked: bool) -> None:
        """Show or hide this row's detail section when its toggle button is clicked."""
        self.detail_widget.setVisible(checked)
        self.detail_toggle.setText("Details ▴" if checked else "Details ▾")

    def _on_apply_override_clicked(self) -> None:
        """Persist the operator's override choice for this scan and refresh the row's match icon to reflect it."""
        automatic_is_match = self.scan.match_result in ("match", "override_match") if self.scan.match_result else None
        # The automated result itself never changes after the fact -- only whichever of
        # 'match'/'no_match' it originally landed on is recoverable from the stored label once
        # an override has already been applied once, which is a known, accepted limitation of
        # not storing the automatic and final results as two separate columns.
        new_label = _final_match_result_label(automatic_is_match, self.override_combo.currentText())
        update_match_result(self.conn, self.scan.id, new_label)
        logger.info("Operator overrode scan %d match_result to '%s'", self.scan.id, new_label)
        self.scan.match_result = new_label
        self.match_icon_label.setText(_match_icon_text(new_label))

    def _on_export_clicked(self) -> None:
        """Write this scan's images and validation summary to an operator-chosen, self-contained HTML file."""
        display_fields = _derive_display_fields(self.scan.doc_fields_json)
        display_name = f"{display_fields.first_name} {display_fields.last_name}".strip()
        report_html = build_scan_report_html(
            self.scan, display_name, display_fields.birth_date_display, display_fields.age_display
        )

        default_name = f"kontroll_scan_{self.scan.id}_{datetime.fromisoformat(self.scan.created_at).strftime('%Y%m%d_%H%M%S')}.html"
        file_path, _ = QFileDialog.getSaveFileName(self, "Export scan report", default_name, "HTML files (*.html)")
        if not file_path:
            return

        Path(file_path).write_text(report_html, encoding="utf-8")
        logger.info("Operator exported scan %d report to %s", self.scan.id, file_path)
        QMessageBox.information(self, "Export scan report", f"Saved to:\n{file_path}")

    def _on_delete_clicked(self) -> None:
        """Permanently delete this scan's DB row and saved images after operator confirmation, and remove this row from the list."""
        confirm = QMessageBox.question(
            self,
            "Delete scan",
            "Permanently delete this scan result, including its saved images? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        delete_scan(self.conn, self.scan.id)
        for image_path in (self.scan.face_image_path, self.scan.doc_image_path):
            if image_path and Path(image_path).exists():
                Path(image_path).unlink()
        logger.info("Operator deleted scan %d", self.scan.id)

        if self._on_delete is not None:
            self._on_delete(self)
        self.setParent(None)
        self.deleteLater()


class ResultAwareCalendarWidget(QCalendarWidget):
    """A date-picker calendar popup where only dates with at least one saved scan are selectable.

    Dates with a scan are painted white; dates without one are painted grey and are effectively
    unclickable -- QCalendarWidget has no native way to disable individual dates, so a click on a
    grey date is instead caught via selectionChanged and immediately reverted to the last valid
    (scan-having) date that was selected.
    """

    def __init__(
        self,
        result_dates_provider: Callable[[], set[date]],
        on_shown: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._result_dates_provider = result_dates_provider
        self._on_shown = on_shown
        self._last_valid_date = self.selectedDate()
        self.selectionChanged.connect(self._on_selection_changed)

    def showEvent(self, event) -> None:
        """Fire the on_shown hook the moment this popup actually becomes visible -- i.e. exactly when the operator opened it, in a single click."""
        super().showEvent(event)
        if self._on_shown is not None:
            self._on_shown()

    def _on_selection_changed(self) -> None:
        if self.selectedDate().toPyDate() in self._result_dates_provider():
            self._last_valid_date = self.selectedDate()
        else:
            self.setSelectedDate(self._last_valid_date)

    def paintCell(self, painter, rect, date) -> None:
        """Paint the native cell first (grid/background/native theme), then recolor just the day-number text on top."""
        super().paintCell(painter, rect, date)
        has_results = date.toPyDate() in self._result_dates_provider()
        painter.save()
        painter.setPen(QColor("white") if has_results else QColor("#808080"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(date.day()))
        painter.restore()


class _ClickAnywhereDateEdit(QDateEdit):
    """A QDateEdit whose calendar popup opens on a click anywhere in the field, not just its narrow dropdown-arrow button.

    QDateTimeEdit's own internal click handling only opens the popup for a click landing exactly
    on its internal calendar-button sub-control -- a hit-test region whose precise geometry is
    determined by the active QStyle, and so isn't reliably reproducible by synthesizing a click
    aimed at a particular sub-control (that approach worked under the Fusion style used in headless
    testing, but not under Windows' native style used when actually running the app). Instead, any
    left-click anywhere in the field directly shows the same popup container QDateEdit itself
    would show, positioned explicitly just below the field -- sidestepping style-specific hit-test
    geometry entirely.
    """

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._popup_is_visible():
            self._show_calendar_popup()
            return
        super().mousePressEvent(event)

    def _popup_is_visible(self) -> bool:
        popup = self.calendarWidget().parent()
        return popup is not None and popup.isVisible()

    def _show_calendar_popup(self) -> None:
        popup = self.calendarWidget().parent()
        if popup is None:
            return
        popup.move(self.mapToGlobal(self.rect().bottomLeft()))
        popup.show()


class MainWindow(QMainWindow):
    """The app's single combined screen: live previews, one capture-and-process button, and the scan history list."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        camera_manager: CameraManager,
        face_analysis_app: FaceAnalysis,
    ) -> None:
        super().__init__()
        self.conn = conn
        self.camera_manager = camera_manager
        self.face_analysis_app = face_analysis_app
        self._settings_window: SettingsWindow | None = None
        # A capture click that gets a frame from only one camera holds onto that successful frame
        # here rather than discarding it, so the next click only has to retry the failed side
        # instead of redoing a capture that already succeeded -- see
        # _on_capture_and_process_clicked and _on_discard_pending_capture_clicked.
        self._pending_face_frame: np.ndarray | None = None
        self._pending_doc_frame: np.ndarray | None = None
        # Consecutive failed preview-tick reads per camera, since the last successful one -- once
        # either crosses _SIGNAL_LOST_TICK_THRESHOLD, _update_previews swaps that camera's frozen
        # last-good frame for an explicit "signal lost" message rather than leaving it looking
        # like a normal (if slightly stale) live feed. A single dropped tick doesn't count; only a
        # sustained run of them does, so an occasional one-off missed frame doesn't flicker it.
        self._face_signal_lost_streak = 0
        self._doc_signal_lost_streak = 0

        self.setWindowTitle("Kontroll")
        self._build_layout()
        self._load_history()
        self._restore_window_state()

        self.capture_shortcut = QShortcut(QKeySequence(get_capture_shortcut(self.conn)), self)
        self.capture_shortcut.activated.connect(self._on_capture_and_process_clicked)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_previews)
        self.preview_timer.start(_PREVIEW_INTERVAL_MS)

    def _build_layout(self) -> None:
        """Construct and arrange every widget on the single combined screen.

        A vertical splitter divides the window into two panes with a drag handle between them,
        positioned right below the capture button: the top pane (previews + capture button) and
        the bottom pane (history headline + scan list). QSplitter's own resize-distribution (via
        setStretchFactor) does not reliably keep one pane fixed on a plain window resize in
        practice, so that's instead enforced explicitly in resizeEvent(), which pins the top pane
        at its last explicit height and hands all resize delta to the bottom pane -- where the
        headline stays fixed-size and the scan list (stretch=1) is what actually absorbs it, so
        the list's own height tracks the window rather than a fixed pixel count. Dragging the
        handle itself still freely resizes the top pane (see _on_splitter_moved), which is what
        lets the previews grow or shrink on demand.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)

        self.top_bottom_splitter = QSplitter(Qt.Orientation.Vertical)
        outer_layout.addWidget(self.top_bottom_splitter)

        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        previews_row = QHBoxLayout()
        self.face_preview_label = self._build_preview_label()
        self.doc_preview_label = self._build_preview_label()
        previews_row.addLayout(self._build_preview_column("Visitor", self.face_preview_label), stretch=1)
        previews_row.addLayout(self._build_preview_column("Identification", self.doc_preview_label), stretch=1)
        top_layout.addLayout(previews_row, stretch=1)

        capture_row = QHBoxLayout()
        self.capture_button = QPushButton("Capture && Process")
        self.capture_button.setMinimumHeight(48)
        self.capture_button.clicked.connect(self._on_capture_and_process_clicked)
        capture_row.addWidget(self.capture_button, stretch=1)

        self.discard_pending_button = QPushButton("Discard held capture")
        self.discard_pending_button.setMinimumHeight(48)
        self.discard_pending_button.setVisible(False)
        self.discard_pending_button.clicked.connect(self._on_discard_pending_capture_clicked)
        capture_row.addWidget(self.discard_pending_button)

        top_layout.addLayout(capture_row)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(QLabel("Previous scans (newest first):"))
        bottom_layout.addLayout(self._build_filter_row())

        self._history_rows: list[HistoryRowWidget] = []

        self.history_scroll_area = QScrollArea()
        self.history_scroll_area.setWidgetResizable(True)
        self.history_scroll_area.setMinimumSize(self._measure_default_history_size())
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.addStretch(1)  # keeps rows top-aligned as they're inserted above this
        self.history_scroll_area.setWidget(self.history_container)
        bottom_layout.addWidget(self.history_scroll_area, stretch=1)  # fills the bottom pane's space and grows/shrinks with it

        self.top_bottom_splitter.addWidget(top_widget)
        self.top_bottom_splitter.addWidget(bottom_widget)
        self.top_bottom_splitter.setCollapsible(0, False)
        self.top_bottom_splitter.setCollapsible(1, False)
        self.top_bottom_splitter.splitterMoved.connect(self._on_splitter_moved)

        self._top_pane_widget = top_widget
        self._top_pane_height: int | None = None  # learned from top_widget's natural sizeHint on first resizeEvent

        self._build_status_bar()

    def _build_status_bar(self) -> None:
        """Build the window's status bar: scan-count/disk-usage stats on the left, a Settings button on the right."""
        # QStatusBar draws a sunken-border frame around every widget added via addWidget/
        # addPermanentWidget by default, which shows up as a vertical divider line next to each
        # one -- this app doesn't want those dividers, so they're switched off here.
        self.statusBar().setStyleSheet("QStatusBar::item { border: none; }")

        self.status_stats_label = QLabel()
        self.statusBar().addWidget(self.status_stats_label)

        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._on_settings_clicked)
        self.statusBar().addPermanentWidget(settings_button)

        self._refresh_status_bar_stats()

    def _refresh_status_bar_stats(self) -> None:
        """Recompute and display the total scan count, total on-disk size of saved scan images, and a low-disk-space warning if applicable."""
        scan_count = count_scans(self.conn)
        storage_mb = calculate_scan_storage_bytes(FACE_IMAGE_DIR, DOC_IMAGE_DIR) / (1024 * 1024)
        stats_text = f"{scan_count} scan(s)  ·  {storage_mb:.1f} MB on disk"

        free_mb = shutil.disk_usage(FACE_IMAGE_DIR).free / (1024 * 1024)
        warning_threshold_mb = get_disk_space_warning_mb(self.conn)
        if free_mb < warning_threshold_mb:
            stats_text += f"  ·  ⚠ Low disk space: {free_mb:,.0f} MB free"
            self.status_stats_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.status_stats_label.setStyleSheet("")

        self.status_stats_label.setText(stats_text)

    def _on_settings_clicked(self) -> None:
        """Open the Settings window, creating it once and reusing it on subsequent clicks."""
        logger.info("Operator opened the Settings window")
        if self._settings_window is None:
            self._settings_window = SettingsWindow(
                self.conn,
                self.camera_manager,
                self.face_analysis_app,
                on_data_changed=self._on_settings_data_changed,
                on_capture_shortcut_changed=self._on_capture_shortcut_changed,
            )
        self._settings_window.show()
        self._settings_window.raise_()
        self._settings_window.activateWindow()

    def _on_settings_data_changed(self) -> None:
        """Reload the history list and refresh the status bar stats after the settings window purges scans out from under it."""
        self._reload_history()
        self._refresh_status_bar_stats()

    def _reload_history(self) -> None:
        """Discard every currently-displayed history row and reload the full list fresh from the database."""
        for row in self._history_rows:
            row.setParent(None)
            row.deleteLater()
        self._history_rows.clear()
        self._load_history()

    def _on_capture_shortcut_changed(self, sequence_text: str) -> None:
        """Re-key the capture keyboard shortcut after it's changed and saved in the settings window."""
        self.capture_shortcut.setKey(QKeySequence(sequence_text))

    def _build_filter_row(self) -> QHBoxLayout:
        """Build the history filter controls: a capture-date filter, a match-result filter, and a free-text name search box."""
        filter_row = QHBoxLayout()

        filter_row.addWidget(QLabel("Date:"))
        self.date_filter_checkbox = QCheckBox("Any date")
        self.date_filter_checkbox.setChecked(True)
        self.date_filter_checkbox.toggled.connect(self._on_date_filter_checkbox_toggled)
        filter_row.addWidget(self.date_filter_checkbox)

        self.date_filter_edit = _ClickAnywhereDateEdit(QDate.currentDate())
        self.date_filter_edit.setCalendarPopup(True)
        # Deliberately left enabled even while "Any date" is checked: a disabled widget swallows
        # the click that would open its calendar dropdown entirely, forcing the operator to
        # uncheck the box first and click a second time. Instead, ResultAwareCalendarWidget's
        # on_shown hook unchecks the box the instant the popup actually opens, so one click both
        # opens the calendar and puts the date picker into effect.
        self.date_filter_edit.setCalendarWidget(
            ResultAwareCalendarWidget(self._scan_result_dates, on_shown=self._on_date_calendar_shown)
        )
        self.date_filter_edit.dateChanged.connect(self._apply_filters)
        self._date_filter_opacity_effect = QGraphicsOpacityEffect(self.date_filter_edit)
        self.date_filter_edit.setGraphicsEffect(self._date_filter_opacity_effect)
        self._update_date_filter_dim_state()
        filter_row.addWidget(self.date_filter_edit)

        filter_row.addWidget(QLabel("Result:"))
        self.match_filter_combo = QComboBox()
        self.match_filter_combo.addItems(["All", "Match", "No Match"])
        self.match_filter_combo.currentTextChanged.connect(self._apply_filters)
        filter_row.addWidget(self.match_filter_combo)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search name...")
        self.search_edit.textChanged.connect(self._apply_filters)
        filter_row.addWidget(self.search_edit, stretch=1)

        clear_filters_button = QPushButton("Clear filters")
        clear_filters_button.clicked.connect(self._on_clear_filters_clicked)
        filter_row.addWidget(clear_filters_button)

        return filter_row

    def _on_date_filter_checkbox_toggled(self, checked: bool) -> None:
        """Dim the date picker to match whether the 'any date' checkbox is checked, then re-apply filters."""
        self._update_date_filter_dim_state()
        self._apply_filters()

    def _update_date_filter_dim_state(self) -> None:
        """Set the date picker's opacity to reflect whether it's currently in effect (unchecked) or not (checked)."""
        is_dimmed = self.date_filter_checkbox.isChecked()
        self._date_filter_opacity_effect.setOpacity(_DATE_FILTER_DIMMED_OPACITY if is_dimmed else 1.0)

    def _on_date_calendar_shown(self) -> None:
        """Uncheck 'Any date' as soon as its calendar dropdown is opened, so picking a date takes effect immediately without a separate uncheck step."""
        self.date_filter_checkbox.setChecked(False)

    def _on_clear_filters_clicked(self) -> None:
        """Reset every filter control to its default (unfiltered) state and refresh the visible rows."""
        self.date_filter_checkbox.setChecked(True)
        self.date_filter_edit.setDate(QDate.currentDate())
        self.match_filter_combo.setCurrentText("All")
        self.search_edit.clear()
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Show or hide each history row so only scans matching the current date/result/search filters remain visible."""
        match_filter = self.match_filter_combo.currentText()
        search_text = self.search_edit.text().strip().lower()
        date_filter_active = not self.date_filter_checkbox.isChecked()
        selected_date = self.date_filter_edit.date().toPyDate()

        for row in self._history_rows:
            scan = row.scan
            visible = True

            if date_filter_active:
                visible = visible and datetime.fromisoformat(scan.created_at).date() == selected_date

            if match_filter == "Match":
                visible = visible and scan.match_result in ("match", "override_match")
            elif match_filter == "No Match":
                visible = visible and scan.match_result in ("no_match", "override_no_match")

            if search_text:
                display_fields = _derive_display_fields(scan.doc_fields_json)
                haystack = f"{display_fields.first_name} {display_fields.last_name}".lower()
                visible = visible and search_text in haystack

            row.setVisible(visible)

    def _scan_result_dates(self) -> set[date]:
        """Return the distinct calendar dates that have at least one saved scan, for ResultAwareCalendarWidget to highlight."""
        return {datetime.fromisoformat(row.scan.created_at).date() for row in self._history_rows}

    def _measure_default_history_size(self) -> QSize:
        """Measure a representative history row to pick the scan list's default minimum size.

        Building a throwaway row with realistic-length sample text (rather than guessing pixel
        values) is what keeps this accurate if fonts/styling change later. Sized
        _DEFAULT_VISIBLE_HISTORY_ROWS rows tall and one row's natural width wide, plus room for
        the scrollbar itself, so a freshly-opened window shows a full row without a horizontal
        scrollbar and ~5 rows without a vertical one -- it's only a floor (setMinimumSize), so the
        list can still grow past it once the window is resized bigger. The extra 15% width
        headroom (beyond the scrollbar's own pixel extent) covers font-metric differences between
        environments -- the measured sizeHint() here is a few percent narrower than an identical
        row actually renders at on a real Windows install, which was otherwise just barely enough
        to trigger a horizontal scrollbar.
        """
        sample_row = HistoryRowWidget(self.conn, _SAMPLE_SCAN_FOR_SIZING)
        row_size = sample_row.sizeHint()
        sample_row.deleteLater()
        scrollbar_extent = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        width = int(row_size.width() * 1.15) + scrollbar_extent
        return QSize(width, row_size.height() * _DEFAULT_VISIBLE_HISTORY_ROWS)

    def _build_preview_label(self) -> QLabel:
        """Build one live preview label that expands to fill whatever space its pane currently has.

        A minimum size (rather than a fixed one) keeps it from collapsing to nothing when the
        splitter handle above is dragged small, while still letting it grow -- _update_previews
        rescales each incoming frame to the label's current size on every tick.
        """
        label = QLabel("No camera signal")
        label.setMinimumSize(*_PREVIEW_SIZE)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setFrameShape(QFrame.Shape.Box)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _build_preview_column(self, headline_text: str, preview_label: QLabel) -> QVBoxLayout:
        """Stack a centered, bold, white headline above one camera preview label."""
        column = QVBoxLayout()

        headline = QLabel(headline_text)
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        headline.setStyleSheet("color: white; font-size: 16pt; font-weight: bold;")
        column.addWidget(headline)
        column.addWidget(preview_label, stretch=1)

        return column

    def _load_history(self) -> None:
        """Populate the history list with every previously saved scan, newest first.

        _add_history_row always inserts at the top, so the scans are fed in here oldest-first
        (the reverse of list_scans' own newest-first order) -- each insert-at-top then pushes
        the previous one down by exactly one, leaving the newest scan on top once the loop ends.
        """
        for scan in reversed(list_scans(self.conn)):
            self._add_history_row(scan)
        self._apply_filters()

    def _add_history_row(self, scan: ScanRecord) -> None:
        """Insert one scan as the top-most row of the history list."""
        row = HistoryRowWidget(self.conn, scan, on_delete=self._on_history_row_deleted)
        self.history_layout.insertWidget(0, row)
        self._history_rows.append(row)

    def _on_history_row_deleted(self, row: HistoryRowWidget) -> None:
        """Drop a deleted row's own bookkeeping entry once it has removed itself (DB row + image files already gone by this point)."""
        if row in self._history_rows:
            self._history_rows.remove(row)
        self._refresh_status_bar_stats()

    def _update_previews(self) -> None:
        """Poll both cameras for their latest frame and refresh the live preview labels. Runs on a timer, independent of capture."""
        face_frame = self.camera_manager.read_face_preview_frame()
        if face_frame is not None:
            self._face_signal_lost_streak = 0
            self._set_preview_pixmap(self.face_preview_label, face_frame)
        else:
            self._face_signal_lost_streak += 1
            self._maybe_show_signal_lost(self.face_preview_label, self._face_signal_lost_streak)

        doc_frame = self.camera_manager.read_doc_preview_frame()
        if doc_frame is not None:
            self._doc_signal_lost_streak = 0
            self._set_preview_pixmap(self.doc_preview_label, doc_frame)
        else:
            self._doc_signal_lost_streak += 1
            self._maybe_show_signal_lost(self.doc_preview_label, self._doc_signal_lost_streak)

    def _maybe_show_signal_lost(self, label: QLabel, lost_streak: int) -> None:
        """Once a camera has failed to return a frame for _SIGNAL_LOST_TICK_THRESHOLD consecutive ticks, replace its frozen last-good frame with an explicit warning rather than leaving it looking like a live feed."""
        if lost_streak == _SIGNAL_LOST_TICK_THRESHOLD:
            label.clear()
            label.setText("⚠ Camera signal lost")
            label.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 12pt;")

    def _set_preview_pixmap(self, label: QLabel, frame: np.ndarray) -> None:
        """Scale a captured frame to fill as much of the label's current size as possible, preserving aspect ratio.

        Rescaling to label.size() on every tick (rather than the old fixed _PREVIEW_SIZE) is what
        lets the live feeds visibly grow or shrink as the operator drags the splitter handle above
        the capture button to resize the preview pane.
        """
        label.setStyleSheet("")  # clears any "signal lost" styling left over from _maybe_show_signal_lost
        pixmap = _bgr_frame_to_pixmap(frame)
        label.setPixmap(
            pixmap.scaled(label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def _on_capture_and_process_clicked(self) -> None:
        """Grab a still frame from both cameras, run the full face-match and document-validation pipelines, save everything, and add it to the history list.

        If a previous click already got a good frame from one camera but not the other, that held
        frame (self._pending_face_frame / self._pending_doc_frame) is reused instead of recaptured
        -- this click only needs to (re)try whichever side is still missing.
        """
        logger.info("Operator triggered capture & process")
        # np.ndarray has no usable truthiness (a plain `or` raises on a multi-element array), so
        # each pending frame is checked explicitly against None rather than short-circuited.
        face_frame = (
            self._pending_face_frame
            if self._pending_face_frame is not None
            else capture_still_frame(self.camera_manager.face_capture)
        )
        doc_frame = (
            self._pending_doc_frame
            if self._pending_doc_frame is not None
            else capture_still_frame(self.camera_manager.doc_capture)
        )

        if face_frame is None or doc_frame is None:
            self._pending_face_frame = face_frame
            self._pending_doc_frame = doc_frame
            self.discard_pending_button.setVisible(face_frame is not None or doc_frame is not None)

            failed_cameras = []
            if face_frame is None:
                failed_cameras.append("visitor (left)")
            if doc_frame is None:
                failed_cameras.append("identification (right)")
            failed_description = " and ".join(failed_cameras)
            logger.warning("Capture failed: could not read a frame from the %s camera", failed_description)

            message = f"Could not read a frame from the {failed_description} camera."
            if face_frame is not None or doc_frame is not None:
                message += (
                    " The other camera's capture was held onto -- click Capture & Process again to "
                    "retry just the failed one, or Discard held capture to start over."
                )
            QMessageBox.warning(self, "Capture failed", message)
            return

        self._pending_face_frame = None
        self._pending_doc_frame = None
        self.discard_pending_button.setVisible(False)

        doc_image_quality = assess_image_quality(doc_frame)
        if doc_image_quality.has_warning:
            logger.info(
                "Document capture flagged low quality (%s): blur_score=%.1f glare_ratio=%.2f",
                doc_image_quality.warning_text,
                doc_image_quality.blur_score,
                doc_image_quality.glare_ratio,
            )

        live_embedding = get_best_face_embedding(detect_faces(self.face_analysis_app, face_frame))
        doc_embedding = get_best_face_embedding(detect_faces(self.face_analysis_app, doc_frame))

        match_result = None
        if live_embedding is not None and doc_embedding is not None:
            threshold = get_face_match_threshold(self.conn)
            match_result = match_faces(live_embedding, doc_embedding, threshold)

        ocr_text = extract_text(doc_frame)
        mrz_result = parse_and_validate_mrz(ocr_text)
        barcode_texts = decode_pdf417_barcodes(doc_frame)
        doc_classification = classify_document(mrz_result, barcode_texts, ocr_text)
        crosscheck_result = determine_validation_status(mrz_result, barcode_texts, ocr_text)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        face_image_path = FACE_IMAGE_DIR / f"{timestamp}_face.jpg"
        doc_image_path = DOC_IMAGE_DIR / f"{timestamp}_doc.jpg"
        cv2.imwrite(str(face_image_path), face_frame)
        cv2.imwrite(str(doc_image_path), doc_frame)

        if mrz_result.found:
            doc_fields = mrz_result.fields
        else:
            # Driving licences carry no MRZ, so name/birth-date have to come from a heuristic OCR
            # read of the licence's EU-harmonized numbered fields instead (see dl_field_parser.py).
            dl_fields_result = parse_driving_licence_fields(ocr_text)
            doc_fields = dl_fields_result.fields if dl_fields_result.found else {}
        doc_fields_json = json.dumps(doc_fields)

        person_flags = check_person_flags(doc_fields)
        if person_flags.is_minor or person_flags.is_expired:
            logger.info(
                "Scan flagged: is_minor=%s (age=%s) is_expired=%s (expiry=%s)",
                person_flags.is_minor,
                person_flags.age_years,
                person_flags.is_expired,
                person_flags.expiry_date_display,
            )

        validation_detail_json = json.dumps(
            {
                "notes": crosscheck_result.notes + person_flags.notes,
                "field_comparisons": [asdict(comparison) for comparison in crosscheck_result.field_comparisons],
                "classified_from": doc_classification.classified_from,
                "image_quality": {"has_warning": doc_image_quality.has_warning, "text": doc_image_quality.warning_text},
                "person_flags": asdict(person_flags),
            }
        )

        scan_id = insert_scan(
            self.conn,
            operator=None,
            face_image_path=str(face_image_path),
            doc_image_path=str(doc_image_path),
            face_embedding=live_embedding.tobytes() if live_embedding is not None else None,
            match_score=match_result.similarity_score if match_result is not None else None,
            match_result=_final_match_result_label(match_result.is_match if match_result else None, "Automatic"),
            doc_type=doc_classification.doc_type,
            doc_country=doc_classification.doc_country,
            doc_fields_json=doc_fields_json,
            validation_status=crosscheck_result.validation_status,
            validation_detail_json=validation_detail_json,
        )

        logger.info(
            "Scan %d saved: match_result=%s, validation_status=%s, doc_type=%s",
            scan_id,
            _final_match_result_label(match_result.is_match if match_result else None, "Automatic"),
            crosscheck_result.validation_status,
            doc_classification.doc_type,
        )

        self._add_history_row(get_scan_by_id(self.conn, scan_id))
        self._apply_filters()
        self._refresh_status_bar_stats()

    def _on_discard_pending_capture_clicked(self) -> None:
        """Drop a held partial capture (one camera's frame from a previous failed click) so the next click starts over from scratch on both sides."""
        self._pending_face_frame = None
        self._pending_doc_frame = None
        self.discard_pending_button.setVisible(False)
        logger.info("Operator discarded a held partial capture")

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """Remember the operator's manually-chosen top-pane height whenever they drag the splitter handle.

        Without this, the next resizeEvent would snap the top pane straight back to its original
        learned height, undoing the drag.
        """
        self._top_pane_height = self.top_bottom_splitter.sizes()[0]

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """Keep the top pane (previews + capture button) at a constant height across a plain window resize.

        The bottom pane (headline + scan list + trailing stretch) absorbs the entire resize delta
        instead. The top pane's height still changes freely when the operator drags the splitter
        handle -- _on_splitter_moved keeps self._top_pane_height in sync with that.
        """
        super().resizeEvent(event)
        splitter = getattr(self, "top_bottom_splitter", None)
        if splitter is None:
            return
        if self._top_pane_height is None:
            self._top_pane_height = self._top_pane_widget.sizeHint().height()
        bottom_height = max(splitter.height() - splitter.handleWidth() - self._top_pane_height, 0)
        splitter.setSizes([self._top_pane_height, bottom_height])

    def _restore_window_state(self) -> None:
        """Restore the window's size/position and the splitter's pane split from the last session, if anything was saved.

        The top pane's height is restored as a plain pixel integer, not via QSplitter's own
        saveState()/restoreState(): this method runs before the window has ever been shown, and
        QSplitter.sizes() reports a proportionally-scaled placeholder (not the real saved value)
        when queried before the splitter has had a real layout pass -- reading that back into
        self._top_pane_height would have resizeEvent() permanently lock onto the wrong number the
        moment the window is actually shown. Storing/restoring the height directly sidesteps that
        entirely: resizeEvent() already knows how to apply self._top_pane_height once real
        geometry exists, whether it came from a user drag, a first-run default, or here.

        Falls back to the existing content-driven default sizing (adjustSize() plus the top
        pane's natural sizeHint, learned lazily by resizeEvent) the first time the app is run,
        before anything has ever been saved.
        """
        geometry_b64 = get_setting_or_none(self.conn, _WINDOW_GEOMETRY_SETTING_KEY)
        if geometry_b64 is not None:
            self.restoreGeometry(QByteArray(base64.b64decode(geometry_b64)))
        else:
            self.adjustSize()

        top_pane_height_str = get_setting_or_none(self.conn, _TOP_PANE_HEIGHT_SETTING_KEY)
        if top_pane_height_str is not None:
            self._top_pane_height = int(top_pane_height_str)

    def _save_window_state(self) -> None:
        """Persist the window's current size/position and the splitter's top-pane height, so the next launch can restore them."""
        geometry_b64 = base64.b64encode(bytes(self.saveGeometry())).decode("ascii")
        set_setting(self.conn, _WINDOW_GEOMETRY_SETTING_KEY, geometry_b64)

        if self._top_pane_height is not None:
            set_setting(self.conn, _TOP_PANE_HEIGHT_SETTING_KEY, str(self._top_pane_height))

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Save the window/splitter layout, close the settings window if open, then release both camera handles, when the window closes."""
        self._save_window_state()
        if self._settings_window is not None:
            self._settings_window.close()
        self.camera_manager.release_all()
        super().closeEvent(event)
