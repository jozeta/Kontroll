"""Single-screen PyQt6 window: two live camera previews, one capture button, and a scan history list."""

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import CameraManager
from capture.frame_capture import capture_still_frame
from config.settings import get_face_match_threshold
from document.barcode_parser import decode_pdf417_barcodes
from document.doc_classifier import classify_document
from document.field_crosscheck import determine_validation_status
from document.mrz_parser import parse_and_validate_mrz
from document.ocr import extract_text
from face.detector import detect_faces
from face.embedder import get_best_face_embedding
from face.matcher import match_faces
from storage.models import ScanRecord, get_scan_by_id, insert_scan, list_scans, update_match_result

FACE_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "faces"
DOC_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

_PREVIEW_INTERVAL_MS = 33  # ~30fps live preview polling
_PREVIEW_SIZE = (320, 240)
_THUMBNAIL_SIZE = (110, 82)

_OVERRIDE_CHOICES = ["Automatic", "Override: Match", "Override: No Match"]


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


def _infer_birth_year(two_digit_year: int) -> int:
    """Infer a full 4-digit birth year from an MRZ's 2-digit year, assuming the birth date must be in the past (not the future)."""
    current_two_digit_year = datetime.now().year % 100
    century = 2000 if two_digit_year <= current_two_digit_year else 1900
    return century + two_digit_year


def _parse_mrz_birth_date(birth_date_field: str) -> date | None:
    """Parse an MRZ birth_date field (YYMMDD) into a real date, inferring the century since the MRZ only encodes 2 digits."""
    if len(birth_date_field) != 6 or not birth_date_field.isdigit():
        return None
    year_2digit, month, day = int(birth_date_field[0:2]), int(birth_date_field[2:4]), int(birth_date_field[4:6])
    year = _infer_birth_year(year_2digit)
    try:
        return date(year, month, day)
    except ValueError:
        # A malformed/misread MRZ can decode to an impossible calendar date (e.g. month 19);
        # that's a data-quality problem to surface as "unknown", not a bug to crash on.
        return None


def _calculate_age(birth_date: date) -> int:
    """Calculate a person's age in whole years as of today, given their birth date."""
    today = date.today()
    age = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age


@dataclass
class DisplayFields:
    """Name/birth-date/age fields derived from a scan's stored MRZ data, for display in the history list."""

    first_name: str
    last_name: str
    birth_date_display: str
    age_display: str


def _derive_display_fields(doc_fields_json: str | None) -> DisplayFields:
    """Extract first name, last name, birth date, and computed age from a scan's stored MRZ fields, if any.

    Only documents with a successfully parsed MRZ carry structured name/birth-date fields in v1
    (see document/doc_classifier.py) -- barcode-only and OCR-only documents show '-' placeholders
    here rather than a guessed value, consistent with how validation_status handles the same gap.
    """
    if not doc_fields_json:
        return DisplayFields("-", "-", "-", "-")

    fields = json.loads(doc_fields_json)
    first_name = fields.get("name", "").strip() or "-"
    last_name = fields.get("surname", "").strip() or "-"

    birth_date = _parse_mrz_birth_date(fields.get("birth_date", ""))
    if birth_date is None:
        return DisplayFields(first_name, last_name, "-", "-")

    return DisplayFields(first_name, last_name, birth_date.isoformat(), str(_calculate_age(birth_date)))


def _final_match_result_label(automatic_is_match: bool | None, override_choice: str) -> str:
    """Compute the match_result string to persist, combining the automated result with any operator override choice."""
    if override_choice == "Override: Match":
        return "override_match"
    if override_choice == "Override: No Match":
        return "override_no_match"
    if automatic_is_match:
        return "match"
    return "no_match"


class HistoryRowWidget(QFrame):
    """One row in the scan history list: both thumbnails, extracted identity fields, the match icon, and an expandable detail/override section."""

    def __init__(self, conn: sqlite3.Connection, scan: ScanRecord, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.conn = conn
        self.scan = scan
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self._build_layout()

    def _build_layout(self) -> None:
        """Build the always-visible summary row and the hidden-by-default detail section beneath it."""
        outer_layout = QVBoxLayout(self)

        summary_row = QHBoxLayout()
        summary_row.addWidget(self._build_thumbnail(self.scan.face_image_path))
        summary_row.addWidget(self._build_thumbnail(self.scan.doc_image_path))

        display_fields = _derive_display_fields(self.scan.doc_fields_json)
        identity_label = QLabel(
            f"{display_fields.first_name} {display_fields.last_name}\n"
            f"DOB: {display_fields.birth_date_display}   Age: {display_fields.age_display}\n"
            f"{self.scan.created_at}"
        )
        summary_row.addWidget(identity_label, stretch=1)

        self.match_icon_label = QLabel(_match_icon_text(self.scan.match_result))
        self.match_icon_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        summary_row.addWidget(self.match_icon_label)

        self.status_label = QLabel(f"Doc: {(self.scan.validation_status or '-').upper()}")
        summary_row.addWidget(self.status_label)

        self.detail_toggle = QPushButton("Details ▾")
        self.detail_toggle.setCheckable(True)
        self.detail_toggle.toggled.connect(self._on_detail_toggle)
        summary_row.addWidget(self.detail_toggle)

        outer_layout.addLayout(summary_row)
        outer_layout.addWidget(self._build_detail_section())

    def _build_thumbnail(self, image_path: str | None) -> QLabel:
        """Build a fixed-size thumbnail label for one of the row's two saved images."""
        label = QLabel()
        label.setFixedSize(*_THUMBNAIL_SIZE)
        pixmap = _load_thumbnail(image_path)
        if pixmap is not None:
            label.setPixmap(pixmap)
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
        self.scan.match_result = new_label
        self.match_icon_label.setText(_match_icon_text(new_label))


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

        self.setWindowTitle("ID Verification")
        self._build_layout()
        self._load_history()

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_previews)
        self.preview_timer.start(_PREVIEW_INTERVAL_MS)

    def _build_layout(self) -> None:
        """Construct and arrange every widget on the single combined screen."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)

        operator_row = QHBoxLayout()
        operator_row.addWidget(QLabel("Operator:"))
        self.operator_field = QLineEdit()
        operator_row.addWidget(self.operator_field)
        outer_layout.addLayout(operator_row)

        previews_row = QHBoxLayout()
        self.face_preview_label = self._build_preview_label()
        self.doc_preview_label = self._build_preview_label()
        previews_row.addWidget(self.face_preview_label)
        previews_row.addWidget(self.doc_preview_label)
        outer_layout.addLayout(previews_row)

        self.capture_button = QPushButton("Capture && Process")
        self.capture_button.clicked.connect(self._on_capture_and_process_clicked)
        outer_layout.addWidget(self.capture_button)

        outer_layout.addWidget(QLabel("Previous scans (newest first):"))

        self.history_scroll_area = QScrollArea()
        self.history_scroll_area.setWidgetResizable(True)
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.addStretch(1)  # keeps rows top-aligned as they're inserted above this
        self.history_scroll_area.setWidget(self.history_container)
        outer_layout.addWidget(self.history_scroll_area, stretch=1)

    def _build_preview_label(self) -> QLabel:
        """Build one fixed-size live preview label."""
        label = QLabel("No camera signal")
        label.setFixedSize(*_PREVIEW_SIZE)
        label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        label.setFrameShape(QFrame.Shape.Box)
        return label

    def _load_history(self) -> None:
        """Populate the history list with every previously saved scan, newest first.

        _add_history_row always inserts at the top, so the scans are fed in here oldest-first
        (the reverse of list_scans' own newest-first order) -- each insert-at-top then pushes
        the previous one down by exactly one, leaving the newest scan on top once the loop ends.
        """
        for scan in reversed(list_scans(self.conn)):
            self._add_history_row(scan)

    def _add_history_row(self, scan: ScanRecord) -> None:
        """Insert one scan as the top-most row of the history list."""
        row = HistoryRowWidget(self.conn, scan)
        self.history_layout.insertWidget(0, row)

    def _update_previews(self) -> None:
        """Poll both cameras for their latest frame and refresh the live preview labels. Runs on a timer, independent of capture."""
        face_frame = self.camera_manager.read_face_preview_frame()
        if face_frame is not None:
            self.face_preview_label.setPixmap(_bgr_frame_to_pixmap(face_frame).scaled(*_PREVIEW_SIZE))

        doc_frame = self.camera_manager.read_doc_preview_frame()
        if doc_frame is not None:
            self.doc_preview_label.setPixmap(_bgr_frame_to_pixmap(doc_frame).scaled(*_PREVIEW_SIZE))

    def _on_capture_and_process_clicked(self) -> None:
        """Grab a still frame from both cameras, run the full face-match and document-validation pipelines, save everything, and add it to the history list."""
        face_frame = capture_still_frame(self.camera_manager.face_capture)
        doc_frame = capture_still_frame(self.camera_manager.doc_capture)
        if face_frame is None or doc_frame is None:
            QMessageBox.warning(self, "Capture failed", "Could not read a frame from both cameras.")
            return

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

        doc_fields_json = json.dumps(mrz_result.fields) if mrz_result.found else "{}"
        validation_detail_json = json.dumps(
            {
                "notes": crosscheck_result.notes,
                "field_comparisons": [asdict(comparison) for comparison in crosscheck_result.field_comparisons],
                "classified_from": doc_classification.classified_from,
            }
        )

        scan_id = insert_scan(
            self.conn,
            operator=self.operator_field.text() or None,
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

        self._add_history_row(get_scan_by_id(self.conn, scan_id))

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Release both camera handles when the window closes."""
        self.camera_manager.release_all()
        super().closeEvent(event)
