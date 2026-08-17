"""Single-screen PyQt6 window: two live camera previews, capture buttons, and the match/validation results panel."""

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QCloseEvent, QImage, QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import CameraManager
from capture.frame_capture import capture_still_frame
from config.settings import get_face_match_threshold
from document.barcode_parser import decode_pdf417_barcodes
from document.doc_classifier import DocumentClassification, classify_document
from document.field_crosscheck import CrosscheckResult, determine_validation_status
from document.mrz_parser import MrzResult, parse_and_validate_mrz
from document.ocr import extract_text
from face.detector import detect_faces
from face.embedder import get_best_face_embedding
from face.matcher import MatchResult, match_faces
from storage.models import insert_scan

FACE_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "faces"
DOC_IMAGE_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"

_PREVIEW_INTERVAL_MS = 33  # ~30fps live preview polling


def _bgr_frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame into a QPixmap suitable for display in a QLabel."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    # .copy() detaches the QImage from rgb_frame's buffer, which is a stack-local that would
    # otherwise be freed once this function returns.
    qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


class MainWindow(QMainWindow):
    """The app's single combined screen: live previews, manual capture buttons, and face/document results."""

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

        # State captured so far in the current scan; cleared by _reset_for_next_scan after a save
        self.face_frame: np.ndarray | None = None
        self.face_embedding: np.ndarray | None = None
        self.match_result: MatchResult | None = None
        self.doc_frame: np.ndarray | None = None
        self.doc_classification: DocumentClassification | None = None
        self.crosscheck_result: CrosscheckResult | None = None
        self.mrz_result: MrzResult | None = None

        self.setWindowTitle("ID Verification")
        self._build_layout()

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_previews)
        self.preview_timer.start(_PREVIEW_INTERVAL_MS)

    def _build_layout(self) -> None:
        """Construct and arrange every widget on the single combined screen."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)

        # Operator name, entered once per scan and stored with the saved scan record
        operator_row = QHBoxLayout()
        operator_row.addWidget(QLabel("Operator:"))
        self.operator_field = QLineEdit()
        operator_row.addWidget(self.operator_field)
        outer_layout.addLayout(operator_row)

        # Camera previews and their capture buttons, side by side
        previews_row = QHBoxLayout()
        previews_row.addLayout(self._build_face_capture_column())
        previews_row.addLayout(self._build_doc_capture_column())
        outer_layout.addLayout(previews_row)

        # Face-match and document-validation results, side by side
        results_row = QHBoxLayout()
        results_row.addWidget(self._build_face_results_group())
        results_row.addWidget(self._build_doc_results_group())
        outer_layout.addLayout(results_row)

        self.save_button = QPushButton("Save Scan")
        self.save_button.clicked.connect(self._on_save_scan_clicked)
        outer_layout.addWidget(self.save_button)

    def _build_face_capture_column(self) -> QVBoxLayout:
        """Build the face camera's live preview label and its capture button."""
        column = QVBoxLayout()
        column.addWidget(QLabel("Face Camera"))
        self.face_preview_label = QLabel("No camera signal")
        self.face_preview_label.setFixedSize(320, 240)
        column.addWidget(self.face_preview_label)
        self.capture_face_button = QPushButton("Capture Face")
        self.capture_face_button.clicked.connect(self._on_capture_face_clicked)
        column.addWidget(self.capture_face_button)
        return column

    def _build_doc_capture_column(self) -> QVBoxLayout:
        """Build the document camera's live preview label and its capture button."""
        column = QVBoxLayout()
        column.addWidget(QLabel("Document Camera"))
        self.doc_preview_label = QLabel("No camera signal")
        self.doc_preview_label.setFixedSize(320, 240)
        column.addWidget(self.doc_preview_label)
        self.capture_doc_button = QPushButton("Capture Document")
        self.capture_doc_button.clicked.connect(self._on_capture_doc_clicked)
        column.addWidget(self.capture_doc_button)
        return column

    def _build_face_results_group(self) -> QGroupBox:
        """Build the face-match results panel: automated badge, raw score, and manual override control."""
        group = QGroupBox("Face Match")
        layout = QVBoxLayout(group)
        self.face_match_badge = QLabel("Not yet captured")
        layout.addWidget(self.face_match_badge)
        self.face_match_score_label = QLabel("Score: -")
        layout.addWidget(self.face_match_score_label)
        layout.addWidget(QLabel("Operator override:"))
        self.face_override_combo = QComboBox()
        self.face_override_combo.addItems(["Automatic", "Override: Match", "Override: No Match"])
        self.face_override_combo.currentTextChanged.connect(self._update_face_badge)
        layout.addWidget(self.face_override_combo)
        return group

    def _build_doc_results_group(self) -> QGroupBox:
        """Build the document validation results panel: status badge, type/country, and an expandable detail view."""
        group = QGroupBox("Document Validation")
        layout = QVBoxLayout(group)
        self.doc_status_badge = QLabel("Not yet captured")
        layout.addWidget(self.doc_status_badge)
        self.doc_type_label = QLabel("Type: -")
        layout.addWidget(self.doc_type_label)

        self.doc_detail_toggle = QPushButton("Show Details")
        self.doc_detail_toggle.setCheckable(True)
        self.doc_detail_toggle.toggled.connect(self._on_detail_toggle)
        layout.addWidget(self.doc_detail_toggle)

        self.doc_detail_text = QTextEdit()
        self.doc_detail_text.setReadOnly(True)
        self.doc_detail_text.setVisible(False)
        layout.addWidget(self.doc_detail_text)
        return group

    def _update_previews(self) -> None:
        """Poll both cameras for their latest frame and refresh the live preview labels. Runs on a timer, independent of capture."""
        face_frame = self.camera_manager.read_face_preview_frame()
        if face_frame is not None:
            self.face_preview_label.setPixmap(_bgr_frame_to_pixmap(face_frame))

        doc_frame = self.camera_manager.read_doc_preview_frame()
        if doc_frame is not None:
            self.doc_preview_label.setPixmap(_bgr_frame_to_pixmap(doc_frame))

    def _on_capture_face_clicked(self) -> None:
        """Grab a still frame from the face camera, run detection/embedding on it, and re-run the match if a document is already captured."""
        frame = capture_still_frame(self.camera_manager.face_capture)
        if frame is None:
            QMessageBox.warning(self, "Capture failed", "Could not read a frame from the face camera.")
            return

        self.face_frame = frame
        faces = detect_faces(self.face_analysis_app, frame)
        self.face_embedding = get_best_face_embedding(faces)

        if self.face_embedding is None:
            self.match_result = None
            self.face_match_badge.setText("No face detected")
            self.face_match_score_label.setText("Score: -")
            return

        self._run_face_match_if_ready()

    def _on_capture_doc_clicked(self) -> None:
        """Grab a still frame from the document camera and run the full document pipeline: OCR, MRZ, barcode, classification, and cross-check."""
        frame = capture_still_frame(self.camera_manager.doc_capture)
        if frame is None:
            QMessageBox.warning(self, "Capture failed", "Could not read a frame from the document camera.")
            return

        self.doc_frame = frame
        ocr_text = extract_text(frame)
        self.mrz_result = parse_and_validate_mrz(ocr_text)
        barcode_texts = decode_pdf417_barcodes(frame)

        self.doc_classification = classify_document(self.mrz_result, barcode_texts, ocr_text)
        self.crosscheck_result = determine_validation_status(self.mrz_result, barcode_texts, ocr_text)
        self._update_doc_results_display()

        # The document camera captures the whole ID document, so the printed photo's face is
        # found by running detection on this same frame, not on a separately captured image.
        self._run_face_match_if_ready()

    def _run_face_match_if_ready(self) -> None:
        """Run the face-match comparison once both a live face capture and a document capture are available."""
        if self.face_embedding is None or self.doc_frame is None:
            return

        doc_faces = detect_faces(self.face_analysis_app, self.doc_frame)
        doc_face_embedding = get_best_face_embedding(doc_faces)
        if doc_face_embedding is None:
            self.match_result = None
            self.face_match_badge.setText("No face detected on document")
            self.face_match_score_label.setText("Score: -")
            return

        threshold = get_face_match_threshold(self.conn)
        self.match_result = match_faces(self.face_embedding, doc_face_embedding, threshold)
        self.face_match_score_label.setText(f"Score: {self.match_result.similarity_score:.4f}")
        self._update_face_badge()

    def _update_face_badge(self) -> None:
        """Refresh the face-match badge to reflect either the automated result or the operator's manual override."""
        if self.match_result is None:
            self.face_match_badge.setText("Not yet matched")
            return

        override = self.face_override_combo.currentText()
        if override == "Override: Match":
            self.face_match_badge.setText("MATCH (operator override)")
        elif override == "Override: No Match":
            self.face_match_badge.setText("NO MATCH (operator override)")
        elif self.match_result.is_match:
            self.face_match_badge.setText("MATCH (automatic)")
        else:
            self.face_match_badge.setText("NO MATCH (automatic)")

    def _update_doc_results_display(self) -> None:
        """Refresh the document validation badge, type/country labels, and the expandable per-field detail text."""
        self.doc_status_badge.setText(f"Status: {self.crosscheck_result.validation_status.upper()}")
        country = self.doc_classification.doc_country or "unknown country"
        self.doc_type_label.setText(f"Type: {self.doc_classification.doc_type} ({country})")

        detail_lines = [f"Classified from: {self.doc_classification.classified_from}"]
        detail_lines.extend(self.crosscheck_result.notes)
        for comparison in self.crosscheck_result.field_comparisons:
            status = "OK" if comparison.found_in_ocr else "MISMATCH"
            detail_lines.append(f"[{status}] {comparison.field_name}: {comparison.machine_read_value}")
        self.doc_detail_text.setPlainText("\n".join(detail_lines))

    def _on_detail_toggle(self, checked: bool) -> None:
        """Show or hide the document validation detail text when the operator clicks the toggle button."""
        self.doc_detail_text.setVisible(checked)
        self.doc_detail_toggle.setText("Hide Details" if checked else "Show Details")

    def _final_match_result_label(self) -> str:
        """Compute the match_result string to persist, combining the automated result with any operator override."""
        override = self.face_override_combo.currentText()
        if override == "Override: Match":
            return "override_match"
        if override == "Override: No Match":
            return "override_no_match"
        if self.match_result is not None and self.match_result.is_match:
            return "match"
        return "no_match"

    def _on_save_scan_clicked(self) -> None:
        """Persist the current scan (both images, embedding, match result, and document validation result) to the database."""
        if self.face_frame is None or self.doc_frame is None:
            QMessageBox.warning(self, "Incomplete scan", "Capture both a face and a document before saving.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        face_image_path = FACE_IMAGE_DIR / f"{timestamp}_face.jpg"
        doc_image_path = DOC_IMAGE_DIR / f"{timestamp}_doc.jpg"
        cv2.imwrite(str(face_image_path), self.face_frame)
        cv2.imwrite(str(doc_image_path), self.doc_frame)

        doc_fields_json = json.dumps(self.mrz_result.fields) if self.mrz_result.found else "{}"
        validation_detail_json = json.dumps(
            {
                "notes": self.crosscheck_result.notes,
                "field_comparisons": [asdict(comparison) for comparison in self.crosscheck_result.field_comparisons],
                "classified_from": self.doc_classification.classified_from,
            }
        )

        insert_scan(
            self.conn,
            operator=self.operator_field.text() or None,
            face_image_path=str(face_image_path),
            doc_image_path=str(doc_image_path),
            face_embedding=self.face_embedding.tobytes() if self.face_embedding is not None else None,
            match_score=self.match_result.similarity_score if self.match_result is not None else None,
            match_result=self._final_match_result_label(),
            doc_type=self.doc_classification.doc_type,
            doc_country=self.doc_classification.doc_country,
            doc_fields_json=doc_fields_json,
            validation_status=self.crosscheck_result.validation_status,
            validation_detail_json=validation_detail_json,
        )

        QMessageBox.information(self, "Scan saved", "The scan was saved successfully.")
        self._reset_for_next_scan()

    def _reset_for_next_scan(self) -> None:
        """Clear all captured-scan state and results widgets so the window is ready for the next operator-triggered scan."""
        self.face_frame = None
        self.face_embedding = None
        self.match_result = None
        self.doc_frame = None
        self.doc_classification = None
        self.crosscheck_result = None
        self.mrz_result = None

        self.face_match_badge.setText("Not yet captured")
        self.face_match_score_label.setText("Score: -")
        self.face_override_combo.setCurrentIndex(0)
        self.doc_status_badge.setText("Not yet captured")
        self.doc_type_label.setText("Type: -")
        self.doc_detail_text.clear()

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Release both camera handles when the window closes."""
        self.camera_manager.release_all()
        super().closeEvent(event)
