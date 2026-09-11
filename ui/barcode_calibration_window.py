"""A standalone window for measuring the barcode/OCR word-overlap ratio distribution on real
document captures, so the barcode_overlap_threshold (Settings) can be set from real desk
conditions (this camera, this lighting, these document types) instead of the shipped default.

Mirrors ui/calibration_window.py's face-match calibration flow, but there is one important
difference in what counts as a "known-good" sample: a face-match pair needs the operator to
personally confirm the live face and the document photo really are the same person, whereas here
the OCR text and the barcode payload are read off the *same physical document* in the same
capture -- there's no separate ground truth to confirm, so every successfully barcode-decoded
capture is a valid calibration sample, not just specially arranged ones. This tool is only useful
for documents that carry a PDF417 barcode without an MRZ (see document/field_crosscheck.py --
mainly EU driving licences); a capture with no barcode simply can't contribute a sample.

Deliberately keeps its captured ratios in memory only, for the duration of one calibration
session -- calibration captures are not real scans and are never written to the scans table or
saved to disk.
"""

import logging
import sqlite3
import statistics
from typing import Callable

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import CameraManager
from document.barcode_parser import decode_pdf417_barcodes
from document.card_detector import detect_and_crop_card
from document.field_crosscheck import compute_barcode_ocr_overlap_ratio
from document.ocr import extract_text
from ui.window_geometry import restore_geometry, save_geometry

logger = logging.getLogger(__name__)

_WINDOW_GEOMETRY_SETTING_KEY = "barcode_calibration_window_geometry"

_HISTOGRAM_BUCKET_COUNT = 20

# A suggestion only becomes meaningful once there are a few samples to work from -- below this,
# a single unusually clean (or noisy) OCR read would swing the suggested value wildly.
_MIN_SAMPLES_FOR_SUGGESTION = 3

# The suggested threshold sits this far below the lowest observed genuine-document ratio, as a
# small safety margin against the very sample that set that floor -- a plain heuristic, not a
# statistically derived value; the operator is always shown the full distribution alongside it and
# free to pick their own value in Settings instead.
_SUGGESTION_MARGIN = 0.05


class _RatioHistogramWidget(QWidget):
    """Draws a simple bar histogram of collected overlap ratios across the fixed 0.0-1.0 range."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ratios: list[float] = []
        self.setMinimumHeight(120)

    def set_ratios(self, ratios: list[float]) -> None:
        self._ratios = ratios
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1E1E1E"))

        if not self._ratios:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No captures yet")
            painter.end()
            return

        bucket_width = 1.0 / _HISTOGRAM_BUCKET_COUNT
        buckets = [0] * _HISTOGRAM_BUCKET_COUNT
        for ratio in self._ratios:
            bucket_index = min(max(int(ratio / bucket_width), 0), _HISTOGRAM_BUCKET_COUNT - 1)
            buckets[bucket_index] += 1
        max_count = max(buckets)

        bar_width = self.width() / _HISTOGRAM_BUCKET_COUNT
        for index, count in enumerate(buckets):
            bar_height = (count / max_count) * (self.height() - 4)
            x = index * bar_width
            y = self.height() - bar_height
            painter.fillRect(int(x) + 1, int(y), max(int(bar_width) - 2, 1), int(bar_height), QColor("#FFB74D"))

        painter.setPen(QPen(QColor("#888888")))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.end()


class BarcodeCalibrationWindow(QWidget):
    """Lets an operator repeatedly capture a barcode-bearing document (driving licence/ID card)
    through the real document camera, showing the resulting OCR/barcode overlap-ratio distribution
    so barcode_overlap_threshold can be set from measured data rather than the unvalidated shipped
    default."""

    def __init__(
        self,
        camera_manager: CameraManager,
        on_apply_threshold: Callable[[float], None],
        conn: sqlite3.Connection | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.camera_manager = camera_manager
        self._on_apply_threshold = on_apply_threshold
        self._conn = conn
        self._ratios: list[float] = []

        self.setWindowTitle("Barcode/OCR overlap calibration")
        self.resize(480, 480)
        if conn is not None:
            restore_geometry(conn, self, _WINDOW_GEOMETRY_SETTING_KEY)
        self._build_layout()

    def closeEvent(self, event) -> None:
        if self._conn is not None:
            save_geometry(self._conn, self, _WINDOW_GEOMETRY_SETTING_KEY)
        super().closeEvent(event)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Capture a real barcode-bearing document (e.g. an EU driving licence or ID card) "
            "repeatedly, ideally several different documents, to measure this desk's typical "
            "OCR/barcode overlap ratio. These captures are never saved as scans. A capture with "
            "no MRZ and no decodable barcode won't produce a sample -- try a different document."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        capture_button = QPushButton("Capture document")
        capture_button.setMinimumHeight(40)
        capture_button.clicked.connect(self._on_capture_document_clicked)
        layout.addWidget(capture_button)

        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

        self.histogram = _RatioHistogramWidget()
        layout.addWidget(self.histogram)

        self.results_list = QListWidget()
        layout.addWidget(self.results_list, stretch=1)

        self.suggestion_label = QLabel()
        self.suggestion_label.setWordWrap(True)
        self.suggestion_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(self.suggestion_label)

        button_row = QHBoxLayout()
        self.apply_button = QPushButton("Apply suggested threshold")
        self.apply_button.setEnabled(False)
        self.apply_button.clicked.connect(self._on_apply_suggested_clicked)
        button_row.addWidget(self.apply_button)

        clear_button = QPushButton("Clear results")
        clear_button.clicked.connect(self._on_clear_clicked)
        button_row.addWidget(clear_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self._refresh_display()

    def _on_capture_document_clicked(self) -> None:
        doc_frame = self.camera_manager.get_doc_latest_frame()
        if doc_frame is None:
            QMessageBox.warning(self, "Capture failed", "Could not read a frame from the document camera. Try again.")
            return

        doc_analysis_frame = detect_and_crop_card(doc_frame)
        ocr_text = extract_text(doc_analysis_frame)
        barcode_texts = decode_pdf417_barcodes(doc_analysis_frame)
        ratio = compute_barcode_ocr_overlap_ratio(barcode_texts, ocr_text)
        if ratio is None:
            QMessageBox.warning(
                self,
                "No barcode found",
                "No PDF417 barcode was decoded in this capture -- try a driving licence or ID card that carries one.",
            )
            return

        self._ratios.append(ratio)
        logger.info("Barcode calibration capture #%d: overlap ratio %.4f", len(self._ratios), ratio)
        self._refresh_display()

    def _on_clear_clicked(self) -> None:
        self._ratios.clear()
        self._refresh_display()

    def _suggested_threshold(self) -> float | None:
        if len(self._ratios) < _MIN_SAMPLES_FOR_SUGGESTION:
            return None
        return max(0.0, min(self._ratios) - _SUGGESTION_MARGIN)

    def _refresh_display(self) -> None:
        self.results_list.clear()
        for index, ratio in enumerate(self._ratios, start=1):
            self.results_list.addItem(f"#{index}: {ratio:.4f}")

        self.histogram.set_ratios(self._ratios)

        if self._ratios:
            mean = statistics.mean(self._ratios)
            stats_text = (
                f"{len(self._ratios)} document(s) captured — "
                f"min {min(self._ratios):.4f}  max {max(self._ratios):.4f}  mean {mean:.4f}"
            )
            if len(self._ratios) > 1:
                stats_text += f"  stdev {statistics.stdev(self._ratios):.4f}"
        else:
            stats_text = "0 documents captured."
        self.stats_label.setText(stats_text)

        suggested = self._suggested_threshold()
        if suggested is None:
            remaining = _MIN_SAMPLES_FOR_SUGGESTION - len(self._ratios)
            self.suggestion_label.setText(f"Capture {remaining} more document(s) for a suggested threshold.")
            self.apply_button.setEnabled(False)
        else:
            self.suggestion_label.setText(
                f"Suggested threshold: {suggested:.4f} (lowest observed ratio minus a small margin)"
            )
            self.apply_button.setEnabled(True)

    def _on_apply_suggested_clicked(self) -> None:
        suggested = self._suggested_threshold()
        if suggested is None:
            return
        self._on_apply_threshold(suggested)
        QMessageBox.information(self, "Threshold applied", f"Barcode/OCR overlap threshold set to {suggested:.4f}.")
