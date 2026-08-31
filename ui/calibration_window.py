"""A standalone window for measuring the face-match similarity-score distribution on known-good
same-person pairs, so the face-match threshold (Settings) can be set from real desk conditions
(this machine's cameras, lighting, and expected document types) instead of the shipped default.

Deliberately keeps its captured scores in memory only, for the duration of one calibration
session -- calibration captures are not real scans and are never written to the scans table or
saved to disk, so they never pollute the operator's actual scan history.
"""

import logging
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
from insightface.app import FaceAnalysis

from capture.camera_manager import CameraManager
from config.settings import set_setting
from face.detector import detect_faces
from face.embedder import get_best_face_embedding
from face.matcher import compute_cosine_similarity

logger = logging.getLogger(__name__)

_HISTOGRAM_BUCKET_COUNT = 20

# A suggestion only becomes meaningful once there are a few samples to work from -- below this,
# a single unlucky (or lucky) capture would swing the suggested value wildly.
_MIN_SAMPLES_FOR_SUGGESTION = 3

# The suggested threshold sits this far below the lowest observed genuine-pair score, as a small
# safety margin against the very sample that set that floor -- a plain heuristic, not a
# statistically derived value; the operator is always shown the full distribution alongside it
# and free to pick their own value in Settings instead.
_SUGGESTION_MARGIN = 0.02


class _ScoreHistogramWidget(QWidget):
    """Draws a simple bar histogram of collected similarity scores across the fixed 0.0-1.0 score range."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scores: list[float] = []
        self.setMinimumHeight(120)

    def set_scores(self, scores: list[float]) -> None:
        self._scores = scores
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1E1E1E"))

        if not self._scores:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No captures yet")
            painter.end()
            return

        bucket_width = 1.0 / _HISTOGRAM_BUCKET_COUNT
        buckets = [0] * _HISTOGRAM_BUCKET_COUNT
        for score in self._scores:
            bucket_index = min(max(int(score / bucket_width), 0), _HISTOGRAM_BUCKET_COUNT - 1)
            buckets[bucket_index] += 1
        max_count = max(buckets)

        bar_width = self.width() / _HISTOGRAM_BUCKET_COUNT
        for index, count in enumerate(buckets):
            bar_height = (count / max_count) * (self.height() - 4)
            x = index * bar_width
            y = self.height() - bar_height
            painter.fillRect(int(x) + 1, int(y), max(int(bar_width) - 2, 1), int(bar_height), QColor("#4FC3F7"))

        painter.setPen(QPen(QColor("#888888")))
        painter.drawLine(0, self.height() - 1, self.width(), self.height() - 1)
        painter.end()


class CalibrationWindow(QWidget):
    """Lets an operator repeatedly capture known-good (same-person) live-face/document-photo pairs
    through the real cameras, showing the resulting similarity-score distribution so the face-match
    threshold can be set from measured data rather than the unvalidated shipped default."""

    def __init__(
        self,
        camera_manager: CameraManager,
        face_analysis_app: FaceAnalysis,
        on_apply_threshold: Callable[[float], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.camera_manager = camera_manager
        self.face_analysis_app = face_analysis_app
        self._on_apply_threshold = on_apply_threshold
        self._scores: list[float] = []

        self.setWindowTitle("Face match calibration")
        self.resize(480, 480)
        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        instructions = QLabel(
            "Capture the same known person's live face and document photo repeatedly (ideally "
            "several different people, several captures each) to measure this desk's typical "
            "genuine-match similarity scores. These captures are never saved as scans."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        capture_button = QPushButton("Capture known-good pair")
        capture_button.setMinimumHeight(40)
        capture_button.clicked.connect(self._on_capture_pair_clicked)
        layout.addWidget(capture_button)

        self.stats_label = QLabel()
        layout.addWidget(self.stats_label)

        self.histogram = _ScoreHistogramWidget()
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

    def _on_capture_pair_clicked(self) -> None:
        face_frame = self.camera_manager.get_face_latest_frame()
        doc_frame = self.camera_manager.get_doc_latest_frame()
        if face_frame is None or doc_frame is None:
            QMessageBox.warning(self, "Capture failed", "Could not read a frame from one or both cameras. Try again.")
            return

        live_embedding = get_best_face_embedding(detect_faces(self.face_analysis_app, face_frame))
        doc_embedding = get_best_face_embedding(detect_faces(self.face_analysis_app, doc_frame))
        if live_embedding is None or doc_embedding is None:
            QMessageBox.warning(self, "No face detected", "Could not detect a face in one or both captures. Try again.")
            return

        score = compute_cosine_similarity(live_embedding, doc_embedding)
        self._scores.append(score)
        logger.info("Calibration pair #%d captured: similarity score %.4f", len(self._scores), score)
        self._refresh_display()

    def _on_clear_clicked(self) -> None:
        self._scores.clear()
        self._refresh_display()

    def _suggested_threshold(self) -> float | None:
        if len(self._scores) < _MIN_SAMPLES_FOR_SUGGESTION:
            return None
        return max(0.0, min(self._scores) - _SUGGESTION_MARGIN)

    def _refresh_display(self) -> None:
        self.results_list.clear()
        for index, score in enumerate(self._scores, start=1):
            self.results_list.addItem(f"#{index}: {score:.4f}")

        self.histogram.set_scores(self._scores)

        if self._scores:
            mean = statistics.mean(self._scores)
            stats_text = (
                f"{len(self._scores)} pair(s) captured — "
                f"min {min(self._scores):.4f}  max {max(self._scores):.4f}  mean {mean:.4f}"
            )
            if len(self._scores) > 1:
                stats_text += f"  stdev {statistics.stdev(self._scores):.4f}"
        else:
            stats_text = "0 pairs captured."
        self.stats_label.setText(stats_text)

        suggested = self._suggested_threshold()
        if suggested is None:
            remaining = _MIN_SAMPLES_FOR_SUGGESTION - len(self._scores)
            self.suggestion_label.setText(f"Capture {remaining} more pair(s) for a suggested threshold.")
            self.apply_button.setEnabled(False)
        else:
            self.suggestion_label.setText(
                f"Suggested threshold: {suggested:.4f} (lowest observed score minus a small margin)"
            )
            self.apply_button.setEnabled(True)

    def _on_apply_suggested_clicked(self) -> None:
        suggested = self._suggested_threshold()
        if suggested is None:
            return
        self._on_apply_threshold(suggested)
        QMessageBox.information(self, "Threshold applied", f"Face match threshold set to {suggested:.4f}.")
