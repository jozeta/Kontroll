"""Shared "click a thumbnail, see it full-size" viewer, used by both the scan history list
(ui/main_window.py) and the flagged-persons list (ui/settings_window.py) -- kept in its own module
(rather than defined once in main_window.py and imported from there) since main_window.py already
imports SettingsWindow from ui/settings_window.py, and a reverse import back into main_window.py
would be circular.
"""

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QResizeEvent
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget


class ImageViewerWindow(QWidget):
    """A standalone, freely-resizable window showing one image at a much larger size.

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


class ClickableThumbnailLabel(QLabel):
    """A thumbnail QLabel that opens the underlying image full-size in a resizable ImageViewerWindow on double-click."""

    def __init__(self, image_path: str | None, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._title = title
        # Kept alive by this reference for as long as the thumbnail itself lives -- a top-level
        # PyQt widget with no Python reference held anywhere is garbage-collected out from under
        # its own open window.
        self._viewer: ImageViewerWindow | None = None

    def mouseDoubleClickEvent(self, event) -> None:
        if self._image_path and Path(self._image_path).exists():
            self._viewer = ImageViewerWindow(self._image_path, self._title)
            self._viewer.show()
        super().mouseDoubleClickEvent(event)
