"""Draws the app's own logo (a simple camera glyph, evoking the capture step at the heart of this
app) as a QPixmap.

Drawn with QPainter rather than loaded from an image file, the same reasoning as
ui/flag_icons.py's curated flags: no bundled asset file to keep track of or lose across a
PyInstaller packaging step, and it stays crisp at any requested size since it's vector-drawn. Used
both for the splash screen (ui/splash.py) and as the actual application/taskbar icon
(QApplication.setWindowIcon in main.py) -- the two previously showed different things (this logo
only ever appeared on the splash; nothing set a window/taskbar icon at all), now unified on one
drawing so the app has one consistent visual identity everywhere it shows up.
"""

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap

_BODY_COLOR = QColor("#37474F")
_BODY_BORDER_COLOR = QColor("#1C262B")
_LENS_OUTER_COLOR = QColor("#0078D7")
_LENS_INNER_COLOR = QColor("#FFFFFF")
_FLASH_COLOR = QColor("#FFFFFF")


def render_app_logo(size: int) -> QPixmap:
    """Render the app logo -- a simple camera body with a viewfinder hump and a lens -- as a square QPixmap of the given pixel size (transparent background)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    border_pen = QPen(_BODY_BORDER_COLOR, max(size * 0.02, 1.0))

    # Small raised viewfinder hump, top-left of the body -- drawn first so the body's own top edge
    # (drawn next) overlaps and hides its lower portion, reading as one continuous camera shape
    # rather than two separate boxes stacked on top of each other.
    hump_rect = QRectF(size * 0.22, size * 0.16, size * 0.30, size * 0.16)
    painter.setPen(border_pen)
    painter.setBrush(_BODY_COLOR)
    painter.drawRoundedRect(hump_rect, size * 0.03, size * 0.03)

    body_rect = QRectF(size * 0.08, size * 0.28, size * 0.84, size * 0.58)
    painter.drawRoundedRect(body_rect, size * 0.08, size * 0.08)

    # Small flash/shutter-button indicator, top-right of the body.
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_FLASH_COLOR)
    painter.drawEllipse(QRectF(size * 0.68, size * 0.35, size * 0.07, size * 0.07))

    # Lens: an outer ring plus a smaller inner "glass" circle.
    lens_center = QPointF(size / 2, size * 0.59)
    lens_outer_radius = size * 0.22
    painter.setPen(border_pen)
    painter.setBrush(_LENS_OUTER_COLOR)
    painter.drawEllipse(lens_center, lens_outer_radius, lens_outer_radius)

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_LENS_INNER_COLOR)
    painter.drawEllipse(lens_center, lens_outer_radius * 0.42, lens_outer_radius * 0.42)

    painter.end()
    return pixmap
