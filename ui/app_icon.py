"""Draws the app's own logo (a shield with a checkmark, evoking identity verification) as a QPixmap.

Drawn with QPainter rather than loaded from an image file, the same reasoning as
ui/flag_icons.py's curated flags: no bundled asset file to keep track of or lose across a
PyInstaller packaging step, and it stays crisp at any requested size since it's vector-drawn.
"""

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QPolygonF

_SHIELD_COLOR = QColor("#0078D7")
_SHIELD_BORDER_COLOR = QColor("#004C8C")
_CHECKMARK_COLOR = QColor("#FFFFFF")


def render_app_logo(size: int) -> QPixmap:
    """Render the app logo as a square QPixmap of the given pixel size (transparent background)."""
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    margin = size * 0.08
    shield_path = QPolygonF(
        [
            QPointF(size / 2, margin),
            QPointF(size - margin, size * 0.28),
            QPointF(size - margin, size * 0.58),
            QPointF(size / 2, size - margin),
            QPointF(margin, size * 0.58),
            QPointF(margin, size * 0.28),
        ]
    )
    painter.setPen(QPen(_SHIELD_BORDER_COLOR, size * 0.03))
    painter.setBrush(_SHIELD_COLOR)
    painter.drawPolygon(shield_path)

    checkmark_pen = QPen(_CHECKMARK_COLOR, size * 0.09)
    checkmark_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    checkmark_pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(checkmark_pen)
    painter.drawPolyline(
        QPolygonF(
            [
                QPointF(size * 0.32, size * 0.50),
                QPointF(size * 0.45, size * 0.63),
                QPointF(size * 0.70, size * 0.35),
            ]
        )
    )

    painter.end()
    return pixmap
