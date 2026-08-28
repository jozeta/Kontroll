"""A splash screen shown while the app's slow startup work (prerequisite check, model loading, camera opening) runs."""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import QLabel, QSplashScreen

from config.version import APP_VERSION
from ui.app_icon import render_app_logo

_SPLASH_SIZE = (420, 280)
_BACKGROUND_COLOR = QColor(30, 30, 40)


def _build_background_pixmap() -> QPixmap:
    """Draw just the logo onto a solid background -- the title/version text is added as QLabel children instead (see build_splash_screen), not painted onto this pixmap.

    Vector shapes (QPainter polygons, as ui/app_icon.py draws) need no font, but text drawn
    directly via QPainter.drawText onto a bare QPixmap depends on whatever font the current
    platform happens to have available at that call site -- a real risk on a fresh/minimal
    machine. Routing the text through ordinary QLabel widgets instead uses the exact same
    font-resolution path every other label in this app already relies on (see e.g.
    ui/main_window.py's headline label), which is the safer, already-proven-to-render choice.
    """
    width, height = _SPLASH_SIZE
    pixmap = QPixmap(width, height)
    pixmap.fill(_BACKGROUND_COLOR)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    logo_size = 96
    painter.drawPixmap((width - logo_size) // 2, 36, render_app_logo(logo_size))
    painter.end()

    return pixmap


def build_splash_screen() -> QSplashScreen:
    """Build the app's splash screen, ready to .show() -- callers drive its status line via .showMessage() and dismiss it via .finish(window) once the main window is ready."""
    width, height = _SPLASH_SIZE
    splash = QSplashScreen(_build_background_pixmap())
    splash.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)

    title_label = QLabel("Kontroll", splash)
    title_label.setStyleSheet("color: white; font-size: 20pt; font-weight: bold;")
    title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    title_label.setGeometry(0, 150, width, 40)
    title_label.show()

    version_label = QLabel(f"Version {APP_VERSION}", splash)
    version_label.setStyleSheet("color: #BEBEC8; font-size: 10pt;")
    version_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    version_label.setGeometry(0, 190, width, 24)
    version_label.show()

    return splash
