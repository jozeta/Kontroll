"""Shared window/dialog size-and-position persistence.

ui/main_window.py::MainWindow and ui/settings_window.py::SettingsWindow each already implement
this same pattern independently (QWidget.saveGeometry()/restoreGeometry(), base64-encoded through
a `settings` table key, saved on closeEvent) -- left as-is here rather than refactored, since
they already work and touching them isn't worth the risk. This module exists so every window/
dialog added *after* those two (ImageViewerWindow, ImageEditDialog, CalibrationWindow,
BarcodeCalibrationWindow, and the newer comparison/edit dialogs) doesn't reimplement it a third,
fourth, fifth... time.

For a QDialog, call save_geometry from a `finished` signal handler, not closeEvent: QDialog.accept()/
reject() call done() -> hide() directly, which never raises a QCloseEvent, so a closeEvent override
alone misses every normal Accept/Cancel-button close and only catches the OS window-close button
(QDialog's own closeEvent handler happens to route that one through reject() -> done()). `finished`
fires for all three (accept, reject, and the OS close button), so it's the one hook that's never
missed. For a plain QWidget window (no accept()/reject() distinction), closeEvent is fine and is
what MainWindow/SettingsWindow already use.
"""

import base64
import sqlite3

from PyQt6.QtCore import QByteArray
from PyQt6.QtWidgets import QWidget

from config.settings import get_setting_or_none, set_setting


def restore_geometry(conn: sqlite3.Connection, widget: QWidget, setting_key: str) -> None:
    """Restore widget's size/position from setting_key, if anything was saved for it previously -- otherwise leaves whatever size/position the caller already set (e.g. a first-run default resize())."""
    geometry_b64 = get_setting_or_none(conn, setting_key)
    if geometry_b64 is not None:
        widget.restoreGeometry(QByteArray(base64.b64decode(geometry_b64)))


def save_geometry(conn: sqlite3.Connection, widget: QWidget, setting_key: str) -> None:
    """Persist widget's current size/position under setting_key, for a later restore_geometry call to restore."""
    geometry_b64 = base64.b64encode(bytes(widget.saveGeometry())).decode("ascii")
    set_setting(conn, setting_key, geometry_b64)
