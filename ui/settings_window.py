"""The Settings window: manual scan purge, camera-to-side assignment, the capture keyboard
shortcut, and a live prerequisites checklist with per-item install instructions."""

import base64
import logging
import shutil
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
from insightface.app import FaceAnalysis
from PyQt6.QtCore import QByteArray, QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QIcon, QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import (
    AUTO_LOWEST,
    CAMERA_BACKENDS,
    CameraDiagnostics,
    CameraInfo,
    CameraManager,
    CameraStats,
    ExposureSettings,
    REQUESTED_FOURCC,
    SLOW_READ_WARN_THRESHOLD_SECONDS,
    enumerate_cameras,
    gather_camera_diagnostics,
    probe_supported_resolutions,
    resolutions_from_native_formats,
    set_best_available_resolution,
    set_exact_resolution,
    set_lowest_available_resolution,
)
from capture.dshow_native_formats import list_native_formats
from config.app_paths import (
    DB_DIR_KEY,
    DOCUMENTS_DIR_KEY,
    FACES_DIR_KEY,
    LOGS_DIR_KEY,
    get_configured_dir,
    get_flagged_images_dir,
    get_pending_dir_change,
    path_label,
    request_dir_change,
)
from config.ocr_languages import ADDITIONAL_OCR_LANGUAGES
from config.settings import (
    RETENTION_DAYS_NEVER,
    get_barcode_overlap_threshold,
    get_capture_shortcut,
    get_debug_logging_enabled,
    get_disk_space_warning_mb,
    get_doc_camera_auto_exposure,
    get_doc_camera_backend,
    get_doc_camera_brightness,
    get_doc_camera_exposure,
    get_doc_camera_resolution,
    get_face_camera_auto_exposure,
    get_face_camera_backend,
    get_face_camera_brightness,
    get_face_camera_exposure,
    get_face_camera_resolution,
    get_face_match_threshold,
    get_history_page_size,
    get_retention_days,
    get_setting_or_none,
    get_theme,
    resolution_to_setting_value,
    set_setting,
)
from document.country_flags import alpha2_for_country_code
from document.flag_matcher import match_basis_label
from document.statistics import (
    compute_daily_volume,
    compute_kpi_trends,
    compute_statistics,
    scans_last_30_days_delta_percent,
    top_n_with_other,
)
from prerequisites import (
    PrerequisiteStatus,
    check_cameras,
    check_insightface_model,
    check_ocr_language_packs,
    check_python_packages,
    check_tesseract,
    download_insightface_model,
    install_ocr_language_packs,
    install_python_packages,
    install_tesseract,
)
from storage.backup import create_backup_archive
from storage.models import (
    FlaggedPerson,
    clear_flagged_match_for_person,
    delete_flagged_person,
    insert_flagged_person,
    list_flagged_persons,
    list_scans,
)
from storage.retention import purge_scans_older_than_days
from ui.app_icon import render_app_logo
from ui.barcode_calibration_window import BarcodeCalibrationWindow
from ui.calibration_window import CalibrationWindow
from ui.charts import BarChartWidget, DonutChartWidget, StatTileWidget, categorical_colors, trend_display
from ui.flag_icons import render_flag_icon
from ui.flag_retroactive import apply_flagged_person_edit, run_retroactive_flag_check
from ui.frame_display import bgr_frame_to_pixmap, shrink_to_fit
from ui.image_viewer import ClickableThumbnailLabel
from ui.statistics_export import build_statistics_report_html
from ui.statistics_labels import (
    doc_type_color,
    doc_type_label,
    flags_and_issues_breakdown,
    validation_status_color,
    validation_status_label,
)
from ui.theme import THEME_CHOICES, THEME_DARK, THEME_LIGHT, THEME_SYSTEM, apply_theme
from ui.window_geometry import restore_geometry, save_geometry

logger = logging.getLogger(__name__)

# Size of each camera's live preview label on the Cameras tab -- deliberately small (this is a
# diagnostic view, not the main capture experience -- see ui/main_window.py's own, larger
# previews for that) so the tab stays a reasonable size with both cameras' previews plus stats
# visible at once without scrolling.
_CAMERA_DIAGNOSTICS_PREVIEW_SIZE = (240, 180)

_PURGE_DAY_OPTIONS = [30, 90, 180, 365]

_THEME_LABELS: dict[str, str] = {THEME_SYSTEM: "System default", THEME_LIGHT: "Light", THEME_DARK: "Dark"}

# Cameras is deliberately absent: check_cameras() always has can_auto_install=False, since there's
# nothing this app can install to fix missing hardware.
_PREREQUISITE_INSTALLERS: dict[str, Callable[[], bool]] = {
    "Python packages": install_python_packages,
    "Tesseract OCR": install_tesseract,
    "OCR language packs": install_ocr_language_packs,
    "InsightFace buffalo_l model": download_insightface_model,
}

# Built from config.ocr_languages.ADDITIONAL_OCR_LANGUAGES rather than hand-typed, so this list
# can never drift out of sync with what document/ocr.py actually requests and
# prerequisites.py::check_ocr_language_packs actually checks for.
_OCR_LANGUAGE_ROSTER_TEXT = "\n".join(
    f"  {display_name} -> {lang_code}.traineddata" for lang_code, display_name in ADDITIONAL_OCR_LANGUAGES.items()
)

_PREREQUISITE_INSTRUCTIONS: dict[str, str] = {
    "Python packages": (
        "Open a terminal in the project folder with the virtual environment activated and run:\n\n"
        "    pip install -r requirements.txt"
    ),
    "Tesseract OCR": (
        "Download and run the installer from the UB Mannheim build:\n"
        "https://github.com/UB-Mannheim/tesseract/wiki\n\n"
        "Or click \"Install now\" below to install it automatically via winget."
    ),
    "OCR language packs": (
        "Covers accented characters for every supported document-issuing country -- without a "
        "language's pack, its accented letters (å/ä/ö, ü/ß, é/ç, ñ, ...) silently OCR as their "
        "plain unaccented equivalents instead of erroring.\n\n"
        "Full list of languages and the exact file each one needs (the checklist above shows "
        "which of these are currently found/missing):\n"
        f"{_OCR_LANGUAGE_ROSTER_TEXT}\n\n"
        "Note: \"Dutch\" is the Netherlands' language (nld.traineddata) -- it is not the English "
        "word for German (\"Deutsch\"), which is the separate \"German\" entry (deu.traineddata) "
        "above. Check the language name against its file name here if a report seems off.\n\n"
        "Download the missing .traineddata file(s) from "
        "https://github.com/tesseract-ocr/tessdata_fast and place them in Tesseract's tessdata\\ "
        "folder (the one named in the checklist above), or click \"Install now\" below to "
        "download them automatically."
    ),
    "InsightFace buffalo_l model": (
        "This ~330MB face-recognition model pack downloads automatically the first time the app "
        "runs, and requires an internet connection.\n\n"
        "Click \"Install now\" below to trigger the download now."
    ),
    "Cameras": (
        "Connect two UVC-compatible USB webcams -- one for the visitor's face, one for the ID "
        "document -- then restart the app. There is nothing this app can install to fix missing "
        "camera hardware."
    ),
}


def _check_non_camera_prerequisites() -> list[PrerequisiteStatus]:
    """Every prerequisite check except Cameras, which is deliberately excluded here -- see
    SettingsWindow._build_prerequisites_group's docstring for why."""
    return [
        check_python_packages(),
        check_tesseract(),
        check_ocr_language_packs(),
        check_insightface_model(),
    ]


def _check_all_prerequisites() -> list[PrerequisiteStatus]:
    """Run every prerequisite check fresh (not cached), including Cameras -- for the manual "refresh after install" and "info" paths, where a real hardware probe is an expected, one-off cost of that explicit action rather than something run unconditionally on window open."""
    return [*_check_non_camera_prerequisites(), check_cameras()]


_PREREQUISITE_INSTALL_INFO_DIALOG_GEOMETRY_KEY = "prerequisite_install_info_dialog_geometry"


class PrerequisiteInstallInfoWindow(QDialog):
    """A small modal window explaining how to satisfy one missing prerequisite, with a one-click installer when available."""

    def __init__(
        self,
        status: PrerequisiteStatus,
        installer: Callable[[], bool] | None,
        on_installed: Callable[[], None],
        conn: sqlite3.Connection,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._installer = installer
        self._on_installed = on_installed
        self._status_name = status.name
        self.finished.connect(lambda _result: save_geometry(conn, self, _PREREQUISITE_INSTALL_INFO_DIALOG_GEOMETRY_KEY))

        self.setWindowTitle(f"{status.name} — install instructions")
        layout = QVBoxLayout(self)

        name_label = QLabel(status.name)
        name_label.setStyleSheet("font-weight: bold; font-size: 12pt;")
        layout.addWidget(name_label)

        detail_label = QLabel(status.detail)
        detail_label.setWordWrap(True)
        layout.addWidget(detail_label)

        instructions_label = QLabel(_PREREQUISITE_INSTRUCTIONS.get(status.name, "No additional instructions available."))
        instructions_label.setWordWrap(True)
        instructions_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(instructions_label)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        if installer is not None:
            install_button = QPushButton("Install now")
            install_button.clicked.connect(self._on_install_clicked)
            button_row.addWidget(install_button)

        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

        restore_geometry(conn, self, _PREREQUISITE_INSTALL_INFO_DIALOG_GEOMETRY_KEY)

    def _on_install_clicked(self) -> None:
        logger.info("Operator triggered install for prerequisite '%s'", self._status_name)
        success = self._installer()
        if success:
            logger.info("Prerequisite '%s' installed successfully", self._status_name)
            QMessageBox.information(self, "Install", "Installed successfully.")
            self._on_installed()
            self.close()
        else:
            logger.warning("Prerequisite '%s' installation failed", self._status_name)
            QMessageBox.warning(self, "Install", "Installation failed. See README.md for manual setup instructions.")


_FLAGGED_THUMBNAIL_SIZE = (48, 48)


_EDIT_FLAGGED_PERSON_DIALOG_GEOMETRY_KEY = "edit_flagged_person_dialog_geometry"


class _EditFlaggedPersonDialog(QDialog):
    """Edit an existing flagged-persons watch-list entry's name/personal number/notes/reference image -- opened via the Edit button on its row (see _FlaggedPersonRowWidget). Saving is handled by the caller via ui/flag_retroactive.py::apply_flagged_person_edit, which warns first if the change would remove the flag from any currently-flagged scan -- this dialog only collects the new values."""

    def __init__(self, flagged_person: FlaggedPerson, conn: sqlite3.Connection, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Flagged Person")
        self._conn = conn
        # Unchanged unless _on_choose_image_clicked picks a new one -- this dialog doesn't force
        # the operator to re-pick a reference image just to fix a typo'd name.
        self._image_path = flagged_person.image_path

        layout = QVBoxLayout(self)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.name_edit = QLineEdit(flagged_person.name or "")
        name_row.addWidget(self.name_edit, stretch=1)
        layout.addLayout(name_row)

        number_row = QHBoxLayout()
        number_row.addWidget(QLabel("Personal number:"))
        self.personal_number_edit = QLineEdit(flagged_person.personal_number or "")
        self.personal_number_edit.setPlaceholderText("YYMMDD-XXXX")
        number_row.addWidget(self.personal_number_edit, stretch=1)
        layout.addLayout(number_row)

        notes_row = QHBoxLayout()
        notes_row.addWidget(QLabel("Notes:"))
        self.notes_edit = QLineEdit(flagged_person.notes or "")
        self.notes_edit.setPlaceholderText("Optional")
        notes_row.addWidget(self.notes_edit, stretch=1)
        layout.addLayout(notes_row)

        image_row = QHBoxLayout()
        self.image_label = QLabel(Path(self._image_path).name if self._image_path else "No image selected")
        self.image_label.setStyleSheet("color: gray;")
        choose_image_button = QPushButton("Choose image...")
        choose_image_button.clicked.connect(self._on_choose_image_clicked)
        image_row.addWidget(choose_image_button)
        image_row.addWidget(self.image_label, stretch=1)
        layout.addLayout(image_row)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_row.addWidget(cancel_button)
        save_button = QPushButton("Save")
        save_button.setDefault(True)
        save_button.clicked.connect(self.accept)
        button_row.addWidget(save_button)
        layout.addLayout(button_row)

        self.finished.connect(lambda _result: save_geometry(self._conn, self, _EDIT_FLAGGED_PERSON_DIALOG_GEOMETRY_KEY))
        restore_geometry(conn, self, _EDIT_FLAGGED_PERSON_DIALOG_GEOMETRY_KEY)

    def _on_choose_image_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose reference image", "", "Images (*.jpg *.jpeg *.png)")
        if not file_path:
            return
        source = Path(file_path)
        destination = get_flagged_images_dir() / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{source.name}"
        shutil.copyfile(source, destination)
        self._image_path = str(destination)
        self.image_label.setText(source.name)

    def values(self) -> tuple[str, str, str, str | None]:
        """Return (name, personal_number, notes, image_path) once the dialog has been accepted."""
        return (
            self.name_edit.text().strip(),
            self.personal_number_edit.text().strip(),
            self.notes_edit.text().strip(),
            self._image_path,
        )


class _FlaggedPersonRowWidget(QFrame):
    """One row in the Flagged Users list: reference thumbnail (if any), name/personal number/notes, and a Remove button."""

    def __init__(self, conn: sqlite3.Connection, flagged_person: FlaggedPerson, on_removed: Callable[[], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.conn = conn
        self.flagged_person = flagged_person
        self._on_removed = on_removed
        self.setFrameShape(QFrame.Shape.StyledPanel)

        row = QHBoxLayout(self)

        thumbnail_label = ClickableThumbnailLabel(flagged_person.image_path, flagged_person.name or "Flagged person", conn=conn)
        thumbnail_label.setFixedSize(*_FLAGGED_THUMBNAIL_SIZE)
        pixmap = QPixmap(flagged_person.image_path) if flagged_person.image_path else None
        if pixmap is not None and not pixmap.isNull():
            thumbnail_label.setPixmap(
                pixmap.scaled(*_FLAGGED_THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
            thumbnail_label.setCursor(Qt.CursorShape.PointingHandCursor)
            thumbnail_label.setToolTip("Double-click to view full size")
        else:
            thumbnail_label.setText("No\nimage")
            thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumbnail_label.setStyleSheet("color: gray; font-size: 8pt; border: 1px solid gray;")
        row.addWidget(thumbnail_label)

        info_column = QVBoxLayout()
        info_column.addWidget(QLabel(f"<b>{flagged_person.name or '(no name)'}</b>"))
        info_column.addWidget(QLabel(f"Personal number: {flagged_person.personal_number or '-'}"))
        if flagged_person.notes:
            info_column.addWidget(QLabel(f"Notes: {flagged_person.notes}"))
        match_basis_label_widget = QLabel(f"Matches on: {match_basis_label(flagged_person)}")
        match_basis_label_widget.setStyleSheet("color: gray; font-size: 8pt;")
        info_column.addWidget(match_basis_label_widget)
        row.addLayout(info_column, stretch=1)

        edit_button = QPushButton("Edit")
        edit_button.clicked.connect(self._on_edit_clicked)
        row.addWidget(edit_button)

        remove_button = QPushButton("Remove")
        remove_button.setStyleSheet("color: #FF1744;")
        remove_button.clicked.connect(self._on_remove_clicked)
        row.addWidget(remove_button)

    def _on_edit_clicked(self) -> None:
        dialog = _EditFlaggedPersonDialog(self.flagged_person, self.conn, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        name, personal_number, notes, image_path = dialog.values()
        if not name and not personal_number:
            QMessageBox.warning(self, "Edit flagged person", "Check and fill in at least a name or a personal number.")
            return
        applied = apply_flagged_person_edit(
            self.conn, self.flagged_person, name or None, personal_number or None, notes or None, image_path, self
        )
        if applied:
            self._on_removed()  # reuses the same "refresh the whole list + notify MainWindow" callback add/remove already use

    def _on_remove_clicked(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Remove flagged person",
            f"Remove '{self.flagged_person.name or self.flagged_person.personal_number}' from the flagged-persons list?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return
        delete_flagged_person(self.conn, self.flagged_person.id)
        cleared_count = clear_flagged_match_for_person(self.conn, self.flagged_person.id)
        logger.info(
            "Operator removed flagged person id=%d (cleared flag from %d scan(s))", self.flagged_person.id, cleared_count
        )
        self._on_removed()


def _format_percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _resolution_compression_label(fourcc: str) -> str:
    """Short "MJPG" / "YUY2 (uncompressed)"-style label for one resolution's negotiated FourCC, shared by the resolution combo and the diagnostics export -- see probe_supported_resolutions' own comment for why compression can vary per resolution on the same camera."""
    if fourcc.upper() == REQUESTED_FOURCC.upper():
        return fourcc
    return f"{fourcc} (uncompressed)"


def _format_camera_stats_html(stats: CameraStats) -> str:
    """Render a CameraStats snapshot as the Cameras tab's per-camera diagnostics text.

    The point of this whole panel is to answer "is this specific camera the reason things feel
    slow, and if so why" without digging through app.log -- so the two figures most directly tied
    to the sluggish-preview/low-CPU failure mode this app has hit in production (see
    capture/camera_manager.py's own comments on MJPG-vs-uncompressed bandwidth throttling) are
    called out in color rather than left as plain numbers: a compression mismatch (requested MJPG,
    camera actually delivering something else) and a slow last-read time.
    """
    if not stats.is_opened:
        return f"<b>Status:</b> <span style='color:#FF5252;'>Offline</span> (device index {stats.camera_index})"

    resolution_text = f"{stats.resolution[0]} × {stats.resolution[1]}" if stats.resolution else "-"

    actual_fourcc_text = stats.actual_fourcc or "unknown"
    compression_matches = actual_fourcc_text.upper() == stats.requested_fourcc.upper()
    compression_color = "#00C853" if compression_matches else "#FF9800"
    compression_note = (
        "" if compression_matches else " ⚠ uncompressed fallback — likely cause of a slow/laggy preview"
    )

    fps_text = f"{stats.fps:.1f}" if stats.fps > 0 else "measuring..."
    fps_color = "#FF5252" if 0 < stats.fps < 10 else "inherit"

    read_ms_threshold = SLOW_READ_WARN_THRESHOLD_SECONDS * 1000
    read_color = "#FF9800" if stats.last_read_ms > read_ms_threshold else "inherit"

    lines = [
        "<b>Status:</b> <span style='color:#00C853;'>Online</span>",
        f"<b>Device index:</b> {stats.camera_index}",
        f"<b>Resolution:</b> {resolution_text}",
        f"<b>FPS (measured):</b> <span style='color:{fps_color};'>{fps_text}</span>",
        f"<b>Compression:</b> requested {stats.requested_fourcc}, actual "
        f"<span style='color:{compression_color};'>{actual_fourcc_text}</span>{compression_note}",
        f"<b>Last read time:</b> <span style='color:{read_color};'>{stats.last_read_ms:.0f} ms</span>",
        f"<b>Consecutive read failures:</b> {stats.consecutive_failures}",
    ]
    if stats.slow_read_streak > 0:
        lines.append(f"<span style='color:#FF9800;'><b>Slow reads in a row:</b> {stats.slow_read_streak}</span>")
    # A sustained (not just momentary) slow-read streak alongside an uncompressed-fallback
    # camera means this specific resolution's raw bandwidth is more than this camera/USB link can
    # sustain (e.g. 1920x1080 YUY2 is ~124MB/s of raw data at 30fps, comfortably over what USB 2.0
    # can carry) -- switching to a lower resolution needs far less bandwidth and commonly restores
    # a smooth preview even though the camera still isn't delivering MJPG. Surfaced here (not just
    # left to app.log's own warning) since this is exactly the kind of thing an operator would
    # otherwise have to ask about -- see _SLOW_READ_LOG_EVERY_N's matching threshold in
    # capture/camera_manager.py for why 30 is "sustained, not a one-off blip."
    if not compression_matches and stats.slow_read_streak >= 30:
        lines.append(
            "<span style='color:#FF9800;'>Try switching this camera's Resolution below to "
            "\"Auto (lowest available)\" (or a lower fixed resolution) -- the current resolution's "
            "raw, uncompressed data rate is likely more than this camera/USB port can sustain.</span>"
        )
    return "<br>".join(lines)


def _format_resolution_preference(resolution) -> str:
    """Human-readable form of a ResolutionPreference (None/AUTO_LOWEST/an exact size) -- mirrors the wording already used in the resolution combo boxes elsewhere in this window."""
    if resolution is None:
        return "Auto (best available)"
    if resolution == AUTO_LOWEST:
        return "Auto (lowest available)"
    return f"{resolution[0]} x {resolution[1]}"


def _format_camera_diagnostics_report(
    side_label: str, diagnostics: CameraDiagnostics, configured_resolution, stats: CameraStats
) -> str:
    """Render a CameraDiagnostics snapshot (plus the configured resolution preference and current
    CameraStats) as the plain-text report shown/copied/saved by "Export diagnostics..." -- see
    SettingsWindow._on_export_camera_diagnostics_clicked. Plain text rather than HTML (unlike
    _format_camera_stats_html) since this is meant to be pasted into a bug report or chat message,
    not just displayed in a label.
    """
    lines = [
        f"=== {side_label} Camera Diagnostics ===",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Device index: {diagnostics.camera_index}",
        f"Status: {'Online' if diagnostics.is_opened else 'Offline'}",
    ]
    if not diagnostics.is_opened:
        lines.append("")
        lines.append("(Camera is not open -- no further properties available.)")
        return "\n".join(lines)

    lines += [
        f"Backend: {diagnostics.backend_name}",
        f"Configured resolution preference: {_format_resolution_preference(configured_resolution)}",
        f"Negotiated compression (FourCC): {diagnostics.fourcc}",
        "",
        "--- Current driver-reported properties ---",
    ]
    lines += [f"{label}: {value:g}" for label, value in diagnostics.properties.items()]

    lines += ["", "--- Supported resolutions (probed via cv2, trial-and-error) ---"]
    if diagnostics.supported_resolutions:
        lines += [
            f"{width} x {height}  ({_resolution_compression_label(fourcc)})"
            for width, height, fourcc in diagnostics.supported_resolutions
        ]
    else:
        lines.append("(none detected)")

    lines += ["", "--- Native modes (exact, via DirectShow IAMStreamConfig) ---"]
    if diagnostics.native_formats:
        lines += [str(native_format) for native_format in diagnostics.native_formats]
    else:
        lines.append("(none detected -- possibly unavailable while the camera is in use, or on this backend)")

    lines += [
        "",
        "--- Measured runtime stats (this session) ---",
        f"FPS (measured): {stats.fps:.1f}" if stats.fps > 0 else "FPS (measured): still measuring",
        f"Last read time: {stats.last_read_ms:.0f} ms",
        f"Consecutive read failures: {stats.consecutive_failures}",
        f"Slow reads in a row: {stats.slow_read_streak}",
    ]
    return "\n".join(lines)


_CAMERA_DIAGNOSTICS_DIALOG_GEOMETRY_KEY = "camera_diagnostics_dialog_geometry"


class _CameraDiagnosticsDialog(QDialog):
    """Shows a camera diagnostics report (see _format_camera_diagnostics_report) as read-only,
    selectable, monospace text, with quick actions to copy it or save it to a file -- for pasting
    into a bug report or sending to whoever is troubleshooting a camera issue remotely."""

    def __init__(self, report_text: str, title: str, conn: sqlite3.Connection, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 560)
        restore_geometry(conn, self, _CAMERA_DIAGNOSTICS_DIALOG_GEOMETRY_KEY)
        self.finished.connect(lambda _result: save_geometry(conn, self, _CAMERA_DIAGNOSTICS_DIALOG_GEOMETRY_KEY))
        self._report_text = report_text

        layout = QVBoxLayout(self)

        text_edit = QPlainTextEdit(report_text)
        text_edit.setReadOnly(True)
        text_edit.setStyleSheet("QPlainTextEdit { font-family: Consolas, 'Courier New', monospace; }")
        text_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(text_edit, stretch=1)

        button_row = QHBoxLayout()
        copy_button = QPushButton("Copy to clipboard")
        copy_button.clicked.connect(self._on_copy_clicked)
        button_row.addWidget(copy_button)
        save_button = QPushButton("Save to file...")
        save_button.clicked.connect(self._on_save_clicked)
        button_row.addWidget(save_button)
        button_row.addStretch(1)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.accept)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _on_copy_clicked(self) -> None:
        QApplication.clipboard().setText(self._report_text)

    def _on_save_clicked(self) -> None:
        default_name = f"kontroll_camera_diagnostics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path, _ = QFileDialog.getSaveFileName(self, "Save camera diagnostics", default_name, "Text files (*.txt)")
        if not file_path:
            return
        Path(file_path).write_text(self._report_text, encoding="utf-8")
        QMessageBox.information(self, "Save camera diagnostics", f"Saved to:\n{file_path}")


# Persists this window's size/position across restarts (see _restore_window_geometry/
# _save_window_geometry) -- distinct from ui/main_window.py's own "window_geometry" key, since the
# two windows' sizes are unrelated to each other.
_SETTINGS_WINDOW_GEOMETRY_SETTING_KEY = "settings_window_geometry"


class _BackendSwitchWorker(QObject):
    """Runs CameraManager.set_face_camera_backend/set_doc_camera_backend on a background thread,
    so a slow device release/reacquire never blocks the UI -- diagnosed against a real operator
    report of the whole app freezing for a long time when switching a camera back to DirectShow
    right after Media Foundation. set_face_camera_backend/set_doc_camera_backend already insert a
    deliberate settling delay and retry a few times on top of that (see
    capture/camera_manager.py::_open_camera_verified) specifically because a UVC device doesn't
    always finish releasing from one capture API before a different one tries to claim it -- worst
    case, several seconds of real blocking hardware I/O, which previously ran directly on
    SettingsWindow._on_backend_changed's calling (UI) thread.
    """

    finished = pyqtSignal(bool, int)  # (succeeded, effective_backend)

    def __init__(self, camera_manager: CameraManager, side: str, backend: int) -> None:
        super().__init__()
        self._camera_manager = camera_manager
        self._side = side
        self._backend = backend

    def run(self) -> None:
        if self._side == "face":
            succeeded, effective_backend = self._camera_manager.set_face_camera_backend(self._backend)
        else:
            succeeded, effective_backend = self._camera_manager.set_doc_camera_backend(self._backend)
        self.finished.emit(succeeded, effective_backend)


class SettingsWindow(QWidget):
    """A standalone settings window covering manual scan purge, camera assignment, the capture
    shortcut, and a live prerequisites checklist."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        camera_manager: CameraManager,
        face_analysis_app: FaceAnalysis,
        on_data_changed: Callable[[], None],
        on_capture_shortcut_changed: Callable[[str], None],
        on_camera_probe_started: Callable[[], None] | None = None,
        on_camera_probe_finished: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        # See ui/main_window.py::MainWindow's identical call for why this is set explicitly rather
        # than relying solely on QApplication.setWindowIcon's application-wide default.
        self.setWindowIcon(QIcon(render_app_logo(64)))
        self.conn = conn
        self.camera_manager = camera_manager
        self.face_analysis_app = face_analysis_app
        self._on_data_changed = on_data_changed
        self._on_capture_shortcut_changed = on_capture_shortcut_changed
        # Lets the main window pause its own live-preview poll (see ui/main_window.py) for the
        # duration of a camera probe below -- both would otherwise read the same VideoCapture at
        # once, since pumping the event loop mid-probe (via QApplication.processEvents, to avoid
        # the app appearing frozen for the ~1-2s/camera this takes) lets that timer's callback run
        # interleaved with the probe's own capture.set()/.read() calls.
        self._on_camera_probe_started = on_camera_probe_started or (lambda: None)
        self._on_camera_probe_finished = on_camera_probe_finished or (lambda: None)
        self._prerequisite_rows: dict[str, tuple[QLabel, QPushButton]] = {}
        self._path_rows: dict[str, tuple[QLabel, QLabel]] = {}
        self._calibration_window: CalibrationWindow | None = None
        self._barcode_calibration_window: BarcodeCalibrationWindow | None = None
        # Keeps the QThread/worker for an in-progress backend switch alive (see _on_backend_changed)
        # -- nothing else would hold a reference to them otherwise, and Qt does not appreciate a
        # QThread being garbage-collected while still running. None means no switch is in progress,
        # also used to ignore a second trigger while one is already running.
        self._backend_switch_thread: QThread | None = None
        self._backend_switch_worker: _BackendSwitchWorker | None = None
        # Populated by _refresh_statistics_tab, the first time the operator opens the Statistics
        # tab (see _on_settings_tab_changed) -- kept on self so the period combo (_on_statistics_
        # period_changed) and the Print Report button (_on_print_statistics_report_clicked) can
        # both reuse the same already-loaded scan list instead of re-querying the database.
        self._statistics_scans: list | None = None
        self._statistics = None
        self._statistics_trends = None
        self._statistics_daily_volume: list | None = None
        self._statistics_daily_volume_days: int = 30

        self.setWindowTitle("Settings")
        self._apply_responsive_size()
        self._restore_window_geometry()
        self._build_layout()

    def _apply_responsive_size(self) -> None:
        """Size the window generously but never larger than the available screen, and let it be resized down further -- tabs + per-tab scroll areas (see _build_scrollable_tab) mean no control ever gets stranded off-screen on a small/low-resolution display.

        This is only the *first-run* default -- _restore_window_geometry (called right after, in
        __init__) overrides it with whatever size/position the operator last left the window at, if
        anything was saved, the same way ui/main_window.py's own window remembers its size.
        """
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        width = min(560, available.width() - 60) if available is not None else 560
        height = min(620, available.height() - 80) if available is not None else 620
        self.resize(max(width, 360), max(height, 320))
        self.setMinimumSize(360, 320)

    def _restore_window_geometry(self) -> None:
        """Restore this window's size/position from the last session, if anything was saved -- leaves _apply_responsive_size's first-run default in place otherwise."""
        geometry_b64 = get_setting_or_none(self.conn, _SETTINGS_WINDOW_GEOMETRY_SETTING_KEY)
        if geometry_b64 is not None:
            self.restoreGeometry(QByteArray(base64.b64decode(geometry_b64)))

    def _save_window_geometry(self) -> None:
        """Persist this window's current size/position so the next time it's opened restores it -- called from closeEvent."""
        geometry_b64 = base64.b64encode(bytes(self.saveGeometry())).decode("ascii")
        set_setting(self.conn, _SETTINGS_WINDOW_GEOMETRY_SETTING_KEY, geometry_b64)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        self._save_window_geometry()
        super().closeEvent(event)

    def is_backend_switch_in_progress(self) -> bool:
        """Whether a camera backend switch (see _on_backend_changed) is currently running on its background thread -- checked by ui/main_window.py::MainWindow.closeEvent before releasing both camera handles out from under it."""
        return self._backend_switch_thread is not None

    def _build_layout(self) -> None:
        """Group settings into tabs (each independently scrollable) rather than one long flat list -- keeps any single tab short enough to fit a small screen and avoids an overwhelming wall of controls."""
        outer_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        outer_layout.addWidget(self.tab_widget)

        self.tab_widget.addTab(self._build_scrollable_tab(self._build_general_tab()), "General")
        self._cameras_tab_index = self.tab_widget.addTab(self._build_scrollable_tab(self._build_cameras_tab()), "Cameras")
        self.tab_widget.addTab(self._build_scrollable_tab(self._build_face_match_tab()), "Face Match")
        self.tab_widget.addTab(self._build_scrollable_tab(self._build_document_validation_tab()), "Document Validation")
        self.tab_widget.addTab(self._build_scrollable_tab(self._build_data_tab()), "Data")
        self.tab_widget.addTab(self._build_scrollable_tab(self._build_flagged_users_tab()), "Flagged Users")
        self._statistics_tab_index = self.tab_widget.addTab(self._build_scrollable_tab(self._build_statistics_tab()), "Statistics")
        self._prerequisites_tab_index = self.tab_widget.addTab(
            self._build_scrollable_tab(self._build_prerequisites_tab()), "Prerequisites"
        )

        # Detecting a camera's supported resolutions is a real hardware round-trip per candidate
        # (~1-2s per camera) -- doing that for both cameras unconditionally while building this
        # window (i.e. on every "Settings" click, since it's a singleton rebuilt only once per
        # session) made opening it noticeably slow. Deferred to the first time the operator
        # actually looks at the Cameras tab instead; _build_cameras_tab only does the cheap,
        # instant part (current setting shown, nothing probed) up front. Device enumeration itself
        # (_populate_camera_assignment, populating which cameras are even available to assign)
        # shares this same flag and is deferred alongside it, for the same reason.
        self._cameras_tab_populated = False
        # Checking camera hardware for the Prerequisites tab (prerequisites.check_cameras) is the
        # same kind of ~10-device hardware round-trip -- deferred the same way, to the first time
        # that tab is actually shown, rather than run unconditionally on window open.
        self._cameras_prerequisite_checked = False
        self.tab_widget.currentChanged.connect(self._on_settings_tab_changed)

    def _on_settings_tab_changed(self, index: int) -> None:
        if index == self._cameras_tab_index and not self._cameras_tab_populated:
            self._cameras_tab_populated = True
            self._populate_camera_assignment()
            self._refresh_resolution_combo("face")
            self._refresh_resolution_combo("doc")
        if index == self._prerequisites_tab_index and not self._cameras_prerequisite_checked:
            self._cameras_prerequisite_checked = True
            self._apply_prerequisite_status(check_cameras())
        if index == self._statistics_tab_index:
            # Unlike the Cameras tab's one-time hardware probe above, this recomputes every time
            # the tab is shown (not just once) -- it's just an in-memory pass over already-loaded
            # scan rows (no hardware round-trip), and new scans can arrive from the main window
            # while this Settings window stays open.
            self._refresh_statistics_tab()

    def _build_scrollable_tab(self, content: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(content)
        return scroll_area

    def _build_general_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_appearance_group())
        layout.addWidget(self._build_shortcut_group())
        layout.addWidget(self._build_disk_space_warning_group())
        layout.addWidget(self._build_history_page_size_group())
        layout.addWidget(self._build_logging_group())
        layout.addStretch(1)
        return page

    def _build_cameras_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_camera_diagnostics_group())
        layout.addWidget(self._build_camera_group())
        layout.addStretch(1)
        return page

    def _build_face_match_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_face_match_threshold_group())
        layout.addStretch(1)
        return page

    def _build_document_validation_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_barcode_overlap_threshold_group())
        layout.addStretch(1)
        return page

    def _build_data_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_data_location_group())
        layout.addWidget(self._build_auto_purge_group())
        layout.addWidget(self._build_purge_group())
        layout.addWidget(self._build_backup_group())
        layout.addStretch(1)
        return page

    def _build_statistics_tab(self) -> QWidget:
        """Build the Statistics tab: KPI tiles, a scan-volume chart, and a handful of breakdown charts, all computed from the saved scan history (see document/statistics.py).

        Left empty of real data at build time -- charts and tiles are populated by
        _refresh_statistics_tab, which _on_settings_tab_changed calls every time this tab is
        actually shown (not just once, unlike the Cameras tab's hardware probe), since new scans
        can be saved from the main window while this Settings window stays open.
        """
        page = QWidget()
        layout = QVBoxLayout(page)

        intro_label = QLabel("Statistics are computed from every saved scan.")
        intro_label.setWordWrap(True)
        layout.addWidget(intro_label)

        self.statistics_tiles_layout = QGridLayout()
        layout.addLayout(self.statistics_tiles_layout)

        self.statistics_volume_group = QGroupBox("Scan volume")
        volume_layout = QVBoxLayout(self.statistics_volume_group)
        period_row = QHBoxLayout()
        period_row.addWidget(QLabel("Period:"))
        self.statistics_period_combo = QComboBox()
        self.statistics_period_combo.addItem("Last 7 days", 7)
        self.statistics_period_combo.addItem("Last 30 days", 30)
        self.statistics_period_combo.addItem("Last 90 days", 90)
        self.statistics_period_combo.addItem("Last 12 months", 365)
        self.statistics_period_combo.setCurrentIndex(1)
        self.statistics_period_combo.currentIndexChanged.connect(self._on_statistics_period_changed)
        period_row.addWidget(self.statistics_period_combo)
        period_row.addStretch(1)
        volume_layout.addLayout(period_row)
        self.statistics_volume_chart = BarChartWidget(orientation="vertical")
        volume_layout.addWidget(self.statistics_volume_chart)
        layout.addWidget(self.statistics_volume_group)

        # Stacked, not side-by-side -- the Settings window's default width (~560px, see
        # _apply_responsive_size) is too narrow to fit two donuts-plus-legend at a readable size
        # without forcing this tab into horizontal scrolling, which the rest of Settings never does.
        validation_group = QGroupBox("Document validation status")
        validation_layout = QVBoxLayout(validation_group)
        self.statistics_validation_chart = DonutChartWidget()
        validation_layout.addWidget(self.statistics_validation_chart)
        layout.addWidget(validation_group)

        doc_type_group = QGroupBox("Document type")
        doc_type_layout = QVBoxLayout(doc_type_group)
        self.statistics_doc_type_chart = DonutChartWidget()
        doc_type_layout.addWidget(self.statistics_doc_type_chart)
        layout.addWidget(doc_type_group)

        age_group = QGroupBox("Age distribution")
        age_layout = QVBoxLayout(age_group)
        age_caption = QLabel("Only scans with a readable birth date are counted.")
        age_caption.setStyleSheet("font-size: 8pt;")
        age_layout.addWidget(age_caption)
        self.statistics_age_chart = DonutChartWidget()
        age_layout.addWidget(self.statistics_age_chart)
        layout.addWidget(age_group)

        nationality_group = QGroupBox("Nationality / issuing country (top 10)")
        nationality_layout = QVBoxLayout(nationality_group)
        self.statistics_nationality_chart = BarChartWidget(orientation="horizontal")
        nationality_layout.addWidget(self.statistics_nationality_chart)
        layout.addWidget(nationality_group)

        flags_group = QGroupBox("Flags && issues")
        flags_layout = QVBoxLayout(flags_group)
        flags_caption = QLabel("Each row is an independent condition -- one scan can count in more than one, so this isn't a breakdown of the total.")
        flags_caption.setWordWrap(True)
        flags_caption.setStyleSheet("font-size: 8pt;")
        flags_layout.addWidget(flags_caption)
        self.statistics_flags_chart = BarChartWidget(orientation="horizontal")
        flags_layout.addWidget(self.statistics_flags_chart)
        layout.addWidget(flags_group)

        report_row = QHBoxLayout()
        report_row.addStretch(1)
        print_report_button = QPushButton("Print Report…")
        print_report_button.clicked.connect(self._on_print_statistics_report_clicked)
        report_row.addWidget(print_report_button)
        layout.addLayout(report_row)

        layout.addStretch(1)
        return page

    def _refresh_statistics_tab(self) -> None:
        scans = list_scans(self.conn, limit=100_000)
        self._statistics_scans = scans
        self._statistics = compute_statistics(scans)
        self._statistics_trends = compute_kpi_trends(scans)
        self._populate_statistics_tiles(self._statistics, self._statistics_trends)
        self._populate_statistics_breakdown_charts(self._statistics)
        self._update_statistics_volume_chart()

    def _populate_statistics_tiles(self, stats, trends) -> None:
        while self.statistics_tiles_layout.count():
            item = self.statistics_tiles_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()

        total_checked = sum(stats.validation_status_counts.values())
        average_score_text = f"avg. similarity {stats.average_match_score:.3f}" if stats.average_match_score is not None else ""

        # Each trend is (arrow text, status color) from trend_display -- higher_is_better is None
        # for a metric with no inherent good/bad direction (raw scan volume, repeat-visit rate),
        # so those two always render in the tile's normal text color, arrow included.
        total_scans_trend = trend_display(scans_last_30_days_delta_percent(stats), None)
        pass_rate_trend = trend_display(trends.verification_pass_rate_delta_pp, True, suffix="pp")
        face_match_trend = trend_display(trends.face_match_rate_delta_pp, True, suffix="pp")
        ocr_trend = trend_display(trends.ocr_extraction_rate_delta_pp, True, suffix="pp")
        security_flags_trend = trend_display(trends.security_flags_delta_percent, False)
        repeat_visit_trend = trend_display(trends.repeat_visit_rate_delta_pp, None, suffix="pp")

        tiles = [
            ("Total scans", str(stats.total_scans), f"{stats.scans_last_30_days} in last 30 days", *total_scans_trend),
            ("Verification pass rate", _format_percent(stats.verification_pass_rate), f"{stats.validation_status_counts.get('valid', 0)} valid of {total_checked} checked", *pass_rate_trend),
            ("Face match rate", _format_percent(stats.face_match_rate), average_score_text, *face_match_trend),
            ("OCR extraction rate", _format_percent(stats.ocr_extraction_rate), f"{stats.image_quality_issue_count} flagged low quality", *ocr_trend),
            ("Security flags", str(stats.security_flags_count), f"{stats.watchlist_matches_count} watch-list, {stats.expired_documents_count} expired", *security_flags_trend),
            ("Repeat visitors", _format_percent(stats.repeat_visit_rate), f"{stats.unique_person_count} unique of {stats.identified_scan_count} identified", *repeat_visit_trend),
        ]
        # 2 columns, not 3 -- the Settings window's default width (~560px) only comfortably fits
        # two of these side by side without truncating the detail line under it.
        columns = 2
        for index, (title, value, detail, trend_text, trend_color) in enumerate(tiles):
            tile = StatTileWidget(title)
            tile.set_value(value, detail, trend_text, trend_color)
            self.statistics_tiles_layout.addWidget(tile, index // columns, index % columns)

    def _populate_statistics_breakdown_charts(self, stats) -> None:
        validation_items = [
            (validation_status_label(status), count, validation_status_color(status))
            for status, count in stats.validation_status_counts.items()
        ]
        self.statistics_validation_chart.set_data(validation_items)

        doc_type_palette = categorical_colors(self.statistics_doc_type_chart)
        doc_type_items = [
            (doc_type_label(doc_type), count, doc_type_color(doc_type, doc_type_palette))
            for doc_type, count in stats.doc_type_counts.items()
        ]
        self.statistics_doc_type_chart.set_data(doc_type_items)

        # Fixed bucket order (youngest to oldest, see document/statistics.py::_AGE_BUCKETS), not
        # sorted by count like the other donut charts above -- age reads naturally left-to-right/
        # young-to-old. Colors are assigned positionally from the categorical palette rather than
        # via a label->color lookup (unlike doc_type_color/validation_status_color) since there's
        # no separately-registered color for an age bucket anywhere else in the app to stay
        # consistent with.
        age_palette = categorical_colors(self.statistics_age_chart)
        age_items = [
            (label, count, age_palette[index % len(age_palette)])
            for index, (label, count) in enumerate(stats.age_bucket_counts.items())
        ]
        self.statistics_age_chart.set_data(age_items)

        nationality_items = top_n_with_other(stats.nationality_counts, 10)
        nationality_icons = [
            render_flag_icon(alpha2_for_country_code(label)) if alpha2_for_country_code(label) else None
            for label, _ in nationality_items
        ]
        self.statistics_nationality_chart.set_data(
            [label for label, _ in nationality_items],
            [float(count) for _, count in nationality_items],
            icons=nationality_icons,
        )

        flags_items = flags_and_issues_breakdown(stats)
        self.statistics_flags_chart.set_data([label for label, _ in flags_items], [float(count) for _, count in flags_items])

    def _update_statistics_volume_chart(self) -> None:
        if self._statistics_scans is None:
            return
        days = self.statistics_period_combo.currentData()
        daily_volume = compute_daily_volume(self._statistics_scans, days)
        self._statistics_daily_volume = daily_volume
        self._statistics_daily_volume_days = days

        date_format = "%m-%d" if days <= 90 else "%Y-%m"
        labels = [point.day.strftime(date_format) for point in daily_volume]
        values = [float(point.count) for point in daily_volume]
        self.statistics_volume_chart.set_data(labels, values)
        self.statistics_volume_group.setTitle(f"Scan volume — last {days} days")

    def _on_statistics_period_changed(self, _index: int) -> None:
        self._update_statistics_volume_chart()

    def _on_print_statistics_report_clicked(self) -> None:
        if self._statistics is None:
            self._refresh_statistics_tab()

        report_html = build_statistics_report_html(
            self._statistics, self._statistics_trends, self._statistics_daily_volume, self._statistics_daily_volume_days
        )
        default_name = f"kontroll_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        file_path, _ = QFileDialog.getSaveFileName(self, "Print statistics report", default_name, "HTML files (*.html)")
        if not file_path:
            return

        Path(file_path).write_text(report_html, encoding="utf-8")
        logger.info("Operator exported the statistics report to %s", file_path)
        QMessageBox.information(self, "Print statistics report", f"Saved to:\n{file_path}")

    def _build_prerequisites_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.addWidget(self._build_prerequisites_group())
        layout.addStretch(1)
        return page

    # -- Appearance -------------------------------------------------------------------------------

    def _build_appearance_group(self) -> QGroupBox:
        group = QGroupBox("Appearance")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Theme:"))

        self.theme_combo = QComboBox()
        for mode in THEME_CHOICES:
            self.theme_combo.addItem(_THEME_LABELS[mode], mode)
        self.theme_combo.setCurrentIndex(self.theme_combo.findData(get_theme(self.conn)))
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        row.addWidget(self.theme_combo, stretch=1)

        return group

    def _on_theme_changed(self, _list_index: int) -> None:
        mode = self.theme_combo.currentData()
        set_setting(self.conn, "theme", mode)
        apply_theme(QApplication.instance(), mode)
        logger.info("Operator set the app theme to '%s'", mode)

    # -- Data storage locations -----------------------------------------------------------------

    def _build_data_location_group(self) -> QGroupBox:
        group = QGroupBox("Data storage locations")
        layout = QVBoxLayout(group)

        for key in (DB_DIR_KEY, FACES_DIR_KEY, DOCUMENTS_DIR_KEY, LOGS_DIR_KEY):
            layout.addWidget(self._build_single_path_row(key))

        return group

    def _build_single_path_row(self, key: str) -> QWidget:
        """Build one path's row: label, current path, an optional pending-move notice, and a Change... button. Each of the three paths is independently configurable."""
        row_widget = QWidget()
        layout = QVBoxLayout(row_widget)
        layout.setContentsMargins(0, 4, 0, 4)

        header_row = QHBoxLayout()
        header_row.addWidget(QLabel(f"{path_label(key)}:"), stretch=1)
        change_button = QPushButton("Change...")
        change_button.clicked.connect(lambda _checked, k=key: self._on_change_dir_clicked(k))
        header_row.addWidget(change_button)
        layout.addLayout(header_row)

        path_value_label = QLabel(str(get_configured_dir(key)))
        path_value_label.setWordWrap(True)
        path_value_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path_value_label)

        pending_notice_label = QLabel()
        pending_notice_label.setWordWrap(True)
        pending_notice_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        layout.addWidget(pending_notice_label)

        self._path_rows[key] = (path_value_label, pending_notice_label)
        self._update_pending_notice(key)

        return row_widget

    def _update_pending_notice(self, key: str) -> None:
        _, pending_notice_label = self._path_rows[key]
        pending = get_pending_dir_change(key)
        if pending is None:
            pending_notice_label.setVisible(False)
            return
        pending_notice_label.setText(f"Pending move to: {pending}\nRestart Kontroll to complete this change.")
        pending_notice_label.setVisible(True)

    def _on_change_dir_clicked(self, key: str) -> None:
        current_dir = get_configured_dir(key)
        chosen_dir = QFileDialog.getExistingDirectory(self, f"Choose new {path_label(key).lower()} folder", str(current_dir))
        if not chosen_dir:
            return

        new_dir = Path(chosen_dir)
        if new_dir.resolve() == current_dir.resolve():
            return

        confirm = QMessageBox.question(
            self,
            f"Change {path_label(key).lower()} folder",
            f"Move the {path_label(key).lower()} data to:\n\n{new_dir}\n\n"
            "This takes effect the next time Kontroll starts. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        request_dir_change(key, new_dir)
        logger.info("Operator requested moving %s from %s to %s (pending restart)", key, current_dir, new_dir)
        self._update_pending_notice(key)
        QMessageBox.information(
            self,
            f"Change {path_label(key).lower()} folder",
            "Saved. Restart Kontroll to move the data and finish switching to the new location.",
        )

    # -- Automatic purge on startup -------------------------------------------------------------

    def _build_auto_purge_group(self) -> QGroupBox:
        group = QGroupBox("Automatic purge")
        row = QHBoxLayout()
        row.addWidget(QLabel("Keep scans for:"))

        current_days = get_retention_days(self.conn)
        self.retention_days_combo = QComboBox()
        for days in _PURGE_DAY_OPTIONS:
            self.retention_days_combo.addItem(f"{days} days", days)
        self.retention_days_combo.addItem("Never", RETENTION_DAYS_NEVER)
        # The persisted value might not be one of the preset options (e.g. hand-edited in the DB)
        # -- add it anyway so the combo reflects the actual setting instead of silently jumping
        # to whatever option happens to be first in the list.
        if current_days not in _PURGE_DAY_OPTIONS and current_days != RETENTION_DAYS_NEVER:
            self.retention_days_combo.addItem(f"{current_days} days (custom)", current_days)
        self.retention_days_combo.setCurrentIndex(self.retention_days_combo.findData(current_days))
        self.retention_days_combo.currentIndexChanged.connect(self._on_retention_days_changed)
        row.addWidget(self.retention_days_combo)
        row.addStretch(1)

        self.auto_purge_note_label = QLabel()
        self.auto_purge_note_label.setWordWrap(True)
        self.auto_purge_note_label.setStyleSheet("color: gray;")
        self._update_auto_purge_note(current_days)

        outer_layout = QVBoxLayout()
        outer_layout.addLayout(row)
        outer_layout.addWidget(self.auto_purge_note_label)
        group.setLayout(outer_layout)

        return group

    def _update_auto_purge_note(self, retention_days: int) -> None:
        if retention_days == RETENTION_DAYS_NEVER:
            self.auto_purge_note_label.setText("Automatic purge is disabled -- scans are kept indefinitely.")
        else:
            self.auto_purge_note_label.setText("Scans older than this are deleted automatically every time Kontroll starts.")

    def _on_retention_days_changed(self, _list_index: int) -> None:
        new_days = self.retention_days_combo.currentData()
        set_setting(self.conn, "retention_days", str(new_days))
        self._update_auto_purge_note(new_days)
        logger.info("Operator set the automatic purge retention period to %d days", new_days)

    # -- Purge old scans -----------------------------------------------------------------------

    def _build_purge_group(self) -> QGroupBox:
        group = QGroupBox("Purge old scans")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Delete scans older than:"))

        self.purge_days_combo = QComboBox()
        for days in _PURGE_DAY_OPTIONS:
            self.purge_days_combo.addItem(f"{days} days", days)
        # A cutoff of 0 days ("older than now") already means "every scan" to
        # purge_scans_older_than_days -- see its docstring -- so this reuses that behavior rather
        # than needing a separate delete-everything code path. Kept out of _PURGE_DAY_OPTIONS
        # (shared with the auto-purge-on-startup combo above) since "delete everything" would be a
        # dangerous, easy-to-fat-finger choice for an *automatic* startup purge, unlike here where
        # it's one deliberate manual action with its own stronger confirmation wording below.
        self.purge_days_combo.addItem("All scans (delete everything)", 0)
        row.addWidget(self.purge_days_combo)

        purge_button = QPushButton("Purge")
        purge_button.clicked.connect(self._on_purge_clicked)
        row.addWidget(purge_button)
        row.addStretch(1)

        return group

    def _on_purge_clicked(self) -> None:
        days = self.purge_days_combo.currentData()
        if days == 0:
            prompt = "Permanently delete ALL scans, including every saved image? This cannot be undone."
        else:
            prompt = f"Permanently delete every scan older than {days} days, including saved images? This cannot be undone."
        confirm = QMessageBox.question(
            self,
            "Purge old scans",
            prompt,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        purged_count = purge_scans_older_than_days(self.conn, days)
        logger.info("Operator purged %d scan(s) (cutoff: %d days)", purged_count, days)
        QMessageBox.information(self, "Purge old scans", f"Deleted {purged_count} scan(s).")
        self._on_data_changed()

    # -- Disk space warning ----------------------------------------------------------------------

    def _build_disk_space_warning_group(self) -> QGroupBox:
        group = QGroupBox("Disk space warning")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Warn in the status bar when free disk space falls below:"))

        self.disk_space_warning_spin = QSpinBox()
        self.disk_space_warning_spin.setRange(100, 500_000)
        self.disk_space_warning_spin.setSingleStep(100)
        self.disk_space_warning_spin.setSuffix(" MB")
        self.disk_space_warning_spin.setValue(get_disk_space_warning_mb(self.conn))
        row.addWidget(self.disk_space_warning_spin)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save_disk_space_warning_clicked)
        row.addWidget(save_button)
        row.addStretch(1)

        return group

    def _on_save_disk_space_warning_clicked(self) -> None:
        new_threshold_mb = self.disk_space_warning_spin.value()
        set_setting(self.conn, "disk_space_warning_mb", str(new_threshold_mb))
        logger.info("Operator set the low-disk-space warning threshold to %d MB", new_threshold_mb)
        QMessageBox.information(self, "Disk space warning", f"Warning threshold set to {new_threshold_mb} MB.")
        self._on_data_changed()  # refreshes the main window's status bar immediately, not just on next capture

    # -- History list loading -----------------------------------------------------------------------

    def _build_history_page_size_group(self) -> QGroupBox:
        group = QGroupBox("History list")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Scans loaded at a time:"))

        self.history_page_size_spin = QSpinBox()
        # Floor of 5 keeps at least a useful initial batch even at the low end; no particular
        # ceiling reason for 500 beyond "large enough that anyone wanting more is really asking to
        # just load everything at once" -- each row is a real per-image decode cost (see
        # DEFAULT_HISTORY_PAGE_SIZE's own comment), so this isn't meant to invite a huge value.
        self.history_page_size_spin.setRange(5, 500)
        self.history_page_size_spin.setValue(get_history_page_size(self.conn))
        row.addWidget(self.history_page_size_spin)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save_history_page_size_clicked)
        row.addWidget(save_button)
        row.addStretch(1)

        return group

    def _on_save_history_page_size_clicked(self) -> None:
        new_page_size = self.history_page_size_spin.value()
        set_setting(self.conn, "history_page_size", str(new_page_size))
        logger.info("Operator set the history list load-batch size to %d", new_page_size)
        QMessageBox.information(self, "History list", f"The history list will now load {new_page_size} scans at a time.")
        self._on_data_changed()  # reloads the main window's history list at the new batch size immediately

    # -- Logging ------------------------------------------------------------------------------------

    def _build_logging_group(self) -> QGroupBox:
        group = QGroupBox("Logging")
        layout = QVBoxLayout(group)

        self.debug_logging_checkbox = QCheckBox("Enable verbose (debug) logging")
        self.debug_logging_checkbox.setChecked(get_debug_logging_enabled(self.conn))
        self.debug_logging_checkbox.toggled.connect(self._on_debug_logging_toggled)
        layout.addWidget(self.debug_logging_checkbox)

        info_label = QLabel(
            "Adds extra detail to the log file (see the log file location under the Data tab) -- "
            "capture-pipeline stage timings, slow camera reads, a full driver-reported snapshot "
            "(resolution, negotiated compression, exposure/gain/focus/etc., and every natively "
            "advertised mode) of each camera every time it's opened, reopened, or switched, and "
            "other diagnostic detail not needed for routine operation. Useful when reporting a bug "
            "or a slowdown; takes effect immediately, no restart needed. Leave off for normal "
            "day-to-day use."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        return group

    def _on_debug_logging_toggled(self, checked: bool) -> None:
        set_setting(self.conn, "debug_logging_enabled", "1" if checked else "0")
        logging.getLogger().setLevel(logging.DEBUG if checked else logging.INFO)
        # comtypes (pulled in transitively by pygrabber for native DirectShow format/device
        # enumeration) logs one line per raw COM call/pointer release at DEBUG -- internal plumbing
        # noise, not camera information -- which would otherwise bury capture/camera_manager.py's own
        # useful per-camera-open verbose dump; see main.py::apply_debug_logging_setting's matching
        # startup-time suppression, mirrored here since this toggle sets the level live without going
        # through that function.
        logging.getLogger("comtypes").setLevel(logging.WARNING)
        logger.info("Operator %s debug logging", "enabled" if checked else "disabled")

    # -- Backup -----------------------------------------------------------------------------------

    def _build_backup_group(self) -> QGroupBox:
        group = QGroupBox("Backup")
        layout = QVBoxLayout(group)

        info_label = QLabel(
            "Saves a single .zip file containing the database and every saved face/document image "
            "-- useful on a USB drive or network share as protection against this machine's drive "
            "failing or being lost, which BitLocker alone does not protect against."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        row = QHBoxLayout()
        backup_button = QPushButton("Backup now...")
        backup_button.clicked.connect(self._on_backup_now_clicked)
        row.addWidget(backup_button)
        row.addStretch(1)
        layout.addLayout(row)

        return group

    def _on_backup_now_clicked(self) -> None:
        default_name = f"kontroll_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        file_path, _ = QFileDialog.getSaveFileName(self, "Backup now", default_name, "Zip files (*.zip)")
        if not file_path:
            return

        # Zipping every face/document image can take a real, noticeable stretch on a large image
        # folder -- a busy cursor is the same minimal feedback used elsewhere in this window for a
        # synchronous operation with no natural progress-reporting point (see
        # _refresh_resolution_combo), so this doesn't look like the window has simply stopped
        # responding for however long the archive takes to write.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            create_backup_archive(
                self.conn, get_configured_dir(FACES_DIR_KEY), get_configured_dir(DOCUMENTS_DIR_KEY), Path(file_path)
            )
        except OSError:
            logger.warning("Backup to %s failed", file_path, exc_info=True)
            QMessageBox.warning(
                self, "Backup failed", f"Could not write the backup archive to:\n{file_path}\n\nSee the log for details."
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        logger.info("Operator created a backup archive at %s", file_path)
        QMessageBox.information(self, "Backup complete", f"Backup saved to:\n{file_path}")

    # -- Flagged Users ----------------------------------------------------------------------------

    def _build_flagged_users_tab(self) -> QWidget:
        """Build the Flagged Users tab: a manual add-entry form plus the current watch list.

        Entries can also be added directly from a scan's detail section (see
        ui/main_window.py::HistoryRowWidget._on_flag_person_clicked), prefilled from that scan's
        extracted fields -- this form is for adding someone the operator already knows about
        without needing a scan of them on hand first.
        """
        page = QWidget()
        layout = QVBoxLayout(page)

        info_label = QLabel(
            "When a new scan's extracted name or personal number matches an entry below, that "
            "scan's row is highlighted red in the history list to warn the operator, and a large "
            "warning appears immediately after capture. Adding an entry here also checks previously "
            "saved scans for a match and offers to flag any it finds."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        layout.addWidget(self._build_add_flagged_person_group())

        self.flagged_persons_list_layout = QVBoxLayout()
        layout.addLayout(self.flagged_persons_list_layout)
        self._refresh_flagged_persons_list()

        layout.addStretch(1)
        return page

    def _build_add_flagged_person_group(self) -> QGroupBox:
        group = QGroupBox("Add flagged person")
        form_layout = QVBoxLayout(group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.new_flagged_name_edit = QLineEdit()
        name_row.addWidget(self.new_flagged_name_edit, stretch=1)
        form_layout.addLayout(name_row)

        number_row = QHBoxLayout()
        number_row.addWidget(QLabel("Personal number:"))
        self.new_flagged_number_edit = QLineEdit()
        self.new_flagged_number_edit.setPlaceholderText("YYMMDD-XXXX")
        number_row.addWidget(self.new_flagged_number_edit, stretch=1)
        form_layout.addLayout(number_row)

        notes_row = QHBoxLayout()
        notes_row.addWidget(QLabel("Notes:"))
        self.new_flagged_notes_edit = QLineEdit()
        self.new_flagged_notes_edit.setPlaceholderText("Optional")
        notes_row.addWidget(self.new_flagged_notes_edit, stretch=1)
        form_layout.addLayout(notes_row)

        image_row = QHBoxLayout()
        self._new_flagged_image_path: str | None = None
        self.new_flagged_image_label = QLabel("No image selected")
        self.new_flagged_image_label.setStyleSheet("color: gray;")
        choose_image_button = QPushButton("Choose image...")
        choose_image_button.clicked.connect(self._on_choose_flagged_image_clicked)
        image_row.addWidget(choose_image_button)
        image_row.addWidget(self.new_flagged_image_label, stretch=1)
        form_layout.addLayout(image_row)

        add_row = QHBoxLayout()
        add_row.addStretch(1)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._on_add_flagged_person_clicked)
        add_row.addWidget(add_button)
        form_layout.addLayout(add_row)

        return group

    def _on_choose_flagged_image_clicked(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Choose reference image", "", "Images (*.jpg *.jpeg *.png)")
        if not file_path:
            return
        self._new_flagged_image_path = file_path
        self.new_flagged_image_label.setText(Path(file_path).name)

    def _on_add_flagged_person_clicked(self) -> None:
        name = self.new_flagged_name_edit.text().strip()
        personal_number = self.new_flagged_number_edit.text().strip()
        notes = self.new_flagged_notes_edit.text().strip()

        if not name and not personal_number:
            QMessageBox.warning(self, "Add flagged person", "Enter at least a name or a personal number.")
            return

        image_path: str | None = None
        if self._new_flagged_image_path is not None:
            source = Path(self._new_flagged_image_path)
            destination = get_flagged_images_dir() / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{source.name}"
            shutil.copyfile(source, destination)
            image_path = str(destination)

        flagged_person_id = insert_flagged_person(self.conn, name or None, personal_number or None, image_path, notes or None)
        logger.info("Operator manually added flagged person id=%d", flagged_person_id)

        self.new_flagged_name_edit.clear()
        self.new_flagged_number_edit.clear()
        self.new_flagged_notes_edit.clear()
        self._new_flagged_image_path = None
        self.new_flagged_image_label.setText("No image selected")
        self._refresh_flagged_persons_list()

        flagged_person = FlaggedPerson(
            id=flagged_person_id,
            created_at=datetime.now().isoformat(),
            name=name or None,
            personal_number=personal_number or None,
            image_path=image_path,
            notes=notes or None,
        )
        run_retroactive_flag_check(self.conn, flagged_person, self)
        self._on_data_changed()

    def refresh_flagged_persons_list(self) -> None:
        """Public entry point for MainWindow to refresh this tab after a flagged-persons change made elsewhere in the app (e.g. a scan's own "Remove Flag" detail button)."""
        self._refresh_flagged_persons_list()

    def _refresh_flagged_persons_list(self) -> None:
        """Rebuild the flagged-persons list from the database -- called on tab build and after any add/remove."""
        while self.flagged_persons_list_layout.count():
            item = self.flagged_persons_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        flagged_people = list_flagged_persons(self.conn)
        if not flagged_people:
            empty_label = QLabel("No flagged persons yet.")
            empty_label.setStyleSheet("color: gray;")
            self.flagged_persons_list_layout.addWidget(empty_label)
            return

        for flagged_person in flagged_people:
            self.flagged_persons_list_layout.addWidget(
                _FlaggedPersonRowWidget(self.conn, flagged_person, on_removed=self._on_flagged_person_removed)
            )

    def _on_flagged_person_removed(self) -> None:
        """After a flagged-persons entry is removed from this tab's own list, rebuild the list itself and tell MainWindow so the history list's badges/red-row tinting clear too."""
        self._refresh_flagged_persons_list()
        self._on_data_changed()

    # -- Live camera preview & diagnostics ------------------------------------------------------

    def _build_camera_diagnostics_group(self) -> QGroupBox:
        """Build the live preview + real-time health panel for both cameras.

        Deliberately reads no camera frames of its own -- see update_camera_diagnostics, which
        ui/main_window.py's own ~30fps preview timer calls once per tick with the very same frame
        it just read for its own preview. A second, independent capture.read() against the same
        VideoCapture handle from a timer/thread of this window's own would race the main window's
        (DirectShow handles aren't safe for concurrent reads), which is exactly the kind of
        hazard on_camera_probe_started/finished already exist elsewhere in this window to avoid.
        Starts out showing a placeholder in both columns; content only appears once this tab is
        actually the one showing (see update_camera_diagnostics's own early-return).
        """
        group = QGroupBox("Live preview && diagnostics")
        layout = QHBoxLayout(group)
        layout.addWidget(self._build_single_camera_diagnostics_column("face", "Visitor camera"))
        layout.addWidget(self._build_single_camera_diagnostics_column("doc", "Identification camera"))
        return group

    def _build_single_camera_diagnostics_column(self, side: str, title: str) -> QWidget:
        column_widget = QWidget()
        layout = QVBoxLayout(column_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel(f"<b>{title}</b>"))

        preview_label = QLabel("Preview starts\nwhile this tab is open")
        preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_label.setFixedSize(*_CAMERA_DIAGNOSTICS_PREVIEW_SIZE)
        preview_label.setStyleSheet("background-color: #000; color: #888; border: 1px solid #444;")
        layout.addWidget(preview_label)

        stats_label = QLabel("Waiting for data...")
        stats_label.setStyleSheet("color: gray;")
        stats_label.setWordWrap(True)
        stats_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(stats_label)

        export_button = QPushButton("Export diagnostics...")
        export_button.setToolTip("Dump this camera's full driver-reported configuration and supported resolutions, for troubleshooting")
        export_button.clicked.connect(lambda _checked, s=side: self._on_export_camera_diagnostics_clicked(s))
        layout.addWidget(export_button)

        if side == "face":
            self.face_diagnostics_preview_label = preview_label
            self.face_diagnostics_stats_label = stats_label
        else:
            self.doc_diagnostics_preview_label = preview_label
            self.doc_diagnostics_stats_label = stats_label

        return column_widget

    def update_camera_diagnostics(self, face_frame: np.ndarray | None, doc_frame: np.ndarray | None) -> None:
        """Refresh the Cameras tab's live preview + stats from a frame pair ui/main_window.py's own preview timer just read.

        Called once per preview tick from MainWindow._update_previews regardless of whether this
        window (or its Cameras tab specifically) is even visible -- this no-ops immediately unless
        it is, so the rest of the time it costs nothing beyond the one attribute-and-index check.
        See _build_camera_diagnostics_group's docstring for why frames are handed in rather than
        read here.
        """
        if not self.isVisible() or self.tab_widget.currentIndex() != self._cameras_tab_index:
            return
        self._update_one_camera_diagnostics(
            self.face_diagnostics_preview_label, self.face_diagnostics_stats_label, face_frame,
            self.camera_manager.get_face_camera_stats(),
        )
        self._update_one_camera_diagnostics(
            self.doc_diagnostics_preview_label, self.doc_diagnostics_stats_label, doc_frame,
            self.camera_manager.get_doc_camera_stats(),
        )

    def _update_one_camera_diagnostics(
        self, preview_label: QLabel, stats_label: QLabel, frame: np.ndarray | None, stats: CameraStats
    ) -> None:
        if frame is not None:
            shrunk = shrink_to_fit(frame, preview_label.width(), preview_label.height())
            pixmap = bgr_frame_to_pixmap(shrunk)
            preview_label.setStyleSheet("background-color: #000; border: 1px solid #444;")
            preview_label.setPixmap(
                pixmap.scaled(
                    preview_label.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
            )
        else:
            preview_label.clear()
            preview_label.setText("No signal" if stats.is_opened else "Offline")
            preview_label.setStyleSheet("background-color: #000; color: #FF5252; font-weight: bold; border: 1px solid #444;")
        stats_label.setText(_format_camera_stats_html(stats))

    def _on_export_camera_diagnostics_clicked(self, side: str) -> None:
        """Gather and show a full diagnostic dump for one camera -- driver-reported configuration, every probed resolution, and current runtime stats -- for the operator to read, copy, or save when troubleshooting/reporting an issue.

        Pauses the main window's live-preview poll for the same reason _refresh_resolution_combo
        does (gather_camera_diagnostics probes resolutions against the same already-open capture
        the preview timer reads from concurrently otherwise), and restores whichever resolution is
        actually configured afterward -- probing leaves the capture set to its last-tried candidate.
        """
        capture = self.camera_manager.face_capture if side == "face" else self.camera_manager.doc_capture
        camera_index = self.camera_manager.face_camera_index if side == "face" else self.camera_manager.doc_camera_index
        configured_resolution = get_face_camera_resolution(self.conn) if side == "face" else get_doc_camera_resolution(self.conn)

        self._on_camera_probe_started()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            diagnostics = gather_camera_diagnostics(
                capture, camera_index, on_candidate_tried=lambda _width, _height: QApplication.processEvents()
            )
        finally:
            if isinstance(configured_resolution, tuple):
                set_exact_resolution(capture, configured_resolution)
            elif configured_resolution == AUTO_LOWEST:
                set_lowest_available_resolution(capture)
            else:
                set_best_available_resolution(capture)
            QApplication.restoreOverrideCursor()
            self._on_camera_probe_finished()

        stats = self.camera_manager.get_face_camera_stats() if side == "face" else self.camera_manager.get_doc_camera_stats()
        side_label = "Visitor" if side == "face" else "Identification"
        report_text = _format_camera_diagnostics_report(side_label, diagnostics, configured_resolution, stats)
        logger.info("Operator exported %s camera (index %d) diagnostics", side_label, camera_index)

        dialog = _CameraDiagnosticsDialog(report_text, f"{side_label} Camera Diagnostics", self.conn, self)
        dialog.exec()

    # -- Camera assignment ----------------------------------------------------------------------

    def _build_camera_group(self) -> QGroupBox:
        """Build the camera-assignment group without touching camera hardware.

        Enumerating connected cameras (enumerate_cameras -- opens up to 10 device indices in
        turn) is a real, multi-second hardware round-trip, and this group used to build it
        unconditionally the moment Settings was first opened, regardless of which tab the operator
        actually landed on -- the single biggest cause of "Settings is slow to open". The combos
        below start out showing only the currently-assigned camera indices (no probe); the real
        device list is filled in by _populate_camera_assignment once the operator actually
        switches to this tab -- see _on_settings_tab_changed.
        """
        group = QGroupBox("Camera assignment")
        layout = QVBoxLayout(group)

        self.camera_detecting_label = QLabel("Camera list loads when this tab is opened.")
        self.camera_detecting_label.setStyleSheet("color: gray;")
        layout.addWidget(self.camera_detecting_label)

        face_row = QHBoxLayout()
        face_row.addWidget(QLabel("Visitor camera (left):"))
        self.face_camera_combo = self._build_camera_combo([], self.camera_manager.face_camera_index)
        self.face_camera_combo.currentIndexChanged.connect(self._on_face_camera_changed)
        face_row.addWidget(self.face_camera_combo, stretch=1)
        layout.addLayout(face_row)
        layout.addWidget(self._build_backend_row("face"))
        layout.addWidget(self._build_resolution_row("face"))
        layout.addWidget(self._build_exposure_row("face"))

        doc_row = QHBoxLayout()
        doc_row.addWidget(QLabel("Identification camera (right):"))
        self.doc_camera_combo = self._build_camera_combo([], self.camera_manager.doc_camera_index)
        self.doc_camera_combo.currentIndexChanged.connect(self._on_doc_camera_changed)
        doc_row.addWidget(self.doc_camera_combo, stretch=1)
        layout.addLayout(doc_row)
        layout.addWidget(self._build_backend_row("doc"))
        layout.addWidget(self._build_resolution_row("doc"))
        layout.addWidget(self._build_exposure_row("doc"))

        info_label = QLabel(
            "\"Auto (lowest available)\" probes the camera and always uses the lowest resolution "
            "it actually supports, for the lowest CPU/bandwidth cost -- the factory default for "
            "the Visitor camera, which doesn't need much detail for face matching. \"Auto (best "
            "available)\" does the same for the highest resolution -- the factory default for the "
            "Identification camera, since OCR/MRZ reading needs real pixel detail on the card's "
            "printed text; switch either camera to the other mode (or a specific resolution) if "
            "needed, e.g. if the live preview warns that a low resolution may affect ID document "
            "reading. Both \"Auto\" modes now prefer a resolution that keeps MJPG compression over "
            "a higher-resolution one that doesn't, and each resolution below shows what compression "
            "it actually negotiates. If a camera keeps falling back to an uncompressed format at "
            "every resolution despite working fine in other applications, try switching its Backend "
            "to Media Foundation above -- some camera/driver combinations negotiate compression far "
            "more reliably over it than the default DirectShow backend. Detecting the list below is "
            "usually near-instant; it falls back to a slower (couple-of-seconds-per-camera) hardware "
            "probe only if the fast method isn't available for this camera."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        return group

    def _populate_camera_assignment(self) -> None:
        """Actually probe connected camera devices and fill in the assignment combos -- see _build_camera_group's docstring for why this is deferred to here rather than run at window build time.

        Pauses the main window's live-preview poll for the same reason _refresh_resolution_combo
        does (see its docstring and on_camera_probe_started/finished) -- enumerate_cameras briefly
        opens a fresh handle to every device index it tries, including whichever ones the main
        preview loop already has open, which would otherwise read from the same hardware handle
        concurrently.
        """
        self._on_camera_probe_started()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            # Opens up to 10 device indices in turn (see enumerate_cameras' docstring) -- pumping
            # the event loop between each keeps this from reading as a freeze.
            cameras = enumerate_cameras(on_index_tried=lambda _index: QApplication.processEvents())
        finally:
            QApplication.restoreOverrideCursor()
            self._on_camera_probe_finished()

        self.camera_detecting_label.setVisible(False)
        self._fill_camera_combo(self.face_camera_combo, cameras, self.camera_manager.face_camera_index)
        self._fill_camera_combo(self.doc_camera_combo, cameras, self.camera_manager.doc_camera_index)

    def _build_resolution_row(self, side: str) -> QWidget:
        """Build one camera's resolution picker: a combo of resolutions actually supported by that device, plus a Refresh button to re-probe on demand (e.g. after swapping hardware without restarting).

        Populated cheaply (no hardware probe) at build time -- see _on_settings_tab_changed for
        where the real probe (a real hardware round-trip per candidate resolution) is deferred to.
        """
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 10)
        row.addWidget(QLabel("Resolution:"))

        combo = QComboBox()
        row.addWidget(combo, stretch=1)

        refresh_button = QPushButton("Refresh")
        refresh_button.setToolTip("Re-detect this camera's supported resolutions")
        refresh_button.clicked.connect(lambda _checked, s=side: self._refresh_resolution_combo(s))
        row.addWidget(refresh_button)

        if side == "face":
            self.face_resolution_combo = combo
        else:
            self.doc_resolution_combo = combo

        self._populate_resolution_combo_minimal(side)
        combo.currentIndexChanged.connect(lambda _list_index, s=side: self._on_resolution_changed(s))

        return row_widget

    def _populate_resolution_combo_minimal(self, side: str) -> None:
        """Populate a resolution combo with both "Auto" choices plus whatever's currently configured, without touching the camera hardware at all -- the instant-and-cheap path used when the row is first built."""
        combo = self.face_resolution_combo if side == "face" else self.doc_resolution_combo
        current_resolution = get_face_camera_resolution(self.conn) if side == "face" else get_doc_camera_resolution(self.conn)

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto (best available)", None)
        combo.addItem("Auto (lowest available)", AUTO_LOWEST)
        if isinstance(current_resolution, tuple):
            combo.addItem(f"{current_resolution[0]} × {current_resolution[1]}", current_resolution)
        combo.setCurrentIndex(combo.findData(current_resolution))
        combo.blockSignals(False)

    def _refresh_resolution_combo(self, side: str) -> None:
        """Repopulate the given camera's resolution combo, preserving the currently configured selection.

        Tries the fast path first: list_native_formats queries DirectShow's own advertised
        (compression, resolution) modes directly (a metadata call, no capture.read() at all -- see
        that function's own docstring), so resolutions_from_native_formats can usually build the
        combo's contents near-instantly instead of the previous unconditional ~1-2s real hardware
        round trip (probe_supported_resolutions, ~10 candidate resolutions each requiring a real
        capture.set()+read()). Only falls back to that slow probe if the fast metadata query itself
        comes back empty (e.g. pygrabber unavailable, or this device doesn't expose IAMStreamConfig)
        -- pumping the event loop between each of its ~10 candidates so the app stays responsive
        instead of reading as frozen, and pausing the live-preview poll for that same span since it
        would otherwise read from this same capture concurrently with the probe's own calls whenever
        a pumped event lets its timer fire.
        """
        combo = self.face_resolution_combo if side == "face" else self.doc_resolution_combo
        capture = self.camera_manager.face_capture if side == "face" else self.camera_manager.doc_capture
        camera_index = self.camera_manager.face_camera_index if side == "face" else self.camera_manager.doc_camera_index
        current_resolution = get_face_camera_resolution(self.conn) if side == "face" else get_doc_camera_resolution(self.conn)

        started_at = time.monotonic()
        resolutions = resolutions_from_native_formats(list_native_formats(camera_index))
        used_fast_path = bool(resolutions)

        if used_fast_path:
            logger.info(
                "Refreshed %s camera (index %d) resolution list from native DirectShow format "
                "metadata in %.2fs -- no hardware probe needed", side, camera_index, time.monotonic() - started_at,
            )
        else:
            self._on_camera_probe_started()
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                resolutions = probe_supported_resolutions(
                    capture, on_candidate_tried=lambda _width, _height: QApplication.processEvents()
                )
            finally:
                QApplication.restoreOverrideCursor()
                self._on_camera_probe_finished()
            logger.info(
                "Refreshed %s camera (index %d) resolution list via hardware probe in %.2fs (native "
                "format metadata was unavailable)", side, camera_index, time.monotonic() - started_at,
            )

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto (best available)", None)
        combo.addItem("Auto (lowest available)", AUTO_LOWEST)
        detected_sizes: set[tuple[int, int]] = set()
        for width, height, fourcc in resolutions:
            combo.addItem(f"{width} × {height} — {_resolution_compression_label(fourcc)}", (width, height))
            detected_sizes.add((width, height))
        if isinstance(current_resolution, tuple) and current_resolution not in detected_sizes:
            combo.addItem(f"{current_resolution[0]} × {current_resolution[1]} (not detected)", current_resolution)
        target_index = combo.findData(current_resolution)
        combo.setCurrentIndex(target_index if target_index >= 0 else 0)
        combo.blockSignals(False)

        # The slow probe above leaves the capture set to whichever candidate it tried last -- restore
        # it to whatever is actually configured so the live preview doesn't stay on a stray mode.
        # Not needed on the fast metadata path, which never touches the live capture at all.
        if not used_fast_path:
            if isinstance(current_resolution, tuple):
                set_exact_resolution(capture, current_resolution)
            elif current_resolution == AUTO_LOWEST:
                set_lowest_available_resolution(capture)
            else:
                set_best_available_resolution(capture)

    def _on_resolution_changed(self, side: str) -> None:
        combo = self.face_resolution_combo if side == "face" else self.doc_resolution_combo
        resolution = combo.currentData()
        setting_key = "face_camera_resolution" if side == "face" else "doc_camera_resolution"
        set_setting(self.conn, setting_key, resolution_to_setting_value(resolution))

        if side == "face":
            applied = self.camera_manager.set_face_camera_resolution(resolution)
        else:
            applied = self.camera_manager.set_doc_camera_resolution(resolution)
        logger.info("Operator set the %s camera resolution to %s (applied %s)", side, resolution or "auto", applied)

        # An exact resolution silently falls back to auto-best inside set_exact_resolution when
        # the camera doesn't honor it (some hardware/drivers ignore CAP_PROP_FRAME_WIDTH/HEIGHT
        # entirely and always deliver one fixed native mode) -- that fallback is otherwise only
        # logged, which looks to an operator like their choice was ignored for no reason. Surface
        # it here instead of leaving it silent.
        if isinstance(resolution, tuple) and applied is not None and applied != resolution:
            QMessageBox.warning(
                self,
                "Resolution not supported",
                f"This camera did not accept {resolution[0]} × {resolution[1]} and is still "
                f"using {applied[0]} × {applied[1]} instead. Click Refresh above to see the "
                "resolutions this camera actually supports.",
            )

    # Slider ranges for manual exposure/brightness -- DirectShow's own native units, not a
    # normalized 0-1 scale (see capture.camera_manager.ExposureSettings' docstring for why there's
    # no reliable way to query a given camera's actual supported range up front). Exposure -13..-1
    # is the common UVC/DirectShow log2-seconds convention (each step roughly halves/doubles
    # exposure time); brightness 0-255 matches the mid-range default already used elsewhere in this
    # app (config.settings.DEFAULT_CAMERA_BRIGHTNESS) and this app's own production camera
    # diagnostics export. Neither range is calibrated per-device -- same unvalidated-default caveat
    # as DEFAULT_FACE_MATCH_THRESHOLD; a camera/driver that clamps or ignores an out-of-range or
    # unsupported value simply behaves as it did before, same accepted-limitation pattern as
    # set_exact_resolution's own resolution fallback.
    _EXPOSURE_SLIDER_RANGE = (-13, -1)
    _BRIGHTNESS_SLIDER_RANGE = (0, 255)

    def _build_exposure_row(self, side: str) -> QWidget:
        """Build one camera's exposure control: an "Auto exposure" checkbox (on by default, matching this app's factory default) plus manual exposure/brightness sliders that only take effect -- and are only enabled -- while auto is unchecked."""
        conn = self.conn
        auto = get_face_camera_auto_exposure(conn) if side == "face" else get_doc_camera_auto_exposure(conn)
        exposure = get_face_camera_exposure(conn) if side == "face" else get_doc_camera_exposure(conn)
        brightness = get_face_camera_brightness(conn) if side == "face" else get_doc_camera_brightness(conn)

        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 10)

        auto_checkbox = QCheckBox("Auto exposure")
        auto_checkbox.setChecked(auto)
        row.addWidget(auto_checkbox)

        row.addWidget(QLabel("Exposure:"))
        exposure_slider = QSlider(Qt.Orientation.Horizontal)
        exposure_slider.setRange(*self._EXPOSURE_SLIDER_RANGE)
        exposure_slider.setValue(exposure)
        exposure_slider.setEnabled(not auto)
        row.addWidget(exposure_slider, stretch=1)

        row.addWidget(QLabel("Brightness:"))
        brightness_slider = QSlider(Qt.Orientation.Horizontal)
        brightness_slider.setRange(*self._BRIGHTNESS_SLIDER_RANGE)
        brightness_slider.setValue(brightness)
        brightness_slider.setEnabled(not auto)
        row.addWidget(brightness_slider, stretch=1)

        if side == "face":
            self.face_auto_exposure_checkbox = auto_checkbox
            self.face_exposure_slider = exposure_slider
            self.face_brightness_slider = brightness_slider
        else:
            self.doc_auto_exposure_checkbox = auto_checkbox
            self.doc_exposure_slider = exposure_slider
            self.doc_brightness_slider = brightness_slider

        auto_checkbox.toggled.connect(lambda _checked, s=side: self._on_exposure_changed(s))
        exposure_slider.valueChanged.connect(lambda _value, s=side: self._on_exposure_changed(s))
        brightness_slider.valueChanged.connect(lambda _value, s=side: self._on_exposure_changed(s))

        return row_widget

    def _on_exposure_changed(self, side: str) -> None:
        """Persist and apply this camera's exposure settings whenever the auto checkbox or either slider changes -- mirrors _on_resolution_changed's wiring."""
        auto_checkbox = self.face_auto_exposure_checkbox if side == "face" else self.doc_auto_exposure_checkbox
        exposure_slider = self.face_exposure_slider if side == "face" else self.doc_exposure_slider
        brightness_slider = self.face_brightness_slider if side == "face" else self.doc_brightness_slider

        auto = auto_checkbox.isChecked()
        exposure_slider.setEnabled(not auto)
        brightness_slider.setEnabled(not auto)

        setting_prefix = "face_camera" if side == "face" else "doc_camera"
        set_setting(self.conn, f"{setting_prefix}_auto_exposure", "1" if auto else "0")
        set_setting(self.conn, f"{setting_prefix}_exposure", str(exposure_slider.value()))
        set_setting(self.conn, f"{setting_prefix}_brightness", str(brightness_slider.value()))

        settings = ExposureSettings(auto=auto, exposure=exposure_slider.value(), brightness=brightness_slider.value())
        if side == "face":
            self.camera_manager.set_face_camera_exposure(settings)
        else:
            self.camera_manager.set_doc_camera_exposure(settings)
        logger.info("Operator set the %s camera exposure to %s", side, settings)

    _BACKEND_LABELS: dict[str, str] = {"dshow": "DirectShow (default)", "msmf": "Media Foundation"}

    def _build_backend_row(self, side: str) -> QWidget:
        """Build one camera's capture-backend picker -- DirectShow (the app-wide default) or Media Foundation, an opt-in alternative for a camera/driver combination that negotiates MJPG more reliably over it than DirectShow does (see capture/camera_manager.py::_open_camera's docstring for the diagnosed real-world case, and its caveat about device-index numbering not always agreeing between the two)."""
        current_backend = get_face_camera_backend(self.conn) if side == "face" else get_doc_camera_backend(self.conn)

        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 10)
        row.addWidget(QLabel("Backend:"))

        combo = QComboBox()
        for backend_key, label in self._BACKEND_LABELS.items():
            combo.addItem(label, CAMERA_BACKENDS[backend_key])
        combo.setCurrentIndex(combo.findData(current_backend))
        row.addWidget(combo, stretch=1)

        if side == "face":
            self.face_backend_combo = combo
        else:
            self.doc_backend_combo = combo
        combo.currentIndexChanged.connect(lambda _list_index, s=side: self._on_backend_changed(s))

        return row_widget

    def _on_backend_changed(self, side: str) -> None:
        """Persist and apply this camera's capture-backend choice -- a full release/reopen under the newly chosen backend (see CameraManager.set_face_camera_backend/set_doc_camera_backend), since a capture handle's backend can't be changed in place.

        Runs on a background thread (see _BackendSwitchWorker) rather than blocking this method's
        own (UI) thread: this operation can legitimately take several real seconds (a deliberate
        settling delay plus a few retries, see _open_camera_verified) -- diagnosed against a real
        operator report of the whole app freezing for a long time on exactly this switch. The combo
        is disabled and the live preview paused for the duration (see _on_camera_probe_started),
        both lifted once _finish_backend_switch runs with the result.
        """
        if self._backend_switch_thread is not None:
            return  # a switch (for either camera) is already running -- ignore a second trigger
        combo = self.face_backend_combo if side == "face" else self.doc_backend_combo
        requested_backend = combo.currentData()

        combo.setEnabled(False)
        self._on_camera_probe_started()
        # A wait cursor here is a "something is happening" hint, not an indication the app is
        # blocked -- unlike every other wait-cursor use in this window, this one is purely cosmetic,
        # since the operation itself now runs on a background thread and the rest of the app (and
        # this combo's own sibling controls) stays fully interactive underneath it.
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        thread = QThread(self)
        worker = _BackendSwitchWorker(self.camera_manager, side, requested_backend)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _cleanup() -> None:
            thread.quit()
            thread.wait()
            self._backend_switch_thread = None
            self._backend_switch_worker = None

        def _handle_finished(succeeded: bool, effective_backend: int) -> None:
            _cleanup()
            QApplication.restoreOverrideCursor()
            self._on_camera_probe_finished()
            combo.setEnabled(True)
            self._finish_backend_switch(side, combo, requested_backend, succeeded, effective_backend)

        worker.finished.connect(_handle_finished)
        self._backend_switch_thread = thread
        self._backend_switch_worker = worker
        thread.start()

    def _finish_backend_switch(
        self, side: str, combo: QComboBox, requested_backend: int, succeeded: bool, effective_backend: int
    ) -> None:
        """Persist/report the result of a backend switch once _BackendSwitchWorker's background thread has produced one -- see _on_backend_changed's docstring.

        On failure, auto-reverts to whichever backend was working before rather than leaving the
        camera stuck non-functional (the previously-diagnosed "won't come back online after
        switching to Media Foundation and back" bug) -- so the combo/setting persisted here always
        reflect the backend actually now in effect, which may not be the one the operator picked.
        """
        setting_key = "face_camera_backend" if side == "face" else "doc_camera_backend"
        effective_key = next(key for key, value in CAMERA_BACKENDS.items() if value == effective_backend)
        set_setting(self.conn, setting_key, effective_key)
        requested_label = self._BACKEND_LABELS[next(k for k, v in CAMERA_BACKENDS.items() if v == requested_backend)]
        logger.info(
            "Operator set the %s camera backend to %s (succeeded=%s, now using %s)",
            side, requested_label, succeeded, self._BACKEND_LABELS[effective_key],
        )

        if not succeeded:
            if effective_backend != requested_backend:
                combo.blockSignals(True)
                combo.setCurrentIndex(combo.findData(effective_backend))
                combo.blockSignals(False)
                QMessageBox.warning(
                    self,
                    "Backend not available",
                    f"This camera failed to come online under {requested_label} and has been "
                    f"switched back to {self._BACKEND_LABELS[effective_key]} automatically.",
                )
            else:
                QMessageBox.warning(
                    self,
                    "Backend not available",
                    f"This camera failed to come online under {requested_label}, and reverting "
                    "also failed -- check that it's still connected.",
                )

        # Re-probing the resolution list matters here: a different backend can expose a different
        # set of supported resolutions/compression combinations for the same physical camera (the
        # whole reason this setting exists), and the combo built from the *previous* backend's
        # probe would otherwise silently go stale.
        self._refresh_resolution_combo(side)

    def _build_camera_combo(self, cameras: list[CameraInfo], current_index: int) -> QComboBox:
        combo = QComboBox()
        self._fill_camera_combo(combo, cameras, current_index)
        return combo

    def _fill_camera_combo(self, combo: QComboBox, cameras: list[CameraInfo], current_index: int) -> None:
        """(Re)populate an existing camera-assignment combo, preserving the given current_index selection.

        Signals are blocked while repopulating -- both the initial (empty-list) build and a later
        refill from _populate_camera_assignment go through here, and neither should fire
        _on_face_camera_changed/_on_doc_camera_changed (which would otherwise try to actually
        switch the live camera handle mid-repopulation).
        """
        combo.blockSignals(True)
        combo.clear()
        for camera in cameras:
            combo.addItem(camera.label, camera.index)
        # The currently assigned camera might not have been detected this run (e.g. temporarily
        # unplugged, or not probed yet -- see _build_camera_group) -- add it anyway so the combo
        # reflects the persisted setting instead of silently jumping to whatever camera happens to
        # be first in the list.
        if current_index not in [camera.index for camera in cameras]:
            combo.addItem(f"Camera {current_index} (not detected)", current_index)
        combo.setCurrentIndex(combo.findData(current_index))
        combo.blockSignals(False)

    def _on_face_camera_changed(self, _list_index: int) -> None:
        new_index = self.face_camera_combo.currentData()
        set_setting(self.conn, "face_camera_index", str(new_index))
        self.camera_manager.switch_face_camera(new_index, get_face_camera_resolution(self.conn))
        logger.info("Operator reassigned the visitor (left) camera to index %d", new_index)
        # A different physical device may support an entirely different set of resolutions --
        # re-probe rather than leaving the combo showing the previous camera's list.
        self._refresh_resolution_combo("face")

    def _on_doc_camera_changed(self, _list_index: int) -> None:
        new_index = self.doc_camera_combo.currentData()
        set_setting(self.conn, "doc_camera_index", str(new_index))
        self.camera_manager.switch_doc_camera(new_index, get_doc_camera_resolution(self.conn))
        logger.info("Operator reassigned the identification (right) camera to index %d", new_index)
        self._refresh_resolution_combo("doc")

    # -- Face match threshold -------------------------------------------------------------------

    def _build_face_match_threshold_group(self) -> QGroupBox:
        group = QGroupBox("Face match threshold")
        outer_layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Minimum similarity score to count as a match:"))

        self.face_match_threshold_spin = QDoubleSpinBox()
        self.face_match_threshold_spin.setRange(0.0, 1.0)
        self.face_match_threshold_spin.setSingleStep(0.01)
        self.face_match_threshold_spin.setDecimals(2)
        self.face_match_threshold_spin.setValue(get_face_match_threshold(self.conn))
        self.face_match_threshold_spin.valueChanged.connect(self._on_face_match_threshold_spin_changed)
        row.addWidget(self.face_match_threshold_spin)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save_face_match_threshold_clicked)
        row.addWidget(save_button)
        row.addStretch(1)
        outer_layout.addLayout(row)

        self.face_match_threshold_band_label = QLabel()
        self.face_match_threshold_band_label.setStyleSheet("font-weight: bold;")
        outer_layout.addWidget(self.face_match_threshold_band_label)
        self._update_face_match_threshold_band_label(self.face_match_threshold_spin.value())

        info_label = QLabel(
            "This is the minimum cosine-similarity score between the live face capture and the "
            "document photo for an automatic \"match\". Lower values are more lenient (fewer "
            "false rejects, but more risk of accepting a non-match); higher values are stricter "
            "(fewer false accepts, but more risk of rejecting a genuine match).\n\n"
            "0.00–0.25: Very lenient — likely to accept lookalikes.\n"
            "0.25–0.45: Typical range for this model (default 0.38); not yet calibrated "
            "against real captures at this desk.\n"
            "0.45–0.60: Strict — may reject some genuine matches, especially in poor lighting.\n"
            "Above 0.60: Very strict — expect frequent false rejects.\n\n"
            "Whatever the automatic result, the raw score is always shown alongside it and an "
            "operator can override it from the history list."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        outer_layout.addWidget(info_label)

        calibrate_row = QHBoxLayout()
        calibrate_button = QPushButton("Calibrate...")
        calibrate_button.clicked.connect(self._on_calibrate_clicked)
        calibrate_row.addWidget(calibrate_button)
        calibrate_row.addWidget(QLabel("Measure real similarity scores from known-good pairs to pick a value above."))
        calibrate_row.addStretch(1)
        outer_layout.addLayout(calibrate_row)

        return group

    def _on_calibrate_clicked(self) -> None:
        """Open the calibration window, creating it once and reusing it on subsequent clicks."""
        logger.info("Operator opened the face match calibration window")
        if self._calibration_window is None:
            self._calibration_window = CalibrationWindow(
                self.camera_manager,
                self.face_analysis_app,
                on_apply_threshold=self._on_calibration_threshold_applied,
                conn=self.conn,
            )
        self._calibration_window.show()
        self._calibration_window.raise_()
        self._calibration_window.activateWindow()

    def _on_calibration_threshold_applied(self, new_threshold: float) -> None:
        """Persist a threshold suggested by the calibration window and reflect it in this window's own spinbox."""
        set_setting(self.conn, "face_match_threshold", str(new_threshold))
        logger.info("Operator applied a calibration-suggested face match threshold of %.4f", new_threshold)
        self.face_match_threshold_spin.setValue(new_threshold)

    def _update_face_match_threshold_band_label(self, value: float) -> None:
        if value < 0.25:
            band = "Very lenient"
        elif value < 0.45:
            band = "Typical range"
        elif value < 0.60:
            band = "Strict"
        else:
            band = "Very strict"
        self.face_match_threshold_band_label.setText(f"Current setting: {value:.2f} — {band}")

    def _on_face_match_threshold_spin_changed(self, value: float) -> None:
        self._update_face_match_threshold_band_label(value)

    def _on_save_face_match_threshold_clicked(self) -> None:
        new_threshold = self.face_match_threshold_spin.value()
        set_setting(self.conn, "face_match_threshold", str(new_threshold))
        logger.info("Operator set the face match threshold to %.2f", new_threshold)
        QMessageBox.information(self, "Face match threshold", f"Face match threshold set to {new_threshold:.2f}.")

    # -- Barcode/OCR overlap threshold ------------------------------------------------------------

    def _build_barcode_overlap_threshold_group(self) -> QGroupBox:
        group = QGroupBox("Barcode/OCR overlap threshold")
        outer_layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.addWidget(QLabel("Minimum OCR/barcode word-overlap ratio to count as valid:"))

        self.barcode_overlap_threshold_spin = QDoubleSpinBox()
        self.barcode_overlap_threshold_spin.setRange(0.0, 1.0)
        self.barcode_overlap_threshold_spin.setSingleStep(0.01)
        self.barcode_overlap_threshold_spin.setDecimals(2)
        self.barcode_overlap_threshold_spin.setValue(get_barcode_overlap_threshold(self.conn))
        row.addWidget(self.barcode_overlap_threshold_spin)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save_barcode_overlap_threshold_clicked)
        row.addWidget(save_button)
        row.addStretch(1)
        outer_layout.addLayout(row)

        info_label = QLabel(
            "Only used for a document that has a decoded PDF417 barcode but no MRZ (mainly EU "
            "driving licences -- see README Known limitations: field-level barcode parsing isn't "
            "implemented, so this word-overlap ratio is the only cross-check available for those). "
            "This is the fraction of OCR-read words that also appear in the raw barcode text; below "
            "this ratio, the document's validation_status is 'review' instead of 'valid'. Not yet "
            "calibrated against real captures at this desk (default 0.50) -- lower values are more "
            "lenient (more likely 'valid' despite a noisy OCR read), higher values are stricter."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        outer_layout.addWidget(info_label)

        calibrate_row = QHBoxLayout()
        calibrate_button = QPushButton("Calibrate...")
        calibrate_button.clicked.connect(self._on_barcode_calibrate_clicked)
        calibrate_row.addWidget(calibrate_button)
        calibrate_row.addWidget(QLabel("Measure real overlap ratios from real document captures to pick a value above."))
        calibrate_row.addStretch(1)
        outer_layout.addLayout(calibrate_row)

        return group

    def _on_barcode_calibrate_clicked(self) -> None:
        """Open the barcode/OCR calibration window, creating it once and reusing it on subsequent clicks."""
        logger.info("Operator opened the barcode/OCR overlap calibration window")
        if self._barcode_calibration_window is None:
            self._barcode_calibration_window = BarcodeCalibrationWindow(
                self.camera_manager,
                on_apply_threshold=self._on_barcode_calibration_threshold_applied,
                conn=self.conn,
            )
        self._barcode_calibration_window.show()
        self._barcode_calibration_window.raise_()
        self._barcode_calibration_window.activateWindow()

    def _on_barcode_calibration_threshold_applied(self, new_threshold: float) -> None:
        """Persist a threshold suggested by the calibration window and reflect it in this window's own spinbox."""
        set_setting(self.conn, "barcode_overlap_threshold", str(new_threshold))
        logger.info("Operator applied a calibration-suggested barcode/OCR overlap threshold of %.4f", new_threshold)
        self.barcode_overlap_threshold_spin.setValue(new_threshold)

    def _on_save_barcode_overlap_threshold_clicked(self) -> None:
        new_threshold = self.barcode_overlap_threshold_spin.value()
        set_setting(self.conn, "barcode_overlap_threshold", str(new_threshold))
        logger.info("Operator set the barcode/OCR overlap threshold to %.2f", new_threshold)
        QMessageBox.information(
            self, "Barcode/OCR overlap threshold", f"Barcode/OCR overlap threshold set to {new_threshold:.2f}."
        )

    # -- Capture shortcut -----------------------------------------------------------------------

    def _build_shortcut_group(self) -> QGroupBox:
        group = QGroupBox("Capture keyboard shortcut")
        row = QHBoxLayout(group)
        row.addWidget(QLabel("Shortcut:"))

        self.shortcut_edit = QKeySequenceEdit(QKeySequence(get_capture_shortcut(self.conn)))
        row.addWidget(self.shortcut_edit, stretch=1)

        save_button = QPushButton("Save")
        save_button.clicked.connect(self._on_save_shortcut_clicked)
        row.addWidget(save_button)

        return group

    def _on_save_shortcut_clicked(self) -> None:
        sequence = self.shortcut_edit.keySequence()
        if sequence.isEmpty():
            QMessageBox.warning(self, "Capture shortcut", "Enter a key combination first.")
            return

        sequence_text = sequence.toString()
        set_setting(self.conn, "capture_shortcut", sequence_text)
        self._on_capture_shortcut_changed(sequence_text)
        logger.info("Operator changed the capture keyboard shortcut to %s", sequence_text)
        QMessageBox.information(self, "Capture shortcut", f"Capture shortcut set to {sequence_text}.")

    # -- Prerequisites ----------------------------------------------------------------------------

    def _build_prerequisites_group(self) -> QGroupBox:
        """Build the prerequisites checklist.

        The Cameras row is added as a "Checking..." placeholder rather than actually checked here
        -- check_cameras() opens/reads/releases up to 10 device indices, a real multi-second
        hardware round-trip (same cost as the Cameras tab's own enumerate_cameras(), see
        _build_camera_group), and this group is built unconditionally the moment Settings is first
        opened regardless of which tab the operator lands on. It's filled in the first time the
        Prerequisites tab is actually shown instead -- see _on_settings_tab_changed.
        """
        group = QGroupBox("Prerequisites")
        layout = QVBoxLayout(group)

        for status in _check_non_camera_prerequisites():
            self._add_prerequisite_row(layout, status)
        self._add_prerequisite_placeholder_row(layout, "Cameras")

        return group

    def _add_prerequisite_row(self, layout: QVBoxLayout, status: PrerequisiteStatus) -> None:
        row = QHBoxLayout()
        row.addWidget(QLabel(status.name), stretch=1)

        status_label = QLabel()
        row.addWidget(status_label)

        info_button = QPushButton()
        info_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        info_button.setFixedSize(22, 22)
        info_button.setToolTip("How to install")
        info_button.clicked.connect(lambda _checked, name=status.name: self._on_prerequisite_info_clicked(name))
        row.addWidget(info_button)

        self._prerequisite_rows[status.name] = (status_label, info_button)
        layout.addLayout(row)
        self._apply_prerequisite_status(status)

    def _add_prerequisite_placeholder_row(self, layout: QVBoxLayout, name: str) -> None:
        """Add a row showing "Checking..." rather than a real satisfied/not-satisfied verdict, for a prerequisite whose real check is deferred -- see _build_prerequisites_group's docstring."""
        row = QHBoxLayout()
        row.addWidget(QLabel(name), stretch=1)

        status_label = QLabel("Checking...")
        status_label.setStyleSheet("color: gray;")
        row.addWidget(status_label)

        info_button = QPushButton()
        info_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxInformation))
        info_button.setFixedSize(22, 22)
        info_button.setToolTip("How to install")
        info_button.setVisible(False)
        info_button.clicked.connect(lambda _checked, n=name: self._on_prerequisite_info_clicked(n))
        row.addWidget(info_button)

        self._prerequisite_rows[name] = (status_label, info_button)
        layout.addLayout(row)

    def _apply_prerequisite_status(self, status: PrerequisiteStatus) -> None:
        status_label, info_button = self._prerequisite_rows[status.name]
        status_label.setText("Installed" if status.satisfied else "Not installed")
        status_label.setStyleSheet(f"color: {'#00C853' if status.satisfied else '#FF1744'}; font-weight: bold;")
        info_button.setVisible(not status.satisfied)

    def _refresh_prerequisites(self) -> None:
        """Re-run every prerequisite check and update the list -- used after an install action, where the state may have just changed."""
        for status in _check_all_prerequisites():
            self._apply_prerequisite_status(status)

    def _on_prerequisite_info_clicked(self, prerequisite_name: str) -> None:
        status = next(status for status in _check_all_prerequisites() if status.name == prerequisite_name)
        # can_auto_install can be False even for an item with a normally-available installer --
        # e.g. "Python packages" when running as a frozen EXE, where there's no venv/pip to
        # install into -- so it must gate the lookup, not just presence in the installers dict.
        installer = _PREREQUISITE_INSTALLERS.get(prerequisite_name) if status.can_auto_install else None
        info_window = PrerequisiteInstallInfoWindow(
            status, installer, on_installed=self._refresh_prerequisites, conn=self.conn, parent=self
        )
        info_window.exec()
