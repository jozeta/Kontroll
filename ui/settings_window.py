"""The Settings window: manual scan purge, camera-to-side assignment, the capture keyboard
shortcut, and a live prerequisites checklist with per-item install instructions."""

import logging
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

from insightface.app import FaceAnalysis
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStyle,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import (
    AUTO_LOWEST,
    CameraInfo,
    CameraManager,
    enumerate_cameras,
    probe_supported_resolutions,
    set_best_available_resolution,
    set_exact_resolution,
    set_lowest_available_resolution,
)
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
    get_disk_space_warning_mb,
    get_doc_camera_resolution,
    get_face_camera_resolution,
    get_face_match_threshold,
    get_retention_days,
    get_theme,
    resolution_to_setting_value,
    set_setting,
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
)
from storage.retention import purge_scans_older_than_days
from ui.barcode_calibration_window import BarcodeCalibrationWindow
from ui.calibration_window import CalibrationWindow
from ui.flag_retroactive import run_retroactive_flag_check
from ui.image_viewer import ClickableThumbnailLabel
from ui.theme import THEME_CHOICES, THEME_DARK, THEME_LIGHT, THEME_SYSTEM, apply_theme

logger = logging.getLogger(__name__)

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


def _check_all_prerequisites() -> list[PrerequisiteStatus]:
    """Run every prerequisite check fresh (not cached), so the settings window always reflects current machine state."""
    return [
        check_python_packages(),
        check_tesseract(),
        check_ocr_language_packs(),
        check_insightface_model(),
        check_cameras(),
    ]


class PrerequisiteInstallInfoWindow(QDialog):
    """A small modal window explaining how to satisfy one missing prerequisite, with a one-click installer when available."""

    def __init__(
        self,
        status: PrerequisiteStatus,
        installer: Callable[[], bool] | None,
        on_installed: Callable[[], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._installer = installer
        self._on_installed = on_installed
        self._status_name = status.name

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


class _FlaggedPersonRowWidget(QFrame):
    """One row in the Flagged Users list: reference thumbnail (if any), name/personal number/notes, and a Remove button."""

    def __init__(self, conn: sqlite3.Connection, flagged_person: FlaggedPerson, on_removed: Callable[[], None], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.conn = conn
        self.flagged_person = flagged_person
        self._on_removed = on_removed
        self.setFrameShape(QFrame.Shape.StyledPanel)

        row = QHBoxLayout(self)

        thumbnail_label = ClickableThumbnailLabel(flagged_person.image_path, flagged_person.name or "Flagged person")
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
        row.addLayout(info_column, stretch=1)

        remove_button = QPushButton("Remove")
        remove_button.setStyleSheet("color: #FF1744;")
        remove_button.clicked.connect(self._on_remove_clicked)
        row.addWidget(remove_button)

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

        self.setWindowTitle("Settings")
        self._apply_responsive_size()
        self._build_layout()

    def _apply_responsive_size(self) -> None:
        """Size the window generously but never larger than the available screen, and let it be resized down further -- tabs + per-tab scroll areas (see _build_scrollable_tab) mean no control ever gets stranded off-screen on a small/low-resolution display."""
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        width = min(560, available.width() - 60) if available is not None else 560
        height = min(620, available.height() - 80) if available is not None else 620
        self.resize(max(width, 360), max(height, 320))
        self.setMinimumSize(360, 320)

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
        self.tab_widget.addTab(self._build_scrollable_tab(self._build_prerequisites_tab()), "Prerequisites")

        # Detecting a camera's supported resolutions is a real hardware round-trip per candidate
        # (~1-2s per camera) -- doing that for both cameras unconditionally while building this
        # window (i.e. on every "Settings" click, since it's a singleton rebuilt only once per
        # session) made opening it noticeably slow. Deferred to the first time the operator
        # actually looks at the Cameras tab instead; _build_cameras_tab only does the cheap,
        # instant part (current setting shown, nothing probed) up front.
        self._resolutions_probed = False
        self.tab_widget.currentChanged.connect(self._on_settings_tab_changed)

    def _on_settings_tab_changed(self, index: int) -> None:
        if index == self._cameras_tab_index and not self._resolutions_probed:
            self._resolutions_probed = True
            self._refresh_resolution_combo("face")
            self._refresh_resolution_combo("doc")

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
        layout.addStretch(1)
        return page

    def _build_cameras_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
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

    # -- Camera assignment ----------------------------------------------------------------------

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Camera assignment")
        layout = QVBoxLayout(group)
        # Opens up to 10 device indices in turn (see enumerate_cameras' docstring) -- pumping the
        # event loop between each keeps this window's first build from reading as a freeze.
        cameras = enumerate_cameras(on_index_tried=lambda _index: QApplication.processEvents())

        face_row = QHBoxLayout()
        face_row.addWidget(QLabel("Visitor camera (left):"))
        self.face_camera_combo = self._build_camera_combo(cameras, self.camera_manager.face_camera_index)
        self.face_camera_combo.currentIndexChanged.connect(self._on_face_camera_changed)
        face_row.addWidget(self.face_camera_combo, stretch=1)
        layout.addLayout(face_row)
        layout.addWidget(self._build_resolution_row("face"))

        doc_row = QHBoxLayout()
        doc_row.addWidget(QLabel("Identification camera (right):"))
        self.doc_camera_combo = self._build_camera_combo(cameras, self.camera_manager.doc_camera_index)
        self.doc_camera_combo.currentIndexChanged.connect(self._on_doc_camera_changed)
        doc_row.addWidget(self.doc_camera_combo, stretch=1)
        layout.addLayout(doc_row)
        layout.addWidget(self._build_resolution_row("doc"))

        info_label = QLabel(
            "\"Auto (lowest available)\" probes the camera and always uses the lowest resolution "
            "it actually supports, for the lowest CPU/bandwidth cost -- the factory default for "
            "the Visitor camera, which doesn't need much detail for face matching. \"Auto (best "
            "available)\" does the same for the highest resolution -- the factory default for the "
            "Identification camera, since OCR/MRZ reading needs real pixel detail on the card's "
            "printed text; switch either camera to the other mode (or a specific resolution) if "
            "needed, e.g. if the live preview warns that a low resolution may affect ID document "
            "reading. Detecting the list below takes a couple of seconds per camera."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: gray;")
        layout.addWidget(info_label)

        return group

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
        """Probe the given camera's actually-supported resolutions (a real hardware round-trip, ~1-2s) and repopulate its combo, preserving the currently configured selection.

        Pumps the event loop between each of the ~10 candidates tried (rather than blocking the
        whole ~1-2s straight through) so the app stays responsive and doesn't read as frozen --
        the live-preview poll is paused for the same span, since it would otherwise read from this
        same capture concurrently with the probe's own capture.set()/.read() calls whenever a
        pumped event lets its timer fire.
        """
        combo = self.face_resolution_combo if side == "face" else self.doc_resolution_combo
        capture = self.camera_manager.face_capture if side == "face" else self.camera_manager.doc_capture
        current_resolution = get_face_camera_resolution(self.conn) if side == "face" else get_doc_camera_resolution(self.conn)

        self._on_camera_probe_started()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            resolutions = probe_supported_resolutions(
                capture, on_candidate_tried=lambda _width, _height: QApplication.processEvents()
            )
        finally:
            QApplication.restoreOverrideCursor()
            self._on_camera_probe_finished()

        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Auto (best available)", None)
        combo.addItem("Auto (lowest available)", AUTO_LOWEST)
        for width, height in resolutions:
            combo.addItem(f"{width} × {height}", (width, height))
        if isinstance(current_resolution, tuple) and current_resolution not in resolutions:
            combo.addItem(f"{current_resolution[0]} × {current_resolution[1]} (not detected)", current_resolution)
        target_index = combo.findData(current_resolution)
        combo.setCurrentIndex(target_index if target_index >= 0 else 0)
        combo.blockSignals(False)

        # Probing above leaves the capture set to whichever candidate it tried last -- restore it
        # to whatever is actually configured so the live preview doesn't stay on a stray mode.
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

    def _build_camera_combo(self, cameras: list[CameraInfo], current_index: int) -> QComboBox:
        combo = QComboBox()
        for camera in cameras:
            combo.addItem(camera.label, camera.index)
        # The currently assigned camera might not have been detected this run (e.g. temporarily
        # unplugged) -- add it anyway so the combo reflects the persisted setting instead of
        # silently jumping to whatever camera happens to be first in the list.
        if current_index not in [camera.index for camera in cameras]:
            combo.addItem(f"Camera {current_index} (not detected)", current_index)
        combo.setCurrentIndex(combo.findData(current_index))
        return combo

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
                self.camera_manager, self.face_analysis_app, on_apply_threshold=self._on_calibration_threshold_applied
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
                self.camera_manager, on_apply_threshold=self._on_barcode_calibration_threshold_applied
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
        group = QGroupBox("Prerequisites")
        layout = QVBoxLayout(group)

        for status in _check_all_prerequisites():
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

        return group

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
        info_window = PrerequisiteInstallInfoWindow(status, installer, on_installed=self._refresh_prerequisites, parent=self)
        info_window.exec()
