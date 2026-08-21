"""The Settings window: manual scan purge, camera-to-side assignment, the capture keyboard
shortcut, and a live prerequisites checklist with per-item install instructions."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Callable

from insightface.app import FaceAnalysis
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeySequence
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QKeySequenceEdit,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import CameraInfo, CameraManager, enumerate_cameras
from config.app_paths import (
    DB_DIR_KEY,
    DOCUMENTS_DIR_KEY,
    FACES_DIR_KEY,
    LOGS_DIR_KEY,
    get_configured_dir,
    get_pending_dir_change,
    path_label,
    request_dir_change,
)
from config.settings import (
    RETENTION_DAYS_NEVER,
    get_capture_shortcut,
    get_disk_space_warning_mb,
    get_face_match_threshold,
    get_retention_days,
    set_setting,
)
from prerequisites import (
    PrerequisiteStatus,
    check_cameras,
    check_insightface_model,
    check_python_packages,
    check_tesseract,
    download_insightface_model,
    install_python_packages,
    install_tesseract,
)
from storage.backup import create_backup_archive
from storage.retention import purge_scans_older_than_days
from ui.calibration_window import CalibrationWindow

logger = logging.getLogger(__name__)

_PURGE_DAY_OPTIONS = [30, 90, 180, 365]

# Cameras is deliberately absent: check_cameras() always has can_auto_install=False, since there's
# nothing this app can install to fix missing hardware.
_PREREQUISITE_INSTALLERS: dict[str, Callable[[], bool]] = {
    "Python packages": install_python_packages,
    "Tesseract OCR": install_tesseract,
    "InsightFace buffalo_l model": download_insightface_model,
}

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
    return [check_python_packages(), check_tesseract(), check_insightface_model(), check_cameras()]


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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent, Qt.WindowType.Window)
        self.conn = conn
        self.camera_manager = camera_manager
        self.face_analysis_app = face_analysis_app
        self._on_data_changed = on_data_changed
        self._on_capture_shortcut_changed = on_capture_shortcut_changed
        self._prerequisite_rows: dict[str, tuple[QLabel, QPushButton]] = {}
        self._path_rows: dict[str, tuple[QLabel, QLabel]] = {}
        self._calibration_window: CalibrationWindow | None = None

        self.setWindowTitle("Settings")
        self.resize(480, 560)
        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._build_data_location_group())
        layout.addWidget(self._build_auto_purge_group())
        layout.addWidget(self._build_purge_group())
        layout.addWidget(self._build_disk_space_warning_group())
        layout.addWidget(self._build_backup_group())
        layout.addWidget(self._build_camera_group())
        layout.addWidget(self._build_face_match_threshold_group())
        layout.addWidget(self._build_shortcut_group())
        layout.addWidget(self._build_prerequisites_group())
        layout.addStretch(1)

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
        row.addWidget(self.purge_days_combo)

        purge_button = QPushButton("Purge")
        purge_button.clicked.connect(self._on_purge_clicked)
        row.addWidget(purge_button)
        row.addStretch(1)

        return group

    def _on_purge_clicked(self) -> None:
        days = self.purge_days_combo.currentData()
        confirm = QMessageBox.question(
            self,
            "Purge old scans",
            f"Permanently delete every scan older than {days} days, including saved images? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        purged_count = purge_scans_older_than_days(self.conn, days)
        logger.info("Operator purged %d scan(s) older than %d days", purged_count, days)
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

        logger.info("Operator created a backup archive at %s", file_path)
        QMessageBox.information(self, "Backup complete", f"Backup saved to:\n{file_path}")

    # -- Camera assignment ----------------------------------------------------------------------

    def _build_camera_group(self) -> QGroupBox:
        group = QGroupBox("Camera assignment")
        layout = QVBoxLayout(group)
        cameras = enumerate_cameras()

        face_row = QHBoxLayout()
        face_row.addWidget(QLabel("Visitor camera (left):"))
        self.face_camera_combo = self._build_camera_combo(cameras, self.camera_manager.face_camera_index)
        self.face_camera_combo.currentIndexChanged.connect(self._on_face_camera_changed)
        face_row.addWidget(self.face_camera_combo, stretch=1)
        layout.addLayout(face_row)

        doc_row = QHBoxLayout()
        doc_row.addWidget(QLabel("Identification camera (right):"))
        self.doc_camera_combo = self._build_camera_combo(cameras, self.camera_manager.doc_camera_index)
        self.doc_camera_combo.currentIndexChanged.connect(self._on_doc_camera_changed)
        doc_row.addWidget(self.doc_camera_combo, stretch=1)
        layout.addLayout(doc_row)

        return group

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
        self.camera_manager.switch_face_camera(new_index)
        logger.info("Operator reassigned the visitor (left) camera to index %d", new_index)

    def _on_doc_camera_changed(self, _list_index: int) -> None:
        new_index = self.doc_camera_combo.currentData()
        set_setting(self.conn, "doc_camera_index", str(new_index))
        self.camera_manager.switch_doc_camera(new_index)
        logger.info("Operator reassigned the identification (right) camera to index %d", new_index)

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
