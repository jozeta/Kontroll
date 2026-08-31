"""App entry point: wires config, DB, retention job, and launches the PyQt6 UI."""

import logging
import logging.handlers
import sys
import time
from types import TracebackType

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from config.app_paths import (
    DOCUMENTS_DIR_KEY,
    FACES_DIR_KEY,
    ensure_configured_dirs_exist,
    get_logs_dir,
    pop_completed_relocation,
)
from prerequisites import run_prerequisite_check

# ui.splash only depends on PyQt6 itself (see that module), so -- unlike ui.main_window/
# ui.settings_window, which pull in insightface/cv2/pytesseract at module level -- it's safe to
# import this early, before run_prerequisite_check(). Shown for the entire remaining startup
# sequence below (prerequisite check, DB init, model load, camera open), which is the slow part an
# operator would otherwise stare at a blank taskbar entry for.
from ui.splash import build_splash_screen

logger = logging.getLogger(__name__)

# Resolved via config.app_paths (LOCALAPPDATA-based, operator-configurable from the settings
# window) rather than a path relative to this file's own location -- once packaged as a
# standalone EXE, a __file__-relative path resolves inside an ephemeral or not-reliably-writable
# location. See config/app_paths.py for the full rationale.
LOG_PATH = get_logs_dir() / "app.log"

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
_LOG_MAX_BYTES = 5 * 1024 * 1024  # rotate at 5MB
_LOG_BACKUP_COUNT = 3  # plus up to 3 rotated files (~20MB ceiling total)


def _log_uncaught_exception(
    exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None
) -> None:
    """Route any exception that escapes the app all the way to the top into the crash log before the default handler prints it."""
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def _configure_logging() -> None:
    """Attach a size-rotated file handler to the root logger and route uncaught exceptions into it.

    Every module logs via its own `logging.getLogger(__name__)`, which by default propagates up to
    the root logger -- attaching the handler there (rather than per-module) is what gets every
    module's log lines into the same file with one consistent format. Rotation (rather than a
    single ever-growing file, the previous approach) matters now that the app logs every
    significant operator action, not just startup/shutdown/errors -- daily use would otherwise
    accumulate an unbounded file.
    """
    handler = logging.handlers.RotatingFileHandler(
        LOG_PATH, maxBytes=_LOG_MAX_BYTES, backupCount=_LOG_BACKUP_COUNT, encoding="utf-8"
    )
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(handler)

    sys.excepthook = _log_uncaught_exception


def apply_debug_logging_setting(enabled: bool) -> None:
    """Switch the root logger between INFO and DEBUG -- called once at startup (see main(), after
    the settings DB is available) and again live from ui/settings_window.py's General tab toggle,
    so an operator can turn on verbose logging to diagnose a reported issue without restarting.
    """
    logging.getLogger().setLevel(logging.DEBUG if enabled else logging.INFO)
    logger.info("Debug logging %s", "enabled" if enabled else "disabled")


def main() -> None:
    """Bootstrap the app: check prerequisites, initialize storage, purge expired scans, load the face model, and launch the UI."""
    _configure_logging()
    logger.info("Starting Kontroll (log file: %s)", LOG_PATH)

    # QApplication must exist before any prerequisite dialog can be shown, so this is
    # constructed first; every other import below is deferred until after the prerequisite
    # check specifically so a missing package is caught by that check (with an offer to
    # install it) rather than crashing the process with an ImportError beforehand.
    app = QApplication(sys.argv)

    splash = build_splash_screen()
    splash.show()
    splash.showMessage("Starting...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, Qt.GlobalColor.white)
    # No event loop is running yet (app.exec() is still well below this point) -- without an
    # explicit processEvents() the splash would stay an unpainted blank window on screen for the
    # entire remaining startup sequence instead of actually appearing.
    app.processEvents()

    # Must run before storage.db/ui.main_window are imported: those modules resolve (and
    # silently auto-create) the same configured directories as a matter of course, which would
    # bypass this operator prompt entirely if they ran first. Passing splash as parent keeps this
    # dialog in front of it -- see the comment on run_prerequisite_check(splash) below for why.
    ensure_configured_dirs_exist(splash)

    splash.showMessage(
        "Checking prerequisites...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, Qt.GlobalColor.white
    )
    app.processEvents()
    # Passing the splash as parent is what keeps these dialogs in front of it -- the splash has
    # WindowStaysOnTopHint set (see ui/splash.py) so it stays above the taskbar/other apps during
    # the slow startup sequence, but that same hint also keeps it above a parentless dialog. A
    # dialog parented to the splash is stacked above its parent by Qt's own window rules, which
    # overrides that -- without a parent, the operator would see an unclickable splash sitting on
    # top of the prerequisite dialog with no way to answer it.
    run_prerequisite_check(splash)

    from capture.camera_manager import CameraManager, ExposureSettings
    from config.settings import (
        ensure_default_settings,
        get_debug_logging_enabled,
        get_doc_camera_auto_exposure,
        get_doc_camera_backend,
        get_doc_camera_brightness,
        get_doc_camera_exposure,
        get_doc_camera_index,
        get_doc_camera_resolution,
        get_face_camera_auto_exposure,
        get_face_camera_backend,
        get_face_camera_brightness,
        get_face_camera_exposure,
        get_face_camera_index,
        get_face_camera_resolution,
        get_theme,
    )
    from face.detector import load_face_analysis_model
    from storage.db import get_connection, initialize_schema
    from storage.models import rewrite_image_directory_paths
    from storage.retention import purge_expired_scans
    from ui.main_window import MainWindow
    from ui.theme import apply_theme

    conn = get_connection()
    initialize_schema(conn)
    ensure_default_settings(conn)
    apply_debug_logging_setting(get_debug_logging_enabled(conn))

    # ensure_configured_dirs_exist() above already moved any relocated faces/documents folder's
    # files on disk, but had no DB connection yet to fix up scan rows saved before the move --
    # finish that here now that one exists. See config/app_paths.py::pop_completed_relocation.
    for path_key in (FACES_DIR_KEY, DOCUMENTS_DIR_KEY):
        relocation = pop_completed_relocation(path_key)
        if relocation is not None:
            old_dir, new_dir = relocation
            updated_count = rewrite_image_directory_paths(conn, old_dir, new_dir)
            logger.info(
                "Repointed %d scan row(s) for %s relocation: %s -> %s", updated_count, path_key, old_dir, new_dir
            )

    # Applied here (rather than immediately after QApplication construction) because it needs
    # config.settings/the DB, both deferred past the prerequisite check further down -- see
    # CLAUDE.md's "Startup sequence and the deferred-import bootstrap". The prerequisite-check
    # dialogs above render with whatever default palette PyQt6/the OS already gave them, which is
    # fine since they contain no custom hardcoded colors of their own.
    apply_theme(app, get_theme(conn))

    purged_count = purge_expired_scans(conn)
    logger.info("Retention purge removed %d expired scan(s)", purged_count)

    splash.showMessage(
        "Loading face recognition model...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, Qt.GlobalColor.white
    )
    app.processEvents()
    logger.info("Loading face analysis model (buffalo_l)...")
    model_load_started_at = time.monotonic()
    face_analysis_app = load_face_analysis_model()
    logger.info("Face analysis model loaded in %.1fs", time.monotonic() - model_load_started_at)

    splash.showMessage(
        "Opening cameras...", Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter, Qt.GlobalColor.white
    )
    app.processEvents()
    camera_manager = CameraManager(
        face_camera_index=get_face_camera_index(conn),
        doc_camera_index=get_doc_camera_index(conn),
        face_resolution=get_face_camera_resolution(conn),
        doc_resolution=get_doc_camera_resolution(conn),
        face_exposure=ExposureSettings(
            auto=get_face_camera_auto_exposure(conn),
            exposure=get_face_camera_exposure(conn),
            brightness=get_face_camera_brightness(conn),
        ),
        doc_exposure=ExposureSettings(
            auto=get_doc_camera_auto_exposure(conn),
            exposure=get_doc_camera_exposure(conn),
            brightness=get_doc_camera_brightness(conn),
        ),
        face_backend=get_face_camera_backend(conn),
        doc_backend=get_doc_camera_backend(conn),
    )

    window = MainWindow(conn, camera_manager, face_analysis_app)
    window.show()
    splash.finish(window)
    logger.info("Main window shown; entering event loop")
    exit_code = app.exec()

    conn.close()
    logger.info("Kontroll exiting with code %d", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
