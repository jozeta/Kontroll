"""App entry point: wires config, DB, retention job, and launches the PyQt6 UI."""

import logging
import sys
from pathlib import Path
from types import TracebackType

from PyQt6.QtWidgets import QApplication

from prerequisites import run_prerequisite_check

LOG_PATH = Path(__file__).resolve().parent / "logs" / "app.log"


def _log_uncaught_exception(
    exc_type: type[BaseException], exc_value: BaseException, exc_traceback: TracebackType | None
) -> None:
    """Route any exception that escapes the app all the way to the top into the crash log before the default handler prints it."""
    logging.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    sys.__excepthook__(exc_type, exc_value, exc_traceback)


def _configure_logging() -> None:
    """Set up basic file logging and route uncaught exceptions into it -- the v1 scope is basic error/crash logging only, not a full audit trail."""
    logging.basicConfig(
        filename=LOG_PATH,
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    sys.excepthook = _log_uncaught_exception


def main() -> None:
    """Bootstrap the app: check prerequisites, initialize storage, purge expired scans, load the face model, and launch the UI."""
    _configure_logging()
    logging.info("Starting ID Verification app")

    # QApplication must exist before any prerequisite dialog can be shown, so this is
    # constructed first; every other import below is deferred until after the prerequisite
    # check specifically so a missing package is caught by that check (with an offer to
    # install it) rather than crashing the process with an ImportError beforehand.
    app = QApplication(sys.argv)
    run_prerequisite_check()

    from capture.camera_manager import CameraManager
    from config.settings import ensure_default_settings, get_doc_camera_index, get_face_camera_index
    from face.detector import load_face_analysis_model
    from storage.db import get_connection, initialize_schema
    from storage.retention import purge_expired_scans
    from ui.main_window import MainWindow

    conn = get_connection()
    initialize_schema(conn)
    ensure_default_settings(conn)

    purged_count = purge_expired_scans(conn)
    logging.info("Retention purge removed %d expired scan(s)", purged_count)

    logging.info("Loading face analysis model (buffalo_l)...")
    face_analysis_app = load_face_analysis_model()

    camera_manager = CameraManager(
        face_camera_index=get_face_camera_index(conn),
        doc_camera_index=get_doc_camera_index(conn),
    )

    window = MainWindow(conn, camera_manager, face_analysis_app)
    window.show()
    exit_code = app.exec()

    conn.close()
    logging.info("ID Verification app exiting with code %d", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
