"""Load/save app settings: retention period, face-match threshold, and camera device selections."""

import sqlite3

from capture.camera_manager import AUTO_LOWEST, CAMERA_BACKENDS, DEFAULT_CAMERA_BACKEND, ResolutionPreference

RETENTION_DAYS_NEVER: int = 0

# Defaults to "Never" (see RETENTION_DAYS_NEVER below) -- an operator has to opt into automatic
# deletion of scan history rather than losing it silently by default.
DEFAULT_RETENTION_DAYS: int = RETENTION_DAYS_NEVER

# Stored in the same retention_days setting as an ordinary value (rather than a separate
# enabled/disabled flag) so there's only ever one source of truth for the startup purge job to
# read -- 0 days of retention is otherwise meaningless (nothing would ever be old enough to
# survive a purge), so it's free to reuse as the "auto-purge disabled" sentinel.

# Starting point for buffalo_l (ArcFace-family) cosine similarity on L2-normalized embeddings.
# Published guidance for this model typically lands in the 0.30-0.45 range for same-domain
# (photo-vs-photo) verification, but this app compares a live webcam capture against a
# printed/laminated ID photo -- a cross-domain comparison that tends to push genuine-pair
# scores lower. Treat this default as unvalidated until recalibrated against real captured
# pairs during pilot use; this is why the UI always shows the raw score and an operator
# override alongside the automated pass/fail badge, rather than only a hard gate.
DEFAULT_FACE_MATCH_THRESHOLD: float = 0.38

# Rough heuristic threshold for document/field_crosscheck.py's barcode/OCR word-overlap check
# (used only when a document has a PDF417 barcode but no MRZ, e.g. most EU driving licences) --
# not calibrated against real captured pairs, same caveat as DEFAULT_FACE_MATCH_THRESHOLD above.
# ui/barcode_calibration_window.py helps an operator pick a better value from real desk captures,
# mirroring ui/calibration_window.py's face-match calibration.
DEFAULT_BARCODE_OVERLAP_THRESHOLD: float = 0.5

DEFAULT_FACE_CAMERA_INDEX: int = 0
DEFAULT_DOC_CAMERA_INDEX: int = 1

# "auto_lowest" means "probe the camera and use the lowest resolution it actually honors" (see
# capture.camera_manager.set_lowest_available_resolution); "auto" means the highest resolution the
# camera honors (capture.camera_manager.set_best_available_resolution). An operator can switch
# either camera to the other mode, or a fixed WIDTHxHEIGHT, from the Settings window.
#
# The two cameras get different factory defaults, not the same DEFAULT_CAMERA_RESOLUTION applied
# to both as before: only the document camera's frame ever goes through OCR/MRZ/barcode reading,
# and document/card_detector.py's own diagnosis of real production captures found the card itself
# often occupies only ~25-30% of that frame -- at a low resolution, that leaves far too few pixels
# on the card's actual printed text for Tesseract to read reliably (this was diagnosed as a
# concrete cause of production OCR failures, not a theoretical concern). The face camera has no
# equivalent floor -- face embedding downsamples to a small fixed input size regardless -- so it
# keeps the low-bandwidth/low-CPU default.
DEFAULT_FACE_CAMERA_RESOLUTION: str = AUTO_LOWEST
DEFAULT_DOC_CAMERA_RESOLUTION: str = "auto"

# Auto exposure is the factory default for both cameras -- matches how both cameras' diagnostics
# were found configured in production (see capture.camera_manager's exposure comment for the
# DirectShow auto/manual convention this maps to). Manual exposure/brightness values are only ever
# applied once an operator explicitly switches a camera to manual in Settings -- these defaults are
# just a reasonable starting point for the slider at that point, not calibrated per-device.
DEFAULT_CAMERA_AUTO_EXPOSURE: str = "1"
DEFAULT_CAMERA_EXPOSURE: int = -6
DEFAULT_CAMERA_BRIGHTNESS: int = 128

# DirectShow is the app-wide default backend (see capture.camera_manager.CAMERA_BACKENDS/
# enumerate_cameras' own comment) -- Media Foundation is an opt-in per-camera alternative for a
# camera/driver combination that negotiates MJPG more reliably over it, see
# capture/camera_manager.py::_open_camera's docstring for the diagnosed real-world case this
# exists for.
DEFAULT_CAMERA_BACKEND_SETTING: str = DEFAULT_CAMERA_BACKEND

DEFAULT_CAPTURE_SHORTCUT: str = "F12"

# Free-space floor below which the status bar shows a low-disk-space warning. 2GB is a rough
# starting point (a handful of scans at most before actually running out), not a calibrated
# value -- same unvalidated-default caveat as DEFAULT_FACE_MATCH_THRESHOLD above.
DEFAULT_DISK_SPACE_WARNING_MB: int = 2000

# "system" defers to whatever palette PyQt6/the OS hands the app by default; "light"/"dark" force
# an explicit QPalette override -- see ui/theme.py.
DEFAULT_THEME: str = "system"

# Off by default -- DEBUG-level logging includes per-tick preview timing and other high-frequency
# noise (see ui/main_window.py's slow-preview-tick warnings and capture/camera_manager.py's
# slow-camera-read warnings, both logged at WARNING regardless of this setting) not needed for
# routine operation, only when actively diagnosing a reported slowdown/bug on a given machine. See
# main.py::_configure_logging and ui/settings_window.py's General tab toggle.
DEFAULT_DEBUG_LOGGING_ENABLED: str = "0"

DEFAULT_SETTINGS: dict[str, str] = {
    "retention_days": str(DEFAULT_RETENTION_DAYS),
    "face_match_threshold": str(DEFAULT_FACE_MATCH_THRESHOLD),
    "barcode_overlap_threshold": str(DEFAULT_BARCODE_OVERLAP_THRESHOLD),
    "face_camera_index": str(DEFAULT_FACE_CAMERA_INDEX),
    "doc_camera_index": str(DEFAULT_DOC_CAMERA_INDEX),
    "face_camera_resolution": DEFAULT_FACE_CAMERA_RESOLUTION,
    "doc_camera_resolution": DEFAULT_DOC_CAMERA_RESOLUTION,
    "face_camera_auto_exposure": DEFAULT_CAMERA_AUTO_EXPOSURE,
    "doc_camera_auto_exposure": DEFAULT_CAMERA_AUTO_EXPOSURE,
    "face_camera_exposure": str(DEFAULT_CAMERA_EXPOSURE),
    "doc_camera_exposure": str(DEFAULT_CAMERA_EXPOSURE),
    "face_camera_brightness": str(DEFAULT_CAMERA_BRIGHTNESS),
    "doc_camera_brightness": str(DEFAULT_CAMERA_BRIGHTNESS),
    "face_camera_backend": DEFAULT_CAMERA_BACKEND_SETTING,
    "doc_camera_backend": DEFAULT_CAMERA_BACKEND_SETTING,
    "capture_shortcut": DEFAULT_CAPTURE_SHORTCUT,
    "disk_space_warning_mb": str(DEFAULT_DISK_SPACE_WARNING_MB),
    "theme": DEFAULT_THEME,
    "debug_logging_enabled": DEFAULT_DEBUG_LOGGING_ENABLED,
}


def ensure_default_settings(conn: sqlite3.Connection) -> None:
    """Insert any missing settings keys with their default values, leaving existing values untouched."""
    for key, value in DEFAULT_SETTINGS.items():
        conn.execute(
            "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
    conn.commit()


def get_setting(conn: sqlite3.Connection, key: str) -> str:
    """Return the raw string value for a settings key."""
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"]


def get_setting_or_none(conn: sqlite3.Connection, key: str) -> str | None:
    """Return the raw string value for a settings key that may not exist yet, or None if it's never been set.

    Unlike get_setting, this doesn't assume the key was pre-populated by ensure_default_settings --
    for settings like saved window geometry, there is no sensible hardcoded default to seed, only
    "nothing saved yet".
    """
    row = conn.execute("SELECT value FROM settings WHERE key = ?", (key,)).fetchone()
    return row["value"] if row is not None else None


def set_setting(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Insert or update a settings key's value."""
    conn.execute(
        """
        INSERT INTO settings (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (key, value),
    )
    conn.commit()


def get_retention_days(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the retention_days setting."""
    return int(get_setting(conn, "retention_days"))


def get_face_match_threshold(conn: sqlite3.Connection) -> float:
    """Typed convenience accessor for the face_match_threshold setting."""
    return float(get_setting(conn, "face_match_threshold"))


def get_barcode_overlap_threshold(conn: sqlite3.Connection) -> float:
    """Typed convenience accessor for the barcode_overlap_threshold setting."""
    return float(get_setting(conn, "barcode_overlap_threshold"))


def get_face_camera_index(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the face_camera_index setting."""
    return int(get_setting(conn, "face_camera_index"))


def get_doc_camera_index(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the doc_camera_index setting."""
    return int(get_setting(conn, "doc_camera_index"))


def _parse_camera_resolution(raw: str) -> ResolutionPreference:
    """Parse a stored camera-resolution setting: "auto_lowest" means AUTO_LOWEST, "auto" (or anything else unrecognized) means None (auto-best), else "WIDTHxHEIGHT"."""
    if raw == AUTO_LOWEST:
        return AUTO_LOWEST
    if "x" not in raw:
        return None
    width_str, _, height_str = raw.partition("x")
    if not width_str.isdigit() or not height_str.isdigit():
        return None
    return (int(width_str), int(height_str))


def get_face_camera_resolution(conn: sqlite3.Connection) -> ResolutionPreference:
    """Typed convenience accessor for the face_camera_resolution setting. None means auto (best available); AUTO_LOWEST means auto (lowest available)."""
    return _parse_camera_resolution(get_setting(conn, "face_camera_resolution"))


def get_doc_camera_resolution(conn: sqlite3.Connection) -> ResolutionPreference:
    """Typed convenience accessor for the doc_camera_resolution setting. None means auto (best available); AUTO_LOWEST means auto (lowest available)."""
    return _parse_camera_resolution(get_setting(conn, "doc_camera_resolution"))


def resolution_to_setting_value(resolution: ResolutionPreference) -> str:
    """Inverse of _parse_camera_resolution: format a resolution preference back to its stored string form."""
    if resolution is None:
        return "auto"
    if resolution == AUTO_LOWEST:
        return AUTO_LOWEST
    return f"{resolution[0]}x{resolution[1]}"


def get_face_camera_auto_exposure(conn: sqlite3.Connection) -> bool:
    """Typed convenience accessor for the face_camera_auto_exposure setting."""
    return get_setting(conn, "face_camera_auto_exposure") == "1"


def get_doc_camera_auto_exposure(conn: sqlite3.Connection) -> bool:
    """Typed convenience accessor for the doc_camera_auto_exposure setting."""
    return get_setting(conn, "doc_camera_auto_exposure") == "1"


def get_face_camera_exposure(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the face_camera_exposure setting (only applied when auto exposure is off -- see get_face_camera_auto_exposure)."""
    return int(get_setting(conn, "face_camera_exposure"))


def get_doc_camera_exposure(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the doc_camera_exposure setting (only applied when auto exposure is off -- see get_doc_camera_auto_exposure)."""
    return int(get_setting(conn, "doc_camera_exposure"))


def get_face_camera_brightness(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the face_camera_brightness setting (only applied when auto exposure is off -- see get_face_camera_auto_exposure)."""
    return int(get_setting(conn, "face_camera_brightness"))


def get_doc_camera_brightness(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the doc_camera_brightness setting (only applied when auto exposure is off -- see get_doc_camera_auto_exposure)."""
    return int(get_setting(conn, "doc_camera_brightness"))


def _parse_camera_backend(raw: str) -> int:
    """Parse a stored camera-backend setting string ("dshow"/"msmf") into its cv2.CAP_* constant, defaulting to DirectShow for an unrecognized/missing value."""
    return CAMERA_BACKENDS.get(raw, CAMERA_BACKENDS[DEFAULT_CAMERA_BACKEND])


def get_face_camera_backend(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the face_camera_backend setting -- a cv2.CAP_* constant (see capture.camera_manager.CAMERA_BACKENDS)."""
    return _parse_camera_backend(get_setting(conn, "face_camera_backend"))


def get_doc_camera_backend(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the doc_camera_backend setting -- see get_face_camera_backend."""
    return _parse_camera_backend(get_setting(conn, "doc_camera_backend"))


def get_capture_shortcut(conn: sqlite3.Connection) -> str:
    """Typed convenience accessor for the capture_shortcut setting (a Qt key-sequence string, e.g. 'F12')."""
    return get_setting(conn, "capture_shortcut")


def get_disk_space_warning_mb(conn: sqlite3.Connection) -> int:
    """Typed convenience accessor for the disk_space_warning_mb setting."""
    return int(get_setting(conn, "disk_space_warning_mb"))


def get_theme(conn: sqlite3.Connection) -> str:
    """Typed convenience accessor for the theme setting ("system" / "light" / "dark")."""
    return get_setting(conn, "theme")


def get_debug_logging_enabled(conn: sqlite3.Connection) -> bool:
    """Typed convenience accessor for the debug_logging_enabled setting."""
    return get_setting(conn, "debug_logging_enabled") == "1"
