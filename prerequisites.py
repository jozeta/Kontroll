"""Checks the app's external prerequisites (Python packages, Tesseract, the InsightFace model, cameras) at startup
and offers to install/download whichever are missing, before the main window is constructed.

Every check function here is written to avoid importing the package it is checking for at module
load time -- main.py imports this module before any of capture/face/document/storage/ui, so a
missing package must be reported through a dialog here rather than crashing the process with an
ImportError before the operator ever sees a prompt. The one hard, unavoidable dependency is
PyQt6 itself: it is what renders the dialogs, so there is no way to check for its own absence
from within the running app -- that failure mode is covered in README.md's setup instructions
instead.
"""

import importlib.util
import logging
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox, QWidget

from config.bundled_deps import (
    INSIGHTFACE_LEGACY_MODEL_FILE,
    INSIGHTFACE_MODEL_FILE,
    INSIGHTFACE_PORTABLE_ROOT,
    TESSDATA_DIR,
    TESSERACT_EXE,
)
from config.ocr_languages import ADDITIONAL_OCR_LANGUAGES

logger = logging.getLogger(__name__)

# Maps each requirements.txt entry to the module name Python actually imports it under, since
# a few of them differ (e.g. the "opencv-python" package is imported as "cv2").
_REQUIRED_PACKAGES = {
    "PyQt6": "PyQt6",
    "opencv-python": "cv2",
    "numpy": "numpy",
    "Pillow": "PIL",
    "onnxruntime": "onnxruntime",
    "insightface": "insightface",
    "pytesseract": "pytesseract",
    "mrz": "mrz",
    "zxing-cpp": "zxingcpp",
}

_TESSERACT_COMMON_INSTALL_PATH = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# "tessdata_fast" (not the full "tessdata" repo) specifically -- its integer-quantized models are
# a few MB each rather than tens of MB, which matters here since these download synchronously on
# the operator's "Install now" click with no progress indicator.
_TESSDATA_FAST_URL_TEMPLATE = "https://github.com/tesseract-ocr/tessdata_fast/raw/main/{lang_code}.traineddata"

# INSIGHTFACE_MODEL_FILE / INSIGHTFACE_LEGACY_MODEL_FILE (imported above) point at the same
# proxy file -- the last one InsightFace's downloader writes for the buffalo_l pack, so its
# presence is a reasonable proxy for "the whole pack finished downloading" -- at the portable
# and legacy machine-wide locations respectively. See config/bundled_deps.py.


@dataclass
class PrerequisiteStatus:
    """Whether one prerequisite is satisfied, plus a human-readable detail for the startup dialog and whether this app can attempt to fix it."""

    name: str
    satisfied: bool
    detail: str
    can_auto_install: bool


def _is_frozen() -> bool:
    """True when running as a PyInstaller-frozen executable rather than from a source checkout."""
    return getattr(sys, "frozen", False)


def check_python_packages() -> PrerequisiteStatus:
    """Confirm every required third-party package actually imports; catches an incomplete or corrupted venv (or, when frozen, an incomplete packaged build)."""
    missing_packages = [
        requirement_name
        for requirement_name, module_name in _REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if not missing_packages:
        return PrerequisiteStatus("Python packages", True, "All required packages are installed", can_auto_install=True)

    if _is_frozen():
        # There's no venv/pip to install into inside a frozen EXE -- a missing package here means
        # the build itself is incomplete, not something an end user can fix by installing anything.
        return PrerequisiteStatus(
            "Python packages", False, f"Missing from this build: {', '.join(missing_packages)}", can_auto_install=False
        )
    return PrerequisiteStatus("Python packages", False, f"Missing: {', '.join(missing_packages)}", can_auto_install=True)


def check_tesseract() -> PrerequisiteStatus:
    """Confirm the Tesseract OCR binary is discoverable: a portable copy next to the app, on PATH, or at the common Windows install location.

    If it's only found at the portable or common-install-path location (not on PATH), point
    pytesseract at it directly here -- otherwise this check would report "satisfied" while
    document/ocr.py's plain pytesseract.image_to_string() call still shells out to a bare
    "tesseract" that isn't on PATH and fails with TesseractNotFoundError. The portable copy is
    checked first (see config/bundled_deps.py) since a machine could have both a portable copy
    and an unrelated system-wide install; preferring the portable one is what makes the app
    folder actually self-contained when both exist.
    """
    if TESSERACT_EXE.exists():
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_EXE)
        os.environ["TESSDATA_PREFIX"] = str(TESSDATA_DIR)
        return PrerequisiteStatus("Tesseract OCR", True, "Found (portable copy)", can_auto_install=True)
    if shutil.which("tesseract") is not None:
        return PrerequisiteStatus("Tesseract OCR", True, "Found", can_auto_install=True)
    if _TESSERACT_COMMON_INSTALL_PATH.exists():
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = str(_TESSERACT_COMMON_INSTALL_PATH)
        return PrerequisiteStatus("Tesseract OCR", True, "Found", can_auto_install=True)
    return PrerequisiteStatus(
        "Tesseract OCR", False, "Not found in the portable tesseract\\ folder, on PATH, or at the default install location",
        can_auto_install=True,
    )


def _resolve_active_tessdata_dir() -> Path | None:
    """Return the tessdata\\ folder belonging to whichever Tesseract install check_tesseract() would resolve (portable copy, PATH, or the common install path), or None if Tesseract itself isn't found anywhere.

    Re-derives the same three locations check_tesseract() checks, in the same priority order,
    rather than reading back pytesseract's global tesseract_cmd state -- that state is only ever
    set as a side effect of check_tesseract() having already run once this session, which this
    function shouldn't have to assume.
    """
    if TESSERACT_EXE.exists():
        return TESSDATA_DIR
    which_path = shutil.which("tesseract")
    if which_path is not None:
        return Path(which_path).resolve().parent / "tessdata"
    if _TESSERACT_COMMON_INSTALL_PATH.exists():
        return _TESSERACT_COMMON_INSTALL_PATH.parent / "tessdata"
    return None


def check_ocr_language_packs() -> PrerequisiteStatus:
    """Confirm every additional Tesseract language pack document/ocr.py requests (config.ocr_languages.ADDITIONAL_OCR_LANGUAGES) is installed alongside whichever Tesseract binary check_tesseract() found.

    Checked independently of check_tesseract() itself since a machine can have Tesseract installed
    with only its default English language data -- winget's silent install of the UB Mannheim
    build doesn't pull in additional language packs -- which is a real, separately-fixable gap:
    without them, document/ocr.py's multi-language OCR pass silently falls back to English-only
    recognition for any language pack that's missing, which has no trained glyph shapes for that
    language's own diacritics (å/ä/ö, ü/ß, é/ç, ñ, ...) and misreads them as plain unaccented
    look-alikes instead of failing loudly.
    """
    tessdata_dir = _resolve_active_tessdata_dir()
    if tessdata_dir is None:
        return PrerequisiteStatus(
            "OCR language packs", False, "Tesseract itself isn't installed yet -- install that first",
            can_auto_install=False,
        )

    # Language display name and code always printed together ("German (deu)"), and the missing
    # count/list kept separate from the full roster below -- a prior version of this message
    # printed display names alone, which real operator confusion showed isn't unambiguous enough
    # on its own (mistaking "Dutch" for "German"/"Deutsch", a classic false-friend pair, when only
    # one of the two was actually reported missing). Spelling out the exact expected filename next
    # to each name removes any doubt about which language a given .traineddata file is for.
    missing = [
        (lang_code, display_name)
        for lang_code, display_name in ADDITIONAL_OCR_LANGUAGES.items()
        if not (tessdata_dir / f"{lang_code}.traineddata").exists()
    ]
    roster_lines = "\n".join(
        f"  [{'x' if (tessdata_dir / f'{code}.traineddata').exists() else ' '}] {name} ({code}.traineddata)"
        for code, name in ADDITIONAL_OCR_LANGUAGES.items()
    )

    if not missing:
        return PrerequisiteStatus(
            "OCR language packs", True,
            f"All {len(ADDITIONAL_OCR_LANGUAGES)} additional language packs found in {tessdata_dir}:\n{roster_lines}",
            can_auto_install=True,
        )
    missing_summary = ", ".join(f"{name} ({code}.traineddata)" for code, name in missing)
    return PrerequisiteStatus(
        "OCR language packs", False,
        f"{len(missing)} of {len(ADDITIONAL_OCR_LANGUAGES)} additional language packs missing from "
        f"{tessdata_dir}:\n{roster_lines}\n\n"
        f"Missing: {missing_summary} -- these languages' accented characters will misread as "
        "their plain unaccented equivalents without them",
        can_auto_install=True,
    )


def check_insightface_model() -> PrerequisiteStatus:
    """Confirm the buffalo_l face-recognition model pack has already been downloaded and cached locally, portably or at the legacy location."""
    if INSIGHTFACE_MODEL_FILE.exists():
        return PrerequisiteStatus("InsightFace buffalo_l model", True, "Found (portable copy)", can_auto_install=True)
    if INSIGHTFACE_LEGACY_MODEL_FILE.exists():
        return PrerequisiteStatus("InsightFace buffalo_l model", True, "Found (already downloaded)", can_auto_install=True)
    return PrerequisiteStatus(
        "InsightFace buffalo_l model", False, "Not downloaded yet (~330MB, requires internet)", can_auto_install=True
    )


def check_cameras() -> PrerequisiteStatus:
    """Confirm at least two UVC cameras are currently detected. Informational only -- missing cameras can't be auto-installed."""
    try:
        import cv2
    except ImportError:
        return PrerequisiteStatus("Cameras", False, "Cannot check -- OpenCV is not installed", can_auto_install=False)

    detected_count = 0
    for index in range(10):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            was_read, _frame = capture.read()
            if was_read:
                detected_count += 1
        capture.release()

    if detected_count >= 2:
        return PrerequisiteStatus("Cameras", True, f"{detected_count} camera(s) detected", can_auto_install=False)
    return PrerequisiteStatus(
        "Cameras", False, f"Only {detected_count} camera(s) detected; the app needs two (face + document)",
        can_auto_install=False,
    )


def install_python_packages() -> bool:
    """Run pip install -r requirements.txt against the current interpreter's environment.

    Never actually reachable when frozen (check_python_packages() reports can_auto_install=False
    in that case) -- guarded here too as a defensive fallback, since sys.executable inside a
    frozen EXE is the EXE itself, not a real Python interpreter, and there's no bundled
    requirements.txt to point pip at either.
    """
    if _is_frozen():
        return False
    requirements_path = Path(__file__).resolve().parent / "requirements.txt"
    result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_path)])
    return result.returncode == 0


def install_tesseract() -> bool:
    """Install Tesseract OCR via winget (the UB Mannheim Windows build), and point pytesseract at it directly.

    Setting pytesseract.pytesseract.tesseract_cmd explicitly (rather than relying on PATH) lets
    the app use Tesseract immediately after this install completes, without needing a restart --
    a newly-installed binary's PATH update would not otherwise be visible to this already-running
    process until it exits and a new one starts.
    """
    result = subprocess.run(
        [
            "winget", "install", "--id", "UB-Mannheim.TesseractOCR", "-e",
            "--accept-package-agreements", "--accept-source-agreements",
        ]
    )
    if result.returncode != 0 or not _TESSERACT_COMMON_INSTALL_PATH.exists():
        return False

    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = str(_TESSERACT_COMMON_INSTALL_PATH)
    return True


def install_ocr_language_packs() -> bool:
    """Download every additional Tesseract language pack found missing by check_ocr_language_packs() straight into the active tessdata\\ folder.

    Each is a single small file fetched directly from the tesseract-ocr project's own
    tessdata_fast GitHub repo via urllib (no new dependency needed just for this) -- not run
    through winget/an installer, since UB Mannheim's package doesn't expose "add a language pack
    to an existing install" as a separate winget-able unit. Downloads whichever packs are still
    missing one at a time and reports overall success only if every one of them ends up present --
    a partial failure partway through (e.g. a dropped connection) still leaves whatever already
    downloaded successfully in place rather than being rolled back, since each is independently
    useful once installed.
    """
    tessdata_dir = _resolve_active_tessdata_dir()
    if tessdata_dir is None:
        return False

    all_succeeded = True
    for lang_code in ADDITIONAL_OCR_LANGUAGES:
        destination = tessdata_dir / f"{lang_code}.traineddata"
        if destination.exists():
            continue
        try:
            with urllib.request.urlopen(_TESSDATA_FAST_URL_TEMPLATE.format(lang_code=lang_code), timeout=30) as response:
                destination.write_bytes(response.read())
        except (OSError, urllib.error.URLError):
            logger.exception("Failed to download the '%s' Tesseract language pack to %s", lang_code, destination)
            all_succeeded = False
            continue
        if not (destination.exists() and destination.stat().st_size > 0):
            all_succeeded = False

    return all_succeeded


def download_insightface_model() -> bool:
    """Trigger the InsightFace buffalo_l model download by constructing a FaceAnalysis instance.

    Downloads straight into the app's own portable "models" folder (INSIGHTFACE_PORTABLE_ROOT --
    see config/bundled_deps.py) rather than the legacy .insightface folder under the user's home
    directory, so a fresh download on a machine with no existing model anywhere ends up
    self-contained by default. An existing legacy-location model is left untouched and still
    found by check_insightface_model() / face/detector.py -- this only affects where a *new*
    download lands.
    """
    from insightface.app import FaceAnalysis

    FaceAnalysis(name="buffalo_l", root=str(INSIGHTFACE_PORTABLE_ROOT))
    return INSIGHTFACE_MODEL_FILE.exists()


def run_prerequisite_check(parent: QWidget | None = None) -> None:
    """Check every external prerequisite and, for anything missing that this app can fix, ask the operator whether to install/download it now.

    Runs once at app startup, before the main window is constructed, so a missing prerequisite is
    caught here (with an offer to fix it) instead of surfacing later as an unexplained failure
    partway through a scan. Camera availability is reported but never prompted on, since there is
    nothing this app can install to fix missing hardware.
    """
    statuses = [
        check_python_packages(),
        check_tesseract(),
        check_ocr_language_packs(),
        check_insightface_model(),
        check_cameras(),
    ]

    for status in statuses:
        logger.info(
            "Prerequisite check: %s -> %s (%s)",
            status.name,
            "satisfied" if status.satisfied else "NOT satisfied",
            status.detail,
        )

    installers = {
        "Python packages": install_python_packages,
        "Tesseract OCR": install_tesseract,
        "OCR language packs": install_ocr_language_packs,
        "InsightFace buffalo_l model": download_insightface_model,
    }

    for status in statuses:
        if status.satisfied:
            continue

        if not status.can_auto_install:
            QMessageBox.warning(parent, "Prerequisite check", f"{status.name}: {status.detail}")
            continue

        answer = QMessageBox.question(
            parent,
            "Prerequisite check",
            f"{status.name} is missing.\n\n{status.detail}\n\nInstall/download it now?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            logger.info("Operator declined to install '%s'", status.name)
            continue

        logger.info("Operator requested install of '%s'", status.name)
        success = installers[status.name]()
        if success:
            logger.info("Prerequisite '%s' installed successfully", status.name)
            QMessageBox.information(parent, "Prerequisite check", f"{status.name} installed successfully.")
        else:
            logger.warning("Prerequisite '%s' installation failed", status.name)
            QMessageBox.warning(
                parent,
                "Prerequisite check",
                f"{status.name} installation failed. See README.md for manual setup instructions.",
            )
