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
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox, QWidget

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

# The recognition model is the last file InsightFace's downloader writes for the buffalo_l pack,
# so its presence is a reasonable proxy for "the whole pack finished downloading".
_INSIGHTFACE_MODEL_FILE = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"


@dataclass
class PrerequisiteStatus:
    """Whether one prerequisite is satisfied, plus a human-readable detail for the startup dialog and whether this app can attempt to fix it."""

    name: str
    satisfied: bool
    detail: str
    can_auto_install: bool


def check_python_packages() -> PrerequisiteStatus:
    """Confirm every required third-party package actually imports; catches an incomplete or corrupted venv."""
    missing_packages = [
        requirement_name
        for requirement_name, module_name in _REQUIRED_PACKAGES.items()
        if importlib.util.find_spec(module_name) is None
    ]
    if missing_packages:
        return PrerequisiteStatus(
            "Python packages", False, f"Missing: {', '.join(missing_packages)}", can_auto_install=True
        )
    return PrerequisiteStatus("Python packages", True, "All required packages are installed", can_auto_install=True)


def check_tesseract() -> PrerequisiteStatus:
    """Confirm the Tesseract OCR binary is discoverable, either on PATH or at the common Windows install location."""
    if shutil.which("tesseract") is not None or _TESSERACT_COMMON_INSTALL_PATH.exists():
        return PrerequisiteStatus("Tesseract OCR", True, "Found", can_auto_install=True)
    return PrerequisiteStatus(
        "Tesseract OCR", False, "Not found on PATH or at the default install location", can_auto_install=True
    )


def check_insightface_model() -> PrerequisiteStatus:
    """Confirm the buffalo_l face-recognition model pack has already been downloaded and cached locally."""
    if _INSIGHTFACE_MODEL_FILE.exists():
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
    """Run pip install -r requirements.txt against the current interpreter's environment."""
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


def download_insightface_model() -> bool:
    """Trigger the InsightFace buffalo_l model download by constructing a FaceAnalysis instance."""
    from insightface.app import FaceAnalysis

    FaceAnalysis(name="buffalo_l")
    return _INSIGHTFACE_MODEL_FILE.exists()


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
        check_insightface_model(),
        check_cameras(),
    ]

    installers = {
        "Python packages": install_python_packages,
        "Tesseract OCR": install_tesseract,
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
            continue

        success = installers[status.name]()
        if success:
            QMessageBox.information(parent, "Prerequisite check", f"{status.name} installed successfully.")
        else:
            QMessageBox.warning(
                parent,
                "Prerequisite check",
                f"{status.name} installation failed. See README.md for manual setup instructions.",
            )
