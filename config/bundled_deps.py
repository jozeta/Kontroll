"""Resolves optional, portable copies of the app's two external, non-pip runtime dependencies --
Tesseract OCR and the InsightFace buffalo_l model pack -- from a location relative to the app's
own install folder, rather than a machine-wide path (system PATH / Program Files / the current
Windows user's home directory).

This is the opposite direction of config/app_paths.py: that module deliberately keeps *writable
application data* under LOCALAPPDATA, independent of where the app is installed, since the EXE's
own folder may not be writable (e.g. under Program Files) and must not be assumed stable. These
two dependencies are the opposite case -- read-only assets the app ships alongside itself -- so
they're exactly what SHOULD live relative to the install folder: put a "tesseract" folder (a
copy of a Tesseract install) and a "models" folder (containing "buffalo_l", InsightFace's own
directory layout) next to the EXE, or next to main.py in a source checkout, and the whole folder
becomes self-contained -- copyable to a USB stick or a different drive/path with no
reconfiguration. See README.md "Portable / USB-stick deployment" for how to populate them.

Neither portable copy is required: prerequisites.py and face/detector.py both fall back to the
previous machine-wide defaults (PATH / Program Files / the user's .insightface home folder) when
the portable copy isn't present, so an existing non-portable install keeps working unchanged.
"""

import sys
from pathlib import Path


def get_app_dir() -> Path:
    """Return the app's own install directory: the frozen EXE's folder, or the project root when running from a source checkout.

    Safe to use here (unlike config/app_paths.py's writable data directories) because these two
    dependencies are read-only assets an operator explicitly chooses to place alongside the app --
    there's no risk of needing write access to a Program Files-style install location, since
    nothing here is ever written back to this path by the app itself.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


_APP_DIR = get_app_dir()

# A portable Tesseract copy is just the contents of a normal Tesseract install folder (tesseract.exe
# plus its tessdata\ subfolder and any DLLs it needs) copied wholesale into this subfolder -- see
# README.md.
TESSERACT_DIR = _APP_DIR / "tesseract"
TESSERACT_EXE = TESSERACT_DIR / "tesseract.exe"
TESSDATA_DIR = TESSERACT_DIR / "tessdata"

# Passed directly as InsightFace's own `root` constructor argument, which downloads/looks for a
# named model pack at `<root>/models/<name>` -- passing the app dir itself as root is what makes
# the resulting layout `<app dir>/models/buffalo_l/...`, matching TESSERACT_DIR's sibling-folder
# convention, rather than inventing a different structure of our own.
INSIGHTFACE_PORTABLE_ROOT = _APP_DIR
INSIGHTFACE_MODEL_FILE = _APP_DIR / "models" / "buffalo_l" / "w600k_r50.onnx"

# The legacy default InsightFace has always used (and what a pre-portable-support install of this
# app already has on disk) -- checked as a fallback so existing installs aren't forced to
# re-download the ~330MB pack just because portable support was added.
INSIGHTFACE_LEGACY_ROOT = "~/.insightface"
INSIGHTFACE_LEGACY_MODEL_FILE = Path.home() / ".insightface" / "models" / "buffalo_l" / "w600k_r50.onnx"


def resolve_insightface_root() -> str:
    """Pick which root InsightFace's FaceAnalysis should use to find (or, if missing everywhere, download) the buffalo_l model.

    Prefers the portable location next to the app itself, including for a fresh download that
    doesn't exist anywhere yet -- so newly-downloaded models default to travelling with the
    install folder going forward. Only falls back to the legacy machine-wide default if the model
    is already there from an older, non-portable install of this app.
    """
    if not INSIGHTFACE_MODEL_FILE.exists() and INSIGHTFACE_LEGACY_MODEL_FILE.exists():
        return INSIGHTFACE_LEGACY_ROOT
    return str(INSIGHTFACE_PORTABLE_ROOT)
