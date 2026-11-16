"""The app's version number, read from the VERSION file (the one place it is kept).

Stdlib only, so it can be imported before the prerequisite check and from tests. In a PyInstaller
build, VERSION is bundled next to the code (``--add-data "VERSION;."``).
"""

import sys
from pathlib import Path


def get_version() -> str:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    try:
        return (base / "VERSION").read_text(encoding="utf-8").strip() or "unknown"
    except OSError:
        return "unknown"
