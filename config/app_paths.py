"""Resolves where the app's persistent data lives on disk: the database, saved face images, saved
document images, and the log file are each an independently operator-configurable directory.

Deliberately NOT derived from __file__ the way earlier code paths were: once packaged as a
standalone EXE, __file__ resolves inside a PyInstaller --onefile temp extraction folder that's
wiped on every exit, or inside a --onedir folder that may sit under Program Files and need admin
rights to write to. Instead, a small fixed-location bootstrap file under the current user's
LOCALAPPDATA records each configured directory -- always writable, persists across app updates,
and distinct per Windows user.

Each directory is configurable from the settings window (see ui/settings_window.py), but changing
one only records a *pending* relocation -- the actual file move happens in get_configured_dir(),
at the very start of the *next* app launch, before anything has opened a file handle inside the
old location. Attempting the move while the database connection and camera preview loop are still
live (mid-session) risks a locked/partially-written file; deferring it to the next cold start
sidesteps that entirely.
"""

import json
import logging
import os
import shutil
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox, QWidget

logger = logging.getLogger(__name__)

APP_NAME = "Kontroll"

_LOCAL_APP_DATA_ROOT = Path(os.getenv("LOCALAPPDATA", str(Path.home()))) / APP_NAME
_BOOTSTRAP_CONFIG_PATH = _LOCAL_APP_DATA_ROOT / "app_config.json"

DB_DIR_KEY = "db_dir"
FACES_DIR_KEY = "faces_dir"
DOCUMENTS_DIR_KEY = "documents_dir"
LOGS_DIR_KEY = "logs_dir"

# The three data directories are prompted for at startup if missing (see
# ensure_configured_dirs_exist) since they hold irreplaceable user content. The log directory is
# deliberately excluded from that prompt -- it's a diagnostic aid, always safe to silently
# recreate, and prompting about it on every fresh machine would just be startup-flow noise.
_DATA_PATH_KEYS = (DB_DIR_KEY, FACES_DIR_KEY, DOCUMENTS_DIR_KEY)
_ALL_PATH_KEYS = (DB_DIR_KEY, FACES_DIR_KEY, DOCUMENTS_DIR_KEY, LOGS_DIR_KEY)

_PATH_LABELS: dict[str, str] = {
    DB_DIR_KEY: "Database",
    FACES_DIR_KEY: "Face images",
    DOCUMENTS_DIR_KEY: "Document images",
    LOGS_DIR_KEY: "Log file",
}

_DEFAULT_DIRS: dict[str, Path] = {
    DB_DIR_KEY: _LOCAL_APP_DATA_ROOT / "data",
    FACES_DIR_KEY: _LOCAL_APP_DATA_ROOT / "data" / "faces",
    DOCUMENTS_DIR_KEY: _LOCAL_APP_DATA_ROOT / "data" / "documents",
    LOGS_DIR_KEY: _LOCAL_APP_DATA_ROOT / "logs",
}


def _read_bootstrap_config() -> dict:
    if not _BOOTSTRAP_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_BOOTSTRAP_CONFIG_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _write_bootstrap_config(config: dict) -> None:
    _BOOTSTRAP_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _BOOTSTRAP_CONFIG_PATH.write_text(json.dumps(config), encoding="utf-8")


def path_label(key: str) -> str:
    """Human-readable label for one of the DB_DIR_KEY / FACES_DIR_KEY / DOCUMENTS_DIR_KEY / LOGS_DIR_KEY constants."""
    return _PATH_LABELS[key]


def resolve_configured_dir(key: str) -> Path:
    """Return the currently configured directory for the given path key, without creating it.

    Also completes any relocation requested from the settings window on a *previous* run for this
    key -- see the module docstring for why that's deferred to here.
    """
    config = _read_bootstrap_config()
    pending = config.get(f"pending_{key}")
    if pending is not None:
        config = _complete_pending_relocation(config, key, Path(pending))
    return Path(config.get(key, _DEFAULT_DIRS[key]))


def get_configured_dir(key: str) -> Path:
    """Return the currently configured directory for the given path key, creating it if needed."""
    resolved_dir = resolve_configured_dir(key)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    return resolved_dir


def _complete_pending_relocation(config: dict, key: str, new_dir: Path) -> dict:
    """Move every file from the old directory for this key into the new one, then update the bootstrap config to point at it.

    Never overwrites a file already at the destination (e.g. a previous, partially completed
    move) -- silently skips it rather than risking clobbering data. On any filesystem error
    (permissions, disk full, path on an unavailable removable drive), the move is abandoned and
    the app falls back to the old location rather than crashing at startup; the operator can retry
    from settings once the underlying issue is fixed.
    """
    old_dir = Path(config.get(key, _DEFAULT_DIRS[key]))
    try:
        new_dir.mkdir(parents=True, exist_ok=True)
        if old_dir.exists() and old_dir.resolve() != new_dir.resolve():
            for item in old_dir.iterdir():
                destination = new_dir / item.name
                if not destination.exists():
                    shutil.move(str(item), str(destination))
        config[key] = str(new_dir)
        # Recorded so a caller with DB access (main.py, once storage.db is available -- this
        # module intentionally has none) can rewrite any absolute image paths stored in old scan
        # rows from old_dir to new_dir. Without this, files physically move here but rows written
        # before the move keep pointing at the old (now-empty) directory and their thumbnails
        # start showing "No image" after the next restart. Popped (see pop_completed_relocation)
        # once consumed so it doesn't get reapplied on a later run.
        if old_dir.resolve() != new_dir.resolve():
            config[f"completed_relocation_{key}"] = {"old": str(old_dir), "new": str(new_dir)}
        logger.info("Completed pending relocation of %s: %s -> %s", key, old_dir, new_dir)
    except OSError:
        logger.warning("Failed to move %s to %s; keeping %s", key, new_dir, old_dir, exc_info=True)
    config.pop(f"pending_{key}", None)
    _write_bootstrap_config(config)
    return config


def request_dir_change(key: str, new_dir: Path) -> None:
    """Record a new directory choice for the given path key. Takes effect (and the actual file move happens) on the next app restart."""
    config = _read_bootstrap_config()
    config[f"pending_{key}"] = str(new_dir)
    _write_bootstrap_config(config)


def pop_completed_relocation(key: str) -> tuple[Path, Path] | None:
    """Return and clear the (old_dir, new_dir) pair for a just-completed relocation of this key, or None if none is pending consumption.

    Meant to be called exactly once per relocation, after a DB connection is available, so the
    caller can rewrite stale absolute image paths stored in old scan rows -- see the comment in
    _complete_pending_relocation for why this two-step handoff exists (this module has no DB
    access of its own).
    """
    config = _read_bootstrap_config()
    record = config.pop(f"completed_relocation_{key}", None)
    if record is None:
        return None
    _write_bootstrap_config(config)
    return (Path(record["old"]), Path(record["new"]))


def get_pending_dir_change(key: str) -> Path | None:
    """Return the pending relocation target for this path key if a change was requested but the app hasn't restarted yet, else None."""
    pending = _read_bootstrap_config().get(f"pending_{key}")
    return Path(pending) if pending is not None else None


def _set_configured_dir_immediately(key: str, new_dir: Path) -> None:
    """Point a path key at a new directory effective immediately (not on next restart) -- used only by ensure_configured_dirs_exist's fallback, where the app hasn't started using the old path yet this run."""
    config = _read_bootstrap_config()
    config[key] = str(new_dir)
    _write_bootstrap_config(config)


def get_db_dir() -> Path:
    """Typed convenience accessor: the directory the database file lives in."""
    return get_configured_dir(DB_DIR_KEY)


def get_faces_dir() -> Path:
    """Typed convenience accessor: the directory saved face-capture images live in."""
    return get_configured_dir(FACES_DIR_KEY)


def get_documents_dir() -> Path:
    """Typed convenience accessor: the directory saved document-capture images live in."""
    return get_configured_dir(DOCUMENTS_DIR_KEY)


def get_logs_dir() -> Path:
    """Typed convenience accessor: the directory the log file lives in."""
    return get_configured_dir(LOGS_DIR_KEY)


def get_flagged_images_dir() -> Path:
    """Directory reference images for the operator-maintained flagged-persons watch list are stored in.

    Deliberately not one of the four operator-configurable/relocatable directories above -- it's a
    small, low-volume folder of manually-added entries (not a per-scan capture output), so it's
    kept simple as a fixed subdirectory next to the faces folder rather than adding a fifth
    relocation flow. It only follows the faces directory if that itself gets relocated (since it's
    resolved fresh from get_faces_dir() every call), it is not itself independently relocatable.
    """
    flagged_dir = get_faces_dir().parent / "flagged"
    flagged_dir.mkdir(parents=True, exist_ok=True)
    return flagged_dir


def ensure_configured_dirs_exist(parent: QWidget | None = None) -> None:
    """At app startup, confirm each of the three data directories (DB, faces, documents) exists
    on disk; for any that doesn't, ask the operator whether to create it automatically.

    Must run once, early in main(), after QApplication exists (so a dialog can actually be shown)
    but before storage.db/ui.main_window are imported -- those modules resolve and auto-create
    these same directories as a matter of course, which would silently bypass this prompt if they
    ran first. Declining falls back to that path's own default location under LOCALAPPDATA
    (created without asking, since it's an app-owned location rather than one the operator
    explicitly redirected to somewhere that might be unavailable, e.g. a disconnected external or
    network drive).

    The log directory deliberately isn't included here -- see _DATA_PATH_KEYS.
    """
    for key in _DATA_PATH_KEYS:
        resolved = resolve_configured_dir(key)
        if resolved.exists():
            continue

        answer = QMessageBox.question(
            parent,
            "Folder not found",
            f"The configured {path_label(key).lower()} folder does not exist:\n\n{resolved}\n\nCreate it automatically?",
        )
        if answer == QMessageBox.StandardButton.Yes:
            resolved.mkdir(parents=True, exist_ok=True)
            continue

        default_dir = _DEFAULT_DIRS[key]
        default_dir.mkdir(parents=True, exist_ok=True)
        _set_configured_dir_immediately(key, default_dir)
        QMessageBox.information(
            parent,
            "Folder not found",
            f"Using the default {path_label(key).lower()} folder instead:\n\n{default_dir}",
        )
