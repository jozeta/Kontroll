"""Shared BGR-ndarray <-> QPixmap display helpers.

Used by every window that shows a live camera frame or a saved capture image (ui/main_window.py,
ui/image_viewer.py, ui/settings_window.py) -- kept in its own module, importing nothing from any
of them, specifically so all three can import from here without a circular import (ui/main_window.py
imports SettingsWindow from ui/settings_window.py, and ui/main_window.py/ui/settings_window.py both
already import ClickableThumbnailLabel from ui/image_viewer.py).
"""

import cv2
import numpy as np
from PyQt6.QtGui import QImage, QPixmap

from storage.models import thumbnail_cache_path

# Shared by every place a saved scan image's cached thumbnail is generated (ui/main_window.py's
# capture flow, ui/image_viewer.py::ImageEditDialog's apply flow) or read back
# (ui/main_window.py::_load_thumbnail) -- one definition, so the two can never disagree on what
# size actually got cached.
THUMBNAIL_SIZE = (110, 82)


def shrink_to_fit(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Downscale a BGR frame (via cheap cv2.resize) to approximately fit within target_width x target_height, preserving aspect ratio -- a no-op if it's already smaller either way.

    Meant to run *before* bgr_frame_to_pixmap's color-space conversion and QImage copy, both of
    which cost proportional to pixel count -- calling it first means those don't pay full-camera-
    resolution (or full-saved-image-resolution) cost on every live-preview tick or thumbnail load
    just to have most of that detail immediately discarded by a later scale-down to a small label.
    A caller's own final Qt-side .scaled() call for exact-size fitting stays in place regardless
    (cheap once the image is already this much smaller) so this doesn't change what's displayed.
    """
    height, width = frame.shape[:2]
    if target_width <= 0 or target_height <= 0 or (width <= target_width and height <= target_height):
        return frame
    scale = min(target_width / width, target_height / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def bgr_frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame into a QPixmap suitable for display in a QLabel."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    # .copy() detaches the QImage from rgb_frame's buffer, which is a stack-local that would
    # otherwise be freed once this function returns.
    qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def write_thumbnail_cache(image_path: str, frame: np.ndarray, target_width: int, target_height: int) -> None:
    """Generate and save a small cached thumbnail for frame -- the full-resolution image just saved/edited at image_path -- so a later ui/main_window.py::_load_thumbnail call can decode this tiny file instead of the full-resolution original every time.

    Called right after cv2.imwrite'ing the full image, from the same in-memory frame that was
    just written (MainWindow._finish_save_capture, ImageEditDialog._on_apply_clicked) -- no extra
    disk read needed to populate the cache at the point an image is actually produced. Overwrites
    any existing cache file unconditionally, unlike the "<name>.original<ext>" backup pattern this
    naming convention otherwise mirrors (see storage.models.thumbnail_cache_path) -- a thumbnail is
    a disposable, always-regenerable-from-the-full-image cache, not something to ever preserve an
    older version of.
    """
    thumbnail = shrink_to_fit(frame, target_width, target_height)
    cv2.imwrite(str(thumbnail_cache_path(image_path)), thumbnail, [cv2.IMWRITE_JPEG_QUALITY, 85])
