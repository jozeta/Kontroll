"""Grabs and returns a single still frame from a given camera on manual operator trigger."""

import cv2
import numpy as np


def capture_still_frame(capture: cv2.VideoCapture) -> np.ndarray | None:
    """Read one frame from the given camera handle right now, returning a copy for the caller to keep.

    Called only in direct response to an operator's capture button press -- there is no
    auto-detection or auto-triggering here, matching the v1 requirement that both captures are
    explicitly operator-initiated. The frame is copied so the returned still is independent of
    whatever the live preview loop reads from the same handle afterward.
    """
    was_read, frame = capture.read()
    if not was_read:
        return None
    return frame.copy()
