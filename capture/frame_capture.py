"""Grabs and returns a single still frame from a given camera on manual operator trigger."""

import time

import cv2
import numpy as np

_READ_RETRIES = 3
_RETRY_DELAY_SECONDS = 0.05


def capture_still_frame(capture: cv2.VideoCapture) -> np.ndarray | None:
    """Read one frame from the given camera handle right now, returning a copy for the caller to keep.

    Called only in direct response to an operator's capture button press -- there is no
    auto-detection or auto-triggering here, matching the v1 requirement that both captures are
    explicitly operator-initiated. The frame is copied so the returned still is independent of
    whatever the live preview loop reads from the same handle afterward.

    A single dropped read is a known, harmless hiccup with the DirectShow backend (CAP_DSHOW) --
    it briefly fails to hand back a buffer even though the device is fine and the very next read
    succeeds -- so this retries a few times before reporting a real failure to the caller.
    """
    for attempt in range(_READ_RETRIES):
        was_read, frame = capture.read()
        if was_read:
            return frame.copy()
        if attempt < _READ_RETRIES - 1:
            time.sleep(_RETRY_DELAY_SECONDS)
    return None
