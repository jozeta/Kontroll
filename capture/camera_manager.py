"""Enumerates UVC cameras and manages the two selected VideoCapture handles (face camera, document camera)."""

import logging
from dataclasses import dataclass
from typing import Callable, Final, Literal

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Sentinel for the "auto, lowest available" resolution preference -- distinct from None (which
# means "auto, highest available", the original and only meaning of an unset resolution before
# this was added). This is the factory default for the face camera specifically (see
# config.settings.DEFAULT_FACE_CAMERA_RESOLUTION), so a fresh install starts it at the lowest CPU/
# bandwidth cost rather than maxing out whatever the camera supports -- the document camera
# defaults to None (best available) instead, since only its frame goes through OCR (see
# config.settings.DEFAULT_DOC_CAMERA_RESOLUTION for why). An operator can still explicitly pick
# either mode, or an exact resolution, for either camera in Settings.
AUTO_LOWEST: Final = "auto_lowest"

ResolutionPreference = tuple[int, int] | None | Literal["auto_lowest"]

# Common UVC sensor resolutions, highest first. Used both to probe what a given camera actually
# supports (for the settings UI) and to pick the best one automatically -- OpenCV/DirectShow has
# no "list supported modes" API, so the standard trick is to request each candidate and read back
# the *actual* decoded frame size, which is authoritative over VideoCapture.get() (some backends
# report the requested value back even when they silently clamped to something else).
_RESOLUTION_CANDIDATES: list[tuple[int, int]] = [
    (3840, 2160),
    (2592, 1944),
    (2048, 1536),
    (1920, 1080),
    (1600, 1200),
    (1280, 960),
    (1280, 720),
    (1024, 768),
    (800, 600),
    (640, 480),
]

# Consecutive failed reads (from the ~30fps preview poll in ui/main_window.py) before
# CameraManager attempts to release and reopen a camera handle itself, rather than just leaving it
# failing forever. ~1.5s -- a little after the UI's own "signal lost" warning already appears, so
# the operator sees the warning and then (usually) sees it clear itself as the reopen succeeds.
_RECONNECT_AFTER_FAILURES = 45


@dataclass
class CameraInfo:
    """One camera device discovered during enumeration, identified by its OpenCV device index."""

    index: int
    label: str


def enumerate_cameras(max_index: int = 10, on_index_tried: Callable[[int], None] | None = None) -> list[CameraInfo]:
    """Probe device indices 0..max_index-1 and return the ones that open and yield a readable frame.

    OpenCV/Windows has no API to list UVC devices by name, so this is the standard way to discover
    which indices correspond to actually-connected cameras: open each index, try to read one frame,
    and keep only the ones that succeed. Devices are labeled generically by index since no
    human-readable name is available this way.

    on_index_tried, if given, is called (with the index just tried) after each one -- this module
    has no GUI dependency of its own, but a caller running a Qt event loop can pass e.g.
    QApplication.processEvents to keep the app responsive across this multi-second, multi-device
    blocking operation instead of it reading as a freeze (see ui/settings_window.py).
    """
    discovered: list[CameraInfo] = []
    for index in range(max_index):
        # CAP_DSHOW (DirectShow) is used instead of the Windows default (Media Foundation)
        # because MSMF is known to be slow/unreliable when repeatedly opening and closing
        # device indices during enumeration, which this probing loop does by nature.
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            was_read, _frame = capture.read()
            if was_read:
                discovered.append(CameraInfo(index=index, label=f"Camera {index}"))
        capture.release()
        if on_index_tried is not None:
            on_index_tried(index)
    return discovered


def _apply_resolution(capture: cv2.VideoCapture, width: int, height: int) -> tuple[int, int] | None:
    """Request a resolution and verify it by reading back a real frame. Returns the actually-decoded (width, height), or None if no frame could be read at all."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    was_read, frame = capture.read()
    if not was_read or frame is None:
        return None
    actual_height, actual_width = frame.shape[:2]
    return (actual_width, actual_height)


def probe_supported_resolutions(
    capture: cv2.VideoCapture, on_candidate_tried: Callable[[int, int], None] | None = None
) -> list[tuple[int, int]]:
    """Test every candidate resolution against an already-open capture and return the distinct ones it actually applies, highest first.

    Leaves the capture set to whichever candidate was tried last -- callers that care about the
    final resolution (e.g. set_best_available_resolution) apply their chosen one explicitly
    afterward rather than relying on probing's side effect.

    on_candidate_tried, if given, is called (with the candidate's width/height) after each one --
    see enumerate_cameras' docstring for why (this is the ~1-2s-per-camera operation that
    docstring's ui/settings_window.py reference is about).
    """
    found: list[tuple[int, int]] = []
    for width, height in _RESOLUTION_CANDIDATES:
        applied = _apply_resolution(capture, width, height)
        if applied is not None and applied not in found:
            found.append(applied)
        if on_candidate_tried is not None:
            on_candidate_tried(width, height)
    found.sort(key=lambda resolution: resolution[0] * resolution[1], reverse=True)
    return found


def set_best_available_resolution(capture: cv2.VideoCapture) -> tuple[int, int] | None:
    """Leave the capture set to the highest-area candidate resolution that actually works. Returns that resolution, or None if the capture can't produce a frame at all.

    Tries _RESOLUTION_CANDIDATES in its existing highest-to-lowest order and stops at the first
    one that works, rather than probing every remaining (necessarily-smaller) candidate too via
    probe_supported_resolutions -- that full 10-candidate probe is only actually needed by the
    Settings window's resolution dropdown, which wants the *complete* list of supported
    resolutions to offer as choices. This function only ever needs the single best answer, and is
    on the path every app startup and camera switch takes (see config.settings.
    DEFAULT_FACE_CAMERA_RESOLUTION/DEFAULT_DOC_CAMERA_RESOLUTION and AUTO_LOWEST above), so the
    full probe here was needlessly turning "open the camera" into up to 10 blocking hardware
    round-trips when as few as 1 usually suffice.
    """
    for width, height in _RESOLUTION_CANDIDATES:
        applied = _apply_resolution(capture, width, height)
        if applied is not None:
            return applied
    return None


def set_lowest_available_resolution(capture: cv2.VideoCapture) -> tuple[int, int] | None:
    """Leave the capture set to the lowest-area candidate resolution that actually works. Returns that resolution, or None if the capture can't produce a frame at all.

    Mirrors set_best_available_resolution but tries _RESOLUTION_CANDIDATES lowest-to-highest
    (reversed) and stops at the first success -- used for the AUTO_LOWEST resolution preference,
    the factory default (see its docstring), so this is the common case at every app startup.
    """
    for width, height in reversed(_RESOLUTION_CANDIDATES):
        applied = _apply_resolution(capture, width, height)
        if applied is not None:
            return applied
    return None


def set_exact_resolution(capture: cv2.VideoCapture, resolution: tuple[int, int]) -> tuple[int, int] | None:
    """Apply an operator-chosen exact resolution, falling back to auto-best if the camera doesn't honor it."""
    applied = _apply_resolution(capture, resolution[0], resolution[1])
    if applied is not None:
        return applied
    logger.warning("Camera did not honor requested resolution %s; falling back to auto-best", resolution)
    return set_best_available_resolution(capture)


def _apply_resolution_preference(capture: cv2.VideoCapture, resolution: ResolutionPreference) -> tuple[int, int] | None:
    """Resolve a ResolutionPreference (an exact size, AUTO_LOWEST, or None for auto-best) against an already-open capture."""
    if resolution is None:
        return set_best_available_resolution(capture)
    if resolution == AUTO_LOWEST:
        return set_lowest_available_resolution(capture)
    return set_exact_resolution(capture, resolution)


def _open_camera(index: int, resolution: ResolutionPreference) -> cv2.VideoCapture:
    """Open a camera at the given device index and apply the given resolution preference (exact size, AUTO_LOWEST, or None for auto-best)."""
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    # Reduces stale-buffered-frame buildup on backends that honor it; harmless no-op otherwise.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    # Without this, DirectShow commonly negotiates an uncompressed format (e.g. YUY2), which is
    # heavily bandwidth-limited over USB -- the camera itself then throttles its frame rate far
    # below 30fps at anything but a very low resolution, with the process sitting idle (low
    # CPU/RAM) waiting on each capture.read() rather than doing any real work. Requesting MJPG
    # compression up front (must happen before the resolution is set, since the driver picks the
    # FourCC/resolution/framerate combination together) is the standard fix; a camera that doesn't
    # support MJPG simply ignores this and behaves as before.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
    if capture.isOpened():
        was_read, _frame = capture.read()
        if not was_read:
            # A camera that flatly can't stream MJPG is expected to just ignore the request above
            # (the common case) rather than break -- but on the rare driver where it does, revert
            # by reopening fresh rather than probing resolutions against a capture that can't
            # produce a single frame.
            logger.warning("Camera index %d failed to read after requesting MJPG; reopening without it", index)
            capture.release()
            capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        applied = _apply_resolution_preference(capture, resolution)
        logger.info("Camera index %d opened at resolution %s (requested %s)", index, applied, resolution or "auto")
    return capture


class CameraManager:
    """Owns the two live VideoCapture handles used by the app: one for the face camera, one for the document camera."""

    def __init__(
        self,
        face_camera_index: int,
        doc_camera_index: int,
        face_resolution: ResolutionPreference = None,
        doc_resolution: ResolutionPreference = None,
    ) -> None:
        self.face_camera_index: int = face_camera_index
        self.doc_camera_index: int = doc_camera_index
        self._face_resolution = face_resolution
        self._doc_resolution = doc_resolution
        self.face_capture: cv2.VideoCapture = _open_camera(face_camera_index, face_resolution)
        self.doc_capture: cv2.VideoCapture = _open_camera(doc_camera_index, doc_resolution)
        self._face_read_failures = 0
        self._doc_read_failures = 0
        logger.info(
            "Camera manager started: face camera index %d (opened=%s), document camera index %d (opened=%s)",
            face_camera_index,
            self.face_capture.isOpened(),
            doc_camera_index,
            self.doc_capture.isOpened(),
        )

    def read_face_preview_frame(self) -> np.ndarray | None:
        """Read the latest available frame from the face camera, for live preview display only.

        Self-heals a dead handle: after enough consecutive failed reads, releases and reopens the
        capture at the same index/resolution rather than leaving it failing forever, and keeps
        retrying on a cooldown if the camera is still unavailable (e.g. genuinely unplugged).
        """
        was_read, frame = self.face_capture.read()
        if was_read:
            self._face_read_failures = 0
            return frame
        self._face_read_failures += 1
        if self._face_read_failures % _RECONNECT_AFTER_FAILURES == 0:
            logger.warning("Face camera index %d unresponsive; attempting to reopen", self.face_camera_index)
            self.face_capture.release()
            self.face_capture = _open_camera(self.face_camera_index, self._face_resolution)
        return None

    def read_doc_preview_frame(self) -> np.ndarray | None:
        """Read the latest available frame from the document camera, for live preview display only.

        See read_face_preview_frame's docstring -- same self-healing reconnect behavior.
        """
        was_read, frame = self.doc_capture.read()
        if was_read:
            self._doc_read_failures = 0
            return frame
        self._doc_read_failures += 1
        if self._doc_read_failures % _RECONNECT_AFTER_FAILURES == 0:
            logger.warning("Document camera index %d unresponsive; attempting to reopen", self.doc_camera_index)
            self.doc_capture.release()
            self.doc_capture = _open_camera(self.doc_camera_index, self._doc_resolution)
        return None

    def switch_face_camera(self, index: int, resolution: ResolutionPreference = None) -> None:
        """Release the current face camera handle and open a new one at the given device index and resolution (None = auto-best)."""
        self.face_capture.release()
        self.face_camera_index = index
        self._face_resolution = resolution
        self.face_capture = _open_camera(index, resolution)
        self._face_read_failures = 0
        logger.info("Switched face camera to index %d (opened=%s)", index, self.face_capture.isOpened())

    def switch_doc_camera(self, index: int, resolution: ResolutionPreference = None) -> None:
        """Release the current document camera handle and open a new one at the given device index and resolution (None = auto-best)."""
        self.doc_capture.release()
        self.doc_camera_index = index
        self._doc_resolution = resolution
        self.doc_capture = _open_camera(index, resolution)
        self._doc_read_failures = 0
        logger.info("Switched document camera to index %d (opened=%s)", index, self.doc_capture.isOpened())

    def set_face_camera_resolution(self, resolution: ResolutionPreference) -> tuple[int, int] | None:
        """Re-apply a resolution preference to the already-open face camera handle without a full release/reopen."""
        self._face_resolution = resolution
        applied = _apply_resolution_preference(self.face_capture, resolution)
        logger.info("Face camera resolution set to %s (requested %s)", applied, resolution or "auto")
        return applied

    def set_doc_camera_resolution(self, resolution: ResolutionPreference) -> tuple[int, int] | None:
        """Re-apply a resolution preference to the already-open document camera handle without a full release/reopen."""
        self._doc_resolution = resolution
        applied = _apply_resolution_preference(self.doc_capture, resolution)
        logger.info("Document camera resolution set to %s (requested %s)", applied, resolution or "auto")
        return applied

    def release_all(self) -> None:
        """Release both camera handles, e.g. on app shutdown."""
        logger.info("Releasing both camera handles")
        self.face_capture.release()
        self.doc_capture.release()
