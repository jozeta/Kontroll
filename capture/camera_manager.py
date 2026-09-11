"""Enumerates UVC cameras and manages the two selected VideoCapture handles (face camera, document camera)."""

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Final, Literal

import cv2
import numpy as np

from capture.dshow_native_formats import NativeFormat, list_native_formats, list_video_input_device_names

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

# The two Windows capture backends an operator can choose between per camera in Settings (see
# ui/settings_window.py's "Backend" combo, and _open_camera's own docstring for why this exists at
# all -- some cameras negotiate MJPG far more reliably over one than the other). Keyed by the same
# short string config.settings persists each camera's choice under. CAP_DSHOW stays the app-wide
# default (see enumerate_cameras' own comment on why DirectShow specifically is used for
# enumeration); CAP_MSMF (Media Foundation) is the opt-in alternative.
CAMERA_BACKENDS: Final[dict[str, int]] = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF}
DEFAULT_CAMERA_BACKEND: Final = "dshow"


def backend_display_name(backend: int) -> str:
    """Human-readable name for a cv2.CAP_* backend constant, for logging/diagnostics."""
    for name, value in CAMERA_BACKENDS.items():
        if value == backend:
            return "DirectShow" if name == "dshow" else "Media Foundation"
    return str(backend)

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

# How many of each camera's most recently read frames _CameraReaderThread keeps (see
# get_recent_frames), on top of the single "latest" one it always kept before this existed. At the
# preview poll's own ~30fps, this covers a bit under a quarter of a second of real recent history --
# enough that a single momentarily-blurry frame (a hand still settling the card, a brief autofocus
# hunt) very likely has a sharper neighbor sitting right next to it in the buffer, without reaching
# so far back that a "recent" frame is actually stale relative to what the operator is looking at
# right now. See MainWindow._pick_sharpest_recent_doc_frame for the actual selection this enables.
_RECENT_FRAME_BUFFER_SIZE = 6

# A single preview-tick capture.read() call taking longer than this is a real, user-visible
# problem, not just a diagnostic curiosity: read_face_preview_frame/read_doc_preview_frame are
# called directly from ui/main_window.py's ~30fps QTimer callback on the main GUI thread, so a
# slow read blocks the whole UI (button clicks, window redraws, scrolling) for its entire
# duration -- and does so while CPU usage looks low, since the process is idle-waiting on the
# camera driver rather than computing anything. Diagnosed as a real production failure mode: a
# camera that silently falls back to an uncompressed format (MJPG not honored, see _open_camera's
# own comment on this) gets bandwidth-throttled hard over USB, and every single read blocks for
# however long the throttled frame takes to arrive. Logging this (throttled, see
# _SLOW_READ_LOG_EVERY_N below) is what turns "the app feels sluggish" into concrete evidence in
# app.log pointing at a specific camera and a specific millisecond figure.
SLOW_READ_WARN_THRESHOLD_SECONDS = 0.15

# Once a camera is producing slow reads, log only the 1st, 30th, 60th, ... occurrence rather than
# every single one -- a sustained stall (the common case, e.g. a camera stuck in an uncompressed
# format for the rest of the session) would otherwise spam app.log at ~30fps without adding any
# new information after the first occurrence already made the point.
_SLOW_READ_LOG_EVERY_N = 30

# DirectShow's own convention (not a generic OpenCV one) for cv2.CAP_PROP_AUTO_EXPOSURE: 0.75 means
# "auto", 0.25 means "manual". Confirmed against this app's own production camera diagnostics
# export, which showed both cameras reporting "Auto exposure: -1" (i.e. this property isn't
# reliably *readable* on that hardware/driver -- OpenCV's own -1-for-unsupported convention, see
# decode_fourcc's docstring for the same pattern on FourCC) while still accepting *writes* to
# CAP_PROP_EXPOSURE/CAP_PROP_BRIGHTNESS -- so this is set unconditionally on every open/change
# rather than skipped when the read-back looks unsupported. Not every UVC camera/driver honors
# manual exposure control at all; a camera that ignores this simply keeps behaving as it did before
# (same accepted-limitation pattern as set_exact_resolution's fallback).
_AUTO_EXPOSURE_VALUE: Final = 0.75
_MANUAL_EXPOSURE_VALUE: Final = 0.25


@dataclass
class ExposureSettings:
    """One camera's exposure configuration: auto, or a manual exposure/brightness pair.

    exposure/brightness are raw driver values (DirectShow's own units -- exposure is typically a
    small negative log2 range, e.g. -1..-13, roughly halving the exposure time per step down;
    brightness is commonly 0-255 but varies by device) rather than a normalized 0-1 range, since
    there's no reliable cross-device way to query a camera's actual supported range up front (the
    same reason probe_supported_resolutions has to probe candidates by trial rather than asking the
    driver directly) -- see ui/settings_window.py's slider, which is calibrated to a reasonable but
    unvalidated default range rather than a per-device-queried one.
    """

    auto: bool
    exposure: int
    brightness: int


def _apply_exposure_settings(capture: cv2.VideoCapture, settings: ExposureSettings) -> None:
    """Apply an ExposureSettings to an already-open capture -- auto mode, or an explicit manual exposure/brightness pair. See _AUTO_EXPOSURE_VALUE's comment for the DirectShow auto/manual convention."""
    capture.set(cv2.CAP_PROP_AUTO_EXPOSURE, _AUTO_EXPOSURE_VALUE if settings.auto else _MANUAL_EXPOSURE_VALUE)
    if not settings.auto:
        capture.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
        capture.set(cv2.CAP_PROP_BRIGHTNESS, settings.brightness)


# The FourCC (compression format) _open_camera requests -- pulled out to a shared constant so
# get_face_camera_stats/get_doc_camera_stats (see CameraStats) can compare it against what the
# camera actually negotiated, surfaced in ui/settings_window.py's Cameras tab diagnostics. See
# _open_camera's own comment for why an uncompressed fallback here matters.
REQUESTED_FOURCC: Final = "MJPG"

# Rolling window used to compute each camera's measured FPS in CameraStats -- long enough to
# smooth out ordinary tick-to-tick jitter, short enough that the figure shown in Settings still
# reflects what's happening right now rather than an average from a minute ago.
_FPS_WINDOW_SECONDS = 2.0


@dataclass
class CameraInfo:
    """One camera device discovered during enumeration, identified by its OpenCV device index."""

    index: int
    label: str


@dataclass
class CameraStats:
    """A snapshot of one camera's current health, for the Cameras tab's live diagnostics (see
    ui/settings_window.py) -- everything an operator needs to tell "camera is fine, the desk is
    just slow for some other reason" apart from "this specific camera is the problem, and here's
    why" (an uncompressed-fallback compression mismatch, a low measured FPS, a slow last read, or
    consecutive failures), without having to go digging in app.log."""

    is_opened: bool
    camera_index: int
    resolution: tuple[int, int] | None
    requested_fourcc: str
    actual_fourcc: str | None
    fps: float
    last_read_ms: float
    consecutive_failures: int
    slow_read_streak: int


def decode_fourcc(raw_value: float) -> str:
    """Decode a cv2.CAP_PROP_FOURCC float (a 4-byte code packed into a double) back to its 4-character string, e.g. "MJPG" or "YUY2". Returns "?" for a backend that doesn't report one (0 or negative)."""
    code = int(raw_value)
    if code <= 0:
        return "?"
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4)).strip() or "?"


# Curated cv2.CAP_PROP_* properties worth showing in the Cameras tab's "Export diagnostics"
# dialog (see ui/settings_window.py) -- far from every property OpenCV defines (dozens are
# irrelevant to a UVC webcam, or never honored by the DirectShow backend this app always uses, see
# enumerate_cameras' own comment on CAP_DSHOW), just the ones that actually affect what an
# operator sees, how fast the preview runs, or commonly explain a "why does this one camera behave
# differently" troubleshooting question. Each entry is (display label, cv2.CAP_PROP_* constant);
# FourCC is handled separately in gather_camera_diagnostics since its raw value needs decode_fourcc
# rather than being shown as a bare number.
_DIAGNOSTIC_PROPERTY_LABELS: list[tuple[str, int]] = [
    ("Frame width", cv2.CAP_PROP_FRAME_WIDTH),
    ("Frame height", cv2.CAP_PROP_FRAME_HEIGHT),
    ("Driver-reported FPS", cv2.CAP_PROP_FPS),
    ("Buffer size", cv2.CAP_PROP_BUFFERSIZE),
    ("Brightness", cv2.CAP_PROP_BRIGHTNESS),
    ("Contrast", cv2.CAP_PROP_CONTRAST),
    ("Saturation", cv2.CAP_PROP_SATURATION),
    ("Hue", cv2.CAP_PROP_HUE),
    ("Gain", cv2.CAP_PROP_GAIN),
    ("Sharpness", cv2.CAP_PROP_SHARPNESS),
    ("Gamma", cv2.CAP_PROP_GAMMA),
    ("Exposure", cv2.CAP_PROP_EXPOSURE),
    ("Auto exposure", cv2.CAP_PROP_AUTO_EXPOSURE),
    ("Auto white balance", cv2.CAP_PROP_AUTO_WB),
    ("White balance temperature", cv2.CAP_PROP_WB_TEMPERATURE),
    ("Focus", cv2.CAP_PROP_FOCUS),
    ("Autofocus", cv2.CAP_PROP_AUTOFOCUS),
    ("Zoom", cv2.CAP_PROP_ZOOM),
    ("Backlight compensation", cv2.CAP_PROP_BACKLIGHT),
    ("Pan", cv2.CAP_PROP_PAN),
    ("Tilt", cv2.CAP_PROP_TILT),
    ("Convert to RGB", cv2.CAP_PROP_CONVERT_RGB),
]


@dataclass
class CameraDiagnostics:
    """A full point-in-time snapshot of one camera's configuration and driver-reported
    capabilities, for the Cameras tab's "Export diagnostics" button -- troubleshooting-oriented, so
    it deliberately goes far beyond CameraStats' operator-facing summary (raw property values,
    backend name, every probed resolution) at the cost of being a real ~1-2s hardware round-trip,
    unlike CameraStats (which is cheap enough to refresh every preview tick).
    """

    camera_index: int
    is_opened: bool
    backend_name: str
    fourcc: str
    properties: dict[str, float]
    supported_resolutions: list[tuple[int, int, str]]  # (width, height, negotiated FourCC)
    native_formats: list[NativeFormat]  # exact DirectShow-advertised modes -- see dshow_native_formats.py


def gather_camera_diagnostics(
    capture: cv2.VideoCapture, camera_index: int, on_candidate_tried: Callable[[int, int], None] | None = None
) -> CameraDiagnostics:
    """Collect a full diagnostic snapshot of an already-open camera: every curated CAP_PROP_*
    value it currently reports, its backend name, and every resolution it actually supports (via
    probe_supported_resolutions -- the same real ~1-2s-per-camera hardware round-trip the Cameras
    tab's resolution dropdown already does).

    Leaves the capture's resolution changed as a side effect of probing, exactly like
    probe_supported_resolutions itself -- the caller is responsible for restoring whichever
    resolution is actually configured afterward (see ui/settings_window.py's caller, which already
    does this same restore for the resolution dropdown's own probe).
    """
    is_opened = capture.isOpened()
    if not is_opened:
        return CameraDiagnostics(
            camera_index=camera_index, is_opened=False, backend_name="?", fourcc="?", properties={},
            supported_resolutions=[], native_formats=[],
        )
    properties = {label: capture.get(prop_id) for label, prop_id in _DIAGNOSTIC_PROPERTY_LABELS}
    backend_name = capture.getBackendName() if hasattr(capture, "getBackendName") else "?"
    fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
    supported_resolutions = probe_supported_resolutions(capture, on_candidate_tried=on_candidate_tried)
    # A DirectShow device the already-open `capture` handle above is exclusively holding may not
    # allow a second, independent enumeration query at the same time -- list_native_formats never
    # raises and simply comes back empty in that case (see its own docstring), same "no worse than
    # not having this at all" degradation as everywhere else this module talks to real hardware.
    native_formats = list_native_formats(camera_index)
    return CameraDiagnostics(
        camera_index=camera_index,
        is_opened=True,
        backend_name=backend_name,
        fourcc=fourcc,
        properties=properties,
        supported_resolutions=supported_resolutions,
        native_formats=native_formats,
    )


def _safe_read(capture: cv2.VideoCapture) -> tuple[bool, np.ndarray | None]:
    """capture.read(), but never lets a driver-level decode failure raise out as an uncaught exception -- treats it exactly like an ordinary failed read (False, None) instead.

    Diagnosed against a real production crash: switching a camera to Media Foundation and
    requesting a specific resolution it can't cleanly deliver made OpenCV's own frame-buffer
    wrapping raise `cv2.error: ... (-215:Assertion failed) _step >= minstep in function
    'cv::Mat::Mat'` -- a driver returning a malformed/inconsistent buffer, not a Python-level
    condition anything here previously anticipated. Every caller in this module treats
    capture.read()'s result as a plain (bool, frame) tuple to test, never as something that can
    raise -- left unguarded, that uncaught exception skipped every step after the failing read
    (releasing the old handle first, then never reaching the code that would install the new one or
    resume the reader thread's own read loop), which is exactly what turned one bad frame into a
    camera stuck fully offline until the whole app was restarted. This is also why the background
    _CameraReaderThread's own read loop uses this too: an uncaught exception there would silently
    kill that thread outright (Python threads don't propagate exceptions to the main thread, they
    just die), which is a strictly worse, silent version of the same failure.
    """
    try:
        return capture.read()
    except cv2.error as error:
        # No exc_info/full traceback here -- this can legitimately fire many times in a row (once
        # per candidate resolution during a probe, times a few retry attempts during a backend
        # switch), and every occurrence is the exact same OpenCV assertion string; a full traceback
        # on each one would flood app.log for a single underlying incident without adding
        # information past the first occurrence.
        logger.warning("capture.read() raised a driver-level error, treating as a failed read: %s", error)
        return False, None


def enumerate_cameras(max_index: int = 10, on_index_tried: Callable[[int], None] | None = None) -> list[CameraInfo]:
    """Return every camera that opens and yields a readable frame, checking only indices a device
    is actually registered at rather than blindly trying every index 0..max_index-1.

    Previously this always opened+read all `max_index` (10) indices in turn regardless of how many
    cameras were actually attached -- each open is a real, multi-second-capable DirectShow hardware
    round trip, so a 1-2 camera desk still paid for 8-9 pointless opens of nonexistent devices every
    time this ran (at every app startup via prerequisites.check_cameras, and every time Settings'
    Cameras tab was opened). `list_video_input_device_names` queries DirectShow's own device
    registry first -- a near-instant metadata call that touches no hardware -- so this only opens
    the indices that actually correspond to a registered device; if that query itself fails (e.g.
    pygrabber unavailable), this falls back to the previous blind 0..max_index-1 loop, so camera
    detection never regresses, only speeds up in the common case. Discovered cameras are labeled
    with their real device name when available, generically by index otherwise.

    on_index_tried, if given, is called (with the index just tried) after each one -- this module
    has no GUI dependency of its own, but a caller running a Qt event loop can pass e.g.
    QApplication.processEvents to keep the app responsive across this operation instead of it
    reading as a freeze (see ui/settings_window.py). Kept even though the fast path above usually
    makes this whole function quick, since the blind-probing fallback path still needs it.
    """
    started_at = time.monotonic()
    device_names = list_video_input_device_names()
    if device_names:
        candidate_indices = list(range(min(len(device_names), max_index)))
        logger.info(
            "Fast DirectShow device enumeration found %d registered video input device(s) (%s); "
            "probing only those indices instead of blindly trying 0..%d",
            len(device_names), device_names, max_index - 1,
        )
    else:
        candidate_indices = list(range(max_index))
        logger.info(
            "Fast DirectShow device enumeration returned no devices (pygrabber unavailable, or a "
            "COM error) -- falling back to blindly probing indices 0..%d", max_index - 1,
        )

    discovered: list[CameraInfo] = []
    for index in candidate_indices:
        # CAP_DSHOW (DirectShow) is used instead of the Windows default (Media Foundation)
        # because MSMF is known to be slow/unreliable when repeatedly opening and closing
        # device indices during enumeration, which this probing loop does by nature.
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if capture.isOpened():
            was_read, _frame = _safe_read(capture)
            if was_read:
                name = device_names[index] if index < len(device_names) else None
                label = f"{name} (index {index})" if name else f"Camera {index}"
                discovered.append(CameraInfo(index=index, label=label))
        capture.release()
        if on_index_tried is not None:
            on_index_tried(index)
    logger.info(
        "Camera enumeration found %d camera(s) in %.2fs (checked %d index(es): %s)",
        len(discovered), time.monotonic() - started_at, len(candidate_indices), candidate_indices,
    )
    return discovered


# UVC negotiates FourCC/resolution/framerate together as one bundled "mode" -- a camera's driver
# commonly exposes several framerate options *per resolution+format combination* (e.g. 30fps at a
# small size, 15 or even 5fps at a larger one), and defaults to whichever one it/the backend
# happens to pick first if nothing requests otherwise. This app previously only ever requested
# FourCC and resolution, never framerate -- diagnosed against a real operator report of a camera
# confirmed to deliver smooth 1080p30 in other applications while measuring far below that here,
# which pointed at a negotiation gap in this app's own request rather than a hardware/bandwidth
# ceiling (already addressed separately by preferring a compressed resolution, see
# _best_candidate_preferring_compression). Requesting 30fps explicitly, in the same call as
# resolution (before the follow-up read that verifies what was actually negotiated), gives the
# driver every input needed to pick its best matching mode instead of an unrelated default -- a
# camera/driver that doesn't support 30fps at the requested resolution simply clamps to whatever it
# does support, same accepted-limitation pattern as every other capability probed this way.
_TARGET_FPS: Final = 30.0


def _apply_resolution(capture: cv2.VideoCapture, width: int, height: int) -> tuple[int, int] | None:
    """Request a resolution (and framerate, see _TARGET_FPS) and verify it by reading back a real frame. Returns the actually-decoded (width, height), or None if no frame could be read at all."""
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, _TARGET_FPS)
    # Re-assert MJPG *after* width/height, not just once at open time in _open_camera: confirmed
    # against a real operator's camera (Logitech StreamCam, natively MJPG-capable up to 1080p30 --
    # verified independently via DirectShow's own IAMStreamConfig) that renegotiating frame size on
    # DirectShow silently drops the filter graph's format node back to its uncompressed default,
    # undoing the earlier CAP_PROP_FOURCC request -- explaining why that camera kept landing on slow
    # (~200-1000ms/frame) uncompressed reads even though it was already requesting MJPG up front. A
    # camera that doesn't support MJPG at this resolution simply ignores this, same as the original
    # request.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*REQUESTED_FOURCC))
    was_read, frame = _safe_read(capture)
    if not was_read or frame is None:
        return None
    actual_height, actual_width = frame.shape[:2]
    return (actual_width, actual_height)


def _apply_resolution_and_get_fourcc(capture: cv2.VideoCapture, width: int, height: int) -> tuple[int, int, str] | None:
    """_apply_resolution, plus the FourCC actually negotiated at that resolution.

    Read *after* applying the resolution, not once at camera-open time: plenty of real UVC webcams
    only expose their compressed (MJPG) mode at some resolutions and silently drop to raw YUY2 at
    others (typically their sensor's native/highest mode, whose hardware encoder can't keep up) --
    diagnosed against a real operator report of a camera that streams smoothly in other
    applications, ruling out a simple "this camera doesn't do MJPG at all" explanation. A single
    FourCC read at open time (what this app previously relied on) can't see this per-resolution
    variation at all -- see probe_supported_resolutions/set_best_available_resolution below, both
    of which use this instead of the plain _apply_resolution to act on it.
    """
    applied = _apply_resolution(capture, width, height)
    if applied is None:
        return None
    fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
    return (applied[0], applied[1], fourcc)


def probe_supported_resolutions(
    capture: cv2.VideoCapture, on_candidate_tried: Callable[[int, int], None] | None = None
) -> list[tuple[int, int, str]]:
    """Test every candidate resolution against an already-open capture and return the distinct ones it actually applies (plus the FourCC negotiated at each one), highest resolution first.

    Leaves the capture set to whichever candidate was tried last -- callers that care about the
    final resolution (e.g. set_best_available_resolution) apply their chosen one explicitly
    afterward rather than relying on probing's side effect.

    on_candidate_tried, if given, is called (with the candidate's width/height) after each one --
    see enumerate_cameras' docstring for why (this is the ~1-2s-per-camera operation that
    docstring's ui/settings_window.py reference is about).
    """
    started_at = time.monotonic()
    found: list[tuple[int, int, str]] = []
    seen_sizes: set[tuple[int, int]] = set()
    for width, height in _RESOLUTION_CANDIDATES:
        applied = _apply_resolution_and_get_fourcc(capture, width, height)
        if applied is not None and (applied[0], applied[1]) not in seen_sizes:
            found.append(applied)
            seen_sizes.add((applied[0], applied[1]))
        if on_candidate_tried is not None:
            on_candidate_tried(width, height)
    found.sort(key=lambda resolution: resolution[0] * resolution[1], reverse=True)
    logger.info(
        "Hardware resolution probe tried %d candidate(s) and found %d supported resolution(s) in %.2fs",
        len(_RESOLUTION_CANDIDATES), len(found), time.monotonic() - started_at,
    )
    return found


def resolutions_from_native_formats(native_formats: list[NativeFormat]) -> list[tuple[int, int, str]]:
    """Collapse a camera's exact native DirectShow modes (see dshow_native_formats.list_native_formats)
    down to one (width, height, fourcc) entry per distinct resolution, in the same shape and sort
    order (highest resolution first) as probe_supported_resolutions -- so a caller that only needs
    "what resolutions does this camera support" can use this fast metadata query (no capture.read()
    at all, see list_native_formats' own docstring) instead of that function's real, ~1-2s-per-
    camera hardware round trip. A native format list commonly repeats the same resolution once per
    supported framerate and/or compression; MJPG is preferred over any other compression found at
    the same resolution, mirroring the preference _best_candidate_preferring_compression applies
    when actually choosing a resolution to use.
    """
    best_fourcc_by_size: dict[tuple[int, int], str] = {}
    for native_format in native_formats:
        size = (native_format.width, native_format.height)
        current = best_fourcc_by_size.get(size)
        if current is None or (current.upper() != REQUESTED_FOURCC.upper() and native_format.fourcc.upper() == REQUESTED_FOURCC.upper()):
            best_fourcc_by_size[size] = native_format.fourcc
    resolutions = [(width, height, fourcc) for (width, height), fourcc in best_fourcc_by_size.items()]
    resolutions.sort(key=lambda resolution: resolution[0] * resolution[1], reverse=True)
    return resolutions


def _best_candidate_preferring_compression(
    capture: cv2.VideoCapture, candidates: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Try each candidate (in the given order) against an already-open capture; among the ones that
    actually work, prefer the first whose negotiated FourCC matches REQUESTED_FOURCC over simply
    the first that works at all -- since a resolution that keeps MJPG is far less likely to hit the
    uncompressed-fallback USB-bandwidth throttling this app has repeatedly diagnosed in production
    (see SLOW_READ_WARN_THRESHOLD_SECONDS's own comment) than one that doesn't, even if that
    compressed resolution is a step down from the outright highest/lowest one available.

    Tries every candidate (rather than stopping at the first success, set_best_available_
    resolution's previous behavior) specifically to keep searching for a compressed option -- an
    acceptable cost since this only runs at camera open/switch time, not per-frame. If nothing
    negotiates the requested compression, falls back to the first candidate that worked at all
    (matching the old behavior exactly), re-applying it since the search loop may have left the
    capture on a later, differently-sized candidate.
    """
    first_working: tuple[int, int] | None = None
    for width, height in candidates:
        applied = _apply_resolution_and_get_fourcc(capture, width, height)
        if applied is None:
            continue
        applied_width, applied_height, fourcc = applied
        if first_working is None:
            first_working = (applied_width, applied_height)
        if fourcc.upper() == REQUESTED_FOURCC.upper():
            return (applied_width, applied_height)
    if first_working is not None:
        _apply_resolution(capture, first_working[0], first_working[1])
    return first_working


def set_best_available_resolution(capture: cv2.VideoCapture) -> tuple[int, int] | None:
    """Leave the capture set to the highest-area candidate resolution that actually works, preferring one that also keeps MJPG compression over a higher-resolution one that doesn't -- see _best_candidate_preferring_compression. Returns that resolution, or None if the capture can't produce a frame at all."""
    return _best_candidate_preferring_compression(capture, _RESOLUTION_CANDIDATES)


def set_lowest_available_resolution(capture: cv2.VideoCapture) -> tuple[int, int] | None:
    """Leave the capture set to the lowest-area candidate resolution that actually works, preferring one that also keeps MJPG compression -- see set_best_available_resolution/_best_candidate_preferring_compression. Used for the AUTO_LOWEST resolution preference, the factory default (see its docstring), so this is the common case at every app startup."""
    return _best_candidate_preferring_compression(capture, list(reversed(_RESOLUTION_CANDIDATES)))


def set_exact_resolution(capture: cv2.VideoCapture, resolution: tuple[int, int]) -> tuple[int, int] | None:
    """Apply an operator-chosen exact resolution, falling back to auto-best if the camera doesn't honor it."""
    applied = _apply_resolution(capture, resolution[0], resolution[1])
    if applied is not None:
        return applied
    logger.warning("Camera did not honor requested resolution %s; falling back to auto-best", resolution)
    return set_best_available_resolution(capture)


def _warn_if_slow_read(label: str, camera_index: int, elapsed_seconds: float, slow_read_streak: int) -> int:
    """Log (throttled) a warning when a preview-tick camera read took long enough to be a real UI-blocking problem -- see SLOW_READ_WARN_THRESHOLD_SECONDS. Returns the updated consecutive-slow-read streak."""
    if elapsed_seconds <= SLOW_READ_WARN_THRESHOLD_SECONDS:
        return 0
    slow_read_streak += 1
    if slow_read_streak == 1 or slow_read_streak % _SLOW_READ_LOG_EVERY_N == 0:
        logger.warning(
            "%s camera index %d preview read took %.0fms (threshold %.0fms) -- this blocks the UI "
            "thread for its duration since preview reads run on the main window's ~30fps timer; "
            "%d consecutive slow read(s) so far. Often means the camera fell back to an "
            "uncompressed format and is bandwidth-throttled over USB (see _open_camera's MJPG "
            "comment) rather than a genuine hang.",
            label,
            camera_index,
            elapsed_seconds * 1000,
            SLOW_READ_WARN_THRESHOLD_SECONDS * 1000,
            slow_read_streak,
        )
    return slow_read_streak


def _record_frame_timestamp(timestamps: list[float], now: float) -> None:
    """Append a successful-read timestamp and drop everything older than _FPS_WINDOW_SECONDS -- keeps the list bounded regardless of how long the app has been running, rather than growing forever."""
    timestamps.append(now)
    cutoff = now - _FPS_WINDOW_SECONDS
    while timestamps and timestamps[0] < cutoff:
        timestamps.pop(0)


def _measured_fps(timestamps: list[float]) -> float:
    """Compute the rolling FPS implied by a _record_frame_timestamp-maintained list -- 0.0 if there's not yet enough history (fewer than 2 reads within the window) to measure a rate from."""
    if len(timestamps) < 2:
        return 0.0
    return (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])


def _log_verbose_camera_snapshot(label: str, capture: cv2.VideoCapture, camera_index: int, backend: int) -> None:
    """Log a full point-in-time snapshot of one camera -- every curated driver-reported property
    (see _DIAGNOSTIC_PROPERTY_LABELS) plus its exact native DirectShow formats -- at DEBUG level.

    Gated on logger.isEnabledFor(DEBUG) so this (a real COM round trip for the native-format query,
    on top of a handful of property reads) only runs when an operator has actually turned on verbose
    logging (config.settings' debug_logging_enabled, applied via main.py::apply_debug_logging_setting
    before CameraManager is ever constructed) -- never on an ordinary camera open/reopen/switch.
    This is what turns "operator turned on verbose logging to diagnose a camera issue" into a
    complete snapshot landing in app.log on its own, without a separate trip through Settings ->
    Cameras -> Export diagnostics (which remains the on-demand, UI-visible equivalent -- see
    gather_camera_diagnostics -- and additionally probes actually-negotiable resolutions, which this
    does not, to stay cheap enough to run on every open).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    properties = {prop_label: capture.get(prop_id) for prop_label, prop_id in _DIAGNOSTIC_PROPERTY_LABELS}
    negotiated_fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
    backend_name = capture.getBackendName() if hasattr(capture, "getBackendName") else backend_display_name(backend)
    native_formats = list_native_formats(camera_index)
    logger.debug(
        "%s camera index %d verbose snapshot -- backend=%s negotiated_fourcc=%s properties=%s native_formats=%s",
        label, camera_index, backend_name, negotiated_fourcc, properties, [str(f) for f in native_formats],
    )


def _apply_resolution_preference(capture: cv2.VideoCapture, resolution: ResolutionPreference) -> tuple[int, int] | None:
    """Resolve a ResolutionPreference (an exact size, AUTO_LOWEST, or None for auto-best) against an already-open capture."""
    if resolution is None:
        return set_best_available_resolution(capture)
    if resolution == AUTO_LOWEST:
        return set_lowest_available_resolution(capture)
    return set_exact_resolution(capture, resolution)


def _open_camera(
    index: int,
    resolution: ResolutionPreference,
    exposure_settings: ExposureSettings | None = None,
    backend: int = cv2.CAP_DSHOW,
    label: str = "Camera",
) -> cv2.VideoCapture:
    """Open a camera at the given device index and apply the given resolution preference (exact size, AUTO_LOWEST, or None for auto-best) and exposure settings (None defaults to plain auto exposure).

    label ("Face"/"Document", from CameraManager's own two roles) is purely cosmetic, threaded
    through only so _log_verbose_camera_snapshot's DEBUG-level dump at the end of this function
    reads clearly in app.log -- see that function's own docstring for when it actually logs
    anything.

    backend picks the underlying Windows capture API (see CAMERA_BACKENDS) -- CAP_DSHOW
    (DirectShow) is the app-wide default (see enumerate_cameras' own comment for why), but some
    camera/driver combinations negotiate MJPG far more reliably over CAP_MSMF (Media Foundation)
    instead -- diagnosed against a real operator report of a camera that streamed smoothly at full
    resolution in other applications (which, on Windows, commonly use Media Foundation rather than
    the legacy DirectShow API) while this app's DSHOW-opened handle kept falling back to
    uncompressed YUY2 at every resolution. ui/settings_window.py exposes this as a per-camera
    "Backend" choice specifically so an operator can test that theory on their own hardware without
    needing a code change -- see CameraManager.set_face_camera_backend/set_doc_camera_backend.

    Caveat worth knowing: DirectShow's and Media Foundation's device-index numbering aren't
    guaranteed to agree on every system (each backend enumerates independently) -- since
    enumerate_cameras() (which populates the Settings camera-assignment dropdown) always probes via
    DSHOW, switching a specific camera's backend to MSMF *could* in principle open a different
    physical camera at the same numeric index on a system with several attached. In practice this
    is rare for a simple one-or-two-camera desk setup, and it's immediately obvious from the live
    preview if it happens -- switch back to DirectShow if so.
    """
    # A pre-open nudge via pygrabber (setting one of the camera's own native DirectShow modes
    # directly through IAMStreamConfig before cv2 ever touches the device) was tried here and
    # verified, against a real operator's camera and app.log, to be actively harmful: opening a
    # separate DirectShow graph to pre-set the format left the device in a state that made cv2's own
    # subsequent open crash repeatedly (`cv2.error: ... _step >= minstep in function 'cv::Mat::Mat'`)
    # at the requested resolution, which this module's existing self-heal logic then (correctly, but
    # unnecessarily) interpreted as "camera did not honor requested resolution" and fell back to a
    # *lower* one than before this nudge existed. Removed for that reason -- see
    # capture/dshow_native_formats.py, whose enumeration (list_native_formats, used for Settings
    # diagnostics only) is unaffected and still safe, since it never runs before cv2's own open.
    capture = cv2.VideoCapture(index, backend)
    # Reduces stale-buffered-frame buildup on backends that honor it; harmless no-op otherwise.
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _apply_exposure_settings(capture, exposure_settings or ExposureSettings(auto=True, exposure=0, brightness=0))
    # Without this, DirectShow commonly negotiates an uncompressed format (e.g. YUY2), which is
    # heavily bandwidth-limited over USB -- the camera itself then throttles its frame rate far
    # below 30fps at anything but a very low resolution, with the process sitting idle (low
    # CPU/RAM) waiting on each capture.read() rather than doing any real work. Requesting MJPG
    # compression up front (must happen before the resolution is set, since the driver picks the
    # FourCC/resolution/framerate combination together) is the standard fix; a camera that doesn't
    # support MJPG simply ignores this and behaves as before.
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*REQUESTED_FOURCC))
    if capture.isOpened():
        was_read, _frame = _safe_read(capture)
        if not was_read:
            # A camera that flatly can't stream MJPG is expected to just ignore the request above
            # (the common case) rather than break -- but on the rare driver where it does, revert
            # by reopening fresh rather than probing resolutions against a capture that can't
            # produce a single frame.
            logger.warning("Camera index %d failed to read after requesting MJPG; reopening without it", index)
            capture.release()
            capture = cv2.VideoCapture(index, backend)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            _apply_exposure_settings(capture, exposure_settings or ExposureSettings(auto=True, exposure=0, brightness=0))
        applied = _apply_resolution_preference(capture, resolution)
        logger.info(
            "Camera index %d opened via %s at resolution %s (requested %s)",
            index, backend_display_name(backend), applied, resolution or "auto",
        )
        _log_verbose_camera_snapshot(label, capture, index, backend)
    return capture


# How long to wait after releasing a capture before reopening the same device index under a
# *different* backend -- see _open_camera_verified's docstring for the diagnosed real failure this
# guards against. Not needed (and not applied) when reopening under the *same* backend a handle
# already used (e.g. the existing self-heal-after-read-failures path), which has never shown this
# problem -- only a genuine backend switch (DirectShow<->Media Foundation) hands the device off
# between two different underlying Windows capture APIs, which is what needs the settling time.
_BACKEND_SWITCH_RELEASE_DELAY_SECONDS = 0.5

# How many open+read attempts _open_camera_verified makes (spaced by
# _BACKEND_SWITCH_RELEASE_DELAY_SECONDS) before giving up on a backend switch.
_BACKEND_SWITCH_RETRY_ATTEMPTS = 3


def _open_camera_verified(
    index: int,
    resolution: ResolutionPreference,
    exposure_settings: ExposureSettings | None,
    backend: int,
    label: str = "Camera",
) -> tuple[cv2.VideoCapture, bool]:
    """Open a camera under a specific backend and confirm it actually delivers a frame, retrying a
    few times with a short delay if not, before giving up -- used specifically for an operator-
    initiated backend switch (see CameraManager.set_face_camera_backend/set_doc_camera_backend),
    not for routine camera opens (plain _open_camera, used everywhere else, has no need for this).

    Diagnosed against a real operator report: switching a camera from DirectShow to Media
    Foundation and back again left it unable to come back online at all. A UVC device doesn't
    always finish tearing down one capture session before a *different* underlying Windows capture
    API tries to claim the same device right after -- the previous, single-attempt-with-no-delay
    _open_camera call could get back a handle that reports isOpened() but never actually reads a
    frame (or fails to open at all), and nothing here used to retry, verify, or fall back, so that
    broken handle was simply kept forever. Returns (capture, succeeded) rather than raising -- a
    persistently failing capture is still returned (never None) so the caller always has *some*
    handle to keep going with, exactly like plain _open_camera's own contract.
    """
    capture = _open_camera(index, resolution, exposure_settings, backend, label)
    for attempt in range(_BACKEND_SWITCH_RETRY_ATTEMPTS):
        if capture.isOpened():
            was_read, _frame = _safe_read(capture)
            if was_read:
                return capture, True
        if attempt < _BACKEND_SWITCH_RETRY_ATTEMPTS - 1:
            logger.warning(
                "Camera index %d did not deliver a frame under %s on attempt %d/%d; retrying after a short delay",
                index, backend_display_name(backend), attempt + 1, _BACKEND_SWITCH_RETRY_ATTEMPTS,
            )
            capture.release()
            time.sleep(_BACKEND_SWITCH_RELEASE_DELAY_SECONDS)
            capture = _open_camera(index, resolution, exposure_settings, backend, label)
    return capture, False


class _CameraReaderThread:
    """Continuously reads frames from one camera's capture handle on a background thread, so a
    slow/throttled capture.read() (see SLOW_READ_WARN_THRESHOLD_SECONDS) never blocks whichever
    thread wants the latest frame -- only the most recent successfully-read frame is kept, fetched
    through get_latest_frame(), which never touches the camera hardware itself and returns
    essentially instantly.

    This is the actual fix for the diagnosed production failure mode described at
    SLOW_READ_WARN_THRESHOLD_SECONDS above: before this, read_face_preview_frame/
    read_doc_preview_frame called capture.read() directly from ui/main_window.py's ~30fps QTimer
    callback (the main GUI thread), so a bandwidth-throttled camera's slow read blocked the entire
    UI for its duration. Now that same call happens here instead, off the UI thread.

    Owns hw_lock, the only thing serializing every access to this camera's capture handle: this
    thread holds it only around its own capture.read() call; any other code that needs to touch the
    same handle directly (resolution probing, diagnostics export, exposure changes -- none of which
    go through this class) must acquire it first via CameraManager.pause_readers()/resume_readers(),
    since DirectShow handles aren't documented safe for concurrent access from multiple threads.
    """

    def __init__(self, label: str, camera_index: int, capture: cv2.VideoCapture) -> None:
        self.label = label
        self.camera_index = camera_index
        self.capture = capture
        self.hw_lock = threading.Lock()

        self._frame_lock = threading.Lock()
        self._latest_frame: np.ndarray | None = None
        # A short rolling history of recently-read frames, alongside the single "latest" one above
        # -- see get_recent_frames and _RECENT_FRAME_BUFFER_SIZE's own comment for why this exists.
        self._recent_frames: deque[np.ndarray] = deque(maxlen=_RECENT_FRAME_BUFFER_SIZE)

        self._paused = threading.Event()  # set = paused (no reads happening)
        self._stop_requested = threading.Event()

        self.read_failures = 0
        self.slow_read_streak = 0
        self.last_read_ms = 0.0
        self.frame_timestamps: list[float] = []

        self._thread = threading.Thread(target=self._run, name=f"{label}CameraReader", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_requested.is_set():
            if self._paused.is_set():
                time.sleep(0.05)
                continue
            with self.hw_lock:
                read_started_at = time.monotonic()
                was_read, frame = _safe_read(self.capture)
                read_finished_at = time.monotonic()
            self.last_read_ms = (read_finished_at - read_started_at) * 1000
            self.slow_read_streak = _warn_if_slow_read(
                self.label, self.camera_index, read_finished_at - read_started_at, self.slow_read_streak
            )
            if was_read:
                self.read_failures = 0
                _record_frame_timestamp(self.frame_timestamps, read_finished_at)
                with self._frame_lock:
                    self._latest_frame = frame
                    self._recent_frames.append(frame)
            else:
                self.read_failures += 1

    def get_latest_frame(self) -> np.ndarray | None:
        with self._frame_lock:
            return self._latest_frame

    def get_recent_frames(self) -> list[np.ndarray]:
        """Return up to the last _RECENT_FRAME_BUFFER_SIZE successfully-read frames, oldest first -- empty if none have been read yet (e.g. right after opening). Each is the same frame object get_latest_frame would have returned at the moment it was read, so a caller comparing/scoring them (e.g. MainWindow._pick_sharpest_recent_doc_frame) is choosing among genuinely distinct recent reads, not repeatedly re-fetching whatever's currently newest."""
        with self._frame_lock:
            return list(self._recent_frames)

    def pause(self) -> None:
        """Stop this thread from issuing any further capture.read() calls, blocking until any
        already-in-flight read has finished -- safe to follow immediately with direct access to
        the capture handle (resolution probing, diagnostics, exposure changes) from another thread.
        """
        self._paused.set()
        with self.hw_lock:
            pass  # waits out a read that was already in progress when pause() was called

    def resume(self) -> None:
        self._paused.clear()

    def replace_capture(self, capture: cv2.VideoCapture) -> None:
        """Point this thread at a freshly (re)opened capture handle -- used after a reconnect/switch. Must be called while paused."""
        self.capture = capture
        self.read_failures = 0
        self.slow_read_streak = 0
        self.frame_timestamps = []
        with self._frame_lock:
            self._latest_frame = None
            self._recent_frames.clear()

    def stop(self) -> None:
        self._stop_requested.set()
        self._thread.join(timeout=2.0)


class CameraManager:
    """Owns the two live VideoCapture handles used by the app: one for the face camera, one for the document camera.

    Each camera's actual capture.read() calls happen on its own background _CameraReaderThread
    (see that class) rather than on whichever thread asks for a frame -- read_face_preview_frame/
    read_doc_preview_frame (the ~30fps UI preview poll) and get_face_latest_frame/
    get_doc_latest_frame (the Capture button's still-frame grab) are both just non-blocking getters
    over that thread's latest buffered frame.
    """

    def __init__(
        self,
        face_camera_index: int,
        doc_camera_index: int,
        face_resolution: ResolutionPreference = None,
        doc_resolution: ResolutionPreference = None,
        face_exposure: ExposureSettings | None = None,
        doc_exposure: ExposureSettings | None = None,
        face_backend: int = cv2.CAP_DSHOW,
        doc_backend: int = cv2.CAP_DSHOW,
    ) -> None:
        self.face_camera_index: int = face_camera_index
        self.doc_camera_index: int = doc_camera_index
        self._face_resolution = face_resolution
        self._doc_resolution = doc_resolution
        self._face_exposure = face_exposure or ExposureSettings(auto=True, exposure=0, brightness=0)
        self._doc_exposure = doc_exposure or ExposureSettings(auto=True, exposure=0, brightness=0)
        self._face_backend = face_backend
        self._doc_backend = doc_backend

        face_capture = _open_camera(face_camera_index, face_resolution, self._face_exposure, self._face_backend, "Face")
        doc_capture = _open_camera(doc_camera_index, doc_resolution, self._doc_exposure, self._doc_backend, "Document")
        self._face_reader = _CameraReaderThread("Face", face_camera_index, face_capture)
        self._doc_reader = _CameraReaderThread("Document", doc_camera_index, doc_capture)

        logger.info(
            "Camera manager started: face camera index %d (opened=%s), document camera index %d (opened=%s)",
            face_camera_index,
            face_capture.isOpened(),
            doc_camera_index,
            doc_capture.isOpened(),
        )

    # Kept as read-only properties (not plain attributes) so existing direct callers --
    # ui/settings_window.py's resolution probing and diagnostics export -- keep working against the
    # raw cv2.VideoCapture unchanged, as long as they bracket that access with pause_readers()/
    # resume_readers() (already true: both call sites already pause/resume the live preview around
    # every such operation, extended in ui/main_window.py to also pause/resume these reader threads
    # -- see MainWindow._pause_camera_previews/_resume_camera_previews).
    @property
    def face_capture(self) -> cv2.VideoCapture:
        return self._face_reader.capture

    @property
    def doc_capture(self) -> cv2.VideoCapture:
        return self._doc_reader.capture

    def pause_readers(self) -> None:
        """Stop both background reader threads from issuing further capture.read() calls, blocking until any in-flight read finishes -- call before any direct access to face_capture/doc_capture from another thread (resolution probing, diagnostics export)."""
        self._face_reader.pause()
        self._doc_reader.pause()

    def resume_readers(self) -> None:
        self._face_reader.resume()
        self._doc_reader.resume()

    def read_face_preview_frame(self) -> np.ndarray | None:
        """Return the face camera's latest buffered frame, for live preview display only -- never blocks on the camera hardware itself (see _CameraReaderThread).

        Self-heals a dead handle: after enough consecutive failed reads, releases and reopens the
        capture at the same index/resolution rather than leaving it failing forever, and keeps
        retrying on a cooldown if the camera is still unavailable (e.g. genuinely unplugged).
        """
        if self._face_reader.read_failures > 0 and self._face_reader.read_failures % _RECONNECT_AFTER_FAILURES == 0:
            self._reopen_face_camera()
        return self._face_reader.get_latest_frame()

    def read_doc_preview_frame(self) -> np.ndarray | None:
        """Return the document camera's latest buffered frame -- see read_face_preview_frame's docstring."""
        if self._doc_reader.read_failures > 0 and self._doc_reader.read_failures % _RECONNECT_AFTER_FAILURES == 0:
            self._reopen_doc_camera()
        return self._doc_reader.get_latest_frame()

    def get_face_latest_frame(self) -> np.ndarray | None:
        """Return the face camera's latest buffered frame, for the Capture button's still-frame grab.

        Identical to read_face_preview_frame's own frame source (both just read the reader thread's
        buffer) -- kept as a separate, non-self-healing method since the Capture button shouldn't
        itself trigger a reconnect attempt (the live preview's own polling already does that, and
        firing two independent reconnects for the same dead camera at once serves no purpose).
        """
        return self._face_reader.get_latest_frame()

    def get_doc_latest_frame(self) -> np.ndarray | None:
        """Return the document camera's latest buffered frame -- see get_face_latest_frame's docstring."""
        return self._doc_reader.get_latest_frame()

    def get_doc_recent_frames(self) -> list[np.ndarray]:
        """Return the document camera's last few buffered frames (see _CameraReaderThread.get_recent_frames), oldest first.

        Document-camera-only (no face-camera equivalent): this exists specifically so the Capture
        button can pick the *sharpest* of several recent frames for OCR (see
        MainWindow._pick_sharpest_recent_doc_frame) rather than whichever single frame happened to
        be newest at the exact instant of the click -- a real source of misread characters/lines
        when that one instant caught motion blur or a momentary hologram glare, even though a
        frame from a fraction of a second earlier or later would have OCR'd cleanly. Face matching
        has no equivalent need for this: it isn't sensitive to which exact millisecond's frame is
        used the way character-level OCR is.
        """
        return self._doc_reader.get_recent_frames()

    def _reopen_face_camera(self) -> None:
        logger.warning("Face camera index %d unresponsive; attempting to reopen", self.face_camera_index)
        self._face_reader.pause()
        self._face_reader.capture.release()
        new_capture = _open_camera(
            self.face_camera_index, self._face_resolution, self._face_exposure, self._face_backend, "Face"
        )
        self._face_reader.replace_capture(new_capture)
        self._face_reader.resume()

    def _reopen_doc_camera(self) -> None:
        logger.warning("Document camera index %d unresponsive; attempting to reopen", self.doc_camera_index)
        self._doc_reader.pause()
        self._doc_reader.capture.release()
        new_capture = _open_camera(
            self.doc_camera_index, self._doc_resolution, self._doc_exposure, self._doc_backend, "Document"
        )
        self._doc_reader.replace_capture(new_capture)
        self._doc_reader.resume()

    def switch_face_camera(self, index: int, resolution: ResolutionPreference = None) -> None:
        """Release the current face camera handle and open a new one at the given device index and resolution (None = auto-best), keeping whatever exposure settings/backend are currently configured for this role."""
        self._face_reader.pause()
        self._face_reader.capture.release()
        self.face_camera_index = index
        self._face_resolution = resolution
        new_capture = _open_camera(index, resolution, self._face_exposure, self._face_backend, "Face")
        self._face_reader.replace_capture(new_capture)
        self._face_reader.resume()
        logger.info("Switched face camera to index %d (opened=%s)", index, new_capture.isOpened())

    def switch_doc_camera(self, index: int, resolution: ResolutionPreference = None) -> None:
        """Release the current document camera handle and open a new one at the given device index and resolution (None = auto-best), keeping whatever exposure settings/backend are currently configured for this role."""
        self._doc_reader.pause()
        self._doc_reader.capture.release()
        self.doc_camera_index = index
        self._doc_resolution = resolution
        new_capture = _open_camera(index, resolution, self._doc_exposure, self._doc_backend, "Document")
        self._doc_reader.replace_capture(new_capture)
        self._doc_reader.resume()
        logger.info("Switched document camera to index %d (opened=%s)", index, new_capture.isOpened())

    def set_face_camera_backend(self, backend: int) -> tuple[bool, int]:
        """Release and reopen the face camera handle under a different capture backend (see CAMERA_BACKENDS/_open_camera's own docstring for why an operator might want this) at the same device index/resolution/exposure.

        Returns (succeeded, backend_now_in_effect). If the new backend fails to actually deliver a
        frame (see _open_camera_verified -- a real, diagnosed failure mode: a UVC device doesn't
        always finish releasing from one capture API before a different one tries to claim it,
        which used to leave this camera stuck non-functional after a DSHOW<->MSMF<->DSHOW round
        trip with no way back short of restarting the app), this automatically reverts to whichever
        backend was previously in effect rather than leaving the camera broken -- the caller should
        check the returned backend_now_in_effect against what it asked for and tell the operator if
        it isn't the same.
        """
        previous_backend = self._face_backend
        self._face_reader.pause()
        try:
            self._face_reader.capture.release()
            time.sleep(_BACKEND_SWITCH_RELEASE_DELAY_SECONDS)
            new_capture, succeeded = _open_camera_verified(
                self.face_camera_index, self._face_resolution, self._face_exposure, backend, "Face"
            )
            if not succeeded and backend != previous_backend:
                logger.warning(
                    "Face camera failed to come online under %s; reverting to %s",
                    backend_display_name(backend), backend_display_name(previous_backend),
                )
                new_capture.release()
                time.sleep(_BACKEND_SWITCH_RELEASE_DELAY_SECONDS)
                new_capture, _reverted_ok = _open_camera_verified(
                    self.face_camera_index, self._face_resolution, self._face_exposure, previous_backend, "Face"
                )
                backend = previous_backend
            self._face_backend = backend
            self._face_reader.replace_capture(new_capture)
        finally:
            self._face_reader.resume()
        logger.info("Face camera backend set to %s (opened=%s)", backend_display_name(backend), new_capture.isOpened())
        return succeeded, backend

    def set_doc_camera_backend(self, backend: int) -> tuple[bool, int]:
        """Release and reopen the document camera handle under a different capture backend -- see set_face_camera_backend's docstring (including the auto-revert-on-failure behavior and its return value)."""
        previous_backend = self._doc_backend
        self._doc_reader.pause()
        try:
            self._doc_reader.capture.release()
            time.sleep(_BACKEND_SWITCH_RELEASE_DELAY_SECONDS)
            new_capture, succeeded = _open_camera_verified(
                self.doc_camera_index, self._doc_resolution, self._doc_exposure, backend, "Document"
            )
            if not succeeded and backend != previous_backend:
                logger.warning(
                    "Document camera failed to come online under %s; reverting to %s",
                    backend_display_name(backend), backend_display_name(previous_backend),
                )
                new_capture.release()
                time.sleep(_BACKEND_SWITCH_RELEASE_DELAY_SECONDS)
                new_capture, _reverted_ok = _open_camera_verified(
                    self.doc_camera_index, self._doc_resolution, self._doc_exposure, previous_backend, "Document"
                )
                backend = previous_backend
            self._doc_backend = backend
            self._doc_reader.replace_capture(new_capture)
        finally:
            self._doc_reader.resume()
        logger.info("Document camera backend set to %s (opened=%s)", backend_display_name(backend), new_capture.isOpened())
        return succeeded, backend

    def set_face_camera_resolution(self, resolution: ResolutionPreference) -> tuple[int, int] | None:
        """Re-apply a resolution preference to the already-open face camera handle without a full release/reopen."""
        self._face_resolution = resolution
        self._face_reader.pause()
        try:
            applied = _apply_resolution_preference(self.face_capture, resolution)
        finally:
            self._face_reader.resume()
        logger.info("Face camera resolution set to %s (requested %s)", applied, resolution or "auto")
        return applied

    def set_doc_camera_resolution(self, resolution: ResolutionPreference) -> tuple[int, int] | None:
        """Re-apply a resolution preference to the already-open document camera handle without a full release/reopen."""
        self._doc_resolution = resolution
        self._doc_reader.pause()
        try:
            applied = _apply_resolution_preference(self.doc_capture, resolution)
        finally:
            self._doc_reader.resume()
        logger.info("Document camera resolution set to %s (requested %s)", applied, resolution or "auto")
        return applied

    def set_face_camera_exposure(self, exposure: ExposureSettings) -> None:
        """Apply new exposure settings (auto, or manual exposure/brightness) to the already-open face camera handle without a full release/reopen. Also remembered so a later switch/reopen of this camera role re-applies it."""
        self._face_exposure = exposure
        self._face_reader.pause()
        try:
            _apply_exposure_settings(self.face_capture, exposure)
        finally:
            self._face_reader.resume()
        logger.info("Face camera exposure set to %s", exposure)

    def set_doc_camera_exposure(self, exposure: ExposureSettings) -> None:
        """Apply new exposure settings to the already-open document camera handle -- see set_face_camera_exposure's docstring."""
        self._doc_exposure = exposure
        self._doc_reader.pause()
        try:
            _apply_exposure_settings(self.doc_capture, exposure)
        finally:
            self._doc_reader.resume()
        logger.info("Document camera exposure set to %s", exposure)

    def get_face_camera_stats(self) -> CameraStats:
        """Snapshot the face camera's current health -- see CameraStats. Only reads capture.get() properties (no capture.read()), but still briefly pauses the reader thread first since DirectShow handles aren't documented safe for concurrent get()/read() from different threads -- cheap enough (a handful of property reads) to not be worth the complexity of a finer-grained lock."""
        return self._build_stats(self._face_reader)

    def get_doc_camera_stats(self) -> CameraStats:
        """Snapshot the document camera's current health -- see get_face_camera_stats' docstring."""
        return self._build_stats(self._doc_reader)

    def _build_stats(self, reader: "_CameraReaderThread") -> CameraStats:
        with reader.hw_lock:
            capture = reader.capture
            is_opened = capture.isOpened()
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if is_opened else 0
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if is_opened else 0
            actual_fourcc = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)) if is_opened else None
        return CameraStats(
            is_opened=is_opened,
            camera_index=reader.camera_index,
            resolution=(width, height) if width > 0 and height > 0 else None,
            requested_fourcc=REQUESTED_FOURCC,
            actual_fourcc=actual_fourcc,
            fps=_measured_fps(reader.frame_timestamps),
            last_read_ms=reader.last_read_ms,
            consecutive_failures=reader.read_failures,
            slow_read_streak=reader.slow_read_streak,
        )

    def release_all(self) -> None:
        """Stop both reader threads and release both camera handles, e.g. on app shutdown."""
        logger.info("Releasing both camera handles")
        self._face_reader.stop()
        self._doc_reader.stop()
        self._face_reader.capture.release()
        self._doc_reader.capture.release()
