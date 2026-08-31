"""Precise DirectShow native-format enumeration via pygrabber.

Diagnosed as necessary against a real operator report: a camera confirmed to deliver compressed
1080p30 in other applications (Teams, Meet, ...) kept landing on an uncompressed, low-resolution
mode in this app. `cv2.VideoCapture` only exposes a generic, best-effort property API
(`CAP_PROP_FRAME_WIDTH`/`HEIGHT`/`FOURCC`) -- it has no way to enumerate a camera's actual native
(resolution, compression, framerate) combinations the way a native application does, so it's
reduced to guessing plausible width/height values and reading back whatever the driver happened to
land on (see `capture/camera_manager.py`'s own `_RESOLUTION_CANDIDATES`/`probe_supported_
resolutions`). `list_native_formats` instead enumerates the camera's exact supported modes straight
from DirectShow's own `IAMStreamConfig::GetStreamCaps`, for accurate diagnostics (see
`ui/settings_window.py`'s Cameras tab) -- this ends any guessing about what a camera actually
supports, e.g. confirming a real operator's Logitech StreamCam natively advertises MJPG up to
1920x1080@30fps.

This module previously also *applied* one of those native modes directly (`IAMStreamConfig::
SetFormat`) before `_open_camera` opened the same device index with `cv2.VideoCapture`, on the
theory that some UVC drivers persist a pre-set format across that handoff. Tested against that same
real camera and its app.log: this made things *worse*, not better -- opening a separate DirectShow
graph to pre-set the format left the device in a state that made cv2's own subsequent open crash
repeatedly (`cv2.error: ... _step >= minstep in function 'cv::Mat::Mat'`) at the requested
resolution, which `camera_manager.py`'s existing self-heal logic then (correctly, but unnecessarily)
read as "camera did not honor requested resolution" and fell back to a *lower* resolution than
before this nudge existed. Removed for that reason -- see `capture/camera_manager.py::
_open_camera`'s own comment at the removal site. The actual fix for the underlying uncompressed-
fallback problem turned out to be simpler: `_apply_resolution` re-asserts `CAP_PROP_FOURCC` to MJPG
*after* setting width/height, since renegotiating frame size was silently resetting DirectShow's
format node back to its uncompressed default on that camera.
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class NativeFormat:
    """One exact native (compression, resolution) mode a camera advertises, as reported by DirectShow's own IAMStreamConfig::GetStreamCaps -- see list_native_formats."""

    index: int
    fourcc: str
    width: int
    height: int

    def __str__(self) -> str:
        return f"{self.width}x{self.height} {self.fourcc}"


def list_native_formats(device_index: int) -> list[NativeFormat]:
    """Return the camera's exact natively-advertised (compression, resolution) modes, straight from DirectShow's own IAMStreamConfig::GetStreamCaps.

    Authoritative, unlike capture/camera_manager.py's own probe_supported_resolutions, which can
    only ever discover what cv2's generic property-set API happens to land on for each candidate
    size, not what the driver actually advertises supporting. Returns an empty list (never raises)
    if enumeration itself fails for this device index -- e.g. a device already exclusively held by
    another process, or one that doesn't expose IAMStreamConfig at all.
    """
    try:
        from pygrabber.dshow_graph import FilterGraph

        graph = FilterGraph()
        graph.add_video_input_device(device_index)
        video_input = graph.get_input_device()
        raw_formats = video_input.get_formats()
        graph.remove_filters()
    except Exception:
        logger.warning("Failed to enumerate native DirectShow formats for device index %d", device_index, exc_info=True)
        return []
    return [
        NativeFormat(index=raw["index"], fourcc=raw["media_type_str"], width=raw["width"], height=raw["height"])
        for raw in raw_formats
    ]
