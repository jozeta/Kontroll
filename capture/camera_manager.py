"""Enumerates UVC cameras and manages the two selected VideoCapture handles (face camera, document camera)."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class CameraInfo:
    """One camera device discovered during enumeration, identified by its OpenCV device index."""

    index: int
    label: str


def enumerate_cameras(max_index: int = 10) -> list[CameraInfo]:
    """Probe device indices 0..max_index-1 and return the ones that open and yield a readable frame.

    OpenCV/Windows has no API to list UVC devices by name, so this is the standard way to discover
    which indices correspond to actually-connected cameras: open each index, try to read one frame,
    and keep only the ones that succeed. Devices are labeled generically by index since no
    human-readable name is available this way.
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
    return discovered


class CameraManager:
    """Owns the two live VideoCapture handles used by the app: one for the face camera, one for the document camera."""

    def __init__(self, face_camera_index: int, doc_camera_index: int) -> None:
        self.face_camera_index: int = face_camera_index
        self.doc_camera_index: int = doc_camera_index
        self.face_capture: cv2.VideoCapture = cv2.VideoCapture(face_camera_index, cv2.CAP_DSHOW)
        self.doc_capture: cv2.VideoCapture = cv2.VideoCapture(doc_camera_index, cv2.CAP_DSHOW)

    def read_face_preview_frame(self) -> np.ndarray | None:
        """Read the latest available frame from the face camera, for live preview display only."""
        was_read, frame = self.face_capture.read()
        return frame if was_read else None

    def read_doc_preview_frame(self) -> np.ndarray | None:
        """Read the latest available frame from the document camera, for live preview display only."""
        was_read, frame = self.doc_capture.read()
        return frame if was_read else None

    def switch_face_camera(self, index: int) -> None:
        """Release the current face camera handle and open a new one at the given device index."""
        self.face_capture.release()
        self.face_camera_index = index
        self.face_capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    def switch_doc_camera(self, index: int) -> None:
        """Release the current document camera handle and open a new one at the given device index."""
        self.doc_capture.release()
        self.doc_camera_index = index
        self.doc_capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    def release_all(self) -> None:
        """Release both camera handles, e.g. on app shutdown."""
        self.face_capture.release()
        self.doc_capture.release()
