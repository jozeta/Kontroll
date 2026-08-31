"""Shared "click a thumbnail, see it full-size" viewer, used by both the scan history list
(ui/main_window.py) and the flagged-persons list (ui/settings_window.py) -- kept in its own module
(rather than defined once in main_window.py and imported from there) since main_window.py already
imports SettingsWindow from ui/settings_window.py, and a reverse import back into main_window.py
would be circular.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PyQt6.QtCore import QPoint, QRect, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap, QResizeEvent, QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from document.card_detector import find_card_bounding_box
from document.dates import parse_mrz_style_date
from document.field_extraction import extract_document_fields
from document.ocr import preview_ocr_preprocessing
from face.embedder import face_bbox_area
from ui.frame_display import bgr_frame_to_pixmap

if TYPE_CHECKING:
    from storage.models import ScanRecord

logger = logging.getLogger(__name__)

ThumbnailKind = Literal["face", "document"]

# Extra margin kept around a detected face's bounding box, as a fraction of the box's own
# width/height -- a flush zoom right at the detector's own box edges tends to clip chin/forehead/
# ears, so this leaves enough headroom that the auto-zoomed view still reads as "the whole face",
# not a tight crop of just eyes-nose-mouth.
_FACE_ZOOM_MARGIN_RATIO = 0.6

# Extra margin kept around a detected card's bounding box, as a fraction of its own width/height.
# document/card_detector.py::find_card_bounding_box already applies its own small (6%) margin for
# the actual pixel-precision OCR crop, but that's tuned for a tight crop, not a preview frame --
# confirmed against a real operator report that the initial zoom in ImageEditDialog/ImageViewerWindow
# could still clip the bottom of the card, since a zoom computed to exactly touch the detected
# box's edges has zero tolerance for any detection imprecision (e.g. a slightly rotated card, or a
# brightness-threshold edge that undershoots by a few percent). A much larger margin here is cheap
# insurance -- unlike the OCR crop, this only affects how much surrounding background the initial
# framing shows, never what gets saved.
_CARD_ZOOM_MARGIN_RATIO = 0.25


def _expand_bbox_with_margin(
    bbox: tuple[int, int, int, int], margin_ratio: float, frame_width: int, frame_height: int
) -> tuple[int, int, int, int] | None:
    """Grow bbox (x1, y1, x2, y2) outward from its own center by margin_ratio of its own width/height, clamped to the frame -- shared by both the face and card auto-zoom framing (see _find_face_zoom_bbox/_find_zoom_bbox) so a tight or slightly imprecise detection still leaves headroom instead of exactly touching the true subject's edges. Returns None if the margined box degenerates (shouldn't happen in practice, but mirrors every other bbox-producing function here in never returning a zero/negative-area box)."""
    x1, y1, x2, y2 = bbox
    box_width, box_height = x2 - x1, y2 - y1
    margin_x, margin_y = box_width * margin_ratio, box_height * margin_ratio
    x1 = max(0, int(x1 - margin_x))
    y1 = max(0, int(y1 - margin_y))
    x2 = min(frame_width, int(x2 + margin_x))
    y2 = min(frame_height, int(y2 + margin_y))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)

# 1.0 = the whole image fitted inside the viewport (the old, only, behavior) -- zoom can never go
# lower than this; there's nothing useful to see beyond the full image for a single still photo.
_MIN_ZOOM_LEVEL = 1.0
# How far past "fitted" the operator can zoom in via the mouse wheel or the detected-region button.
_MAX_ZOOM_LEVEL = 6.0
# Multiplicative zoom change per wheel notch.
_ZOOM_WHEEL_STEP = 1.2
# Hard cap on the rendered pixmap's longer edge, regardless of zoom level -- without this, zooming
# in on a high-resolution document capture (this app's document camera can go up to ~4K) could
# scale up to a many-thousand-pixel QPixmap on every single wheel tick, which is both slow to
# render and a multi-hundred-MB allocation. 6000px is generous headroom above any real screen
# while keeping that bounded.
_MAX_SCALED_DIMENSION = 6000


def _find_face_zoom_bbox(image_bgr: np.ndarray, face_analysis_app: FaceAnalysis) -> tuple[int, int, int, int] | None:
    """Return the bounding box (plus margin) of the largest detected face in image_bgr, or None if no face was found.

    Selects by bounding-box area, not detection confidence -- deliberately mirrors face/embedder.py
    ::get_best_face_embedding's own selection rule (imported from there, not reimplemented, so the
    two can never drift apart). Using a different "which face is the subject" rule here than the
    one the actual face-match pipeline uses would let this zoom view highlight a different face
    than the one actually being verified against the document photo -- confusing at best, actively
    misleading at worst. See get_best_face_embedding's own docstring for the concrete misdetection
    (a smaller, off-center background face outscoring the large, centered subject on RetinaFace's
    own det_score) this rule exists to avoid.
    """
    faces = face_analysis_app.get(image_bgr)
    if not faces:
        return None
    best_face = max(faces, key=face_bbox_area)
    x1, y1, x2, y2 = best_face.bbox
    frame_height, frame_width = image_bgr.shape[:2]
    return _expand_bbox_with_margin((int(x1), int(y1), int(x2), int(y2)), _FACE_ZOOM_MARGIN_RATIO, frame_width, frame_height)


def _find_zoom_bbox(
    image_bgr: np.ndarray, kind: ThumbnailKind | None, face_analysis_app: FaceAnalysis | None
) -> tuple[int, int, int, int] | None:
    """Dispatch to the right auto-zoom region finder for a thumbnail's kind, or None if the kind is unset/unsupported or nothing was found to zoom to."""
    if kind == "face" and face_analysis_app is not None:
        return _find_face_zoom_bbox(image_bgr, face_analysis_app)
    if kind == "document":
        bbox = find_card_bounding_box(image_bgr)
        if bbox is None:
            return None
        frame_height, frame_width = image_bgr.shape[:2]
        return _expand_bbox_with_margin(bbox, _CARD_ZOOM_MARGIN_RATIO, frame_width, frame_height)
    return None


def _zoom_params_for_bbox(
    image_width: int, image_height: int, bbox: tuple[int, int, int, int]
) -> tuple[tuple[float, float], float]:
    """Convert a bounding box in original-image pixel coordinates into (center_fraction, zoom_level) -- the params _ZoomableImageView.zoom_to_region expects -- such that the box roughly fills the viewport."""
    x1, y1, x2, y2 = bbox
    center_fraction = ((x1 + x2) / 2 / image_width, (y1 + y2) / 2 / image_height)
    region_width_fraction = max((x2 - x1) / image_width, 0.01)
    region_height_fraction = max((y2 - y1) / image_height, 0.01)
    zoom_level = max(_MIN_ZOOM_LEVEL, min(1 / max(region_width_fraction, region_height_fraction), _MAX_ZOOM_LEVEL))
    return center_fraction, zoom_level


class _ZoomableImageView(QScrollArea):
    """Displays one image with scroll-wheel zoom (centered on the cursor) and click-drag panning.

    Built on QScrollArea rather than a plain QLabel specifically so panning around a zoomed-in
    image -- once it no longer fits the viewport -- comes for free from native scrollbars/dragging,
    instead of hand-rolling viewport math on top of a bare label. zoom_level is relative to
    "fitted" (1.0 = whole image visible, scaled to just fit the viewport, the only zoom level the
    previous, crop-based version of this viewer ever showed); wheel-zooming and
    zoom_to_region/zoom_to_fit move it within [_MIN_ZOOM_LEVEL, _MAX_ZOOM_LEVEL].

    The displayed image is always the *original, full* pixmap, rescaled -- never a cropped copy --
    so nothing outside whatever region was auto-detected is ever actually discarded; the operator
    can always pan/zoom out to see the rest.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # The label is purely a display surface -- mouse/wheel events should reach this scroll
        # area (for panning/zooming) rather than being swallowed by the label sitting on top of it.
        self.image_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWidget(self.image_label)

        self._original_pixmap: QPixmap | None = None
        self._zoom_level = _MIN_ZOOM_LEVEL
        self._panning = False
        self._pan_start_pos = QPoint()
        self._pan_start_scroll = (0, 0)
        self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

    def set_image(
        self, pixmap: QPixmap, initial_zoom_level: float = _MIN_ZOOM_LEVEL, initial_center_fraction: tuple[float, float] = (0.5, 0.5)
    ) -> None:
        """Load an image, optionally already framed on a specific region (e.g. a detected face/card).

        The "fitted" framing (the default, and effectively all this method did before zoom/pan was
        added) is applied immediately -- safe, since a fitted image has nothing to scroll either
        way regardless of the viewport's current geometry. Any *other* requested framing is instead
        applied on a deferred (0ms) timer: at the point this is normally called (from
        ImageViewerWindow.__init__, before the window has actually been shown), the QScrollArea's
        viewport doesn't yet have its final size, and scrollbar ranges for a not-yet-laid-out
        widget aren't reliably established either -- positioning the scroll immediately here can
        silently clamp to (0, 0), i.e. the top-left corner, rather than the intended region. This
        was especially visible for an off-center subject (a card sitting to one side of a capture)
        and easy to miss for a roughly-centered one (a face), which is exactly the asymmetry
        reported: the face zoom "mostly" working while the card zoom opened stuck in the corner.
        Deferring via a 0ms singleShot -- rather than needing a smarter one-shot showEvent guard --
        is the standard, reliable way to guarantee this runs only once the window has actually been
        shown and laid out, since ImageViewerWindow.show() always happens before control returns to
        the event loop that the deferred call is scheduled on.
        """
        self._original_pixmap = pixmap
        self._rescale(_MIN_ZOOM_LEVEL, (0.5, 0.5))
        if initial_zoom_level > _MIN_ZOOM_LEVEL or initial_center_fraction != (0.5, 0.5):
            QTimer.singleShot(0, lambda: self._rescale(initial_zoom_level, initial_center_fraction))

    def zoom_to_fit(self) -> None:
        self._rescale(_MIN_ZOOM_LEVEL, (0.5, 0.5))

    def zoom_to_region(self, center_fraction: tuple[float, float], zoom_level: float) -> None:
        self._rescale(zoom_level, center_fraction)

    def _fit_scale(self) -> float:
        """The scale factor at which the whole original image just fits the current viewport."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return 1.0
        viewport_size = self.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return 1.0
        return min(
            viewport_size.width() / self._original_pixmap.width(),
            viewport_size.height() / self._original_pixmap.height(),
        )

    def _label_px_to_fraction(self, point_px: tuple[float, float]) -> tuple[float, float]:
        """Convert a point in the *currently displayed* label pixmap's own pixel coordinates to a 0..1 fraction of the original image -- the stable representation used to keep a point fixed across a rescale."""
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull() or pixmap.width() == 0 or pixmap.height() == 0:
            return (0.5, 0.5)
        return (point_px[0] / pixmap.width(), point_px[1] / pixmap.height())

    def _fraction_to_label_px(self, fraction: tuple[float, float]) -> tuple[float, float]:
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull():
            return (0.0, 0.0)
        return (fraction[0] * pixmap.width(), fraction[1] * pixmap.height())

    def _current_center_in_label_px(self) -> tuple[float, float]:
        return (
            self.horizontalScrollBar().value() + self.viewport().width() / 2,
            self.verticalScrollBar().value() + self.viewport().height() / 2,
        )

    def _set_center_in_label_px(self, point_px: tuple[float, float]) -> None:
        self.horizontalScrollBar().setValue(round(point_px[0] - self.viewport().width() / 2))
        self.verticalScrollBar().setValue(round(point_px[1] - self.viewport().height() / 2))

    def _apply_current_scale(self) -> None:
        """Rescale the label's pixmap to reflect self._zoom_level against the current viewport size, without touching scroll position -- callers that care where the view ends up afterward set it themselves (see _rescale and the cursor-anchored zoom in wheelEvent)."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        scale = self._fit_scale() * self._zoom_level
        scale = min(scale, _MAX_SCALED_DIMENSION / self._original_pixmap.width(), _MAX_SCALED_DIMENSION / self._original_pixmap.height())
        scaled = self._original_pixmap.scaled(
            max(1, round(self._original_pixmap.width() * scale)),
            max(1, round(self._original_pixmap.height() * scale)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def _rescale(self, zoom_level: float, center_fraction: tuple[float, float]) -> None:
        """Jump to a specific zoom level, centered on a specific point (a 0..1 fraction of the original image) -- used for the initial view, zoom_to_fit/zoom_to_region, and preserving the current center across a window resize."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        self._zoom_level = max(_MIN_ZOOM_LEVEL, min(zoom_level, _MAX_ZOOM_LEVEL))
        self._apply_current_scale()
        self._set_center_in_label_px(self._fraction_to_label_px(center_fraction))

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        if self._original_pixmap is None or self._original_pixmap.isNull():
            return
        # Preserve whatever point is currently centered (in stable, resolution-independent
        # fractional terms) rather than resetting the view -- this also transparently keeps
        # "fitted" mode (zoom_level == _MIN_ZOOM_LEVEL) tracking the window size, since a fitted
        # view's center is always the image center regardless.
        current_center = self._label_px_to_fraction(self._current_center_in_label_px())
        self._rescale(self._zoom_level, current_center)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom in/out, keeping the image point under the cursor fixed on screen -- the standard "zoom to cursor" interaction."""
        if self._original_pixmap is None or self._original_pixmap.isNull():
            event.ignore()
            return

        cursor_viewport_pos = self.viewport().mapFromGlobal(event.globalPosition().toPoint())
        old_label_px = (
            self.horizontalScrollBar().value() + cursor_viewport_pos.x(),
            self.verticalScrollBar().value() + cursor_viewport_pos.y(),
        )
        cursor_fraction = self._label_px_to_fraction(old_label_px)

        zoom_multiplier = _ZOOM_WHEEL_STEP if event.angleDelta().y() > 0 else (1 / _ZOOM_WHEEL_STEP)
        self._zoom_level = max(_MIN_ZOOM_LEVEL, min(self._zoom_level * zoom_multiplier, _MAX_ZOOM_LEVEL))
        self._apply_current_scale()

        new_label_px = self._fraction_to_label_px(cursor_fraction)
        self.horizontalScrollBar().setValue(round(new_label_px[0] - cursor_viewport_pos.x()))
        self.verticalScrollBar().setValue(round(new_label_px[1] - cursor_viewport_pos.y()))
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._original_pixmap is not None:
            self._panning = True
            self._pan_start_pos = event.position().toPoint()
            self._pan_start_scroll = (self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self._pan_start_scroll[0] - delta.x())
            self.verticalScrollBar().setValue(self._pan_start_scroll[1] - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._panning:
            self._panning = False
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)


# Crop-selection rectangle color -- matches the app's previous capture-review crop tool
# (formerly ui/main_window.py's DocumentCropLabel, removed once capture stopped freezing for
# review -- see ImageEditDialog's own docstring for where that moved to).
_CROP_RECT_COLOR = QColor("#FFEB3B")

# Minimum drag distance, in on-screen (viewport) pixels -- measured at the current zoom level,
# not against the zoom-independent base_pixmap coordinates crop_rect is actually stored in -- for
# a drag to count as a real crop selection rather than an accidental single click/tiny jitter. See
# _CropZoomView.mouseReleaseEvent.
_MIN_EDIT_CROP_DRAG_PX = 20

# Brightness is an additive offset (cv2.convertScaleAbs beta); contrast is a percentage multiplier
# (mapped to alpha = value/100, so 100 = unchanged). Ranges are generous enough to visibly fix a
# genuinely over/under-exposed capture without being so wide most of the slider is useless.
_EDIT_BRIGHTNESS_RANGE = (-100, 100)
_EDIT_CONTRAST_RANGE = (50, 200)

# Sharpen strength (0 = off) for the unsharp-mask sharpening applied after brightness/contrast --
# see _sharpen_image. 0-100 rather than a raw kernel/weight value so the slider reads as a plain
# "how much" knob; mapped internally to an addWeighted blend factor.
_EDIT_SHARPEN_RANGE = (0, 100)
_EDIT_SHARPEN_MAX_WEIGHT = 1.5

# Initial window size and hard minimum for ImageEditDialog -- generous enough that the image
# preview (which now shows real full-resolution detail via scroll-wheel zoom, see _CropZoomView)
# has real room, and that the adjustments panel below it isn't cramped against the image. The
# dialog is resizable (unlike its previous fixed-size-to-sizeHint() behavior) specifically so an
# operator can make the preview bigger still on a large monitor.
_EDIT_DIALOG_INITIAL_SIZE = (1050, 850)
_EDIT_DIALOG_MIN_SIZE = (700, 600)

# Fixed height of the field-check panel's scrollable text area (see its construction in
# ImageEditDialog.__init__) -- confirmed against a real operator report that the panel's text
# alternates between a short "waiting for changes to settle…" line and a multi-row comparison
# table as sliders are dragged, and without a fixed height that swing in required layout height
# forced the whole (resizable) dialog window to visibly grow and shrink on every single slider
# tick. A fixed-height scroll area keeps the dialog's layout completely stable regardless of
# content length -- a longer table simply scrolls internally instead of resizing anything.
_FIELD_CHECK_PANEL_HEIGHT = 170

# How long to wait after the operator stops moving a slider (or changes the crop) before
# automatically re-running the field-extraction check -- long enough that a drag across the whole
# slider only triggers one real OCR/MRZ/barcode pass at the end, not one per intermediate value
# (each pass costs a few hundred ms to ~1s, which would make dragging visibly stutter if run on
# every valueChanged tick).
_FIELD_CHECK_DEBOUNCE_MS = 700

# Which of document/field_extraction.py's possible doc_fields keys to show in the edit dialog's
# extracted-fields comparison, and in what order/under what label -- curated to the identity fields
# an operator actually cares about verifying (name, personal number, dates, ...), skipping the few
# keys that exist purely for internal classification/cross-check use (document_type, optional_data)
# rather than showing every raw key a parser happens to produce.
_FIELD_DISPLAY_ORDER: list[tuple[str, str]] = [
    ("surname", "Surname"),
    ("name", "Given name(s)"),
    ("personal_number", "Personal number"),
    ("birth_date", "Date of birth"),
    ("expiry_date", "Expiry date"),
    ("nationality", "Nationality"),
    ("sex", "Sex"),
    ("document_number", "Document number"),
]


def _render_field_comparison_html(original_fields: dict[str, str], new_fields: dict[str, str]) -> str:
    """Render an "Original (saved) / New (this image)" comparison table for the fields both dicts have -- a changed value is highlighted so a real difference stands out from a row that read identically both times.

    Shows a row for every field in _FIELD_DISPLAY_ORDER that either dict has a value for -- a field
    neither found is omitted entirely rather than shown as a confusing blank/blank row.
    """
    rows = []
    for key, label in _FIELD_DISPLAY_ORDER:
        original_value = _format_field_value(key, original_fields.get(key, ""))
        new_value = _format_field_value(key, new_fields.get(key, ""))
        if not original_value and not new_value:
            continue
        changed = original_value != new_value
        new_cell_style = "color:#FF9800; font-weight:bold;" if changed else ""
        rows.append(
            f"<tr><td><b>{label}</b></td>"
            f"<td>{original_value or '<i>-</i>'}</td>"
            f"<td style='{new_cell_style}'>{new_value or '<i>-</i>'}</td></tr>"
        )
    if not rows:
        return "<i>No identity fields extracted from either version.</i>"
    return (
        "<table cellspacing='6'>"
        "<tr><th align='left'>Field</th><th align='left'>Original (saved)</th><th align='left'>New (this image)</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def _format_field_value(key: str, value: str) -> str:
    """Render one doc_fields value for display -- birth_date/expiry_date are stored in whichever
    raw format their source parser produced (MRZ's 6-digit YYMMDD or an OCR parser's ISO
    YYYY-MM-DD, see document/dates.py::parse_mrz_style_date), so both are normalized to the same
    ISO display form here rather than showing the operator two different raw encodings depending on
    which parser happened to win. Falls back to the raw value unchanged if it doesn't parse as a
    date at all (so a malformed/misread value is still visible instead of silently disappearing).
    """
    if not value:
        return ""
    if key in ("birth_date", "expiry_date"):
        parsed = parse_mrz_style_date(value, assume_past=(key == "birth_date"))
        if parsed is not None:
            return parsed.isoformat()
    return value


class _CropZoomView(QScrollArea):
    """Displays one (already brightness/contrast-adjusted) image with scroll-wheel zoom (centered
    on the cursor, mirroring _ZoomableImageView's identical wheel behavior) and click-drag
    crop-rectangle selection -- ImageEditDialog's canvas.

    Unlike _ZoomableImageView, left-drag here draws a crop rectangle instead of panning -- once
    zoomed in, panning around is done via the scrollbars this QScrollArea already provides for
    free, rather than overloading left-drag for two different purposes.

    crop_rect is tracked in base_pixmap's own pixel coordinates -- i.e. the *original, full-
    resolution* adjusted image, not whatever's currently rescaled on screen -- a stable coordinate
    space that stays valid across zoom/pan changes and across a brightness/contrast tweak (which
    changes base_pixmap's pixel content but never its size), so ImageEditDialog can convert it
    straight to original-image pixel bounds with a 1:1 scale.

    zoom_level 1.0 means "base_pixmap fitted to the current viewport", exactly like
    _ZoomableImageView's own zoom_level -- see _fit_scale/_display_scale. base_pixmap itself is
    always the full, un-shrunk adjusted image (ImageEditDialog._refresh_preview passes the real
    resolution straight through, no pre-downscale), so scrolling past zoom level 1.0 reveals real
    pixel detail instead of upscaling an already-shrunk preview -- a low-resolution base_pixmap here
    was confirmed to be exactly what made brightness/contrast/sharpen edits look pixelated
    regardless of how good the underlying edit actually was, the same root cause
    _ZoomableImageView's own docstring already documents for the plain (non-edit) viewer.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(False)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFrameShape(QScrollArea.Shape.NoFrame)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Mouse events reach this scroll area (for crop-dragging) rather than being swallowed by
        # the label sitting on top of it -- same technique _ZoomableImageView uses for panning.
        self.image_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setWidget(self.image_label)

        self._base_pixmap: QPixmap | None = None
        self._zoom_level = _MIN_ZOOM_LEVEL
        self.crop_rect: QRect | None = None
        self._drag_origin_base: QPoint | None = None
        # Screen-pixel (not base-pixel) start position of the current drag, purely to decide
        # whether it was a real crop selection or an accidental click/tiny jitter once released --
        # see mouseReleaseEvent. Measuring this in screen pixels (what the operator actually
        # perceives dragging) rather than base pixels keeps that "was this deliberate" judgment
        # consistent regardless of the current zoom level.
        self._drag_origin_viewport: QPoint | None = None
        # Fired (with no arguments) once a crop drag finishes, whether it produced a real
        # selection or was cleared as too small to count -- ImageEditDialog uses this to know when
        # to re-run its OCR check, since a changed crop region changes the effective test image the
        # same way a slider does. None (the default) means no one is listening.
        self.crop_changed: Callable[[], None] | None = None
        # Right-button drag pans (mirroring _ZoomableImageView's left-button pan) -- kept as a
        # separate button from crop selection specifically so both are available at once without
        # a mode toggle: left-drag draws/adjusts a crop rectangle, right-drag pans around when
        # zoomed in, exactly like a typical image editor's space-bar/middle-click pan convention.
        self._panning = False
        self._pan_start_pos = QPoint()
        self._pan_start_scroll = (0, 0)

    @property
    def base_pixmap(self) -> QPixmap | None:
        return self._base_pixmap

    def set_base_pixmap(self, pixmap: QPixmap) -> None:
        """Set the fit-scale ("zoom level 1.0") pixmap to display, preserving the current zoom level and crop rectangle -- called every time brightness/contrast changes, since only pixel content (not size) changes between calls."""
        self._base_pixmap = pixmap
        self._redraw()

    def reset_crop(self) -> None:
        had_crop = self.crop_rect is not None
        self.crop_rect = None
        self._drag_origin_base = None
        self._redraw()
        if had_crop and self.crop_changed is not None:
            self.crop_changed()

    def reset_zoom(self) -> None:
        self._zoom_level = _MIN_ZOOM_LEVEL
        self._redraw()

    def zoom_to_region(self, center_fraction: tuple[float, float], zoom_level: float) -> None:
        """Jump to a specific zoom level, centered on a specific point (a 0..1 fraction of base_pixmap) -- mirrors _ZoomableImageView.zoom_to_region, used here to open already-zoomed on a detected face/card region (see ImageEditDialog.__init__) instead of the whole (often mostly-background) captured frame."""
        self._zoom_level = max(_MIN_ZOOM_LEVEL, min(zoom_level, _MAX_ZOOM_LEVEL))
        self._redraw()
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        scale = self._display_scale()
        target_x = center_fraction[0] * self._base_pixmap.width() * scale
        target_y = center_fraction[1] * self._base_pixmap.height() * scale
        self.horizontalScrollBar().setValue(round(target_x - self.viewport().width() / 2))
        self.verticalScrollBar().setValue(round(target_y - self.viewport().height() / 2))

    def _fit_scale(self) -> float:
        """The scale factor at which the whole base_pixmap just fits the current viewport -- see class docstring; mirrors _ZoomableImageView._fit_scale exactly."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return 1.0
        viewport_size = self.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            return 1.0
        return min(
            viewport_size.width() / self._base_pixmap.width(),
            viewport_size.height() / self._base_pixmap.height(),
        )

    def _display_scale(self) -> float:
        """The actual on-screen pixel scale currently applied to base_pixmap -- fit scale times the operator's own zoom level on top of it."""
        return self._fit_scale() * self._zoom_level

    def _current_center_fraction(self) -> tuple[float, float]:
        """The 0..1 fraction of base_pixmap currently centered in the viewport -- used by resizeEvent to keep the same content centered across a window resize, mirroring _ZoomableImageView's identical need."""
        pixmap = self.image_label.pixmap()
        if pixmap is None or pixmap.isNull() or pixmap.width() == 0 or pixmap.height() == 0:
            return (0.5, 0.5)
        center_x = self.horizontalScrollBar().value() + self.viewport().width() / 2
        center_y = self.verticalScrollBar().value() + self.viewport().height() / 2
        return (center_x / pixmap.width(), center_y / pixmap.height())

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """Keep the same content centered when the dialog (and this view along with it) is resized -- without this, _fit_scale changing out from under an unchanged zoom_level would otherwise silently shift what's visible. See _ZoomableImageView.resizeEvent, same rationale."""
        super().resizeEvent(event)
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        center_fraction = self._current_center_fraction()
        self._redraw()
        scale = self._display_scale()
        target_x = center_fraction[0] * self._base_pixmap.width() * scale
        target_y = center_fraction[1] * self._base_pixmap.height() * scale
        self.horizontalScrollBar().setValue(round(target_x - self.viewport().width() / 2))
        self.verticalScrollBar().setValue(round(target_y - self.viewport().height() / 2))

    def _redraw(self) -> None:
        """Rescale _base_pixmap to the current display scale (fit x zoom), paint the crop rectangle (if any) on top, and display it -- the shared tail end of every zoom/crop/base-pixmap change."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        scale = self._display_scale()
        scale = min(scale, _MAX_SCALED_DIMENSION / self._base_pixmap.width(), _MAX_SCALED_DIMENSION / self._base_pixmap.height())
        scaled = self._base_pixmap.scaled(
            max(1, round(self._base_pixmap.width() * scale)),
            max(1, round(self._base_pixmap.height() * scale)),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if self.crop_rect is not None and not self.crop_rect.isNull():
            # .copy() detaches from the shared implicit data the .scaled() call above may still be
            # sharing, so drawing on it doesn't risk mutating a cached copy elsewhere.
            scaled = scaled.copy()
            painter = QPainter(scaled)
            pen = QPen(_CROP_RECT_COLOR)
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(
                QRect(
                    round(self.crop_rect.left() * scale),
                    round(self.crop_rect.top() * scale),
                    round(self.crop_rect.width() * scale),
                    round(self.crop_rect.height() * scale),
                )
            )
            painter.end()
        self.image_label.setPixmap(scaled)
        self.image_label.resize(scaled.size())

    def _viewport_pos_to_base(self, viewport_pos: QPoint) -> QPoint:
        """Convert a click position (in this scroll area's viewport-local pixels) into base_pixmap's own stable pixel coordinate space, accounting for both the current scroll offset and zoom level.

        Clamped to base_pixmap's own bounds -- a drag that continues past the label's edge (e.g.
        into the scrollbar area, or beyond the viewport while zoomed in) would otherwise produce a
        crop_rect coordinate outside the image entirely.
        """
        label_px_x = self.horizontalScrollBar().value() + viewport_pos.x()
        label_px_y = self.verticalScrollBar().value() + viewport_pos.y()
        scale = self._display_scale() or 1.0
        base_x = round(label_px_x / scale)
        base_y = round(label_px_y / scale)
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            base_x = max(0, min(base_x, self._base_pixmap.width() - 1))
            base_y = max(0, min(base_y, self._base_pixmap.height() - 1))
        return QPoint(base_x, base_y)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self._base_pixmap is not None:
            self._panning = True
            self._pan_start_pos = event.position().toPoint()
            self._pan_start_scroll = (self.horizontalScrollBar().value(), self.verticalScrollBar().value())
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton and self._base_pixmap is not None:
            viewport_pos = event.position().toPoint()
            self._drag_origin_viewport = viewport_pos
            self._drag_origin_base = self._viewport_pos_to_base(viewport_pos)
            self.crop_rect = QRect(self._drag_origin_base, self._drag_origin_base)
            self._redraw()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self.horizontalScrollBar().setValue(self._pan_start_scroll[0] - delta.x())
            self.verticalScrollBar().setValue(self._pan_start_scroll[1] - delta.y())
            event.accept()
            return
        if self._drag_origin_base is not None:
            current_base = self._viewport_pos_to_base(event.position().toPoint())
            self.crop_rect = QRect(self._drag_origin_base, current_base).normalized()
            self._redraw()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.RightButton and self._panning:
            self._panning = False
            self.viewport().unsetCursor()
            event.accept()
            return
        if self._drag_origin_base is not None:
            release_viewport = event.position().toPoint()
            current_base = self._viewport_pos_to_base(release_viewport)
            self.crop_rect = QRect(self._drag_origin_base, current_base).normalized()
            drag_span = release_viewport - self._drag_origin_viewport
            if abs(drag_span.x()) < _MIN_EDIT_CROP_DRAG_PX and abs(drag_span.y()) < _MIN_EDIT_CROP_DRAG_PX:
                self.crop_rect = None  # too small to count as a real selection -- an accidental click/jitter
            self._drag_origin_base = None
            self._drag_origin_viewport = None
            self._redraw()
            if self.crop_changed is not None:
                self.crop_changed()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Zoom in/out, keeping the image point under the cursor fixed on screen -- identical interaction to _ZoomableImageView.wheelEvent, just operating on base_pixmap/zoom_level instead of that class's own state."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            event.ignore()
            return

        cursor_viewport_pos = self.viewport().mapFromGlobal(event.globalPosition().toPoint())
        cursor_base = self._viewport_pos_to_base(cursor_viewport_pos)

        zoom_multiplier = _ZOOM_WHEEL_STEP if event.angleDelta().y() > 0 else (1 / _ZOOM_WHEEL_STEP)
        self._zoom_level = max(_MIN_ZOOM_LEVEL, min(self._zoom_level * zoom_multiplier, _MAX_ZOOM_LEVEL))
        self._redraw()

        scale = self._display_scale()
        new_label_px_x = cursor_base.x() * scale
        new_label_px_y = cursor_base.y() * scale
        self.horizontalScrollBar().setValue(round(new_label_px_x - cursor_viewport_pos.x()))
        self.verticalScrollBar().setValue(round(new_label_px_y - cursor_viewport_pos.y()))
        event.accept()


def _sharpen_image(image: np.ndarray, strength: int) -> np.ndarray:
    """Apply unsharp-mask sharpening to image, strength 0 (no-op, returns image unchanged) to 100 (strongest) -- helps OCR read a slightly soft/out-of-focus capture's printed text edges more cleanly, without the noise amplification a naive edge-enhance kernel would add.

    Standard unsharp-mask technique: blur the image, then push the original *away* from that blur
    by weight_factor -- pixels that differ most from their local blurred average (i.e. edges) are
    amplified most, flat regions barely change.
    """
    if strength <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=2.0)
    weight_factor = 1.0 + (strength / 100) * _EDIT_SHARPEN_MAX_WEIGHT
    return cv2.addWeighted(image, weight_factor, blurred, 1.0 - weight_factor, 0)


class ImageEditDialog(QDialog):
    """Lets the operator crop and adjust brightness/contrast on one saved scan image (face or
    document), applied in place to the file at image_path -- opened from ImageViewerWindow's
    "Edit image…" button.

    This is where cropping moved to once capture stopped freezing both previews for operator
    review before every single save (see ui/main_window.py::MainWindow._run_capture_and_process):
    most captures don't need cropping or a brightness/contrast fix, so it's now an explicit,
    occasional action on an already-saved thumbnail instead of a step every capture had to pass
    through. Brightness/contrast/sharpen are applied to the whole image first (live-previewed),
    then any drawn crop rectangle is cut from the *adjusted* image, matching the order an operator
    would expect (fix the image up, then trim to just the card/face).

    A genuinely low-quality source capture is hard to judge by eye alone -- "does this look
    better?" doesn't tell you whether the document fields will actually extract any better. Two
    things address that directly: the "Preview: what OCR sees" toggle swaps the displayed image for
    the literal preprocessed (grayscale, contrast-normalized, thresholded) input Tesseract receives
    (document/ocr.py::preview_ocr_preprocessing) instead of the color original, and -- for a
    document image (kind="document") -- the field-check panel runs the real document classification
    + field-extraction pipeline (document/field_extraction.py::extract_document_fields, the same
    code the actual capture pipeline uses) against the current adjusted-and-cropped image and shows
    a side-by-side "Original (saved) / New (this image)" comparison of the identity fields it found
    (surname, personal number, dates, ...), with a changed value highlighted -- concrete extracted
    data instead of a bare word count, auto-refreshed shortly after each adjustment settles (see
    _FIELD_CHECK_DEBOUNCE_MS) or on demand via "Test extraction now". Omitted entirely for a face
    image (kind="face"), which has no document fields to extract.

    Never destructively loses the original capture: the first time an image at a given path is
    edited, the original file is copied aside to "<name>.original<ext>" (if that backup doesn't
    already exist yet) before this dialog overwrites it -- see _backup_original_if_needed.
    """

    def __init__(
        self,
        image_path: str,
        title: str,
        kind: ThumbnailKind | None = None,
        face_analysis_app: FaceAnalysis | None = None,
        scan: "ScanRecord | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self._image_path = image_path
        self._kind = kind
        self._original_bgr = cv2.imread(image_path)
        self._original_doc_fields: dict[str, str] = (
            json.loads(scan.doc_fields_json) if scan is not None and scan.doc_fields_json else {}
        )
        self.applied = False

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        if self._original_bgr is None:
            layout.addWidget(QLabel("Image not found -- cannot edit."))
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(self.reject)
            buttons.button(QDialogButtonBox.StandardButton.Close).clicked.connect(self.reject)
            layout.addWidget(buttons)
            return

        # Expanding, not fixed -- the whole dialog is resizable now (see the setMinimumSize/resize
        # calls at the end of this method), so the image preview grows with it rather than being
        # capped at a small fixed canvas. See _CropZoomView's own docstring for why this also fixed
        # the pixelated-when-zoomed complaint: it displays the full, un-shrunk adjusted image now,
        # not a pre-downscaled copy.
        self.crop_view = _CropZoomView()
        self.crop_view.setMinimumSize(480, 380)
        self.crop_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.crop_view.crop_changed = self._schedule_field_check
        layout.addWidget(self.crop_view, stretch=1)

        crop_row = QHBoxLayout()
        crop_row.addWidget(QLabel("Scroll to zoom. Drag on the image to select a crop region."))
        crop_row.addStretch(1)
        reset_zoom_button = QPushButton("Reset zoom")
        reset_zoom_button.clicked.connect(self.crop_view.reset_zoom)
        crop_row.addWidget(reset_zoom_button)
        clear_crop_button = QPushButton("Clear crop")
        clear_crop_button.clicked.connect(self.crop_view.reset_crop)
        crop_row.addWidget(clear_crop_button)
        layout.addLayout(crop_row)

        # All the adjustment controls live in their own titled group, clearly separated from the
        # image above -- previously these sat as a bare stack of rows directly under a small,
        # fixed-size image canvas with no visual grouping, which read as the controls crowding the
        # image rather than being its own distinct panel.
        adjustments_group = QGroupBox("Adjustments")
        adjustments_layout = QVBoxLayout(adjustments_group)
        adjustments_layout.setSpacing(8)
        layout.addWidget(adjustments_group)

        self.ocr_preview_checkbox = QCheckBox("Preview: show what OCR sees")
        self.ocr_preview_checkbox.setToolTip(
            "Shows the exact preprocessed (grayscale, contrast-normalized, thresholded) image "
            "Tesseract reads, instead of the color original -- the real test of whether an edit "
            "helps is how this view looks, not the color image."
        )
        self.ocr_preview_checkbox.toggled.connect(self._refresh_preview)
        adjustments_layout.addWidget(self.ocr_preview_checkbox)

        brightness_row = QHBoxLayout()
        brightness_row.addWidget(QLabel("Brightness:"))
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(*_EDIT_BRIGHTNESS_RANGE)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self._on_adjustment_changed)
        brightness_row.addWidget(self.brightness_slider, stretch=1)
        adjustments_layout.addLayout(brightness_row)

        contrast_row = QHBoxLayout()
        contrast_row.addWidget(QLabel("Contrast:"))
        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(*_EDIT_CONTRAST_RANGE)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._on_adjustment_changed)
        contrast_row.addWidget(self.contrast_slider, stretch=1)
        adjustments_layout.addLayout(contrast_row)

        sharpen_row = QHBoxLayout()
        sharpen_row.addWidget(QLabel("Sharpen:"))
        self.sharpen_slider = QSlider(Qt.Orientation.Horizontal)
        self.sharpen_slider.setRange(*_EDIT_SHARPEN_RANGE)
        self.sharpen_slider.setValue(0)
        self.sharpen_slider.setToolTip("Sharpens printed text edges -- helps a slightly soft/out-of-focus capture read more cleanly.")
        self.sharpen_slider.valueChanged.connect(self._on_adjustment_changed)
        sharpen_row.addWidget(self.sharpen_slider, stretch=1)
        adjustments_layout.addLayout(sharpen_row)

        self.grayscale_checkbox = QCheckBox("Black && white")
        self.grayscale_checkbox.setToolTip(
            "Converts the image to grayscale -- can help OCR on a poorly color-balanced capture, "
            "or simply match how a scanned black-and-white document should look."
        )
        self.grayscale_checkbox.toggled.connect(self._on_adjustment_changed)
        adjustments_layout.addWidget(self.grayscale_checkbox)

        reset_button = QPushButton("Reset brightness/contrast/sharpen")
        reset_button.clicked.connect(self._on_reset_adjustments_clicked)
        adjustments_layout.addWidget(reset_button)

        # Only meaningful for a document image -- a face photo has no identity fields to extract,
        # and running card-detection/OCR/MRZ/barcode against one would just waste time.
        self._field_check_timer: QTimer | None = None
        if kind == "document":
            field_check_row = QHBoxLayout()
            self.field_check_label = QLabel("Extracted fields: not checked yet.")
            self.field_check_label.setWordWrap(True)
            self.field_check_label.setTextFormat(Qt.TextFormat.RichText)
            self.field_check_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            # Fixed-height scroll area, not the label directly -- see _FIELD_CHECK_PANEL_HEIGHT's
            # own comment for why: this is what stops the dialog itself from growing/shrinking as
            # the label's content alternates between a short status line and a multi-row table.
            field_check_scroll = QScrollArea()
            field_check_scroll.setWidget(self.field_check_label)
            field_check_scroll.setWidgetResizable(True)
            field_check_scroll.setFixedHeight(_FIELD_CHECK_PANEL_HEIGHT)
            field_check_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            field_check_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            field_check_row.addWidget(field_check_scroll, stretch=1)
            test_extraction_button = QPushButton("Test extraction now")
            test_extraction_button.clicked.connect(self._run_field_check)
            field_check_row.addWidget(test_extraction_button, alignment=Qt.AlignmentFlag.AlignTop)
            layout.addLayout(field_check_row)

            # Debounces automatic re-checks (see _schedule_field_check) so dragging a slider across
            # its whole range only triggers one real OCR/MRZ/barcode pass once it settles, not one
            # per intermediate value -- each pass costs a few hundred ms to ~1s.
            self._field_check_timer = QTimer(self)
            self._field_check_timer.setSingleShot(True)
            self._field_check_timer.timeout.connect(self._run_field_check)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Apply")
        buttons.accepted.connect(self._on_apply_clicked)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # Open already zoomed in on the detected face/card region, not the whole captured frame --
        # a document capture in particular commonly frames the whole light-box/desk area, with the
        # card itself occupying only a modest fraction of it (see document/card_detector.py's own
        # diagnosis), so without this an operator opening Edit would first have to manually scroll-
        # zoom to the small region that's actually relevant before any brightness/contrast/sharpen
        # adjustment is even visible at a useful scale. Deferred via singleShot(0, ...) for the same
        # reason ImageViewerWindow's own auto-zoom is: crop_view's scrollbar ranges aren't reliably
        # established until after this dialog has actually been shown and laid out once.
        bbox = None
        if kind is not None:
            try:
                bbox = _find_zoom_bbox(self._original_bgr, kind, face_analysis_app)
            except Exception:
                logger.warning("Auto-zoom detection failed for %s image at %s", kind, image_path, exc_info=True)
        if bbox is not None:
            image_height, image_width = self._original_bgr.shape[:2]
            center_fraction, zoom_level = _zoom_params_for_bbox(image_width, image_height, bbox)
            QTimer.singleShot(0, lambda: self.crop_view.zoom_to_region(center_fraction, zoom_level))

        # Resizable, not fixed -- an operator can make the preview larger still on a big monitor;
        # _CropZoomView.resizeEvent keeps the same content centered when this happens.
        self.setMinimumSize(*_EDIT_DIALOG_MIN_SIZE)
        self.resize(*_EDIT_DIALOG_INITIAL_SIZE)
        self._refresh_preview()
        if self._field_check_timer is not None:
            # Run once automatically so the operator sees the Original/New comparison immediately
            # on opening, without first having to touch a slider -- deferred via singleShot(0, ...)
            # rather than called directly here so the dialog itself appears right away instead of
            # this (up to ~1s) extraction pass blocking its own construction.
            QTimer.singleShot(0, self._run_field_check)

    def _adjusted_full_image(self) -> np.ndarray:
        alpha = self.contrast_slider.value() / 100
        beta = self.brightness_slider.value()
        adjusted = cv2.convertScaleAbs(self._original_bgr, alpha=alpha, beta=beta)
        adjusted = _sharpen_image(adjusted, self.sharpen_slider.value())
        if self.grayscale_checkbox.isChecked():
            # Converted back to 3-channel BGR (not left single-channel) so every downstream
            # consumer -- crop-bounds math against base_pixmap, bgr_frame_to_pixmap, cv2.imwrite --
            # keeps working against a plain BGR frame without a separate grayscale code path.
            gray = cv2.cvtColor(adjusted, cv2.COLOR_BGR2GRAY)
            adjusted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return adjusted

    def _current_test_image(self) -> np.ndarray:
        """The adjusted image, cropped to the current selection if one is drawn -- i.e. exactly what Apply would write to disk. Used for the OCR check (and by _on_apply_clicked itself), so both always reflect the same effective result.

        Deliberately NOT what crop_view displays: the crop-selection canvas always shows the full,
        uncropped adjusted image (see _refresh_preview) so the operator has the whole picture to
        drag a selection against -- cropping the display image the moment a selection exists would
        remove the very reference frame needed to keep adjusting it.
        """
        adjusted = self._adjusted_full_image()
        crop_bounds = self._crop_rect_to_image_bounds(adjusted)
        if crop_bounds is None:
            return adjusted
        x1, y1, x2, y2 = crop_bounds
        return adjusted[y1:y2, x1:x2].copy()

    def _refresh_preview(self) -> None:
        """Redraw crop_view's canvas from the current brightness/contrast/sharpen/grayscale settings -- always the *full* adjusted image (crop_view overlays its own independently-tracked crop rectangle on top of whatever's shown here), optionally shown as the OCR-preprocessed view instead of color when that toggle is checked.

        The resulting pixmap is crop_view's "base" image at full, un-shrunk resolution -- unlike the
        old behavior here (which downscaled to a small fixed preview size before this point),
        crop_view itself now handles fitting it to the viewport and only magnifies real pixel
        detail on scroll-wheel zoom (see _CropZoomView's own docstring) rather than ever upscaling
        an already-shrunk copy, which is what made brightness/contrast/sharpen edits look pixelated
        regardless of how good the underlying edit actually was.
        """
        display_image = self._adjusted_full_image()
        if self.ocr_preview_checkbox.isChecked():
            preprocessed = preview_ocr_preprocessing(display_image)
            display_image = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
        pixmap = bgr_frame_to_pixmap(display_image)
        self.crop_view.set_base_pixmap(pixmap)

    def _on_adjustment_changed(self) -> None:
        self._refresh_preview()
        self._schedule_field_check()

    def _schedule_field_check(self) -> None:
        if self._field_check_timer is None:  # no field-check panel for a face image -- see __init__
            return
        self.field_check_label.setText("Extracted fields: waiting for changes to settle…")
        self._field_check_timer.start(_FIELD_CHECK_DEBOUNCE_MS)

    def _run_field_check(self) -> None:
        """Run the real document classification + field-extraction pipeline over the current adjusted-and-cropped image and show a before/after comparison against the scan's currently-saved fields -- an objective answer to "is this edit actually helping extraction" instead of judging a low-quality image by eye. Synchronous (blocks this modal dialog only, with a wait cursor) -- matches this codebase's existing pattern for other occasional, few-hundred-ms-to-~1s operations (e.g. resolution probing, backup)."""
        if self._field_check_timer is None:
            return
        self._field_check_timer.stop()
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new_fields = extract_document_fields(self._current_test_image()).doc_fields
        finally:
            QApplication.restoreOverrideCursor()
        self.field_check_label.setText(_render_field_comparison_html(self._original_doc_fields, new_fields))

    def _on_reset_adjustments_clicked(self) -> None:
        self.brightness_slider.setValue(0)
        self.contrast_slider.setValue(100)
        self.sharpen_slider.setValue(0)

    def _crop_rect_to_image_bounds(self, image: np.ndarray) -> tuple[int, int, int, int] | None:
        """Convert the crop rectangle dragged on crop_view (already in its base_pixmap's own, zoom-independent pixel coordinates -- see _CropZoomView's docstring) into (x1, y1, x2, y2) pixel bounds within image's own resolution, or None if there's no usable selection."""
        crop_rect = self.crop_view.crop_rect
        base_pixmap = self.crop_view.base_pixmap
        if crop_rect is None or crop_rect.isNull() or base_pixmap is None or base_pixmap.isNull():
            return None

        image_height, image_width = image.shape[:2]
        scale_x = image_width / base_pixmap.width()
        scale_y = image_height / base_pixmap.height()
        x1 = max(int(crop_rect.left() * scale_x), 0)
        y1 = max(int(crop_rect.top() * scale_y), 0)
        x2 = min(int((crop_rect.right() + 1) * scale_x), image_width)
        y2 = min(int((crop_rect.bottom() + 1) * scale_y), image_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _backup_original_if_needed(self) -> None:
        path = Path(self._image_path)
        backup_path = path.with_name(f"{path.stem}.original{path.suffix}")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    def _on_apply_clicked(self) -> None:
        result = self._current_test_image()
        if self.crop_view.crop_rect is not None:
            logger.info("Operator cropped %s to %dx%d via the image editor", self._image_path, result.shape[1], result.shape[0])

        self._backup_original_if_needed()
        cv2.imwrite(self._image_path, result)
        logger.info("Operator applied brightness/contrast/sharpen/crop edits to %s", self._image_path)
        self.applied = True
        self.accept()


class ImageViewerWindow(QWidget):
    """A standalone, freely-resizable window showing one image at a much larger size, with
    scroll-wheel zoom (centered on the cursor) and click-drag panning -- see _ZoomableImageView.

    When kind is given ("face" or "document"), opens already zoomed in on the detected face or ID
    card region -- the whole original image is still there, just scrolled/scaled to that region,
    never cropped away -- since this is the common case an operator double-clicks a thumbnail for:
    to actually see the face or card clearly, not to see the whole capture including surrounding
    desk/background. A "Zoom to detected face/card" button jumps back to that framing after the
    operator has zoomed/panned elsewhere; "Fit to window" always jumps back to the whole image.
    Falls back to opening fitted with no region-jump button at all if nothing was detected (e.g. no
    face found, or the card-detection heuristic didn't find a confident match -- see
    document/card_detector.py's own caveats).

    scan/on_reanalyze, if both given, add an "Edit image…" button that opens ImageEditDialog for
    this same image_path; once an edit is applied, the operator is asked whether to re-analyze scan
    with the corrected image, via on_reanalyze (wired by callers to the same re-analyze pipeline the
    history list's own "Re-analyze" button already uses -- see
    ui/main_window.py::MainWindow._on_reanalyze_scan_requested). Omitted (both default to None) by
    callers with no scan/re-analyze concept, e.g. the flagged-persons reference-image viewer in
    ui/settings_window.py, which simply gets no Edit button.
    """

    def __init__(
        self,
        image_path: str,
        title: str,
        kind: ThumbnailKind | None = None,
        face_analysis_app: FaceAnalysis | None = None,
        scan: "ScanRecord | None" = None,
        on_reanalyze: Callable[["ScanRecord"], None] | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self._image_path = image_path
        self._kind = kind
        self._face_analysis_app = face_analysis_app
        self._scan = scan
        self._on_reanalyze = on_reanalyze
        # (center_fraction, zoom_level) for the detected region, once/if found -- see
        # _try_find_zoom_region. None means nothing was detected (or kind is None), in which case
        # zoom_to_region_button never appears.
        self._zoom_region_params: tuple[tuple[float, float], float] | None = None

        layout = QVBoxLayout(self)

        self.image_view = _ZoomableImageView()
        layout.addWidget(self.image_view, stretch=1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        self.fit_button = QPushButton("Fit to window")
        self.fit_button.clicked.connect(self.image_view.zoom_to_fit)
        button_row.addWidget(self.fit_button)
        self.zoom_to_region_button = QPushButton()
        self.zoom_to_region_button.clicked.connect(self._on_zoom_to_region_clicked)
        self.zoom_to_region_button.setVisible(False)
        button_row.addWidget(self.zoom_to_region_button)
        if scan is not None and on_reanalyze is not None:
            self.edit_button = QPushButton("Edit image…")
            self.edit_button.clicked.connect(self._on_edit_image_clicked)
            button_row.addWidget(self.edit_button)
        layout.addLayout(button_row)

        # Sized before loading the image -- see _ZoomableImageView's docstring on why "fit" scale
        # depends on the viewport already having its real size; a resize after loading would leave
        # the very first frame computed against a stale/default size.
        self.resize(900, 700)

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_view.hide()
            self.fit_button.setVisible(False)
            error_label = QLabel("Image not found")
            error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.insertWidget(0, error_label)
            return

        initial_center, initial_zoom = self._try_find_zoom_region(image_path, pixmap, face_analysis_app)
        self.image_view.set_image(pixmap, initial_zoom_level=initial_zoom, initial_center_fraction=initial_center)

    def _try_find_zoom_region(
        self, image_path: str, pixmap: QPixmap, face_analysis_app: FaceAnalysis | None
    ) -> tuple[tuple[float, float], float]:
        """Return the (center_fraction, zoom_level) to open the viewer at: the detected face/card region if one was found, else the whole image fitted. Never raises -- a detection failure just falls back to the fitted view, same as no kind being given at all."""
        if self._kind is not None:
            image_bgr = cv2.imread(image_path)
            if image_bgr is not None:
                # Face detection in particular is a real, if brief (well under a second on CPU for
                # one still image), blocking call -- a wait cursor matches how this codebase
                # already signals other one-off blocking operations (e.g. resolution probing,
                # backup) elsewhere.
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                try:
                    bbox = _find_zoom_bbox(image_bgr, self._kind, face_analysis_app)
                except Exception:
                    logger.warning("Auto-zoom detection failed for %s capture at %s", self._kind, image_path, exc_info=True)
                    bbox = None
                finally:
                    QApplication.restoreOverrideCursor()
                if bbox is not None:
                    self._zoom_region_params = _zoom_params_for_bbox(pixmap.width(), pixmap.height(), bbox)
                    self.zoom_to_region_button.setText(
                        "Zoom to detected face" if self._kind == "face" else "Zoom to detected card"
                    )
                    self.zoom_to_region_button.setVisible(True)
                    return self._zoom_region_params
        return ((0.5, 0.5), _MIN_ZOOM_LEVEL)

    def _on_zoom_to_region_clicked(self) -> None:
        if self._zoom_region_params is not None:
            center_fraction, zoom_level = self._zoom_region_params
            self.image_view.zoom_to_region(center_fraction, zoom_level)

    def _on_edit_image_clicked(self) -> None:
        """Open ImageEditDialog for this image; on Apply, reload the viewer's own display and offer to re-analyze the scan with the corrected image."""
        dialog = ImageEditDialog(
            self._image_path,
            self.windowTitle(),
            kind=self._kind,
            face_analysis_app=self._face_analysis_app,
            scan=self._scan,
            parent=self,
        )
        dialog.exec()
        if not dialog.applied:
            return

        pixmap = QPixmap(self._image_path)
        if not pixmap.isNull():
            self.image_view.set_image(pixmap)

        assert self._scan is not None and self._on_reanalyze is not None  # guarded by edit_button's own construction
        confirm = QMessageBox.question(
            self,
            "Re-analyze scan",
            "Image updated. Re-run face-match and document analysis on this scan using the "
            "corrected image?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self._on_reanalyze(self._scan)


class ClickableThumbnailLabel(QLabel):
    """A thumbnail QLabel that opens the underlying image full-size in a resizable ImageViewerWindow on double-click.

    kind/face_analysis_app, if given, let the opened viewer auto-zoom to the detected face or ID
    card region -- see ImageViewerWindow's docstring. scan/on_reanalyze, if both given, thread
    through to the same viewer's optional "Edit image…" button -- see ImageViewerWindow's docstring.
    All four default to None/unset, which keeps this label's previous behavior (open showing the
    full, unzoomed image, no edit capability) unchanged for any caller that doesn't pass them.
    """

    def __init__(
        self,
        image_path: str | None,
        title: str,
        kind: ThumbnailKind | None = None,
        face_analysis_app: FaceAnalysis | None = None,
        scan: "ScanRecord | None" = None,
        on_reanalyze: Callable[["ScanRecord"], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._image_path = image_path
        self._title = title
        self._kind = kind
        self._face_analysis_app = face_analysis_app
        self._scan = scan
        self._on_reanalyze = on_reanalyze
        # Kept alive by this reference for as long as the thumbnail itself lives -- a top-level
        # PyQt widget with no Python reference held anywhere is garbage-collected out from under
        # its own open window.
        self._viewer: ImageViewerWindow | None = None

    def mouseDoubleClickEvent(self, event) -> None:
        if self._image_path and Path(self._image_path).exists():
            self._viewer = ImageViewerWindow(
                self._image_path,
                self._title,
                kind=self._kind,
                face_analysis_app=self._face_analysis_app,
                scan=self._scan,
                on_reanalyze=self._on_reanalyze,
            )
            self._viewer.show()
        super().mouseDoubleClickEvent(event)
