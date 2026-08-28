"""Locates the ID card/document within a captured document-camera frame and crops+upscales to it before OCR.

Diagnosed from real production captures (see the operator-reported "OCR can't extract names/DOB"
issue): the document camera frames the whole light-box/desk area, not just the card, so even a
high-resolution capture (e.g. 1920x1080) can leave the card itself occupying only ~25-30% of the
frame -- well under the effective pixel density Tesseract needs for the small printed text on an
ID-1 card. Whole-word OCR confidence can still look fine (Tesseract finds the words it does find
with confidence), while position-sensitive parsing (document/dl_field_parser.py's per-line field
anchors, or the MRZ's fixed-width lines) intermittently fails because a line break or a leading
digit gets dropped at that pixel density.

This is a heuristic best-effort crop, not a validated implementation -- same caveat as
dl_field_parser.py and barcode_parser.py. It assumes the app's own lightbox-style capture setup:
a light-colored card against a much darker background (true of every production screenshot seen
so far), found via simple brightness thresholding rather than a trained detector. Any capture
setup where that assumption doesn't hold (a light background, an off-angle non-rectangular view)
just fails to find a confident match and falls back to the original, uncropped frame -- this never
makes a capture worse, only possibly no better.

Also corrects for rotation: a card set down even slightly turned on the desk (routine at a manual
desk, not a guided scanner slot) previously still got an axis-aligned crop, which both left extra
background inside the frame (an axis-aligned box around a rotated rectangle is always bigger than
the rectangle itself, working against the whole point of a tight crop) and left the printed text
lines themselves tilted -- exactly the kind of thing that trips up Tesseract's line segmentation
and, downstream, dl_field_parser.py's per-line field anchors, independent of raw pixel density.
detect_and_crop_card now finds the card's minimum-area (i.e. possibly rotated) rectangle and warps
it to an upright, tightly-cropped view instead of taking an axis-aligned bounding box.
"""

import cv2
import numpy as np

# A found bright region must span at least this fraction of the frame to be worth cropping to (too
# small: almost certainly a lighting artifact or one of the light-box's corner registration marks,
# not a card) and at most this fraction (if it already covers nearly the whole frame, cropping
# would gain nothing).
_MIN_CARD_AREA_RATIO = 0.03
_MAX_CARD_AREA_RATIO = 0.95

# ID-1 format (as used by driving licences and most national ID cards): 85.6mm x 53.98mm.
_CARD_ASPECT_RATIO = 85.6 / 53.98
_ASPECT_RATIO_TOLERANCE = 0.35

# Extra margin kept around the detected rectangle (as a fraction of its own size, applied in its
# own -- possibly rotated -- orientation) so a card edge/corner isn't clipped if the brightness
# threshold slightly underestimates its true extent.
_CROP_MARGIN_RATIO = 0.06

# After cropping, upscale (never downscale) so the card's own text has enough effective pixel
# density for OCR even if the original capture resolution was modest.
_MIN_CROPPED_WIDTH = 1600


def _order_corners_clockwise(points: np.ndarray) -> np.ndarray:
    """Sort 4 arbitrary corner points into (top-left, top-right, bottom-right, bottom-left) order.

    cv2.boxPoints doesn't guarantee a fixed corner order relative to the frame's own axes (it
    depends on the rotated rect's angle), so the order has to be recovered from the points'
    positions themselves rather than assumed -- this is the standard sum/difference trick: the
    top-left corner has the smallest x+y, the bottom-right the largest; the top-right has the
    smallest y-x, the bottom-left the largest.
    """
    ordered = np.zeros((4, 2), dtype=np.float32)
    coordinate_sum = points.sum(axis=1)
    coordinate_diff = np.diff(points, axis=1).flatten()
    ordered[0] = points[np.argmin(coordinate_sum)]  # top-left
    ordered[2] = points[np.argmax(coordinate_sum)]  # bottom-right
    ordered[1] = points[np.argmin(coordinate_diff)]  # top-right
    ordered[3] = points[np.argmax(coordinate_diff)]  # bottom-left
    return ordered


def _warp_to_upright_card(frame: np.ndarray, corners: np.ndarray) -> np.ndarray | None:
    """Perspective-warp a (possibly rotated) quadrilateral region of frame to an upright, landscape-oriented rectangle. Returns None if the corners describe a degenerate (near-zero-area) region."""
    top_left, top_right, bottom_right, bottom_left = corners

    # Measured from the actual corner points rather than trusting cv2.minAreaRect's own width/
    # height directly -- more robust to the small imprecisions boxPoints/rounding can introduce.
    top_edge = np.linalg.norm(top_right - top_left)
    bottom_edge = np.linalg.norm(bottom_right - bottom_left)
    left_edge = np.linalg.norm(bottom_left - top_left)
    right_edge = np.linalg.norm(bottom_right - top_right)
    edge_a = max(top_edge, bottom_edge)
    edge_b = max(left_edge, right_edge)
    if edge_a < 1 or edge_b < 1:
        return None

    # Always output landscape (wider than tall), matching the ID-1 card shape, regardless of
    # whether the card happened to be presented closer to portrait -- a card rotated ~90 degrees
    # from "upright" is still a fully valid capture, just measured with width/height swapped.
    dest_width, dest_height = (max(edge_a, edge_b), min(edge_a, edge_b))
    if edge_a < edge_b:
        # The long edge was actually top/bottom rather than left/right -- rotate the corner
        # assignment a quarter turn so the warp still maps onto a landscape destination rather
        # than stretching a portrait-shaped source into a landscape box.
        top_left, top_right, bottom_right, bottom_left = bottom_left, top_left, top_right, bottom_right

    destination_points = np.array(
        [[0, 0], [dest_width - 1, 0], [dest_width - 1, dest_height - 1], [0, dest_height - 1]], dtype=np.float32
    )
    source_points = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(source_points, destination_points)
    return cv2.warpPerspective(frame, transform, (int(round(dest_width)), int(round(dest_height))))


def detect_and_crop_card(frame: np.ndarray) -> np.ndarray:
    """Return a perspective-corrected, cropped-and-upscaled view of the card-shaped bright region in frame, or frame itself unchanged if no confident match is found."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return frame

    frame_height, frame_width = frame.shape[:2]
    frame_area = frame_height * frame_width

    best_rect = None
    best_area = 0
    for contour in contours:
        # minAreaRect (the smallest-area rectangle enclosing the contour, at any angle) rather
        # than boundingRect (axis-aligned) -- a rotated card's axis-aligned bounding box is always
        # larger than the card itself, which would both under-count how "card-shaped" a rotated
        # region looks (skewing the aspect-ratio check below) and bake extra background into the
        # crop this function produces.
        rect = cv2.minAreaRect(contour)
        _, (width, height), _ = rect
        area = width * height
        area_ratio = area / frame_area
        if not (_MIN_CARD_AREA_RATIO <= area_ratio <= _MAX_CARD_AREA_RATIO):
            continue

        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if abs(aspect_ratio - _CARD_ASPECT_RATIO) > _ASPECT_RATIO_TOLERANCE:
            continue

        if area > best_area:
            best_area = area
            best_rect = rect

    if best_rect is None:
        return frame

    corners = cv2.boxPoints(best_rect)
    # Grows the quadrilateral uniformly outward from its own center by the margin fraction (rather
    # than clamping in axis-aligned x/y like a simple bounding-box margin would) -- this keeps the
    # margin meaningful regardless of the card's rotation angle. Points landing outside the frame
    # afterward are clamped to it; warpPerspective would otherwise just sample a thin, harmless
    # black border there, but clamping keeps the source coordinates well-defined either way.
    center = corners.mean(axis=0)
    corners = center + (corners - center) * (1 + 2 * _CROP_MARGIN_RATIO)
    corners[:, 0] = np.clip(corners[:, 0], 0, frame_width - 1)
    corners[:, 1] = np.clip(corners[:, 1], 0, frame_height - 1)

    ordered_corners = _order_corners_clockwise(corners)
    warped = _warp_to_upright_card(frame, ordered_corners)
    if warped is None or warped.shape[0] == 0 or warped.shape[1] == 0:
        return frame

    if warped.shape[1] < _MIN_CROPPED_WIDTH:
        scale = _MIN_CROPPED_WIDTH / warped.shape[1]
        warped = cv2.resize(warped, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    return warped
