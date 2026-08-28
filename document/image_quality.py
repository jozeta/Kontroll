"""Assesses a captured document frame for basic image-quality problems (blur, glare, low
resolution, hard-to-read text) before it's handed to OCR/MRZ/PDF417 extraction.

This is a passive signal only -- it never blocks, retries, or prompts on capture. It exists so a
poor OCR/MRZ read can be attributed to a visibly bad source image (shown as a banner in the
history list) rather than looking like an unexplained black-box failure.

All thresholds below are unvalidated starting points, not calibrated against real captured
document samples -- same caveat as config.settings.DEFAULT_FACE_MATCH_THRESHOLD and
config.settings.DEFAULT_BARCODE_OVERLAP_THRESHOLD.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from document.ocr import OcrWord

# Variance of the Laplacian is a standard cheap blur proxy: a sharp image has lots of
# high-frequency edge content (high variance), a blurry one is smoothed out (low variance). This
# is measured over the *whole* frame, so a high-frequency background can mask a blurry document
# that only occupies a small part of the frame -- see is_low_resolution/is_text_hard_to_read below
# for signals that catch that case instead.
_BLUR_VARIANCE_THRESHOLD = 100.0

# Fraction of near-white (blown-out) pixels above which the frame is considered glare-heavy --
# laminated ID cards under desk lighting commonly reflect a hard, small, very bright patch.
_GLARE_PIXEL_RATIO_THRESHOLD = 0.15
_GLARE_BRIGHTNESS_LEVEL = 245  # 0-255 grayscale value considered "blown out"

# Frame dimensions below which a capture is flagged low-resolution regardless of how sharp it
# looks -- catches a camera silently defaulting to a low mode (e.g. 640x480) where a document
# occupying only part of the frame ends up too low-detail to OCR reliably, even though whole-frame
# blur/glare metrics don't fire (background texture can keep the Laplacian variance high).
_MIN_FRAME_WIDTH = 1000
_MIN_FRAME_HEIGHT = 700


def is_resolution_too_low_for_ocr(width: int, height: int) -> bool:
    """Whether a frame at this size is below assess_image_quality's is_low_resolution floor.

    Exposed separately (rather than only inline in assess_image_quality below) so a caller that
    only has frame dimensions -- e.g. ui/main_window.py's live document-camera preview, warning an
    operator before a frame is even captured -- can reuse the same threshold without running a
    full quality assessment (blur/glare/OCR) on every preview tick.
    """
    return width < _MIN_FRAME_WIDTH or height < _MIN_FRAME_HEIGHT


def compute_blur_score(frame: np.ndarray) -> float:
    """Variance of the Laplacian over frame -- see _BLUR_VARIANCE_THRESHOLD's comment for what this measures. Exposed separately for the same reason as is_resolution_too_low_for_ocr above: a live preview can check this against the same threshold without the rest of assess_image_quality's work."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def is_frame_too_blurry(frame: np.ndarray) -> bool:
    """Whether frame's blur score is below assess_image_quality's is_blurry floor -- see compute_blur_score."""
    return compute_blur_score(frame) < _BLUR_VARIANCE_THRESHOLD

# OCR-confidence-based legibility check, using the per-word confidence Tesseract already reports
# (document.ocr.extract_words) -- a low mean confidence or very few recognized words suggests the
# text was too small/blurry/pixelated for Tesseract to read reliably, independent of what the
# blur/glare/resolution metrics say.
_MIN_MEAN_OCR_CONFIDENCE = 45.0
_MIN_OCR_WORD_COUNT = 3


@dataclass
class ImageQualityResult:
    """The result of a quality check on one captured frame: which problem(s) it looks like it has, plus the raw measurements behind that call."""

    is_blurry: bool
    is_glare_heavy: bool
    is_low_resolution: bool
    is_text_hard_to_read: bool
    blur_score: float
    glare_ratio: float
    width: int
    height: int
    mean_ocr_confidence: float | None

    @property
    def has_warning(self) -> bool:
        return self.is_blurry or self.is_glare_heavy or self.is_low_resolution or self.is_text_hard_to_read

    @property
    def warning_text(self) -> str:
        """Short human-readable summary of which problem(s) were detected, for display in a banner."""
        reasons = []
        if self.is_low_resolution:
            reasons.append("low resolution")
        if self.is_blurry:
            reasons.append("blurry")
        if self.is_glare_heavy:
            reasons.append("glare")
        if self.is_text_hard_to_read:
            reasons.append("hard to read text")
        return " & ".join(reasons)


def assess_image_quality(frame: np.ndarray, ocr_words: list[OcrWord] | None = None) -> ImageQualityResult:
    """Measure blur, glare, resolution, and (if OCR words are supplied) text legibility on a captured BGR frame, and decide whether any crosses its warning threshold.

    ocr_words is optional -- pass document.ocr.extract_words(frame)'s result to enable the
    hard-to-read-text check; omit it (None) to skip that check entirely (e.g. for callers that
    don't run OCR at all). An empty list (OCR ran and found nothing) is treated the same as very
    low confidence, since a real ID capture is expected to contain readable text.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float(np.mean(gray >= _GLARE_BRIGHTNESS_LEVEL))
    height, width = frame.shape[:2]
    # (blur_score is computed inline above rather than via compute_blur_score, to avoid converting
    # to grayscale twice -- glare_ratio needs that same `gray` array right below.)

    mean_ocr_confidence: float | None = None
    is_text_hard_to_read = False
    if ocr_words is not None:
        if ocr_words:
            mean_ocr_confidence = sum(word.confidence for word in ocr_words) / len(ocr_words)
        else:
            mean_ocr_confidence = 0.0
        is_text_hard_to_read = mean_ocr_confidence < _MIN_MEAN_OCR_CONFIDENCE or len(ocr_words) < _MIN_OCR_WORD_COUNT

    return ImageQualityResult(
        is_blurry=blur_score < _BLUR_VARIANCE_THRESHOLD,
        is_glare_heavy=glare_ratio >= _GLARE_PIXEL_RATIO_THRESHOLD,
        is_low_resolution=is_resolution_too_low_for_ocr(width, height),
        is_text_hard_to_read=is_text_hard_to_read,
        blur_score=blur_score,
        glare_ratio=glare_ratio,
        width=width,
        height=height,
        mean_ocr_confidence=mean_ocr_confidence,
    )
