"""Assesses a captured document frame for basic image-quality problems (blur, glare) before it's
handed to OCR/MRZ/barcode extraction.

This is a passive signal only -- it never blocks, retries, or prompts on capture. It exists so a
poor OCR/MRZ read can be attributed to a visibly bad source image (shown as a banner in the
history list) rather than looking like an unexplained black-box failure.

Both thresholds below are unvalidated starting points, not calibrated against real captured
document samples -- same caveat as config.settings.DEFAULT_FACE_MATCH_THRESHOLD and
document.field_crosscheck._BARCODE_OVERLAP_VALID_RATIO.
"""

from dataclasses import dataclass

import cv2
import numpy as np

# Variance of the Laplacian is a standard cheap blur proxy: a sharp image has lots of
# high-frequency edge content (high variance), a blurry one is smoothed out (low variance).
_BLUR_VARIANCE_THRESHOLD = 100.0

# Fraction of near-white (blown-out) pixels above which the frame is considered glare-heavy --
# laminated ID cards under desk lighting commonly reflect a hard, small, very bright patch.
_GLARE_PIXEL_RATIO_THRESHOLD = 0.15
_GLARE_BRIGHTNESS_LEVEL = 245  # 0-255 grayscale value considered "blown out"


@dataclass
class ImageQualityResult:
    """The result of a quality check on one captured frame: whether it looks blurry and/or glare-heavy, plus the raw measurements behind that call."""

    is_blurry: bool
    is_glare_heavy: bool
    blur_score: float
    glare_ratio: float

    @property
    def has_warning(self) -> bool:
        return self.is_blurry or self.is_glare_heavy

    @property
    def warning_text(self) -> str:
        """Short human-readable summary of which problem(s) were detected, for display in a banner."""
        reasons = []
        if self.is_blurry:
            reasons.append("blurry")
        if self.is_glare_heavy:
            reasons.append("glare")
        return " & ".join(reasons)


def assess_image_quality(frame: np.ndarray) -> ImageQualityResult:
    """Measure blur and glare on a captured BGR frame and decide whether either crosses its warning threshold."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    glare_ratio = float(np.mean(gray >= _GLARE_BRIGHTNESS_LEVEL))
    return ImageQualityResult(
        is_blurry=blur_score < _BLUR_VARIANCE_THRESHOLD,
        is_glare_heavy=glare_ratio >= _GLARE_PIXEL_RATIO_THRESHOLD,
        blur_score=blur_score,
        glare_ratio=glare_ratio,
    )
