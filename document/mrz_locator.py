"""Locates the two-line MRZ band within a document frame, for OCR-cropping to it directly.

Diagnosed against a real passport capture that photographs the entire open passport book (both
pages) rather than just the bio-data page -- physically unavoidable for a bound passport, unlike a
driving licence or ID card which can be laid flat on its own. That capture style buries the MRZ
among a huge amount of unrelated printed text and security-pattern background, which both
document.ocr's general OCR pass and its MRZ-alphabet-whitelisted fallback (applied to the *whole*
frame) have been observed failing to recover reliably. Cropping tightly to just the MRZ band before
running that whitelisted OCR pass recovers it far more consistently.

The locator works by row-wise edge density, not fixed position -- MRZ text is dense, monospaced,
and spans nearly the full page width, unlike a name field's sparse, partial-width printed text. But
raw density alone is NOT enough: a passport's decorative security-pattern background (guilloché
lines, watermarks) can have *higher* average edge density than the MRZ, since it's printed
uniformly across a large area, while the MRZ sits on a plain background with near-zero density
immediately above and below it. What distinguishes the MRZ is PROMINENCE -- a narrow, sharp density
spike standing out against its immediate surroundings -- not the pattern's broad, sustained
moderate density. Searching the whole frame height (not just wherever the bio-data page happens to
land) also makes this independent of which page ends up on top -- verified this doesn't matter
against a real capture: the prominence peak lands in the same place whether the search covers the
full frame or is restricted to the lower half.

This is a heuristic best-effort locator, not a validated implementation -- same caveat as every
other document/*_parser.py module. The prominence threshold below was picked from one real MRZ
sample (0.22) against eight real non-MRZ documents (driving licences/national ID cards, 0.10-0.15)
-- a real gap, but from a small sample. Falls back to returning the original frame unchanged if no
band clears the threshold, the same safe-fallback philosophy as document/card_detector.py.
"""

import cv2
import numpy as np

# Edge-magnitude threshold (Sobel) above which a pixel counts as part of a character stroke,
# rather than background noise/gradient.
_EDGE_MAGNITUDE_THRESHOLD = 30.0

# Target band height as a fraction of frame height -- sized for "two MRZ lines", not the WHOLE
# passport's worth of printed text (which is what a plain highest-density search converges on
# instead, per this module's docstring).
_BAND_HEIGHT_FRACTION = 0.055

# How much of the frame height, immediately above and below a candidate band, is checked to see
# whether the band actually stands out from its surroundings (prominence) rather than just being
# part of a broader dense region.
_CONTEXT_HEIGHT_FRACTION = 0.05

# See module docstring for how this was picked.
_MIN_PROMINENCE = 0.18

# Extra margin (as a fraction of the detected band's own height) kept on each side of the crop, so
# character ascenders/descenders right at the band's edge aren't clipped.
_CROP_MARGIN_RATIO = 0.5


def _row_active_width_fraction(gray: np.ndarray) -> np.ndarray:
    """For each row, the fraction of columns with a strong horizontal intensity gradient -- high and sustained across nearly the full width for MRZ text, low for blank margins."""
    sobel = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    return (sobel > _EDGE_MAGNITUDE_THRESHOLD).mean(axis=1)


def locate_mrz_band(frame: np.ndarray) -> np.ndarray:
    """Return a tight crop around the frame's most MRZ-band-shaped row region, or the original frame unchanged if nothing clears the confidence threshold."""
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    density = _row_active_width_fraction(gray)

    smoothing_window = max(3, height // 300)
    smoothed = np.convolve(density, np.ones(smoothing_window) / smoothing_window, mode="same")
    cumulative = np.cumsum(smoothed)

    def window_mean(start: int, end: int) -> float:
        start, end = max(0, start), min(height, end)
        if end <= start:
            return 0.0
        return (cumulative[end - 1] - (cumulative[start - 1] if start > 0 else 0.0)) / (end - start)

    band_height = max(1, int(height * _BAND_HEIGHT_FRACTION))
    context_height = max(1, int(height * _CONTEXT_HEIGHT_FRACTION))

    best_prominence = -1.0
    best_band: tuple[int, int] | None = None
    for start in range(0, height - band_height):
        end = start + band_height
        band_density = window_mean(start, end)
        surrounding_density = max(window_mean(start - context_height, start), window_mean(end, end + context_height))
        prominence = band_density - surrounding_density
        if prominence > best_prominence:
            best_prominence = prominence
            best_band = (start, end)

    if best_band is None or best_prominence < _MIN_PROMINENCE:
        return frame

    start, end = best_band
    margin = int((end - start) * _CROP_MARGIN_RATIO)
    return frame[max(0, start - margin) : min(height, end + margin), :]
