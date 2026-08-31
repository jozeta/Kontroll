"""Runs Tesseract OCR over a document image and returns extracted text and per-word confidence data."""

from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract

from config.ocr_languages import TESSERACT_LANG_STRING

# Documents from any one supported country routinely carry that country's own diacritics in names
# (å/ä/ö, ü/ß, é/è/ç, ñ, ...) -- Tesseract defaults to "eng" alone when no lang is given, whose
# model has no trained glyph shapes for most of those and instead of failing loudly, silently
# reads them as their plain unaccented look-alikes. Running every supported language's model
# together (see config/ocr_languages.py for the exact set and why it's shared with
# prerequisites.py) rather than picking just one means generic Latin/numeric text -- MRZ leftovers,
# punctuation, digits -- that any single language model alone is less confident on still recognizes
# well, and one OCR pass works regardless of which supported country's document is being scanned
# without first having to know its country. Requires each listed language's .traineddata pack to
# actually be installed alongside Tesseract -- see prerequisites.py::check_ocr_language_packs /
# install_ocr_language_packs.
_OCR_LANG = TESSERACT_LANG_STRING

# PSM 4 ("assume a single column of text of variable sizes") rather than Tesseract's PSM-3 default
# ("fully automatic page segmentation", the effective setting when no --psm is given at all).
#
# PSM 6 ("assume a single uniform block of text") was tried first and reverted: an ID-1 card is
# NOT actually a single uniform block even after document/card_detector.py's crop -- it still has
# a photo (and often a hologram/security-pattern strip) occupying a large fraction of it alongside
# the text fields, and PSM 6's "just read straight through, everything is text" assumption means
# Tesseract tries to read the photo/hologram pixel noise as characters too, which measurably
# degrades or even swamps the real field text with garbage lines (confirmed with a real Tesseract
# install against synthetic card images built specifically to include a photo block, which PSM 6
# read far worse than PSM 3/4 on -- the difference doesn't show up on a text-only test image with
# no photo, which is why this wasn't caught before PSM 6 shipped). PSM 4 assumes a single column
# rather than one uniform block, which tolerates the photo/hologram regions being visually distinct
# from the text column instead of forcing everything into one undifferentiated read -- verified
# against the same synthetic card-with-photo images, at multiple rotation angles, to consistently
# produce clean per-field lines where both PSM 3 and PSM 6 produced fragmented or garbage-laden
# output. Tesseract's MRZ-focused pass below (extract_mrz_candidate_text) keeps PSM 6, since that
# crop is just the isolated MRZ strip itself with no photo mixed in -- a genuinely uniform block.
#
# --dpi 300 tells Tesseract the resolution to assume for an image with no embedded DPI metadata (a
# raw numpy array has none) -- matching the >=300dpi-equivalent effective density
# card_detector.py's own upscaling (_MIN_CROPPED_WIDTH) is specifically aiming for, rather than
# leaving Tesseract to guess. Confirmed accepted by a real Tesseract 5.4 install (not just passed
# through unvalidated).
_GENERAL_OCR_CONFIG = "--psm 4 --dpi 300"

# CLAHE (Contrast Limited Adaptive Histogram Equalization) tile grid size and clip limit --
# normalizes local contrast for a genuinely over/under-exposed capture (the software-side
# complement to capture/camera_manager.py's ExposureSettings, which fixes the capture itself at the
# source). A plain global histogram-equalize was deliberately not used instead: it applies one
# uniform gain across the whole card, fighting _preprocess_for_ocr's own adaptive threshold (already
# local-neighborhood-based specifically to handle uneven lighting, see that function's docstring).
# CLAHE's clip limit caps how far any one tile's contrast gets stretched, which is what keeps it
# from amplifying noise the same uniform equalize would. Tile size 8x8 and clip limit 2.0 are
# OpenCV's own commonly-used starting values for this kind of document/text preprocessing, not
# calibrated specifically against this app's cameras.
#
# Applied conditionally, gated on _apply_clahe_if_low_contrast's own std-deviation check below --
# NOT unconditionally on every capture. Verified with a real driving-licence capture (cropped from
# an operator-reported over-bright production screenshot) plus synthetic over-/under-exposed
# variants of it: CLAHE clearly helped once contrast was actually low (mean OCR word confidence
# 26.5->35.9 on an overexposed variant, 33.8->45.7 on an underexposed one), but on that same
# capture's original, already-reasonably-exposed contrast, applying it unconditionally *hurt*
# (more low-confidence noise words, lower mean confidence) -- adaptiveThreshold's own local
# thresholding already handles a normally-lit capture fine on its own. Gating on low contrast keeps
# this strictly additive (an already-good capture's OCR is untouched) instead of a blanket change
# that regresses the common case to fix the uncommon one.
_CLAHE_CLIP_LIMIT = 2.0
_CLAHE_TILE_GRID_SIZE = (8, 8)

# Grayscale standard deviation below which a capture is considered low-contrast enough (whether
# from over- or under-exposure -- both compress the pixel value range toward one end) to be worth
# CLAHE-normalizing before thresholding. A normally-lit real capture measured ~116; simulated over-
# and under-exposed variants of that same capture measured ~53 and ~31 respectively (see this
# module's own verification above) -- 70 sits comfortably between the two without being so high it
# would also catch a normal capture.
_LOW_CONTRAST_STD_THRESHOLD = 70.0


@dataclass
class OcrWord:
    """One OCR-recognized word: its text and Tesseract's confidence score for that recognition (0-100)."""

    text: str
    confidence: float


def _preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Convert a captured document frame to grayscale and adaptive-threshold it before handing it to Tesseract.

    Diagnosed against a real driving licence capture whose card background carried a pastel
    hologram/security-pattern gradient: fed to Tesseract as a raw color image, this produced
    almost no usable text at all (3 garbled lines out of 7+ printed fields), while an identical
    crop converted to grayscale and adaptive-thresholded first recovered nearly every field
    cleanly. Adaptive (not a single global Otsu) threshold is used specifically because a card
    photo commonly has uneven lighting across its surface (glare on one side, shadow on the
    other) that a single global threshold value can't handle -- adaptive thresholding recomputes
    the local threshold in each neighborhood instead. Verified not to regress any previously-
    working capture in a side-by-side comparison across every real sample available.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.std() < _LOW_CONTRAST_STD_THRESHOLD:
        clahe = cv2.createCLAHE(clipLimit=_CLAHE_CLIP_LIMIT, tileGridSize=_CLAHE_TILE_GRID_SIZE)
        gray = clahe.apply(gray)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)


def preview_ocr_preprocessing(image: np.ndarray) -> np.ndarray:
    """Public wrapper around _preprocess_for_ocr, for a caller that wants to *show* the operator
    exactly what Tesseract will actually see (grayscale, CLAHE-normalized if low-contrast, then
    adaptive-thresholded) rather than run OCR itself -- see ui/image_viewer.py::ImageEditDialog's
    "Preview: what OCR sees" toggle, which exists specifically because judging whether an edit
    (brightness/contrast/sharpen/crop) actually helps OCR by eye on the original color image is
    unreliable; showing the real preprocessed input makes that immediately visible instead."""
    return _preprocess_for_ocr(image)


def extract_text(image: np.ndarray) -> str:
    """Return the full raw text Tesseract recognizes on the document image."""
    return pytesseract.image_to_string(_preprocess_for_ocr(image), lang=_OCR_LANG, config=_GENERAL_OCR_CONFIG)


def _words_from_preprocessed(preprocessed: np.ndarray) -> list[OcrWord]:
    """Shared by extract_words and extract_words_and_text -- see extract_words' docstring for the confidence-filtering rationale."""
    data = pytesseract.image_to_data(
        preprocessed, lang=_OCR_LANG, config=_GENERAL_OCR_CONFIG, output_type=pytesseract.Output.DICT
    )
    words: list[OcrWord] = []
    for text, confidence in zip(data["text"], data["conf"]):
        stripped_text = text.strip()
        if stripped_text and float(confidence) >= 0:
            words.append(OcrWord(text=stripped_text, confidence=float(confidence)))
    return words


def extract_words(image: np.ndarray) -> list[OcrWord]:
    """Return each recognized word individually with its confidence score, for use in field cross-checking.

    Tesseract reports a confidence of -1 for layout elements that aren't recognized text (e.g.
    detected blank regions); those are filtered out here along with empty/whitespace-only entries,
    since they carry no field-matching value.
    """
    return _words_from_preprocessed(_preprocess_for_ocr(image))


def extract_words_and_text(image: np.ndarray) -> tuple[list[OcrWord], str]:
    """Equivalent to calling extract_words(image) and extract_text(image) separately, but preprocessing image only once.

    The two underlying Tesseract calls (image_to_data vs image_to_string) can't be merged into one
    -- image_to_data's per-word layout doesn't reliably reproduce image_to_string's spacing, which
    mrz_parser/dl_field_parser depend on, so both are still run -- but they were each independently
    re-running the identical grayscale+adaptive-threshold preprocessing on the same input image.
    Use this instead of calling both separately when, as in the main capture pipeline, both are
    wanted for the same image.
    """
    preprocessed = _preprocess_for_ocr(image)
    words = _words_from_preprocessed(preprocessed)
    text = pytesseract.image_to_string(preprocessed, lang=_OCR_LANG, config=_GENERAL_OCR_CONFIG)
    return words, text


def _ocr_quality_score(words: list[OcrWord]) -> float:
    """Rough proxy for how well an OCR pass went, used only to compare a few candidate rotations of the same image against each other -- not a meaningful number on its own. More recognized words, each at a reasonable confidence, scores higher than the sparse, low-confidence handful of words a wrongly-rotated pass typically produces."""
    if not words:
        return 0.0
    return len(words) * (sum(word.confidence for word in words) / len(words))

# Empirically, an image that's actually upright scores in the thousands or higher on a normal ID
# document (dozens of words at reasonable confidence); a genuinely sideways/upside-down capture --
# where Tesseract has no real text lines to find -- scores in the low hundreds or less. This sits
# comfortably between the two, confirmed against a real capture that scored 907 as originally
# oriented and 16873 once rotated the right way round.
_ORIENTATION_RETRY_SCORE_THRESHOLD = 2000.0


def correct_text_orientation(image: np.ndarray) -> tuple[np.ndarray, list[OcrWord], str]:
    """Return (oriented_image, words, text) for whichever of image's 4 axis-aligned orientations OCRs best -- image unchanged if the original orientation already looks fine.

    document/card_detector.py's crop only ever guarantees a landscape-shaped output, not which of
    the two landscape orientations (right-side-up or upside-down) is the readable one -- there's no
    way to tell a rectangle's true reading orientation from its geometry alone, and when its own
    long/short-edge swap has to pick one of two ways to reach landscape, it can pick the wrong one.
    Diagnosed against a real passport capture: the photographed spread's natural landscape axis ran
    vertically in the portrait camera frame, and the fixed-direction 90-degree correction
    card_detector.py applies to reach landscape happened to leave the text sideways rather than
    upright. Tesseract does not recognize sideways text at all (it assumes roughly-horizontal text
    and returns next to nothing), which is what turned into "no text found" for that capture
    specifically -- 13 words at low confidence as originally oriented, versus 393 words at high
    confidence rotated 90 degrees the other way. ID cards/driving licences captured closer to
    already-upright in the lightbox don't hit this.

    Callers should use the returned image (not just the returned words/text) for everything
    downstream that assumes an upright frame -- document/mrz_locator.py's MRZ-band search and
    document/barcode_parser.py's decode both need the correctly-oriented image too. The words/text
    are the exact extract_words_and_text() result already computed for the winning orientation
    while scoring it -- returned so the caller (ui/main_window.py::_analyze_capture) can use them
    directly instead of immediately re-running an identical OCR pass on the same image right after
    this returns, which is what happened here previously (this function returned only the image,
    and every caller needed extract_words_and_text() right afterward anyway) -- a real, silent,
    guaranteed-on-every-single-capture extra Tesseract pass in the common (already-upright) case
    specifically, since that path used to score orientation via extract_words() and then discard
    the result.

    Tries the extra 3 rotations only when the first (unrotated) attempt scores poorly -- see
    _ORIENTATION_RETRY_SCORE_THRESHOLD -- so the common already-upright case (every lightbox
    capture seen so far except the one that motivated this) pays for exactly one OCR pass total,
    not two.
    """
    best_image = image
    best_words, best_text = extract_words_and_text(image)
    best_score = _ocr_quality_score(best_words)
    if best_score >= _ORIENTATION_RETRY_SCORE_THRESHOLD:
        return best_image, best_words, best_text

    for rotation in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE):
        rotated = cv2.rotate(image, rotation)
        words, text = extract_words_and_text(rotated)
        score = _ocr_quality_score(words)
        if score > best_score:
            best_image, best_words, best_text, best_score = rotated, words, text, score

    return best_image, best_words, best_text


# MRZ lines use only these characters (ICAO 9303) -- restricting Tesseract's whitelist to exactly
# this set is what makes extract_mrz_candidate_text() below far more accurate than the general
# extract_text() pass for MRZ specifically: it can't misread a character into e.g. lowercase noise
# or punctuation outside this set at all, since those glyphs are excluded from consideration
# entirely rather than merely being less likely.
_MRZ_CHARSET_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"


def extract_mrz_candidate_text(image: np.ndarray) -> str:
    """Run a second, MRZ-tuned OCR pass over a document image, for use as a fallback when extract_text()'s general-purpose pass finds no valid MRZ.

    Diagnosed against a real passport capture: general-purpose OCR (extract_text) found no valid
    MRZ at all (its two fixed-width lines came out full of noise, mixed in among garbled OCR of
    all the surrounding printed/security-pattern content). Two changes recover the MRZ text
    dramatically better: an Otsu (global) rather than adaptive (local) threshold -- MRZ printing is
    high-contrast and uniformly lit along its own strip, so global thresholding suits it better than
    the adaptive threshold extract_text() uses for whole-card lighting unevenness -- and, more
    importantly, restricting Tesseract's character whitelist to the MRZ alphabet itself
    (_MRZ_CHARSET_WHITELIST), which stops it from ever considering a lowercase or punctuation
    misreading for what should always be one of only 37 possible characters. This does not by
    itself guarantee the *whole* MRZ decodes perfectly (a heavily-noisy capture can still corrupt
    individual characters) -- document.mrz_parser's checksum validation is what actually decides
    whether the recovered text is trustworthy, exactly as for the general-purpose OCR path; this
    function only improves the odds of handing it something worth checksumming in the first place.

    Deliberately not run as the primary/only OCR pass: the character whitelist would make it
    actively worse than extract_text() for driving-licence/national-ID field text (names, Swedish
    'Ö'/'Å'/'Ä', punctuation), so it's only worth the extra ~1s Tesseract call as a fallback once
    extract_text()'s result has already been tried and found no valid MRZ.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return pytesseract.image_to_string(
        thresholded, config=f"--psm 6 --dpi 300 -c tessedit_char_whitelist={_MRZ_CHARSET_WHITELIST}"
    )
