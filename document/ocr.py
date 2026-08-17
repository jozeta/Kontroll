"""Runs Tesseract OCR over a document image and returns extracted text and per-word confidence data."""

from dataclasses import dataclass

import numpy as np
import pytesseract


@dataclass
class OcrWord:
    """One OCR-recognized word: its text and Tesseract's confidence score for that recognition (0-100)."""

    text: str
    confidence: float


def extract_text(image: np.ndarray) -> str:
    """Return the full raw text Tesseract recognizes on the document image."""
    return pytesseract.image_to_string(image)


def extract_words(image: np.ndarray) -> list[OcrWord]:
    """Return each recognized word individually with its confidence score, for use in field cross-checking.

    Tesseract reports a confidence of -1 for layout elements that aren't recognized text (e.g.
    detected blank regions); those are filtered out here along with empty/whitespace-only entries,
    since they carry no field-matching value.
    """
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words: list[OcrWord] = []
    for text, confidence in zip(data["text"], data["conf"]):
        stripped_text = text.strip()
        if stripped_text and float(confidence) >= 0:
            words.append(OcrWord(text=stripped_text, confidence=float(confidence)))
    return words
