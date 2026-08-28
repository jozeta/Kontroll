"""Central list of Tesseract language packs this app's document OCR pass requests, shared between
document/ocr.py (which actually runs OCR with them) and prerequisites.py (which checks they're
installed and offers to download whichever are missing).

Kept in its own tiny, dependency-free module rather than defined directly in document/ocr.py:
prerequisites.py must not import anything that pulls in cv2/pytesseract/numpy at module load time
(see prerequisites.py's own docstring on this) -- it has to be able to report "pytesseract isn't
installed" without importing pytesseract, and document/ocr.py imports pytesseract itself at module
level.
"""

# Tesseract language codes (ISO 639-2/B) for this app's supported document-issuing countries.
# English (always included separately below, not listed here since it's the one language pack
# Tesseract ships by default) covers the USA/UK/Ireland directly with no extra pack needed.
#
# Trimmed to this desk's actual visitor mix (Nordic + French) rather than every Latin-script EU
# language this app could in principle support -- every additional simultaneously-loaded language
# costs extra Tesseract recognition time per capture (every listed language is evaluated together,
# not one at a time) and measurably increases cross-language diacritic misreads: a dozen
# languages' worth of similar-looking accented glyphs and dictionaries competing for the same
# character raised real, diagnosed misreads (e.g. Swedish 'Ö' recognized as Polish/Spanish 'Ó')
# that a narrower set is far less prone to. If this desk's visitor mix broadens later, add the
# missing country back here rather than restoring the full original list -- narrower is better
# whenever it's actually true, not a one-time tradeoff to revisit only once.
ADDITIONAL_OCR_LANGUAGES: dict[str, str] = {
    "swe": "Swedish",
    "dan": "Danish",
    "nor": "Norwegian",
    "fin": "Finnish",
    "fra": "French",
}

# Passed to Tesseract as a single "+"-joined lang string. "eng" always leads -- it's the one
# language pack guaranteed present alongside any Tesseract install (see prerequisites.py::
# check_tesseract, which only ever confirms the Tesseract binary itself, not which language packs
# came with it) and is what covers the USA/UK/Ireland with no additional pack needed.
TESSERACT_LANG_STRING = "+".join(["eng", *ADDITIONAL_OCR_LANGUAGES.keys()])
