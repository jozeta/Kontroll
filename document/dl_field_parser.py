"""Parses name and birth-date fields out of a Swedish (EU-harmonized) driving licence's raw OCR text.

Swedish driving licences follow the EU's harmonized driving licence layout, where each printed
field is prefixed with its EU-standard field number (Council Directive 2006/126/EC, Annex I):
1. surname, 2. given name(s), 3. date (and place) of birth. Those field-number prefixes are used
as anchors here rather than the printed Swedish/English labels themselves (e.g. 'Efternamn'/
'Surname'), since OCR is more likely to misread small label text than a single digit, and the
numbering is consistent across EU member states regardless of which two languages a given
licence's labels are printed in.

This is a heuristic best-effort parser, not a validated implementation -- no independently-
verified real driving licence OCR sample was available to test it against (see barcode_parser.py's
similar caveat for the same document type). A parse failure here degrades gracefully to '-'
placeholders in the UI, the same as for any other document with no structured fields available.
"""

import re
from dataclasses import dataclass, field


@dataclass
class DlFieldsResult:
    """Whether driving-licence-style name/birth-date fields were found in the OCR text, and what was extracted."""

    found: bool
    fields: dict[str, str] = field(default_factory=dict)


# Anchored on the field's leading digit (optionally followed by '.'/':' punctuation OCR often
# drops or mangles) at the start of its own printed line, since Tesseract typically emits one
# printed line per output line for this kind of card layout.
_SURNAME_PATTERN = re.compile(r"(?m)^\s*1[.:]?\s+([A-ZÅÄÖÉ' -]{2,})\s*$")
_GIVEN_NAME_PATTERN = re.compile(r"(?m)^\s*2[.:]?\s+([A-ZÅÄÖÉ' -]{2,})\s*$")
_BIRTH_LINE_PATTERN = re.compile(r"(?m)^\s*3[.:]?\s+(.*\d.*)$")

# Field 3 is "date and place of birth", so only the date portion of that line is extracted here;
# multiple formats are tried since EU guidance specifies dd.mm.yyyy but Swedish practice commonly
# prints ISO-style yyyy-mm-dd instead.
_DATE_PATTERNS = (
    re.compile(r"(\d{4})-(\d{2})-(\d{2})"),
    re.compile(r"(\d{2})\.(\d{2})\.(\d{4})"),
    re.compile(r"(\d{2})/(\d{2})/(\d{4})"),
)


def _extract_birth_date(birth_line: str) -> str | None:
    """Pull an ISO 'YYYY-MM-DD' date out of a driving licence's date-of-birth line, trying each supported date format in turn."""
    for pattern in _DATE_PATTERNS:
        match = pattern.search(birth_line)
        if match is None:
            continue
        first, second, third = match.groups()
        # The ISO pattern's groups are already (year, month, day); the other two are (day, month, year).
        year, month, day = (first, second, third) if len(first) == 4 else (third, second, first)
        return f"{year}-{month}-{day}"
    return None


def parse_driving_licence_fields(ocr_text: str) -> DlFieldsResult:
    """Best-effort extraction of surname, given name(s), and birth date from a driving licence's raw OCR text."""
    surname_match = _SURNAME_PATTERN.search(ocr_text)
    given_name_match = _GIVEN_NAME_PATTERN.search(ocr_text)
    birth_line_match = _BIRTH_LINE_PATTERN.search(ocr_text)

    fields: dict[str, str] = {}
    if surname_match:
        fields["surname"] = surname_match.group(1).strip()
    if given_name_match:
        fields["name"] = given_name_match.group(1).strip()
    if birth_line_match:
        birth_date = _extract_birth_date(birth_line_match.group(1))
        if birth_date:
            fields["birth_date"] = birth_date

    return DlFieldsResult(found=bool(fields), fields=fields)
