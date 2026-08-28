"""Parses name and personal-number fields out of a Swedish passport's raw OCR text, as a fallback for when its MRZ can't be read.

The MRZ is always the preferred, checksum-verified source for a passport (see
document/mrz_parser.py and document.ocr.extract_mrz_candidate_text) -- this parser is only worth
running once that has already failed. The Swedish passport bio-data page happens to print its own
numbered fields (1. Efternamn/Surname, 2. Förnamn/Given names, 3a. Personnr/Personal Id. No.) in a
similar spirit to the EU driving licence (document/dl_field_parser.py), but the physical layout
differs in a way that breaks that parser's same-line assumption: a passport prints each field's
label on its own line with the actual value on a *separate* line below it (form-style), rather than
the driving licence's "number and value share one printed line" convention -- so dl_field_parser's
patterns (anchored on finding the value immediately after the field number on one line) do not
match a passport's real layout at all.

Diagnosed against a real capture of an *entire open passport book* (both pages, not just the
bio-data page) held up to the camera: the sheer amount of extra text/security-pattern noise from
the other page caused Tesseract to badly scramble reading order -- the OCR'd text has the given-
name block appearing *before* the surname block, the reverse of their real top-to-bottom order on
the page. That rules out the technique document/national_id_parser.py uses (find one anchor, then
take the next couple of name-shaped lines in sequence) -- with reading order this unreliable,
"next" doesn't mean what it should. Instead, each field here is found by its OWN nearby label,
independent of any other field's position, so a scrambled order between fields doesn't cause one
field's value to be mistaken for another's.

This is a heuristic best-effort parser, not a validated implementation -- same caveat as
dl_field_parser.py and national_id_parser.py. Given names extraction in particular depends on
"GIVEN NAME"/"FÖRNAMN" having OCR'd legibly enough to anchor on, which is not guaranteed (observed
failing on the noisy two-page capture above, where it degraded to simply not finding a given name
rather than guessing). Expiry date is not attempted at all: it sits in a two-column area next to
height/sex whose reading order has proven just as unreliable as the name fields' above, and this
parser only reports fields it has reasonable confidence in.
"""

import re
from dataclasses import dataclass, field

# Tolerant of OCR punctuation noise around the label text itself -- these anchor on the label,
# not the field number, since a passport's field-number-to-label spacing has been observed
# collapsing unpredictably (e.g. "1, Efternamn/ Surname"). English label text is checked first
# (mandatory on every ICAO 9303 passport, and unaccented, so less prone to misreading than the
# Swedish label sharing the same line). Case-insensitive: unlike the driving licence/national-ID-
# card's all-caps field labels, a passport's are printed in mixed case ("Surname", not "SURNAME").
_SURNAME_LABEL_PATTERN = re.compile(r"SURNAME|EFTERNAMN", re.IGNORECASE)
_GIVEN_NAME_LABEL_PATTERN = re.compile(r"GIVEN\s*NAMES?|F[OÖ]RNAMN", re.IGNORECASE)

# Deliberately no end-of-line anchor and a tolerant leading-junk prefix -- see
# dl_field_parser.py's _SURNAME_PATTERN for the same reasoning (OCR noise before/after the actual
# name text).
_NAME_CANDIDATE_PATTERN = re.compile(r"(?m)^[^A-ZÅÄÖÉ\n]{0,5}([A-ZÅÄÖÉ][A-ZÅÄÖÉ]+(?:[ '\-][A-ZÅÄÖÉ]+)*)")

# Non-name lines that are also purely-uppercase and would otherwise look name-shaped -- checked as
# substrings since OCR's exact misreading of these varies. Place-of-birth values (e.g. "MATTEUS,
# STOCKHOLM") are excluded structurally instead, by _NAME_CANDIDATE_PATTERN stopping at the comma.
_NAME_CANDIDATE_DENYLIST_SUBSTRINGS = ("SURNAME", "EFTERNAMN", "GIVEN", "FORNAMN", "PASSPORT", "PASSEPORT", "SVERIGE", "SWEDEN", "SUEDE")

# Same personnummer pattern as national_id_parser.py -- searched for directly rather than anchored
# on the "Personnr/Personal Id. No." label, since that label has been observed OCR'd into
# unrecognizable noise ("Persil Ny)") even in a capture that read other fields cleanly, while the
# digit pattern itself is distinctive enough to find on its own.
_PERSONAL_NUMBER_PATTERN = re.compile(r"(\d{6})[-+.\s]+(\d{4})")


@dataclass
class PassportFieldsResult:
    """Whether Swedish-passport-style fields were found in the OCR text, and what was extracted."""

    found: bool
    fields: dict[str, str] = field(default_factory=dict)


def _find_name_after_label(ocr_text: str, label_pattern: re.Pattern) -> str | None:
    """Find label_pattern in the text, then return the first name-shaped candidate line after it, or None if either isn't found."""
    label_match = label_pattern.search(ocr_text)
    if label_match is None:
        return None
    for candidate_match in _NAME_CANDIDATE_PATTERN.finditer(ocr_text[label_match.end() :]):
        candidate = candidate_match.group(1)
        if any(denied in candidate for denied in _NAME_CANDIDATE_DENYLIST_SUBSTRINGS):
            continue
        return candidate
    return None


def parse_passport_fields(ocr_text: str) -> PassportFieldsResult:
    """Best-effort extraction of surname, given name(s), personal number, and birth date from a Swedish passport's raw OCR text."""
    fields: dict[str, str] = {}

    surname = _find_name_after_label(ocr_text, _SURNAME_LABEL_PATTERN)
    if surname:
        fields["surname"] = surname

    given_name = _find_name_after_label(ocr_text, _GIVEN_NAME_LABEL_PATTERN)
    if given_name:
        fields["name"] = given_name

    personal_number_match = _PERSONAL_NUMBER_PATTERN.search(ocr_text)
    if personal_number_match:
        personal_number_digits, personal_number_suffix = personal_number_match.groups()
        fields["personal_number"] = f"{personal_number_digits}-{personal_number_suffix}"
        # See national_id_parser.py's identical derivation for why this is safe to reuse here.
        fields["birth_date"] = personal_number_digits

    return PassportFieldsResult(found=bool(fields), fields=fields)
