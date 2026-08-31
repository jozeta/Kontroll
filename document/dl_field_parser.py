"""Parses name, birth-date, and expiry-date fields out of a Swedish (EU-harmonized) driving licence's raw OCR text.

Swedish driving licences follow the EU's harmonized driving licence layout, where each printed
field is prefixed with its EU-standard field number (Council Directive 2006/126/EC, Annex I):
1. surname, 2. given name(s), 3. date (and place) of birth, 4b. date of expiry (4a, date of
issue, and 4c, issuing authority, are printed alongside 4b but not extracted here -- nothing
downstream needs them). Those field-number prefixes are used as anchors here rather than the
printed Swedish/English labels themselves (e.g. 'Efternamn'/'Surname'), since OCR is more likely
to misread small label text than a digit-plus-letter code, and the numbering is consistent
across EU member states regardless of which two languages a given licence's labels are printed
in.

This is a heuristic best-effort parser, not a validated implementation -- no independently-
verified real driving licence OCR sample was available to test it against (see barcode_parser.py's
similar caveat for the same document type). A parse failure here degrades gracefully to '-'
placeholders in the UI, the same as for any other document with no structured fields available.

Field 5 (personal_number) is Sweden-specific -- the EU-harmonized layout leaves field 5 to each
member state's own use, and Sweden prints the personnummer there (format YYMMDD-XXXX, or with a
'+' separator instead of '-' for someone who has turned 100) -- extracted the same field-number-
anchored way as the others, for document/flag_matcher.py to cross-check against the operator's
flagged-persons list.
"""

import re
from dataclasses import dataclass, field

from document.dates import extract_date_from_line


@dataclass
class DlFieldsResult:
    """Whether driving-licence-style name/birth-date fields were found in the OCR text, and what was extracted."""

    found: bool
    fields: dict[str, str] = field(default_factory=dict)


# Anchored on the field's leading digit (optionally followed by '.'/':' punctuation OCR often
# drops or mangles) near the start of its own printed line, since Tesseract typically emits one
# printed line per output line for this kind of card layout. Deliberately has no end-of-line
# anchor: a real capture has shown Tesseract appending a stray misread artifact after the actual
# name (e.g. "CHRISTOPHER [", the '[' being a misread scan-frame/table-border pixel) -- capturing
# only the leading run of valid name characters and letting the match end there tolerates that
# kind of trailing noise instead of rejecting the whole line because of it.
#
# "Near" the start, not exactly at it: .{0,8} tolerates a short run of leading OCR noise before
# the marker digit itself, rather than requiring it to be the very first non-whitespace character
# -- diagnosed against two real captures: a category-code fragment bled onto the same line as
# field 1 ("CE 1. NYBERG"), which a bare \s* prefix (matching only whitespace) rejected outright
# even though the actual marker and name were both read correctly; and, separately, leading noise
# that itself happened to contain an unrelated stray digit ("A 4 1. UTAS") -- excluding digits
# from the skip class (an earlier version of this pattern) rejected that one too, since the
# noise's own "4" blocked it from ever reaching the real "1" marker three characters later.
# Bounded to 8 characters (rather than unbounded) so this still can't wander off and match an
# unrelated digit elsewhere on a longer, noisier line.
#
# The whitespace between the field marker and its value is \s* (optional), not \s+ (required): a
# real capture had Tesseract drop the space after "4b." entirely ("4b.3108 2026" with no gap at
# all before the digits), which a mandatory \s+ would reject outright.
#
# The name-value character class is a broad Unicode range (A-Z plus every accented uppercase
# Latin-1 letter, À-Ö and Ø-Þ -- U+00D7 the multiplication sign is the one non-letter code point
# in that span and is excluded) rather than a hand-picked subset of just this app's supported-
# country accents (the previous Å/Ä/Ö/É-only class) -- diagnosed against a real capture where
# Tesseract's multi-language model (config/ocr_languages.py loads a dozen languages at once,
# specifically to read any supported country's diacritics) misread a printed 'Ö' as 'Ó' (O-acute,
# a Polish/Spanish/Hungarian glyph, not O-diaeresis) instead of failing cleanly on it -- which
# silently truncated "LARS GÖRAN" to "LARS G" mid-match, since 'Ó' wasn't in the narrower
# allowlist. Cross-language diacritic confusion between a dozen simultaneously-loaded languages'
# similar-looking accented glyphs isn't limited to any one specific pair, so this widens the net to
# every accented Latin uppercase letter instead of enumerating confusions one at a time as found.
_ACCENTED_UPPERCASE = "A-ZÀ-ÖØ-Þ"
_SURNAME_PATTERN = re.compile(rf"(?m)^.{{0,8}}?1[.:]?\s*([{_ACCENTED_UPPERCASE}][{_ACCENTED_UPPERCASE}' -]+)")
_GIVEN_NAME_PATTERN = re.compile(rf"(?m)^.{{0,8}}?2[.:]?\s*([{_ACCENTED_UPPERCASE}][{_ACCENTED_UPPERCASE}' -]+)")
_BIRTH_LINE_PATTERN = re.compile(r"(?m)^.{0,8}?3[.:]?\s*(.*\d.*)$")
# Anchored on "4b" specifically, not a bare "4" -- fields 4a (issue date) and 4c (authority) are
# printed right alongside it, and a bare "4" would ambiguously match 4a's line too.
_EXPIRY_LINE_PATTERN = re.compile(r"(?m)^.{0,8}?4b[.:]?\s*(.*\d.*)$")
# Separator and trailing-anchor tolerance matching the other fields above (see their comments) --
# this pattern hadn't been revisited for the same real-world OCR noise until a live capture of a
# previously-clean-reading card intermittently dropped its personal_number despite every other
# field on the same OCR pass succeeding, the same class of fragility already fixed for names/
# dates. The two-group-plus-explicit-join form matches national_id_parser.py's identical pattern.
#
# The field marker itself accepts '§' alongside '5': a live capture's diagnostic log caught
# Tesseract reading the bold '5.' as the section-sign symbol '§' instead of the digit -- a
# plausible glyph confusion at typical capture resolution, and not a one-off, since it broke this
# exact field on an otherwise clean OCR pass (every other field read correctly) twice in a row.
_PERSONAL_NUMBER_PATTERN = re.compile(r"(?m)^.{0,8}?[5§][.:]?\s*(\d{6})[-+.\s]+(\d{4})")


def parse_driving_licence_fields(ocr_text: str) -> DlFieldsResult:
    """Best-effort extraction of surname, given name(s), birth date, and expiry date from a driving licence's raw OCR text."""
    surname_match = _SURNAME_PATTERN.search(ocr_text)
    given_name_match = _GIVEN_NAME_PATTERN.search(ocr_text)
    birth_line_match = _BIRTH_LINE_PATTERN.search(ocr_text)
    expiry_line_match = _EXPIRY_LINE_PATTERN.search(ocr_text)
    personal_number_match = _PERSONAL_NUMBER_PATTERN.search(ocr_text)

    fields: dict[str, str] = {}
    if surname_match:
        fields["surname"] = surname_match.group(1).strip()
    if given_name_match:
        fields["name"] = given_name_match.group(1).strip()
    if birth_line_match:
        birth_date = extract_date_from_line(birth_line_match.group(1))
        if birth_date:
            fields["birth_date"] = birth_date
    if expiry_line_match:
        expiry_date = extract_date_from_line(expiry_line_match.group(1))
        if expiry_date:
            fields["expiry_date"] = expiry_date
    if personal_number_match:
        personal_number_digits, personal_number_suffix = personal_number_match.groups()
        fields["personal_number"] = f"{personal_number_digits}-{personal_number_suffix}"
        if "birth_date" not in fields:
            # Field 3 (the printed date-of-birth line) and field 5 (personnummer) are OCR'd and
            # matched independently above -- a real capture can have field 3's line come out too
            # garbled for extract_date_from_line to recover while field 5 still reads cleanly. A Swedish
            # personnummer's own first 6 digits are its holder's YYMMDD birth date (the same
            # encoding document/national_id_parser.py already derives birth_date from directly,
            # rather than separately OCR'd), so that's used here as a fallback source rather than
            # leaving birth_date -- and the DOB/age/is_minor flag that depends on it -- unpopulated
            # when the personnummer alone would have been enough.
            fields["birth_date"] = personal_number_digits

    return DlFieldsResult(found=bool(fields), fields=fields)
