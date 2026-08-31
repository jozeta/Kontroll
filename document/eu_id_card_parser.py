"""Parses name/birth-date/nationality/expiry-date fields out of a non-Swedish EU/EEA national ID
card's raw OCR text, using the English-language field labels EU Regulation 2019/1157 (the "eID
card" regulation) has required on every member state's national ID card since 2 August 2021:
'Surname', 'Given name(s)', 'Nationality', 'Date of birth', 'Sex', 'Date of expiry' (Annex I).

document/national_id_parser.py already covers the Swedish national ID card specifically (anchored
on its own printed title, since its labels have been observed OCR'ing unreliably) and is tried
first for any document classified as national_id -- see ui/main_window.py::_analyze_capture. This
module is the fallback for every *other* country's card, anchored on the regulation-mandated
English labels themselves rather than a per-country translated label table, since:

- Every field is printed in the issuing country's own language(s) *plus* at least one other
  official EU language, and English is near-universally chosen as that second language in practice
  -- so anchoring on the English label works across member states without needing a translated
  label table for each one (the same reasoning document/dl_field_parser.py already uses for the EU
  driving licence's field-number layout, just via a shared label vocabulary instead of shared field
  numbers).
- No real non-Swedish EU ID card OCR sample was available to validate a per-country layout against
  (the operator-reported gap was specifically "extracts nothing for non-Swedish national IDs", not
  a known-wrong specific-country layout) -- this is a heuristic best-effort parser, not a validated
  implementation, same caveat as dl_field_parser.py/national_id_parser.py/barcode_parser.py.

A parse failure here degrades gracefully to '-' placeholders in the UI, same as any other document
with no structured fields available.
"""

import re
from dataclasses import dataclass, field

from document.dates import extract_date_from_line

# Each label is searched as its own anchor; the value is taken from whatever follows it on the
# same line (skipping a translated label/separator that may sit between the English label and the
# actual value, e.g. "Efternamn / Surname NYBERG") or, failing that, from the next line (covering a
# layout where the label and value are printed on separate lines entirely). See
# _extract_value_after_label for the shared same-line-then-next-line search this uses for every
# field below, rather than each field re-implementing its own line-lookahead.
_SURNAME_LABEL = re.compile(r"SURNAME")
_GIVEN_NAME_LABEL = re.compile(r"GIVEN NAMES?")
_NATIONALITY_LABEL = re.compile(r"NATIONALITY")
_BIRTH_DATE_LABEL = re.compile(r"DATE OF BIRTH")
_EXPIRY_DATE_LABEL = re.compile(r"DATE OF EXPIRY")

# Same broad accented-Latin-uppercase character class dl_field_parser.py/national_id_parser.py both
# use, for the same reason (this app's dozen-language OCR setup has been observed misreading one
# accented letter as a visually similar one from a different language's alphabet -- see those
# modules' own comments on _ACCENTED_UPPERCASE for the specific diagnosed case).
_NAME_VALUE_PATTERN = re.compile(r"[A-ZÀ-ÖØ-Þ]{2,}(?:['\-][A-ZÀ-ÖØ-Þ]{2,})*(?:\s+[A-ZÀ-ÖØ-Þ]{2,})*")

# ICAO 3166-1 alpha-3-shaped nationality code (e.g. "DEU", "FRA") -- printed as a plain code on most
# EU ID cards' nationality field, matching the same alpha-3 convention this app's MRZ parsing
# already relies on elsewhere (document/mrz_parser.py).
_NATIONALITY_CODE_PATTERN = re.compile(r"\b([A-Z]{3})\b")


@dataclass
class EuIdCardFieldsResult:
    """Whether EU-eID-card-style English field labels were found in the OCR text, and what was extracted."""

    found: bool
    fields: dict[str, str] = field(default_factory=dict)


def _extract_value_after_label(text: str, label_pattern: re.Pattern, value_pattern: re.Pattern) -> str | None:
    """Find label_pattern anywhere in text, then return a value_pattern match from either an unambiguous "LABEL: VALUE" same-line form, or (the far more common real layout) the next line entirely -- None if the label itself isn't found or no value matches either place.

    Only two same-line forms are trusted: an explicit "LABEL: VALUE" (colon-separated), or the
    value starting immediately after the label with nothing but whitespace in between (e.g.
    "Surname MUELLER"). A bilingual label with no colon (e.g. "Given names / Vorname HANS PETER")
    matches neither -- there's no reliable boundary between its own second-language translation and
    an actual value that might follow it, both looking exactly as name-shaped to value_pattern as
    the real value does, and there's no real sample available to derive a more precise rule from
    (see this module's docstring) -- so that case always falls through to the next line instead,
    which is where a value actually printed on its own separate line is anyway.
    """
    label_match = label_pattern.search(text)
    if label_match is None:
        return None

    line_end = text.find("\n", label_match.end())
    same_line_remainder = text[label_match.end() : line_end if line_end != -1 else len(text)]

    colon_index = same_line_remainder.find(":")
    if colon_index != -1:
        same_line_value = value_pattern.search(same_line_remainder[colon_index + 1 :])
        if same_line_value:
            return same_line_value.group(0).strip()

    immediate_value = value_pattern.match(same_line_remainder.lstrip())
    if immediate_value:
        return immediate_value.group(0).strip()

    if line_end == -1:
        return None
    next_line_end = text.find("\n", line_end + 1)
    next_line = text[line_end + 1 : next_line_end if next_line_end != -1 else len(text)]
    next_line_value = value_pattern.search(next_line)
    return next_line_value.group(0).strip() if next_line_value else None


def parse_eu_id_card_fields(ocr_text: str) -> EuIdCardFieldsResult:
    """Best-effort extraction of surname, given name(s), nationality, birth date, and expiry date from a non-Swedish EU/EEA national ID card's raw OCR text, anchored on EU Regulation 2019/1157's mandatory English field labels."""
    normalized_text = ocr_text.upper()
    fields: dict[str, str] = {}

    surname = _extract_value_after_label(normalized_text, _SURNAME_LABEL, _NAME_VALUE_PATTERN)
    if surname:
        fields["surname"] = surname

    given_name = _extract_value_after_label(normalized_text, _GIVEN_NAME_LABEL, _NAME_VALUE_PATTERN)
    if given_name:
        fields["name"] = given_name

    nationality = _extract_value_after_label(normalized_text, _NATIONALITY_LABEL, _NATIONALITY_CODE_PATTERN)
    if nationality:
        fields["nationality"] = nationality

    birth_date_line = _extract_value_after_label(normalized_text, _BIRTH_DATE_LABEL, re.compile(r".*\d.*"))
    if birth_date_line:
        birth_date = extract_date_from_line(birth_date_line)
        if birth_date:
            fields["birth_date"] = birth_date

    expiry_date_line = _extract_value_after_label(normalized_text, _EXPIRY_DATE_LABEL, re.compile(r".*\d.*"))
    if expiry_date_line:
        expiry_date = extract_date_from_line(expiry_date_line)
        if expiry_date:
            fields["expiry_date"] = expiry_date

    return EuIdCardFieldsResult(found=bool(fields), fields=fields)
