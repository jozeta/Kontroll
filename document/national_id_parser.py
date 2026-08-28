"""Parses name, personal-number, birth-date, and expiry-date fields out of a Swedish national ID card's raw OCR text.

Unlike the EU driving licence (document/dl_field_parser.py), the Swedish national ID card
("Nationellt ID-kort") has no numbered-field layout to anchor on -- its printed labels
('Efternamn/Surname', 'Förnamn/Given names', ...) are the only structural markers, and OCR has
been observed mangling those labels heavily and inconsistently (e.g. 'Efternamn/ Sumame, s',
'vee. Efternamryy Sumome') while leaving the actual data values themselves comparatively clean.
So this parser deliberately does NOT anchor on the (unreliable) labels at all. Instead:

- The card's title line, 'NATIONELLT IDENTITETSKORT', has been reliably OCR'd intact across every
  real sample seen so far (no accented characters, plain bold caps) -- used as a start-of-content
  anchor to search after, skipping the country/title lines above it.
- Surname and given name(s) are found as the first two standalone, purely-uppercase name-shaped
  lines after that anchor, explicitly skipping known non-name label lines ('IDENTITY CARD' and its
  OCR variants, 'SVERIGE'/'SWEDEN', the 'CAN' card-access-number label) that would otherwise also
  look name-shaped.
- The personal number (personnummer, format YYMMDD-XXXX) is found by its own distinctive digit
  pattern, tolerant of the separator OCR misreads as a period or drops into a stray space, and its
  first 6 digits double as the birth date -- a personnummer's date-of-birth encoding is exactly
  the same YYMMDD document.dates.parse_mrz_style_date already knows how to century-resolve for
  MRZ documents, so birth date is derived from it directly rather than separately OCR'd from the
  card's bilingual 'DD MON/MON YY' date line (which has proven less reliable to parse than the
  numeric personnummer).
- The expiry date is read from that same bilingual 'DD MON/MON YY' line -- there are exactly two
  such lines on the card (birth date, then expiry date, in that reading order), so the LAST match
  in the text is taken as expiry.

This is a heuristic best-effort parser, not a validated implementation -- same caveat as
dl_field_parser.py. A parse failure here degrades gracefully to '-' placeholders in the UI, the
same as for any other document with no structured fields available.
"""

import re
from dataclasses import dataclass, field

# Swedish month abbreviations (as printed on the card) mapped to their number; English equivalents
# added for the two months where Swedish and English abbreviations differ (MAJ/May, OKT/Oct) in
# case OCR happens to pick up the English half of the bilingual "MAJ/MAY"-style pair instead.
_MONTH_ABBREVIATIONS: dict[str, int] = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAJ": 5, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OKT": 10, "OCT": 10, "NOV": 11, "DEC": 12,
}

_TITLE_ANCHOR_PATTERN = re.compile(r"IDENTITETSKORT")

# Matches one uppercase name-shaped token anywhere on a line, rather than anchoring a single regex
# to a bounded run of leading junk at the very start of the line (the previous approach) --
# diagnosed against a real capture where that anchored form missed both actual name lines
# outright: "mea > NYBERG" has 6 characters of leading OCR noise before the name, one more than
# the old {0,5} skip budget allowed, and "N LARS GÖRAN . N" has a single stray OCR-noise
# letter glued directly onto the front of the name with no separator at all, which the old
# pattern's start-of-line anchor could never skip past. Both failures left an unrelated,
# coincidentally all-caps-shaped garbage line (a garbled nationality field, "MM SVENSKEEZZE")
# matching *instead* and being mistaken for the real name. Scanning the whole line for uppercase
# runs -- see _find_name_candidates -- finds the real name regardless of how much or how little
# junk surrounds it on either side.
#
# Requires 3+ letters per token (not 2+): a real capture showed a faint repeating security
# watermark printed diagonally across the photo ("SVE"/"SWE") get OCR'd as short, low-confidence
# fragments ("TEA", "SN") that otherwise looked exactly as name-shaped as a genuine short token
# would. Real Swedish/European given names and surnames are essentially never under 3 letters, so
# this filters the shortest, least name-like noise without a real risk of dropping a genuine name
# -- it doesn't fully solve the watermark-noise problem (a 3+ letter fragment can still slip
# through, as "TEA" itself does), which would need per-word OCR confidence threaded through here
# to fix properly; the two OCR calls this app makes over the same image (image_to_string for this
# text, image_to_data for confidence) don't reproduce each other's line structure closely enough
# to correlate a given line back to its own words' confidence, so this is a partial mitigation,
# not a complete one.
#
# Character class is a broad Unicode range (see dl_field_parser.py's _ACCENTED_UPPERCASE for the
# identical reasoning: this app's dozen-language OCR setup has been observed misreading one
# accented letter as a different one, e.g. 'Ö' as 'Ó', which a narrower hand-picked allowlist
# wouldn't accept).
_NAME_TOKEN_PATTERN = re.compile(r"[A-ZÀ-ÖØ-Þ]{3,}(?:['\-][A-ZÀ-ÖØ-Þ]{3,})*")

# Non-name lines that are also purely-uppercase and would otherwise look name-shaped to
# _NAME_TOKEN_PATTERN -- checked as substrings since OCR's exact misreading of these varies
# ("IDENTITY CARD" / "IDENTITY CARO" / "TOENTITY CARD" all seen in real captures).
_NAME_CANDIDATE_DENYLIST_SUBSTRINGS = ("IDENT", "CARD", "CARO", "SVERIGE", "SWEDEN", "CAN")

# Separator between the personnummer's two digit groups, tolerant of OCR turning the printed '-'
# into a '.' or dropping it into a bare space (both seen in real captures).
_PERSONAL_NUMBER_PATTERN = re.compile(r"(\d{6})[-+.\s]+(\d{4})")

# The card's two bilingual "DD MON/MON YY" date lines (birth date, then expiry date) -- e.g.
# "9 JAN/JAN 69" or "29 JUN/JUN 31". Only the month code is captured; the trailing one (after the
# separator) is intentionally ignored since it's the same date repeated in the other language. The
# separator itself is a single optional character of (almost) any kind, not specifically '/' --
# diagnosed against a real capture where OCR read "JUN/JUN" as "JUNIJUN" (the '/' misread as 'I'
# with no separator character recognizable as such at all), which made this pattern fail to match
# the expiry-date line entirely while the birth-date line's cleanly-OCR'd '/' still matched --
# leaving only one match overall where two were expected, so parse_national_id_fields' "last match
# is the expiry date" logic silently grabbed the birth date's own match instead.
_MONTH_NAME_DATE_PATTERN = re.compile(r"(\d{1,2})\s+([A-ZÅÄÖ]{3}).?[A-ZÅÄÖ]{3}\s+(\d{2})")


@dataclass
class NationalIdFieldsResult:
    """Whether Swedish national-ID-card-style fields were found in the OCR text, and what was extracted."""

    found: bool
    fields: dict[str, str] = field(default_factory=dict)


def _find_name_candidates(text_after_anchor: str) -> list[str]:
    """Return every line's run of uppercase name-shaped tokens after the card-title anchor, minus known non-name labels, in reading order.

    Checks the denylist against the whole line (not just the matched tokens) so a label line like
    "IDENTITY CARD" is still excluded even though "IDENTITY"/"CARD" are exactly the kind of
    all-caps tokens _NAME_TOKEN_PATTERN looks for -- excluding by content, not by position, is
    what _NAME_TOKEN_PATTERN's line-wide scan needs since it no longer assumes a name starts at a
    predictable offset into the line.
    """
    candidates = []
    for line in text_after_anchor.splitlines():
        if any(denied in line for denied in _NAME_CANDIDATE_DENYLIST_SUBSTRINGS):
            continue
        tokens = _NAME_TOKEN_PATTERN.findall(line)
        if not tokens:
            continue
        candidates.append(" ".join(tokens))
    return candidates


def _month_name_date_to_yymmdd(day: str, month_code: str, year_2digit: str) -> str | None:
    month_number = _MONTH_ABBREVIATIONS.get(month_code)
    if month_number is None:
        return None
    return f"{year_2digit}{month_number:02d}{int(day):02d}"


def parse_national_id_fields(ocr_text: str) -> NationalIdFieldsResult:
    """Best-effort extraction of surname, given name(s), personal number, birth date, and expiry date from a Swedish national ID card's raw OCR text."""
    fields: dict[str, str] = {}

    title_match = _TITLE_ANCHOR_PATTERN.search(ocr_text)
    if title_match is not None:
        # Skip to the start of the *next* line, not just past the matched text itself -- the title
        # anchor doesn't always land on its own dedicated OCR line (a real capture had it fused
        # onto a noisier line, "�SEE | NATIONELLT IDENTITETSKORT EI", where the title match ends
        # mid-line with trailing garbage -- "EI" -- still ahead of it before the next real
        # newline). Searching from title_match.end() directly let that trailing garbage count as
        # the first "line" _find_name_candidates saw, and its "EI" token beat the real surname to
        # become the first (wrong) candidate.
        next_line_start = ocr_text.find("\n", title_match.end())
        search_start = next_line_start + 1 if next_line_start != -1 else len(ocr_text)
        name_candidates = _find_name_candidates(ocr_text[search_start:])
        if len(name_candidates) >= 1:
            fields["surname"] = name_candidates[0]
        if len(name_candidates) >= 2:
            fields["name"] = name_candidates[1]

    personal_number_match = _PERSONAL_NUMBER_PATTERN.search(ocr_text)
    if personal_number_match:
        personal_number_digits, personal_number_suffix = personal_number_match.groups()
        fields["personal_number"] = f"{personal_number_digits}-{personal_number_suffix}"
        # A personnummer's own first 6 digits are its holder's YYMMDD birth date -- the same raw
        # format document.dates.parse_mrz_style_date already century-resolves for MRZ documents,
        # reused here rather than separately (and less reliably) OCR'd off the card's bilingual
        # date line. Not true for a samordningsnummer (coordination number, day-of-month +60) --
        # an accepted gap, consistent with this parser's best-effort heuristic scope.
        fields["birth_date"] = personal_number_digits

    date_line_matches = _MONTH_NAME_DATE_PATTERN.findall(ocr_text)
    if date_line_matches:
        # Birth date (first match) is derived from the personnummer above instead; expiry date is
        # the LAST such line in reading order -- see module docstring.
        expiry_yymmdd = _month_name_date_to_yymmdd(*date_line_matches[-1])
        if expiry_yymmdd:
            fields["expiry_date"] = expiry_yymmdd

    return NationalIdFieldsResult(found=bool(fields), fields=fields)
