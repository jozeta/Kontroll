"""Date parsing and age calculation shared between document field cross-checking (age/expiry
flags in field_crosscheck.py) and UI display of extracted MRZ/driving-licence fields
(ui/main_window.py).
"""

import re
from datetime import date, datetime

# Shared by every OCR-heuristic field parser that has to pull a printed calendar date out of a raw
# text line (document/dl_field_parser.py's field 3/4b lines, document/eu_id_card_parser.py's "Date
# of birth"/"Date of expiry" label lines): multiple formats are tried since EU guidance specifies
# dd.mm.yyyy but Swedish practice commonly prints ISO-style yyyy-mm-dd instead. The separator is a
# character class and OPTIONAL (a '?', not required), because real captures have shown Tesseract
# mangling the printed '.' between date components in several different ways -- dropped into a
# plain space ("28.07.1981" -> "28.07 1981"), dropped entirely with the digits running together
# ("18.12.1972" -> "18121972"), or dropped for only one of the two separators ("22.07.1968" ->
# "2207 1968"). Making the separator optional handles all of these (and the normal, cleanly-
# punctuated case) with one pattern each, rather than a combinatorial explosion of "which separator
# was dropped where" variants. The day<=31/month<=12/year bounds in extract_date_from_line below are
# what keep this from over-matching once separators are no longer required between every digit pair.
_DATE_SEPARATOR = r"[-./ ]?"
_DATE_LINE_PATTERNS = (
    re.compile(rf"(\d{{4}}){_DATE_SEPARATOR}(\d{{2}}){_DATE_SEPARATOR}(\d{{2}})"),
    re.compile(rf"(\d{{2}}){_DATE_SEPARATOR}(\d{{2}}){_DATE_SEPARATOR}(\d{{4}})"),
)


def extract_date_from_line(line: str) -> str | None:
    """Pull an ISO 'YYYY-MM-DD' date out of a printed document field line, trying each supported date format in turn -- see _DATE_LINE_PATTERNS' comment for why both are needed and why the separator is optional."""
    for pattern in _DATE_LINE_PATTERNS:
        match = pattern.search(line)
        if match is None:
            continue
        first, second, third = match.groups()
        # The ISO pattern's groups are already (year, month, day); the other two are (day, month, year).
        year, month, day = (first, second, third) if len(first) == 4 else (third, second, first)
        # The year bound matters as much as the day/month one below, not just for plausibility: a
        # caller's own capture line can run on to include the *next* printed field if OCR drops the
        # line break between them (diagnosed in dl_field_parser.py's expiry-date line, where this
        # bound is what stopped an unrelated card-number tail from being matched as a date instead
        # of the real one earlier in the same line). 1900-2099 comfortably covers both a birth date
        # and an expiry date for any document this app will see.
        if not (1900 <= int(year) <= 2099 and 1 <= int(day) <= 31 and 1 <= int(month) <= 12):
            continue
        return f"{year}-{month}-{day}"
    return None


def _infer_full_year_for_past_date(two_digit_year: int) -> int:
    """Infer a full 4-digit year for a two-digit year assumed to lie in the past (birth dates): whichever century doesn't put the date in the future."""
    current_two_digit_year = datetime.now().year % 100
    century = 2000 if two_digit_year <= current_two_digit_year else 1900
    return century + two_digit_year


def parse_mrz_style_date(date_field: str, *, assume_past: bool) -> date | None:
    """Parse a stored date field into a real date.

    Two source formats are supported: an MRZ-style 6-digit 'YYMMDD' string (2-digit year, century
    inferred since the MRZ only encodes 2 digits), and a full ISO 'YYYY-MM-DD' string as produced
    by the driving-licence OCR heuristic in document/dl_field_parser.py (which already has an
    unambiguous 4-digit year, so no century inference is needed there).

    assume_past picks the century-inference rule for the 6-digit case. Birth dates are always in
    the past, so a 2-digit year greater than the current 2-digit year must mean 19XX, not a future
    birth (assume_past=True). Expiry dates run the other way -- they're commonly a decade or so in
    the future -- and since the MRZ format itself postdates 1980, century is simply always taken
    as 20XX for those (assume_past=False); a live document old enough for that to be wrong isn't a
    realistic case at a staffed desk.
    """
    try:
        return date.fromisoformat(date_field)
    except ValueError:
        pass

    if len(date_field) != 6 or not date_field.isdigit():
        return None
    year_2digit, month, day = int(date_field[0:2]), int(date_field[2:4]), int(date_field[4:6])
    year = _infer_full_year_for_past_date(year_2digit) if assume_past else 2000 + year_2digit
    try:
        return date(year, month, day)
    except ValueError:
        # A malformed/misread MRZ can decode to an impossible calendar date (e.g. month 19);
        # that's a data-quality problem to surface as "unknown", not a bug to crash on.
        return None


def calculate_age(birth_date: date, as_of: date | None = None) -> int:
    """Calculate age in whole years as of a given date (defaults to today)."""
    as_of = as_of or date.today()
    age = as_of.year - birth_date.year
    if (as_of.month, as_of.day) < (birth_date.month, birth_date.day):
        age -= 1
    return age
