"""Date parsing and age calculation shared between document field cross-checking (age/expiry
flags in field_crosscheck.py) and UI display of extracted MRZ/driving-licence fields
(ui/main_window.py).
"""

from datetime import date, datetime


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
