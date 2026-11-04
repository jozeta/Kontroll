"""Locates, parses, and checksum-validates an ICAO 9303 Machine Readable Zone block."""

import re
from dataclasses import dataclass

from mrz.checker.td1 import TD1CodeChecker
from mrz.checker.td2 import TD2CodeChecker
from mrz.checker.td3 import TD3CodeChecker

# The three ICAO 9303 MRZ formats this app recognizes, keyed by their fixed per-line character
# width -- widths are unique across formats (30 / 36 / 44), so a candidate line's length alone is
# enough to tell which format it belongs to, with no ambiguity between them.
# (format name, characters per line, number of lines, checker class)
_MRZ_FORMATS = [
    ("TD1", 30, 3, TD1CodeChecker),  # national ID cards
    ("TD2", 36, 2, TD2CodeChecker),  # some ID cards and visas
    ("TD3", 44, 2, TD3CodeChecker),  # passports
]

_MRZ_LINE_PATTERN = re.compile(r"^[A-Z0-9<]+$")

# TD3 (passport) line 1 encodes only document type, issuing country, and name -- ICAO 9303 puts no
# check digit on any of that, only on line 2's data fields (document number, birth date, expiry
# date, personal number, plus a final composite check). So a genuine, cleanly-read line 2 is fully
# self-verifying on its own, independent of whatever line 1 says. This matters because a real
# capture of an *entire open passport book* (both pages, not just the bio-data page -- physically
# unavoidable for a bound passport, see document/mrz_locator.py) has repeatedly OCR'd line 2
# perfectly while line 1 gets occasional filler-character noise (a '<' misread as 'K'/'X') that
# throws off its required exact 44-character length -- meaning a good line 2 was being discarded
# entirely just because line 1 didn't also come out clean.
#
# Pairing a real line 2 with this placeholder line 1 (a syntactically valid but content-free
# TD3 line 1) satisfies the `mrz` library's requirement for two lines without asserting anything
# about the real name/country -- verified empirically that every data checksum still reports
# correctly against a real line 2 this way; only the 'identifier' check (a line-1-content sanity
# check, not a checksum) fails, which _TD3_LINE1_DEPENDENT_CHECKS excludes below.
_TD3_LINE_LENGTH = 44
_TD3_PLACEHOLDER_LINE1 = "P<XXX" + "<" * (_TD3_LINE_LENGTH - 5)

# Checks that read line 1's own content and are therefore meaningless (neither trustworthy if they
# pass nor a sign of bad data if they fail) against the placeholder -- excluded so only line 2's
# real data checksums decide whether this recovery counts as valid.
_TD3_LINE1_DEPENDENT_CHECKS = {"identifier", "valid country code", "document type format"}


@dataclass
class MrzResult:
    """The outcome of locating and validating an MRZ block: whether one was found, whether checksums passed, and its parsed fields."""

    found: bool
    format_name: str | None
    checksum_valid: bool
    fields: dict[str, str]
    failed_checks: list[str]


def _normalize_lines(raw_text: str) -> list[str]:
    """Uppercase and strip incidental whitespace from OCR text lines, keeping only lines using the MRZ character set (A-Z, 0-9, '<')."""
    normalized_lines = (line.strip().upper().replace(" ", "") for line in raw_text.splitlines())
    return [line for line in normalized_lines if _MRZ_LINE_PATTERN.match(line)]


def _sanitize_fields(fields: dict[str, str]) -> dict[str, str]:
    """Replace the literal string "None" with an empty string in every field value.

    The `mrz` library returns the literal string "None" (not an actual empty value) for a field
    it couldn't parse out of a corrupted MRZ line -- observed on a real capture whose line 1 (the
    name line) had OCR noise substituted into its filler characters, checksum-independent of line
    2 (which carried valid data and is why this MRZ still counted as "found"). Left unsanitized,
    that literal "None" string flows straight through to the UI (e.g. `_derive_display_fields`'s
    `.strip() or "-"` treats "None" as a normal non-empty name and displays it verbatim) instead of
    degrading to the usual "-" placeholder for a field that wasn't actually recovered.
    """
    return {key: ("" if value == "None" else value) for key, value in fields.items()}


def _with_swedish_personal_number(fields: dict[str, str]) -> dict[str, str]:
    """Derive a personnummer-formatted "personal_number" field from a Swedish passport's TD3 "optional_data" field, if present.

    TD3's "optional data" field (line 2, positions 29-42) is a country-specific free-use area under
    ICAO 9303, not a standardized field -- Swedish passports use it to encode the holder's full
    personnummer (century + YYMMDD + 4-digit suffix, e.g. "198107280292" for personnummer
    "810728-0292"), but this isn't a universal convention, so it's only applied here when the
    document's nationality/country is Sweden, to avoid misreading another country's arbitrary use
    of that field as a fake personnummer. Reformatted to the same "YYMMDD-XXXX" convention used
    everywhere else in this app (document/dl_field_parser.py, document/national_id_parser.py) for
    document/flag_matcher.py to cross-check consistently regardless of source document type.
    """
    is_swedish = fields.get("nationality", "").strip("<").upper() == "SWE"
    optional_data_digits = fields.get("optional_data", "").strip("<")
    if is_swedish and len(optional_data_digits) >= 10 and optional_data_digits.isdigit():
        personnummer_digits = optional_data_digits[-10:]
        fields = {**fields, "personal_number": f"{personnummer_digits[:6]}-{personnummer_digits[6:]}"}
    return fields


def _try_td3_from_line2_only(candidate_lines: list[str]) -> MrzResult | None:
    """Attempt TD3 (passport) recovery from a lone valid-length line 2, when no valid-length line 1 was also found. See _TD3_PLACEHOLDER_LINE1's comment for why this is safe."""
    for candidate in candidate_lines:
        if len(candidate) != _TD3_LINE_LENGTH:
            continue
        mrz_code = _TD3_PLACEHOLDER_LINE1 + "\n" + candidate
        try:
            checker = TD3CodeChecker(mrz_code)
        except Exception:  # noqa: S112 -- expected for non-MRZ candidates; try the next one
            # A genuine line 1 (misclassified here as a "lone line 2" candidate) or other
            # malformed input can make the checker's own parsing raise -- try the next candidate
            # rather than letting this best-effort recovery attempt crash the capture.
            continue

        failed_checks = [
            description
            for description, passed in checker.report.fields
            if not passed and description not in _TD3_LINE1_DEPENDENT_CHECKS
        ]
        if failed_checks:
            continue

        fields = _sanitize_fields(checker.fields()._asdict())
        # "country" came from the fabricated placeholder line 1 and is meaningless; "nationality"
        # is real (read from line 2) and, for the overwhelming majority of passports, identical to
        # the issuing country -- used here so doc_classifier still gets a usable country code
        # instead of losing it entirely for this recovery path.
        fields["country"] = fields.get("nationality", "")
        fields = _with_swedish_personal_number(fields)
        return MrzResult(
            found=True,
            format_name="TD3 (line 2 only)",
            checksum_valid=True,
            fields=fields,
            failed_checks=[],
        )
    return None


def parse_and_validate_mrz(raw_text: str) -> MrzResult:
    """Locate an MRZ block within raw OCR text and, if found, checksum-validate and parse it against the matching ICAO 9303 format.

    Real captured documents produce OCR text with noise (extra whitespace, misread characters,
    stray non-MRZ lines) mixed in around the MRZ block; this only requires a candidate line to be
    composed entirely of the MRZ character set and match one of the three known line widths -- it
    does not attempt to correct misrecognized characters within a line, so a badly-OCR'd MRZ will
    correctly fail checksum validation rather than being silently "corrected" into passing.
    """
    candidate_lines = _normalize_lines(raw_text)

    for format_name, line_length, line_count, checker_class in _MRZ_FORMATS:
        matching_lines = [line for line in candidate_lines if len(line) == line_length]
        if len(matching_lines) < line_count:
            continue

        mrz_code = "\n".join(matching_lines[:line_count])
        checker = checker_class(mrz_code)
        failed_checks = [description for description, passed in checker.report.fields if not passed]
        fields = _sanitize_fields(checker.fields()._asdict())
        if format_name == "TD3":
            fields = _with_swedish_personal_number(fields)
        return MrzResult(
            found=True,
            format_name=format_name,
            checksum_valid=bool(checker),
            fields=fields,
            failed_checks=failed_checks,
        )

    line2_only_result = _try_td3_from_line2_only(candidate_lines)
    if line2_only_result is not None:
        return line2_only_result

    return MrzResult(found=False, format_name=None, checksum_valid=False, fields={}, failed_checks=[])
