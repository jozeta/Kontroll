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
        return MrzResult(
            found=True,
            format_name=format_name,
            checksum_valid=bool(checker),
            fields=checker.fields()._asdict(),
            failed_checks=failed_checks,
        )

    return MrzResult(found=False, format_name=None, checksum_valid=False, fields={}, failed_checks=[])
