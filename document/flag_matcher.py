"""Cross-checks a scan's extracted document fields against the operator-maintained flagged-persons watch list.

Matching is deliberately best-effort/heuristic, the same as the rest of the OCR-derived field
extraction in this app (see dl_field_parser.py's docstring) -- a false negative (a flagged person
not caught) degrades to the normal unflagged experience, while a false positive only costs the
operator a moment's extra attention on a row that turns red, so leaning permissive is the safer
failure direction for a desk-facing warning rather than an automated block.

Personal-number matching currently only works for documents whose doc_fields includes a
"personal_number" key -- today that's Swedish driving licences only (document/dl_field_parser.py's
field 5). MRZ documents (passports, national ID cards) have no equivalent field extracted yet, so
those fall back to name-only matching -- a known limitation, not a bug.
"""

import re
from dataclasses import dataclass

from storage.models import FlaggedPerson

_NAME_TOKEN_PATTERN = re.compile(r"[A-ZÅÄÖÉ]+")


@dataclass
class FlagMatch:
    """A flagged-persons watch-list entry that matched this scan, and which field the match was found on."""

    flagged_person: FlaggedPerson
    matched_on: str  # "name" / "personal_number"


def normalize_personal_number(raw: str) -> str:
    """Strip everything but digits and keep only the last 10 -- the YYMMDD-XXXX personnummer body -- so a 12-digit (century-prefixed) and a 10-digit rendering of the same number compare equal."""
    digits = re.sub(r"\D", "", raw)
    return digits[-10:] if len(digits) >= 10 else digits


def normalize_name_tokens(raw: str) -> frozenset[str]:
    """Break a name into a set of uppercase alphabetic tokens (2+ letters), order- and whitespace-insensitive, for comparing 'NYBERG LARS GÖRAN' against 'Lars Göran Nyberg' alike."""
    return frozenset(token for token in _NAME_TOKEN_PATTERN.findall(raw.upper()) if len(token) > 1)


def _scan_name_tokens(doc_fields: dict[str, str]) -> frozenset[str]:
    full_name = f"{doc_fields.get('name', '')} {doc_fields.get('surname', '')}"
    return normalize_name_tokens(full_name)


def match_basis_label(flagged_person: FlaggedPerson) -> str:
    """Describe which of a flagged-persons entry's own fields find_matching_flagged_person will actually check a scan against -- shown next to the entry in Settings' Flagged Users tab (and, per-scan, via a matched scan's own stored matched_on -- see ui/main_window.py) so an operator can see at a glance what's driving the match, not just that a match happened."""
    has_name = bool(flagged_person.name)
    has_personal_number = bool(flagged_person.personal_number)
    if has_name and has_personal_number:
        return "Name & personal number"
    if has_personal_number:
        return "Personal number"
    if has_name:
        return "Name"
    return "Nothing set -- this entry can never match a scan"


def matched_on_display(matched_on: str) -> str:
    """Human-readable form of a stored FlagMatch.matched_on / flagged_match["matched_on"] value ("name"/"personal_number") -- shared so every place that surfaces it (the retroactive-match dialog, a scan's own flagged badge) uses identical wording."""
    return "personal number" if matched_on == "personal_number" else "name"


def find_matching_flagged_person(doc_fields: dict[str, str], flagged_people: list[FlaggedPerson]) -> FlagMatch | None:
    """Return the first flagged-persons entry whose name and/or personal number matches this scan's extracted fields, else None.

    A personal-number match requires both sides to normalize to a full 10-digit body (guards
    against a short/garbled OCR read coincidentally matching a short garbled watch-list entry). A
    name match requires every token the operator entered to appear somewhere in the scan's
    extracted name -- so a flagged entry of just a surname still catches a full extracted name,
    without requiring an exact full-name match either direction.
    """
    scan_personal_number = normalize_personal_number(doc_fields.get("personal_number", ""))
    scan_name_tokens = _scan_name_tokens(doc_fields)

    for flagged_person in flagged_people:
        if flagged_person.personal_number and scan_personal_number:
            flagged_number = normalize_personal_number(flagged_person.personal_number)
            if len(flagged_number) == 10 and flagged_number == scan_personal_number:
                return FlagMatch(flagged_person=flagged_person, matched_on="personal_number")

        if flagged_person.name and scan_name_tokens:
            flagged_tokens = normalize_name_tokens(flagged_person.name)
            if flagged_tokens and flagged_tokens.issubset(scan_name_tokens):
                return FlagMatch(flagged_person=flagged_person, matched_on="name")

    return None
