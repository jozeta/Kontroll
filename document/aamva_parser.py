"""Parses AAMVA-format PDF417 barcode payloads -- the standardized data layout printed on the
back of every US state's (and Canadian province's) driver's license/ID card, per the AAMVA DL/ID
Card Design Standard.

Unlike the Swedish driving licence PDF417 layout (see barcode_parser.py's caveat, which has never
been validated against a real sample), AAMVA is a single, precisely published, and essentially
universal standard used identically by every issuing US/Canadian jurisdiction -- every real US
driver's license/ID barcode follows it -- so its data elements can be parsed deterministically
here with confidence, without needing a real physical sample to validate the layout against the
way barcode_parser.py's Swedish caveat does.

Only the small subset of AAMVA's ~40 data elements this app's document pipeline actually uses
elsewhere (name, birth date, expiry date, license number, jurisdiction/country -- the same fields
document/mrz_parser.py and document/dl_field_parser.py populate for other document types) are
extracted; every other element (address, height, eye color, sex, ...) is left unparsed since
nothing downstream needs them.
"""

import re
from dataclasses import dataclass, field

# Present near the start of every compliant AAMVA barcode payload, right after the compliance
# indicator/separator control characters -- used here purely to detect "this barcode is AAMVA-
# formatted" before attempting to parse it as one, since any other barcode this app might scan
# (e.g. a Swedish driving licence's PDF417 payload) won't contain this literal text.
_AAMVA_HEADER_MARKER = "ANSI "

# "ANSI " + 6-digit IIN + 2-digit AAMVA version + 2-digit jurisdiction version + 2-digit entry
# count, immediately followed by the first subfile's own 2-character type designator -- "DL" for
# a driver's licence, "ID" for an identification card. Per the AAMVA header layout, that
# designator's position is fixed relative to "ANSI ", so this is an exact structural match, not a
# heuristic.
_HEADER_SUBFILE_TYPE_PATTERN = re.compile(r"ANSI \d{6}\d{2}\d{2}\d{2}(DL|ID)")

# Each element ID is matched only right after a segment boundary -- the start of the payload, a
# data-element-separator (LF) / segment-terminator (CR) / record-separator (RS) control character,
# or the subfile type marker ("DL"/"ID") that immediately precedes the *first* element in a
# subfile with no separator of its own before it (e.g. "...0008DLDAQT64235789" -- "DAQ" starts
# right after the literal "DL", not after a control character). Anchoring this way, rather than a
# bare substring search, avoids a false match on a 3-character sequence that happens to appear
# inside a *previous* element's own value.
_ELEMENT_BOUNDARY = r"(?:^|[\n\r\x1e]|DL|ID)"


def _extract_element(payload: str, element_id: str) -> str | None:
    """Return the value following one AAMVA element ID (e.g. "DCS" for last name) in the payload, or None if that element isn't present."""
    match = re.search(rf"{_ELEMENT_BOUNDARY}{element_id}([^\n\r\x1e]*)", payload)
    if match is None:
        return None
    value = match.group(1).strip()
    return value or None


def _parse_aamva_date(raw: str | None) -> str | None:
    """Convert an AAMVA MMDDCCYY-format date string (the modern standard's format for DBB/date-of-birth and DBA/expiration-date) to an ISO 'YYYY-MM-DD' string, or None if it isn't a well-formed 8-digit date."""
    if not raw or len(raw) != 8 or not raw.isdigit():
        return None
    month, day, year = raw[0:2], raw[2:4], raw[4:8]
    if not (1 <= int(month) <= 12 and 1 <= int(day) <= 31):
        return None
    return f"{year}-{month}-{day}"


@dataclass
class AamvaResult:
    """Whether a barcode payload was recognized as AAMVA-formatted, and which of this app's document fields could be extracted from it."""

    found: bool
    is_id_card: bool = False  # True for an "ID" (identification card) subfile, False for "DL" (driver's licence)
    fields: dict[str, str] = field(default_factory=dict)


def parse_aamva_barcode(payload: str) -> AamvaResult:
    """Parse one decoded PDF417 barcode payload as AAMVA data, if it looks like one.

    Population of `fields` mirrors the key names document/mrz_parser.py and
    document/dl_field_parser.py already use ("surname", "name", "birth_date", "expiry_date",
    "document_number") so downstream code (display, flagged-persons matching, under-18/expiry
    flags, visit-history matching) works the same regardless of which document type/country
    produced them -- see ui/main_window.py's capture handler for where this is dispatched from.
    "country" (ISO alpha-3, e.g. "USA"/"CAN") and "jurisdiction" (the issuing state/province's
    2-letter code) are also populated for document/doc_classifier.py's use, but aren't otherwise
    read elsewhere in the pipeline.
    """
    if _AAMVA_HEADER_MARKER not in payload:
        return AamvaResult(found=False)

    header_match = _HEADER_SUBFILE_TYPE_PATTERN.search(payload)
    is_id_card = header_match is not None and header_match.group(1) == "ID"

    fields: dict[str, str] = {}

    surname = _extract_element(payload, "DCS")
    if surname:
        fields["surname"] = surname

    given_name = _extract_element(payload, "DAC")
    middle_name = _extract_element(payload, "DAD")
    if middle_name and middle_name.upper() == "NONE":
        middle_name = None
    name_parts = [part for part in (given_name, middle_name) if part]
    if name_parts:
        fields["name"] = " ".join(name_parts)

    birth_date = _parse_aamva_date(_extract_element(payload, "DBB"))
    if birth_date:
        fields["birth_date"] = birth_date

    expiry_date = _parse_aamva_date(_extract_element(payload, "DBA"))
    if expiry_date:
        fields["expiry_date"] = expiry_date

    document_number = _extract_element(payload, "DAQ")
    if document_number:
        fields["document_number"] = document_number

    country = _extract_element(payload, "DCG")
    if country:
        fields["country"] = country

    jurisdiction = _extract_element(payload, "DAJ")
    if jurisdiction:
        fields["jurisdiction"] = jurisdiction

    return AamvaResult(found=bool(fields), is_id_card=is_id_card, fields=fields)


def parse_first_aamva_barcode(barcode_texts: list[str]) -> AamvaResult:
    """Try each decoded barcode in turn (a capture could in principle contain more than one) and return the first one that parses as AAMVA data, or a not-found result if none do."""
    for text in barcode_texts:
        result = parse_aamva_barcode(text)
        if result.found:
            return result
    return AamvaResult(found=False)
