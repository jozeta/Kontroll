"""Finds a scan's prior visits by matching its extracted document fields against every other saved
scan -- powers the "seen before" visit counter and the last-5-visits list shown in the scan detail
view (see ui/main_window.py::HistoryRowWidget).

Reuses document/flag_matcher.py's normalization helpers (personnummer digits-only/last-10, name
tokens uppercase/order-insensitive) but applies a stricter match rule than the flagged-persons
watch list: an exact personal-number match, or otherwise an exact name-token-*set* match (not the
watch list's subset match). A false "already seen" hit here only adds noise to an operator-facing
counter, but the watch list's looser subset match (a flagged surname alone catching a fuller
extracted name) would be too eager here -- it would merge two different people who happen to share
a surname into one visit history, which the flagged-persons list's asymmetry (operator-entered
watch-list token vs. machine-read name) doesn't have to worry about since both sides here are
machine-read.
"""

import json

from document.flag_matcher import normalize_name_tokens, normalize_personal_number
from storage.models import ScanRecord


def _scan_doc_fields(scan: ScanRecord) -> dict[str, str]:
    return json.loads(scan.doc_fields_json) if scan.doc_fields_json else {}


def find_previous_scans(doc_fields: dict[str, str], scans: list[ScanRecord], exclude_scan_id: int | None = None) -> list[ScanRecord]:
    """Return every scan in `scans` that represents the same person as `doc_fields`, newest first.

    `scans` is expected newest-first (as storage.models.list_scans already returns), which this
    function preserves rather than re-sorts. `exclude_scan_id` is normally the scan `doc_fields`
    itself came from, so it doesn't count as its own "previous visit".
    """
    personal_number = normalize_personal_number(doc_fields.get("personal_number", ""))
    name_tokens = normalize_name_tokens(f"{doc_fields.get('name', '')} {doc_fields.get('surname', '')}")

    matches = []
    for scan in scans:
        if exclude_scan_id is not None and scan.id == exclude_scan_id:
            continue
        other_fields = _scan_doc_fields(scan)

        if len(personal_number) == 10:
            other_number = normalize_personal_number(other_fields.get("personal_number", ""))
            if len(other_number) == 10 and other_number == personal_number:
                matches.append(scan)
                continue

        if name_tokens:
            other_tokens = normalize_name_tokens(f"{other_fields.get('name', '')} {other_fields.get('surname', '')}")
            if other_tokens and other_tokens == name_tokens:
                matches.append(scan)

    return matches
