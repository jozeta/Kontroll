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
from collections import defaultdict
from dataclasses import dataclass, field

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


@dataclass
class PersonIndex:
    """A same-person lookup index built once over a full scan list, so a caller building many
    HistoryRowWidgets in a row (see ui/main_window.py::MainWindow._load_history) doesn't have each
    one independently re-scan the entire list to find its own previous visits (what
    find_previous_scans does, and still does -- unchanged -- for the small-scale callers where
    that's fine, e.g. _refresh_same_person_rows over just the currently-displayed rows).

    A scan can be filed under both a personal-number bucket and a name-token bucket at once (its
    own fields may supply either or both), exactly mirroring find_previous_scans's two independent
    match conditions -- this index is a lookup structure over the same match rule, not a different,
    reduced one (contrast document/statistics.py's own _person_key, which deliberately collapses
    to a single key as an accepted approximation for aggregate counting -- not appropriate to reuse
    here where per-scan display accuracy matters).
    """

    by_personal_number: dict[str, list[ScanRecord]] = field(default_factory=dict)
    by_name_tokens: dict[frozenset[str], list[ScanRecord]] = field(default_factory=dict)


def build_person_index(scans: list[ScanRecord]) -> PersonIndex:
    """Build a PersonIndex over `scans` (see PersonIndex's own docstring), preserving whatever
    order `scans` is already in within each bucket -- callers passing a newest-first list (as
    storage.models.list_scans already returns) get newest-first buckets for free."""
    by_personal_number: dict[str, list[ScanRecord]] = defaultdict(list)
    by_name_tokens: dict[frozenset[str], list[ScanRecord]] = defaultdict(list)

    for scan in scans:
        fields = _scan_doc_fields(scan)
        personal_number = normalize_personal_number(fields.get("personal_number", ""))
        if len(personal_number) == 10:
            by_personal_number[personal_number].append(scan)

        name_tokens = normalize_name_tokens(f"{fields.get('name', '')} {fields.get('surname', '')}")
        if name_tokens:
            by_name_tokens[name_tokens].append(scan)

    return PersonIndex(by_personal_number=dict(by_personal_number), by_name_tokens=dict(by_name_tokens))


def find_previous_scans_indexed(
    doc_fields: dict[str, str], index: PersonIndex, exclude_scan_id: int | None = None
) -> list[ScanRecord]:
    """Indexed equivalent of find_previous_scans -- same pairwise match rule (exact personal-number
    match, or exact name-token-set match), but an O(1)-average bucket lookup instead of an O(N) scan
    over every saved scan. See PersonIndex's own docstring for why this is a safe drop-in rather
    than a behavior change.
    """
    personal_number = normalize_personal_number(doc_fields.get("personal_number", ""))
    name_tokens = normalize_name_tokens(f"{doc_fields.get('name', '')} {doc_fields.get('surname', '')}")

    matches: list[ScanRecord] = []
    seen_ids: set[int] = set()

    if len(personal_number) == 10:
        for scan in index.by_personal_number.get(personal_number, []):
            if scan.id != exclude_scan_id and scan.id not in seen_ids:
                matches.append(scan)
                seen_ids.add(scan.id)

    if name_tokens:
        for scan in index.by_name_tokens.get(name_tokens, []):
            if scan.id != exclude_scan_id and scan.id not in seen_ids:
                matches.append(scan)
                seen_ids.add(scan.id)

    # A scan matched via both buckets, or the two buckets concatenated above, don't necessarily
    # come out newest-first even when the index was built from a newest-first scan list (e.g. a
    # personal-number match found on an older scan than a name-only match) -- re-sort explicitly
    # rather than relying on bucket-concatenation order, since callers (the "5 most recent
    # previous visits" list) depend on newest-first.
    matches.sort(key=lambda scan: scan.created_at, reverse=True)
    return matches
