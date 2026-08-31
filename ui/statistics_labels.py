"""Shared display labels and color assignments for the Statistics tab (ui/settings_window.py) and
its printable report (ui/statistics_export.py), kept in one place so the two never show the same
breakdown with a different label or a different color for the same category.
"""

from ui.charts import STATUS_CRITICAL, STATUS_GOOD, STATUS_MUTED, STATUS_WARNING

# document/doc_classifier.py's DocumentClassification.doc_type values.
DOC_TYPE_LABELS = {
    "passport": "Passport",
    "national_id": "National ID",
    "drivers_licence": "Driving licence",
    "unknown": "Unknown",
}
_DOC_TYPE_CATEGORICAL_ORDER = ["passport", "national_id", "drivers_licence"]  # fixed slot order; "unknown" always gets STATUS_MUTED, never a hue

# document/field_crosscheck.py's determine_validation_status return values -- these are mutually
# exclusive per scan (exactly one per row), which is what makes them a legitimate donut/pie
# candidate; see ui/charts.py's DonutChartWidget docstring.
VALIDATION_STATUS_LABELS = {
    "valid": "Valid",
    "review": "Needs review",
    "invalid": "Invalid",
    "unverifiable": "Unverifiable",
}
VALIDATION_STATUS_COLORS = {
    "valid": STATUS_GOOD,
    "review": STATUS_WARNING,
    "invalid": STATUS_CRITICAL,
    "unverifiable": STATUS_MUTED,
}

# storage.models/ui.main_window's match_result values.
MATCH_RESULT_LABELS = {
    "match": "Match",
    "no_match": "No match",
    "override_match": "Match (operator override)",
    "override_no_match": "No match (operator override)",
}


def doc_type_label(doc_type: str) -> str:
    return DOC_TYPE_LABELS.get(doc_type, doc_type)


def doc_type_color(doc_type: str, categorical_palette: list[str]) -> str:
    """Assign doc_type a fixed categorical slot (same slot regardless of which other doc types are present in this dataset) -- 'unknown' always gets the muted status color instead of a hue, since it isn't a real category, just the absence of a classification."""
    if doc_type in _DOC_TYPE_CATEGORICAL_ORDER:
        return categorical_palette[_DOC_TYPE_CATEGORICAL_ORDER.index(doc_type)]
    return STATUS_MUTED


def validation_status_label(status: str) -> str:
    return VALIDATION_STATUS_LABELS.get(status, status)


def validation_status_color(status: str) -> str:
    return VALIDATION_STATUS_COLORS.get(status, STATUS_MUTED)


def match_result_label(result: str) -> str:
    return MATCH_RESULT_LABELS.get(result, result)


def flags_and_issues_breakdown(stats) -> list[tuple[str, int]]:
    """Build the (label, count) list behind the Statistics tab's "Flags & issues" bar chart: every
    image-quality warning reason plus the person/security flags, each an independent, possibly-
    overlapping condition a scan can have -- not a partition, so this is deliberately a ranked bar
    list rather than a pie (see ui/charts.py's DonutChartWidget docstring for why).
    """
    items: list[tuple[str, int]] = []
    for reason, count in stats.image_quality_reason_counts.items():
        items.append((reason.title(), count))
    items.append(("Expired document", stats.expired_documents_count))
    items.append(("Under 18", stats.minors_count))
    items.append(("Watch-list match", stats.watchlist_matches_count))
    items.append(("Face no-match", stats.match_result_counts.get("no_match", 0)))
    items = [(label, count) for label, count in items if count > 0]
    items.sort(key=lambda item: item[1], reverse=True)
    return items
