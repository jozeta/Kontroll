"""Aggregate statistics over saved scans, for the Settings "Statistics" tab (ui/settings_window.py)
and its printable report (ui/statistics_export.py).

Every figure here is derived only from what the pipeline actually records (see CLAUDE.md's
validation_status/person_flags/match_result semantics). This app has no processing-latency
instrumentation and no liveness/spoof-detection step, so neither is reported here -- inventing
numbers for signals the pipeline doesn't actually produce would be actively misleading in a report
an operator might rely on.
"""

import json
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from document.flag_matcher import normalize_name_tokens, normalize_personal_number
from storage.models import ScanRecord

# Average calendar-month/year lengths, used only to convert an exact avg-per-day figure into an
# approximate avg-per-month/year one -- not a claim about any specific month/year.
_DAYS_PER_MONTH = 30.44
_DAYS_PER_YEAR = 365.25


def _parse_created_at(scan: ScanRecord) -> datetime:
    return datetime.fromisoformat(scan.created_at)


def _doc_fields(scan: ScanRecord) -> dict:
    if not scan.doc_fields_json:
        return {}
    try:
        return json.loads(scan.doc_fields_json)
    except json.JSONDecodeError:
        return {}


def _detail(scan: ScanRecord) -> dict:
    if not scan.validation_detail_json:
        return {}
    try:
        return json.loads(scan.validation_detail_json)
    except json.JSONDecodeError:
        return {}


def _person_key(fields: dict) -> str | None:
    """Group scans by the same identity rule document/person_matcher.py uses pairwise (exact
    personal-number match, else exact name-token-*set* match) -- but reduced to a single hashable
    key instead of an O(n^2) all-pairs comparison, so repeat-visit counting here stays O(n) even
    over a scan history with no retention limit.
    """
    personal_number = normalize_personal_number(fields.get("personal_number", ""))
    if len(personal_number) == 10:
        return f"pn:{personal_number}"
    name_tokens = normalize_name_tokens(f"{fields.get('name', '')} {fields.get('surname', '')}")
    if name_tokens:
        return "name:" + "|".join(sorted(name_tokens))
    return None


@dataclass
class DailyCount:
    """One calendar day's scan count, as returned by compute_daily_volume -- zero-filled for days with no scans, so a caller can plot a continuous series without gaps."""

    day: date
    count: int


@dataclass
class Statistics:
    """Every figure shown on the Statistics tab / printed report, computed once from a scan list by compute_statistics."""

    generated_at: datetime
    total_scans: int
    first_scan_at: datetime | None
    last_scan_at: datetime | None

    scans_last_7_days: int
    scans_prev_7_days: int
    scans_last_30_days: int
    scans_prev_30_days: int

    avg_per_day: float
    avg_per_week: float
    avg_per_month: float
    avg_per_year: float

    validation_status_counts: dict[str, int]
    verification_pass_rate: float | None  # 'valid' scans / scans with any validation_status at all

    match_result_counts: dict[str, int]
    face_match_rate: float | None  # (match + override_match) / scans with a match_result at all
    average_match_score: float | None

    ocr_extracted_count: int
    ocr_extraction_rate: float | None  # scans with >=1 extracted doc field / total_scans

    image_quality_issue_count: int
    image_quality_issue_rate: float | None
    image_quality_reason_counts: dict[str, int]

    minors_count: int
    expired_documents_count: int
    watchlist_matches_count: int
    security_flags_count: int  # watchlist matches + expired documents

    identified_scan_count: int  # scans whose extracted fields were enough to compute a person_key
    unique_person_count: int
    repeat_visit_rate: float | None  # (identified_scan_count - unique_person_count) / identified_scan_count

    doc_type_counts: dict[str, int]
    nationality_counts: dict[str, int]  # ISO alpha-3 -> count, every country seen (caller truncates for display)


def _delta_percent(current: int, previous: int) -> float | None:
    """Percent change from previous to current, or None if there's no previous-period baseline to compare against (0 scans then -- not the same as a 0% change)."""
    if previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def compute_statistics(scans: list[ScanRecord], now: datetime | None = None) -> Statistics:
    """Compute every Statistics figure in one pass over `scans` (as returned by storage.models.list_scans)."""
    now = now or datetime.now()
    total_scans = len(scans)

    validation_status_counts: Counter[str] = Counter()
    match_result_counts: Counter[str] = Counter()
    image_quality_reason_counts: Counter[str] = Counter()
    doc_type_counts: Counter[str] = Counter()
    nationality_counts: Counter[str] = Counter()
    person_counts: Counter[str] = Counter()

    match_scores: list[float] = []
    created_at_list: list[datetime] = []
    ocr_extracted_count = 0
    image_quality_issue_count = 0
    minors_count = 0
    expired_documents_count = 0
    watchlist_matches_count = 0
    identified_scan_count = 0

    for scan in scans:
        created_at = _parse_created_at(scan)
        created_at_list.append(created_at)

        if scan.validation_status:
            validation_status_counts[scan.validation_status] += 1
        if scan.match_result:
            match_result_counts[scan.match_result] += 1
        if scan.match_score is not None:
            match_scores.append(scan.match_score)
        if scan.doc_type:
            doc_type_counts[scan.doc_type] += 1
        if scan.doc_country:
            nationality_counts[scan.doc_country] += 1

        fields = _doc_fields(scan)
        if fields:
            ocr_extracted_count += 1

        detail = _detail(scan)
        image_quality = detail.get("image_quality") or {}
        if image_quality.get("has_warning"):
            image_quality_issue_count += 1
            for reason in (image_quality.get("text") or "").split(" & "):
                if reason:
                    image_quality_reason_counts[reason] += 1

        person_flags = detail.get("person_flags") or {}
        if person_flags.get("is_minor"):
            minors_count += 1
        if person_flags.get("is_expired"):
            expired_documents_count += 1
        if detail.get("flagged_match"):
            watchlist_matches_count += 1

        person_key = _person_key(fields)
        if person_key is not None:
            identified_scan_count += 1
            person_counts[person_key] += 1

    first_scan_at = min(created_at_list) if created_at_list else None
    last_scan_at = max(created_at_list) if created_at_list else None

    scans_last_7_days = sum(1 for c in created_at_list if c >= now - timedelta(days=7))
    scans_prev_7_days = sum(1 for c in created_at_list if now - timedelta(days=14) <= c < now - timedelta(days=7))
    scans_last_30_days = sum(1 for c in created_at_list if c >= now - timedelta(days=30))
    scans_prev_30_days = sum(1 for c in created_at_list if now - timedelta(days=60) <= c < now - timedelta(days=30))

    if first_scan_at is not None:
        days_span = max((now.date() - first_scan_at.date()).days + 1, 1)
    else:
        days_span = 1
    avg_per_day = total_scans / days_span
    avg_per_week = avg_per_day * 7
    avg_per_month = avg_per_day * _DAYS_PER_MONTH
    avg_per_year = avg_per_day * _DAYS_PER_YEAR

    total_with_status = sum(validation_status_counts.values())
    verification_pass_rate = (validation_status_counts.get("valid", 0) / total_with_status) if total_with_status else None

    total_with_match_result = sum(match_result_counts.values())
    matched_count = match_result_counts.get("match", 0) + match_result_counts.get("override_match", 0)
    face_match_rate = (matched_count / total_with_match_result) if total_with_match_result else None
    average_match_score = (sum(match_scores) / len(match_scores)) if match_scores else None

    ocr_extraction_rate = (ocr_extracted_count / total_scans) if total_scans else None
    image_quality_issue_rate = (image_quality_issue_count / total_scans) if total_scans else None

    unique_person_count = len(person_counts)
    repeat_visit_rate = (
        (identified_scan_count - unique_person_count) / identified_scan_count if identified_scan_count else None
    )

    return Statistics(
        generated_at=now,
        total_scans=total_scans,
        first_scan_at=first_scan_at,
        last_scan_at=last_scan_at,
        scans_last_7_days=scans_last_7_days,
        scans_prev_7_days=scans_prev_7_days,
        scans_last_30_days=scans_last_30_days,
        scans_prev_30_days=scans_prev_30_days,
        avg_per_day=avg_per_day,
        avg_per_week=avg_per_week,
        avg_per_month=avg_per_month,
        avg_per_year=avg_per_year,
        validation_status_counts=dict(validation_status_counts),
        verification_pass_rate=verification_pass_rate,
        match_result_counts=dict(match_result_counts),
        face_match_rate=face_match_rate,
        average_match_score=average_match_score,
        ocr_extracted_count=ocr_extracted_count,
        ocr_extraction_rate=ocr_extraction_rate,
        image_quality_issue_count=image_quality_issue_count,
        image_quality_issue_rate=image_quality_issue_rate,
        image_quality_reason_counts=dict(image_quality_reason_counts),
        minors_count=minors_count,
        expired_documents_count=expired_documents_count,
        watchlist_matches_count=watchlist_matches_count,
        security_flags_count=watchlist_matches_count + expired_documents_count,
        identified_scan_count=identified_scan_count,
        unique_person_count=unique_person_count,
        repeat_visit_rate=repeat_visit_rate,
        doc_type_counts=dict(doc_type_counts),
        nationality_counts=dict(nationality_counts),
    )


def scans_last_30_days_delta_percent(stats: Statistics) -> float | None:
    return _delta_percent(stats.scans_last_30_days, stats.scans_prev_30_days)


def _pp_delta(current: float | None, previous: float | None) -> float | None:
    """Percentage-point difference between two 0-1 rates (e.g. 0.62 -> 0.58 is -4.0, not -6.5%) -- the natural unit for comparing a rate to itself period-over-period, as opposed to _delta_percent's percent-of-a-percent framing for comparing two counts. None if either side has no data to compute a rate from at all (not the same as a rate of exactly 0)."""
    if current is None or previous is None:
        return None
    return (current - previous) * 100.0


@dataclass
class KpiTrends:
    """Period-over-period comparison for the Statistics tab's KPI tiles: each field is the change
    from the 30 days before last to the last 30 days, computed once by compute_kpi_trends so ui/
    settings_window.py's tab and ui/statistics_export.py's report always show the same trend.
    None means one of the two 30-day windows had no data to compute that figure from at all --
    not the same as an unchanged (0) value.
    """

    verification_pass_rate_delta_pp: float | None
    face_match_rate_delta_pp: float | None
    ocr_extraction_rate_delta_pp: float | None
    security_flags_delta_percent: float | None
    repeat_visit_rate_delta_pp: float | None


def compute_kpi_trends(scans: list[ScanRecord], now: datetime | None = None) -> KpiTrends:
    """Compare the last 30 days against the 30 days before that for every rate/count KPI tile
    that isn't already covered by Statistics' own scans_last_30_days/scans_prev_30_days fields
    (see scans_last_30_days_delta_percent for that one). Reuses compute_statistics itself on each
    30-day slice rather than re-deriving each rate's formula a second time here.
    """
    now = now or datetime.now()
    last_30_days = [scan for scan in scans if _parse_created_at(scan) >= now - timedelta(days=30)]
    previous_30_days = [scan for scan in scans if now - timedelta(days=60) <= _parse_created_at(scan) < now - timedelta(days=30)]
    current = compute_statistics(last_30_days, now)
    previous = compute_statistics(previous_30_days, now)

    return KpiTrends(
        verification_pass_rate_delta_pp=_pp_delta(current.verification_pass_rate, previous.verification_pass_rate),
        face_match_rate_delta_pp=_pp_delta(current.face_match_rate, previous.face_match_rate),
        ocr_extraction_rate_delta_pp=_pp_delta(current.ocr_extraction_rate, previous.ocr_extraction_rate),
        security_flags_delta_percent=_delta_percent(current.security_flags_count, previous.security_flags_count),
        repeat_visit_rate_delta_pp=_pp_delta(current.repeat_visit_rate, previous.repeat_visit_rate),
    )


def compute_daily_volume(scans: list[ScanRecord], days: int, now: datetime | None = None) -> list[DailyCount]:
    """Return one DailyCount per calendar day in the trailing `days`-day window ending today (inclusive), zero-filled for days with no scans -- the data behind the Statistics tab's time-series bar chart."""
    now = now or datetime.now()
    end_day = now.date()
    start_day = end_day - timedelta(days=days - 1)

    counts: Counter[date] = Counter()
    for scan in scans:
        day = _parse_created_at(scan).date()
        if start_day <= day <= end_day:
            counts[day] += 1

    return [DailyCount(day=start_day + timedelta(days=offset), count=counts[start_day + timedelta(days=offset)]) for offset in range(days)]


def top_n_with_other(counts: dict[str, int], n: int, other_label: str = "Other") -> list[tuple[str, int]]:
    """Return the n largest (label, count) entries from `counts`, sorted descending, with every remaining entry folded into one trailing `other_label` bucket (omitted if empty) -- keeps a breakdown chart to a fixed, readable number of bars regardless of how many distinct values actually occur."""
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    top = ranked[:n]
    remainder = sum(count for _, count in ranked[n:])
    if remainder:
        top.append((other_label, remainder))
    return top
