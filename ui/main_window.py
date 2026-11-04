"""Single-screen PyQt6 window: two live camera previews, one capture button, and a scan history list."""

import base64
import html
import json
import logging
import shutil
import sqlite3
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PyQt6.QtCore import QByteArray, QDate, QObject, QPoint, QPropertyAnimation, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QCursor,
    QIcon,
    QKeySequence,
    QMouseEvent,
    QPixmap,
    QResizeEvent,
    QShortcut,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCalendarWidget,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QStyle,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import CameraManager
from config.app_paths import get_documents_dir, get_faces_dir, get_flagged_images_dir
from config.settings import (
    get_barcode_overlap_threshold,
    get_capture_shortcut,
    get_disk_space_warning_mb,
    get_face_match_threshold,
    get_history_page_size,
    get_setting_or_none,
    set_setting,
)
from document.country_flags import alpha2_for_country_code
from document.dates import calculate_age, parse_mrz_style_date
from document.doc_classifier import DocumentClassification
from document.field_crosscheck import (
    CrosscheckResult,
    PersonFlagsResult,
    check_person_flags,
    determine_validation_status,
)
from document.field_extraction import extract_document_fields
from document.flag_matcher import FlagMatch, find_matching_flagged_person, match_basis_label, matched_on_display
from document.image_quality import (
    ImageQualityResult,
    assess_image_quality,
    compute_blur_score,
    is_frame_too_blurry,
    is_resolution_too_low_for_ocr,
)
from face.detector import detect_faces
from face.embedder import get_best_face_embedding
from face.matcher import MatchResult, match_faces
from document.person_matcher import PersonIndex, build_person_index, find_previous_scans, find_previous_scans_indexed
from storage.models import (
    FlaggedPerson,
    ScanRecord,
    add_scan_comment,
    calculate_scan_storage_bytes,
    clear_flagged_match_for_person,
    count_scans,
    delete_flagged_person,
    delete_scan,
    delete_scan_comment,
    get_flagged_person_by_id,
    get_scan_by_id,
    insert_flagged_person,
    insert_scan,
    list_flagged_persons,
    list_scan_comments,
    list_scans,
    thumbnail_cache_path,
    update_match_result,
    update_scan_analysis,
    update_scan_doc_fields,
)
from ui.app_icon import render_app_logo
from ui.flag_icons import render_flag_icon
from ui.flag_retroactive import apply_flagged_person_edit, run_retroactive_flag_check
from ui.frame_display import (
    THUMBNAIL_SIZE,
    bgr_frame_to_pixmap as _bgr_frame_to_pixmap,
    shrink_to_fit as _shrink_to_fit,
    write_thumbnail_cache,
)
from ui.image_viewer import ClickableThumbnailLabel
from ui.scan_export import build_batch_report_html, build_scan_report_html
from ui.settings_window import SettingsWindow
from ui.window_geometry import restore_geometry, save_geometry

# Each independently resolved (and auto-created if missing) via config.app_paths rather than a
# path relative to this file's own location -- see that module for why (packaged-EXE safety) and
# how each is independently operator-configurable.
FACE_IMAGE_DIR = get_faces_dir()
DOC_IMAGE_DIR = get_documents_dir()
FLAGGED_IMAGE_DIR = get_flagged_images_dir()

logger = logging.getLogger(__name__)

_PREVIEW_INTERVAL_MS = 33  # ~30fps live preview polling
_PREVIEW_SIZE = (320, 240)
_THUMBNAIL_SIZE = THUMBNAIL_SIZE  # re-exported under this module's existing private name -- see ui/frame_display.py

# How long the camera-flash-style white overlay takes to fade back out after a capture -- see
# _FlashCapableLabel.flash_white/MainWindow._flash_capture_previews. Short enough to read as a
# quick "dip", not a lingering wash-out that would still be fading while the operator is already
# looking at the next visitor.
_CAPTURE_FLASH_FADE_MS = 250

# How many preview ticks between live blur checks on the document camera (see
# MainWindow._update_doc_quality_warning) -- roughly 3 times/sec at _PREVIEW_INTERVAL_MS, cheap
# enough to stay responsive without running a Laplacian convolution over the whole frame on every
# single ~30fps tick.
_QUALITY_CHECK_INTERVAL_TICKS = 10

# Consecutive failed preview-tick reads (at _PREVIEW_INTERVAL_MS apart) before a camera's frozen
# last-good preview frame is replaced with an explicit "signal lost" message -- see
# MainWindow._maybe_show_signal_lost. ~1 second's worth, so a single dropped tick (which happens
# occasionally and harmlessly) doesn't flicker the warning on and off.
_SIGNAL_LOST_TICK_THRESHOLD = round(1000 / _PREVIEW_INTERVAL_MS)

# Colors for the small always-visible camera-status badge next to each preview headline -- see
# MainWindow._set_camera_status_badge.
_CAMERA_STATUS_LIVE_COLOR = "#2E7D32"
_CAMERA_STATUS_RECONNECTING_COLOR = "#F9A825"
_CAMERA_STATUS_LOST_COLOR = "#C62828"

_OVERRIDE_CHOICES = ["Automatic", "Override: Match", "Override: No Match"]

# Opacity applied to the date-filter dropdown while "Any date" is checked -- dims it to signal
# it's inactive without actually disabling it, since a disabled widget can't receive the click
# that opens its calendar popup (see MainWindow._on_date_calendar_shown).
_DATE_FILTER_DIMMED_OPACITY = 0.4

# How many rows the history list is sized to show, without scrolling, the first time the app is
# opened -- see MainWindow._restore_window_state, which pads the window out to this height on a
# fresh install (no saved geometry yet). Not an enforced floor by itself -- see
# _MIN_VISIBLE_HISTORY_ROWS below for the actual resize floor -- and the list's own stretch factor
# lets it grow past this whenever the window is resized bigger.
_DEFAULT_VISIBLE_HISTORY_ROWS = 5

# Absolute floor (via QScrollArea.setMinimumSize) for the scan list's height once the operator
# drags the splitter handle or shrinks the window -- kept low so the whole window still fits on a
# small/low-resolution screen even with the top pane (previews + capture button, pinned to its own
# fixed height by resizeEvent) taking up its share.
_MIN_VISIBLE_HISTORY_ROWS = 2

# How close (in pixels) to the bottom of the history list's scrollbar the operator has to scroll
# before the next chunk of scans loads -- see MainWindow._on_history_scrolled/_load_more_rows.
# Loading a little before the true bottom (rather than only once it's fully reached) means the
# next chunk is usually already in place by the time the operator actually gets there, instead of
# a visible pause right as they hit the end of the currently-loaded rows.
_HISTORY_SCROLL_LOAD_MORE_THRESHOLD_PX = 300

# settings-table keys the window's size/position and the top pane's height are persisted under
# (see MainWindow._restore_window_state / _save_window_state). The geometry is base64 text
# wrapping the raw QByteArray that Qt's own saveGeometry() produces; the pane height is stored as
# a plain pixel integer instead of via QSplitter's saveState()/restoreState() -- see
# _restore_window_state's docstring for why.
_WINDOW_GEOMETRY_SETTING_KEY = "window_geometry"
_TOP_PANE_HEIGHT_SETTING_KEY = "top_pane_height"


def _format_timestamp(created_at: str) -> str:
    """Format a scan's stored ISO-format created_at timestamp as 'YYYY-MM-DD HH:MM:SS' for display."""
    return datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M:%S")


def _load_thumbnail(image_path: str | None) -> QPixmap | None:
    """Load a saved scan image's cached thumbnail from disk (generating it first if this is the first time) and scale it to thumbnail size for the history list.

    storage.models.thumbnail_cache_path names a small "<name>.thumb.jpg" file kept alongside the
    full-resolution image -- decoding that tiny file is far cheaper than decoding the full-
    resolution original every time, which matters here since this runs on every single history row
    built, including, on a large scan history, the potentially hundreds of rows
    MainWindow._jump_to_scan's catch-up load builds in one go. MainWindow._finish_save_capture and
    ImageEditDialog._on_apply_clicked both generate this cache eagerly (from the same in-memory
    frame they're already writing to disk, no extra read needed) whenever an image is produced or
    edited, so the fallback path below -- decoding the full image and writing the cache from that
    -- is normally only ever hit once per scan, lazily backfilling scans captured before this cache
    existed rather than needing a migration pass over old rows. That fallback decode itself uses
    cv2.IMREAD_REDUCED_COLOR_2 (a real decode-time saving for a JPEG, built into libjpeg's own
    DCT-based scaled decoding, not just a resize done after a full decode) -- halving each
    dimension still comfortably exceeds _THUMBNAIL_SIZE for every resolution this app captures at
    (640x480 and up -- see capture/camera_manager.py), so this never undershoots the target size.
    """
    if not image_path or not Path(image_path).exists():
        return None

    cache_path = thumbnail_cache_path(image_path)
    if cache_path.exists():
        frame = cv2.imread(str(cache_path))
    else:
        frame = cv2.imread(image_path, cv2.IMREAD_REDUCED_COLOR_2)
        if frame is not None:
            write_thumbnail_cache(image_path, frame, _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1])
    if frame is None:
        return None

    frame = _shrink_to_fit(frame, _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1])
    pixmap = _bgr_frame_to_pixmap(frame)
    return pixmap.scaled(
        _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


def _delete_image_and_cache(image_path: str | None) -> int:
    """Delete a saved scan image file and its cached thumbnail (see storage.models.thumbnail_cache_path), returning the total bytes freed.

    Shared by HistoryRowWidget._on_delete_clicked and, in effect, mirrored by
    storage/retention.py::_delete_image_file_if_present for the automatic/manual purge jobs -- the
    only two places in the app that ever delete a saved scan's image files, so a thumbnail cache
    file can never be orphaned behind an already-deleted full image.
    """
    freed_bytes = 0
    if not image_path:
        return freed_bytes
    for path in (Path(image_path), thumbnail_cache_path(image_path)):
        if path.exists():
            # Sized before unlinking -- lets the caller adjust its cached storage-usage total by
            # exactly this amount instead of re-walking the whole image directory tree (see
            # MainWindow._add_to_cached_storage_bytes).
            freed_bytes += path.stat().st_size
            path.unlink()
    return freed_bytes


def _match_icon_text(match_result: str | None) -> str:
    """Return a short, visually distinct glyph+label summarizing the face-match result for the history list."""
    if match_result in ("match", "override_match"):
        return "✅ MATCH"
    if match_result in ("no_match", "override_no_match"):
        return "❌ NO MATCH"
    return "➖ N/A"


_TOOLTIP_ACCENT_COLOR = "#90CAF9"  # a light, theme-agnostic blue for a tooltip's own title -- Qt tooltips always render on a fixed (not theme-following) background, so this doesn't need to react to light/dark mode the way in-window widgets do.


_TOOLTIP_FONT_SIZE = "8pt"  # smaller than Qt's own default tooltip font -- these tooltips pack in several data rows, not a single short hint, so keeping them compact matters more than for an ordinary one-line tooltip.


def _tooltip_html(title: str, rows: list[str], footnote: str | None = None) -> str:
    """Build a small, consistently-styled rich-text tooltip: a colored title, a divider, one row per line, and an optional italic footnote below a blank line.

    Shared by every hover tooltip in the history list (see _match_confidence_tooltip,
    _flagged_match_tooltip) so they read as one consistent design rather than each rolling its own
    ad hoc formatting -- Qt tooltips render basic HTML, so a bold/colored title and a rule read far
    more like a purpose-built info panel than a plain wrapped line of text.
    """
    html_parts = [
        f"<div style='font-size:{_TOOLTIP_FONT_SIZE};'>",
        f"<b style='color:{_TOOLTIP_ACCENT_COLOR};'>{title}</b><hr>",
        "<br>".join(rows),
    ]
    if footnote:
        html_parts.append(f"<br><br><i>{footnote}</i>")
    html_parts.append("</div>")
    return "".join(html_parts)


def _match_confidence_tooltip(conn: sqlite3.Connection, match_score: float | None, match_result: str | None) -> str:
    """Build the hover-tooltip for a row's match_icon_label: the raw similarity score, whether the
    result was automatic or an operator override, and the currently configured face-match
    threshold for reference.

    This detail used to sit as a permanent inline label next to the icon, crowding every single
    row with a number that's only occasionally worth checking -- moved to a hover tooltip instead,
    shown only when an operator actually wants it.

    The threshold shown is *today's* setting (config.settings.get_face_match_threshold), not
    necessarily whatever was active when this scan was actually captured -- recalibration
    (ui/calibration_window.py) can change it afterward, so this is deliberately labeled as current/
    reference-only rather than presented as if it explains this specific row's own stored
    match_result.
    """
    if match_score is None:
        return _tooltip_html(
            "Face Match", ["No similarity score available -- a face wasn't detected in one of the two images."]
        )

    rows = [f"Similarity score: <b>{match_score:.3f}</b>"]

    if match_result in ("override_match", "override_no_match"):
        outcome = "Match" if match_result == "override_match" else "No match"
        rows.append(f"Result: {outcome} (manually overridden)")
    elif match_result in ("match", "no_match"):
        outcome = "Match" if match_result == "match" else "No match"
        rows.append(f"Result: {outcome} (automatic)")

    threshold = get_face_match_threshold(conn)
    margin = match_score - threshold
    rows.append(f"Current threshold: {threshold:.3f} (score is {margin:+.3f} relative to it)")

    return _tooltip_html("Face Match", rows, footnote="Threshold may have been changed since this scan was captured.")


def _flagged_match_tooltip(conn: sqlite3.Connection, flagged_match: dict) -> str:
    """Build the hover-tooltip for a flagged scan's red badge: the full flagged-persons watch-list
    entry this scan matched (name, personal number, notes, when it was added) plus which of the
    scan's own fields is what actually matched -- detail that used to spill into the badge's own
    visible text, crowding it on every flagged row (see HistoryRowWidget._build_layout).

    Looks up the *live* storage.models.FlaggedPerson row by id rather than only trusting the
    smaller snapshot already stored in the scan's own flagged_match blob (name/matched_on/notes) --
    the watch-list entry may have been edited since this scan was flagged (see
    ui/flag_retroactive.py::apply_flagged_person_edit), and the tooltip should reflect its current
    details, not whatever was true at match time. Falls back to the stored snapshot if the entry
    has since been deleted -- shouldn't normally happen, since deleting an entry clears every
    scan's flag against it (storage.models.clear_flagged_match_for_person), but a defensive
    fallback costs nothing here.
    """
    flagged_person_id = flagged_match.get("flagged_person_id")
    flagged_person = get_flagged_person_by_id(conn, flagged_person_id) if flagged_person_id is not None else None

    name = flagged_person.name if flagged_person is not None else flagged_match.get("flagged_person_name")
    personal_number = flagged_person.personal_number if flagged_person is not None else None
    notes = flagged_person.notes if flagged_person is not None else flagged_match.get("notes")

    rows = []
    if name:
        rows.append(f"Name: {html.escape(name)}")
    if personal_number:
        rows.append(f"Personal number: {html.escape(personal_number)}")

    matched_on = flagged_match.get("matched_on")
    if matched_on:
        rows.append(f"Matched via: {matched_on_display(matched_on).capitalize()}")
    if flagged_person is not None:
        rows.append(f"Watch-list match fields: {match_basis_label(flagged_person)}")
    if notes:
        rows.append(f"Notes: {html.escape(notes)}")
    if flagged_person is not None and flagged_person.created_at:
        rows.append(f"Added to watch list: {_format_timestamp(flagged_person.created_at)}")
    if not rows:
        rows.append("No further details recorded.")

    return _tooltip_html("Flagged Person Match", rows, footnote="Manage this entry from Settings → Flagged Users.")


@dataclass
class DisplayFields:
    """Name/birth-date/age/personal-number fields derived from a scan's stored MRZ data, for display in the history list."""

    first_name: str
    last_name: str
    birth_date_display: str
    age_display: str
    personal_number_display: str


def _derive_display_fields(doc_fields_json: str | None) -> DisplayFields:
    """Extract first name, last name, birth date, computed age, and personal number from a scan's stored structured fields, if any.

    doc_fields_json holds whichever structured fields were available for this scan: MRZ fields for
    passports/ID cards, or the OCR-heuristic-derived fields from document/dl_field_parser.py
    (driving licences) or document/national_id_parser.py (Swedish national ID cards) -- see
    ui/main_window.py's capture handler for which one wins. Documents with neither show '-'
    placeholders here rather than a guessed value, consistent with how validation_status handles
    the same gap. personal_number is only ever populated by the latter two (MRZ fields carry no
    equivalent key today), so it's read independently of the birth_date early-return below.
    """
    if not doc_fields_json:
        return DisplayFields("-", "-", "-", "-", "-")

    fields = json.loads(doc_fields_json)
    first_name = fields.get("name", "").strip() or "-"
    last_name = fields.get("surname", "").strip() or "-"
    personal_number_display = fields.get("personal_number", "").strip() or "-"

    birth_date = parse_mrz_style_date(fields.get("birth_date", ""), assume_past=True)
    if birth_date is None:
        return DisplayFields(first_name, last_name, "-", "-", personal_number_display)

    return DisplayFields(
        first_name, last_name, birth_date.isoformat(), str(calculate_age(birth_date)), personal_number_display
    )


# Never persisted or displayed -- used only to measure a representative row's natural size for
# sizing the history list (see MainWindow._measure_history_list_size). The name/status text
# lengths are chosen to be reasonably long, similar to real MRZ data, so the measured width isn't
# optimistically narrow.
_SAMPLE_SCAN_FOR_SIZING = ScanRecord(
    id=0,
    created_at=datetime.now().isoformat(),
    operator=None,
    face_image_path=None,
    doc_image_path=None,
    face_embedding=None,
    match_score=0.9876,
    match_result="match",
    doc_type="passport",
    doc_country="SWE",
    doc_fields_json=json.dumps(
        {"name": "MARGARETHA", "surname": "LINDQVIST-JOHANSSON", "birth_date": "850615", "personal_number": "850615-1234"}
    ),
    validation_status="valid",
    validation_detail_json=json.dumps({"notes": [], "field_comparisons": []}),
    operator_comment=None,
)


def _final_match_result_label(automatic_is_match: bool | None, override_choice: str) -> str:
    """Compute the match_result string to persist, combining the automated result with any operator override choice."""
    if override_choice == "Override: Match":
        return "override_match"
    if override_choice == "Override: No Match":
        return "override_no_match"
    if automatic_is_match:
        return "match"
    return "no_match"


@dataclass
class CaptureAnalysis:
    """Every derived signal the document/face pipeline computes from one face+document frame pair.

    Produced by _analyze_capture, and shared by a fresh capture
    (MainWindow._run_capture_and_process) and a saved scan's "Re-analyze" action
    (MainWindow._on_reanalyze_scan_requested) -- extracted into its own function specifically so
    those two can never drift apart from each other as the pipeline evolves.
    """

    live_embedding: np.ndarray | None
    match_result: MatchResult | None
    doc_classification: DocumentClassification
    doc_fields: dict[str, str]
    doc_fields_json: str
    crosscheck_result: CrosscheckResult
    person_flags: PersonFlagsResult
    flag_match: FlagMatch | None
    doc_image_quality: ImageQualityResult
    validation_detail_json: str


def _analyze_capture(
    face_analysis_app: FaceAnalysis,
    face_frame: np.ndarray,
    doc_frame: np.ndarray,
    face_match_threshold: float,
    barcode_overlap_threshold: float,
    flagged_persons: list[FlaggedPerson],
) -> CaptureAnalysis:
    """Run the full face-match + document OCR/MRZ/barcode/classification pipeline over one already-captured face+document frame pair and return every derived signal, without touching the database or saved image files itself.

    Takes plain frames rather than a ScanRecord/camera handle so the exact same logic works
    whether the frames just came off a live camera (a fresh capture) or were loaded back from a
    previously saved scan's own image files (re-analysis) -- see the two call sites above.

    Takes the face-match/barcode-overlap thresholds and the flagged-persons list as plain
    already-fetched values rather than a live DB connection, so this function does no I/O of its
    own and is safe to run on a background thread (see MainWindow._start_capture_analysis) --
    sqlite3 connections aren't safe to share across threads, but these three are cheap
    single-table reads the caller does up front on the main thread instead.
    """
    # Stage timings logged at the end of this function (see the final logger.info call below) --
    # this pipeline runs on a background thread already (see _CaptureAnalysisWorker), so it never
    # blocks the UI, but a slow stage on weaker production hardware is otherwise invisible; this is
    # what turns "capture feels slow" into a concrete per-stage millisecond breakdown in app.log.
    _pipeline_started_at = time.monotonic()

    # Crop-to-card, OCR, MRZ/barcode location+validation, document classification, and structured
    # field extraction -- see document/field_extraction.py for the detailed per-step rationale
    # (extracted there, out of this function's original home, so ui/image_viewer.py::
    # ImageEditDialog's "what would this image extract to" preview can run the exact same logic
    # rather than risk drifting out of sync with it).
    document_fields_result = extract_document_fields(doc_frame)
    ocr_words = document_fields_result.ocr_words
    ocr_text = document_fields_result.ocr_text
    mrz_result = document_fields_result.mrz_result
    barcode_texts = document_fields_result.barcode_texts
    doc_classification = document_fields_result.doc_classification
    doc_fields = document_fields_result.doc_fields
    _document_extraction_elapsed = time.monotonic() - _pipeline_started_at

    # ocr_words' per-word confidence feeds assess_image_quality's hard-to-read-text check below;
    # Blur/glare/resolution are still measured against the original doc_frame (a diagnostic about
    # the raw camera capture itself, e.g. "camera silently defaulted to a low resolution mode") --
    # only the OCR-confidence-based legibility check benefits from using the cropped, higher-
    # effective-density frame the field parsers actually run on.
    doc_image_quality = assess_image_quality(doc_frame, ocr_words)
    _quality_elapsed = time.monotonic() - _pipeline_started_at - _document_extraction_elapsed
    if doc_image_quality.has_warning:
        logger.info(
            "Document capture flagged low quality (%s): blur_score=%.1f glare_ratio=%.2f "
            "resolution=%dx%d mean_ocr_confidence=%s",
            doc_image_quality.warning_text,
            doc_image_quality.blur_score,
            doc_image_quality.glare_ratio,
            doc_image_quality.width,
            doc_image_quality.height,
            doc_image_quality.mean_ocr_confidence,
        )

    live_embedding = get_best_face_embedding(detect_faces(face_analysis_app, face_frame))
    doc_embedding = get_best_face_embedding(detect_faces(face_analysis_app, doc_frame))

    match_result = None
    if live_embedding is not None and doc_embedding is not None:
        match_result = match_faces(live_embedding, doc_embedding, face_match_threshold)
    _face_elapsed = time.monotonic() - _pipeline_started_at - _document_extraction_elapsed - _quality_elapsed

    crosscheck_result = determine_validation_status(mrz_result, barcode_texts, ocr_text, barcode_overlap_threshold)
    doc_fields_json = json.dumps(doc_fields)

    person_flags = check_person_flags(doc_fields)
    if person_flags.is_minor or person_flags.is_expired:
        logger.info(
            "Scan flagged: is_minor=%s (age=%s) is_expired=%s (expiry=%s)",
            person_flags.is_minor,
            person_flags.age_years,
            person_flags.is_expired,
            person_flags.expiry_date_display,
        )

    flag_match = find_matching_flagged_person(doc_fields, flagged_persons)
    if flag_match is not None:
        logger.warning(
            "Scan matches flagged person id=%d (matched on %s)", flag_match.flagged_person.id, flag_match.matched_on
        )

    validation_detail_json = json.dumps(
        {
            "notes": crosscheck_result.notes + person_flags.notes,
            "field_comparisons": [asdict(comparison) for comparison in crosscheck_result.field_comparisons],
            "classified_from": doc_classification.classified_from,
            "image_quality": {"has_warning": doc_image_quality.has_warning, "text": doc_image_quality.warning_text},
            "person_flags": asdict(person_flags),
            "flagged_match": (
                {
                    "flagged_person_id": flag_match.flagged_person.id,
                    "flagged_person_name": flag_match.flagged_person.name,
                    "matched_on": flag_match.matched_on,
                    "notes": flag_match.flagged_person.notes,
                }
                if flag_match is not None
                else None
            ),
        }
    )

    _total_elapsed = time.monotonic() - _pipeline_started_at
    logger.info(
        "Capture analysis pipeline took %.2fs total (document-extraction=%.2fs quality=%.2fs face=%.2fs "
        "crosscheck/flags=%.2fs)",
        _total_elapsed,
        _document_extraction_elapsed,
        _quality_elapsed,
        _face_elapsed,
        _total_elapsed - _document_extraction_elapsed - _quality_elapsed - _face_elapsed,
    )

    return CaptureAnalysis(
        live_embedding=live_embedding,
        match_result=match_result,
        doc_classification=doc_classification,
        doc_fields=doc_fields,
        doc_fields_json=doc_fields_json,
        crosscheck_result=crosscheck_result,
        person_flags=person_flags,
        flag_match=flag_match,
        doc_image_quality=doc_image_quality,
        validation_detail_json=validation_detail_json,
    )


class _CaptureAnalysisWorker(QObject):
    """Runs _analyze_capture on a background thread -- see MainWindow._start_capture_analysis, which owns the QThread this is moved to.

    A plain QObject moved to a QThread via moveToThread, rather than a QThread subclass -- the
    officially recommended PyQt pattern, since it keeps the worker's own state off the thread
    object itself and makes it trivial to reuse the same worker shape for both call sites that
    need this (a fresh capture and a saved scan's "Re-analyze" action).
    """

    finished = pyqtSignal(object)  # CaptureAnalysis
    failed = pyqtSignal(str)  # traceback text

    def __init__(
        self,
        face_analysis_app: FaceAnalysis,
        face_frame: np.ndarray,
        doc_frame: np.ndarray,
        face_match_threshold: float,
        barcode_overlap_threshold: float,
        flagged_persons: list[FlaggedPerson],
    ) -> None:
        super().__init__()
        self._face_analysis_app = face_analysis_app
        self._face_frame = face_frame
        self._doc_frame = doc_frame
        self._face_match_threshold = face_match_threshold
        self._barcode_overlap_threshold = barcode_overlap_threshold
        self._flagged_persons = flagged_persons

    def run(self) -> None:
        try:
            analysis = _analyze_capture(
                self._face_analysis_app,
                self._face_frame,
                self._doc_frame,
                self._face_match_threshold,
                self._barcode_overlap_threshold,
                self._flagged_persons,
            )
        except Exception:
            # Caught and forwarded as text rather than left to propagate: an exception raised on a
            # QThread doesn't reach the main thread's try/except at all -- Qt just logs it to
            # stderr and the signal that was supposed to carry the result never fires, which is
            # exactly the "click did nothing, no dialog, no history row" failure mode
            # _process_capture's own docstring already describes avoiding for the synchronous path.
            self.failed.emit(traceback.format_exc())
            return
        self.finished.emit(analysis)


_FLAG_PERSON_DIALOG_GEOMETRY_KEY = "flag_person_dialog_geometry"
_EDIT_FIELDS_DIALOG_GEOMETRY_KEY = "edit_fields_dialog_geometry"


class _FlagPersonDialog(QDialog):
    """Prompt to add a scan's person to the operator-maintained flagged-persons watch list, prefilled from the scan's own extracted fields.

    Name, personal number, and notes each have their own checkbox to their left; the field is only
    enabled (and only contributes data on submit -- see values()) once its checkbox is checked, so
    the operator explicitly opts each piece of data into the watch-list entry rather than every
    prefilled/typed field being saved by default. Name and personal number are pre-filled but
    always editable once checked -- the OCR-derived name/personnummer can be wrong or missing (see
    document/dl_field_parser.py's best-effort caveat), and the operator may want to flag someone by
    name alone even without a reliable personnummer read, or vice versa.
    """

    def __init__(
        self,
        suggested_name: str,
        suggested_personal_number: str,
        has_face_image: bool,
        has_doc_image: bool,
        conn: sqlite3.Connection,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add to Flagged Users")
        self._has_face_image = has_face_image
        self._has_doc_image = has_doc_image
        self.finished.connect(lambda _result: save_geometry(conn, self, _FLAG_PERSON_DIALOG_GEOMETRY_KEY))

        form = QFormLayout(self)

        self.name_checkbox, self.name_edit = self._add_checkable_row(form, "Name:", suggested_name)
        self.personal_number_checkbox, self.personal_number_edit = self._add_checkable_row(
            form, "Personal number:", suggested_personal_number
        )

        self.image_combo = QComboBox()
        if has_face_image:
            self.image_combo.addItem("Use this scan's face capture", "face")
        if has_doc_image:
            self.image_combo.addItem("Use this scan's document capture", "doc")
        self.image_combo.addItem("No reference image", None)
        # Defaults to the face capture (falling back to the document capture, then "No reference
        # image" only if neither exists) rather than defaulting to "No reference image" itself --
        # the whole point of flagging from a scan is normally to carry its photo forward, and a
        # silent "No reference image" default meant an operator who didn't think to touch this
        # dropdown ended up with a photo-less watch-list entry every time.
        self.image_combo.setCurrentIndex(0)
        form.addRow("Reference image:", self.image_combo)

        self.notes_checkbox, self.notes_edit = self._add_checkable_row(form, "Notes:", "")
        self.notes_edit.setPlaceholderText("Reason for flagging")

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        restore_geometry(conn, self, _FLAG_PERSON_DIALOG_GEOMETRY_KEY)

    def _add_checkable_row(self, form: QFormLayout, label_text: str, initial_text: str) -> tuple[QCheckBox, QLineEdit]:
        """Add a form row of [checkbox][line edit], the edit starting disabled and only enabled while its checkbox is checked. Checking a row with prefilled text checks the box by default."""
        checkbox = QCheckBox()
        line_edit = QLineEdit(initial_text)
        line_edit.setEnabled(False)
        checkbox.toggled.connect(line_edit.setEnabled)
        checkbox.setChecked(bool(initial_text))

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(checkbox)
        row_layout.addWidget(line_edit, stretch=1)
        form.addRow(label_text, row_widget)
        return checkbox, line_edit

    def values(self) -> tuple[str, str, str | None, str]:
        """Return (name, personal_number, image_source, notes) once the dialog has been accepted -- each field is empty unless its checkbox was checked, and image_source is "face"/"doc"/None."""
        return (
            self.name_edit.text().strip() if self.name_checkbox.isChecked() else "",
            self.personal_number_edit.text().strip() if self.personal_number_checkbox.isChecked() else "",
            self.image_combo.currentData(),
            self.notes_edit.text().strip() if self.notes_checkbox.isChecked() else "",
        )


class _EditFieldsDialog(QDialog):
    """Prompt to manually correct a scan's extracted given name/surname/personal number -- for when OCR/MRZ/barcode extraction got a field wrong or missed it, without needing a full re-capture.

    Prefilled from the scan's current doc_fields_json; any of the three fields left unchanged is
    saved back unchanged too, so leaving one blank on purpose (there was never a value) doesn't
    behave differently from leaving it blank because the operator didn't touch it -- both just
    write through whatever's currently in the box.
    """

    def __init__(
        self, given_name: str, surname: str, personal_number: str, conn: sqlite3.Connection, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Name / Personal Number")
        self.finished.connect(lambda _result: save_geometry(conn, self, _EDIT_FIELDS_DIALOG_GEOMETRY_KEY))

        form = QFormLayout(self)
        self.given_name_edit = QLineEdit(given_name)
        form.addRow("Given name(s):", self.given_name_edit)
        self.surname_edit = QLineEdit(surname)
        form.addRow("Surname:", self.surname_edit)
        self.personal_number_edit = QLineEdit(personal_number)
        form.addRow("Personal number:", self.personal_number_edit)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        restore_geometry(conn, self, _EDIT_FIELDS_DIALOG_GEOMETRY_KEY)

    def values(self) -> tuple[str, str, str]:
        """Return (given_name, surname, personal_number) once the dialog has been accepted."""
        return self.given_name_edit.text().strip(), self.surname_edit.text().strip(), self.personal_number_edit.text().strip()


_SIDE_BY_SIDE_DIALOG_GEOMETRY_KEY = "side_by_side_dialog_geometry"
_SIDE_BY_SIDE_MIN_IMAGE_SIZE = 120
# Rough allowance for everything in the dialog besides the two images themselves -- the title
# labels' own row, and the layout's outer margins/inter-column spacing. Doesn't need to be exact:
# it only sets how tightly the window's size cap (see _constrain_window_size) hugs the images: a
# little slack just leaves a few harmless extra pixels of gray, not a functional problem.
_SIDE_BY_SIDE_CHROME_ALLOWANCE_PX = 56
# Available-screen-size fraction used to cap how big this dialog can ever get for a very
# high-resolution capture or a very wide document capture -- without this, e.g. two 4K images side
# by side would demand a window bigger than most screens.
_SIDE_BY_SIDE_MAX_SCREEN_FRACTION = 0.85


class _AspectRatioImageLabel(QLabel):
    """A QLabel that keeps a stored source pixmap rescaled to fill as much of its own current size
    as possible (preserving aspect ratio), redone on every resize -- used by
    _SideBySideImageDialog so its two images grow/shrink with the dialog window instead of staying
    fixed at whatever size they were first drawn at.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(_SIDE_BY_SIDE_MIN_IMAGE_SIZE, _SIDE_BY_SIDE_MIN_IMAGE_SIZE)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        self._source_pixmap = pixmap
        self._rescale()

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._source_pixmap is None or self._source_pixmap.isNull():
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
        )


class _SideBySideImageDialog(QDialog):
    """A quick-glance comparison view of a scan's live face capture and document capture side by
    side, opened via the small "⇄" compare button between the two thumbnails in HistoryRowWidget's
    summary row. Deliberately static (no zoom/pan/crop) -- that's already available by double-
    clicking either thumbnail individually (see ui/image_viewer.py::ImageViewerWindow).

    Each title stays pinned to the top of its column (fixed height, no stretch) while its image
    label takes stretch=1 and _AspectRatioImageLabel's own resizeEvent keeps it rescaled to fill
    whatever space that leaves -- so shrinking this window (see _constrain_window_size for why it
    can only shrink, never grow past the images' own native size) still scales the images down
    smoothly rather than just clipping them.
    """

    def __init__(
        self, face_image_path: str | None, doc_image_path: str | None, conn: sqlite3.Connection, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compare Face and Document")
        self._conn = conn
        self.finished.connect(lambda _result: save_geometry(self._conn, self, _SIDE_BY_SIDE_DIALOG_GEOMETRY_KEY))

        layout = QHBoxLayout(self)
        face_column, face_pixmap = self._build_image_column("Live Face Capture", face_image_path)
        doc_column, doc_pixmap = self._build_image_column("Document Capture", doc_image_path)
        layout.addLayout(face_column)
        layout.addLayout(doc_column)

        self._constrain_window_size(face_pixmap, doc_pixmap)
        restore_geometry(conn, self, _SIDE_BY_SIDE_DIALOG_GEOMETRY_KEY)

    def _build_image_column(self, title: str, image_path: str | None) -> tuple[QVBoxLayout, QPixmap | None]:
        column = QVBoxLayout()
        title_label = QLabel(f"<b>{title}</b>")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(title_label)  # no stretch -- stays pinned to the top of the column

        image_label = _AspectRatioImageLabel()
        pixmap = QPixmap(image_path) if image_path and Path(image_path).exists() else None
        if pixmap is not None and not pixmap.isNull():
            image_label.set_source_pixmap(pixmap)
        else:
            image_label.setText("No image")
            pixmap = None
        column.addWidget(image_label, stretch=1)  # absorbs all remaining vertical space
        return column, pixmap

    def _constrain_window_size(self, face_pixmap: QPixmap | None, doc_pixmap: QPixmap | None) -> None:
        """Cap the window at roughly the combined size of both images shown at their own native
        resolution (never smaller than a modest per-image minimum), and open at that size by
        default -- growing this window bigger than that would only ever add empty gray space
        around already-full-resolution images, never reveal more real detail, which is exactly the
        excess gray space reported around a dialog left at its old fixed initial size regardless of
        how small the actual images were. Still freely shrinkable below the cap (image labels
        rescale down via _AspectRatioImageLabel), just not growable past it.
        """
        screen = self.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else None
        max_screen_height = int(available.height() * _SIDE_BY_SIDE_MAX_SCREEN_FRACTION) if available is not None else 900
        max_screen_width = int(available.width() * _SIDE_BY_SIDE_MAX_SCREEN_FRACTION) if available is not None else 1400

        def total_width_at_height(image_height: int) -> int:
            def width_at(pixmap: QPixmap | None) -> int:
                if pixmap is None or pixmap.isNull() or pixmap.height() <= 0:
                    return _SIDE_BY_SIDE_MIN_IMAGE_SIZE
                return round(pixmap.width() * (image_height / pixmap.height()))

            return width_at(face_pixmap) + width_at(doc_pixmap)

        native_height = max(
            (face_pixmap.height() if face_pixmap is not None else 0),
            (doc_pixmap.height() if doc_pixmap is not None else 0),
            _SIDE_BY_SIDE_MIN_IMAGE_SIZE,
        )
        target_image_height = min(native_height, max_screen_height)

        # A very wide document capture (e.g. a light-box shot with the card occupying only a
        # fraction of the frame) can demand a combined width well past the screen even at a modest
        # shared height -- shrink the shared height further, proportionally, until the combined
        # width fits too, rather than only ever capping height.
        max_content_width = max_screen_width - _SIDE_BY_SIDE_CHROME_ALLOWANCE_PX
        combined_width = total_width_at_height(target_image_height)
        if combined_width > max_content_width and combined_width > 0:
            target_image_height = max(1, round(target_image_height * (max_content_width / combined_width)))

        window_width = total_width_at_height(target_image_height) + _SIDE_BY_SIDE_CHROME_ALLOWANCE_PX
        window_height = target_image_height + _SIDE_BY_SIDE_CHROME_ALLOWANCE_PX

        self.setMaximumSize(window_width, window_height)
        self.resize(window_width, window_height)


class _InstantTooltipLabel(QLabel):
    """A QLabel whose tooltip (see setToolTip) appears the instant the cursor enters it, instead of after Qt's own default hover delay (QStyle.StyleHint.SH_ToolTip_WakeUpDelay, commonly ~700ms).

    Used for the match-confidence and flagged-person badges (see HistoryRowWidget._build_layout):
    both moved a detail panel that used to be permanent inline text into a hover tooltip, so a
    real delay before that tooltip appears works against the whole point of it being there --
    unlike an ordinary tooltip that's just a supplementary hint, this one *is* the feature.
    """

    def enterEvent(self, event) -> None:
        super().enterEvent(event)
        tooltip_text = self.toolTip()
        if tooltip_text:
            QToolTip.showText(QCursor.pos(), tooltip_text, self)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        QToolTip.hideText()


class HistoryRowWidget(QFrame):
    """One row in the scan history list: both thumbnails, extracted identity fields, the match icon, and an expandable detail/override section."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        scan: ScanRecord,
        face_analysis_app: FaceAnalysis | None = None,
        on_delete: Callable[["HistoryRowWidget", int], None] | None = None,
        on_flagged_persons_changed: Callable[[], None] | None = None,
        on_jump_to_scan: Callable[[int], None] | None = None,
        on_reanalyze: Callable[[ScanRecord], None] | None = None,
        on_export: Callable[[ScanRecord, list[ScanRecord]], None] | None = None,
        on_selection_toggled: Callable[[ScanRecord, bool], None] | None = None,
        all_scans: list[ScanRecord] | None = None,
        person_index: PersonIndex | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.conn = conn
        self.scan = scan
        # Passed through to each thumbnail's ClickableThumbnailLabel (see _build_thumbnail) so
        # double-clicking the face thumbnail can auto-zoom to the detected face -- see
        # ui/image_viewer.py. None is a valid, supported value (e.g. _SAMPLE_SCAN_FOR_SIZING's
        # throwaway sizing row never gets double-clicked), and just disables that thumbnail's
        # auto-zoom rather than erroring.
        self.face_analysis_app = face_analysis_app
        self._on_delete = on_delete
        self._on_flagged_persons_changed = on_flagged_persons_changed
        self._on_jump_to_scan = on_jump_to_scan
        self._on_reanalyze = on_reanalyze
        self._on_export = on_export
        self._on_selection_toggled = on_selection_toggled
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Parsed once here rather than by every individual call site that needs it (previously
        # _build_layout and the flagged-user dialog each re-ran json.loads(doc_fields_json) + date
        # parsing independently) -- cheap per row, but adds up across many rows. Note
        # MainWindow._recompute_filtered_scans (the history search/filter row) computes its own
        # copy directly from each ScanRecord via _derive_display_fields rather than reading this
        # cached one, since it has to run before a scan's row (and this cache) even exists -- see
        # that method's own docstring.
        self.display_fields = _derive_display_fields(self.scan.doc_fields_json)
        self._previous_scans_cache = self._compute_previous_scans(all_scans, person_index)
        self._build_layout()
        self._apply_flagged_row_style()

    def _visit_position_and_total(self) -> tuple[int, int]:
        """This scan's 1-indexed chronological position among every saved scan for the same person (including itself), and the total count -- e.g. (2, 3) for "the 2nd of 3 visits on record so far".

        Previously the badge showed only the total (len(_previous_scans_cache) + 1), which is
        identical on every one of a person's rows regardless of which visit each one actually was
        -- correct as a count, but not a position, and looked like it hadn't updated correctly
        after deleting one of several visits (the *total* drops for everyone, as expected, but
        every remaining row still shows that same new total rather than its own distinct spot in
        the sequence). Derived from _previous_scans_cache (every *other* same-person scan, already
        computed at row-build time) plus this scan itself, sorted oldest-first.
        """
        all_visits = [*self._previous_scans_cache, self.scan]
        all_visits.sort(key=lambda scan: scan.created_at)
        position = next(index for index, scan in enumerate(all_visits, start=1) if scan.id == self.scan.id)
        return position, len(all_visits)

    def _compute_previous_scans(
        self, all_scans: list[ScanRecord] | None, person_index: PersonIndex | None = None
    ) -> list[ScanRecord]:
        """Find every other saved scan that represents the same person as this one, newest first -- see document/person_matcher.py. Computed once at row build time and cached, since it's read both for the summary row's visit-count badge and the detail section's visit history.

        person_index, if given, is a PersonIndex built once by the caller (e.g.
        MainWindow._load_history) over every saved scan -- an O(1)-average lookup instead of the
        O(N) linear scan find_previous_scans does over all_scans, which is what turned jumping to
        an old visit (MainWindow._jump_to_scan, which must synchronously build every row between
        the top of the list and the target) into an effectively O(N^2) UI-thread freeze on a large
        production scan history. Falls back to all_scans (a caller-supplied full scan list, reused
        to avoid each row independently re-running its own full-table query -- see the older
        docstring this replaced) when no index is given, e.g. the single fresh-capture insert path
        where building a whole index for just one row isn't worth it.
        """
        if not self.scan.doc_fields_json:
            return []
        doc_fields = json.loads(self.scan.doc_fields_json)
        if not doc_fields:
            return []
        if person_index is not None:
            return find_previous_scans_indexed(doc_fields, person_index, exclude_scan_id=self.scan.id)
        scans = all_scans if all_scans is not None else list_scans(self.conn, limit=100_000)
        return find_previous_scans(doc_fields, scans, exclude_scan_id=self.scan.id)

    def _build_layout(self) -> None:
        """Build the always-visible summary row and the hidden-by-default detail section beneath it."""
        outer_layout = QVBoxLayout(self)

        summary_row = QHBoxLayout()

        self.export_selection_checkbox = QCheckBox()
        self.export_selection_checkbox.setToolTip("Select for batch export")
        self.export_selection_checkbox.toggled.connect(self._on_export_selection_toggled)
        summary_row.addWidget(self.export_selection_checkbox)

        summary_row.addWidget(self._build_thumbnail(self.scan.face_image_path, "Live Face Capture", kind="face"))
        summary_row.addWidget(self._build_compare_button())
        summary_row.addWidget(self._build_doc_thumbnail_column())

        display_fields = self.display_fields
        identity_column = QVBoxLayout()
        identity_column.addWidget(QLabel(f"{display_fields.first_name} {display_fields.last_name}"))
        identity_column.addWidget(QLabel(f"DOB: {display_fields.birth_date_display}   Age: {display_fields.age_display}"))
        identity_column.addWidget(QLabel(f"Personnummer: {display_fields.personal_number_display}"))
        identity_column.addWidget(self._build_country_row())
        identity_column.addWidget(QLabel(_format_timestamp(self.scan.created_at)))
        summary_row.addLayout(identity_column, stretch=1)
        summary_row.addSpacing(32)

        self.match_icon_label = _InstantTooltipLabel(_match_icon_text(self.scan.match_result))
        self.match_icon_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
        # The raw similarity score, whether the result was automatic or an operator override, and
        # the current face-match threshold for reference are all in the hover tooltip rather than
        # a permanent inline label next to the icon -- that used to crowd every single row with a
        # number that's only occasionally worth checking; a tooltip surfaces it exactly when an
        # operator actually wants it, on demand, without it competing for space on every row.
        self.match_icon_label.setToolTip(_match_confidence_tooltip(self.conn, self.scan.match_score, self.scan.match_result))
        summary_row.addWidget(self.match_icon_label)

        self.status_label = QLabel(f"Doc: {(self.scan.validation_status or '-').upper()}")
        summary_row.addWidget(self.status_label)

        person_flags_text = self._person_flags_badge_text()
        if person_flags_text:
            self.person_flags_label = QLabel(f"⚠ {person_flags_text}")
            self.person_flags_label.setStyleSheet(
                "color: white; background-color: #C62828; font-weight: bold; padding: 2px 6px; border-radius: 3px;"
            )
            summary_row.addWidget(self.person_flags_label)

        flagged_match = self._flagged_match_info()
        if flagged_match is not None:
            flagged_reason = flagged_match.get("notes") or flagged_match.get("flagged_person_name") or "Flagged person"
            # Just the icon and reason -- name, personal number, which field matched, and when the
            # entry was added to the watch list all moved into the hover tooltip instead (see
            # _flagged_match_tooltip) rather than crowding the badge's own visible text with a
            # "(matched by ...)" suffix on every flagged row.
            self.flagged_match_label = _InstantTooltipLabel(f"🚩 {flagged_reason}")
            self.flagged_match_label.setToolTip(_flagged_match_tooltip(self.conn, flagged_match))
            self.flagged_match_label.setStyleSheet(
                "color: white; background-color: #B71C1C; font-weight: bold; padding: 2px 6px; border-radius: 3px;"
            )
            summary_row.addWidget(self.flagged_match_label)

        if self._previous_scans_cache:
            position, total = self._visit_position_and_total()
            self.visit_count_label = QLabel(f"🔁 Visit {position} of {total}")
            self.visit_count_label.setStyleSheet(
                "color: white; background-color: #1565C0; font-weight: bold; padding: 2px 6px; border-radius: 3px;"
            )
            summary_row.addWidget(self.visit_count_label)

        self.detail_toggle = QPushButton("Details ▾")
        self.detail_toggle.setCheckable(True)
        self.detail_toggle.toggled.connect(self._on_detail_toggle)
        summary_row.addWidget(self.detail_toggle)

        outer_layout.addLayout(summary_row)
        outer_layout.addWidget(self._build_detail_section())

    def _build_country_row(self) -> QWidget:
        """Build the row showing the identified document country: a small drawn flag icon (see ui/flag_icons.py) plus the country code as plain text."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(6)

        alpha2_code = alpha2_for_country_code(self.scan.doc_country)
        if alpha2_code:
            icon_label = QLabel()
            icon_label.setPixmap(render_flag_icon(alpha2_code))
            row_layout.addWidget(icon_label)

        text_label = QLabel(self.scan.doc_country or "-")
        text_label.setTextFormat(Qt.TextFormat.PlainText)
        row_layout.addWidget(text_label)
        row_layout.addStretch(1)

        return row_widget

    _QUALITY_BANNER_HEIGHT = 14

    def _build_doc_thumbnail_column(self) -> QWidget:
        """Build the document thumbnail, with a passive low-quality banner overlaid flush across its top edge when the capture was flagged blurry/glare-heavy.

        The banner is purely informational -- it was decided at capture time (see
        MainWindow._on_capture_and_process_clicked) and stored in validation_detail_json's
        "image_quality" key, never re-evaluated here, and never blocks or prompts anything.
        It's positioned with explicit geometry (rather than added to a layout) so it sits exactly
        flush against the image's top edge at full thumbnail width, instead of a layout-managed
        widget shrinking to its own size hint and floating disconnected from the image. This also
        keeps the container's own size fixed to the thumbnail size regardless of whether a banner
        is present, so the document thumbnail never shifts relative to the face thumbnail beside
        it (which never gets a banner).
        """
        container = QWidget()
        container.setFixedSize(*_THUMBNAIL_SIZE)

        thumbnail = self._build_thumbnail(self.scan.doc_image_path, "Document Capture", kind="document")
        thumbnail.setParent(container)
        thumbnail.move(0, 0)

        quality_warning_text = self._doc_image_quality_warning_text()
        if quality_warning_text:
            banner = QLabel(f"⚠ {quality_warning_text}", container)
            banner.setStyleSheet(
                "color: white; background-color: rgba(255, 152, 0, 130); font-weight: bold; font-size: 7pt;"
            )
            banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
            banner.setGeometry(0, 0, _THUMBNAIL_SIZE[0], self._QUALITY_BANNER_HEIGHT)
            banner.raise_()

        return container

    def _person_flags_badge_text(self) -> str | None:
        """Return a short banner label for this scan if it was flagged as under 18 and/or expired, else None.

        Reads the "person_flags" key stored in validation_detail_json at capture time (see
        MainWindow._on_capture_and_process_clicked / document.field_crosscheck.check_person_flags)
        -- never re-evaluated here.
        """
        if not self.scan.validation_detail_json:
            return None
        detail = json.loads(self.scan.validation_detail_json)
        flags = detail.get("person_flags")
        if not flags:
            return None
        labels = []
        if flags.get("is_minor"):
            labels.append("UNDER 18")
        if flags.get("is_expired"):
            labels.append("EXPIRED")
        return " / ".join(labels) if labels else None

    def _flagged_match_info(self) -> dict | None:
        """Return this scan's stored "flagged_match" detail dict if it matches an entry on the operator's flagged-persons watch list, else None.

        Just reads whatever is currently stored -- never re-evaluated here. Normally set once, at
        capture time (see MainWindow._run_capture_and_process / document.flag_matcher.
        find_matching_flagged_person), but can also be set afterward -- either retroactively, when
        a new watch-list entry is added and this scan turns out to already match it (see
        ui/flag_retroactive.py), or cleared, when the operator removes the flag (see
        _on_remove_flag_clicked / storage.models.clear_flagged_match_for_person).
        """
        if not self.scan.validation_detail_json:
            return None
        return json.loads(self.scan.validation_detail_json).get("flagged_match")

    def _apply_flagged_row_style(self) -> None:
        """Tint the whole row's background red when it matched the flagged-persons watch list, so it stands out in the history list at a glance rather than only in the small badge text."""
        if self._flagged_match_info() is not None:
            self.setStyleSheet("HistoryRowWidget { background-color: rgba(198, 40, 40, 40); }")

    def _doc_image_quality_warning_text(self) -> str | None:
        """Return the stored image-quality warning text for this scan's document capture, or None if it was fine (or never assessed)."""
        if not self.scan.validation_detail_json:
            return None
        detail = json.loads(self.scan.validation_detail_json)
        quality = detail.get("image_quality")
        if not quality or not quality.get("has_warning"):
            return None
        return f"Low quality ({quality.get('text', 'unknown')})"

    def _build_compare_button(self) -> QWidget:
        """A small icon button between the two thumbnails that opens both saved images side by side (see _SideBySideImageDialog) -- a quicker glance than double-clicking each thumbnail individually and mentally flipping between two separate viewer windows."""
        button = QPushButton("⇄")
        button.setFixedSize(18, 18)
        button.setToolTip("Compare face and document side by side")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setStyleSheet("QPushButton { font-size: 9pt; padding: 0; }")
        button.clicked.connect(self._on_compare_clicked)
        return button

    def _on_compare_clicked(self) -> None:
        dialog = _SideBySideImageDialog(self.scan.face_image_path, self.scan.doc_image_path, self.conn, parent=self)
        dialog.exec()

    def _build_thumbnail(self, image_path: str | None, title: str, kind: str | None = None) -> QLabel:
        """Build a fixed-size thumbnail label for one of the row's two saved images. Double-clicking it opens the full image in a resizable viewer window, auto-zoomed to the detected face/card region when kind is given, with an "Edit image…" (crop/brightness/contrast, then optional re-analyze) action -- see ui/image_viewer.py."""
        label = ClickableThumbnailLabel(
            image_path,
            title,
            kind=kind,
            face_analysis_app=self.face_analysis_app,
            scan=self.scan,
            on_reanalyze=self._on_reanalyze,
            conn=self.conn,
        )
        label.setFixedSize(*_THUMBNAIL_SIZE)
        pixmap = _load_thumbnail(image_path)
        if pixmap is not None:
            label.setPixmap(pixmap)
            label.setCursor(Qt.CursorShape.PointingHandCursor)
            label.setToolTip("Double-click to view full size")
        else:
            label.setText("No image")
        return label

    def _build_detail_section(self) -> QWidget:
        """Build the collapsible detail section: score, document classification, cross-check notes, and the override control."""
        self.detail_widget = QWidget()
        self.detail_widget.setVisible(False)
        layout = QVBoxLayout(self.detail_widget)

        score_text = f"{self.scan.match_score:.4f}" if self.scan.match_score is not None else "-"
        layout.addWidget(QLabel(f"Raw similarity score: {score_text}"))
        layout.addWidget(QLabel(f"Document type: {self.scan.doc_type or '-'}   Country: {self.scan.doc_country or '-'}"))

        if self.scan.validation_detail_json:
            detail = json.loads(self.scan.validation_detail_json)
            for note in detail.get("notes", []):
                layout.addWidget(QLabel(f"• {note}"))
            for comparison in detail.get("field_comparisons", []):
                status = "OK" if comparison.get("found_in_ocr") else "MISMATCH"
                layout.addWidget(QLabel(f"[{status}] {comparison.get('field_name')}: {comparison.get('machine_read_value')}"))

        if self._previous_scans_cache:
            layout.addWidget(QLabel(f"<b>Previous visits ({len(self._previous_scans_cache)}):</b>"))
            for previous_scan in self._previous_scans_cache[:5]:
                layout.addWidget(self._build_previous_visit_link(previous_scan))

        layout.addWidget(QLabel("<b>Comments:</b>"))
        self.comments_list_layout = QVBoxLayout()
        layout.addLayout(self.comments_list_layout)
        self._refresh_comments_list()

        add_comment_row = QHBoxLayout()
        self.new_comment_edit = QLineEdit()
        self.new_comment_edit.setPlaceholderText("Add a comment about this scan...")
        add_comment_row.addWidget(self.new_comment_edit, stretch=1)
        add_comment_button = QPushButton("Add")
        add_comment_button.clicked.connect(self._on_add_comment_clicked)
        add_comment_row.addWidget(add_comment_button)
        layout.addLayout(add_comment_row)

        override_row = QHBoxLayout()
        override_row.addWidget(QLabel("Operator override:"))
        self.override_combo = QComboBox()
        self.override_combo.addItems(_OVERRIDE_CHOICES)
        self.override_combo.setCurrentText(self._initial_override_choice())
        override_row.addWidget(self.override_combo)

        apply_button = QPushButton("Apply")
        apply_button.clicked.connect(self._on_apply_override_clicked)
        override_row.addWidget(apply_button)

        export_button = QPushButton("Export report")
        export_button.clicked.connect(self._on_export_clicked)
        override_row.addWidget(export_button)

        flag_button = QPushButton("Add to Flagged Users…")
        flag_button.clicked.connect(self._on_flag_person_clicked)
        override_row.addWidget(flag_button)

        edit_fields_button = QPushButton("Edit Name/ID…")
        edit_fields_button.clicked.connect(self._on_edit_fields_clicked)
        override_row.addWidget(edit_fields_button)

        reanalyze_button = QPushButton("Re-analyze")
        reanalyze_button.clicked.connect(self._on_reanalyze_clicked)
        override_row.addWidget(reanalyze_button)

        override_row_buttons = [apply_button, export_button, flag_button, edit_fields_button, reanalyze_button]

        if self._flagged_match_info() is not None:
            remove_flag_button = QPushButton("Remove Flag")
            remove_flag_button.setStyleSheet("color: #FF1744;")
            remove_flag_button.clicked.connect(self._on_remove_flag_clicked)
            override_row.addWidget(remove_flag_button)
            override_row_buttons.append(remove_flag_button)

        delete_button = QPushButton("Delete")
        delete_button.setStyleSheet("color: #FF1744;")
        delete_button.clicked.connect(self._on_delete_clicked)
        override_row.addWidget(delete_button)
        override_row_buttons.append(delete_button)

        # Every button in this row shares one common width instead of each auto-sizing to its own
        # text, which left "Add to Flagged Users…"/"Export report" much wider than "Apply"/the old
        # fixed-70px "Delete" -- an inconsistent-looking row. Sized to the widest of the row's
        # *other* (already reasonably short) buttons, measured via sizeHint() rather than a
        # hardcoded guess -- not to "Add to Flagged Users…"/"Export report" themselves, since with
        # six buttons now in this row, stretching everything out to their much wider natural size
        # would make the whole row uncomfortably wide. Those two instead get their button text
        # elided to fit this narrower common width, with the full label still available as a
        # tooltip and unchanged as their actual accessible text/logic.
        long_label_buttons = {export_button, flag_button}
        common_button_width = max(
            button.sizeHint().width() for button in override_row_buttons if button not in long_label_buttons
        )
        for button in override_row_buttons:
            if button in long_label_buttons:
                full_text = button.text()
                button.setToolTip(full_text)
                available_text_width = common_button_width - 20  # rough allowance for the button's own padding/chrome
                button.setText(button.fontMetrics().elidedText(full_text, Qt.TextElideMode.ElideRight, available_text_width))
            button.setFixedWidth(common_button_width)

        layout.addLayout(override_row)

        return self.detail_widget

    def _build_previous_visit_link(self, previous_scan: ScanRecord) -> QWidget:
        """Build one clickable "previous visit" entry that jumps the operator straight to that earlier scan's own row (see MainWindow._jump_to_scan) rather than just listing its timestamp as plain text.

        A flat QPushButton styled to read as a hyperlink -- not a QLabel with rich-text <a> markup
        -- so the click target is the whole line (a real button's hit-testing), not just wherever
        the underlined text happens to sit, and no linkActivated href-string parsing is needed to
        recover which scan was clicked.
        """
        link_button = QPushButton(f"• {_format_timestamp(previous_scan.created_at)}")
        link_button.setFlat(True)
        link_button.setCursor(Qt.CursorShape.PointingHandCursor)
        link_button.setStyleSheet(
            "QPushButton { color: #1565C0; text-decoration: underline; border: none; "
            "background: transparent; padding: 0; text-align: left; }"
        )
        link_button.setToolTip("Jump to this scan")
        link_button.clicked.connect(lambda _checked, scan_id=previous_scan.id: self._on_previous_visit_link_clicked(scan_id))
        return link_button

    def _on_previous_visit_link_clicked(self, scan_id: int) -> None:
        if self._on_jump_to_scan is not None:
            self._on_jump_to_scan(scan_id)

    def _refresh_comments_list(self) -> None:
        """Rebuild this scan's comments list from the database -- called on build and after any add/delete."""
        while self.comments_list_layout.count():
            item = self.comments_list_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        comments = list_scan_comments(self.conn, self.scan.id)
        if not comments:
            empty_label = QLabel("No comments yet.")
            empty_label.setStyleSheet("color: gray;")
            self.comments_list_layout.addWidget(empty_label)
            return

        for comment in comments:
            self.comments_list_layout.addWidget(self._build_comment_row(comment))

    def _build_comment_row(self, comment: dict) -> QWidget:
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        timestamp_display = _format_timestamp(comment["created_at"]) if comment.get("created_at") else "-"
        text_label = QLabel(f"[{timestamp_display}] {comment.get('text', '')}")
        text_label.setWordWrap(True)
        row_layout.addWidget(text_label, stretch=1)

        delete_button = QPushButton("×")
        delete_button.setFixedWidth(24)
        delete_button.setStyleSheet("color: #FF1744;")
        delete_button.setToolTip("Delete this comment")
        delete_button.clicked.connect(lambda _checked, comment_id=comment.get("id"): self._on_delete_comment_clicked(comment_id))
        row_layout.addWidget(delete_button)

        return row_widget

    def _on_add_comment_clicked(self) -> None:
        """Add a new comment to this scan, and to every other saved scan that matches the same person (see document/person_matcher.py -- the same matching that powers the visit-count badge and previous-visits list), so a note travels with a person's whole visit history rather than being scoped to just the one scan the operator happened to have open."""
        text = self.new_comment_edit.text().strip()
        if not text:
            return

        add_scan_comment(self.conn, self.scan.id, text)
        matching_scans = self._previous_scans_cache  # already this same-person set, computed once at row build time
        for matching_scan in matching_scans:
            add_scan_comment(self.conn, matching_scan.id, text)
        logger.info(
            "Operator added a comment to scan %d (propagated to %d matching scan(s))", self.scan.id, len(matching_scans)
        )

        self.new_comment_edit.clear()
        self._refresh_comments_list()
        status_bar = self.window().statusBar() if isinstance(self.window(), QMainWindow) else None
        if status_bar is not None:
            suffix = f" (and {len(matching_scans)} matching scan(s))" if matching_scans else ""
            status_bar.showMessage(f"Comment added{suffix}", 3000)

    def _on_delete_comment_clicked(self, comment_id: int) -> None:
        """Delete one comment from this scan only -- does not cascade to the other matching scans it may have been propagated to when added, since each scan's copy is independent once created."""
        delete_scan_comment(self.conn, self.scan.id, comment_id)
        logger.info("Operator deleted comment %d from scan %d", comment_id, self.scan.id)
        self._refresh_comments_list()

    def _on_edit_fields_clicked(self) -> None:
        """Open a dialog to manually correct this scan's given name/surname/personal number, for when OCR/MRZ/barcode extraction got a field wrong or missed it entirely -- without needing a full re-capture.

        If this scan is currently flagged, also propagates a corrected name/personal number into
        the linked flagged-persons watch-list entry (see _sync_flagged_person_from_edit) -- an
        operator correcting a flagged visitor's misread name here would otherwise leave the master
        watch-list entry (and thus future matching) still using the old, wrong value.
        """
        doc_fields = json.loads(self.scan.doc_fields_json) if self.scan.doc_fields_json else {}
        dialog = _EditFieldsDialog(
            doc_fields.get("name", ""),
            doc_fields.get("surname", ""),
            doc_fields.get("personal_number", ""),
            conn=self.conn,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        given_name, surname, personal_number = dialog.values()
        for key, value in (("name", given_name), ("surname", surname), ("personal_number", personal_number)):
            if value:
                doc_fields[key] = value
            else:
                doc_fields.pop(key, None)

        doc_fields_json = json.dumps(doc_fields)
        update_scan_doc_fields(self.conn, self.scan.id, doc_fields_json)
        self.scan.doc_fields_json = doc_fields_json
        logger.info("Operator manually edited name/personal-number fields for scan %d", self.scan.id)

        self._sync_flagged_person_from_edit(f"{given_name} {surname}".strip(), personal_number)

        if self._on_flagged_persons_changed is not None:
            # Reuses the same "refresh everything" callback flag changes use -- editing a name/
            # personal number can change which other scans this one's visit history/count (see
            # document/person_matcher.py) or watch-list match reflects, not just this row's own
            # display, so a full history reload is needed here too.
            self._on_flagged_persons_changed()

    def _sync_flagged_person_from_edit(self, corrected_name: str, corrected_personal_number: str) -> None:
        """If this scan is currently flagged, propagate a corrected name/personal number into the linked flagged-persons entry via ui/flag_retroactive.py::apply_flagged_person_edit -- the same shared apply path Settings' own Edit button uses, so a change here still warns first if it would unflag some other scan matching the watch-list entry's *old* details. A no-op if nothing relevant actually changed, or if the linked entry was since removed from the watch list."""
        flagged_match = self._flagged_match_info()
        if flagged_match is None or flagged_match.get("flagged_person_id") is None:
            return
        flagged_person = get_flagged_person_by_id(self.conn, flagged_match["flagged_person_id"])
        if flagged_person is None:
            return

        name_changed = bool(corrected_name) and corrected_name != (flagged_person.name or "")
        number_changed = bool(corrected_personal_number) and corrected_personal_number != (flagged_person.personal_number or "")
        if not (name_changed or number_changed):
            return

        apply_flagged_person_edit(
            self.conn,
            flagged_person,
            corrected_name or flagged_person.name,
            corrected_personal_number or flagged_person.personal_number,
            flagged_person.notes,
            flagged_person.image_path,
            self,
        )

    def _on_reanalyze_clicked(self) -> None:
        if self._on_reanalyze is not None:
            self._on_reanalyze(self.scan)

    def _on_flag_person_clicked(self) -> None:
        """Open the Add-to-Flagged-Users dialog prefilled from this scan's extracted fields, save the entry the operator confirms, then offer to retroactively mark this and any other matching saved scan (see ui/flag_retroactive.py)."""
        display_fields = self.display_fields
        suggested_name = f"{display_fields.first_name} {display_fields.last_name}".strip(" -")
        doc_fields = json.loads(self.scan.doc_fields_json) if self.scan.doc_fields_json else {}
        suggested_personal_number = doc_fields.get("personal_number", "")

        has_face_image = bool(self.scan.face_image_path and Path(self.scan.face_image_path).exists())
        has_doc_image = bool(self.scan.doc_image_path and Path(self.scan.doc_image_path).exists())

        dialog = _FlagPersonDialog(
            suggested_name, suggested_personal_number, has_face_image, has_doc_image, conn=self.conn, parent=self
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        name, personal_number, image_source, notes = dialog.values()
        if not name and not personal_number:
            QMessageBox.warning(self, "Add to Flagged Users", "Check and fill in at least a name or a personal number.")
            return

        image_path = self._copy_flagged_reference_image(image_source)
        flagged_person_id = insert_flagged_person(self.conn, name or None, personal_number or None, image_path, notes or None)
        logger.info("Operator added flagged person id=%d from scan %d", flagged_person_id, self.scan.id)
        QMessageBox.information(self, "Add to Flagged Users", f"Added '{name or personal_number}' to the flagged-persons list.")

        flagged_person = FlaggedPerson(
            id=flagged_person_id,
            created_at=datetime.now().isoformat(),
            name=name or None,
            personal_number=personal_number or None,
            image_path=image_path,
            notes=notes or None,
        )
        run_retroactive_flag_check(self.conn, flagged_person, self)
        if self._on_flagged_persons_changed is not None:
            self._on_flagged_persons_changed()

    def _on_remove_flag_clicked(self) -> None:
        """Remove this scan's flag by deleting the underlying flagged-persons entry entirely, which cascades to every other scan currently flagged against it (see storage.models.clear_flagged_match_for_person) -- a flag is a property of the watch-list entry, not of one individual scan."""
        flagged_match = self._flagged_match_info()
        if flagged_match is None:
            return
        flagged_person_id = flagged_match.get("flagged_person_id")
        confirm = QMessageBox.question(
            self,
            "Remove flag",
            "Remove this flag? This deletes the underlying entry from the flagged-persons list in "
            "Settings and clears the flag from every scan matching it, not just this one.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        if flagged_person_id is not None:
            delete_flagged_person(self.conn, flagged_person_id)
            cleared_count = clear_flagged_match_for_person(self.conn, flagged_person_id)
            logger.info(
                "Operator removed flag for scan %d (flagged_person_id=%d, cleared %d scan(s))",
                self.scan.id,
                flagged_person_id,
                cleared_count,
            )
        if self._on_flagged_persons_changed is not None:
            self._on_flagged_persons_changed()

    def _copy_flagged_reference_image(self, image_source: str | None) -> str | None:
        """Copy this scan's face or document image into the flagged-persons image store under a new unique name, so it survives independently of the original scan (which may later be deleted or purged)."""
        if image_source is None:
            return None
        source_path = Path(self.scan.face_image_path if image_source == "face" else self.scan.doc_image_path)
        destination_path = FLAGGED_IMAGE_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_{source_path.name}"
        shutil.copyfile(source_path, destination_path)
        return str(destination_path)

    def _initial_override_choice(self) -> str:
        """Pre-select the override combo to reflect this scan's currently stored match_result, so a prior override isn't lost from view."""
        if self.scan.match_result == "override_match":
            return "Override: Match"
        if self.scan.match_result == "override_no_match":
            return "Override: No Match"
        return "Automatic"

    def _on_detail_toggle(self, checked: bool) -> None:
        """Show or hide this row's detail section when its toggle button is clicked."""
        self.detail_widget.setVisible(checked)
        self.detail_toggle.setText("Details ▴" if checked else "Details ▾")

    def _on_apply_override_clicked(self) -> None:
        """Persist the operator's override choice for this scan and refresh the row's match icon to reflect it."""
        automatic_is_match = self.scan.match_result in ("match", "override_match") if self.scan.match_result else None
        # The automated result itself never changes after the fact -- only whichever of
        # 'match'/'no_match' it originally landed on is recoverable from the stored label once
        # an override has already been applied once, which is a known, accepted limitation of
        # not storing the automatic and final results as two separate columns.
        new_label = _final_match_result_label(automatic_is_match, self.override_combo.currentText())
        update_match_result(self.conn, self.scan.id, new_label)
        logger.info("Operator overrode scan %d match_result to '%s'", self.scan.id, new_label)
        self.scan.match_result = new_label
        self.match_icon_label.setText(_match_icon_text(new_label))
        self.match_icon_label.setToolTip(_match_confidence_tooltip(self.conn, self.scan.match_score, new_label))

    def _on_export_clicked(self) -> None:
        """Delegate to MainWindow._on_export_scan_requested, passing along this scan's already-computed previous-visits cache -- MainWindow owns the actual file-save dialog and (if the operator has previous visits and opts in) bundling them into one combined report, since that's shared with the checkbox-driven batch export below."""
        if self._on_export is not None:
            self._on_export(self.scan, self._previous_scans_cache)

    def _on_export_selection_toggled(self, checked: bool) -> None:
        if self._on_selection_toggled is not None:
            self._on_selection_toggled(self.scan, checked)

    def _on_delete_clicked(self) -> None:
        """Permanently delete this scan's DB row and saved images after operator confirmation, and remove this row from the list."""
        confirm = QMessageBox.question(
            self,
            "Delete scan",
            "Permanently delete this scan result, including its saved images? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        delete_scan(self.conn, self.scan.id)
        freed_bytes = 0
        for image_path in (self.scan.face_image_path, self.scan.doc_image_path):
            freed_bytes += _delete_image_and_cache(image_path)
        logger.info("Operator deleted scan %d", self.scan.id)

        if self._on_delete is not None:
            self._on_delete(self, freed_bytes)
        self.setParent(None)
        self.deleteLater()


class ResultAwareCalendarWidget(QCalendarWidget):
    """A date-picker calendar popup where only dates with at least one saved scan are selectable.

    Dates with a scan are painted white; dates without one are painted grey and are effectively
    unclickable -- QCalendarWidget has no native way to disable individual dates, so a click on a
    grey date is instead caught via selectionChanged and immediately reverted to the last valid
    (scan-having) date that was selected.
    """

    def __init__(
        self,
        result_dates_provider: Callable[[], set[date]],
        on_shown: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._result_dates_provider = result_dates_provider
        self._on_shown = on_shown
        self._last_valid_date = self.selectedDate()
        # Populated fresh each time the popup actually becomes visible (see showEvent), not here --
        # this widget is constructed as part of MainWindow's own layout building, before
        # MainWindow._history_rows exists yet, so calling result_dates_provider() this early would
        # fail. An empty set is a safe placeholder in the meantime: paintCell/_on_selection_changed
        # can't fire before this widget is shown for the first time regardless. Once shown, this is
        # refreshed on every open rather than calling result_dates_provider() from scratch on every
        # paintCell -- Qt repaints ~35-42 visible day cells on every open/month-navigation/hover,
        # and the provider rebuilds its date set from every history row each time it's called, so
        # that was up to ~40x more history-row work than necessary per single calendar repaint.
        self._cached_result_dates: set[date] = set()
        self.selectionChanged.connect(self._on_selection_changed)

    def showEvent(self, event) -> None:
        """Fire the on_shown hook the moment this popup actually becomes visible -- i.e. exactly when the operator opened it, in a single click."""
        super().showEvent(event)
        self._cached_result_dates = self._result_dates_provider()
        if self._on_shown is not None:
            self._on_shown()

    def _on_selection_changed(self) -> None:
        if self.selectedDate().toPyDate() in self._cached_result_dates:
            self._last_valid_date = self.selectedDate()
        else:
            self.setSelectedDate(self._last_valid_date)

    def paintCell(self, painter, rect, date) -> None:
        """Paint the native cell first (grid/background/native theme), then recolor just the day-number text on top."""
        super().paintCell(painter, rect, date)
        has_results = date.toPyDate() in self._cached_result_dates
        painter.save()
        painter.setPen(QColor("white") if has_results else QColor("#808080"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, str(date.day()))
        painter.restore()


class _ClickAnywhereDateEdit(QDateEdit):
    """A QDateEdit whose calendar popup opens on a click anywhere in the field, not just its narrow dropdown-arrow button.

    QDateTimeEdit's own internal click handling only opens the popup for a click landing exactly
    on its internal calendar-button sub-control -- a hit-test region whose precise geometry is
    determined by the active QStyle, and so isn't reliably reproducible by synthesizing a click
    aimed at a particular sub-control (that approach worked under the Fusion style used in headless
    testing, but not under Windows' native style used when actually running the app). Instead, any
    left-click anywhere in the field directly shows the same popup container QDateEdit itself
    would show, positioned explicitly just below the field -- sidestepping style-specific hit-test
    geometry entirely.
    """

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton and not self._popup_is_visible():
            self._show_calendar_popup()
            return
        super().mousePressEvent(event)

    def _popup_is_visible(self) -> bool:
        popup = self.calendarWidget().parent()
        return popup is not None and popup.isVisible()

    def _show_calendar_popup(self) -> None:
        popup = self.calendarWidget().parent()
        if popup is None:
            return
        popup.move(self.mapToGlobal(self.rect().bottomLeft()))
        popup.show()


class _FlashCapableLabel(QLabel):
    """A live camera preview label that can briefly flash solid white -- see flash_white -- for
    immediate on-screen feedback that a capture just happened, like a camera's own shutter flash.

    A plain QLabel showing the live camera feed can't just have its own background/stylesheet
    swapped to white for this: _update_previews calls setPixmap() on it every ~33ms (see
    _PREVIEW_INTERVAL_MS), which would paint the live frame straight back over any such change on
    the very next tick. Instead this adds a separate always-on-top child widget (_flash_overlay,
    full white, otherwise hidden) that _update_previews never touches -- flash_white shows it at
    full opacity and fades it back out via a QPropertyAnimation on its own QGraphicsOpacityEffect,
    independent of the preview tick's own repainting.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._flash_overlay = QLabel(self)
        self._flash_overlay.setStyleSheet("background-color: white;")
        self._flash_overlay.setVisible(False)
        self._flash_opacity_effect = QGraphicsOpacityEffect(self._flash_overlay)
        self._flash_opacity_effect.setOpacity(1.0)
        self._flash_overlay.setGraphicsEffect(self._flash_opacity_effect)
        # Parented to self (the label) as well as kept as an attribute -- Qt parent-child ownership
        # alone would be enough to keep this alive for the animation's duration, but the explicit
        # attribute reference makes that non-obvious-at-a-glance guarantee visible in the code too.
        self._flash_animation = QPropertyAnimation(self._flash_opacity_effect, b"opacity", self)
        self._flash_animation.setDuration(_CAPTURE_FLASH_FADE_MS)
        self._flash_animation.setStartValue(1.0)
        self._flash_animation.setEndValue(0.0)
        self._flash_animation.finished.connect(lambda: self._flash_overlay.setVisible(False))
        self._position_flash_overlay()

    def flash_white(self) -> None:
        """Show this preview at solid white, immediately fading back to reveal the live feed underneath -- see class docstring."""
        self._position_flash_overlay()
        self._flash_overlay.raise_()  # on top of any other overlay this label hosts (e.g. the quality-warning banner)
        self._flash_overlay.setVisible(True)
        self._flash_animation.stop()
        self._flash_opacity_effect.setOpacity(1.0)
        self._flash_animation.start()

    def _position_flash_overlay(self) -> None:
        self._flash_overlay.setGeometry(0, 0, self.width(), self.height())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_flash_overlay()


class _QualityOverlayLabel(_FlashCapableLabel):
    """The document preview label, extended to host both the live blur/resolution quality-warning banner (see set_quality_overlay) and the capture-flash overlay (see _FlashCapableLabel) -- kept as its own QLabel subclass (rather than a plain QLabel) purely so the banner can be positioned with explicit geometry inside the label's own frame rather than displacing it in the surrounding layout.

    No longer adds crop-selection support -- capture now saves immediately with no operator review/
    crop step (see MainWindow._run_capture_and_process); cropping moved to a post-hoc "Edit image…"
    action on a saved thumbnail (see ui/image_viewer.py).
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._quality_overlay: QLabel | None = None

    def set_quality_overlay(self, overlay: QLabel) -> None:
        """Parent a warning banner directly onto this label so it paints on top of the live preview instead of occupying its own row above it.

        Positioned with explicit geometry inside this label's own frame (see _position_quality_
        overlay), so toggling it visible/hidden never changes this label's size -- previously, a
        separate layout row above the preview label pushed the boxed preview down by exactly the
        banner's height whenever it appeared, misaligning this preview's top border against the
        sibling preview's border (which never has a banner).
        """
        overlay.setParent(self)
        self._quality_overlay = overlay
        self._position_quality_overlay()

    def _position_quality_overlay(self) -> None:
        if self._quality_overlay is None:
            return
        inset = self.frameWidth()
        overlay_width = max(self.width() - 2 * inset, 0)
        self._quality_overlay.setFixedWidth(overlay_width)
        self._quality_overlay.adjustSize()
        self._quality_overlay.move(inset, inset)
        self._quality_overlay.raise_()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_quality_overlay()


class MainWindow(QMainWindow):
    """The app's single combined screen: live previews, one capture-and-process button, and the scan history list."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        camera_manager: CameraManager,
        face_analysis_app: FaceAnalysis,
    ) -> None:
        super().__init__()
        # QApplication.setWindowIcon (main.py) already supplies this as the default for every
        # window that doesn't set its own -- set explicitly here too since Windows' title-bar icon
        # specifically (as opposed to the taskbar/Alt-Tab icon) doesn't always reliably pick up
        # that application-wide default for every top-level window.
        self.setWindowIcon(QIcon(render_app_logo(64)))
        self.conn = conn
        self.camera_manager = camera_manager
        self.face_analysis_app = face_analysis_app
        self._settings_window: SettingsWindow | None = None
        # Kept up to date incrementally (see _add_to_cached_storage_bytes) rather than
        # recalculated by walking the whole faces/documents directory tree on every single status-
        # bar refresh (_refresh_status_bar_stats runs after nearly every action) -- only a full
        # recount actually needs the real walk (_recompute_storage_bytes), which happens once here
        # at startup and again after a bulk change with an unknown-in-advance file count (a purge).
        self._cached_storage_bytes = 0

        # A capture click that gets a frame from only one camera holds onto that successful frame
        # here rather than discarding it, so the next click only has to retry the failed side
        # instead of redoing a capture that already succeeded -- see
        # _on_capture_and_process_clicked and _on_discard_pending_capture_clicked.
        self._pending_face_frame: np.ndarray | None = None
        self._pending_doc_frame: np.ndarray | None = None
        # Consecutive failed preview-tick reads per camera, since the last successful one -- once
        # either crosses _SIGNAL_LOST_TICK_THRESHOLD, _update_previews swaps that camera's frozen
        # last-good frame for an explicit "signal lost" message rather than leaving it looking
        # like a normal (if slightly stale) live feed. A single dropped tick doesn't count; only a
        # sustained run of them does, so an occasional one-off missed frame doesn't flicker it.
        self._face_signal_lost_streak = 0
        self._doc_signal_lost_streak = 0
        # Last-known (width, height) of each camera's own frame, in pixels -- kept even across a
        # tick where a fresh frame didn't arrive (see _update_top_pane_height_cap, which uses these
        # to cap how far down the splitter handle above the capture button can be dragged, so the
        # previews can't be stretched past the point where the camera feed itself can no longer
        # usefully grow to fill the extra height).
        self._face_frame_size: tuple[int, int] | None = None
        self._doc_frame_size: tuple[int, int] | None = None
        # The cap most recently applied to self._top_pane_widget's maximum height -- cached purely
        # so _update_top_pane_height_cap can skip the setMaximumHeight() call (and the layout
        # invalidation it triggers) on the very frequent ticks where the cap hasn't actually changed.
        self._top_pane_height_cap: int | None = None
        # See _warn_if_preview_tick_slow -- throttles how often a slow preview tick gets logged.
        self._slow_preview_tick_streak = 0
        # Backing state for the live document-camera blur warning (see
        # _update_doc_quality_warning) -- the tick count drives the throttled check interval, and
        # the last result is kept between checks so the banner doesn't flicker off on the ticks a
        # fresh check isn't due.
        self._doc_quality_check_tick_count = 0
        self._doc_frame_looks_blurry = False
        # Guards against re-entrant capture runs. The actual analysis pipeline runs on a background
        # thread now (see _start_capture_analysis), not synchronously -- without this, a click (or
        # shortcut press) during that window would fire a second, overlapping capture, and worse,
        # a second Re-analyze/capture could end up calling into the shared face_analysis_app model
        # from two worker threads at once. Set by both _on_capture_and_process_clicked and
        # _on_reanalyze_scan_requested, the two callers of _start_capture_analysis.
        self._is_processing = False
        # Set just before _start_capture_analysis is called and read back once its background
        # thread finishes -- kept alive on self since nothing else would hold a reference to the
        # running QThread/worker otherwise (see _start_capture_analysis's own docstring).
        self._capture_analysis_thread: QThread | None = None
        self._capture_analysis_worker: _CaptureAnalysisWorker | None = None
        # The capture button's own text, saved by _on_capture_and_process_clicked just before it's
        # overwritten with "Capturing…"/"Processing…", so _end_capture_processing can restore it
        # afterward.
        self._capture_button_original_text = ""
        # Scan ids checked via each row's export-selection checkbox, for the "Export selected"
        # batch-export button in the filter row -- see _on_history_row_selection_toggled /
        # _on_export_selected_clicked. IDs rather than ScanRecord/HistoryRowWidget references so
        # this survives a _reload_history rebuild trivially (it's explicitly cleared there instead,
        # since the fresh rows all start unchecked -- see that method).
        self._selected_scan_ids: set[int] = set()

        # History list incremental-loading state -- see _load_history/_recompute_filtered_scans/
        # _load_more_rows/_on_history_scrolled. self._all_scans is every saved scan, newest-first,
        # refreshed on every full reload (_load_history) and incrementally on a fresh capture
        # (_finish_save_capture); self._filtered_scans is that list narrowed by the current filter
        # row controls (see _recompute_filtered_scans); self._loaded_count is how many of
        # self._filtered_scans, counted from the front (newest), are currently built as
        # HistoryRowWidgets -- growing self._history_page_size at a time as the operator scrolls
        # near the bottom, rather than every matching scan being built as a widget up front (each
        # row is a real per-image decode cost -- see HistoryRowWidget._build_thumbnail).
        # self._history_page_size is re-read from settings on every full reload so a change made in
        # Settings (General tab) takes effect immediately (see
        # SettingsWindow._on_save_history_page_size_clicked's on_data_changed callback).
        self._all_scans: list[ScanRecord] = []
        # Rebuilt (see _rebuild_person_index) every time self._all_scans changes -- an O(1)-average
        # same-person lookup shared by every row, instead of each row's own O(N) scan over the full
        # list. See document/person_matcher.py::PersonIndex for why this exists: reaching an old
        # visit via _jump_to_scan used to mean building thousands of rows on a large production
        # history, each doing its own full-list scan -- effectively O(N^2) on the UI thread.
        self._person_index: PersonIndex | None = None
        self._filtered_scans: list[ScanRecord] = []
        self._loaded_count: int = 0
        self._history_page_size: int = get_history_page_size(self.conn)

        self.setWindowTitle("Kontroll")
        self._build_layout()
        self._load_history()
        self._restore_window_state()
        # Scheduled for the very next event-loop iteration, after the window has actually been
        # shown (main.py calls window.show() next, then app.exec()) -- see
        # _nudge_layout_to_match_actual_geometry's own docstring for why this fixes a real reported
        # startup glitch (the live camera previews rendering letterboxed/undersized within their own
        # box until the operator nudged the window with a manual resize).
        QTimer.singleShot(0, self._nudge_layout_to_match_actual_geometry)

        self.capture_shortcut = QShortcut(QKeySequence(get_capture_shortcut(self.conn)), self)
        self.capture_shortcut.activated.connect(self._on_capture_and_process_clicked)

        self.preview_timer = QTimer(self)
        self.preview_timer.timeout.connect(self._update_previews)
        self.preview_timer.start(_PREVIEW_INTERVAL_MS)

    def _build_layout(self) -> None:
        """Construct and arrange every widget on the single combined screen.

        A vertical splitter divides the window into two panes with a drag handle between them,
        positioned right below the capture button: the top pane (previews + capture button) and
        the bottom pane (history headline + scan list). QSplitter's own resize-distribution (via
        setStretchFactor) does not reliably keep one pane fixed on a plain window resize in
        practice, so that's instead enforced explicitly in resizeEvent(), which pins the top pane
        at its last explicit height and hands all resize delta to the bottom pane -- where the
        headline stays fixed-size and the scan list (stretch=1) is what actually absorbs it, so
        the list's own height tracks the window rather than a fixed pixel count. Dragging the
        handle itself still freely resizes the top pane (see _on_splitter_moved), which is what
        lets the previews grow or shrink on demand.
        """
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QVBoxLayout(central_widget)

        self.top_bottom_splitter = QSplitter(Qt.Orientation.Vertical)
        outer_layout.addWidget(self.top_bottom_splitter)

        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        previews_row = QHBoxLayout()
        self.face_preview_label = self._build_preview_label(_FlashCapableLabel)
        self.doc_preview_label = self._build_preview_label(_QualityOverlayLabel)
        self.doc_quality_warning_label = self._build_quality_warning_label()
        self.doc_preview_label.set_quality_overlay(self.doc_quality_warning_label)
        face_column, self.face_status_badge = self._build_preview_column("Visitor", self.face_preview_label)
        doc_column, self.doc_status_badge = self._build_preview_column("Identification", self.doc_preview_label)
        previews_row.addLayout(face_column, stretch=1)
        previews_row.addLayout(doc_column, stretch=1)
        top_layout.addLayout(previews_row, stretch=1)

        capture_row = QHBoxLayout()
        self.capture_button = QPushButton("Capture")
        self.capture_button.setMinimumHeight(48)
        self.capture_button.clicked.connect(self._on_capture_and_process_clicked)
        capture_row.addWidget(self.capture_button, stretch=1)

        self.discard_pending_button = QPushButton("Discard held capture")
        self.discard_pending_button.setMinimumHeight(48)
        self.discard_pending_button.setVisible(False)
        self.discard_pending_button.clicked.connect(self._on_discard_pending_capture_clicked)
        capture_row.addWidget(self.discard_pending_button)

        top_layout.addLayout(capture_row)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        # Hidden by default to keep the history list itself as roomy as possible -- toggled by the
        # "Search" button down by Settings in the status bar (see _build_status_bar/
        # _on_search_toggle_clicked). A QWidget wrapper is what makes the whole row hide/show as one
        # unit -- a bare QHBoxLayout (what _build_filter_row returns) has no setVisible of its own.
        self.filter_row_widget = QWidget()
        self.filter_row_widget.setLayout(self._build_filter_row())
        self.filter_row_widget.setVisible(False)
        bottom_layout.addWidget(self.filter_row_widget)

        self.history_status_label = QLabel()
        self.history_status_label.setStyleSheet("color: gray;")
        bottom_layout.addWidget(self.history_status_label)

        self._history_rows: list[HistoryRowWidget] = []

        self.history_scroll_area = QScrollArea()
        self.history_scroll_area.setWidgetResizable(True)
        default_history_size = self._measure_history_list_size(_DEFAULT_VISIBLE_HISTORY_ROWS)
        min_history_size = self._measure_history_list_size(_MIN_VISIBLE_HISTORY_ROWS)
        self.history_scroll_area.setMinimumSize(min_history_size)
        # Extra height beyond the enforced minimum that a fresh install (no saved window geometry
        # yet) should pad the window out by, so it still opens showing _DEFAULT_VISIBLE_HISTORY_ROWS
        # rows instead of just the _MIN_VISIBLE_HISTORY_ROWS floor -- see _restore_window_state.
        self._default_history_extra_height = default_history_size.height() - min_history_size.height()
        self.history_container = QWidget()
        self.history_layout = QVBoxLayout(self.history_container)
        self.history_layout.addStretch(1)  # keeps rows top-aligned as they're inserted above this
        self.history_scroll_area.setWidget(self.history_container)
        # Loads the next chunk of scans as the operator scrolls near the bottom -- see
        # _on_history_scrolled/_load_more_rows.
        self.history_scroll_area.verticalScrollBar().valueChanged.connect(self._on_history_scrolled)
        bottom_layout.addWidget(self.history_scroll_area, stretch=1)  # fills the bottom pane's space and grows/shrinks with it

        self.top_bottom_splitter.addWidget(top_widget)
        self.top_bottom_splitter.addWidget(bottom_widget)
        self.top_bottom_splitter.setCollapsible(0, False)
        self.top_bottom_splitter.setCollapsible(1, False)
        self.top_bottom_splitter.splitterMoved.connect(self._on_splitter_moved)

        self._top_pane_widget = top_widget
        self._top_pane_height: int | None = None  # learned from top_widget's natural sizeHint on first resizeEvent

        self._build_status_bar()

    def _build_status_bar(self) -> None:
        """Build the window's status bar: scan-count/disk-usage stats on the left, a Search toggle and a Settings button on the right."""
        # QStatusBar draws a sunken-border frame around every widget added via addWidget/
        # addPermanentWidget by default, which shows up as a vertical divider line next to each
        # one -- this app doesn't want those dividers, so they're switched off here.
        self.statusBar().setStyleSheet("QStatusBar::item { border: none; }")

        self.status_stats_label = QLabel()
        self.statusBar().addWidget(self.status_stats_label)

        # Checkable so its own pressed/sunken look doubles as the "search panel is open" indicator,
        # with no extra state to keep in sync -- see _on_search_toggle_clicked/filter_row_widget.
        self.search_toggle_button = QPushButton("Search")
        self.search_toggle_button.setCheckable(True)
        self.search_toggle_button.toggled.connect(self._on_search_toggle_clicked)
        self.statusBar().addPermanentWidget(self.search_toggle_button)

        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._on_settings_clicked)
        self.statusBar().addPermanentWidget(settings_button)

        self._recompute_storage_bytes()
        self._refresh_status_bar_stats()

    def _on_search_toggle_clicked(self, checked: bool) -> None:
        """Show or hide the history search/filter row -- see filter_row_widget, hidden by default so it doesn't take up space until the operator actually wants it."""
        self.filter_row_widget.setVisible(checked)
        if checked:
            self.search_edit.setFocus()

    def _recompute_storage_bytes(self) -> None:
        """Walk the faces/documents directories to get the real on-disk total, and cache it.

        This is the one place that pays calculate_scan_storage_bytes' real filesystem walk -- at
        startup, and after a bulk change whose exact freed/added byte count isn't known here (a
        purge, via _on_settings_data_changed). Everywhere else that image files are added or
        removed one scan at a time (a fresh capture, a single delete), the cache is adjusted
        directly instead -- see _add_to_cached_storage_bytes.
        """
        self._cached_storage_bytes = calculate_scan_storage_bytes(FACE_IMAGE_DIR, DOC_IMAGE_DIR)

    def _add_to_cached_storage_bytes(self, delta: int) -> None:
        """Adjust the cached on-disk total by a known delta (positive for newly-written files, negative for just-deleted ones) instead of re-walking the filesystem to find the new total."""
        self._cached_storage_bytes = max(0, self._cached_storage_bytes + delta)

    def _refresh_status_bar_stats(self) -> None:
        """Display the total scan count, cached on-disk size of saved scan images, and a low-disk-space warning if applicable.

        Reads the cached storage total rather than recomputing it -- see _recompute_storage_bytes/
        _add_to_cached_storage_bytes -- since this runs after nearly every operator action and a
        full filesystem walk on each of those was a real, easily-avoided cost that only grows with
        total accumulated scan history.
        """
        scan_count = count_scans(self.conn)
        storage_mb = self._cached_storage_bytes / (1024 * 1024)
        stats_text = f"{scan_count} scan(s)  ·  {storage_mb:.1f} MB on disk"

        free_mb = shutil.disk_usage(FACE_IMAGE_DIR).free / (1024 * 1024)
        warning_threshold_mb = get_disk_space_warning_mb(self.conn)
        if free_mb < warning_threshold_mb:
            stats_text += f"  ·  ⚠ Low disk space: {free_mb:,.0f} MB free"
            self.status_stats_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.status_stats_label.setStyleSheet("")

        self.status_stats_label.setText(stats_text)

    def _on_settings_clicked(self) -> None:
        """Open the Settings window, creating it once and reusing it on subsequent clicks."""
        logger.info("Operator opened the Settings window")
        if self._settings_window is None:
            self._settings_window = SettingsWindow(
                self.conn,
                self.camera_manager,
                self.face_analysis_app,
                on_data_changed=self._on_settings_data_changed,
                on_capture_shortcut_changed=self._on_capture_shortcut_changed,
                on_camera_probe_started=self._pause_camera_previews,
                on_camera_probe_finished=self._resume_camera_previews,
            )
        self._settings_window.show()
        self._settings_window.raise_()
        self._settings_window.activateWindow()

    def _pause_camera_previews(self) -> None:
        """Stop the live-preview poll and both cameras' background reader threads -- see SettingsWindow's on_camera_probe_started docstring for why this is handed to it as a callback rather than it reaching into preview_timer/camera_manager directly.

        Pausing camera_manager's reader threads (see capture.camera_manager._CameraReaderThread)
        matters now, not just the QTimer: Settings' resolution probing/diagnostics export/exposure
        changes access face_capture/doc_capture directly, which would otherwise race against those
        background threads' own concurrent capture.read() calls on the same handle.
        """
        self.preview_timer.stop()
        self.camera_manager.pause_readers()

    def _resume_camera_previews(self) -> None:
        self.camera_manager.resume_readers()
        self.preview_timer.start(_PREVIEW_INTERVAL_MS)

    def _on_settings_data_changed(self) -> None:
        """Reload the history list and refresh the status bar stats after the settings window purges scans out from under it."""
        self._reload_history()
        # A purge can remove an arbitrary, not-known-here number of files -- unlike a single
        # capture/delete, there's no cheap exact delta to apply, so this is the other place (along
        # with startup) that pays for a real recount.
        self._recompute_storage_bytes()
        self._refresh_status_bar_stats()

    def _reload_history(self) -> None:
        """Discard every currently-displayed history row and reload the full list fresh from the database (see _load_history, which rebuilds from the front).

        Also clears the batch-export selection: a purge/delete/re-analyze can change or remove
        exactly the rows a selection refers to, so starting fresh (every checkbox unchecked) is
        safer than trying to carry a possibly-stale selection across a full reload.
        """
        self._selected_scan_ids.clear()
        self._load_history()

    def _on_capture_shortcut_changed(self, sequence_text: str) -> None:
        """Re-key the capture keyboard shortcut after it's changed and saved in the settings window."""
        self.capture_shortcut.setKey(QKeySequence(sequence_text))

    def _build_filter_row(self) -> QHBoxLayout:
        """Build the history filter controls: a capture-date filter, a match-result filter, and a free-text name search box."""
        filter_row = QHBoxLayout()

        filter_row.addWidget(QLabel("Date:"))
        self.date_filter_checkbox = QCheckBox("Any date")
        self.date_filter_checkbox.setChecked(True)
        self.date_filter_checkbox.toggled.connect(self._on_date_filter_checkbox_toggled)
        filter_row.addWidget(self.date_filter_checkbox)

        self.date_filter_edit = _ClickAnywhereDateEdit(QDate.currentDate())
        self.date_filter_edit.setCalendarPopup(True)
        # Deliberately left enabled even while "Any date" is checked: a disabled widget swallows
        # the click that would open its calendar dropdown entirely, forcing the operator to
        # uncheck the box first and click a second time. Instead, ResultAwareCalendarWidget's
        # on_shown hook unchecks the box the instant the popup actually opens, so one click both
        # opens the calendar and puts the date picker into effect.
        self.date_filter_edit.setCalendarWidget(
            ResultAwareCalendarWidget(self._scan_result_dates, on_shown=self._on_date_calendar_shown)
        )
        self.date_filter_edit.dateChanged.connect(self._on_filters_changed)
        self._date_filter_opacity_effect = QGraphicsOpacityEffect(self.date_filter_edit)
        self.date_filter_edit.setGraphicsEffect(self._date_filter_opacity_effect)
        self._update_date_filter_dim_state()
        filter_row.addWidget(self.date_filter_edit)

        filter_row.addWidget(QLabel("Result:"))
        self.match_filter_combo = QComboBox()
        self.match_filter_combo.addItems(["All", "Match", "No Match"])
        self.match_filter_combo.currentTextChanged.connect(self._on_filters_changed)
        filter_row.addWidget(self.match_filter_combo)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search name...")
        self.search_edit.textChanged.connect(self._on_filters_changed)
        filter_row.addWidget(self.search_edit, stretch=1)

        clear_filters_button = QPushButton("Clear filters")
        clear_filters_button.clicked.connect(self._on_clear_filters_clicked)
        filter_row.addWidget(clear_filters_button)

        self.export_selected_button = QPushButton("Export selected")
        self.export_selected_button.setToolTip("Check scans in the list below (checkbox next to each row) to batch-export them as one combined report.")
        self.export_selected_button.setEnabled(False)
        self.export_selected_button.clicked.connect(self._on_export_selected_clicked)
        filter_row.addWidget(self.export_selected_button)

        return filter_row

    def _on_date_filter_checkbox_toggled(self, checked: bool) -> None:
        """Dim the date picker to match whether the 'any date' checkbox is checked, then re-apply filters."""
        self._update_date_filter_dim_state()
        self._on_filters_changed()

    def _update_date_filter_dim_state(self) -> None:
        """Set the date picker's opacity to reflect whether it's currently in effect (unchecked) or not (checked)."""
        is_dimmed = self.date_filter_checkbox.isChecked()
        self._date_filter_opacity_effect.setOpacity(_DATE_FILTER_DIMMED_OPACITY if is_dimmed else 1.0)

    def _on_date_calendar_shown(self) -> None:
        """Uncheck 'Any date' as soon as its calendar dropdown is opened, so picking a date takes effect immediately without a separate uncheck step."""
        self.date_filter_checkbox.setChecked(False)

    def _on_clear_filters_clicked(self) -> None:
        """Reset every filter control to its default (unfiltered) state and refresh the visible rows."""
        self.date_filter_checkbox.setChecked(True)
        self.date_filter_edit.setDate(QDate.currentDate())
        self.match_filter_combo.setCurrentText("All")
        self.search_edit.clear()
        self._on_filters_changed()

    def _on_filters_changed(self) -> None:
        """A filter control changed -- recompute the filtered scan list and reload the history list from its front (whatever was previously loaded/scrolled-to is meaningless against a new filter)."""
        self._recompute_filtered_scans()
        self._reset_loaded_rows()

    def _recompute_filtered_scans(self) -> None:
        """Recompute self._filtered_scans from self._all_scans using the current filter controls' state.

        Pure computation over already-fetched records, no DB query and no widget rendering (see
        _reset_loaded_rows/_load_more_rows for that) -- this is what lets a filter/search reach
        every saved scan, not just whichever ones happen to be loaded so far.
        """
        match_filter = self.match_filter_combo.currentText()
        search_text = self.search_edit.text().strip().lower()
        date_filter_active = not self.date_filter_checkbox.isChecked()
        selected_date = self.date_filter_edit.date().toPyDate()

        filtered: list[ScanRecord] = []
        for scan in self._all_scans:
            if date_filter_active and datetime.fromisoformat(scan.created_at).date() != selected_date:
                continue
            if match_filter == "Match" and scan.match_result not in ("match", "override_match"):
                continue
            if match_filter == "No Match" and scan.match_result not in ("no_match", "override_no_match"):
                continue
            if search_text:
                display_fields = _derive_display_fields(scan.doc_fields_json)
                haystack = f"{display_fields.first_name} {display_fields.last_name}".lower()
                if search_text not in haystack:
                    continue
            filtered.append(scan)
        self._filtered_scans = filtered

    def _reset_loaded_rows(self) -> None:
        """Tear down every currently-built history row and reload from the front of self._filtered_scans -- used on a full reload and whenever the filter controls change."""
        for row in self._history_rows:
            row.setParent(None)
            row.deleteLater()
        self._history_rows.clear()
        self._loaded_count = 0
        self._load_more_rows()
        self._fill_viewport_with_rows()

    def _load_more_rows(self) -> None:
        """Build and append the next chunk (self._history_page_size scans) of self._filtered_scans below whatever's already loaded.

        Called once for the initial load, again whenever the operator scrolls near the bottom (see
        _on_history_scrolled), and again by _fill_viewport_with_rows if the loaded rows don't yet
        fill the visible viewport. A no-op once every filtered scan is already loaded.
        """
        if self._loaded_count >= len(self._filtered_scans):
            return
        next_chunk = self._filtered_scans[self._loaded_count : self._loaded_count + self._history_page_size]
        for scan in next_chunk:
            self._append_history_row(scan, all_scans=self._all_scans)
        self._loaded_count += len(next_chunk)
        self._update_history_status_label()

    def _fill_viewport_with_rows(self) -> None:
        """Keep loading additional chunks until either every filtered scan is loaded or the loaded rows already overflow the visible viewport (i.e. a scrollbar actually exists).

        Without this, a tall window/screen could end up showing fewer rows than actually fit on
        screen with no way to trigger loading more: _on_history_scrolled only ever fires once
        there's something to scroll, which never happens if the loaded content doesn't yet exceed
        the viewport height in the first place. guard is just a sane upper bound in case a screen
        somehow never reports a scrollable range -- avoids ever looping forever.
        """
        guard = 0
        while (
            self._loaded_count < len(self._filtered_scans)
            and self.history_scroll_area.verticalScrollBar().maximum() == 0
            and guard < 50
        ):
            self._load_more_rows()
            guard += 1

    def _on_history_scrolled(self, value: int) -> None:
        """Load the next chunk once the operator has scrolled near the bottom of the currently-loaded rows -- see _load_more_rows."""
        scrollbar = self.history_scroll_area.verticalScrollBar()
        if value >= scrollbar.maximum() - _HISTORY_SCROLL_LOAD_MORE_THRESHOLD_PX:
            self._load_more_rows()

    def _update_history_status_label(self) -> None:
        """Refresh the small status line below the filter row showing how much of the filtered scan list is currently loaded."""
        total = len(self._filtered_scans)
        if self._loaded_count >= total:
            self.history_status_label.setText(f"Showing all {total} scan(s)")
        else:
            self.history_status_label.setText(f"Showing {self._loaded_count} of {total} scan(s) -- scroll down for more")

    def _scan_result_dates(self) -> set[date]:
        """Return the distinct calendar dates that have at least one saved scan, for ResultAwareCalendarWidget to highlight.

        Reads self._all_scans (every saved scan) rather than self._history_rows (only the rows
        currently loaded/built) -- otherwise the calendar would only ever highlight dates among
        whatever's been scrolled into view so far.
        """
        return {datetime.fromisoformat(scan.created_at).date() for scan in self._all_scans}

    def _measure_history_list_size(self, visible_rows: int) -> QSize:
        """Measure a representative history row to pick a scan-list size showing exactly `visible_rows` rows.

        Building a throwaway row with realistic-length sample text (rather than guessing pixel
        values) is what keeps this accurate if fonts/styling change later. Sized `visible_rows`
        rows tall and one row's natural width wide, plus room for the scrollbar itself, so a
        freshly-opened window shows a full row without a horizontal scrollbar. The extra 15% width
        headroom (beyond the scrollbar's own pixel extent) covers font-metric differences between
        environments -- the measured sizeHint() here is a few percent narrower than an identical
        row actually renders at on a real Windows install, which was otherwise just barely enough
        to trigger a horizontal scrollbar.
        """
        # all_scans=[] -- this row is never shown, so its previous-visits lookup result is
        # irrelevant; skips what would otherwise be a real full-table query purely to size a
        # throwaway widget, on every window construction.
        sample_row = HistoryRowWidget(self.conn, _SAMPLE_SCAN_FOR_SIZING, all_scans=[])
        row_size = sample_row.sizeHint()
        sample_row.deleteLater()
        scrollbar_extent = self.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
        width = int(row_size.width() * 1.15) + scrollbar_extent
        return QSize(width, row_size.height() * visible_rows)

    def _build_preview_label(self, label_class: type[QLabel] = _FlashCapableLabel) -> QLabel:
        """Build one live preview label that expands to fill whatever space its pane currently has.

        A minimum size (rather than a fixed one) keeps it from collapsing to nothing when the
        splitter handle above is dragged small, while still letting it grow -- _update_previews
        rescales each incoming frame to the label's current size on every tick. label_class lets
        the document preview use _QualityOverlayLabel (adding the live quality-warning banner on
        top of the capture-flash overlay both preview labels get from _FlashCapableLabel) while the
        face preview only needs the plain flash-capable base, without duplicating either setup.
        """
        label = label_class("No camera signal")
        label.setMinimumSize(*_PREVIEW_SIZE)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setFrameShape(QFrame.Shape.Box)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _build_preview_column(self, headline_text: str, preview_label: QLabel) -> tuple[QVBoxLayout, QLabel]:
        """Stack a centered, bold headline (with a small always-visible camera-status badge beside it) above one camera preview label. Returns the column layout and the status badge label -- kept updated every preview tick by _set_camera_status_badge, called from _update_previews alongside its existing signal-lost-streak bookkeeping."""
        column = QVBoxLayout()

        headline_row = QHBoxLayout()
        headline_row.addStretch(1)
        headline = QLabel(headline_text)
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Deliberately no hardcoded color here -- leaving it unset lets the label inherit the
        # current QPalette's WindowText color (see ui/theme.py), which stays readable under both
        # light and dark themes and updates live if the operator switches themes, unlike a
        # hardcoded "color: white" (the previous approach, unreadable under a light theme since
        # this label sits directly on the window background with no fixed backdrop).
        headline.setStyleSheet("font-size: 16pt; font-weight: bold;")
        headline_row.addWidget(headline)
        status_badge = QLabel()
        status_badge.setStyleSheet("font-size: 9pt; font-weight: bold; margin-left: 6px;")
        headline_row.addWidget(status_badge)
        headline_row.addStretch(1)
        column.addLayout(headline_row)
        column.addWidget(preview_label, stretch=1)

        return column, status_badge

    def _build_quality_warning_label(self) -> QLabel:
        """Build the (initially hidden) semi-transparent banner overlaid on top of the live document preview when the camera frame looks too low-resolution or too blurry for reliable OCR -- see _update_doc_quality_warning.

        Attached to the preview label itself via _QualityOverlayLabel.set_quality_overlay rather
        than placed in the surrounding layout, so it paints over the live feed instead of
        displacing the boxed preview below it.
        """
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet(
            "color: white; background-color: rgba(211, 47, 47, 175); font-weight: bold; padding: 4px;"
        )
        label.setVisible(False)
        return label

    def _load_history(self) -> None:
        """Fetch every saved scan from the database and (re)build the history list from its first chunk, ready to load more as the operator scrolls -- see _load_more_rows/_on_history_scrolled.

        Fetches the full scan list once up front (self._all_scans) and hands it to every row,
        rather than letting each displayed row independently run its own full-table query to find
        its own previous visits (HistoryRowWidget._compute_previous_scans) -- that used to make
        every history load do O(n) redundant full-table scans for n displayed rows. The same
        self._all_scans also backs the filter row (_recompute_filtered_scans) and the calendar's
        scan-date highlighting (_scan_result_dates), so only rows for whatever's actually been
        loaded/scrolled-to are ever built as widgets -- see _reset_loaded_rows for why that matters
        (HistoryRowWidget construction is a real per-image decode cost, not a cheap layout op).

        Also re-reads the load-batch size from settings, so a change made in Settings (General tab)
        takes effect on the next reload without needing a restart.
        """
        self._all_scans = list_scans(self.conn, limit=100_000)
        self._rebuild_person_index()
        self._history_page_size = get_history_page_size(self.conn)
        self._recompute_filtered_scans()
        self._reset_loaded_rows()

    def _rebuild_person_index(self) -> None:
        """Rebuild self._person_index from self._all_scans -- call after any mutation of self._all_scans (a full reload, a fresh capture inserted, a scan deleted, a re-analysis changing a scan's own fields) so every row's cheap indexed previous-visit lookup stays in sync with what's actually loaded. See self._person_index's own comment for why this exists."""
        self._person_index = build_person_index(self._all_scans)

    def _build_history_row(self, scan: ScanRecord, all_scans: list[ScanRecord] | None) -> HistoryRowWidget:
        """Construct one HistoryRowWidget with every standard callback wired up -- shared by _add_history_row (a brand new row) and _replace_history_row (rebuilding an existing row in place), so the two can never drift apart in which callbacks they pass."""
        return HistoryRowWidget(
            self.conn,
            scan,
            face_analysis_app=self.face_analysis_app,
            on_delete=self._on_history_row_deleted,
            on_flagged_persons_changed=self._on_flagged_persons_changed,
            on_jump_to_scan=self._jump_to_scan,
            on_reanalyze=self._on_reanalyze_scan_requested,
            on_export=self._on_export_scan_requested,
            on_selection_toggled=self._on_history_row_selection_toggled,
            all_scans=all_scans,
            person_index=self._person_index,
        )

    def _add_history_row(self, scan: ScanRecord, all_scans: list[ScanRecord] | None = None) -> None:
        """Insert one scan as the new top-most row of the history list -- used only for a brand new capture (see _finish_save_capture), which is always the newest scan on record and so always belongs above every currently-loaded row. Loading a chunk of older scans (initial load, scroll-triggered, or filling the viewport) goes through _append_history_row instead, which adds below whatever's already loaded rather than above it.

        all_scans, if the caller already has the full scan list in hand, is forwarded to
        HistoryRowWidget to avoid a redundant query -- see its docstring.
        """
        row = self._build_history_row(scan, all_scans)
        self.history_layout.insertWidget(0, row)
        self._history_rows.append(row)

    def _append_history_row(self, scan: ScanRecord, all_scans: list[ScanRecord] | None = None) -> None:
        """Build one HistoryRowWidget and add it below every currently-loaded row (just above the trailing layout stretch, see _build_layout's `history_layout.addStretch(1)`) -- see _load_more_rows, which always feeds scans in newest-first order, so appending them in that same order naturally keeps the on-screen list newest-to-oldest top-to-bottom without needing to reverse anything.

        Restores this row's export-selection checkbox if the scan is already part of the batch-
        export selection (self._selected_scan_ids), so scrolling to load more rows never visually
        loses a selection made earlier -- see _on_history_row_selection_toggled.
        """
        row = self._build_history_row(scan, all_scans)
        self.history_layout.insertWidget(self.history_layout.count() - 1, row)
        self._history_rows.append(row)
        if scan.id in self._selected_scan_ids:
            row.export_selection_checkbox.setChecked(True)

    def _refresh_same_person_rows(self, reference_scan: ScanRecord, all_scans: list[ScanRecord]) -> None:
        """Rebuild in place every *other* already-displayed row that represents the same person as reference_scan, so their visit-count badge/previous-visits list picks up whatever changed -- a new visit added (reference_scan is the just-added scan, still present in all_scans) or one removed (reference_scan is the just-deleted scan, already absent from all_scans -- see _on_history_row_deleted).

        Without this, a person's other row(s) keep showing whatever visit count was true at the
        moment they were originally built -- e.g. their very first visit's row never gained a
        "Visit 1 of N" badge at all (built with no prior visits on record, so
        HistoryRowWidget._build_layout never added the badge in the first place -- see its
        `if self._previous_scans_cache:` guard) even after a repeat visit made one of these,
        because that first row is never touched again by the single-row insert _add_history_row
        does for an ordinary capture. Deliberately targeted (only this person's rows, rebuilt in
        place) rather than a full _reload_history(), which would tear down and rebuild the whole
        list and reset the scroll position -- this runs on every single capture and delete, common
        actions where doing that would cost far more (and be far more disruptive) than the handful
        of rows either one actually touches.
        """
        if not reference_scan.doc_fields_json:
            return
        doc_fields = json.loads(reference_scan.doc_fields_json)
        if not doc_fields:
            return
        same_person_scans = find_previous_scans(
            doc_fields, [row.scan for row in self._history_rows], exclude_scan_id=reference_scan.id
        )
        if not same_person_scans:
            return
        same_person_ids = {scan.id for scan in same_person_scans}
        for stale_row in [row for row in self._history_rows if row.scan.id in same_person_ids]:
            self._replace_history_row(stale_row, all_scans)

    def _replace_history_row(
        self, old_row: HistoryRowWidget, all_scans: list[ScanRecord], scan: ScanRecord | None = None
    ) -> HistoryRowWidget:
        """Swap one already-displayed HistoryRowWidget for a freshly-built one at the same position in the list, e.g. to pick up a visit-count change -- see _refresh_same_person_rows. `scan`, if given, replaces old_row's own (stale) scan record -- e.g. _finish_reanalyze, where the underlying record itself changed, not just some other row's view of it -- otherwise old_row.scan is reused unchanged."""
        layout_index = self.history_layout.indexOf(old_row)
        self.history_layout.removeWidget(old_row)
        old_row.setParent(None)
        old_row.deleteLater()

        new_row = self._build_history_row(scan if scan is not None else old_row.scan, all_scans)
        self.history_layout.insertWidget(layout_index, new_row)
        self._history_rows[self._history_rows.index(old_row)] = new_row
        return new_row

    def _on_history_row_selection_toggled(self, scan: ScanRecord, checked: bool) -> None:
        """Track/untrack one scan's id in the batch-export selection as its row checkbox is toggled, and refresh the Export selected button's count/enabled state."""
        if checked:
            self._selected_scan_ids.add(scan.id)
        else:
            self._selected_scan_ids.discard(scan.id)
        self._update_export_selected_button()

    def _update_export_selected_button(self) -> None:
        count = len(self._selected_scan_ids)
        self.export_selected_button.setText(f"Export selected ({count})" if count else "Export selected")
        self.export_selected_button.setEnabled(count > 0)

    def _on_export_selected_clicked(self) -> None:
        """Batch-export every scan the operator checked via its row checkbox, newest first, as one combined report -- see _export_scans.

        Resolved from self._all_scans by id rather than by filtering self._history_rows: a
        selection can outlive the row that was checked (e.g. a filter change reloads the list from
        its front without clearing the selection -- see _on_filters_changed/_reset_loaded_rows), so
        only whatever's currently loaded/built as a widget would no longer be enough to recover
        every selected scan.
        """
        scans_by_id = {scan.id: scan for scan in self._all_scans}
        selected_scans = [scans_by_id[scan_id] for scan_id in self._selected_scan_ids if scan_id in scans_by_id]
        selected_scans.sort(key=lambda scan: scan.created_at, reverse=True)
        self._export_scans(selected_scans)

    def _on_export_scan_requested(self, scan: ScanRecord, previous_scans: list[ScanRecord]) -> None:
        """Handle a HistoryRowWidget's Export report button: if this person has previous visits on record (see document/person_matcher.py), offer to bundle them into one combined report before delegating to the shared _export_scans save flow."""
        scans_to_export = [scan]
        if previous_scans:
            answer = QMessageBox.question(
                self,
                "Export scan report",
                f"This person has {len(previous_scans)} previous visit(s) on record.\n\n"
                "Include them in this export as one combined report?",
            )
            if answer == QMessageBox.StandardButton.Yes:
                scans_to_export.extend(previous_scans)
                scans_to_export.sort(key=lambda scan: scan.created_at, reverse=True)
        self._export_scans(scans_to_export)

    def _export_scans(self, scans: list[ScanRecord]) -> None:
        """Shared save-dialog + file-write flow behind every export path (single-scan, previous-visits-bundled, and checkbox-driven batch): exactly one scan uses the plain single-scan report, more than one uses the combined batch report, so the two paths can never drift apart from each other."""
        if not scans:
            return

        entries = []
        for scan in scans:
            display_fields = _derive_display_fields(scan.doc_fields_json)
            display_name = f"{display_fields.first_name} {display_fields.last_name}".strip()
            entries.append(
                (
                    scan,
                    display_name,
                    display_fields.birth_date_display,
                    display_fields.age_display,
                    display_fields.personal_number_display,
                )
            )

        if len(entries) == 1:
            scan = entries[0][0]
            report_html = build_scan_report_html(*entries[0])
            default_name = f"kontroll_scan_{scan.id}_{datetime.fromisoformat(scan.created_at).strftime('%Y%m%d_%H%M%S')}.html"
        else:
            report_html = build_batch_report_html(entries)
            default_name = f"kontroll_batch_export_{len(entries)}_scans_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"

        file_path, _ = QFileDialog.getSaveFileName(self, "Export scan report", default_name, "HTML files (*.html)")
        if not file_path:
            return

        Path(file_path).write_text(report_html, encoding="utf-8")
        logger.info("Operator exported %d scan(s) to %s", len(entries), file_path)
        QMessageBox.information(self, "Export scan report", f"Saved to:\n{file_path}")

    def _on_flagged_persons_changed(self) -> None:
        """Reload the history list (so flag badges/red-row tinting and visit counters reflect the change) and refresh the Settings window's Flagged Users tab if it's currently open, after a flagged-persons change made from a scan's own detail section."""
        self._reload_history()
        if self._settings_window is not None:
            self._settings_window.refresh_flagged_persons_list()

    def _on_reanalyze_scan_requested(self, scan: ScanRecord) -> None:
        """Re-run the face-match + document analysis pipeline (see _analyze_capture) against a saved scan's own stored images, and persist whichever fields come out different -- e.g. after an OCR engine update, a newly added document parser, or a recalibrated face-match threshold.

        Never silently discards an operator's own prior decision: an existing manual match-result
        override is left untouched (only a still-"Automatic" result is refreshed), and an existing
        flag on this scan (whether auto-detected at capture time or retroactively applied -- see
        ui/flag_retroactive.py) is kept even if this fresh pass doesn't independently re-derive it,
        so a temporary OCR regression can't silently clear a real flag. A genuine removal (the
        flagged-persons entry itself deleted) is unaffected by this, since there's then nothing
        left in the watch list to match against either way. Manually edited fields (see
        HistoryRowWidget._on_edit_fields_clicked), however, *are* overwritten -- re-analysis is
        explicitly about refreshing extracted data, and the confirmation prompt says so.
        """
        if not (scan.face_image_path and Path(scan.face_image_path).exists()):
            QMessageBox.warning(self, "Re-analyze", "This scan's saved face image can no longer be found on disk.")
            return
        if not (scan.doc_image_path and Path(scan.doc_image_path).exists()):
            QMessageBox.warning(self, "Re-analyze", "This scan's saved document image can no longer be found on disk.")
            return
        if self._is_processing:
            # Shares _is_processing with the main capture flow (see _start_capture_analysis's
            # docstring) rather than having its own separate flag -- both ultimately call into the
            # one shared face_analysis_app model, which shouldn't be asked to run two analyses on
            # two different background threads at once.
            QMessageBox.information(
                self, "Re-analyze", "A capture or another re-analysis is already running -- try again once it finishes."
            )
            return

        confirm = QMessageBox.question(
            self,
            "Re-analyze scan",
            "Re-run face-match and document analysis on this scan's saved images with the current "
            "engine? Extracted fields, document validation, and (unless already manually "
            "overridden) the match result will be refreshed -- including overwriting any manual "
            "field edits with whatever this pass finds.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        face_frame = cv2.imread(scan.face_image_path)
        doc_frame = cv2.imread(scan.doc_image_path)
        if face_frame is None or doc_frame is None:
            QMessageBox.warning(self, "Re-analyze", "Could not read one of this scan's saved images from disk.")
            return

        self._is_processing = True
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self._start_capture_analysis(
            face_frame,
            doc_frame,
            on_finished=lambda analysis: self._finish_reanalyze(scan, analysis),
            on_failed=lambda: self._on_reanalyze_failed(scan),
        )

    def _on_reanalyze_failed(self, scan: ScanRecord) -> None:
        """Handle _start_capture_analysis's worker raising during a Re-analyze -- mirrors the try/except _on_reanalyze_scan_requested itself used to have directly around its (then-synchronous) _analyze_capture call."""
        QApplication.restoreOverrideCursor()
        self._is_processing = False
        logger.error("Re-analysis failed for scan %d", scan.id)
        QMessageBox.critical(self, "Re-analyze", "An unexpected error occurred while re-analyzing. See the log for details.")

    def _finish_reanalyze(self, scan: ScanRecord, analysis: CaptureAnalysis) -> None:
        """Persist whichever fields came out different and report the change, once _start_capture_analysis's background pipeline has produced a result for a Re-analyze request. See _on_reanalyze_scan_requested's docstring for the override/flag-preservation rules applied here."""
        QApplication.restoreOverrideCursor()
        self._is_processing = False

        previous_doc_fields = json.loads(scan.doc_fields_json) if scan.doc_fields_json else {}
        previous_detail = json.loads(scan.validation_detail_json) if scan.validation_detail_json else {}
        existing_flagged_match = previous_detail.get("flagged_match")

        is_already_overridden = scan.match_result in ("override_match", "override_no_match")
        new_match_result = (
            scan.match_result
            if is_already_overridden
            else _final_match_result_label(analysis.match_result.is_match if analysis.match_result else None, "Automatic")
        )

        new_detail = json.loads(analysis.validation_detail_json)
        if new_detail.get("flagged_match") is None and existing_flagged_match is not None:
            new_detail["flagged_match"] = existing_flagged_match

        update_scan_analysis(
            self.conn,
            scan.id,
            face_embedding=analysis.live_embedding.tobytes() if analysis.live_embedding is not None else None,
            match_score=analysis.match_result.similarity_score if analysis.match_result else None,
            match_result=new_match_result,
            doc_type=analysis.doc_classification.doc_type,
            doc_country=analysis.doc_classification.doc_country,
            doc_fields_json=analysis.doc_fields_json,
            validation_status=analysis.crosscheck_result.validation_status,
            validation_detail_json=json.dumps(new_detail),
        )
        logger.info("Operator re-analyzed scan %d", scan.id)

        changed_fields = sorted(set(previous_doc_fields) | set(analysis.doc_fields))
        changed_summary = [
            f"{field_name}: '{previous_doc_fields.get(field_name, '')}' → '{analysis.doc_fields.get(field_name, '')}'"
            for field_name in changed_fields
            if previous_doc_fields.get(field_name) != analysis.doc_fields.get(field_name)
        ]

        # Replace just this one row in place rather than a full _reload_history() -- a full reload
        # tears down and rebuilds the whole list from the front, which reset the scroll position
        # back up near the top (a real reported annoyance, the same class of bug already fixed for
        # deletion -- see _on_history_row_deleted's own docstring). Mirrors that same targeted
        # in-place-update pattern: refresh self._all_scans/_person_index for the changed record,
        # swap just this row (and any other same-person rows whose visit-history/badge might now
        # differ), then keep the operator's eye on it via the same center+flash used by a
        # previous-visit jump.
        updated_scan = get_scan_by_id(self.conn, scan.id)
        for index, existing_scan in enumerate(self._all_scans):
            if existing_scan.id == updated_scan.id:
                self._all_scans[index] = updated_scan
                break
        self._rebuild_person_index()
        self._recompute_filtered_scans()

        old_row = next((row for row in self._history_rows if row.scan.id == updated_scan.id), None)
        if old_row is not None:
            was_expanded = old_row.detail_toggle.isChecked()
            new_row = self._replace_history_row(old_row, self._all_scans, scan=updated_scan)
            self._refresh_same_person_rows(updated_scan, self._all_scans)
            new_row.detail_toggle.setChecked(True if changed_summary else was_expanded)
            self._center_row_in_history_view(new_row)
            self._flash_row(new_row)
        self._update_history_status_label()

        if changed_summary:
            QMessageBox.information(self, "Re-analyze scan", "Fields updated:\n" + "\n".join(changed_summary))
        else:
            QMessageBox.information(self, "Re-analyze scan", "Re-analysis complete — no field changes found.")

    def _jump_to_scan(self, scan_id: int) -> None:
        """Take the operator directly to a specific scan's own row in the history list, from a "previous visit" link in another scan's detail section (see HistoryRowWidget._build_previous_visit_link).

        Clearing every filter first (rather than, say, only setting the date filter to that visit's
        date) is what makes this an unambiguous, always-works jump straight to that one scan,
        rather than a filter that could still leave several same-day visits for the operator to
        pick back out by hand, or leave the target hidden if an unrelated match-result/search
        filter was active. Also loads however many more chunks are needed to actually reach the
        target scan (it's frequently not among whatever was already loaded, once history grows
        past one load-batch's worth of scans -- see _load_more_rows). The row is then scrolled into
        view, its detail section expanded, and briefly outlined so it's obvious which one the
        operator landed on.
        """
        self._on_clear_filters_clicked()
        target_index = next((i for i, scan in enumerate(self._filtered_scans) if scan.id == scan_id), None)
        if target_index is None:
            return

        # Reaching an old visit can mean synchronously building hundreds/thousands of rows on a
        # large production history (each _load_more_rows call is itself cheap, but there can be
        # many of them) -- a wait cursor plus periodic processEvents() keeps the app visibly
        # responsive during that catch-up instead of looking like the click did nothing (the
        # original, silently-frozen version of this loop is what a large production scan history
        # made look completely broken -- see document/person_matcher.py::PersonIndex for the other
        # half of this fix, which makes each individual row cheap to build in the first place).
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            while self._loaded_count <= target_index and self._loaded_count < len(self._filtered_scans):
                self._load_more_rows()
                QApplication.processEvents()
        finally:
            QApplication.restoreOverrideCursor()

        target_row = next((row for row in self._history_rows if row.scan.id == scan_id), None)
        if target_row is None:
            return
        target_row.detail_toggle.setChecked(True)
        self._center_row_in_history_view(target_row)
        self._flash_row(target_row)

    def _center_row_in_history_view(self, row: "HistoryRowWidget") -> None:
        """Scroll history_scroll_area so `row`'s vertical midpoint lands in the middle of the visible viewport, rather than merely scrolling the minimum distance needed to bring it into view at all (what QScrollArea.ensureWidgetVisible does, and what previously left a jumped-to row sitting at the very edge of the viewport instead of clearly in focus)."""
        container = self.history_scroll_area.widget()
        row_center_y = row.mapTo(container, QPoint(0, row.height() // 2)).y()
        scrollbar = self.history_scroll_area.verticalScrollBar()
        viewport_height = self.history_scroll_area.viewport().height()
        target = row_center_y - viewport_height // 2
        scrollbar.setValue(max(0, min(target, scrollbar.maximum())))

    def _flash_row(self, row: "HistoryRowWidget") -> None:
        """Briefly outline a history row so the operator's eye is drawn to it right after a jump, then restore its normal styling (preserving whatever flagged/unflagged tint it already had)."""
        original_stylesheet = row.styleSheet()
        row.setStyleSheet(original_stylesheet + " HistoryRowWidget { border: 3px solid #1565C0; }")

        def _restore() -> None:
            try:
                row.setStyleSheet(original_stylesheet)
            except RuntimeError:
                pass  # the row was deleted (e.g. a reload/purge happened) before the flash timer fired

        QTimer.singleShot(2000, _restore)

    def _on_history_row_deleted(self, row: HistoryRowWidget, freed_bytes: int) -> None:
        """Remove just this one scan from the history list's own bookkeeping (DB row + image files already gone by this point) -- deliberately not a full _reload_history().

        A full reload tore down and rebuilt every row from the front, which reset the scroll
        position back up near the top of a now-much-shorter list -- a real reported annoyance when
        deleting a scan while scrolled partway down. HistoryRowWidget._on_delete_clicked already
        tears down the deleted row's own widget immediately after this callback returns, so this
        only needs to drop it from self._all_scans/_filtered_scans/_history_rows/_loaded_count
        (and self._selected_scan_ids, in case it happened to be checked) so it doesn't linger as a
        stale reference -- and to refresh every other already-displayed row for the same person
        (see _refresh_same_person_rows, also used for a fresh capture) so their visit-count badge/
        previous-visits list drops the deleted scan too, without touching anything else on screen.
        """
        deleted_scan = row.scan
        self._all_scans = [scan for scan in self._all_scans if scan.id != deleted_scan.id]
        self._rebuild_person_index()
        self._filtered_scans = [scan for scan in self._filtered_scans if scan.id != deleted_scan.id]
        if row in self._history_rows:
            self._history_rows.remove(row)
            self._loaded_count = max(0, self._loaded_count - 1)
        self._selected_scan_ids.discard(deleted_scan.id)
        self._refresh_same_person_rows(deleted_scan, self._all_scans)
        self._update_history_status_label()
        self._update_export_selected_button()
        self._add_to_cached_storage_bytes(-freed_bytes)
        self._refresh_status_bar_stats()

    def _update_previews(self) -> None:
        """Poll both cameras for their latest frame and refresh the live preview labels. Runs on a timer, independent of capture -- never frozen or paused during/after a capture, since capture now saves+processes immediately in the background with no operator review step (see _run_capture_and_process)."""
        tick_started_at = time.monotonic()

        face_frame = self.camera_manager.read_face_preview_frame()
        if face_frame is not None:
            self._face_signal_lost_streak = 0
            self._set_preview_pixmap(self.face_preview_label, face_frame)
        else:
            self._face_signal_lost_streak += 1
            self._maybe_show_signal_lost(
                self.face_preview_label, self._face_signal_lost_streak, self.camera_manager.face_capture.isOpened()
            )
        self._set_camera_status_badge(
            self.face_status_badge, self._face_signal_lost_streak, self.camera_manager.face_capture.isOpened()
        )

        doc_frame = self.camera_manager.read_doc_preview_frame()
        if doc_frame is not None:
            self._doc_signal_lost_streak = 0
            self._set_preview_pixmap(self.doc_preview_label, doc_frame)
            self._update_doc_quality_warning(doc_frame)
        else:
            self._doc_signal_lost_streak += 1
            # Hide the (possibly now-stale) blur/resolution banner immediately rather than waiting
            # for the signal-lost threshold below -- a warning about the *content* of a frame we no
            # longer have is meaningless, and leaving the last one up produced exactly the
            # confusing "black screen with an 'image looks blurry' message still showing over it"
            # combination reported from a genuinely offline camera.
            self.doc_quality_warning_label.setVisible(False)
            self._doc_frame_looks_blurry = False
            self._maybe_show_signal_lost(
                self.doc_preview_label, self._doc_signal_lost_streak, self.camera_manager.doc_capture.isOpened()
            )
        self._set_camera_status_badge(
            self.doc_status_badge, self._doc_signal_lost_streak, self.camera_manager.doc_capture.isOpened()
        )

        # Reuses this tick's own already-read frames rather than letting Settings do its own
        # independent capture.read() -- see SettingsWindow.update_camera_diagnostics' docstring
        # for why a second reader on the same handle would be unsafe. No-ops immediately unless
        # Settings is open and its Cameras tab is the one currently showing.
        if self._settings_window is not None:
            self._settings_window.update_camera_diagnostics(face_frame, doc_frame)

        self._update_top_pane_height_cap(face_frame, doc_frame)
        self._warn_if_preview_tick_slow(time.monotonic() - tick_started_at)

    def _update_top_pane_height_cap(self, face_frame: np.ndarray | None, doc_frame: np.ndarray | None) -> None:
        """Cap how far down the splitter handle above the capture button can be dragged, so the operator can't stretch the previews taller than the point where the camera feed itself can still usefully grow to fill the extra height.

        _set_preview_pixmap scales each frame to fit its label's box preserving aspect ratio
        (Qt.AspectRatioMode.KeepAspectRatio) -- once the box's height exceeds what the frame's own
        aspect ratio needs for the box's current width, every bit of extra height beyond that just
        becomes blank gray padding above/below the picture (the same letterboxing an operator
        screenshot showed at startup, see _nudge_layout_to_match_actual_geometry), not a bigger
        picture. Capping self._top_pane_widget's maximumHeight at that point is what stops the
        splitter handle drag from ever reaching that dead zone in the first place -- QSplitter
        already respects a managed widget's maximumHeight when the operator drags a handle, so
        nothing else is needed to actually enforce this.

        Recomputed on every preview tick (cheap: arithmetic plus one setMaximumHeight() call,
        skipped entirely when the computed cap hasn't actually changed) so it always reflects the
        *current* camera resolution and window width -- both of which can change live (an operator
        picking a different resolution in Settings, or simply resizing the window). Falls back to
        the last frame size seen from each camera on a tick where a fresh one didn't arrive (a
        single dropped/offline tick shouldn't suddenly lift or tighten the cap), and leaves it
        unrestricted entirely until at least one real frame has been seen from *both* cameras --
        nothing to compute either one's aspect ratio from before that.
        """
        if face_frame is not None:
            self._face_frame_size = (face_frame.shape[1], face_frame.shape[0])  # (width, height)
        if doc_frame is not None:
            self._doc_frame_size = (doc_frame.shape[1], doc_frame.shape[0])
        if self._face_frame_size is None or self._doc_frame_size is None:
            return

        # For each preview: given its current on-screen width (fixed by the window's own width,
        # unaffected by the vertical splitter handle -- previews_row splits width evenly, only
        # height is what the handle controls) and its camera's native aspect ratio, how tall could
        # this preview usefully get before any more height stops growing the picture -- translated
        # into the equivalent cap on self._top_pane_widget's own total height, by adding back
        # whatever fixed overhead (headline label, capture button row, margins) isn't the preview
        # itself. That overhead is architecturally constant regardless of the current splitter
        # position, so measuring it from the *current* live geometry here is always accurate, not
        # just at the moment this happens to run.
        candidate_caps = []
        for preview_label, (frame_width, frame_height) in (
            (self.face_preview_label, self._face_frame_size),
            (self.doc_preview_label, self._doc_frame_size),
        ):
            label_width = preview_label.width()
            if label_width <= 0 or frame_width <= 0:
                continue
            max_useful_preview_height = round(label_width * frame_height / frame_width)
            overhead = self._top_pane_widget.height() - preview_label.height()
            candidate_caps.append(overhead + max_useful_preview_height)
        if not candidate_caps:
            return

        # The more restrictive (smaller) of the two previews' caps -- past that point at least one
        # of them would already be showing gray padding, even if the other could still grow a bit
        # more. Floored at the pane's own minimumSizeHint so this can never shrink it below what
        # it's structurally allowed to be.
        cap = max(min(candidate_caps), self._top_pane_widget.minimumSizeHint().height())
        # The setMaximumHeight() call itself is skipped when the cap hasn't actually changed (cheap
        # but pointless to repeat every tick), but the clamp-check below always runs regardless --
        # self._top_pane_height could need pulling down to a still-unchanged cap for a reason other
        # than the cap itself just having moved, e.g. a value restored from a previous session that
        # was saved back when the camera was configured at a larger/differently-shaped resolution.
        if cap != self._top_pane_height_cap:
            self._top_pane_height_cap = cap
            self._top_pane_widget.setMaximumHeight(cap)
        if self._top_pane_height is not None and self._top_pane_height > cap:
            self._top_pane_height = cap
            self._sync_splitter_pane_heights()

    def _warn_if_preview_tick_slow(self, elapsed_seconds: float) -> None:
        """Log (throttled) when one full preview tick -- both camera reads plus both pixmap conversions -- takes long enough to be a user-visible stutter, since this whole method runs on the main GUI thread's timer and blocks everything else (button clicks, redraws) for its duration.

        Threshold is deliberately looser than capture.camera_manager's own per-camera slow-read
        threshold (which already logs the more specific root cause when a single camera read is
        the culprit) -- this one is a catch-all for the tick as a whole, including cases where
        neither individual read was slow enough alone to trip that warning but the combination
        still added up to a noticeable stutter.
        """
        if elapsed_seconds <= _PREVIEW_INTERVAL_MS / 1000 * 2:
            self._slow_preview_tick_streak = 0
            return
        self._slow_preview_tick_streak += 1
        if self._slow_preview_tick_streak == 1 or self._slow_preview_tick_streak % 30 == 0:
            logger.warning(
                "Preview tick took %.0fms (budget ~%dms at %dfps) -- the UI may feel sluggish "
                "during this; %d consecutive slow tick(s) so far",
                elapsed_seconds * 1000,
                _PREVIEW_INTERVAL_MS,
                round(1000 / _PREVIEW_INTERVAL_MS),
                self._slow_preview_tick_streak,
            )

    def _maybe_show_signal_lost(self, label: QLabel, lost_streak: int, camera_is_opened: bool) -> None:
        """Once a camera has failed to return a frame for _SIGNAL_LOST_TICK_THRESHOLD consecutive ticks, replace its frozen last-good frame with an explicit warning rather than leaving it looking like a live feed.

        Distinguishes a camera whose handle isn't even open (camera_is_opened=False -- genuinely
        disconnected/powered off, or never available in the first place) from one that's open but
        has stopped delivering frames (a transient stall CameraManager's own reconnect logic --
        see capture.camera_manager -- is expected to recover from shortly) so the operator sees an
        accurate "offline" instead of a "reconnecting" message that will never actually resolve.
        """
        if lost_streak == _SIGNAL_LOST_TICK_THRESHOLD:
            label.clear()
            # CameraManager attempts its own automatic reopen shortly after this point (see
            # capture.camera_manager's reconnect logic) -- this streak resets to 0 the moment a
            # frame reads successfully again, which clears this warning with no operator action.
            label.setText("⚠ Camera offline" if not camera_is_opened else "⚠ Camera signal lost — reconnecting...")
            label.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 12pt;")

    def _set_camera_status_badge(self, badge: QLabel, lost_streak: int, camera_is_opened: bool) -> None:
        """Update one preview's small always-visible status badge from the same signal-lost bookkeeping _update_previews/_maybe_show_signal_lost already track every tick -- no extra per-tick cost, just a different presentation of the same numbers.

        Three states, unlike _maybe_show_signal_lost's own all-or-nothing swap (which only ever
        shows once lost_streak reaches _SIGNAL_LOST_TICK_THRESHOLD): green while frames are
        arriving normally, amber for the stretch between a first dropped read and that threshold
        (earlier visibility into a camera that's just starting to have trouble, before it's bad
        enough to blank the preview itself), red once the threshold is actually hit -- reusing the
        same opened-vs-not distinction _maybe_show_signal_lost makes so the two never disagree
        about whether a camera is "offline" vs. merely "signal lost".
        """
        if lost_streak == 0:
            badge.setText("● Live")
            badge.setStyleSheet(f"font-size: 9pt; font-weight: bold; margin-left: 6px; color: {_CAMERA_STATUS_LIVE_COLOR};")
        elif lost_streak < _SIGNAL_LOST_TICK_THRESHOLD:
            badge.setText("● Reconnecting…")
            badge.setStyleSheet(
                f"font-size: 9pt; font-weight: bold; margin-left: 6px; color: {_CAMERA_STATUS_RECONNECTING_COLOR};"
            )
        else:
            badge.setText("● Camera offline" if not camera_is_opened else "● Signal lost")
            badge.setStyleSheet(f"font-size: 9pt; font-weight: bold; margin-left: 6px; color: {_CAMERA_STATUS_LOST_COLOR};")

    def _update_doc_quality_warning(self, frame: np.ndarray) -> None:
        """Show/hide the quality banner above the document preview based on the live frame's actual resolution and sharpness.

        Only checked for the document camera, not the face camera, since OCR/MRZ/barcode reading
        is the pipeline step that actually needs resolution/sharpness (see document/
        image_quality.py's is_low_resolution/is_blurry checks, reused here via
        is_resolution_too_low_for_ocr/is_frame_too_blurry) -- face matching doesn't have an
        equivalent floor. Warning live, before any capture, lets an operator raise the resolution
        in Settings or refocus/reposition the document ahead of time, rather than only finding out
        from a banner on the saved scan after the fact -- which is exactly the gap that let a
        blurry-but-otherwise-fine-looking live feed produce a string of unreadable captures in a
        real production deployment before anyone noticed why.

        The blur check (cv2.Laplacian over the whole frame) is real, if modest, per-call cost --
        cheap enough for one shot, but not something to redo on every single ~30fps preview tick
        for both cameras. It's throttled to once every _QUALITY_CHECK_INTERVAL_TICKS ticks instead;
        blur from focus/hand-shake doesn't change fast enough for the gap to be noticeable, and the
        resolution check (just reading frame.shape, no computation) still runs every tick.
        """
        height, width = frame.shape[:2]
        if is_resolution_too_low_for_ocr(width, height):
            self.doc_quality_warning_label.setText(
                f"⚠ Low camera resolution ({width}×{height}) — may affect ID document reading"
            )
            self.doc_quality_warning_label.setVisible(True)
            return

        self._doc_quality_check_tick_count += 1
        if self._doc_quality_check_tick_count % _QUALITY_CHECK_INTERVAL_TICKS == 0:
            self._doc_frame_looks_blurry = is_frame_too_blurry(frame)

        if self._doc_frame_looks_blurry:
            self.doc_quality_warning_label.setText("⚠ Image looks blurry — may affect ID document reading")
            self.doc_quality_warning_label.setVisible(True)
        else:
            self.doc_quality_warning_label.setVisible(False)

    def _set_preview_pixmap(self, label: QLabel, frame: np.ndarray) -> None:
        """Scale a captured frame to fill as much of the label's current size as possible, preserving aspect ratio.

        Rescaling to label.size() on every tick (rather than the old fixed _PREVIEW_SIZE) is what
        lets the live feeds visibly grow or shrink as the operator drags the splitter handle above
        the capture button to resize the preview pane.
        """
        label.setStyleSheet("")  # clears any "signal lost" styling left over from _maybe_show_signal_lost
        label_size = label.size()
        frame = _shrink_to_fit(frame, label_size.width(), label_size.height())
        pixmap = _bgr_frame_to_pixmap(frame)
        label.setPixmap(
            pixmap.scaled(label_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        )

    def _on_capture_and_process_clicked(self) -> None:
        """Entry point for the capture button/shortcut: guards against re-entrant clicks, gives the operator clear busy feedback, then delegates to _run_capture_and_process.

        The capture button stays disabled (progressing from "Capturing…" to "Processing…") for the
        whole span from this click through the background analysis pipeline finishing -- see
        _run_capture_and_process/_finish_save_capture/_end_capture_processing -- but the live
        preview keeps running throughout (there is no freeze-for-review step, see
        _run_capture_and_process's own docstring), so the operator can watch the next visitor
        approach while a scan is still processing. The re-entrancy guard is what stops a click or
        shortcut press queued up mid-capture from firing a second, overlapping capture before this
        one's analysis has even started.
        """
        if self._is_processing:
            return
        self._is_processing = True
        self._capture_button_original_text = self.capture_button.text()
        self.capture_button.setEnabled(False)
        self.capture_button.setText("Capturing…")
        QApplication.processEvents()
        self._run_capture_and_process()

    def _run_capture_and_process(self) -> None:
        """Grab a still frame from both cameras and immediately kick off the face-match/OCR/document-validation pipeline on a background thread (see _start_capture_analysis) -- the scan is saved as soon as that finishes, with no operator review/crop step in between and the live preview never frozen.

        Cropping or brightness/contrast correction, when actually needed, happens after the fact:
        double-click a saved thumbnail in the history list to open the editor (see
        ui/image_viewer.py), which can also trigger a re-analysis of the corrected image (reusing
        the same _start_capture_analysis pipeline via _on_reanalyze_scan_requested) -- rather than
        blocking every single capture on an optional review step most captures don't need.

        If a previous click already got a good frame from one camera but not the other, that held
        frame (self._pending_face_frame / self._pending_doc_frame) is reused instead of recaptured
        -- this click only needs to (re)try whichever side is still missing.
        """
        logger.info("Operator triggered capture & process")
        # np.ndarray has no usable truthiness (a plain `or` raises on a multi-element array), so
        # each pending frame is checked explicitly against None rather than short-circuited.
        face_frame = (
            self._pending_face_frame
            if self._pending_face_frame is not None
            else self.camera_manager.get_face_latest_frame()
        )
        doc_frame = (
            self._pending_doc_frame
            if self._pending_doc_frame is not None
            else self._pick_sharpest_recent_doc_frame()
        )

        if face_frame is None or doc_frame is None:
            self._pending_face_frame = face_frame
            self._pending_doc_frame = doc_frame
            self.discard_pending_button.setVisible(face_frame is not None or doc_frame is not None)

            failed_cameras = []
            if face_frame is None:
                failed_cameras.append("visitor (left)")
            if doc_frame is None:
                failed_cameras.append("identification (right)")
            failed_description = " and ".join(failed_cameras)
            logger.warning("Capture failed: could not read a frame from the %s camera", failed_description)

            message = f"Could not read a frame from the {failed_description} camera."
            if face_frame is not None or doc_frame is not None:
                message += (
                    " The other camera's capture was held onto -- click Capture again to "
                    "retry just the failed one, or Discard held capture to start over."
                )
            QMessageBox.warning(self, "Capture failed", message)
            self._end_capture_processing()
            return

        self._pending_face_frame = None
        self._pending_doc_frame = None
        self.discard_pending_button.setVisible(False)
        self._flash_capture_previews()

        self.capture_button.setText("Processing…")
        self._start_capture_analysis(
            face_frame,
            doc_frame,
            on_finished=lambda analysis: self._finish_save_capture(analysis, face_frame, doc_frame),
            on_failed=self._on_save_capture_failed,
        )

    def _pick_sharpest_recent_doc_frame(self) -> np.ndarray | None:
        """Return the least-blurry of the document camera's last few buffered frames (see CameraManager.get_doc_recent_frames), not just whichever was newest at the exact instant of the click.

        A single fixed snapshot is vulnerable to catching a genuinely bad instant -- a hand still
        settling the card, a brief autofocus hunt, a momentary hologram glare -- even though a
        frame from a fraction of a second earlier or later would have OCR'd cleanly; this was a
        real, reported source of misread characters, truncated names, and OCR picking up the wrong
        (or multiple merged) printed lines, all of which blur specifically makes worse: sharper
        edges are what let Tesseract's own line/character segmentation tell lines and glyphs apart
        correctly in the first place, not just read each one more legibly. Scored with
        document.image_quality.compute_blur_score -- the same cheap (no OCR needed) Laplacian-
        variance proxy this app already uses elsewhere to flag a blurry capture -- rather than
        running the full, much more expensive OCR/MRZ/barcode pipeline on every candidate and
        comparing *that*: only the actual winner ever goes through the real pipeline, exactly as
        before, so this adds a few milliseconds of scoring, not several extra seconds of OCR.
        """
        candidates = self.camera_manager.get_doc_recent_frames()
        if not candidates:
            return None
        return max(candidates, key=compute_blur_score)

    def _flash_capture_previews(self) -> None:
        """Briefly flash both live preview panes white -- an immediate, camera-shutter-style visual cue that a capture just happened, fired the moment a still frame was actually grabbed from both cameras (not merely on the button click, which can still be waiting on one camera -- see the frame-still-missing branch above this call site). Purely cosmetic: runs synchronously and returns immediately, well before the background analysis pipeline (which can take several seconds) finishes."""
        self.face_preview_label.flash_white()
        self.doc_preview_label.flash_white()

    def _start_capture_analysis(
        self,
        face_frame: np.ndarray,
        doc_frame: np.ndarray,
        on_finished: Callable[[CaptureAnalysis], None],
        on_failed: Callable[[], None],
    ) -> None:
        """Run _analyze_capture on a background thread and deliver its result back to on_finished/on_failed on this (the main/UI) thread once it completes.

        This is what actually fixes the multi-second UI freeze on every Save/Re-analyze click: the
        OCR/face-match/MRZ/barcode pipeline is pure CPU-bound work over plain frames once it has
        what it needs, so it doesn't have to run on the Qt event-loop thread at all. The three
        values _analyze_capture used to read from the DB itself (face-match threshold,
        barcode-overlap threshold, the flagged-persons list) are fetched here instead, up front, on
        the main thread -- sqlite3 connections aren't safe to share across threads, and these are
        cheap single-table reads anyway, unlike the work being handed off.

        Callers must check self._is_processing before calling this and set it True first -- this
        method doesn't check or set it itself, since the two call sites want different busy-UI
        treatment (the main capture button vs. a wait cursor for the rarer Re-analyze action), but
        both must still serialize against each other: face_analysis_app (InsightFace) is one
        shared model instance, and calling into it concurrently from two worker threads at once
        isn't a guarantee its ONNX Runtime session makes.

        Keeps the QThread/worker alive via self._capture_analysis_thread/_worker -- nothing else
        would hold a reference to them once this method returns, and Qt does not appreciate a
        QThread being garbage-collected while still running.
        """
        face_match_threshold = get_face_match_threshold(self.conn)
        barcode_overlap_threshold = get_barcode_overlap_threshold(self.conn)
        flagged_persons = list_flagged_persons(self.conn)

        thread = QThread(self)
        worker = _CaptureAnalysisWorker(
            self.face_analysis_app, face_frame, doc_frame, face_match_threshold, barcode_overlap_threshold, flagged_persons
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)

        def _cleanup() -> None:
            thread.quit()
            thread.wait()
            self._capture_analysis_thread = None
            self._capture_analysis_worker = None

        def _handle_finished(analysis: CaptureAnalysis) -> None:
            _cleanup()
            on_finished(analysis)

        def _handle_failed(traceback_text: str) -> None:
            logger.error("Capture analysis failed on a background thread:\n%s", traceback_text)
            _cleanup()
            on_failed()

        worker.finished.connect(_handle_finished)
        worker.failed.connect(_handle_failed)

        self._capture_analysis_thread = thread
        self._capture_analysis_worker = worker
        thread.start()

    def _end_capture_processing(self) -> None:
        """Restore the capture button and clear the processing guard -- the shared tail end of every path out of _run_capture_and_process (a failed camera read, a failed background analysis, or a successfully saved scan)."""
        self.capture_button.setEnabled(True)
        self.capture_button.setText(self._capture_button_original_text)
        self._is_processing = False

    def _on_save_capture_failed(self) -> None:
        """Handle _start_capture_analysis's worker raising during a capture -- mirrors the except branch _finish_save_capture itself still has for errors in the (fast, synchronous) save/DB-write step below the analysis."""
        self._end_capture_processing()
        QMessageBox.critical(
            self,
            "Capture failed",
            "An unexpected error occurred while processing the capture. See the log for details.",
        )

    def _finish_save_capture(self, analysis: CaptureAnalysis, face_frame: np.ndarray, doc_frame: np.ndarray) -> None:
        """Save the analyzed capture and add it to the history list, once _start_capture_analysis's background pipeline has produced a result.

        Everything here that isn't itself an operator-facing decision (i.e. writing the images,
        the DB insert, and updating the UI) is wrapped in a single try/except: previously an
        exception anywhere in this chain was swallowed by main.py's sys.excepthook (logged to file
        only, no dialog, no history row added) -- which is exactly what made some clicks look like
        they silently did nothing at all.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            face_image_path = FACE_IMAGE_DIR / f"{timestamp}_face.jpg"
            doc_image_path = DOC_IMAGE_DIR / f"{timestamp}_doc.jpg"
            cv2.imwrite(str(face_image_path), face_frame)
            cv2.imwrite(str(doc_image_path), doc_frame)
            # Generated eagerly from the same in-memory frames just written above (no extra disk
            # read) so _load_thumbnail's cache-hit path covers this scan from its very first
            # history-list render, rather than falling back to its own lazy-generate-on-first-read
            # path -- see that function's own docstring.
            write_thumbnail_cache(str(face_image_path), face_frame, _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1])
            write_thumbnail_cache(str(doc_image_path), doc_frame, _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1])
            # Sized after writing rather than re-walking the whole image directory tree to find the
            # new total -- see MainWindow._add_to_cached_storage_bytes. Includes each image's
            # cached thumbnail file too, so this incremental count doesn't drift from a full
            # recount (which walks the whole directory and picks those files up regardless).
            self._add_to_cached_storage_bytes(
                face_image_path.stat().st_size
                + doc_image_path.stat().st_size
                + thumbnail_cache_path(str(face_image_path)).stat().st_size
                + thumbnail_cache_path(str(doc_image_path)).stat().st_size
            )

            match_result = analysis.match_result
            scan_id = insert_scan(
                self.conn,
                operator=None,
                face_image_path=str(face_image_path),
                doc_image_path=str(doc_image_path),
                face_embedding=analysis.live_embedding.tobytes() if analysis.live_embedding is not None else None,
                match_score=match_result.similarity_score if match_result is not None else None,
                match_result=_final_match_result_label(match_result.is_match if match_result else None, "Automatic"),
                doc_type=analysis.doc_classification.doc_type,
                doc_country=analysis.doc_classification.doc_country,
                doc_fields_json=analysis.doc_fields_json,
                validation_status=analysis.crosscheck_result.validation_status,
                validation_detail_json=analysis.validation_detail_json,
            )

            logger.info(
                "Scan %d saved: match_result=%s, validation_status=%s, doc_type=%s",
                scan_id,
                _final_match_result_label(match_result.is_match if match_result else None, "Automatic"),
                analysis.crosscheck_result.validation_status,
                analysis.doc_classification.doc_type,
            )

            new_scan = get_scan_by_id(self.conn, scan_id)
            self._all_scans.insert(0, new_scan)  # newest-first, matches list_scans' own order
            self._rebuild_person_index()
            self._recompute_filtered_scans()
            # A fresh capture is always the newest scan on record, so if it matches the current
            # filters it always belongs directly above every currently-loaded row -- insert it there
            # (cheap: one new row) rather than reloading the whole list, since this runs on every
            # single capture, the app's most frequent action. If it doesn't match the current
            # filters (e.g. an active search for a different name), self._filtered_scans is
            # unchanged and nothing here needs to touch the widget tree at all.
            if self._filtered_scans and self._filtered_scans[0].id == new_scan.id:
                self._add_history_row(new_scan, all_scans=self._all_scans)
                self._loaded_count += 1
                self._refresh_same_person_rows(new_scan, self._all_scans)
                self._update_history_status_label()
            self._refresh_status_bar_stats()
            self.statusBar().showMessage(f"✓ Scan #{scan_id} saved", 4000)

            if analysis.flag_match is not None:
                self._show_flagged_person_warning(analysis.flag_match, str(face_image_path))
        except Exception:
            logger.exception("Capture & process failed unexpectedly")
            QMessageBox.critical(
                self,
                "Capture failed",
                "An unexpected error occurred while processing the capture. See the log for details.",
            )
        finally:
            self._end_capture_processing()

    def _show_flagged_person_warning(self, flag_match, live_face_image_path: str) -> None:
        """Show a large, modal, impossible-to-miss warning dialog immediately after a capture matches an entry on the flagged-persons watch list.

        This is deliberately in addition to (not instead of) the passive red-row tint and
        FLAGGED badge in the history list -- the moment right after a fresh capture is the one
        point in the workflow where a flag being easy to overlook would matter most, so it's
        surfaced as a blocking dialog the operator must explicitly acknowledge here, rather than
        relying on them noticing the row further down the (possibly already long) history list.
        """
        flagged_person = flag_match.flagged_person
        logger.warning("Displayed flagged-person warning for flagged_person id=%d", flagged_person.id)

        dialog = QDialog(self)
        dialog.setWindowTitle("FLAGGED PERSON")
        dialog.setMinimumSize(520, 420)
        layout = QVBoxLayout(dialog)

        banner = QLabel("⚠ FLAGGED PERSON ⚠")
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setStyleSheet(
            "color: white; background-color: #B71C1C; font-size: 22pt; font-weight: bold; "
            "padding: 18px; border-radius: 4px;"
        )
        layout.addWidget(banner)

        reason = flagged_person.notes or "(no reason recorded)"
        info_label = QLabel(
            f"<b>{flagged_person.name or '(no name)'}</b><br>"
            f"Personal number: {flagged_person.personal_number or '-'}<br>"
            f"Matched on: {flag_match.matched_on}<br>"
            f"Reason: {reason}"
        )
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("font-size: 13pt; padding: 8px;")
        layout.addWidget(info_label)

        images_row = QHBoxLayout()
        images_row.addLayout(self._build_warning_image_column("Live capture", live_face_image_path))
        if flagged_person.image_path:
            images_row.addLayout(self._build_warning_image_column("Watch-list reference", flagged_person.image_path))
        layout.addLayout(images_row)

        acknowledge_button = QPushButton("Acknowledge")
        acknowledge_button.setStyleSheet("font-size: 12pt; padding: 8px;")
        acknowledge_button.clicked.connect(dialog.accept)
        layout.addWidget(acknowledge_button)

        dialog.exec()

    def _build_warning_image_column(self, title: str, image_path: str | None) -> QVBoxLayout:
        """Build one labeled, fixed-size image column for _show_flagged_person_warning -- larger than the history list's own thumbnails, since this dialog is meant to be read at a glance from a normal viewing distance."""
        column = QVBoxLayout()
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        column.addWidget(title_label)

        image_label = QLabel()
        image_label.setFixedSize(200, 150)
        pixmap = QPixmap(image_path) if image_path else None
        if pixmap is not None and not pixmap.isNull():
            image_label.setPixmap(
                pixmap.scaled(200, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            )
        else:
            image_label.setText("No image")
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setStyleSheet("border: 1px solid gray; color: gray;")
        column.addWidget(image_label)

        return column

    def _on_discard_pending_capture_clicked(self) -> None:
        """Drop a held partial capture (one camera's frame from a previous failed click) so the next click starts over from scratch on both sides."""
        self._pending_face_frame = None
        self._pending_doc_frame = None
        self.discard_pending_button.setVisible(False)
        logger.info("Operator discarded a held partial capture")

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """Remember the operator's manually-chosen top-pane height whenever they drag the splitter handle.

        Without this, the next resizeEvent would snap the top pane straight back to its original
        learned height, undoing the drag.
        """
        self._top_pane_height = self.top_bottom_splitter.sizes()[0]

    def resizeEvent(self, event: QResizeEvent | None) -> None:
        """Keep the top pane (previews + capture button) at a constant height across a plain window resize.

        The bottom pane (headline + scan list + trailing stretch) absorbs the entire resize delta
        instead. The top pane's height still changes freely when the operator drags the splitter
        handle -- _on_splitter_moved keeps self._top_pane_height in sync with that.
        """
        super().resizeEvent(event)
        self._sync_splitter_pane_heights()

    def _sync_splitter_pane_heights(self) -> None:
        """Apply self._top_pane_height (or, the first time, a freshly learned one) to the splitter -- the actual work behind resizeEvent, pulled out into its own method purely for readability.

        Learning self._top_pane_height here (from self._top_pane_widget.sizeHint(), when nothing's
        been learned or restored yet) only happens once -- every later call just reapplies whatever
        value is already cached, never recomputing it, which is normally exactly right (a resize
        shouldn't change the pinned top-pane height).
        """
        splitter = getattr(self, "top_bottom_splitter", None)
        if splitter is None:
            return
        if self._top_pane_height is None:
            self._top_pane_height = self._top_pane_widget.sizeHint().height()
        bottom_height = max(splitter.height() - splitter.handleWidth() - self._top_pane_height, 0)
        splitter.setSizes([self._top_pane_height, bottom_height])

    def _nudge_layout_to_match_actual_geometry(self) -> None:
        """Force Qt to fully redo this window's entire layout once it has actually been shown -- fixes a real reported bug where the live camera previews render letterboxed (a smaller-than-their-actual-box image, with gray padding around it) at startup, until the operator manually resizes the window, at which point the previews properly grow to fill their box.

        Diagnosed from an operator screenshot: the splitter's own top/bottom pane split, the
        headline labels, the capture button, and the history list below were all already exactly
        where they should be *before* any manual resize -- only the previews themselves were
        undersized within an already-correctly-sized box, which rules out the splitter's own
        top-pane height as the cause (a first attempt at this fix targeted that instead, gated to
        only run once on a fresh install with nothing saved yet -- confirmed, from the same
        screenshot, to *not* actually be the problem, since it reproduced on a returning operator's
        window with previously-saved geometry too).

        _set_preview_pixmap re-queries each preview label's own .size() fresh on every ~30fps tick
        (see _PREVIEW_INTERVAL_MS) to decide how large a frame to scale into it -- so the previews
        coming out undersized despite that means the label's own .size() was itself reporting a
        smaller box than what it's actually rendered at. This is a known category of Qt/PyQt6 quirk:
        a QSizePolicy.Expanding child nested inside a QSplitter can end up reporting a stale,
        smaller geometry than its true final layout allocation until something forces a real,
        whole-window resize/layout pass -- merely recomputing and re-applying a splitter pane size
        (this method's first version) observably wasn't enough to trigger that on its own.

        The fix is to reproduce a real manual resize programmatically: grow the window by a single
        pixel and immediately shrink it back. Each of those is a genuine QResizeEvent -- what
        actually forces Qt to redo geometry for every descendant widget, previews included -- and
        resizeEvent()/_sync_splitter_pane_heights() already runs as a normal side effect of that,
        so the splitter's own pane split stays correctly in sync too. The 1px growth/shrink is
        imperceptible on screen and the window ends up at exactly its original size. Scheduled via
        QTimer.singleShot(0, ...) at the end of __init__, so this runs on the very next event-loop
        iteration after main.py's window.show() call -- by which point the window is actually on
        screen, which a same-tick resize (tried first) can't rely on.
        """
        current_size = self.size()
        self.resize(current_size.width() + 1, current_size.height())
        self.resize(current_size)

    def _restore_window_state(self) -> None:
        """Restore the window's size/position and the splitter's pane split from the last session, if anything was saved.

        The top pane's height is restored as a plain pixel integer, not via QSplitter's own
        saveState()/restoreState(): this method runs before the window has ever been shown, and
        QSplitter.sizes() reports a proportionally-scaled placeholder (not the real saved value)
        when queried before the splitter has had a real layout pass -- reading that back into
        self._top_pane_height would have resizeEvent() permanently lock onto the wrong number the
        moment the window is actually shown. Storing/restoring the height directly sidesteps that
        entirely: resizeEvent() already knows how to apply self._top_pane_height once real
        geometry exists, whether it came from a user drag, a first-run default, or here.

        Falls back to the existing content-driven default sizing (adjustSize() plus the top
        pane's natural sizeHint, learned lazily by resizeEvent) the first time the app is run,
        before anything has ever been saved -- padded out by self._default_history_extra_height
        so that default sizing still shows _DEFAULT_VISIBLE_HISTORY_ROWS rows rather than just the
        smaller _MIN_VISIBLE_HISTORY_ROWS floor enforced by history_scroll_area's minimum size.
        """
        geometry_b64 = get_setting_or_none(self.conn, _WINDOW_GEOMETRY_SETTING_KEY)
        if geometry_b64 is not None:
            self.restoreGeometry(QByteArray(base64.b64decode(geometry_b64)))
        else:
            self.adjustSize()
            self.resize(self.width(), self.height() + self._default_history_extra_height)

        top_pane_height_str = get_setting_or_none(self.conn, _TOP_PANE_HEIGHT_SETTING_KEY)
        if top_pane_height_str is not None:
            self._top_pane_height = int(top_pane_height_str)

    def _save_window_state(self) -> None:
        """Persist the window's current size/position and the splitter's top-pane height, so the next launch can restore them."""
        geometry_b64 = base64.b64encode(bytes(self.saveGeometry())).decode("ascii")
        set_setting(self.conn, _WINDOW_GEOMETRY_SETTING_KEY, geometry_b64)

        if self._top_pane_height is not None:
            set_setting(self.conn, _TOP_PANE_HEIGHT_SETTING_KEY, str(self._top_pane_height))

    def closeEvent(self, event: QCloseEvent | None) -> None:
        """Save the window/splitter layout, close the settings window if open, then release both camera handles, when the window closes.

        Declines to close while a capture or re-analysis is still running on its background thread
        (see _start_capture_analysis) -- letting the window close out from under it would either
        silently lose that scan (it's saved only once the thread's result comes back) or risk
        tearing down the QThread while it's still running. Same reasoning for a camera backend
        switch still running on Settings' own background thread (see SettingsWindow.
        _on_backend_changed) -- release_all() below would race its still-in-flight release/reopen
        of the very same camera handles.
        """
        if self._is_processing:
            QMessageBox.information(
                self,
                "Please wait",
                "A capture or re-analysis is still being processed -- please wait for it to finish before closing.",
            )
            if event is not None:
                event.ignore()
            return
        if self._settings_window is not None and self._settings_window.is_backend_switch_in_progress():
            QMessageBox.information(
                self,
                "Please wait",
                "A camera backend change is still being applied -- please wait for it to finish before closing.",
            )
            if event is not None:
                event.ignore()
            return
        self._save_window_state()
        if self._settings_window is not None:
            self._settings_window.close()
        self.camera_manager.release_all()
        super().closeEvent(event)
