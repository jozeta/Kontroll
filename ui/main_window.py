"""Single-screen PyQt6 window: two live camera previews, one capture button, and a scan history list."""

import base64
import json
import logging
import shutil
import sqlite3
import traceback
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PyQt6.QtCore import QByteArray, QDate, QObject, QPoint, QRect, QSize, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import (
    QCloseEvent,
    QColor,
    QImage,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QPen,
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
    QVBoxLayout,
    QWidget,
)

from capture.camera_manager import CameraManager
from capture.frame_capture import capture_still_frame
from config.app_paths import get_documents_dir, get_faces_dir, get_flagged_images_dir
from config.settings import (
    get_barcode_overlap_threshold,
    get_capture_shortcut,
    get_disk_space_warning_mb,
    get_face_match_threshold,
    get_setting_or_none,
    set_setting,
)
from document.aamva_parser import parse_first_aamva_barcode
from document.barcode_parser import decode_pdf417_barcodes
from document.card_detector import detect_and_crop_card
from document.country_flags import alpha2_for_country_code
from document.dates import calculate_age, parse_mrz_style_date
from document.dl_field_parser import parse_driving_licence_fields
from document.doc_classifier import DocumentClassification, classify_document
from document.field_crosscheck import (
    CrosscheckResult,
    PersonFlagsResult,
    check_person_flags,
    determine_validation_status,
)
from document.flag_matcher import FlagMatch, find_matching_flagged_person
from document.image_quality import (
    ImageQualityResult,
    assess_image_quality,
    is_frame_too_blurry,
    is_resolution_too_low_for_ocr,
)
from document.mrz_locator import locate_mrz_band
from document.mrz_parser import parse_and_validate_mrz
from document.national_id_parser import parse_national_id_fields
from document.ocr import correct_text_orientation, extract_mrz_candidate_text, extract_words_and_text
from document.passport_parser import parse_passport_fields
from face.detector import detect_faces
from face.embedder import get_best_face_embedding
from face.matcher import MatchResult, match_faces
from document.person_matcher import find_previous_scans
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
    get_scan_by_id,
    insert_flagged_person,
    insert_scan,
    list_flagged_persons,
    list_scan_comments,
    list_scans,
    update_match_result,
    update_scan_analysis,
    update_scan_doc_fields,
)
from ui.flag_icons import render_flag_icon
from ui.flag_retroactive import run_retroactive_flag_check
from ui.image_viewer import ClickableThumbnailLabel
from ui.scan_export import build_batch_report_html, build_scan_report_html
from ui.settings_window import SettingsWindow

# Each independently resolved (and auto-created if missing) via config.app_paths rather than a
# path relative to this file's own location -- see that module for why (packaged-EXE safety) and
# how each is independently operator-configurable.
FACE_IMAGE_DIR = get_faces_dir()
DOC_IMAGE_DIR = get_documents_dir()
FLAGGED_IMAGE_DIR = get_flagged_images_dir()

logger = logging.getLogger(__name__)

_PREVIEW_INTERVAL_MS = 33  # ~30fps live preview polling
_PREVIEW_SIZE = (320, 240)
_THUMBNAIL_SIZE = (110, 82)

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

_OVERRIDE_CHOICES = ["Automatic", "Override: Match", "Override: No Match"]

# Color the crop-selection rectangle is drawn in over the frozen document preview (see
# DocumentCropLabel) -- bright yellow so it reads clearly against a document photo of any color.
_CROP_RECT_COLOR = QColor("#FFEB3B")

# Minimum crop-rectangle side length, in on-screen label pixels, for a drag to count as a real
# crop selection rather than an accidental single click/tiny jitter -- below this,
# MainWindow._crop_rect_to_frame_bounds treats it as no selection and the full frame is used.
_MIN_CROP_DRAG_PX = 20

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


def _shrink_to_fit(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Downscale a BGR frame (via cheap cv2.resize) to approximately fit within target_width x target_height, preserving aspect ratio -- a no-op if it's already smaller either way.

    Meant to run *before* _bgr_frame_to_pixmap's color-space conversion and QImage copy, both of
    which cost proportional to pixel count -- calling it first means those don't pay full-camera-
    resolution (or full-saved-image-resolution) cost on every live-preview tick or thumbnail load
    just to have most of that detail immediately discarded by a later scale-down to a small label.
    The final Qt-side .scaled() callers already do for exact-size fitting stays in place regardless
    (cheap once the image is already this much smaller) so this doesn't change what's displayed.
    """
    height, width = frame.shape[:2]
    if target_width <= 0 or target_height <= 0 or (width <= target_width and height <= target_height):
        return frame
    scale = min(target_width / width, target_height / height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)


def _bgr_frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    """Convert an OpenCV BGR frame into a QPixmap suitable for display in a QLabel."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_frame.shape
    bytes_per_line = channels * width
    # .copy() detaches the QImage from rgb_frame's buffer, which is a stack-local that would
    # otherwise be freed once this function returns.
    qimage = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def _load_thumbnail(image_path: str | None) -> QPixmap | None:
    """Load a saved scan image from disk and scale it down to thumbnail size for the history list."""
    if not image_path or not Path(image_path).exists():
        return None
    frame = cv2.imread(image_path)
    if frame is None:
        return None
    frame = _shrink_to_fit(frame, _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1])
    pixmap = _bgr_frame_to_pixmap(frame)
    return pixmap.scaled(
        _THUMBNAIL_SIZE[0], _THUMBNAIL_SIZE[1], Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
    )


def _match_icon_text(match_result: str | None) -> str:
    """Return a short, visually distinct glyph+label summarizing the face-match result for the history list."""
    if match_result in ("match", "override_match"):
        return "✅ MATCH"
    if match_result in ("no_match", "override_no_match"):
        return "❌ NO MATCH"
    return "➖ N/A"


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
    # The document camera frames the whole light-box/desk area, so the card itself typically
    # occupies only a modest fraction of even a high-resolution capture -- too low an effective
    # pixel density for OCR to read reliably. detect_and_crop_card finds and crops+upscales to the
    # card's own bounding box before any OCR/MRZ/barcode work (falling back to the original frame
    # untouched if no confident match is found), while face detection uses the original,
    # uncropped doc_frame -- see document/card_detector.py.
    doc_analysis_frame = detect_and_crop_card(doc_frame)
    # card_detector.py's crop only guarantees a landscape shape, not which way up it reads --
    # correct_text_orientation salvages the (otherwise-total) OCR failure that follows when its
    # fixed-direction rotation happens to leave the text sideways. Applied before every downstream
    # use of doc_analysis_frame (OCR, MRZ band location, barcode decode all follow below), not just
    # this function's own OCR pass, since all of them assume an upright frame.
    doc_analysis_frame = correct_text_orientation(doc_analysis_frame)

    # extract_words_and_text runs OCR twice against this same preprocessed image (via
    # pytesseract.image_to_data and image_to_string) -- kept as two separate Tesseract calls rather
    # than reconstructing one from the other's output, since image_to_data's per-word layout
    # doesn't reliably reproduce image_to_string's spacing, which mrz_parser/dl_field_parser depend
    # on, but sharing one preprocessing pass between them rather than each redoing it independently.
    # ocr_words' per-word confidence feeds assess_image_quality's hard-to-read-text check below;
    # ocr_text isn't needed until the MRZ parsing further down, but there's no reason to preprocess
    # doc_analysis_frame a second time just to defer computing it until then.
    ocr_words, ocr_text = extract_words_and_text(doc_analysis_frame)
    # Blur/glare/resolution are still measured against the original doc_frame (a diagnostic about
    # the raw camera capture itself, e.g. "camera silently defaulted to a low resolution mode") --
    # only the OCR-confidence-based legibility check benefits from using the cropped, higher-
    # effective-density frame the field parsers actually run on.
    doc_image_quality = assess_image_quality(doc_frame, ocr_words)
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

    mrz_result = parse_and_validate_mrz(ocr_text)
    if not mrz_result.found:
        # General-purpose OCR's two fixed-width MRZ lines can get lost entirely amid surrounding
        # printed text/security-pattern noise (diagnosed against a real passport capture where
        # this happened) -- a second, MRZ-alphabet-whitelisted OCR pass recovers it far more
        # reliably. Only tried once the cheaper general-purpose attempt has already failed, since
        # the extra Tesseract call costs ~1s and would actively hurt name-field text (which needs
        # the full alphabet, not just MRZ's). locate_mrz_band crops tightly to the MRZ itself
        # first when it can find one (a bound passport photographs as its whole open book, not
        # just the bio-data page -- see document/mrz_locator.py), falling back to the full frame
        # unchanged otherwise, so this is a safe no-op improvement for every other document type.
        mrz_band = locate_mrz_band(doc_analysis_frame)
        mrz_candidate_text = extract_mrz_candidate_text(mrz_band)
        mrz_result = parse_and_validate_mrz(mrz_candidate_text)
    barcode_texts = decode_pdf417_barcodes(doc_analysis_frame)
    doc_classification = classify_document(mrz_result, barcode_texts, ocr_text)
    crosscheck_result = determine_validation_status(mrz_result, barcode_texts, ocr_text, barcode_overlap_threshold)

    if mrz_result.found:
        doc_fields = mrz_result.fields
        if doc_classification.doc_type == "passport" and not (doc_fields.get("surname") and doc_fields.get("name")):
            # The MRZ's line 2 alone (see mrz_parser.py's TD3-line-2-only recovery) carries every
            # checksummed data field but never a name -- ICAO 9303 puts no check digit on line 1
            # (name) at all, so a corrupted line 1 is recovered from here by falling back to
            # passport_parser.py's independent, OCR-label-anchored name search instead, without
            # touching the MRZ-derived (and more trustworthy) dates/document number/personal
            # number already in doc_fields.
            passport_fields = parse_passport_fields(ocr_text).fields
            doc_fields = {**doc_fields, **{key: passport_fields[key] for key in ("surname", "name") if passport_fields.get(key)}}
    elif doc_classification.classified_from == "aamva_barcode":
        # A US/Canadian driver's licence or ID card's fields come from its AAMVA barcode payload,
        # not OCR -- see document/aamva_parser.py. Checked ahead of the doc_type branches below
        # since AAMVA can classify as either "drivers_licence" or "national_id" (see
        # document/doc_classifier.py::_classify_from_aamva_barcode) and both need this same
        # barcode-sourced extraction, not the OCR-heuristic one those branches otherwise use for
        # EU/Swedish documents.
        doc_fields = parse_first_aamva_barcode(barcode_texts).fields
    elif doc_classification.doc_type == "drivers_licence":
        # Driving licences carry no MRZ, so name/birth-date have to come from a heuristic OCR read
        # of the licence's EU-harmonized numbered fields instead (see dl_field_parser.py). Gated
        # on doc_classification rather than just "no MRZ found" -- a passport or national ID card
        # whose MRZ failed to read is still not a driving licence, and running dl_field_parser's
        # field-number-anchored regexes against arbitrary OCR noise from an unrelated document
        # layout risks extracting a spurious fake name (diagnosed against a real passport capture
        # where this produced exactly that).
        dl_fields_result = parse_driving_licence_fields(ocr_text)
        doc_fields = dl_fields_result.fields if dl_fields_result.found else {}
    elif doc_classification.doc_type == "national_id":
        # A Swedish national ID card's MRZ is printed on the back -- a front-only capture (the
        # common case at a desk) never has one to read, so name/personnummer/dates come from an
        # OCR-only heuristic read instead, same idea as the driving licence path above (see
        # document/national_id_parser.py).
        national_id_result = parse_national_id_fields(ocr_text)
        doc_fields = national_id_result.fields if national_id_result.found else {}
    elif doc_classification.doc_type == "passport":
        # The MRZ is always preferred for a passport (see the extract_mrz_candidate_text fallback
        # above) -- this only runs once that has already failed, e.g. a heavily noisy capture. See
        # document/passport_parser.py for why this needs its own parser rather than reusing
        # dl_field_parser/national_id_parser.
        passport_result = parse_passport_fields(ocr_text)
        doc_fields = passport_result.fields if passport_result.found else {}
    else:
        doc_fields = {}
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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add to Flagged Users")
        self._has_face_image = has_face_image
        self._has_doc_image = has_doc_image

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

    def __init__(self, given_name: str, surname: str, personal_number: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit Name / Personal Number")

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

    def values(self) -> tuple[str, str, str]:
        """Return (given_name, surname, personal_number) once the dialog has been accepted."""
        return self.given_name_edit.text().strip(), self.surname_edit.text().strip(), self.personal_number_edit.text().strip()


class HistoryRowWidget(QFrame):
    """One row in the scan history list: both thumbnails, extracted identity fields, the match icon, and an expandable detail/override section."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        scan: ScanRecord,
        on_delete: Callable[["HistoryRowWidget", int], None] | None = None,
        on_flagged_persons_changed: Callable[[], None] | None = None,
        on_jump_to_scan: Callable[[int], None] | None = None,
        on_reanalyze: Callable[[ScanRecord], None] | None = None,
        on_export: Callable[[ScanRecord, list[ScanRecord]], None] | None = None,
        on_selection_toggled: Callable[[ScanRecord, bool], None] | None = None,
        all_scans: list[ScanRecord] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.conn = conn
        self.scan = scan
        self._on_delete = on_delete
        self._on_flagged_persons_changed = on_flagged_persons_changed
        self._on_jump_to_scan = on_jump_to_scan
        self._on_reanalyze = on_reanalyze
        self._on_export = on_export
        self._on_selection_toggled = on_selection_toggled
        self.setFrameShape(QFrame.Shape.StyledPanel)
        # Parsed once here rather than by every individual call site that needs it (previously
        # _build_layout, the flagged-user dialog, and MainWindow._apply_filters each re-ran
        # json.loads(doc_fields_json) + date parsing independently) -- cheap per row, but adds up
        # when _apply_filters re-reads it for every row on every filter keystroke.
        self.display_fields = _derive_display_fields(self.scan.doc_fields_json)
        self._previous_scans_cache = self._compute_previous_scans(all_scans)
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

    def _compute_previous_scans(self, all_scans: list[ScanRecord] | None) -> list[ScanRecord]:
        """Find every other saved scan that represents the same person as this one, newest first -- see document/person_matcher.py. Computed once at row build time and cached, since it's read both for the summary row's visit-count badge and the detail section's visit history.

        all_scans, if given, is the full scan list a caller building many rows at once (e.g.
        MainWindow._load_history) already fetched -- reusing it here avoids each of those rows
        independently re-running its own full-table `list_scans(limit=100_000)` query, which
        turned every history load into O(n) redundant full-table scans for n displayed rows. A
        caller adding just one new row (e.g. after a fresh capture) can omit it; the one query that
        costs is no longer a hot loop when it only happens once.
        """
        if not self.scan.doc_fields_json:
            return []
        doc_fields = json.loads(self.scan.doc_fields_json)
        if not doc_fields:
            return []
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

        summary_row.addWidget(self._build_thumbnail(self.scan.face_image_path, "Live Face Capture"))
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

        self.match_icon_label = QLabel(_match_icon_text(self.scan.match_result))
        self.match_icon_label.setStyleSheet("font-size: 16pt; font-weight: bold;")
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
            self.flagged_match_label = QLabel(f"🚩 FLAGGED: {flagged_reason}")
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

        thumbnail = self._build_thumbnail(self.scan.doc_image_path, "Document Capture")
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

    def _build_thumbnail(self, image_path: str | None, title: str) -> QLabel:
        """Build a fixed-size thumbnail label for one of the row's two saved images. Double-clicking it opens the full image in a resizable viewer window."""
        label = ClickableThumbnailLabel(image_path, title)
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
        """Open a dialog to manually correct this scan's given name/surname/personal number, for when OCR/MRZ/barcode extraction got a field wrong or missed it entirely -- without needing a full re-capture."""
        doc_fields = json.loads(self.scan.doc_fields_json) if self.scan.doc_fields_json else {}
        dialog = _EditFieldsDialog(
            doc_fields.get("name", ""), doc_fields.get("surname", ""), doc_fields.get("personal_number", ""), parent=self
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

        if self._on_flagged_persons_changed is not None:
            # Reuses the same "refresh everything" callback flag changes use -- editing a name/
            # personal number can change which other scans this one's visit history/count (see
            # document/person_matcher.py) or watch-list match reflects, not just this row's own
            # display, so a full history reload is needed here too.
            self._on_flagged_persons_changed()

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

        dialog = _FlagPersonDialog(suggested_name, suggested_personal_number, has_face_image, has_doc_image, parent=self)
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
            if image_path and Path(image_path).exists():
                # Sized before unlinking -- lets the caller adjust its cached storage-usage total
                # by exactly this amount instead of re-walking the whole image directory tree (see
                # MainWindow._add_to_cached_storage_bytes).
                freed_bytes += Path(image_path).stat().st_size
                Path(image_path).unlink()
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


class DocumentCropLabel(QLabel):
    """The document preview label, extended to let the operator drag out a crop rectangle over a frozen captured frame.

    Only interactive while crop_mode_enabled is True (toggled by MainWindow when it's showing a
    just-captured document frame for review) -- during normal live preview, mouse events pass
    straight through to the base QLabel untouched, so this has no effect on ordinary preview
    behavior. The rectangle is tracked and painted in *label* pixel coordinates (what the operator
    visually sees, drag handles included) rather than the underlying frame's own resolution;
    MainWindow._crop_rect_to_frame_bounds is what converts it back to frame coordinates against
    the currently displayed pixmap's scale and centering offset before the numpy frame is actually
    sliced.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.crop_mode_enabled = False
        self.crop_rect: QRect | None = None
        self._drag_origin: QPoint | None = None

    def reset_crop(self) -> None:
        self.crop_rect = None
        self._drag_origin = None
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self.crop_mode_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._drag_origin = event.position().toPoint()
            self.crop_rect = QRect(self._drag_origin, self._drag_origin)
            self.update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self.crop_mode_enabled and self._drag_origin is not None:
            self.crop_rect = QRect(self._drag_origin, event.position().toPoint()).normalized()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self.crop_mode_enabled and self._drag_origin is not None:
            self.crop_rect = QRect(self._drag_origin, event.position().toPoint()).normalized()
            self._drag_origin = None
            self.update()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self.crop_rect is None or self.crop_rect.isNull():
            return
        painter = QPainter(self)
        pen = QPen(_CROP_RECT_COLOR)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRect(self.crop_rect)
        painter.end()


class MainWindow(QMainWindow):
    """The app's single combined screen: live previews, one capture-and-process button, and the scan history list."""

    def __init__(
        self,
        conn: sqlite3.Connection,
        camera_manager: CameraManager,
        face_analysis_app: FaceAnalysis,
    ) -> None:
        super().__init__()
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
        # Set once both frames of a capture are successfully in hand, holding them frozen for
        # operator review (optionally cropping the document image) before the face-match/OCR
        # pipeline actually runs -- see _enter_document_review / _exit_document_review. Distinct
        # from _pending_face_frame/_pending_doc_frame above, which hold a *partial* capture (one
        # camera failed) awaiting a retry, not a full pair awaiting operator review.
        self._review_face_frame: np.ndarray | None = None
        self._review_doc_frame: np.ndarray | None = None
        self._in_document_review = False
        # Consecutive failed preview-tick reads per camera, since the last successful one -- once
        # either crosses _SIGNAL_LOST_TICK_THRESHOLD, _update_previews swaps that camera's frozen
        # last-good frame for an explicit "signal lost" message rather than leaving it looking
        # like a normal (if slightly stale) live feed. A single dropped tick doesn't count; only a
        # sustained run of them does, so an occasional one-off missed frame doesn't flicker it.
        self._face_signal_lost_streak = 0
        self._doc_signal_lost_streak = 0
        # Backing state for the live document-camera blur warning (see
        # _update_doc_quality_warning) -- the tick count drives the throttled check interval, and
        # the last result is kept between checks so the banner doesn't flicker off on the ticks a
        # fresh check isn't due.
        self._doc_quality_check_tick_count = 0
        self._doc_frame_looks_blurry = False
        # Guards against re-entrant capture runs. The actual analysis pipeline runs on a background
        # thread now (see _start_capture_analysis), not synchronously -- without this, a click (or
        # shortcut press) during that window would fire a second, overlapping capture, and worse,
        # a second Re-analyze/Save could end up calling into the shared face_analysis_app model
        # from two worker threads at once. Set by both _on_save_capture_clicked and
        # _on_reanalyze_scan_requested, the two callers of _start_capture_analysis.
        self._is_processing = False
        # Set just before _start_capture_analysis is called and read back once its background
        # thread finishes -- kept alive on self since nothing else would hold a reference to the
        # running QThread/worker otherwise (see _start_capture_analysis's own docstring).
        self._capture_analysis_thread: QThread | None = None
        self._capture_analysis_worker: _CaptureAnalysisWorker | None = None
        # The capture button's own text, saved by _on_save_capture_clicked just before it's
        # overwritten with "Processing…", so _end_capture_processing can restore it afterward.
        self._capture_button_original_text = ""
        # Scan ids checked via each row's export-selection checkbox, for the "Export selected"
        # batch-export button in the filter row -- see _on_history_row_selection_toggled /
        # _on_export_selected_clicked. IDs rather than ScanRecord/HistoryRowWidget references so
        # this survives a _reload_history rebuild trivially (it's explicitly cleared there instead,
        # since the fresh rows all start unchecked -- see that method).
        self._selected_scan_ids: set[int] = set()

        self.setWindowTitle("Kontroll")
        self._build_layout()
        self._load_history()
        self._restore_window_state()

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
        self.face_preview_label = self._build_preview_label()
        self.doc_preview_label = self._build_preview_label(DocumentCropLabel)
        self.doc_quality_warning_label = self._build_quality_warning_label()
        previews_row.addLayout(self._build_preview_column("Visitor", self.face_preview_label), stretch=1)
        previews_row.addLayout(
            self._build_preview_column("Identification", self.doc_preview_label, self.doc_quality_warning_label),
            stretch=1,
        )
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
        top_layout.addWidget(self._build_review_controls_row())

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.addWidget(QLabel("Previous scans (newest first):"))
        bottom_layout.addLayout(self._build_filter_row())

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
        """Build the window's status bar: scan-count/disk-usage stats on the left, a Settings button on the right."""
        # QStatusBar draws a sunken-border frame around every widget added via addWidget/
        # addPermanentWidget by default, which shows up as a vertical divider line next to each
        # one -- this app doesn't want those dividers, so they're switched off here.
        self.statusBar().setStyleSheet("QStatusBar::item { border: none; }")

        self.status_stats_label = QLabel()
        self.statusBar().addWidget(self.status_stats_label)

        settings_button = QPushButton("Settings")
        settings_button.clicked.connect(self._on_settings_clicked)
        self.statusBar().addPermanentWidget(settings_button)

        self._recompute_storage_bytes()
        self._refresh_status_bar_stats()

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
        """Stop the live-preview poll -- see SettingsWindow's on_camera_probe_started docstring for why this is handed to it as a callback rather than it reaching into preview_timer directly."""
        self.preview_timer.stop()

    def _resume_camera_previews(self) -> None:
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
        """Discard every currently-displayed history row and reload the full list fresh from the database.

        Also clears the batch-export selection: the fresh HistoryRowWidgets start with their
        checkboxes unchecked, so a selection referring to the old (now-discarded) widgets would
        otherwise silently disagree with what's actually shown as selected on screen.
        """
        for row in self._history_rows:
            row.setParent(None)
            row.deleteLater()
        self._history_rows.clear()
        self._selected_scan_ids.clear()
        self._update_export_selected_button()
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
        self.date_filter_edit.dateChanged.connect(self._apply_filters)
        self._date_filter_opacity_effect = QGraphicsOpacityEffect(self.date_filter_edit)
        self.date_filter_edit.setGraphicsEffect(self._date_filter_opacity_effect)
        self._update_date_filter_dim_state()
        filter_row.addWidget(self.date_filter_edit)

        filter_row.addWidget(QLabel("Result:"))
        self.match_filter_combo = QComboBox()
        self.match_filter_combo.addItems(["All", "Match", "No Match"])
        self.match_filter_combo.currentTextChanged.connect(self._apply_filters)
        filter_row.addWidget(self.match_filter_combo)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search name...")
        self.search_edit.textChanged.connect(self._apply_filters)
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
        self._apply_filters()

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
        self._apply_filters()

    def _apply_filters(self) -> None:
        """Show or hide each history row so only scans matching the current date/result/search filters remain visible."""
        match_filter = self.match_filter_combo.currentText()
        search_text = self.search_edit.text().strip().lower()
        date_filter_active = not self.date_filter_checkbox.isChecked()
        selected_date = self.date_filter_edit.date().toPyDate()

        for row in self._history_rows:
            scan = row.scan
            visible = True

            if date_filter_active:
                visible = visible and datetime.fromisoformat(scan.created_at).date() == selected_date

            if match_filter == "Match":
                visible = visible and scan.match_result in ("match", "override_match")
            elif match_filter == "No Match":
                visible = visible and scan.match_result in ("no_match", "override_no_match")

            if search_text:
                display_fields = row.display_fields
                haystack = f"{display_fields.first_name} {display_fields.last_name}".lower()
                visible = visible and search_text in haystack

            row.setVisible(visible)

    def _scan_result_dates(self) -> set[date]:
        """Return the distinct calendar dates that have at least one saved scan, for ResultAwareCalendarWidget to highlight."""
        return {datetime.fromisoformat(row.scan.created_at).date() for row in self._history_rows}

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

    def _build_preview_label(self, label_class: type[QLabel] = QLabel) -> QLabel:
        """Build one live preview label that expands to fill whatever space its pane currently has.

        A minimum size (rather than a fixed one) keeps it from collapsing to nothing when the
        splitter handle above is dragged small, while still letting it grow -- _update_previews
        rescales each incoming frame to the label's current size on every tick. label_class lets
        the document preview use DocumentCropLabel (adding crop-selection support) while the face
        preview stays a plain QLabel, without duplicating this setup.
        """
        label = label_class("No camera signal")
        label.setMinimumSize(*_PREVIEW_SIZE)
        label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        label.setFrameShape(QFrame.Shape.Box)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        return label

    def _build_review_controls_row(self) -> QWidget:
        """Build the row of buttons shown only while a just-captured frame pair is frozen for operator review (see _enter_document_review), hidden the rest of the time.

        Save/Discard mirror the two outcomes described in the feature request this implements: the
        operator can accept the capture as-is, optionally after drawing a crop box on the document
        image to trim it down to just the card/passport itself (discarding background/glare
        outside that box), or discard the whole pair and retake. Crop mode only affects the
        document preview (DocumentCropLabel) -- the face capture is never manually cropped, since
        detector.py already finds and isolates the face itself for matching.
        """
        row = QHBoxLayout()

        self.crop_toggle_button = QPushButton("Draw crop box")
        self.crop_toggle_button.setCheckable(True)
        self.crop_toggle_button.setMinimumHeight(40)
        self.crop_toggle_button.toggled.connect(self._on_toggle_crop_mode_clicked)
        row.addWidget(self.crop_toggle_button)

        self.clear_crop_button = QPushButton("Clear crop")
        self.clear_crop_button.setMinimumHeight(40)
        self.clear_crop_button.clicked.connect(self._on_clear_crop_clicked)
        row.addWidget(self.clear_crop_button)

        self.save_capture_button = QPushButton("Save ✓")
        self.save_capture_button.setMinimumHeight(40)
        self.save_capture_button.setStyleSheet("font-weight: bold;")
        self.save_capture_button.clicked.connect(self._on_save_capture_clicked)
        row.addWidget(self.save_capture_button, stretch=1)

        self.discard_capture_button = QPushButton("Discard ✗")
        self.discard_capture_button.setMinimumHeight(40)
        self.discard_capture_button.clicked.connect(self._on_discard_capture_clicked)
        row.addWidget(self.discard_capture_button)

        self.review_controls_widget = QWidget()
        self.review_controls_widget.setLayout(row)
        self.review_controls_widget.setVisible(False)
        return self.review_controls_widget

    def _build_preview_column(
        self, headline_text: str, preview_label: QLabel, warning_label: QLabel | None = None
    ) -> QVBoxLayout:
        """Stack a centered, bold, white headline (and, optionally, a warning banner -- e.g. low camera resolution) above one camera preview label."""
        column = QVBoxLayout()

        headline = QLabel(headline_text)
        headline.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Deliberately no hardcoded color here -- leaving it unset lets the label inherit the
        # current QPalette's WindowText color (see ui/theme.py), which stays readable under both
        # light and dark themes and updates live if the operator switches themes, unlike a
        # hardcoded "color: white" (the previous approach, unreadable under a light theme since
        # this label sits directly on the window background with no fixed backdrop).
        headline.setStyleSheet("font-size: 16pt; font-weight: bold;")
        column.addWidget(headline)
        if warning_label is not None:
            column.addWidget(warning_label)
        column.addWidget(preview_label, stretch=1)

        return column

    def _build_quality_warning_label(self) -> QLabel:
        """Build the (initially hidden) banner shown above the document preview when the live camera frame looks too low-resolution or too blurry for reliable OCR -- see _update_doc_quality_warning."""
        label = QLabel()
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("color: #FF5252; font-weight: bold;")
        label.setVisible(False)
        return label

    def _load_history(self) -> None:
        """Populate the history list with every previously saved scan, newest first.

        _add_history_row always inserts at the top, so the scans are fed in here oldest-first
        (the reverse of list_scans' own newest-first order) -- each insert-at-top then pushes
        the previous one down by exactly one, leaving the newest scan on top once the loop ends.

        Fetches the full scan list once up front (all_scans) and hands it to every row, rather
        than letting each of the up-to-100 displayed rows independently run its own full-table
        query to find its own previous visits (HistoryRowWidget._compute_previous_scans) -- that
        used to make every history load do O(n) redundant full-table scans for n displayed rows.
        """
        all_scans = list_scans(self.conn, limit=100_000)
        displayed_scans = list_scans(self.conn)
        for scan in reversed(displayed_scans):
            self._add_history_row(scan, all_scans=all_scans)
        self._apply_filters()

    def _add_history_row(self, scan: ScanRecord, all_scans: list[ScanRecord] | None = None) -> None:
        """Insert one scan as the top-most row of the history list.

        all_scans, if the caller already has the full scan list in hand (e.g. _load_history
        loading many rows at once), is forwarded to HistoryRowWidget to avoid a redundant query --
        see its docstring. Omitted when adding a single new row (e.g. after a fresh capture),
        where one query is just a normal one-off cost, not a hot loop.
        """
        row = HistoryRowWidget(
            self.conn,
            scan,
            on_delete=self._on_history_row_deleted,
            on_flagged_persons_changed=self._on_flagged_persons_changed,
            on_jump_to_scan=self._jump_to_scan,
            on_reanalyze=self._on_reanalyze_scan_requested,
            on_export=self._on_export_scan_requested,
            on_selection_toggled=self._on_history_row_selection_toggled,
            all_scans=all_scans,
        )
        self.history_layout.insertWidget(0, row)
        self._history_rows.append(row)

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
        """Batch-export every scan the operator checked via its row checkbox, newest first, as one combined report -- see _export_scans."""
        selected_scans = [row.scan for row in self._history_rows if row.scan.id in self._selected_scan_ids]
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

        self._reload_history()
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
        filter was active. The row is then scrolled into view, its detail section expanded, and
        briefly outlined so it's obvious which one the operator landed on.
        """
        self._on_clear_filters_clicked()
        target_row = next((row for row in self._history_rows if row.scan.id == scan_id), None)
        if target_row is None:
            return
        target_row.detail_toggle.setChecked(True)
        self.history_scroll_area.ensureWidgetVisible(target_row, 0, 60)
        self._flash_row(target_row)

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
        """Reload the whole history list after a scan is deleted (DB row + image files already gone by this point), so every other row belonging to the same person recomputes its visit-count badge and previous-visits list (see document/person_matcher.py) without the deleted scan, not just this one's own row disappearing."""
        self._reload_history()
        self._add_to_cached_storage_bytes(-freed_bytes)
        self._refresh_status_bar_stats()

    def _update_previews(self) -> None:
        """Poll both cameras for their latest frame and refresh the live preview labels. Runs on a timer, independent of capture.

        Skipped entirely while a captured frame pair is frozen for operator review (see
        _enter_document_review) -- otherwise the next live tick would immediately overwrite the
        very frames the operator is reviewing/cropping.
        """
        if self._in_document_review:
            return

        face_frame = self.camera_manager.read_face_preview_frame()
        if face_frame is not None:
            self._face_signal_lost_streak = 0
            self._set_preview_pixmap(self.face_preview_label, face_frame)
        else:
            self._face_signal_lost_streak += 1
            self._maybe_show_signal_lost(self.face_preview_label, self._face_signal_lost_streak)

        doc_frame = self.camera_manager.read_doc_preview_frame()
        if doc_frame is not None:
            self._doc_signal_lost_streak = 0
            self._set_preview_pixmap(self.doc_preview_label, doc_frame)
            self._update_doc_quality_warning(doc_frame)
        else:
            self._doc_signal_lost_streak += 1
            self._maybe_show_signal_lost(self.doc_preview_label, self._doc_signal_lost_streak)

    def _maybe_show_signal_lost(self, label: QLabel, lost_streak: int) -> None:
        """Once a camera has failed to return a frame for _SIGNAL_LOST_TICK_THRESHOLD consecutive ticks, replace its frozen last-good frame with an explicit warning rather than leaving it looking like a live feed."""
        if lost_streak == _SIGNAL_LOST_TICK_THRESHOLD:
            label.clear()
            # CameraManager attempts its own automatic reopen shortly after this point (see
            # capture.camera_manager's reconnect logic) -- this streak resets to 0 the moment a
            # frame reads successfully again, which clears this warning with no operator action.
            label.setText("⚠ Camera signal lost — reconnecting...")
            label.setStyleSheet("color: #FF5252; font-weight: bold; font-size: 12pt;")

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

        Grabbing the two still frames is quick, so the busy feedback here is brief -- the actual
        slow face-match/OCR/MRZ/barcode pipeline no longer runs synchronously off this click at
        all; it's deferred until the operator explicitly clicks Save in the review controls that
        appear once both frames are in hand (see _enter_document_review / _on_save_capture_clicked).
        The re-entrancy guard still matters here regardless, since a click or shortcut press queued
        up mid-capture would otherwise fire a second, overlapping capture the instant this one
        returns. Also declines while a previous capture is already frozen for review
        (self._in_document_review) -- the capture button is hidden then, but the keyboard shortcut
        stays bound to this same handler regardless of button visibility, and firing a fresh
        capture over a held review would silently discard whatever the operator was reviewing.
        """
        if self._is_processing or self._in_document_review:
            return
        self._is_processing = True
        original_button_text = self.capture_button.text()
        self.capture_button.setEnabled(False)
        self.capture_button.setText("Capturing…")
        QApplication.processEvents()
        try:
            self._run_capture_and_process()
        finally:
            self.capture_button.setEnabled(True)
            self.capture_button.setText(original_button_text)
            self._is_processing = False

    def _run_capture_and_process(self) -> None:
        """Grab a still frame from both cameras and hand them off for operator review (see _enter_document_review) -- the actual face-match/document-validation pipeline runs later, from _on_save_capture_clicked, once the operator confirms (optionally after cropping the document image).

        If a previous click already got a good frame from one camera but not the other, that held
        frame (self._pending_face_frame / self._pending_doc_frame) is reused instead of recaptured
        -- this click only needs to (re)try whichever side is still missing.

        Everything from the frame-capture check onward that isn't itself an operator-facing
        decision (i.e. the whole face-match/OCR/MRZ/barcode/save pipeline) is wrapped in a single
        try/except: previously an exception anywhere in that chain was swallowed by main.py's
        sys.excepthook (logged to file only, no dialog, no history row added) -- which is exactly
        what made some clicks look like they silently did nothing at all.
        """
        logger.info("Operator triggered capture & process")
        # np.ndarray has no usable truthiness (a plain `or` raises on a multi-element array), so
        # each pending frame is checked explicitly against None rather than short-circuited.
        face_frame = (
            self._pending_face_frame
            if self._pending_face_frame is not None
            else capture_still_frame(self.camera_manager.face_capture)
        )
        doc_frame = (
            self._pending_doc_frame
            if self._pending_doc_frame is not None
            else capture_still_frame(self.camera_manager.doc_capture)
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
            return

        self._pending_face_frame = None
        self._pending_doc_frame = None
        self.discard_pending_button.setVisible(False)

        self._enter_document_review(face_frame, doc_frame)

    def _enter_document_review(self, face_frame: np.ndarray, doc_frame: np.ndarray) -> None:
        """Freeze both previews on a just-captured frame pair and show the review controls, holding off on the actual face-match/OCR pipeline until the operator explicitly saves.

        Both previews are frozen (via _in_document_review, checked by _update_previews) for visual
        consistency even though only the document image can be cropped -- the operator is looking
        at a still pair, not one live feed next to one frozen one.
        """
        self._review_face_frame = face_frame
        self._review_doc_frame = doc_frame
        self._in_document_review = True

        self._set_preview_pixmap(self.face_preview_label, face_frame)
        self._set_preview_pixmap(self.doc_preview_label, doc_frame)
        self.doc_preview_label.reset_crop()

        self.capture_button.setVisible(False)
        self.crop_toggle_button.setChecked(False)
        self.review_controls_widget.setVisible(True)

        self.statusBar().showMessage(
            "Review the capture: optionally draw a crop box on the document image, then Save or Discard."
        )
        logger.info("Capture held for operator review")

    def _exit_document_review(self) -> None:
        """Drop the held review frame pair and return to the normal live-preview capture button, whether the operator saved or discarded."""
        self._review_face_frame = None
        self._review_doc_frame = None
        self._in_document_review = False
        self.doc_preview_label.reset_crop()
        self._set_crop_mode_enabled(False)
        self.crop_toggle_button.setChecked(False)
        self.review_controls_widget.setVisible(False)
        self.capture_button.setVisible(True)

    def _on_toggle_crop_mode_clicked(self, checked: bool) -> None:
        self._set_crop_mode_enabled(checked)

    def _set_crop_mode_enabled(self, enabled: bool) -> None:
        self.doc_preview_label.crop_mode_enabled = enabled
        self.doc_preview_label.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)

    def _on_clear_crop_clicked(self) -> None:
        self.doc_preview_label.reset_crop()

    def _on_discard_capture_clicked(self) -> None:
        """Drop a reviewed frame pair without processing or saving it -- the operator retakes from scratch with a fresh Capture click."""
        logger.info("Operator discarded a reviewed capture")
        self._exit_document_review()

    def _crop_rect_to_frame_bounds(self, label: QLabel, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        """Convert the crop rectangle the operator dragged on `label` (in on-screen label pixels) into (x1, y1, x2, y2) pixel bounds within `frame`'s own resolution, or None if there's no usable selection.

        None covers both "no rectangle drawn" and "drawn but too small or entirely outside the
        displayed image" -- either way the caller should fall back to using the frame uncropped,
        exactly as if the operator had clicked Save without ever entering crop mode. The displayed
        pixmap's rect is recovered from its own size centered in the label (matching the
        AlignCenter alignment _build_preview_label sets), since a KeepAspectRatio scale can
        letterbox the pixmap smaller than the label itself.
        """
        crop_rect: QRect | None = getattr(label, "crop_rect", None)
        if crop_rect is None or crop_rect.isNull():
            return None
        if crop_rect.width() < _MIN_CROP_DRAG_PX or crop_rect.height() < _MIN_CROP_DRAG_PX:
            return None

        pixmap = label.pixmap()
        if pixmap is None or pixmap.isNull():
            return None
        displayed_rect = QRect(
            (label.width() - pixmap.width()) // 2,
            (label.height() - pixmap.height()) // 2,
            pixmap.width(),
            pixmap.height(),
        )
        clipped = crop_rect.intersected(displayed_rect)
        if clipped.width() < _MIN_CROP_DRAG_PX or clipped.height() < _MIN_CROP_DRAG_PX:
            return None

        frame_height, frame_width = frame.shape[:2]
        scale_x = frame_width / displayed_rect.width()
        scale_y = frame_height / displayed_rect.height()
        x1 = max(int((clipped.left() - displayed_rect.left()) * scale_x), 0)
        y1 = max(int((clipped.top() - displayed_rect.top()) * scale_y), 0)
        x2 = min(int((clipped.right() + 1 - displayed_rect.left()) * scale_x), frame_width)
        y2 = min(int((clipped.bottom() + 1 - displayed_rect.top()) * scale_y), frame_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

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

    def _on_save_capture_clicked(self) -> None:
        """Entry point for the review controls' Save button: applies any drawn document crop, then kicks off the face-match/OCR/document-validation pipeline on a background thread (see _start_capture_analysis) and returns immediately -- _finish_save_capture/_on_save_capture_failed pick up once it's done."""
        if self._is_processing:
            return
        face_frame = self._review_face_frame
        doc_frame = self._review_doc_frame
        if face_frame is None or doc_frame is None:
            return

        crop_bounds = self._crop_rect_to_frame_bounds(self.doc_preview_label, doc_frame)
        if crop_bounds is not None:
            x1, y1, x2, y2 = crop_bounds
            doc_frame = doc_frame[y1:y2, x1:x2].copy()
            logger.info("Operator cropped document image to %dx%d before processing", x2 - x1, y2 - y1)

        self._exit_document_review()

        self._is_processing = True
        self._capture_button_original_text = self.capture_button.text()
        self.capture_button.setEnabled(False)
        self.capture_button.setText("Processing…")

        self._start_capture_analysis(
            face_frame,
            doc_frame,
            on_finished=lambda analysis: self._finish_save_capture(analysis, face_frame, doc_frame),
            on_failed=self._on_save_capture_failed,
        )

    def _end_capture_processing(self) -> None:
        """Restore the capture button and clear the processing guard -- the shared tail end of every path out of _on_save_capture_clicked's background analysis (success or failure)."""
        self.capture_button.setEnabled(True)
        self.capture_button.setText(self._capture_button_original_text)
        self._is_processing = False

    def _on_save_capture_failed(self) -> None:
        """Handle _start_capture_analysis's worker raising during a Save -- mirrors the except branch _finish_save_capture itself still has for errors in the (fast, synchronous) save/DB-write step below the analysis."""
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
            # Sized after writing rather than re-walking the whole image directory tree to find the
            # new total -- see MainWindow._add_to_cached_storage_bytes.
            self._add_to_cached_storage_bytes(face_image_path.stat().st_size + doc_image_path.stat().st_size)

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

            self._add_history_row(get_scan_by_id(self.conn, scan_id))
            self._apply_filters()
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
        splitter = getattr(self, "top_bottom_splitter", None)
        if splitter is None:
            return
        if self._top_pane_height is None:
            self._top_pane_height = self._top_pane_widget.sizeHint().height()
        bottom_height = max(splitter.height() - splitter.handleWidth() - self._top_pane_height, 0)
        splitter.setSizes([self._top_pane_height, bottom_height])

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
        tearing down the QThread while it's still running.
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
        self._save_window_state()
        if self._settings_window is not None:
            self._settings_window.close()
        self.camera_manager.release_all()
        super().closeEvent(event)
