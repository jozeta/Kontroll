# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A local, offline-first Windows 11 desktop app (Python + PyQt6) that verifies identity at a
staffed desk: it matches a live webcam face capture against the photo on a scanned ID document,
and validates the document itself via OCR/MRZ/PDF417 cross-checking. An operator manually
triggers each capture and reviews each result; no data leaves the local machine at runtime.

## Commands

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Building a standalone/portable EXE (PyInstaller `--onedir`, plus optionally bundling Tesseract
and the InsightFace model as portable folder copies) is a separate, longer workflow — see
README.md "Building a standalone EXE" and "Portable / USB-stick deployment", not repeated here.

`tests/test_documents.py` covers the pure document logic (printed and MRZ dates, ages, AAMVA
barcodes, under-18/expiry flags) and needs only `pytest` and `mrz`: `python -m pytest -q`. CI runs
it with `ruff check .` (settings in `ruff.toml`, incl. the security rules). Everything that needs the
camera, Qt, OpenCV or InsightFace has no automated tests; verification there has been done with throwaway ad-hoc scripts run against the venv interpreter,
exercising real library behavior rather than mocks wherever practical (e.g. generating a
genuinely valid ICAO 9303 MRZ via `mrz.generator.td3.TD3CodeGenerator` and checksum-validating
it, or a real PDF417 encode→decode round trip via `zxingcpp.create_barcode`/`read_barcodes`,
or running detection against InsightFace's own bundled sample images at
`venv/Lib/site-packages/insightface/data/images/`).

**Testing the PyQt6 UI headlessly** (no display attached): set `QT_QPA_PLATFORM=offscreen`
before importing PyQt6. Any code path that shows a `QMessageBox` (`.information`/`.warning`/
`.question`) will block forever in this mode since there's nothing to click — stub those
methods to no-ops (or a canned return value) before invoking such a path in a script.
`window.show()` must be called for any later `widget.isVisible()` assertions to be meaningful
(Qt visibility is hierarchical), and newly-added child widgets may need an
`app.processEvents()` pass before their visibility reflects a just-made `setVisible()` call,
since no `app.exec()` event loop is running in a script.

**Windows PATH gotcha**: a long-lived shell/process does not see PATH changes made by a
`winget install` (or similar) that happened after that shell started — new tools installed
mid-session need either a fresh process or an explicit full-path reference, not a bare command
name, until that process restarts.

## Architecture

### Startup sequence and the deferred-import bootstrap (main.py + prerequisites.py)

`main.py` constructs `QApplication` first, then calls `prerequisites.run_prerequisite_check()`
**before** importing anything from `capture/`, `face/`, `document/`, `storage/`, or `ui/`. Those
imports are deliberately deferred to inside `main()`, after the prerequisite check, so that a
missing third-party package is caught by the checker (which offers to `pip install` it) rather
than crashing the process with an `ImportError` before the operator ever sees a dialog.

`prerequisites.py` mirrors this discipline internally: every check function avoids importing the
package it's checking for at module load time (e.g. it checks for the InsightFace model and
Tesseract via plain `pathlib`/`shutil` filesystem checks, not by importing `insightface` or
`pytesseract`; `check_cameras()` imports `cv2` lazily inside a try/except). The one hard,
unavoidable dependency is PyQt6 itself, since it's what renders the dialogs — there's no way to
check for its own absence from within the running app.

When editing either file, preserve this ordering. Moving a heavy import back to module level in
either file defeats the point of the checker.

### Pipeline flow

A single "Capture" button click (`ui/main_window.py::MainWindow._on_capture_and_process_clicked` →
`_run_capture_and_process`) grabs a still frame from both cameras and immediately runs the full
pipeline below on a background thread (`_start_capture_analysis`/`_CaptureAnalysisWorker`), saving
the scan as soon as it finishes -- there is no operator review/crop step in between, and the live
preview is never frozen (see "Capture flow: immediate save, no review step" below for why and what
replaced the old review UI). The steps run, in order:

1. **Capture**: `capture/camera_manager.py::CameraManager.get_face_latest_frame()`/
   `get_doc_latest_frame()` return whichever frame each camera's own background reader thread most
   recently read (see "Camera frame reading" below) -- this never itself blocks on camera hardware.
2. **Face pipeline** (`face/`): `detector.py` wraps a single shared
   `insightface.app.FaceAnalysis(name="buffalo_l")` instance (loaded once at startup — expensive,
   never reloaded per-scan) to detect faces in both the live frame and the document frame;
   `embedder.py` picks the highest-confidence face and returns its 512-d normalized embedding;
   `matcher.py` computes cosine similarity against the configurable threshold and returns both
   the raw score and a bool — never just one or the other, since the UI must always show both.
3. **Document pipeline** (`document/`): `ocr.py::extract_words` runs first so `image_quality.py`
   can flag a blurry (Laplacian-variance proxy), glare-heavy, low-resolution, or hard-to-read
   (mean OCR word confidence too low) capture — passive only, shown as a banner on the scan's
   thumbnail, never blocks or retries. `ocr.py::extract_text` (a second, separate Tesseract pass,
   kept apart from `extract_words` since `image_to_data`'s per-word layout doesn't reliably
   reproduce `image_to_string`'s spacing that MRZ/driving-licence field parsing depends on) then
   feeds `mrz_parser.py` (locates
   MRZ-shaped lines by character set + fixed line width, then checksum-validates via the `mrz`
   library — TD1/30-char/3-line, TD2/36-char/2-line, TD3/44-char/2-line, widths are unique so
   there's no format ambiguity) → `barcode_parser.py` (PDF417 only, via zxing-cpp, raw decoded
   text — no structured field parsing for a Swedish driving licence specifically, since that
   layout is unvalidated against a real sample; a US/Canadian driver's licence or ID card's
   payload *is* structurally parsed, by `aamva_parser.py`, since AAMVA is a published standard) →
   `doc_classifier.py` (type/country from MRZ document code first, else a decoded AAMVA payload,
   else OCR keyword fallback, else bare barcode presence) → `field_crosscheck.py::
   determine_validation_status` (the decision point for `validation_status`) and
   `field_crosscheck.py::check_person_flags` (separate under-18/expired-document flags — see
   below). When no MRZ is found and the barcode didn't decode as AAMVA (e.g. an EU driving
   licence), `dl_field_parser.py` heuristically extracts name/birth-date/expiry-date from the OCR
   text instead, anchored on the EU-harmonized numbered field layout (fields 1/2/3/4b) shared by
   every EU/EEA member state's driving licence regardless of language; if the printed date-of-birth
   line itself didn't OCR cleanly, birth date falls back to the first 6 digits of field 5's
   Swedish personnummer (Sweden-specific, same YYMMDD encoding `national_id_parser.py` already
   derives birth date from directly — see that module). For a `national_id`-classified document,
   `national_id_parser.py` (Sweden, anchored on the card's own reliably-OCR'd title) is tried
   first regardless of detected country; if it finds nothing, `eu_id_card_parser.py` is the
   fallback for every other EU/EEA member state's national ID card, anchored on the English field
   labels EU Regulation 2019/1157 has mandated on every member state's card since August 2021
   ('Surname', 'Given name(s)', 'Nationality', 'Date of birth', 'Date of expiry') rather than a
   per-country translated-label table — its own extracted `nationality` field (an ICAO alpha-3
   code) also backfills `doc_country` in `ui/main_window.py::_analyze_capture` when
   `doc_classifier.py`'s title-keyword match didn't already resolve one. OCR itself runs Tesseract
   with every supported document-issuing country's language pack loaded together (see
   `config/ocr_languages.py`), not just English, so each country's own diacritics OCR correctly;
   `document/ocr.py::_preprocess_for_ocr` also conditionally runs CLAHE contrast normalization
   before thresholding when the captured frame's grayscale standard deviation is low enough to
   indicate a genuinely over/under-exposed capture (a normally-exposed capture skips it entirely,
   verified not to regress that common case — see that function's own comments for the exact
   threshold and how it was derived).
4. **Persist**: both images saved to the face/document image directories resolved by
   `config/app_paths.py` (LOCALAPPDATA-based by default, independently operator-configurable —
   *not* a fixed `data/` path relative to the project), one row written via
   `storage/models.py::insert_scan`, then prepended to the on-screen history list.

### Capture flow: immediate save, no review step

Capture used to freeze both live previews on every click so the operator could optionally draw a
crop box on the document image before an explicit Save committed it — diagnosed as adding an
unwanted pause to every single capture when most don't need cropping at all. `MainWindow` now runs
the whole pipeline above immediately off the Capture click (`_run_capture_and_process` →
`_start_capture_analysis` → `_finish_save_capture`, all on a background `QThread` — see that
method's own docstring) with the live preview never paused (`_update_previews` runs unconditionally
on its `QTimer` now); `_is_processing` still guards against a re-entrant click firing a second,
overlapping capture while one is still being analyzed. Cropping and brightness/contrast correction
moved to an explicit, occasional post-hoc action instead: double-click a saved face/document
thumbnail to open `ui/image_viewer.py::ImageViewerWindow`, then its "Edit image…" button opens
`ImageEditDialog` — crop (with scroll-wheel zoom via `_CropZoomView`, mirroring
`ImageViewerWindow`'s own zoom/pan interaction), brightness/contrast/sharpen/black-and-white
controls grouped in their own titled "Adjustments" box (visually separated from the image preview
above rather than a bare stack of rows crowding it), a "Preview: what OCR sees" toggle (swaps the
display for the literal preprocessed image `document/ocr.py::preview_ocr_preprocessing` produces,
since judging whether an edit helps OCR by eye on the color original is unreliable), and (document
images only, see below) a field-check panel that runs the real document classification +
field-extraction pipeline against the current adjusted-and-cropped image and shows a side-by-side
"Original (saved) / New (this image)" comparison of the identity fields found (surname, personal
number, dates, ...), a changed value highlighted (debounced ~700ms after the last change, or on
demand via "Test extraction now") — all applied in place to the scan's saved image file. The
field-check pipeline is `document/field_extraction.py::extract_document_fields` — extracted out of
`ui/main_window.py::_analyze_capture` (its original home) into its own module specifically so this
dialog and the real capture pipeline can never independently drift apart on what a given image
extracts to; `_analyze_capture` now just calls it and handles face-matching/crosscheck/flags
separately. Omitted entirely for a face image (`kind="face"`), which has no document fields to
extract. All edits run against the full-resolution saved file (`cv2.imread(image_path)`), never a
downscaled copy; the black-and-white checkbox converts to grayscale then back to 3-channel BGR
(`cv2.cvtColor` round-trip) so every downstream consumer -- crop-bounds math, the pixmap converter,
`cv2.imwrite` -- keeps working against a plain BGR frame with no separate code path.

The dialog itself is resizable (previously fixed to its own `sizeHint()`, cramping the controls
against a small fixed-size image canvas) — `_CropZoomView` now adopts the same "base pixmap at
full/native resolution, zoom level 1.0 means fitted-to-viewport" architecture
`ImageViewerWindow`'s own `_ZoomableImageView` already used (`_fit_scale`/`_display_scale`, plus a
`resizeEvent` that keeps the same content centered across a resize), rather than pre-shrinking the
adjusted image to a small fixed preview size before ever handing it to the view. That pre-shrink
was the actual root cause of the editor's brightness/contrast/sharpen preview looking pixelated —
scroll-wheel zoom was upscaling an already-downscaled copy, not revealing real pixel detail from
the source; confirmed via a real synthetic high-resolution capture that the view's base pixmap
now matches the source image's own resolution exactly, with real magnification (not
Qt-interpolated fake detail) available past "fitted". The dialog also opens already zoomed in on
the detected face/card region (reusing the same `_find_zoom_bbox` detection `ImageViewerWindow`'s
own auto-zoom uses) rather than the whole captured frame, since a document capture in particular
commonly frames the whole light-box/desk area with the card itself occupying only a modest
fraction of it. `_find_zoom_bbox` pads whatever `document/card_detector.py::find_card_bounding_box`
finds by `_CARD_ZOOM_MARGIN_RATIO` (25%, via the shared `_expand_bbox_with_margin` helper also used
for the face case's own margin) before computing the zoom -- confirmed against a real operator
report that a zoom computed to exactly touch the detected box's edges left zero tolerance for any
detection imprecision (a slightly rotated card, a brightness-threshold edge that undershoots by a
few percent), visibly clipping the bottom of the card on open; this margin is purely extra
preview-framing headroom, separate from `find_card_bounding_box`'s own much smaller (6%)
pixel-precision margin used for the actual OCR crop. `_CropZoomView` also supports right-mouse-
button drag panning (`_panning`/`_pan_start_pos`/`_pan_start_scroll`, mirroring
`_ZoomableImageView`'s left-button pan) alongside its existing left-button crop-drag, so both are
available at once with no mode toggle. The field-check panel's text area is a fixed-height
(`_FIELD_CHECK_PANEL_HEIGHT`) `QScrollArea`, not a plain label -- diagnosed against a real
operator report that the label's content alternates between a short "waiting for changes to
settle…" line and a multi-row comparison table as sliders are dragged, and without a fixed height
that swing in required layout height forced the whole (resizable) dialog window to visibly grow
and shrink on every single slider tick; a longer table now simply scrolls internally instead. The
very first edit to a given image copies the original aside to
`<name>.original<ext>` first (only if that backup doesn't already exist) so the source capture is
never destructively lost. Applying an edit offers to re-analyze the scan with the corrected image,
reusing the exact same `_start_capture_analysis` pipeline `HistoryRowWidget`'s own "Re-analyze"
button already calls (via `on_reanalyze`, threaded down from `MainWindow._on_reanalyze_scan_requested`
through `HistoryRowWidget`/`ClickableThumbnailLabel`/`ImageViewerWindow`) — no separate re-analyze
logic exists for an edited image.

### Camera frame reading and exposure (`capture/camera_manager.py`)

Diagnosed from a production `app.log`: a camera that falls back to an uncompressed capture format
(see `_open_camera`'s own MJPG comment) can take up to ~1s per `capture.read()` call once
bandwidth-throttled over USB. Previously this call happened directly on `ui/main_window.py`'s
~30fps preview `QTimer` callback — the main GUI thread — so a throttled camera froze the entire UI
for however long each read took. Each camera's `capture.read()` now runs continuously on its own
daemon background thread (`_CameraReaderThread`); `CameraManager.read_face_preview_frame`/
`read_doc_preview_frame` (the live preview) and `get_face_latest_frame`/`get_doc_latest_frame` (the
Capture button's still-frame grab, replacing the old `capture/frame_capture.py::capture_still_frame`
retry-loop approach — a continuously-refreshed buffer makes a one-off dropped read invisible to the
consumer without needing its own retry) are now just non-blocking getters over that thread's latest
buffered frame. `CameraManager.pause_readers()`/`resume_readers()` (called from
`MainWindow._pause_camera_previews`/`_resume_camera_previews`) must bracket any *direct* access to
`camera_manager.face_capture`/`doc_capture` from another thread — resolution probing, diagnostics
export, exposure changes in `ui/settings_window.py` — since DirectShow handles aren't documented
safe for concurrent access; `pause_readers()` blocks until any read already in flight finishes
before returning, so it's always safe to touch the raw capture immediately afterward.

Each camera also has operator-configurable exposure settings (`ExposureSettings`: auto, or a manual
exposure/brightness pair in DirectShow's own native units) — default is auto, matching how both
cameras were found configured in production; a manual override is applied via
`CameraManager.set_face_camera_exposure`/`set_doc_camera_exposure`, persisted under the
`face_camera_auto_exposure`/`face_camera_exposure`/`face_camera_brightness` settings keys (and the
`doc_camera_*` equivalents — see `config/settings.py`), and exposed as a checkbox plus two sliders
per camera in Settings → Cameras. Not every camera/driver honors manual exposure control — same
accepted-limitation pattern as `set_exact_resolution`'s own resolution fallback.

**Compression varies by resolution, not just by camera.** Diagnosed against a real operator report:
a camera that streams smoothly at full resolution in other applications still fell back to
uncompressed YUY2 in this app at every resolution it was tried at. `probe_supported_resolutions`
now reads back the negotiated FourCC *at each candidate resolution* (`_apply_resolution_and_get_
fourcc`), not just once at camera-open time — plenty of real UVC webcams only expose MJPG below
their sensor's native/highest mode. `set_best_available_resolution`/`set_lowest_available_
resolution` (`_best_candidate_preferring_compression`) now prefer the highest/lowest candidate that
*also* keeps MJPG over one that doesn't, even if that means a resolution a step down from the
outright highest; the Settings resolution dropdown and diagnostics export both show each
resolution's negotiated compression (`ui/settings_window.py::_resolution_compression_label`).

**Capture backend is now per-camera configurable** (`CameraManager.set_face_camera_backend`/
`set_doc_camera_backend`, persisted under `face_camera_backend`/`doc_camera_backend`, values from
`capture.camera_manager.CAMERA_BACKENDS`) — DirectShow stays the app-wide default (see
`enumerate_cameras`' own comment for why it's used for *enumeration* specifically), but Media
Foundation is offered as a per-camera opt-in alternative in Settings → Cameras, since DirectShow's
MJPG negotiation is a known-flaky OpenCV/Windows quirk on some camera/driver combinations that
Media Foundation often handles more reliably — exactly the mismatch a camera behaving fine in
other (typically Media-Foundation-or-native-SDK-based) applications but not this one suggests.
Caveat documented in `_open_camera`'s own docstring: DirectShow's and Media Foundation's device-index
numbering aren't guaranteed to agree on a system with several cameras attached, since
`enumerate_cameras` always probes via DSHOW — switching a specific camera's backend could in
principle open a different physical device at the same index; this is immediately obvious from the
live preview if it happens, and easy to revert.

**Backend switching is retried, verified, and self-healing** (`_open_camera_verified`,
`_BACKEND_SWITCH_RELEASE_DELAY_SECONDS`/`_BACKEND_SWITCH_RETRY_ATTEMPTS`) — diagnosed from a real
operator report: switching a camera to Media Foundation and back to DirectShow left it unable to
come back online at all. A UVC device doesn't always finish releasing from one capture API before a
*different* one tries to claim it right after; the original single-attempt-with-no-delay
`_open_camera` call could get back a handle that reports `isOpened()` but never actually delivers a
frame, and nothing verified, retried, or fell back, so that broken handle was simply kept forever.
`set_face_camera_backend`/`set_doc_camera_backend` now insert a short settling delay before
(re)opening under a different backend, retry a few times, and — if the newly requested backend
still doesn't deliver a frame — automatically revert to whichever backend was last known to work,
returning `(succeeded, backend_now_in_effect)` so `ui/settings_window.py::_on_backend_changed` can
persist/display whichever backend is *actually* in effect (which may not be the one the operator
picked) and tell them if it had to fall back. This only applies to genuine backend switches, not the
routine reconnect-after-read-failures path (`_reopen_face_camera`/`_reopen_doc_camera`), which

**The switch itself runs on a background thread** (`ui/settings_window.py::_BackendSwitchWorker`,
mirroring `ui/main_window.py`'s `_CaptureAnalysisWorker`/`QThread` pattern) — diagnosed against a
real operator report that the whole app froze for a long time switching back to DirectShow. The
retry/settling-delay machinery above can legitimately take several real seconds of blocking
hardware I/O; running it directly on `_on_backend_changed`'s own (UI) thread, as the very first cut
of this feature did, blocked the entire app for that whole span. The affected combo is disabled
(and the live preview paused, via the existing `_on_camera_probe_started`/`_on_camera_probe_
finished` callbacks) for the duration, both lifted once the worker's `finished` signal delivers the
result to `_finish_backend_switch` back on the main thread. `MainWindow.closeEvent` also now
declines to close (matching its existing guard for an in-progress capture/re-analysis) while
`SettingsWindow.is_backend_switch_in_progress()` is true, since `release_all()` would otherwise race
the worker thread's own still-in-flight release/reopen of the same camera handles.
reopens under the *same* backend a handle already used and has never shown this failure mode.

**`capture.read()` can raise, not just fail — `_safe_read` wraps every call site in this module.**
The retry/revert logic above initially still weren't enough: the real failure turned out to be
OpenCV itself raising `cv2.error: ... (-215:Assertion failed) _step >= minstep in function
'cv::Mat::Mat'` *out of* `capture.read()` when Media Foundation returned a malformed/inconsistent
frame buffer for a specific requested resolution — an uncaught Python exception, not a `(False,
None)` result. Every call site in this module (including `_CameraReaderThread`'s own background
read loop) treated `capture.read()`'s result as a plain tuple to test, never as something that
could raise; left unguarded, that exception unwound mid-operation and skipped every step after the
failing read — including the code that would install the new capture handle or resume the reader
thread — which is exactly what turned one bad frame into a camera stuck fully offline until the
whole app was restarted (worse for the reader thread specifically: an uncaught exception there
silently kills the thread outright, since Python threads don't propagate exceptions to the main
thread). `_safe_read(capture)` now wraps every `capture.read()` call in this module and catches
`cv2.error`, returning `(False, None)` like an ordinary failed read instead — the pre-existing
per-resolution/per-backend fallback logic above then handles it exactly like any other failed
candidate, no special-casing needed once the exception itself can no longer escape. Logs the
caught error's message only (no traceback) since this can legitimately repeat many times in a row
during one probe/retry sequence, all with the identical underlying message.

**Framerate is now explicitly requested too** (`_TARGET_FPS`, applied in `_apply_resolution`
alongside width/height) — UVC negotiates FourCC/resolution/framerate together as one bundled mode,
and this app previously never requested a framerate at all, leaving the driver/backend to default to
whichever one it happened to pick for a given resolution+format (commonly a low one at larger
resolutions). Diagnosed from an operator report of a camera confirmed to deliver smooth 1080p30 in
other applications while measuring far below that here — pointing at a negotiation gap in this app's
own request rather than a hardware/bandwidth ceiling (already addressed separately by preferring a
compressed resolution, see `_best_candidate_preferring_compression` above). A camera/driver that
doesn't support 30fps at a given resolution simply clamps to what it does support, same
accepted-limitation pattern as every other capability probed this way.

### Precise native-format enumeration via pygrabber (`capture/dshow_native_formats.py`)

`cv2.VideoCapture` only exposes a generic, best-effort property API for resolution/FourCC/FPS --
it has no way to ask a camera what it *actually* natively supports the way a native application
does, so every negotiation this app does is really a guess (a candidate size/format is requested,
then whatever the driver actually landed on is read back). Diagnosed against a real operator report
-- and confirmed directly against their own attached camera (a Logitech StreamCam) using this
module: it natively advertises MJPG at every tested resolution up to and including 1920x1080,
proving the "only uncompressed/only 640x480 available" behavior seen through cv2 was a negotiation
precision gap in this app's own request, not a hardware limitation.

`list_native_formats(device_index)` queries DirectShow's own `IAMStreamConfig::GetStreamCaps`
(via the `pygrabber` package) for a camera's exact advertised (compression, resolution) modes --
authoritative ground truth, surfaced in the Settings Cameras tab's diagnostics export alongside
(not replacing) the existing cv2-probed list, specifically so an operator/developer never has to
guess again.

This module previously also went a step further with `try_apply_best_native_format`, which
selected and applied one of those exact native modes directly via `IAMStreamConfig::SetFormat`,
called from `_open_camera` immediately *before* cv2 ever opened the same device index, on the
theory that a UVC driver might persist that as the device's current mode across the handoff.
**Confirmed against the operator's own real `app.log` to be actively harmful, and removed**:
opening a second DirectShow graph to pre-set the format left the device in a state that made cv2's
own subsequent open crash repeatedly (`cv2.error: ... _step >= minstep in function
'cv::Mat::Mat'`) at the requested resolution, which `_open_camera`'s existing self-heal logic then
(correctly, but unnecessarily) read as "camera did not honor requested resolution" and fell back
to a resolution *lower* than before this nudge existed (observed: a requested 1920x1080 falling
all the way back to 640x480). The actual root cause of the uncompressed-fallback problem turned
out to be simpler and is fixed directly in `capture/camera_manager.py::_apply_resolution`: setting
`CAP_PROP_FRAME_WIDTH`/`HEIGHT` on this camera's DirectShow driver was silently resetting the
filter graph's format node back to its uncompressed default, undoing the `CAP_PROP_FOURCC` MJPG
request `_open_camera` makes at open time -- `_apply_resolution` now re-asserts
`CAP_PROP_FOURCC` to MJPG *after* width/height are set, on every resolution application (auto and
operator-chosen exact alike), which is when the actual final negotiation happens.

### `validation_status` semantics (decided in `document/field_crosscheck.py`)

- `invalid` — MRZ found but its checksum failed (structurally wrong, not just an OCR mismatch).
- `review` — MRZ or barcode found and checksum/decode succeeded, but cross-checked fields don't
  appear in the OCR text (could be an OCR misread or a genuine forgery — not auto-resolved).
- `valid` — MRZ or barcode cross-check passed cleanly.
- `unverifiable` — neither MRZ nor barcode present; only OCR text available, nothing to
  cross-check it against. This must never be silently upgraded to `valid`.

### Person flags: under-18 / expired document (`document/field_crosscheck.py::check_person_flags`)

Independent of `validation_status` — a document can be structurally authentic
(`validation_status='valid'`) and still trip one of these; neither flag ever changes
`validation_status` itself. `is_minor` comes from the extracted `birth_date`; `is_expired` from
the extracted `expiry_date` -- both MRZ documents and driving licences (`dl_field_parser.py`,
field 4b) can populate this now. Both dates are parsed via
`document/dates.py::parse_mrz_style_date`, which is deliberately asymmetric about century
inference for the MRZ 6-digit `YYMMDD` case: birth dates assume the past (a 2-digit year greater
than the current one implies 19XX), expiry dates always resolve to 20XX (documents don't have
multi-decade validity and the MRZ format itself postdates 1980) -- moot for driving licences,
whose OCR-extracted dates are already unambiguous 4-digit-year ISO strings. Stored under
`validation_detail_json`'s `person_flags` key and shown as a red banner directly on the scan row
(not hidden behind "Details"), and included in the exported HTML report (`ui/scan_export.py`).

### `match_result` semantics (`face/matcher.py` + `ui/main_window.py`)

Stored as one of `match` / `no_match` / `override_match` / `override_no_match`. The `override_*`
values mean an operator changed the result after the automatic one was computed. Overrides are
applied from the history list (`storage/models.py::update_match_result`), not at capture time —
`HistoryRowWidget` reconstructs the current override-combo selection from whichever of these
four strings is currently stored, which means the original automatic result is not separately
recoverable once an override has been applied (accepted limitation, not a bug).

### Derived display fields are computed, not stored

First name, last name, birth date, and age shown in the history list are **not** separate DB
columns — they're derived on the fly in `ui/main_window.py::_derive_display_fields` by parsing
the `name`/`surname`/`birth_date` keys out of the scan's stored `doc_fields_json` (MRZ fields, or
`dl_field_parser.py`'s heuristic fields for an MRZ-less driving licence; empty `"{}"` when
neither found anything). Date parsing and age calculation (`parse_mrz_style_date`,
`calculate_age`) live in `document/dates.py`, shared with `field_crosscheck.py::check_person_flags`
above rather than duplicated — update both call sites' expectations together if this changes.

### Database

SQLite, schema fixed in `storage/db.py::initialize_schema` (two tables: `scans`, `settings` —
see that function for the authoritative column list; don't infer it from `ScanRecord` alone,
which is the read-side dataclass). `settings` is a plain key/value table with typed accessor
functions in `config/settings.py` (defaults: 90-day retention, 0.38 face-match threshold, camera
indices 0/1 — the threshold is explicitly flagged in a code comment there as an unvalidated
starting point, not a calibrated production value). No app-level encryption of the DB or images
— relies entirely on BitLocker; see README "Security posture" before changing this assumption.

`scans.validation_detail_json` is an unstructured, additively-grown JSON blob, not a fixed
schema — it currently holds `notes` and `field_comparisons` (from `determine_validation_status`),
`classified_from` (from `doc_classifier.py`), `image_quality` (from `image_quality.py`), and
`person_flags` (from `check_person_flags`). Any new capture-time signal that isn't its own
top-level column belongs here as another key, read defensively with `.get()` — old rows written
before a given key existed will lack it.

### Other windows and operator-facing features

- `ui/settings_window.py` — data storage locations (via `config/app_paths.py`), manual purge,
  camera-to-side assignment, capture keyboard shortcut, the live prerequisites checklist, and
  triggering a backup (`storage/backup.py`, a single zip of the DB + all images for an operator
  to copy off-machine — supplements BitLocker, doesn't replace it). Its own window size/position
  persists across restarts (`_restore_window_geometry`/`_save_window_geometry`, under the
  `settings_window_geometry` setting key), mirroring `ui/main_window.py`'s identical mechanism for
  its own window (`window_geometry`) — the two are independent settings, one per window.
- `ui/calibration_window.py` — captures a series of known-good same-person face-match score
  samples on the operator's own hardware/lighting and suggests a threshold from that
  distribution; scores are kept in memory only for the session and never written to the `scans`
  table, so calibration captures never pollute real scan history.
- `ui/scan_export.py` — builds one self-contained HTML report per scan (images as base64 data
  URIs) for an operator to print or save as PDF via the browser; no PDF-generation dependency.
- `ui/flag_icons.py` + `document/country_flags.py` — drawn (non-emoji) flag icons for a curated
  set of ~24 countries, keyed off the MRZ nationality/issuing-country alpha-3 code converted to
  alpha-2; unrecognized/uncurated codes show a neutral grey tile rather than a guess.

### Windows-specific choices worth knowing before "fixing" them

- Camera enumeration/opening always passes `cv2.CAP_DSHOW` explicitly (`capture/camera_manager.py`,
  `prerequisites.py::check_cameras`) rather than the OpenCV default (Media Foundation), which is
  slow/unreliable when repeatedly opening and closing device indices during probing.
- `prerequisites.py::install_tesseract` installs via `winget install --id UB-Mannheim.TesseractOCR`
  and then sets `pytesseract.pytesseract.tesseract_cmd` explicitly to the known install path
  rather than relying on a PATH refresh, so OCR works immediately in the same running process.

### PyInstaller build requires `pyinstaller_hooks/hook-insightface.py`

`--additional-hooks-dir=pyinstaller_hooks` (see README "Building a standalone EXE") is not
optional. InsightFace's `insightface/data/pickle_object.py::get_object()` loads
`meanshape_68.pkl` (used only by the 3D-68 landmark model's pose estimation, part of buffalo_l)
by hand via a file path, not an import — with an explicit `if sys.frozen: base_dir =
sys._MEIPASS` branch that expects the file at `<bundle root>/objects/meanshape_68.pkl`, a
different layout than the installed package's own `insightface/data/objects/meanshape_68.pkl`.
PyInstaller's default import-graph analysis has no way to discover this, so a build without the
hook completes fine and the app starts and loads the face model fine — the crash only surfaces
on the *first actual face capture* (`AttributeError: 'NoneType' object has no attribute 'shape'`
deep inside `insightface/utils/transform.py::estimate_affine_matrix_3d23d`), which is why
"the app launched, looked fine" isn't sufficient EXE verification — see README's "click Capture &
Process at least once" note.

### Portable deployment (`config/bundled_deps.py`)

Tesseract and the InsightFace model can each optionally be bundled as a plain folder copy next to
the app itself (`tesseract\` and `models\buffalo_l\`, next to `main.py` in a source checkout or
next to `Kontroll.exe` in a PyInstaller build) instead of relying on a machine-wide install —
see README "Portable / USB-stick deployment" for the operator-facing how-to. `check_tesseract`/
`check_insightface_model` (`prerequisites.py`) and `load_face_analysis_model` (`face/detector.py`)
all check this portable location *first*, falling back to the previous machine-wide defaults
(PATH / `C:\Program Files\Tesseract-OCR\` / `%USERPROFILE%\.insightface\`) so a non-portable
install keeps working unchanged. This is deliberately the opposite direction of
`config/app_paths.py`: that module keeps *writable* data under LOCALAPPDATA specifically because
the install folder isn't guaranteed writable, whereas these two are read-only assets an operator
explicitly opts into placing alongside the app, so resolving them relative to `sys.executable`
(frozen) / the project root (source) is exactly what makes the whole folder portable.

## Known limitations (see README.md for full detail)

- Face-match threshold (0.38) and the barcode/OCR word-overlap heuristic threshold (0.5, in
  `document/field_crosscheck.py`) are both unvalidated starting points, not calibrated values —
  `ui/calibration_window.py` helps an operator pick a better face-match threshold from real
  same-desk samples, but ships no equivalent tool for the barcode/OCR threshold.
- PDF417 field-level parsing for Swedish driving licences is not implemented (no real sample was
  available to validate against) — `barcode_parser.py` returns raw decoded text only for that
  case. US/Canadian driver's licences and ID cards *do* get field-level parsing, since their
  PDF417 payload follows the published AAMVA standard rather than an unvalidated one — see
  `document/aamva_parser.py`.
- Curated flag icons (`ui/flag_icons.py`) cover ~25 countries (24 plus the US); others show a
  neutral grey tile — this is cosmetic only, independent of which countries' documents the OCR/
  MRZ/AAMVA/EU-driving-licence pipelines actually support (see `config/ocr_languages.py` for the
  OCR language set and `document/doc_classifier.py` for supported document types).
- `document/eu_id_card_parser.py` (the non-Swedish EU/EEA national-ID-card fallback, anchored on
  EU Regulation 2019/1157's mandatory English field labels) is a heuristic best-effort parser with
  no real non-Swedish national ID card OCR sample available to validate its same-line/next-line
  label-to-value heuristic against — same caveat as `dl_field_parser.py`/`barcode_parser.py`. A
  parse failure degrades gracefully to '-' placeholders, same as any other document with no
  structured fields.
- Manual camera exposure/brightness control (`ui/settings_window.py`'s Cameras tab,
  `capture/camera_manager.py::ExposureSettings`) depends on the camera/driver actually honoring
  `cv2.CAP_PROP_AUTO_EXPOSURE`/`CAP_PROP_EXPOSURE`/`CAP_PROP_BRIGHTNESS` over DirectShow, which
  varies by device — a camera that ignores these simply keeps behaving as it did before, same
  accepted-limitation pattern as `set_exact_resolution`'s own resolution fallback. The slider
  ranges are reasonable starting points, not calibrated per-device.

## App standards

This repo follows the app standards in `~/Desktop/code/tools/standards/STANDARDS.md` (rule IDs,
severities; decisions and waivers in `.claude-standards`). `claude-check --repo .` checks it.
