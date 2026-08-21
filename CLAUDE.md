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

There is no automated test suite (no pytest, no `tests/` directory). Verification during
development has been done with throwaway ad-hoc scripts run against the venv interpreter,
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

A single "Capture & Process" button click (`ui/main_window.py:_on_capture_and_process_clicked`)
runs, in order:

1. **Capture** (`capture/frame_capture.py`): grab one still frame from each of the two camera
   handles owned by `capture/camera_manager.py::CameraManager`.
2. **Face pipeline** (`face/`): `detector.py` wraps a single shared
   `insightface.app.FaceAnalysis(name="buffalo_l")` instance (loaded once at startup — expensive,
   never reloaded per-scan) to detect faces in both the live frame and the document frame;
   `embedder.py` picks the highest-confidence face and returns its 512-d normalized embedding;
   `matcher.py` computes cosine similarity against the configurable threshold and returns both
   the raw score and a bool — never just one or the other, since the UI must always show both.
3. **Document pipeline** (`document/`): `image_quality.py` flags a blurry (Laplacian-variance
   proxy) or glare-heavy capture first — passive only, shown as a banner on the scan's thumbnail,
   never blocks or retries → `ocr.py` (Tesseract via pytesseract) → `mrz_parser.py` (locates
   MRZ-shaped lines by character set + fixed line width, then checksum-validates via the `mrz`
   library — TD1/30-char/3-line, TD2/36-char/2-line, TD3/44-char/2-line, widths are unique so
   there's no format ambiguity) → `barcode_parser.py` (PDF417 only, via zxing-cpp, raw text only
   — no structured field parsing, since the Swedish driving licence PDF417 layout is unvalidated
   against a real sample) → `doc_classifier.py` (type/country from MRZ document code, or OCR
   keyword fallback for barcode-only docs) → `field_crosscheck.py::determine_validation_status`
   (the decision point for `validation_status`) and `field_crosscheck.py::check_person_flags`
   (separate under-18/expired-document flags — see below). When no MRZ is found (e.g. a driving
   licence), `dl_field_parser.py` heuristically extracts name/birth-date from the OCR text
   instead, anchored on the EU-harmonized numbered field layout — no expiry date is available
   that way, so the expired-document flag never fires for MRZ-less documents.
4. **Persist**: both images saved to the face/document image directories resolved by
   `config/app_paths.py` (LOCALAPPDATA-based by default, independently operator-configurable —
   *not* a fixed `data/` path relative to the project), one row written via
   `storage/models.py::insert_scan`, then prepended to the on-screen history list.

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
the extracted `expiry_date` (MRZ documents only — driving licences have none, see above). Both
dates are parsed via `document/dates.py::parse_mrz_style_date`, which is deliberately asymmetric
about century inference for the MRZ 6-digit `YYMMDD` case: birth dates assume the past (a 2-digit
year greater than the current one implies 19XX), expiry dates always resolve to 20XX (documents
don't have multi-decade validity and the MRZ format itself postdates 1980). Stored under
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
  to copy off-machine — supplements BitLocker, doesn't replace it).
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
  available to validate against) — `barcode_parser.py` returns raw decoded text only.
- Curated flag icons (`ui/flag_icons.py`) cover ~24 countries; others show a neutral grey tile.
