# Local ID Verification App

A local, offline-first Windows 11 desktop application that verifies a person's identity by
matching a live webcam photo of their face against the photo on a scanned ID document
(passport, national ID card, or driving licence), and validates the authenticity of the
document itself. Intended for use at a staffed desk, where an operator manually triggers
each capture and reviews each result. No data leaves the local machine at runtime.

## Setup

### 1. Python environment

Requires Python 3.11+ (developed and tested against 3.13).

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Tesseract OCR (external dependency)

OCR of visible document fields is performed by [Tesseract OCR](https://github.com/tesseract-ocr/tesseract),
which is **not** a Python package — it is a separate binary that must be installed on the
machine. This app calls it via `pytesseract`, which shells out to the binary.

- Install the Windows build from the UB Mannheim distribution:
  https://github.com/UB-Mannheim/tesseract/wiki
- By default, `pytesseract` expects `tesseract.exe` to be on the system `PATH`. If it isn't
  (or you don't want to modify `PATH`), set the path explicitly before use:

  ```python
  import pytesseract
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

### 3. InsightFace face-recognition model (one-time download)

Face detection and embedding use the InsightFace `buffalo_l` model pack (~330 MB). This
model is **not** bundled in the `insightface` pip package — it is downloaded automatically
the first time `insightface.app.FaceAnalysis(name="buffalo_l")` is constructed, and cached
to:

```
%USERPROFILE%\.insightface\models\buffalo_l\
```

This download requires an internet connection, but it is a one-time setup step, not a
runtime dependency — it does not conflict with the app's offline-first design, which
describes normal operation at the desk (no data or images are ever sent anywhere), not the
initial machine setup.

**For a machine with no internet access at all**: download the `buffalo_l` model pack on a
machine that does have internet access, then copy the resulting `buffalo_l` folder to the
same path (`%USERPROFILE%\.insightface\models\buffalo_l\`) on the offline machine before
first launch.

### 4. Two USB webcams

The app expects two generic UVC-class USB webcams: one pointed at the operator/subject for
the face capture, one pointed at the document for the ID capture. Device index assignment
(which OS-visible camera is "camera 0" vs "camera 1") is not guaranteed to be stable across
reboots or reconnects — use the camera picker in Settings to (re-)assign face/document
cameras to the correct physical device index if previews appear swapped or blank.

## Running

```
venv\Scripts\activate
python main.py
```

## Data retention

Scan records (images, embeddings, extracted fields, and results) are stored locally in
`data/app.db` and `data/faces/` / `data/documents/`. A configurable retention period
(default 90 days, editable in Settings) controls how long this data is kept; records and
image files older than the retention period are purged automatically on app startup.

## Security posture (read before deployment)

This application does **not** encrypt its SQLite database or stored images at the
application level. This is a deliberate v1 design decision, not an oversight: the app is
built to run on a desk machine where **OS-level disk encryption (BitLocker) is assumed to
be enabled**, which protects data at rest against loss/theft of the physical machine or
drive. If BitLocker (or an equivalent full-disk encryption) is not enabled on the deployment
machine, the biometric and document data stored by this app is **not** protected at rest,
and BitLocker should be enabled before the app is used with real data. This assumption is
also documented as a code comment at the point where the database and image files are
written (`storage/db.py`).

## Known limitations (v1)

- **Face-match threshold is a starting default, not a validated production value.** The
  default cosine-similarity threshold (0.38) is based on published guidance for the
  underlying ArcFace-family model on same-domain photo pairs; this app's actual comparison
  is cross-domain (live webcam capture vs. a printed/laminated ID photo), which tends to
  push genuine-pair similarity scores lower. Recalibrate this threshold against real
  captured sample pairs during pilot use — this is why the app always shows the raw
  similarity score and allows a manual override alongside the automated pass/fail badge,
  rather than only showing a hard pass/fail gate.
- **PDF417 barcode support is not yet validated against a real Swedish driving licence
  sample.** The barcode decoder (`zxing-cpp`) is a general-purpose, actively maintained
  PDF417 reader, but Swedish driving licence barcode encoding has not been confirmed to
  match AAMVA-style (US) formats. Validate against a real sample before relying on this
  path in production; a fallback pure-Python decoder (`pdf417decoder`) is listed
  (commented out) in `requirements.txt` in case it proves necessary.
- **Documents without an MRZ or barcode are marked `unverifiable`, not `valid`.** Many
  countries' ID documents have no machine-readable zone and no barcode. For these, only
  OCR-extracted fields are available, with no independent source to cross-check them
  against. The app always reflects this by setting `validation_status = 'unverifiable'`
  rather than presenting OCR-only documents as equivalently trustworthy to MRZ/barcode
  cross-validated ones.
- **No glare/blur quality check on the captured document frame in v1.** Lighting and glare
  from laminated ID cards under normal desk lighting can degrade OCR/MRZ read reliability.
  A pre-extraction image quality check is a reasonable v2 addition, not built here.
- Basic error/crash logging only (`logs/app.log`) — there is no full per-scan operator audit
  trail in v1.

## Project structure

```
main.py                  Entry point: wires config, DB, retention job, launches the UI
ui/                       PyQt6 main window (single combined screen)
capture/                  Camera enumeration and manual still-frame capture
face/                     Face detection, embedding, and similarity matching
document/                 OCR, MRZ parsing, barcode parsing, classification, cross-checking
storage/                  SQLite schema, typed query functions, retention purge job
config/                   Settings load/save (retention days, match threshold, camera selection)
data/                     Runtime-created: app.db, captured face/document images (not committed)
logs/                     Runtime-created: app.log (not committed)
```
