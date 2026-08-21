# Kontroll — Local ID Verification App

A local, offline-first Windows 11 desktop application (window title: **Kontroll**) that verifies
a person's identity by matching a live webcam photo of their face against the photo on a scanned
ID document (passport, national ID card, or driving licence), and validates the authenticity of
the document itself. Intended for use at a staffed desk, where an operator manually triggers each
capture and reviews each result. No data leaves the local machine at runtime.

## Setup

Every time the app starts, `main.py` runs an automatic prerequisite check (`prerequisites.py`)
before opening the main window: it looks for Tesseract, the downloaded InsightFace model, and
two connected cameras, and for anything missing that it can fix itself (Tesseract, the model,
or a Python package), it asks via a dialog whether to install/download it now. Cameras are
reported but never prompted on, since missing hardware can't be installed. The manual steps
below are what that checker automates, and remain here for anyone who prefers to set the
machine up ahead of time, without clicking through dialogs at first launch, or without a
Python environment at all (see "Building a standalone EXE" and "Setting up Tesseract/InsightFace
with only the EXE" below).

### 1. Python environment

Requires Python 3.11+ (developed and tested against 3.13).

**Use the official python.org installer, not the Microsoft Store package.** The Store build of
Python runs under Windows' UWP/MSIX app-container file-system virtualization: any file the app
writes under `%LOCALAPPDATA%` (the database, images, logs — see "Where data is stored" below)
gets silently redirected to a hidden per-package cache folder instead of the real path shown in
Settings, and a Store-Python venv generally cannot be frozen into a clean standalone EXE either.
Download the installer from https://www.python.org/downloads/windows/ (or `winget install --id
Python.Python.3.13 -e`) and confirm `python -c "import sys; print(sys.executable)"` does **not**
print a path under `...\WindowsApps\...` before proceeding.

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

  The app itself already does this automatically once it finds Tesseract at the default install
  path above (see `prerequisites.py::check_tesseract`), so this manual step is only needed if
  Tesseract is installed somewhere else.

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

## Building a standalone EXE

The app can be packaged into a distributable Windows executable with
[PyInstaller](https://pyinstaller.org/), so it can run on a machine with no Python installation
at all.

```
venv\Scripts\activate
pip install pyinstaller
pyinstaller --onedir --name Kontroll --noconfirm --additional-hooks-dir=pyinstaller_hooks main.py
```

`--additional-hooks-dir=pyinstaller_hooks` is required, not optional: `pyinstaller_hooks/hook-insightface.py`
bundles one data file (`meanshape_68.pkl`) that InsightFace's landmark model loads by hand at
runtime rather than importing, so PyInstaller's normal import-graph analysis never discovers it
on its own. Without this flag, the build still completes and the app still starts and loads the
face model fine — the crash only happens on the *first actual face capture*, deep inside
InsightFace with `AttributeError: 'NoneType' object has no attribute 'shape'`, which makes it an
easy step to accidentally skip. See `pyinstaller_hooks/hook-insightface.py`'s own docstring for
the full mechanism.

This produces `dist\Kontroll\Kontroll.exe` plus a `dist\Kontroll\_internal\` folder holding the
bundled interpreter, packages, and Qt plugins — copy the whole `dist\Kontroll\` folder together
when distributing it, not just the `.exe`. `--onedir` (rather than `--onefile`) is intentional:
a `--onefile` build re-extracts itself into a temp directory on every launch and is meaningfully
slower to start; `--onedir` starts instantly and is easier to troubleshoot if something's
missing from the build.

The `dist\Kontroll\` folder itself can be freely copied/renamed to wherever you're staging a
build for distribution (e.g. a `release\` folder) — nothing in it depends on the `dist\Kontroll\`
path specifically, only on the `tesseract\`/`models\` layout described in "Portable / USB-stick
deployment" below staying relative to wherever `Kontroll.exe` ends up.

**Must be built with the python.org Python** described in "Python environment" above — a
Microsoft Store-based venv cannot produce a clean standalone build.

The app already accounts for running frozen: `prerequisites.py::check_python_packages` detects
`sys.frozen` and skips the (meaningless, once bundled) "install missing packages via pip" flow,
and every data/log path (see below) resolves via `%LOCALAPPDATA%` rather than a path relative to
the script/EXE location, which would otherwise land inside PyInstaller's temp extraction folder.

After building, verify it actually runs before distributing it: launch `dist\Kontroll\Kontroll.exe`
directly and confirm the main window opens and the log file (Settings → log file location) is
being written to. **Also click "Capture & Process" at least once** — startup alone doesn't
exercise the face detection code path, which is exactly where a missing PyInstaller data file
(see the `--additional-hooks-dir` note above) shows up.

### Setting up Tesseract OCR and the InsightFace model with only the EXE

If you've been handed just the `Kontroll.exe` folder (no Python, no source checkout), Tesseract
and the InsightFace model still aren't bundled into the EXE — they're external, machine-level
dependencies either way. The built-in prerequisite check (see "Setup" above) runs exactly the
same way in the frozen EXE and will offer to install/download both automatically on first launch,
**provided the machine has internet access and `winget` available** (present by default on
Windows 11). If that's the case, just run `Kontroll.exe` and click "Install/download it now?"
when prompted for each.

If automatic install isn't available (offline machine, no `winget`, or the automatic attempt
fails), fall back to the manual steps from "Setup" above — they don't require Python or the
source checkout, since both are just downloading a normal Windows installer / a model folder:

- **Tesseract OCR**: download and run the UB Mannheim installer
  (https://github.com/UB-Mannheim/tesseract/wiki) directly on the target machine. No further
  configuration needed if installed to the default location
  (`C:\Program Files\Tesseract-OCR\`) — the app already checks there.
- **InsightFace `buffalo_l` model**: on a machine *with* internet access, either run the app
  once (source or EXE) to trigger the automatic download, or download the pack directly; then
  copy the resulting `buffalo_l` folder to `%USERPROFILE%\.insightface\models\buffalo_l\` on the
  offline target machine, under the same Windows user account that will run `Kontroll.exe`.

### Portable / USB-stick deployment

By default, Tesseract and the InsightFace model install to machine-wide locations (`C:\Program
Files\...`, `%USERPROFILE%\.insightface\...`) that don't travel with the app. To make the whole
`dist\Kontroll\` folder self-contained — copyable to a USB stick or a different drive/path with
no reinstallation and no reconfiguration — place a copy of each dependency directly inside it:

```
dist\Kontroll\
  Kontroll.exe
  tesseract\
    tesseract.exe
    tessdata\
    ...(rest of a normal Tesseract install)
  models\
    buffalo_l\
      ...(the *.onnx files)
```

The app checks these exact locations first, before falling back to PATH / Program Files /
`%USERPROFILE%\.insightface` (see `config/bundled_deps.py`) — nothing needs to be configured;
having the files there is enough, and this works the same way running from source with `tesseract\`
and `models\` placed next to `main.py` at the project root instead of next to `Kontroll.exe`.

- **To populate `tesseract\`**: install Tesseract once anywhere (via the app's own prerequisite
  check, or the UB Mannheim installer above), then copy the entire resulting install folder
  (normally `C:\Program Files\Tesseract-OCR\`) into `tesseract\`. Tesseract's Windows build is
  otherwise self-contained, so a plain folder copy works.
- **To populate `models\`**: on a machine with no prior InsightFace install anywhere, clicking
  "Install/download it now?" for the model now downloads directly into this portable location by
  default (see `prerequisites.py::download_insightface_model`). If it's already downloaded at the
  legacy location instead, copy `%USERPROFILE%\.insightface\models\buffalo_l\` to
  `models\buffalo_l\`.

An existing non-portable install (Tesseract on PATH, model at the legacy location) keeps working
unchanged if you don't do this — bundling is opt-in, not required.

## Where data is stored

The database, saved face-capture images, saved document-capture images, and the log file each
live in their own folder under `%LOCALAPPDATA%\Kontroll\` by default:

```
%LOCALAPPDATA%\Kontroll\data\app.db          Scan records (SQLite)
%LOCALAPPDATA%\Kontroll\data\faces\          Saved live face-capture images
%LOCALAPPDATA%\Kontroll\data\documents\      Saved document-capture images
%LOCALAPPDATA%\Kontroll\logs\app.log         Log file (see "Logging" below)
```

Each of these four locations is independently configurable from **Settings → Data storage
locations** — an operator might, for example, want scan images on a larger drive or a network
path while keeping the database and logs local. Changing one only takes effect on the *next*
restart (the move itself happens then, before anything has the old files open); the settings
window shows a "pending move" notice for anything changed but not yet applied. If a configured
folder is missing at startup (e.g. it lived on a drive that isn't currently connected), the app
asks whether to recreate it automatically — declining falls back to the default location above
rather than leaving the app unable to start.

A configurable retention period (default 90 days, editable in Settings) additionally controls
how long scan records are kept regardless of location; records and image files older than the
retention period are purged automatically on every app startup, and an operator can also
manually purge everything older than a chosen number of days (30/90/180/365) from Settings at
any time.

**Settings → Backup** writes a single zip archive containing a consistent snapshot of the
database plus every saved face/document image (`storage/backup.py`), for an operator to copy to
a USB drive or network share. This is a supplement to BitLocker (see "Security posture" below),
not a replacement — BitLocker protects data at rest on this machine, but does nothing against
drive failure or the machine itself being lost.

## Logging

Every significant operator action is logged: captures and their result summary, deletes,
overrides, purges, camera/shortcut/path changes, prerequisite checks and install attempts, and
app start/shutdown — in addition to warnings and uncaught exceptions. Log lines use a plain
timestamped text format (`2026-08-19 22:55:42,464 INFO     module.name: message`), one line per
event, so the file can be opened and read directly by an operator without any tooling.

The log file rotates at 5MB (keeping 3 older backups, ~20MB ceiling total) so continuous
day-to-day action logging can't grow it unbounded. Its folder is configurable the same way as
the three data folders above (Settings → Data storage locations → Log file).

## Validation checks

Each scan produces two independent results, shown side by side and never conflated: a **face
match** (automated pass/fail badge plus the raw cosine-similarity score, always overridable by
the operator from the history list) and a **document validation status** (`valid` / `review` /
`invalid` / `unverifiable` — see `document/field_crosscheck.py` for exactly what drives each).

On top of that, every scan is separately checked for two business-rule conditions that are
independent of document authenticity — a document can be internally consistent and unaltered
while still failing one of these:

- **Under 18**: flagged when the extracted birth date puts the person's current age below 18.
- **Expired document**: flagged when the extracted expiry date is in the past (MRZ documents
  only — driving licences have no machine-readable expiry field to check against, see
  `document/dl_field_parser.py`).

Either condition shows a red **⚠ UNDER 18 / EXPIRED** banner directly on that scan's row in the
history list (not hidden behind "Details"), and is included in the exported HTML report.

## Security posture (read before deployment)

This application does **not** encrypt its SQLite database or stored images at the
application level. This is a deliberate v1 design decision, not an oversight: the app is
built to run on a desk machine where **OS-level disk encryption (BitLocker) is assumed to
be enabled**, which protects data at rest against loss/theft of the physical machine or
drive. If BitLocker (or an equivalent full-disk encryption) is not enabled on the deployment
machine, the biometric and document data stored by this app is **not** protected at rest,
and BitLocker should be enabled before the app is used with real data. This assumption is
also documented as a code comment at the point where the database is opened (`storage/db.py`).

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
- **Document image quality (blur/glare) and face-match threshold calibration are both
  unvalidated heuristics.** `document/image_quality.py` flags a captured document frame as
  blurry (Laplacian-variance proxy) or glare-heavy and shows a passive banner on that scan's
  thumbnail, purely so a poor OCR/MRZ read can be attributed to a visibly bad source image
  rather than looking like an unexplained failure — it never blocks or retries a capture. Both
  its thresholds, and the face-match threshold above, are starting points; **Settings →
  Calibration** (`ui/calibration_window.py`) lets an operator capture a series of known-good
  same-person score samples on the actual desk hardware/lighting and suggests a threshold from
  that distribution, rather than shipping a single fixed value as ground truth.
- **Curated flag icons cover common countries only.** The scan history's country flag icon
  (`ui/flag_icons.py`) draws accurate flags for a curated set of ~24 frequently-seen (mostly
  European) countries; any other recognized country code shows a neutral grey tile rather than
  a guessed or wrong-looking flag design.

## Project structure

```
main.py                  Entry point: logging, config, DB, retention job, launches the UI
prerequisites.py         Startup checks for Tesseract/InsightFace model/cameras/packages, with install/download prompts
ui/
  main_window.py          PyQt6 main window (previews, capture, history list, status bar)
  settings_window.py       Settings: data locations, purge, camera assignment, shortcut, prerequisites, backup
  calibration_window.py    Face-match threshold calibration against known-good score samples
  scan_export.py           Builds a self-contained printable HTML report for one scan
  flag_icons.py            Drawn (non-emoji) national flag icons for the history list
capture/                  Camera enumeration and manual still-frame capture
face/                     Face detection, embedding, and similarity matching
document/
  ocr.py, mrz_parser.py, barcode_parser.py, doc_classifier.py, field_crosscheck.py
  dl_field_parser.py       OCR-heuristic name/DOB extraction for MRZ-less driving licences
  country_flags.py         ISO 3166-1 alpha-3 -> alpha-2 lookup used by ui/flag_icons.py
  image_quality.py         Blur/glare check on the captured document frame
  dates.py                 Shared MRZ/ISO date parsing and age calculation (birth date, expiry date)
storage/                  SQLite schema, typed query functions, retention purge job, backup archive
config/
  settings.py              DB-backed settings (retention days, match threshold, camera indices, capture shortcut)
  app_paths.py              Resolves/relocates the four LOCALAPPDATA-based data & log directories
  bundled_deps.py           Resolves optional portable tesseract\/models\ copies next to the app (see "Portable / USB-stick deployment")
```

Runtime-generated data (database, captured images, logs) is **not** stored under the project
directory — see "Where data is stored" above.

## License

MIT — see [LICENSE](LICENSE).
