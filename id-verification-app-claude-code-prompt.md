# Build Prompt: Local ID Verification App (Windows 11, Python)

## Project Summary

Build a local, offline-first Windows 11 desktop application in Python that verifies a person's identity by matching a live webcam photo of their face against the photo on a scanned ID document (passport, national ID card, or driving licence), and validates the authenticity of the document itself. The app is used at a staffed desk — an operator manually triggers each capture and reviews each result. No data leaves the local machine.

## Core Requirements

1. **Two webcam inputs**: one dedicated to capturing a person's face, one dedicated to capturing an ID document. Both are generic USB UVC webcams — do not assume a specific model, use OpenCV's standard `VideoCapture` interface with a camera picker/selector since device indices can shift.
2. **Manual capture**: both captures are triggered by explicit operator button presses. No auto-detection/auto-trigger logic is needed for v1.
3. **Face matching**: generate face embeddings for both captured images and compute similarity. Present both an automated pass/fail badge AND the raw numeric similarity score, and allow the operator to manually override the automated result.
4. **Document validation**: extract and validate data from the ID document using:
   - OCR (Tesseract via `pytesseract`) for visible text fields
   - MRZ (Machine Readable Zone) parsing and checksum validation for documents that have one (most passports, many national ID cards) — ICAO 9303 standard
   - Barcode parsing (PDF417) for documents that use one (e.g. Swedish driving licences)
   - Cross-checking of OCR-extracted fields against MRZ/barcode-extracted fields where both exist, flagging mismatches
   - Document type and issuing country classification
5. **Broad document support**: must handle Swedish IDs, passports, and driving licences as the primary case, but should be architected to reasonably extend to other countries' documents. Many countries' documents will have no MRZ and no barcode — the app must clearly degrade to "OCR-only, unverifiable via cross-check" rather than silently presenting these as equally trustworthy as MRZ-verified documents.
6. **Storage**: every scan session (face image, document image, embeddings, extracted fields, match result, validation result, timestamp) is persisted to a local SQLite database.
7. **Data retention**: a configurable retention period (in days), stored in app settings, with a background/startup job that purges scan records and associated image files older than that period.
8. **UI**: a single combined screen showing both camera live previews, capture buttons for each, and the match/validation results (with an expandable detail view for the per-check validation breakdown) — not a multi-step wizard.
9. **Logging**: basic error/crash logging only. No requirement for a full per-scan operator audit trail in v1.
10. **Security posture**: no application-level encryption of the database or stored images is required for v1 — the app relies on OS-level disk encryption (BitLocker) on the deployment machine. Document this assumption clearly in code comments and README as an accepted residual risk, not an oversight.

## Technology Stack (fixed — do not substitute without discussion)

| Concern | Choice |
|---|---|
| Language | Python 3.11+ |
| UI framework | PyQt6 (needed for reliable live video widget embedding) |
| Camera capture | OpenCV (`cv2.VideoCapture`) |
| Face detection + embedding | InsightFace (ONNX Runtime), `buffalo_l` model pack |
| Face similarity | Cosine similarity between embeddings, configurable threshold |
| OCR | Tesseract via `pytesseract` (assume Tesseract binary is installed separately; document the dependency) |
| MRZ parsing | `mrz` library, or a small hand-rolled ICAO 9303 parser if the library proves inadequate — implement checksum validation regardless |
| Barcode/PDF417 | `pdf417decoder` or `zxing-cpp` bindings — test against a real Swedish driving licence sample if available, do not assume US/AAMVA barcode formats apply unmodified |
| Database | `sqlite3` (Python stdlib) |
| Config | A simple local settings file (JSON or SQLite table) — must include retention period in days, face-match similarity threshold, and camera device selections, all editable without touching code |

## Project Structure

```
id-verify-app/
├── main.py
├── ui/
│   └── main_window.py
├── capture/
│   ├── camera_manager.py
│   └── frame_capture.py
├── face/
│   ├── detector.py
│   ├── embedder.py
│   └── matcher.py
├── document/
│   ├── ocr.py
│   ├── mrz_parser.py
│   ├── barcode_parser.py
│   ├── field_crosscheck.py
│   └── doc_classifier.py
├── storage/
│   ├── db.py
│   ├── models.py
│   └── retention.py
├── config/
│   └── settings.py
├── logs/
│   └── app.log
├── requirements.txt
└── README.md
```

## Database Schema

```sql
CREATE TABLE scans (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    operator TEXT,
    face_image_path TEXT,
    doc_image_path TEXT,
    face_embedding BLOB,
    match_score REAL,
    match_result TEXT,             -- 'match' / 'no_match' / 'override_match' / 'override_no_match'
    doc_type TEXT,                 -- 'passport' / 'national_id' / 'drivers_licence'
    doc_country TEXT,
    doc_fields_json TEXT,
    validation_status TEXT,        -- 'valid' / 'review' / 'invalid' / 'unverifiable'
    validation_detail_json TEXT
);

CREATE TABLE settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

Note the `'unverifiable'` validation status — this must be used (not `'valid'` or a default pass) whenever a document has no MRZ and no barcode to cross-check against, so OCR-only documents are never visually indistinguishable from cross-validated ones in the UI.

## Coding Conventions (follow strictly)

- Prioritize clarity and readability over cleverness or compactness. Prefer explicit code over implicit magic.
- Scale structure to complexity: simple utility functions can be flat; the face/document pipelines should be modular functions with clear separation of concerns, matching the module breakdown above.
- Add inline comments throughout that explain the *intent and logic* of each meaningful step — not comments that just restate the code syntactically.
- Provide a one-line summary comment/docstring before each function or module describing what it does.
- **Do not add error handling / try-except blocks by default.** Only add error handling where explicitly requested in a follow-up. Let failures surface naturally during development.
- Use type hints on function signatures throughout.

## Build Order (please build in this sequence, confirming each stage before moving to the next)

1. **Project scaffolding**: folder structure, `requirements.txt`, empty module stubs, `README.md` with setup instructions (including the external Tesseract binary dependency).
2. **Storage layer**: `storage/db.py` (schema creation), `storage/models.py` (insert/query scan records), `config/settings.py` (load/save settings including retention period, threshold, camera selection).
3. **Camera capture layer**: `capture/camera_manager.py` (enumerate + select two UVC cameras), `capture/frame_capture.py` (grab a still frame on manual trigger).
4. **Face pipeline**: `face/detector.py` + `face/embedder.py` (InsightFace integration), `face/matcher.py` (cosine similarity + threshold logic against configurable setting).
5. **Document pipeline**: `document/ocr.py` (Tesseract text extraction), `document/mrz_parser.py` (MRZ detection + checksum validation), `document/barcode_parser.py` (PDF417 decode), `document/doc_classifier.py` (type/country detection), `document/field_crosscheck.py` (cross-check logic + `unverifiable` handling for documents without MRZ/barcode).
6. **Retention job**: `storage/retention.py` — purge scans + image files older than the configured N days, run on app startup.
7. **UI**: `ui/main_window.py` — single-screen PyQt6 layout wiring together both camera previews, capture buttons, and results panel (summary badge + expandable detail for both face match and document validation).
8. **Integration**: `main.py` wiring everything together into a runnable app.

At each stage, stop and summarize what was built plus any open questions or assumptions made, before proceeding to the next stage.

## Known Risks to Flag During Implementation (do not silently work around these — surface them)

- PDF417 barcode structure on Swedish driving licences may not match AAMVA-style US formats exactly; validate against a real sample early rather than assuming a library "just works."
- Lighting/glare from laminated ID cards under normal desk lighting will affect OCR and MRZ read reliability — consider a basic blur/glare quality check on the captured document frame before running the extraction pipeline, and flag this as a possible v2 addition if not built now.
- Face matching accuracy will vary between a live webcam face capture and a printed/laminated ID photo (different lighting, color reproduction, compression) — this is a known cross-domain matching challenge; do not assume webcam-quality thresholds tuned on same-domain face pairs will transfer directly.
- Any document without a machine-readable zone or barcode must be marked `'unverifiable'` in validation_status, never silently treated as validated.
