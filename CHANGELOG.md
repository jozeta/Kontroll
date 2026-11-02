# Changelog

All notable changes to Kontroll are listed here, newest first.

The project has no tagged releases yet, so changes are grouped by date. Entries before November 2026 were reconstructed from the commit history.

## 2026-11-02

- Security: Pillow 12.3 → 13.0, fixing 41 Dependabot alerts (5 critical, 23 high). Pillow 13 needs Python 3.11+, which the app already requires.
- Dependencies: onnxruntime 1.23 → 1.31, PyQt6 6.11 → 6.12, numpy 2.2 → 2.5, OpenCV 4.12 → 4.14 (numpy 2.5 needs OpenCV 4.13+). Verified on Linux with OCR, MRZ, PDF417 and face matching; webcam capture (Windows DirectShow) still needs a check on Windows.
- Add Dependabot config, security policy

## 2026-11-01

- Ignore macOS, Claude Code and secret files

## 2026-09-11

- Fix history-list performance/UX bugs, add image editor improvements, and expand flagged-persons/statistics/status features
- Pick the sharpest of several recent document-camera frames for OCR, not just the newest one

## 2026-09-04

- Speed up camera detection startup and paginate scan history

## 2026-08-31

- Add self-healing camera backend switching, image editor, and statistics dashboard

## 2026-08-28

- Add structured document parsers, card detection, person/flag matching, and new operator UI windows
- Wire in structured parsing, card detection, watch-list flagging, and visit history

## 2026-08-21

- Add operator tooling, document/age/expiry checks, and portable deployment
- Fix frozen-EXE crash on first face capture (missing InsightFace data file)
- Extract driving licence expiry date so the expired-document flag works for them

## 2026-08-17

- Stage 1-2: project scaffolding, storage layer, and settings
- Ignore .claude/ harness state directory
- Stage 3: camera capture layer (enumeration, dual handle management, manual still capture)
- Stage 4: face pipeline (InsightFace detector/embedder, cosine-similarity matcher)
- Stage 5: document pipeline (OCR, MRZ checksum validation, PDF417 decoding, classification, crosscheck)
- Stage 6: retention job (startup purge of expired scans and image files)
- Stage 7: UI (single-screen main window wiring cameras, capture, and results)
- Stage 8: integration (main.py bootstraps storage, retention, face model, cameras, and UI)
- Add prerequisite checker: startup checks for Tesseract, InsightFace model, cameras, and packages with install/download prompts
- Remove stray log file accidentally committed
- Rebuild UI: single capture-and-process button, scan history list with name/DOB/age and match icon
- Add CLAUDE.md
