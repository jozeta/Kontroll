"""Builds a self-contained, printable HTML report for a single scan result.

One .html file with both captured images embedded as base64 data URIs plus a plain-text
validation summary -- an operator opens it in any browser and can use the browser's own
Print / Save-as-PDF if a paper or PDF record is needed, without this app taking on a PDF-
generation dependency (none is in requirements.txt).
"""

import base64
import html
import json
from datetime import datetime
from pathlib import Path

from storage.models import ScanRecord


def _image_data_uri(image_path: str | None) -> str | None:
    if not image_path or not Path(image_path).exists():
        return None
    encoded = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _image_figure_html(image_path: str | None, caption: str) -> str:
    data_uri = _image_data_uri(image_path)
    body = f'<img src="{data_uri}" alt="{html.escape(caption)}">' if data_uri else "<p>No image</p>"
    return f"<figure>{body}<figcaption>{html.escape(caption)}</figcaption></figure>"


def _format_created_at(created_at: str) -> str:
    return datetime.fromisoformat(created_at).strftime("%Y-%m-%d %H:%M:%S")


def build_scan_report_html(scan: ScanRecord, display_name: str, birth_date_display: str, age_display: str) -> str:
    """Render one scan's images and validation summary as a single self-contained HTML document."""
    detail = json.loads(scan.validation_detail_json) if scan.validation_detail_json else {}
    notes = detail.get("notes", [])
    field_comparisons = detail.get("field_comparisons", [])
    person_flags = detail.get("person_flags") or {}

    flag_labels = []
    if person_flags.get("is_minor"):
        flag_labels.append("UNDER 18")
    if person_flags.get("is_expired"):
        flag_labels.append(f"EXPIRED ({person_flags.get('expiry_date_display', '?')})")
    flags_text = " / ".join(flag_labels) or "None"

    notes_html = "".join(f"<li>{html.escape(note)}</li>" for note in notes) or "<li>None</li>"
    comparisons_html = (
        "".join(
            f"<li>[{'OK' if comparison.get('found_in_ocr') else 'MISMATCH'}] "
            f"{html.escape(comparison.get('field_name', ''))}: {html.escape(str(comparison.get('machine_read_value', '')))}</li>"
            for comparison in field_comparisons
        )
        or "<li>None</li>"
    )

    score_text = f"{scan.match_score:.4f}" if scan.match_score is not None else "-"

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Kontroll scan report #{scan.id}</title>
<style>
  body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 2em; color: #111; }}
  h1 {{ font-size: 1.3em; }}
  h2 {{ font-size: 1.05em; margin-top: 1.5em; }}
  .images {{ display: flex; gap: 24px; margin: 1em 0; }}
  .images figure {{ margin: 0; text-align: center; }}
  .images img {{ max-width: 320px; max-height: 240px; border: 1px solid #999; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  td {{ padding: 4px 12px 4px 0; vertical-align: top; }}
  td.label {{ font-weight: bold; white-space: nowrap; }}
  ul {{ margin: 0.2em 0; padding-left: 1.2em; }}
  @media print {{ body {{ margin: 0.5cm; }} }}
</style>
</head>
<body>
<h1>Kontroll scan report #{scan.id} &mdash; {html.escape(_format_created_at(scan.created_at))}</h1>
<div class="images">
  {_image_figure_html(scan.face_image_path, "Live face capture")}
  {_image_figure_html(scan.doc_image_path, "Document capture")}
</div>
<table>
  <tr><td class="label">Name</td><td>{html.escape(display_name)}</td></tr>
  <tr><td class="label">Date of birth</td><td>{html.escape(birth_date_display)} (age {html.escape(age_display)})</td></tr>
  <tr><td class="label">Document type</td><td>{html.escape(scan.doc_type or '-')}</td></tr>
  <tr><td class="label">Document country</td><td>{html.escape(scan.doc_country or '-')}</td></tr>
  <tr><td class="label">Face match result</td><td>{html.escape(scan.match_result or '-')}</td></tr>
  <tr><td class="label">Raw similarity score</td><td>{html.escape(score_text)}</td></tr>
  <tr><td class="label">Document validation status</td><td>{html.escape((scan.validation_status or '-').upper())}</td></tr>
  <tr><td class="label">Flags</td><td>{html.escape(flags_text)}</td></tr>
</table>
<h2>Validation notes</h2>
<ul>{notes_html}</ul>
<h2>Field comparisons</h2>
<ul>{comparisons_html}</ul>
</body>
</html>
"""
