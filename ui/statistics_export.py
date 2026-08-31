"""Builds a self-contained, printable HTML statistics report -- the Settings > Statistics tab's
"Print Report" button.

Same approach as ui/scan_export.py: one .html file (charts drawn as inline SVG rather than
embedded raster images) that an operator opens in any browser and prints or saves as PDF from
there, so no PDF-generation dependency is needed. Always rendered in a fixed light color scheme
(like ui/scan_export.py's report), independent of the app's own current theme -- it's meant to be
printed or read as a standalone document, not viewed side-by-side with the themed app window.
"""

import html
import math
from datetime import datetime

from document.country_flags import alpha2_for_country_code
from document.statistics import DailyCount, KpiTrends, Statistics, scans_last_30_days_delta_percent, top_n_with_other
from ui.charts import CATEGORICAL_LIGHT, evenly_spaced_indices, trend_display
from ui.flag_icons import ICON_SIZE, flag_icon_data_uri
from ui.statistics_labels import doc_type_color, doc_type_label, flags_and_issues_breakdown, validation_status_color, validation_status_label

_PAGE_STYLE = """
  body { font-family: "Segoe UI", Arial, sans-serif; margin: 2em; color: #111; }
  h1 { font-size: 1.4em; margin-bottom: 0.2em; }
  h2 { font-size: 1.05em; margin-top: 2em; border-top: 2px solid #ccc; padding-top: 0.75em; }
  .generated { color: #666; font-size: 0.85em; margin-bottom: 1.5em; }
  .tiles { display: flex; flex-wrap: wrap; gap: 16px; margin: 1em 0; }
  .tile { border: 1px solid #ccc; border-radius: 6px; padding: 10px 14px; min-width: 170px; }
  .tile .label { font-size: 0.8em; color: #555; }
  .tile .value { font-size: 1.6em; font-weight: bold; }
  .tile .detail { font-size: 0.75em; color: #777; margin-top: 2px; }
  .tile .trend { font-size: 0.75em; font-weight: bold; margin-top: 2px; }
  .chart-row { display: flex; flex-wrap: wrap; gap: 32px; align-items: flex-start; }
  .chart-block figcaption { font-size: 0.8em; color: #555; margin-top: 4px; }
  table.legend { border-collapse: collapse; font-size: 0.85em; }
  table.legend td { padding: 2px 8px 2px 0; }
  .swatch { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }
  table.averages td { padding: 2px 16px 2px 0; }
  @media print { body { margin: 0.5cm; } .chart-row { page-break-inside: avoid; } }
"""


def _tile_html(label: str, value: str, detail: str = "", trend_text: str = "", trend_color: str | None = None) -> str:
    """trend_text/trend_color are normally trend_display's own return value, passed straight
    through -- see ui/charts.py's StatTileWidget, which renders the on-screen equivalent of this
    same tile the same way, so the two never disagree on wording or color."""
    detail_html = f'<div class="detail">{html.escape(detail)}</div>' if detail else ""
    trend_html = f'<div class="trend" style="color:{trend_color or "#777"};">{html.escape(trend_text)}</div>' if trend_text else ""
    return f'<div class="tile"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div>{detail_html}{trend_html}</div>'


def _percent(value: float | None) -> str:
    return f"{value * 100:.1f}%" if value is not None else "—"


def _bar_chart_svg(labels: list[str], values: list[float], color: str, width: int = 640, height: int = 190, max_tick_labels: int = 12) -> str:
    if not values:
        return "<p>No data yet.</p>"
    max_value = max(values) or 1.0
    margin_left, margin_bottom, margin_top, margin_right = 34, 22, 6, 6
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_bottom - margin_top
    count = len(values)
    slot = plot_w / count
    bar_w = max(slot * 0.65, 1.0)
    tick_indices = evenly_spaced_indices(count, max_tick_labels)

    parts = []
    for fraction in (0.0, 0.5, 1.0):
        y = margin_top + plot_h * (1 - fraction)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#ddd" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 4}" y="{y + 3:.1f}" font-size="9" text-anchor="end" fill="#666">{max_value * fraction:.0f}</text>')

    for index, (label, value) in enumerate(zip(labels, values)):
        bar_h = (value / max_value) * plot_h if max_value else 0.0
        x = margin_left + index * slot + (slot - bar_w) / 2
        y = margin_top + (plot_h - bar_h)
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="2" fill="{color}"><title>{html.escape(label)}: {value:g}</title></rect>')
        if index in tick_indices:
            label_x = margin_left + index * slot + slot / 2
            parts.append(f'<text x="{label_x:.1f}" y="{height - 6}" font-size="9" text-anchor="middle" fill="#666">{html.escape(label)}</text>')

    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _horizontal_bar_chart_svg(
    items: list[tuple[str, float]], color: str, width: int = 480, row_height: int = 24, icons: list[str | None] | None = None
) -> str:
    """icons: one data-URI string per row (parallel to `items`), e.g. from
    ui/flag_icons.py::flag_icon_data_uri -- None entries draw no icon for that row (e.g. a
    top_n_with_other "Other" bucket) but every row still lines up in the same icon column."""
    if not items:
        return "<p>No data yet.</p>"
    max_value = max(value for _, value in items) or 1.0
    label_col, value_col = 150, 40
    icon_width, icon_height = ICON_SIZE
    icon_col = (4 + icon_width + 6) if icons and any(icon is not None for icon in icons) else 0
    bar_col = label_col + icon_col
    bar_area = max(width - bar_col - value_col, 20)
    height = len(items) * row_height + 8

    parts = []
    for index, (label, value) in enumerate(items):
        y = index * row_height
        bar_w = (value / max_value) * bar_area if max_value else 0.0
        parts.append(f'<text x="{label_col - 6}" y="{y + row_height / 2 + 4:.1f}" font-size="10" text-anchor="end" fill="#111">{html.escape(label)}</text>')
        icon = icons[index] if icons and index < len(icons) else None
        if icon is not None:
            icon_y = y + (row_height - icon_height) / 2
            parts.append(f'<image x="{label_col + 4}" y="{icon_y:.1f}" width="{icon_width}" height="{icon_height}" href="{icon}"/>')
        parts.append(f'<rect x="{bar_col}" y="{y + 4:.1f}" width="{max(bar_w, 2):.1f}" height="{row_height - 8}" rx="2" fill="{color}"><title>{html.escape(label)}: {value:g}</title></rect>')
        parts.append(f'<text x="{bar_col + bar_w + 6:.1f}" y="{y + row_height / 2 + 4:.1f}" font-size="10" fill="#555">{value:g}</text>')

    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">{"".join(parts)}</svg>'


def _pie_chart_svg(items: list[tuple[str, float, str]], size: int = 170) -> str:
    """items: (label, value, hex_color). Angle 0 in SVG's y-down coordinate system points east;
    starting at -90 degrees (pointing up, i.e. 12 o'clock) and sweeping by increasing angle draws
    each wedge clockwise, matching ui/charts.py's DonutChartWidget reading direction."""
    total = sum(value for _, value, _ in items)
    if not total:
        return "<p>No data yet.</p>"
    cx = cy = size / 2
    radius = size / 2 - 4

    # A single 100% category can't be drawn as an SVG arc path -- its start and end points
    # coincide (a full-circle sweep), which the path syntax below treats as a zero-length arc and
    # renders as nothing at all. Draw a plain circle instead; ui/charts.py's DonutChartWidget
    # doesn't need this special case since Qt's drawPie handles a full-circle span natively.
    non_zero_items = [item for item in items if item[1] > 0]
    if len(non_zero_items) == 1:
        label, value, color = non_zero_items[0]
        return (
            f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img">'
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" fill="{color}">'
            f"<title>{html.escape(label)}: {int(value)} (100%)</title></circle></svg>"
        )

    parts = []
    angle = -90.0
    for label, value, color in items:
        sweep = (value / total) * 360.0
        x1 = cx + radius * math.cos(math.radians(angle))
        y1 = cy + radius * math.sin(math.radians(angle))
        end_angle = angle + sweep
        x2 = cx + radius * math.cos(math.radians(end_angle))
        y2 = cy + radius * math.sin(math.radians(end_angle))
        large_arc = 1 if sweep > 180 else 0
        path = f"M {cx:.1f},{cy:.1f} L {x1:.2f},{y1:.2f} A {radius:.2f},{radius:.2f} 0 {large_arc} 1 {x2:.2f},{y2:.2f} Z"
        percent = value / total * 100.0
        parts.append(f'<path d="{path}" fill="{color}"><title>{html.escape(label)}: {int(value)} ({percent:.0f}%)</title></path>')
        angle = end_angle

    return f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" role="img">{"".join(parts)}</svg>'


def _legend_table_html(items: list[tuple[str, float, str]]) -> str:
    total = sum(value for _, value, _ in items) or 1.0
    rows = "".join(
        f'<tr><td><span class="swatch" style="background:{color}"></span>{html.escape(label)}</td>'
        f'<td>{int(value)}</td><td>{value / total * 100:.0f}%</td></tr>'
        for label, value, color in items
    )
    return f'<table class="legend">{rows}</table>'


def _chart_block_html(chart_svg: str, legend_html: str, caption: str) -> str:
    return f'<div class="chart-block"><div style="display:flex; gap:16px; align-items:center;">{chart_svg}{legend_html}</div><figcaption>{html.escape(caption)}</figcaption></div>'


def build_statistics_report_html(stats: Statistics, trends: KpiTrends, daily_volume: list[DailyCount], daily_volume_days: int) -> str:
    """Render one self-contained HTML statistics report from an already-computed Statistics snapshot, its KpiTrends (see document/statistics.py::compute_kpi_trends), and a daily-volume series."""
    generated = stats.generated_at.strftime("%Y-%m-%d %H:%M:%S")
    range_text = ""
    if stats.first_scan_at and stats.last_scan_at:
        range_text = f"Scan history: {stats.first_scan_at.strftime('%Y-%m-%d')} to {stats.last_scan_at.strftime('%Y-%m-%d')}"

    tiles_html = "".join(
        [
            _tile_html(
                "Total scans",
                str(stats.total_scans),
                f"{stats.scans_last_30_days} in the last 30 days",
                *trend_display(scans_last_30_days_delta_percent(stats), None),
            ),
            _tile_html(
                "Verification pass rate",
                _percent(stats.verification_pass_rate),
                f"{stats.validation_status_counts.get('valid', 0)} valid of {sum(stats.validation_status_counts.values())} checked",
                *trend_display(trends.verification_pass_rate_delta_pp, True, suffix="pp"),
            ),
            _tile_html(
                "Face match rate",
                _percent(stats.face_match_rate),
                f"average similarity score {stats.average_match_score:.3f}" if stats.average_match_score is not None else "",
                *trend_display(trends.face_match_rate_delta_pp, True, suffix="pp"),
            ),
            _tile_html(
                "OCR extraction rate",
                _percent(stats.ocr_extraction_rate),
                f"{stats.image_quality_issue_count} scans flagged low image quality",
                *trend_display(trends.ocr_extraction_rate_delta_pp, True, suffix="pp"),
            ),
            _tile_html(
                "Security flags",
                str(stats.security_flags_count),
                f"{stats.watchlist_matches_count} watch-list match(es), {stats.expired_documents_count} expired document(s)",
                *trend_display(trends.security_flags_delta_percent, False),
            ),
            _tile_html(
                "Repeat visitors",
                _percent(stats.repeat_visit_rate),
                f"{stats.unique_person_count} unique of {stats.identified_scan_count} identified scans",
                *trend_display(trends.repeat_visit_rate_delta_pp, None, suffix="pp"),
            ),
        ]
    )

    volume_labels = [point.day.strftime("%m-%d") for point in daily_volume]
    volume_values = [float(point.count) for point in daily_volume]
    volume_chart_html = _bar_chart_svg(volume_labels, volume_values, CATEGORICAL_LIGHT[0])

    validation_items = [
        (validation_status_label(status), count, validation_status_color(status)) for status, count in stats.validation_status_counts.items()
    ]
    doc_type_items = [
        (doc_type_label(doc_type), count, doc_type_color(doc_type, CATEGORICAL_LIGHT)) for doc_type, count in stats.doc_type_counts.items()
    ]

    nationality_items = top_n_with_other(stats.nationality_counts, 10)
    nationality_icons = [
        flag_icon_data_uri(alpha2_for_country_code(label)) if alpha2_for_country_code(label) else None
        for label, _ in nationality_items
    ]
    flags_items = flags_and_issues_breakdown(stats)

    averages_html = f"""<table class="averages">
<tr><td>Average scans / day</td><td>{stats.avg_per_day:.2f}</td></tr>
<tr><td>Average scans / week</td><td>{stats.avg_per_week:.1f}</td></tr>
<tr><td>Average scans / month</td><td>{stats.avg_per_month:.1f}</td></tr>
<tr><td>Average scans / year</td><td>{stats.avg_per_year:.0f}</td></tr>
</table>"""

    body = f"""<h1>Kontroll statistics report</h1>
<div class="generated">Generated {html.escape(generated)}{' &mdash; ' + html.escape(range_text) if range_text else ''}</div>

<div class="tiles">{tiles_html}</div>

<h2>Scan volume &mdash; last {daily_volume_days} days</h2>
{volume_chart_html}

<h2>Averages</h2>
{averages_html}

<h2>Breakdowns</h2>
<div class="chart-row">
{_chart_block_html(_pie_chart_svg(validation_items), _legend_table_html(validation_items), "Document validation status")}
{_chart_block_html(_pie_chart_svg(doc_type_items), _legend_table_html(doc_type_items), "Document type")}
</div>

<h2>Nationality / issuing country (top 10)</h2>
{_horizontal_bar_chart_svg([(label, count) for label, count in nationality_items], CATEGORICAL_LIGHT[0], icons=nationality_icons)}

<h2>Flags &amp; issues</h2>
<p style="font-size:0.85em; color:#555;">Each condition below is independent -- a scan can appear in more than one row (e.g. both blurry and expired), so this is a frequency count, not a partition of the total.</p>
{_horizontal_bar_chart_svg(flags_items, CATEGORICAL_LIGHT[1])}
"""

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Kontroll statistics report</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
{body}
</body>
</html>
"""
