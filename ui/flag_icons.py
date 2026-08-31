"""Renders small, in-color national flag icons as QPixmaps for the scan history list.

Deliberately does not use Unicode flag emoji: Qt's QPainter/QLabel text rendering does not
reliably paint the color-font glyphs Windows' Segoe UI Emoji font uses for flags (a known Qt
limitation with COLR/CPAL color fonts), so those characters tend to render as nothing at all
rather than a flag. Drawing plain vector shapes sidesteps that entirely.

Only a curated set of frequently-seen (mostly European) countries have a real flag pattern
defined -- picked for simple, accurately-representable geometry (stripes/Nordic cross) rather than
attempting every country's flag. Any other recognized alpha-2 code falls back to a neutral gray
tile with the code itself drawn on it (e.g. "PL") instead of a guessed or wrong-looking flag
design -- still identifies the country at a glance without pretending to be real flag art.
"""

import base64

from PyQt6.QtCore import QBuffer, QIODevice, QRectF, Qt
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap

ICON_SIZE = (24, 16)

_FALLBACK_COLOR = "#B0B0B0"

# Each entry is ("stripes", orientation, [(color, weight), ...]) for equal or weighted stripes,
# ("nordic_cross", background_color, cross_color, border_color_or_None) for the offset-cross
# Nordic flag design, or ("canton", stripe_color_a, stripe_color_b, stripe_count, canton_color) for
# a horizontally-striped field with a solid-color rectangular canton in the top-left corner (the
# US flag's layout, simplified to a plain canton block rather than 50 individual stars, which
# would be illegible at this icon's size anyway). Colors are standard flag reference hex values;
# stripe weights default to equal (1) unless a real flag's proportions are visibly unequal (e.g.
# Spain, Portugal, Latvia).
_FLAG_PATTERNS: dict[str, tuple] = {
    "SE": ("nordic_cross", "#006AA7", "#FECC02", None),
    "NO": ("nordic_cross", "#EF2B2D", "#FFFFFF", "#002868"),
    "DK": ("nordic_cross", "#C60C30", "#FFFFFF", None),
    "FI": ("nordic_cross", "#FFFFFF", "#003580", None),
    "IS": ("nordic_cross", "#02529C", "#FFFFFF", "#DC1E35"),
    "DE": ("stripes", "horizontal", [("#000000", 1), ("#DD0000", 1), ("#FFCE00", 1)]),
    "NL": ("stripes", "horizontal", [("#AE1C28", 1), ("#FFFFFF", 1), ("#21468B", 1)]),
    "BE": ("stripes", "vertical", [("#000000", 1), ("#FAE042", 1), ("#ED2939", 1)]),
    "FR": ("stripes", "vertical", [("#0055A4", 1), ("#FFFFFF", 1), ("#EF4135", 1)]),
    "IT": ("stripes", "vertical", [("#008C45", 1), ("#F4F5F0", 1), ("#CD212A", 1)]),
    "IE": ("stripes", "vertical", [("#169B62", 1), ("#FFFFFF", 1), ("#FF883E", 1)]),
    "PL": ("stripes", "horizontal", [("#FFFFFF", 1), ("#DC143C", 1)]),
    "AT": ("stripes", "horizontal", [("#ED2939", 1), ("#FFFFFF", 1), ("#ED2939", 1)]),
    "RO": ("stripes", "vertical", [("#002B7F", 1), ("#FCD116", 1), ("#CE1126", 1)]),
    "BG": ("stripes", "horizontal", [("#FFFFFF", 1), ("#00966E", 1), ("#D62612", 1)]),
    "HU": ("stripes", "horizontal", [("#CE2939", 1), ("#FFFFFF", 1), ("#477050", 1)]),
    "LT": ("stripes", "horizontal", [("#FDB913", 1), ("#006A44", 1), ("#C1272D", 1)]),
    "EE": ("stripes", "horizontal", [("#0072CE", 1), ("#000000", 1), ("#FFFFFF", 1)]),
    "LV": ("stripes", "horizontal", [("#9E3039", 2), ("#FFFFFF", 1), ("#9E3039", 2)]),
    "LU": ("stripes", "horizontal", [("#ED2939", 1), ("#FFFFFF", 1), ("#00A1DE", 1)]),
    "RU": ("stripes", "horizontal", [("#FFFFFF", 1), ("#0039A6", 1), ("#D52B1E", 1)]),
    "UA": ("stripes", "horizontal", [("#0057B7", 1), ("#FFD700", 1)]),
    "ES": ("stripes", "horizontal", [("#AA151B", 1), ("#F1BF00", 2), ("#AA151B", 1)]),
    "PT": ("stripes", "vertical", [("#046A38", 2), ("#DA291C", 3)]),
    "US": ("canton", "#B22234", "#FFFFFF", 7, "#3C3B6E"),
}


def _paint_stripes(painter: QPainter, rect: QRectF, orientation: str, stripes: list[tuple[str, int]]) -> None:
    total_weight = sum(weight for _, weight in stripes)
    offset = 0.0
    for color, weight in stripes:
        if orientation == "horizontal":
            stripe_height = rect.height() * weight / total_weight
            painter.fillRect(QRectF(rect.left(), rect.top() + offset, rect.width(), stripe_height), QColor(color))
            offset += stripe_height
        else:
            stripe_width = rect.width() * weight / total_weight
            painter.fillRect(QRectF(rect.left() + offset, rect.top(), stripe_width, rect.height()), QColor(color))
            offset += stripe_width


def _paint_canton(painter: QPainter, rect: QRectF, stripe_color_a: str, stripe_color_b: str, stripe_count: int, canton_color: str) -> None:
    stripe_height = rect.height() / stripe_count
    for index in range(stripe_count):
        color = stripe_color_a if index % 2 == 0 else stripe_color_b
        painter.fillRect(QRectF(rect.left(), rect.top() + index * stripe_height, rect.width(), stripe_height), QColor(color))

    # A canton spanning the top-left ~40% width / ~55% height, over whichever stripes fall
    # beneath it -- proportions chosen to read clearly at this icon's small size, not to match
    # the real flag's exact 50-star field dimensions.
    canton_rect = QRectF(rect.left(), rect.top(), rect.width() * 0.4, rect.height() * 0.55)
    painter.fillRect(canton_rect, QColor(canton_color))


def _paint_nordic_cross(painter: QPainter, rect: QRectF, bg_color: str, cross_color: str, border_color: str | None) -> None:
    painter.fillRect(rect, QColor(bg_color))

    bar_thickness = rect.height() * 0.32
    vertical_x = rect.width() * 0.32
    horizontal_y = (rect.height() - bar_thickness) / 2

    if border_color:
        border_extra = bar_thickness * 0.22
        painter.fillRect(
            QRectF(rect.left() + vertical_x - border_extra, rect.top(), bar_thickness + 2 * border_extra, rect.height()),
            QColor(border_color),
        )
        painter.fillRect(
            QRectF(rect.left(), rect.top() + horizontal_y - border_extra, rect.width(), bar_thickness + 2 * border_extra),
            QColor(border_color),
        )

    painter.fillRect(QRectF(rect.left() + vertical_x, rect.top(), bar_thickness, rect.height()), QColor(cross_color))
    painter.fillRect(QRectF(rect.left(), rect.top() + horizontal_y, rect.width(), bar_thickness), QColor(cross_color))


_icon_cache: dict[str, QPixmap] = {}


def render_flag_icon(alpha2_code: str) -> QPixmap:
    """Render a small colored flag icon for a recognized alpha-2 country code, or a neutral gray tile if its design isn't curated.

    Memoized by code -- there are only ~25 distinct icons ever drawn, but this is called once per
    history row that has a recognized country, so without caching the same handful of icons get
    redrawn with QPainter from scratch on every row build. QPixmap is safe to share this way: it's
    implicitly-shared/copy-on-write in Qt, so handing the same cached instance to many QLabels via
    setPixmap is the normal, intended usage, not a mutation hazard.
    """
    cached = _icon_cache.get(alpha2_code)
    if cached is not None:
        return cached

    pixmap = QPixmap(*ICON_SIZE)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    rect = QRectF(0, 0, *ICON_SIZE)

    pattern = _FLAG_PATTERNS.get(alpha2_code)
    if pattern is None:
        painter.fillRect(rect, QColor(_FALLBACK_COLOR))
        font = painter.font()
        font.setPointSizeF(7.5)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QColor("#404040"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, alpha2_code)
    elif pattern[0] == "stripes":
        _, orientation, stripes = pattern
        _paint_stripes(painter, rect, orientation, stripes)
    elif pattern[0] == "nordic_cross":
        _, bg_color, cross_color, border_color = pattern
        _paint_nordic_cross(painter, rect, bg_color, cross_color, border_color)
    elif pattern[0] == "canton":
        _, stripe_color_a, stripe_color_b, stripe_count, canton_color = pattern
        _paint_canton(painter, rect, stripe_color_a, stripe_color_b, stripe_count, canton_color)

    painter.setPen(QPen(QColor(0, 0, 0, 60), 1))
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.drawRect(rect.adjusted(0, 0, -1, -1))
    painter.end()

    _icon_cache[alpha2_code] = pixmap
    return pixmap


_data_uri_cache: dict[str, str] = {}


def flag_icon_data_uri(alpha2_code: str) -> str:
    """Same flag art as render_flag_icon, encoded as a base64 PNG data URI for embedding directly
    in generated HTML (see ui/statistics_export.py's nationality breakdown chart) -- reuses the
    exact same drawing instead of re-implementing every flag pattern a second time in SVG, so the
    printed report and the on-screen chart never disagree on what a country's flag looks like.
    """
    cached = _data_uri_cache.get(alpha2_code)
    if cached is not None:
        return cached

    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    render_flag_icon(alpha2_code).save(buffer, "PNG")
    encoded = base64.b64encode(bytes(buffer.data())).decode("ascii")
    data_uri = f"data:image/png;base64,{encoded}"

    _data_uri_cache[alpha2_code] = data_uri
    return data_uri
