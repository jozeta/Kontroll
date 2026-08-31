"""Lightweight QPainter-based bar and donut chart widgets for the Settings "Statistics" tab
(ui/settings_window.py) -- no charting-library dependency added, consistent with this app's
existing minimal-dependency stance elsewhere (e.g. ui/scan_export.py using a browser Print/
Save-as-PDF flow rather than a PDF-generation library).

Colors are the fixed eight-hue categorical order and four-role status palette from this project's
validated data-visualization reference palette (never reassigned per-chart or per-filter -- a
color always means the same thing everywhere it's used), read per-widget at paint time from the
current QPalette (light vs. dark) the same way ui/main_window.py's preview headlines pick up theme
changes live, so switching Settings -> General -> Appearance updates chart colors without a
restart.
"""

import math
from collections.abc import Callable

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPaintEvent, QMouseEvent, QPixmap
from PyQt6.QtWidgets import QFrame, QLabel, QSizePolicy, QToolTip, QVBoxLayout, QWidget

# Fixed categorical hue order -- see the dataviz skill's references/palette.md. Light/dark are the
# same eight hues, each stepped for its own surface.
CATEGORICAL_LIGHT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
CATEGORICAL_DARK = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]

# Status palette is fixed -- never themed, never reused as a categorical slot -- reserved for an
# actual good/warning/critical outcome (e.g. validation_status), not general-purpose series color.
STATUS_GOOD = "#0ca30c"
STATUS_WARNING = "#fab219"
STATUS_SERIOUS = "#ec835a"
STATUS_CRITICAL = "#d03b3b"
STATUS_MUTED = "#898781"

_GRIDLINE_LIGHT = QColor("#e1e0d9")
_GRIDLINE_DARK = QColor("#2c2c2a")


def is_dark_palette(widget: QWidget) -> bool:
    return widget.palette().color(widget.backgroundRole()).lightnessF() < 0.5


def categorical_colors(widget: QWidget) -> list[str]:
    return CATEGORICAL_DARK if is_dark_palette(widget) else CATEGORICAL_LIGHT


def _text_color(widget: QWidget) -> QColor:
    return widget.palette().color(widget.foregroundRole())


def _muted_color(widget: QWidget) -> QColor:
    color = _text_color(widget)
    color.setAlphaF(0.6)
    return color


def _gridline_color(widget: QWidget) -> QColor:
    return _GRIDLINE_DARK if is_dark_palette(widget) else _GRIDLINE_LIGHT


def evenly_spaced_indices(count: int, max_labels: int) -> set[int]:
    """Choose up to `max_labels` indices spread evenly across range(count), always including both
    endpoints -- used to thin out x-axis tick labels on a many-bar chart (e.g. a 365-day volume
    series). A fixed stride from index 0 (show every Nth label, always force the last one) looks
    even until the final forced label lands close to the last regular one, which crowds/overlaps --
    picking evenly-spaced indices by construction avoids that instead of special-casing it.

    Shared by ui/charts.py's BarChartWidget and ui/statistics_export.py's static report, so the
    on-screen chart and the printed one thin their labels the same way.
    """
    if count <= max_labels:
        return set(range(count))
    if max_labels <= 1:
        return {count - 1}
    return {round(i * (count - 1) / (max_labels - 1)) for i in range(max_labels)}


def trend_display(delta: float | None, higher_is_better: bool | None, suffix: str = "%") -> tuple[str, str | None]:
    """Format a period-over-period delta as an arrow + magnitude, plus the status color it should
    render in -- the shared logic behind every KPI tile's trend line (ui/settings_window.py's
    Statistics tab and ui/statistics_export.py's printed report both call this, so a tile and its
    printed equivalent are never colored or worded differently).

    delta: None means one of the two periods being compared has no data at all -- not the same as
    an unchanged (0) value, which still gets a flat "▬" arrow.
    higher_is_better: True if going up is the good direction (e.g. verification pass rate), False
    if going up is the bad one (e.g. security flags), or None if the metric has no inherent
    good/bad direction (e.g. raw scan volume) -- that last case, and an exact 0 delta, both return
    a None color so the tile falls back to its normal (uncolored) text.
    """
    if delta is None:
        return "no prior-period data", None
    if delta == 0:
        return "▬ flat vs. previous period", None

    arrow = "▲" if delta > 0 else "▼"
    text = f"{arrow} {abs(delta):.1f}{suffix} vs. previous period"
    if higher_is_better is None:
        return text, None
    return text, STATUS_GOOD if (delta > 0) == higher_is_better else STATUS_CRITICAL


class BarChartWidget(QWidget):
    """A single-series bar chart, drawn vertically (e.g. a scan-volume time series) or
    horizontally (e.g. a ranked breakdown) depending on `orientation`.

    One hue for every bar -- per the dataviz reference, a ranked/ordered single-series chart
    doesn't need per-item color to carry identity, since position and the direct label already do.
    """

    _BAR_GAP_FRACTION = 0.35  # fraction of each bar's slot left as gap to its neighbor
    _HORIZONTAL_ROW_HEIGHT = 26
    _ICON_LEFT_PADDING = 4
    _ICON_RIGHT_PADDING = 6

    def __init__(self, orientation: str = "vertical", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        if orientation not in ("vertical", "horizontal"):
            raise ValueError(f"Unknown orientation: {orientation!r}")
        self._orientation = orientation
        self._labels: list[str] = []
        self._values: list[float] = []
        self._value_formatter: Callable[[float], str] = lambda value: f"{value:g}"
        self._max_tick_labels = 12  # thin x-axis labels down to at most this many (vertical mode); see evenly_spaced_indices
        self._icons: list[QPixmap | None] | None = None  # per-row icon (horizontal mode only), e.g. a flag -- see set_data
        self._bar_rects: list[QRectF] = []
        self.setMouseTracking(True)
        if orientation == "vertical":
            self.setMinimumHeight(220)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def set_data(
        self,
        labels: list[str],
        values: list[float],
        value_formatter: Callable[[float], str] | None = None,
        max_tick_labels: int = 12,
        icons: list[QPixmap | None] | None = None,
    ) -> None:
        """icons (horizontal mode only): one QPixmap per row (e.g. a flag from ui/flag_icons.py),
        parallel to `labels`/`values`; None entries in the list draw no icon for that row (e.g. a
        top_n_with_other "Other" bucket) but every row still lines up in the same icon column."""
        self._labels = labels
        self._values = values
        if value_formatter is not None:
            self._value_formatter = value_formatter
        self._max_tick_labels = max(max_tick_labels, 1)
        self._icons = icons
        if self._orientation == "horizontal":
            self.setMinimumHeight(max(len(labels), 1) * self._HORIZONTAL_ROW_HEIGHT + 16)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            if not self._values:
                painter.setPen(_muted_color(self))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data yet")
                self._bar_rects = []
                return
            if self._orientation == "vertical":
                self._paint_vertical(painter)
            else:
                self._paint_horizontal(painter)
        finally:
            painter.end()

    def _paint_vertical(self, painter: QPainter) -> None:
        color = QColor(categorical_colors(self)[0])
        muted = _muted_color(self)
        grid = _gridline_color(self)
        metrics = QFontMetrics(self.font())

        max_value = max(self._values) or 1.0
        left_margin = metrics.horizontalAdvance(self._value_formatter(max_value)) + 10
        bottom_margin = metrics.height() + 8
        top_margin = 6
        plot_rect = QRectF(left_margin, top_margin, self.width() - left_margin - 6, self.height() - top_margin - bottom_margin)

        # Gridlines + axis value labels at 0/50/100% of the max, recessive (thin, muted) per the
        # dataviz mark spec -- these orient the eye without competing with the bars themselves.
        painter.setPen(grid)
        for fraction in (0.0, 0.5, 1.0):
            y = plot_rect.bottom() - fraction * plot_rect.height()
            painter.drawLine(QPointF(plot_rect.left(), y), QPointF(plot_rect.right(), y))
            painter.setPen(muted)
            label = self._value_formatter(max_value * fraction)
            painter.drawText(QRectF(0, y - metrics.height() / 2, left_margin - 6, metrics.height()), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, label)
            painter.setPen(grid)

        slot_width = plot_rect.width() / len(self._values)
        bar_width = slot_width * (1 - self._BAR_GAP_FRACTION)
        self._bar_rects = []
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(color)
        for index, value in enumerate(self._values):
            bar_height = (value / max_value) * plot_rect.height() if max_value else 0.0
            x = plot_rect.left() + index * slot_width + (slot_width - bar_width) / 2
            rect = QRectF(x, plot_rect.bottom() - bar_height, bar_width, bar_height)
            painter.drawRoundedRect(rect, min(3.0, bar_width / 2), min(3.0, bar_width / 2))
            self._bar_rects.append(rect)

        painter.setPen(muted)
        tick_indices = evenly_spaced_indices(len(self._labels), self._max_tick_labels)
        for index, label in enumerate(self._labels):
            if index not in tick_indices:
                continue
            x = plot_rect.left() + index * slot_width
            label_rect = QRectF(x, plot_rect.bottom() + 4, slot_width, bottom_margin)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _paint_horizontal(self, painter: QPainter) -> None:
        color = QColor(categorical_colors(self)[0])
        muted = _muted_color(self)
        text_color = _text_color(self)
        metrics = QFontMetrics(self.font())

        max_value = max(self._values) or 1.0
        max_label_width = max((metrics.horizontalAdvance(label) for label in self._labels), default=0)
        label_column_width = min(max_label_width + 10, self.width() * 0.4)
        value_column_width = metrics.horizontalAdvance(self._value_formatter(max_value)) + 12

        icon_column_width = 0.0
        if self._icons:
            max_icon_width = max((icon.width() for icon in self._icons if icon is not None), default=0)
            if max_icon_width:
                icon_column_width = self._ICON_LEFT_PADDING + max_icon_width + self._ICON_RIGHT_PADDING

        bar_area_left = label_column_width + icon_column_width
        bar_area_width = max(self.width() - bar_area_left - value_column_width - 6, 10)

        self._bar_rects = []
        row_height = self._HORIZONTAL_ROW_HEIGHT
        bar_height = row_height * (1 - self._BAR_GAP_FRACTION)
        for index, (label, value) in enumerate(zip(self._labels, self._values)):
            row_top = index * row_height
            painter.setPen(text_color)
            label_rect = QRectF(0, row_top, label_column_width - 6, row_height)
            elided = metrics.elidedText(label, Qt.TextElideMode.ElideRight, int(label_rect.width()))
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, elided)

            icon = self._icons[index] if self._icons and index < len(self._icons) else None
            if icon is not None:
                icon_y = row_top + (row_height - icon.height()) / 2
                painter.drawPixmap(QPointF(label_column_width + self._ICON_LEFT_PADDING, icon_y), icon)

            bar_width = (value / max_value) * bar_area_width if max_value else 0.0
            bar_rect = QRectF(bar_area_left, row_top + (row_height - bar_height) / 2, max(bar_width, 2.0), bar_height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(bar_rect, 3.0, 3.0)
            self._bar_rects.append(QRectF(0, row_top, self.width(), row_height))

            painter.setPen(muted)
            value_rect = QRectF(bar_area_left + bar_width + 6, row_top, value_column_width, row_height)
            painter.drawText(value_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._value_formatter(value))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        for index, rect in enumerate(self._bar_rects):
            if rect.contains(position):
                text = f"{self._labels[index]}: {self._value_formatter(self._values[index])}"
                QToolTip.showText(event.globalPosition().toPoint(), text, self)
                return
        QToolTip.hideText()

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)


class DonutChartWidget(QWidget):
    """A donut chart for one mutually-exclusive-category breakdown (e.g. validation_status, where
    every scan has exactly one value) -- a pie/donut is a poor fit for overlapping conditions (a
    scan can be both expired *and* a no-match), so callers with that shape should reach for
    BarChartWidget's horizontal mode instead (see Statistics tab's "Flags & issues" chart).
    """

    _HOLE_FRACTION = 0.55
    _LEGEND_ROW_HEIGHT = 20

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._items: list[tuple[str, float, str]] = []  # (label, value, hex_color)
        self._wedges: list[tuple[float, float]] = []  # (start_angle_degrees, span_degrees), parallel to _items
        self._center = QPointF()
        self._radius = 0.0
        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

    def set_data(self, items: list[tuple[str, float, str]]) -> None:
        """items: list of (label, value, hex_color) -- hex_color decides what each slice means, so
        callers pass a fixed semantic color (e.g. status colors for validation_status) rather than
        relying on draw order."""
        self._items = [item for item in items if item[1] > 0]
        self.setMinimumHeight(max(160, len(self._items) * self._LEGEND_ROW_HEIGHT + 40))
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        try:
            if not self._items:
                painter.setPen(_muted_color(self))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No data yet")
                self._wedges = []
                return
            self._paint_donut_and_legend(painter)
        finally:
            painter.end()

    def _paint_donut_and_legend(self, painter: QPainter) -> None:
        total = sum(value for _, value, _ in self._items)
        diameter = min(self.height() - 12, self.width() * 0.45)
        diameter = max(diameter, 80.0)
        donut_rect = QRectF(8, (self.height() - diameter) / 2, diameter, diameter)
        self._center = donut_rect.center()
        self._radius = diameter / 2

        background = self.palette().color(self.backgroundRole())
        painter.setPen(Qt.PenStyle.NoPen)

        start_angle = 90 * 16  # 12 o'clock, Qt angles are in 1/16ths of a degree
        self._wedges = []
        for label, value, hex_color in self._items:
            span_degrees = (value / total) * 360.0
            span_angle = -round(span_degrees * 16)  # negative = clockwise
            painter.setBrush(QColor(hex_color))
            painter.drawPie(donut_rect, start_angle, span_angle)
            self._wedges.append((start_angle / 16.0, span_angle / 16.0))
            start_angle += span_angle

        hole_diameter = diameter * self._HOLE_FRACTION
        hole_rect = QRectF(0, 0, hole_diameter, hole_diameter)
        hole_rect.moveCenter(self._center)
        painter.setBrush(background)
        painter.drawEllipse(hole_rect)

        painter.setPen(_text_color(self))
        total_rect = QRectF(0, 0, hole_diameter, hole_diameter)
        total_rect.moveCenter(self._center)
        painter.drawText(total_rect, Qt.AlignmentFlag.AlignCenter, f"{int(total)}")

        self._paint_legend(painter, legend_left=donut_rect.right() + 16)

    def _paint_legend(self, painter: QPainter, legend_left: float) -> None:
        total = sum(value for _, value, _ in self._items)
        metrics = QFontMetrics(self.font())
        swatch_size = 10
        top = (self.height() - len(self._items) * self._LEGEND_ROW_HEIGHT) / 2
        for index, (label, value, hex_color) in enumerate(self._items):
            row_top = top + index * self._LEGEND_ROW_HEIGHT
            swatch_rect = QRectF(legend_left, row_top + (self._LEGEND_ROW_HEIGHT - swatch_size) / 2, swatch_size, swatch_size)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(hex_color))
            painter.drawRoundedRect(swatch_rect, 2.0, 2.0)

            percent = (value / total * 100.0) if total else 0.0
            text = f"{label} — {int(value)} ({percent:.0f}%)"
            painter.setPen(_text_color(self))
            text_rect = QRectF(legend_left + swatch_size + 8, row_top, self.width() - legend_left - swatch_size - 16, self._LEGEND_ROW_HEIGHT)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(text_rect.width())))

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        position = event.position()
        offset = position - self._center
        distance = (offset.x() ** 2 + offset.y() ** 2) ** 0.5
        if self._radius and self._radius * self._HOLE_FRACTION <= distance <= self._radius:
            angle_degrees = math.degrees(math.atan2(-offset.y(), offset.x())) % 360
            for index, (start_angle, span_angle) in enumerate(self._wedges):
                sweep = -span_angle  # stored as negative (clockwise); the wedge covers `sweep` degrees clockwise from start_angle
                normalized = (start_angle - angle_degrees) % 360
                if normalized <= sweep:
                    label, value, _ = self._items[index]
                    QToolTip.showText(event.globalPosition().toPoint(), f"{label}: {int(value)}", self)
                    return
        QToolTip.hideText()

    def leaveEvent(self, event) -> None:
        QToolTip.hideText()
        super().leaveEvent(event)


class StatTileWidget(QFrame):
    """One KPI "stat tile": a small title, a large headline value, an optional muted detail line
    (factual context, e.g. "145 valid of 240 checked"), and an optional colored trend line (a
    period-over-period delta, see trend_display) -- the Statistics tab's top row of cards.

    The detail and trend lines are two separate labels, not one, specifically so a trend's status
    color (good/bad) never bleeds onto the plain factual detail line above it.

    Deliberately no hardcoded text color on the title/value/detail labels, same reasoning as ui/
    main_window.py's preview headlines: leaving it unset inherits the current QPalette's
    WindowText/text color, which stays readable under both light and dark themes and updates live
    if the operator switches themes. The trend line is the one exception -- see set_value.
    """

    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet("font-size: 9pt;")
        title_label.setWordWrap(True)
        layout.addWidget(title_label)

        self._value_label = QLabel("—")
        self._value_label.setStyleSheet("font-size: 19pt; font-weight: bold;")
        layout.addWidget(self._value_label)

        self._detail_label = QLabel()
        self._detail_label.setStyleSheet("font-size: 8pt;")
        self._detail_label.setWordWrap(True)
        self._detail_label.setVisible(False)
        layout.addWidget(self._detail_label)

        self._trend_label = QLabel()
        self._trend_label.setWordWrap(True)
        self._trend_label.setVisible(False)
        layout.addWidget(self._trend_label)

    def set_value(self, value_text: str, detail_text: str = "", trend_text: str = "", trend_color: str | None = None) -> None:
        """trend_text/trend_color: normally trend_display's own return value, passed straight
        through -- trend_color of None inherits the tile's normal (theme-aware) text color instead
        of forcing one, same as leaving detail_text uncolored.
        """
        self._value_label.setText(value_text)
        self._detail_label.setText(detail_text)
        self._detail_label.setVisible(bool(detail_text))

        self._trend_label.setText(trend_text)
        self._trend_label.setVisible(bool(trend_text))
        color_rule = f" color: {trend_color};" if trend_color else ""
        self._trend_label.setStyleSheet(f"font-size: 8pt; font-weight: bold;{color_rule}")
