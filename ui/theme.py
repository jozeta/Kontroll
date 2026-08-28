"""App-wide Light/Dark/System theming via a QPalette override paired with the "Fusion" QStyle.

Both together, not the palette alone: Windows' native style ("windowsvista"/"windows11") only
partially respects an app-level QPalette override -- many widgets (QGroupBox, QTabWidget,
QPushButton chrome) fall back to a visually inconsistent flat rendering under it instead of
properly recoloring, which is what produced the "background and a lot of text turns white"
bug when Light/Dark was forced with native style still active. Fusion is the standard Qt style
that fully and uniformly respects every QPalette role for every widget, which is why the
established "Fusion + custom palette" recipe is used here instead. "System" reverts both the
style and the palette to whatever PyQt6/the OS handed the app by default, so native rendering
(and native OS dark-mode following, if any) is untouched unless a theme is explicitly forced.
"""

from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

THEME_SYSTEM = "system"
THEME_LIGHT = "light"
THEME_DARK = "dark"

THEME_CHOICES = [THEME_SYSTEM, THEME_LIGHT, THEME_DARK]

# Captured lazily on the first apply_theme() call, before any override has ever been applied --
# see apply_theme's docstring.
_system_default_palette: QPalette | None = None
_system_default_style_name: str | None = None


def _build_light_palette() -> QPalette:
    palette = QPalette()
    window = QColor(240, 240, 240)
    base = QColor(255, 255, 255)
    text = QColor(20, 20, 20)
    disabled_text = QColor(150, 150, 150)
    highlight = QColor(0, 120, 215)

    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, window)
    palette.setColor(QPalette.ColorRole.ToolTipBase, base)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, window)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.ColorRole.Link, highlight)
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    return palette


def _build_dark_palette() -> QPalette:
    palette = QPalette()
    window = QColor(45, 45, 45)
    base = QColor(30, 30, 30)
    alternate_base = QColor(53, 53, 53)
    text = QColor(220, 220, 220)
    disabled_text = QColor(120, 120, 120)
    highlight = QColor(42, 130, 218)

    palette.setColor(QPalette.ColorRole.Window, window)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, alternate_base)
    palette.setColor(QPalette.ColorRole.ToolTipBase, window)
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, window)
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.BrightText, QColor(255, 80, 80))
    palette.setColor(QPalette.ColorRole.Link, highlight)
    palette.setColor(QPalette.ColorRole.Highlight, highlight)
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(0, 0, 0))

    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, disabled_text)
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, disabled_text)
    return palette


_LIGHT_PALETTE = _build_light_palette()
_DARK_PALETTE = _build_dark_palette()


def apply_theme(app: QApplication, mode: str) -> None:
    """Apply the given theme mode ("system" / "light" / "dark") app-wide.

    The very first call caches app.palette() and app.style()'s name -- at that point still the
    untouched OS/PyQt6 defaults, since nothing has called setPalette()/setStyle() before this
    module does -- as the "system" baseline, so switching back to "system" later restores exactly
    what the app looked like before any override, rather than guessing at OS colors/style.

    Style is set before palette (for both the forced and the system-revert paths) since
    QApplication.setStyle() can itself reset the palette to that style's own default as a side
    effect -- setting the palette second guarantees it's always the one left in effect.
    """
    global _system_default_palette, _system_default_style_name
    if _system_default_palette is None:
        _system_default_palette = QPalette(app.palette())
        _system_default_style_name = app.style().objectName()

    if mode == THEME_LIGHT:
        app.setStyle("Fusion")
        app.setPalette(_LIGHT_PALETTE)
    elif mode == THEME_DARK:
        app.setStyle("Fusion")
        app.setPalette(_DARK_PALETTE)
    else:
        app.setStyle(_system_default_style_name)
        app.setPalette(_system_default_palette)
