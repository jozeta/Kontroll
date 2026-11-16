"""The version comes from the VERSION file and matches the README and CHANGELOG."""

import re
from pathlib import Path

from app_version import get_version

ROOT = Path(__file__).resolve().parents[1]


def test_version_is_semver_from_version_file():
    assert get_version() == (ROOT / "VERSION").read_text().strip()
    assert re.fullmatch(r"\d+\.\d+\.\d+", get_version())


def test_readme_and_changelog_show_the_version():
    v = get_version()
    assert f"**{v}**" in (ROOT / "README.md").read_text(encoding="utf-8")
    assert re.search(rf"^## {re.escape(v)}\b", (ROOT / "CHANGELOG.md").read_text(encoding="utf-8"), re.M)
