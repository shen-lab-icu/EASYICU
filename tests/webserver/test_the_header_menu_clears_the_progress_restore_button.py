"""The conversation header's "More" menu stays clear of the progress restore button.

UX audit 2026-11-19 (73877009c, 1542x1000 and 1280x760): with the research
progress panel collapsed, "Show research progress" (.gd-aside-restore, fixed at
the top right of the shell) covered the header's "More" summary, so a click on
it opened nothing. Wherever the restore button shows on a desktop width, the
header reserves its corner. Geometry is checked live by tools/audit_web_fit.py.
"""

import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[2] / "src/easyicu/webserver/static"


def _px(block: str, prop: str) -> int:
    return int(re.search(rf"(?<![-\w]){prop}:(\d+)px", block).group(1))


def test_the_header_reserves_the_restore_buttons_corner() -> None:
    panels = (STATIC / "css/guided-panels.css").read_text()
    restore = re.search(r"^\.gd-aside-restore\{([^}]*)\}", panels, re.M).group(1)
    corner = _px(restore, "right") + _px(restore, "width")
    desktop = (STATIC / "css/guided-pi-desktop.css").read_text()
    block = desktop[desktop.index("@media(min-width:1181px){\n .gd-main.gpi-empty-session-focus"):]
    block = block[: block.index("\n}\n")]
    for shown in (
        ".gd-main.gd-context-aside-collapsed:not(.gpi-preview-open) .gpi-head",
        ".gd-main.gpi-empty-session-focus:not(.gpi-empty-aside-open):not(.gpi-preview-open):not(.gpi-source-open) .gpi-head",
    ):
        assert shown in block
    padding = int(re.search(r"\.gpi-head\{padding-right:(\d+)px\}", block).group(1))
    assert padding >= corner + 8
    index = (STATIC / "index.html").read_text()
    assert "css/guided-pi-desktop.css?v=20261119-b11" in index
