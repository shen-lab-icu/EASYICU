"""The conversation header's "More" menu stays clear of the progress restore button.

UX audit 2026-11-19 (73877009c, 1542x1000 and 1280x760): with the research
progress panel collapsed, "Show research progress" (.gd-aside-restore, fixed at
the top right of the shell) covered the header's "More" summary, so a click on
it opened nothing. Wherever the restore button shows on a desktop width, the
header reserves its corner. Geometry is checked live by tools/audit_web_fit.py.
"""

import os
import re
from pathlib import Path

import pytest

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


@pytest.mark.requires_web_server
@pytest.mark.parametrize("width,height", [(1280, 760), (1542, 1000)])
def test_a_press_on_more_opens_the_menu_with_the_panel_collapsed(width: int, height: int) -> None:
    # S1500: on the project entry page with the progress panel collapsed, a
    # real mouse press on "More" opens its menu (it reached the restore
    # button instead at 73877009c).
    from tools.web_ui_audit_routes import new_audit_page, settle

    playwright = pytest.importorskip("playwright.sync_api")
    base = os.environ["EASYICU_WEB_AUDIT_BASE"]
    session = os.environ.get("EASYICU_WEB_AUDIT_SESSION", "")
    if not session:
        pytest.skip("the project entry page needs EASYICU_WEB_AUDIT_SESSION (pi_project=…&pi_session=…)")
    with playwright.sync_playwright() as runtime:
        browser = runtime.chromium.launch(headless=True)
        page = new_audit_page(browser, base, width, height)
        page.goto(base + f"/?{session}#guided", wait_until="load")
        settle(page)
        page.click("[data-gpi-rail-new]:visible, .gpi-head-new:visible")
        settle(page)
        if not page.is_visible(".gd-aside-restore"):
            page.click("#gdContextAside [data-context-aside-toggle]")
            page.wait_for_timeout(400)
        assert page.is_visible(".gd-aside-restore"), "the progress panel did not collapse"
        box = page.locator(".gpi-head-overflow > summary:visible").first.bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.wait_for_timeout(300)
        opened = page.evaluate("() => !!document.querySelector('.gpi-head-overflow[open]')")
        browser.close()
    assert opened, f"More did not open at {width}x{height}"
