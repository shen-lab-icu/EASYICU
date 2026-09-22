"""Static fit audit of the native web UI at rest.

On every route and viewport size this finds controls that cannot be pressed
(covered at their centre by something else), floating elements resting outside
the viewport, text silently clipped by ``overflow: hidden`` without an ellipsis,
controls too small to hit, and page-level horizontal overflow. It complements
``audit_web_popovers.py``, which exercises the menus.

Requires a running EasyICU web server and Playwright with a Chrome channel::

    python tools/audit_web_fit.py --base http://127.0.0.1:8765 \
        --session "pi_project=<draft id>&pi_session=<session id>" \
        --out output/ui-audit/fit.json

The exit status is 1 when any check flags something.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.web_ui_audit_routes import DEFAULT_VIEWPORTS, ROUTES_WITHOUT_SESSION, settle  # noqa: E402

MIN_HIT = 24

JS = r"""
(minHit) => {
  const inClosedDetails = el => { for (let a = el; a && a !== document.body; a = a.parentElement) { const p = a.parentElement; if (p && p.tagName === 'DETAILS' && !p.open && a.tagName !== 'SUMMARY') return true; } return false; };
  // An element scrolled out of its scroll container's box is not on screen.
  const inScrollBox = el => { const r = el.getBoundingClientRect(); for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) { const acs = getComputedStyle(a); if (/(auto|scroll|hidden)/.test(acs.overflowY + acs.overflowX) && getComputedStyle(el).position !== 'fixed') { const ar = a.getBoundingClientRect(); if (r.bottom <= ar.top + 1 || r.top >= ar.bottom - 1 || r.right <= ar.left + 1 || r.left >= ar.right - 1) return false; } } return true; };
  // Partly scrolled past a scroll container's edge is a scroll position, not a defect.
  const fullyInScrollBox = el => { const r = el.getBoundingClientRect(); for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) { const acs = getComputedStyle(a); if (/(auto|scroll|hidden)/.test(acs.overflowY + acs.overflowX) && getComputedStyle(el).position !== 'fixed') { const ar = a.getBoundingClientRect(); if (r.top < ar.top - 1 || r.bottom > ar.bottom + 1 || r.left < ar.left - 1 || r.right > ar.right + 1) return false; } } return true; };
  const cls = el => el.tagName.toLowerCase() + '.' + String(el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className || '').split(/\s+/).filter(Boolean).slice(0, 2).join('.');
  const label = el => (el.getAttribute('aria-label') || el.getAttribute('title') || el.getAttribute('placeholder') || el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 40);
  const vis = el => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 0 && r.height > 0 && cs.visibility !== 'hidden' && cs.display !== 'none' && Number(cs.opacity) > 0 && !inClosedDetails(el) && inScrollBox(el); };
  const vw = innerWidth, vh = innerHeight;
  const out = { pageOverflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth, covered: [], floatingOutside: [], clippedText: [], tiny: [] };
  document.querySelectorAll('button, summary, a[href], input, select, textarea, [role="button"], [role="tab"]').forEach(el => {
    if (!vis(el) || el.matches('.shell-sr-only, .sr-only')) return;
    const r = el.getBoundingClientRect();
    if (r.top < 0 || r.bottom > vh || r.left < 0 || r.right > vw || !fullyInScrollBox(el)) return;
    const cx = r.left + r.width / 2, cy = r.top + r.height / 2;
    const hit = document.elementFromPoint(cx, cy);
    if (hit && hit !== el && !el.contains(hit) && !hit.contains(el)) out.covered.push({ el: cls(el), label: label(el), rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)], by: cls(hit) });
    if ((r.width < minHit || r.height < minHit) && el.tagName !== 'INPUT' && el.tagName !== 'A') out.tiny.push({ el: cls(el), label: label(el), rect: [Math.round(r.width), Math.round(r.height)] });
  });
  document.querySelectorAll('body *').forEach(el => {
    if (!vis(el) || el.matches('.shell-sr-only, .sr-only, i, svg, path')) return;
    const cs = getComputedStyle(el); const r = el.getBoundingClientRect();
    if ((cs.position === 'absolute' || cs.position === 'fixed') && r.width > 8 && r.height > 8 && (r.left < -0.5 || r.top < -0.5 || r.right > vw + 0.5 || r.bottom > vh + 0.5)) {
      let scrollParent = null; for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) { const acs = getComputedStyle(a); if (/(auto|scroll)/.test(acs.overflowY + acs.overflowX)) { scrollParent = a; break; } }
      if (!scrollParent) out.floatingOutside.push({ el: cls(el), rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)] });
    }
    if (cs.overflowX === 'hidden' && cs.whiteSpace === 'nowrap' && cs.textOverflow !== 'ellipsis' && el.scrollWidth > el.clientWidth + 2 && el.children.length === 0 && el.textContent.trim()) {
      out.clippedText.push({ el: cls(el), text: el.textContent.trim().slice(0, 40), hidden: el.scrollWidth - el.clientWidth });
    }
  });
  return out;
}
"""



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--session", default="", help="query naming a Guided conversation, e.g. pi_project=…&pi_session=…")
    parser.add_argument("--viewports", default=",".join(DEFAULT_VIEWPORTS))
    parser.add_argument("--min-hit", type=int, default=MIN_HIT, help="smallest acceptable control side in px")
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)
    from playwright.sync_api import sync_playwright

    routes: dict[str, tuple[str, bool]] = {}
    if args.session:
        routes["guided"] = (f"/?{args.session}#guided", False)
        routes["guided-entry"] = (f"/?{args.session}#guided", True)
    routes.update({name: (url, False) for name, url in ROUTES_WITHOUT_SESSION.items()})
    report: dict[str, dict] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for size in args.viewports.split(","):
            width, height = (int(part) for part in size.lower().split("x"))
            page = browser.new_page(viewport={"width": width, "height": height})
            for name, (url, new_conversation) in routes.items():
                page.goto(args.base + url, wait_until="load")
                settle(page)
                if new_conversation:
                    page.click("[data-gpi-rail-new]:visible, .gpi-head-new:visible")
                    settle(page)
                report[f"{name}@{size}"] = page.evaluate(JS, args.min_hit)
            page.close()
        browser.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    total = 0
    for key, result in report.items():
        flags: list[str] = []
        if result["pageOverflowX"] > 0:
            flags.append(f"page overflows horizontally by {result['pageOverflowX']}px")
        flags += [f"covered: {c['el']} «{c['label']}» {c['rect']} by {c['by']}" for c in result["covered"]]
        flags += [f"floating outside viewport: {f['el']} {f['rect']}" for f in result["floatingOutside"]]
        flags += [f"clipped text: {t['el']} «{t['text']}» ({t['hidden']}px hidden)" for t in result["clippedText"]]
        flags += [f"small control (<{args.min_hit}px): {t['el']} «{t['label']}» {t['rect']}" for t in result["tiny"]]
        total += len(flags)
        print(f"== {key}: {len(flags)} flags")
        for flag in flags:
            print("   -", flag)
    print(f"{total} flagged" if total else "nothing flagged")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
