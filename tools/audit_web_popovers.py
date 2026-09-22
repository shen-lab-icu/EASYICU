"""Press every menu and disclosure of the native web UI and check where it lands.

Native <details> only toggles from its own summary, and a popover written for
one placement can land off-screen after its control moves (the model menu
opened below the composer once it left the header). This audit opens each
opener — ``details > summary``, ``[aria-haspopup]``, ``[aria-expanded]`` — with
a real mouse press on every route at several viewport sizes and, for the
floating content it reveals, checks: fully inside the viewport, not clipped by
an overflow ancestor, not covered at its centre, closes on Escape, closes on an
outside press, and toggles closed from its own opener. Inline disclosures are
listed, not judged.

Requires a running EasyICU web server and Playwright with a Chrome channel::

    python tools/audit_web_popovers.py --base http://127.0.0.1:8765 \
        --session "pi_project=<draft id>&pi_session=<session id>" \
        --out output/ui-audit/popovers.json

``--session`` names an existing Guided Copilot conversation so the conversation
composer and the entry composer (a new conversation is created from the rail)
are both audited; without it only the routes that need no conversation run.
The exit status is 1 when any floating menu fails a check.
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


JS = r"""
(() => {
  // Chrome keeps layout boxes for the content of a closed <details>, so
  // anything under a closed details (other than its summary) is not visible.
  const inClosedDetails = el => { for (let a = el; a && a !== document.body; a = a.parentElement) { const p = a.parentElement; if (p && p.tagName === 'DETAILS' && !p.open && a.tagName !== 'SUMMARY') return true; } return false; };
  const vis = el => { const r = el.getBoundingClientRect(); const cs = getComputedStyle(el); return r.width > 8 && r.height > 8 && cs.visibility !== 'hidden' && cs.display !== 'none' && Number(cs.opacity) > 0 && !inClosedDetails(el) && !el.matches('.shell-sr-only, .sr-only, i, svg, path'); };
  const cls = el => el.tagName.toLowerCase() + '.' + String(el.className && el.className.baseVal !== undefined ? el.className.baseVal : el.className || '').split(/\s+/).filter(Boolean).slice(0, 2).join('.');
  const sig = el => { const r = el.getBoundingClientRect(); return `${cls(el)}@${Math.round(r.left)},${Math.round(r.top)},${Math.round(r.width)}x${Math.round(r.height)}`; };
  const label = el => (el.getAttribute('aria-label') || el.getAttribute('title') || el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 48);
  const key = el => { const d = el.tagName === 'SUMMARY' ? el.parentElement : el; return cls(d) + '|' + label(el); };
  const OPENERS = 'details > summary, [aria-haspopup], [aria-expanded]';
  function openers() {
    const seen = {};
    return Array.from(document.querySelectorAll(OPENERS)).filter(vis).map(el => {
      const k = key(el); seen[k] = (seen[k] || 0) + 1;
      const details = el.tagName === 'SUMMARY' ? el.parentElement : null;
      return { key: k, nth: seen[k] - 1, label: label(el), owner: cls(details || el), kind: details ? 'details' : 'button', wasOpen: details ? details.open : el.getAttribute('aria-expanded') === 'true' };
    });
  }
  function find(k, nth) {
    let n = 0;
    for (const el of document.querySelectorAll(OPENERS)) {
      if (!vis(el)) continue;
      if (key(el) === k) { if (n === nth) return el; n += 1; }
    }
    return null;
  }
  function isOpen(k, nth) { const el = find(k, nth); if (!el) return null; return el.tagName === 'SUMMARY' ? el.parentElement.open : el.getAttribute('aria-expanded') === 'true'; }
  function floating() { return Array.from(document.querySelectorAll('body *')).filter(el => { const cs = getComputedStyle(el); return (cs.position === 'absolute' || cs.position === 'fixed') && vis(el); }); }
  function floatingSigs() { return floating().map(sig); }
  function center(k, nth) { const el = find(k, nth); if (!el) return null; el.scrollIntoView({ block: 'center', inline: 'nearest' }); const r = el.getBoundingClientRect(); return [r.left + r.width / 2, r.top + r.height / 2]; }
  function under(x, y) { const el = document.elementFromPoint(x, y); return el ? cls(el) : null; }
  function measure(sigs) {
    const els = floating().filter(el => sigs.includes(sig(el)));
    const outer = els.filter(el => !els.some(o => o !== el && o.contains(el)));
    return outer.map(el => {
      const r = el.getBoundingClientRect(); const vw = innerWidth, vh = innerHeight; const issues = [];
      if (r.left < -0.5 || r.top < -0.5 || r.right > vw + 0.5 || r.bottom > vh + 0.5) issues.push(`outside viewport: (${Math.round(r.left)},${Math.round(r.top)})–(${Math.round(r.right)},${Math.round(r.bottom)}) in ${vw}x${vh}`);
      const cs = getComputedStyle(el);
      for (let a = el.parentElement; a && a !== document.body; a = a.parentElement) {
        const acs = getComputedStyle(a);
        if (cs.position === 'fixed') break;
        if ([acs.overflow, acs.overflowX, acs.overflowY].some(v => v !== 'visible')) {
          const ar = a.getBoundingClientRect();
          if (r.left < ar.left - 0.5 || r.top < ar.top - 0.5 || r.right > ar.right + 0.5 || r.bottom > ar.bottom + 0.5) { issues.push(`clipped by ${cls(a)} (overflow ${acs.overflowX}/${acs.overflowY})`); break; }
        }
      }
      const cx = Math.min(Math.max(r.left + r.width / 2, 0), vw - 1), cy = Math.min(Math.max(r.top + r.height / 2, 0), vh - 1);
      const hit = document.elementFromPoint(cx, cy);
      if (hit && !el.contains(hit) && hit !== el) issues.push(`covered at centre by ${cls(hit)}`);
      return { el: cls(el), rect: [Math.round(r.left), Math.round(r.top), Math.round(r.width), Math.round(r.height)], issues };
    });
  }
  function neutral() {
    // A heading is text, not a control: a safe outside press.
    const h = document.querySelector('.gpi-head-title, .eusk-title, main h1, h1, h2');
    if (h) { const r = h.getBoundingClientRect(); if (r.width && r.height && r.top >= 0 && r.bottom <= innerHeight) return [r.left + Math.min(16, r.width / 2), r.top + r.height / 2]; }
    return [innerWidth / 2, 2];
  }
  function reset(initialOpen) { document.querySelectorAll('details[open]').forEach(d => { if (!initialOpen.includes(sig(d))) d.open = false; }); }
  function openDetailsSigs() { return Array.from(document.querySelectorAll('details[open]')).map(sig); }
  window.__audit = { openers, isOpen, floatingSigs, center, under, measure, neutral, reset, openDetailsSigs };
})();
"""

DISMISSAL_KEYS = ("escapeCloses", "outsideCloses", "ownToggleCloses")
QUIET_NOTES = ("", "inline content", "already open (disclosure)")


def _ev(page, expr, *args):
    page.evaluate(JS)
    return page.evaluate(expr, *args) if args else page.evaluate(expr)


def _route_hash(url: str) -> str:
    return url.split("#")[-1].split("&")[0]



def audit_route(page, base: str, url: str, *, new_conversation: bool) -> list[dict]:
    page.goto(base + url, wait_until="load")
    settle(page)
    if new_conversation:
        page.click("[data-gpi-rail-new]:visible, .gpi-head-new:visible")
        settle(page)
    initial_open = _ev(page, "() => window.__audit.openDetailsSigs()")
    rows: list[dict] = []
    for opener in _ev(page, "() => window.__audit.openers()"):
        if opener["wasOpen"]:
            rows.append({**opener, "note": "already open (disclosure)"})
            continue
        _ev(page, "s => window.__audit.reset(s)", initial_open)
        page.keyboard.press("Escape")
        page.wait_for_timeout(120)
        args = [opener["key"], opener["nth"]]
        point = _ev(page, "([k, n]) => window.__audit.center(k, n)", args)
        if not point:
            rows.append({**opener, "note": "gone before it could be opened (re-render)"})
            continue
        page.wait_for_timeout(120)
        point = _ev(page, "([k, n]) => window.__audit.center(k, n)", args) or point
        covered_by = _ev(page, "([x, y]) => window.__audit.under(x, y)", point)
        before = set(_ev(page, "() => window.__audit.floatingSigs()"))
        page.mouse.click(point[0], point[1])
        page.wait_for_timeout(300)
        if _route_hash(page.url) != _route_hash(url):
            rows.append({**opener, "note": f"navigated to #{page.url.split('#')[-1]}"})
            page.goto(base + url, wait_until="load")
            settle(page)
            continue
        after = set(_ev(page, "() => window.__audit.floatingSigs()"))
        new = sorted(after - before)
        opened = _ev(page, "([k, n]) => window.__audit.isOpen(k, n)", args)
        row = {**opener, "opened": opened, "floating": _ev(page, "s => window.__audit.measure(s)", new) if new else []}
        if opened is False and not new:
            row["note"] = f"press did not open it (point was on {covered_by})"
        elif not new:
            row["note"] = "inline content"
        else:
            still_open = lambda: bool(set(_ev(page, "() => window.__audit.floatingSigs()")) & set(new))  # noqa: E731

            def reopen() -> None:
                if not _ev(page, "([k, n]) => window.__audit.isOpen(k, n)", args):
                    where = _ev(page, "([k, n]) => window.__audit.center(k, n)", args) or point
                    page.mouse.click(where[0], where[1])
                    page.wait_for_timeout(250)

            page.keyboard.press("Escape")
            page.wait_for_timeout(200)
            row["escapeCloses"] = not still_open()
            reopen()
            nx, ny = _ev(page, "() => window.__audit.neutral()")
            page.mouse.click(nx, ny)
            page.wait_for_timeout(250)
            row["outsideCloses"] = not still_open()
            reopen()
            where = _ev(page, "([k, n]) => window.__audit.center(k, n)", args) or point
            page.mouse.click(where[0], where[1])
            page.wait_for_timeout(250)
            row["ownToggleCloses"] = not still_open()
        rows.append(row)
    _ev(page, "s => window.__audit.reset(s)", initial_open)
    return rows


def flagged(row: dict) -> tuple[list[str], list[str], str]:
    issues = [issue for item in row.get("floating", []) for issue in item["issues"]]
    dismiss = [key for key in DISMISSAL_KEYS if key in row and not row[key]]
    note = row.get("note", "")
    return issues, dismiss, note if note not in QUIET_NOTES else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", default="http://127.0.0.1:8765")
    parser.add_argument("--session", default="", help="query naming a Guided conversation, e.g. pi_project=…&pi_session=…")
    parser.add_argument("--viewports", default=",".join(DEFAULT_VIEWPORTS))
    parser.add_argument("--out", type=Path, default=None, help="write the full report as JSON")
    args = parser.parse_args(argv)
    from playwright.sync_api import sync_playwright

    routes: dict[str, tuple[str, bool]] = {}
    if args.session:
        routes["guided-conversation"] = (f"/?{args.session}#guided", False)
        routes["guided-entry"] = (f"/?{args.session}#guided", True)
    routes.update({name: (url, False) for name, url in ROUTES_WITHOUT_SESSION.items()})
    report: dict[str, list[dict]] = {}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for size in args.viewports.split(","):
            width, height = (int(part) for part in size.lower().split("x"))
            page = browser.new_page(viewport={"width": width, "height": height})
            for name, (url, new_conversation) in routes.items():
                try:
                    report[f"{name}@{size}"] = audit_route(page, args.base, url, new_conversation=new_conversation)
                except Exception as exc:  # one broken route must not hide the others
                    report[f"{name}@{size}"] = [{"note": f"route failed: {str(exc).splitlines()[0][:160]}"}]
            page.close()
        browser.close()
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    failures = 0
    for key, rows in report.items():
        floating = sum(1 for row in rows if row.get("floating"))
        print(f"== {key}: {len(rows)} openers, {floating} floating")
        for row in rows:
            issues, dismiss, note = flagged(row)
            if not (row.get("floating") or issues or dismiss or note):
                continue
            bad = bool(issues or dismiss or note)
            failures += bad
            where = "; ".join(f"{item['el']} {item['rect']}" for item in row.get("floating", []))
            print(f"  {'!!' if bad else 'OK'} {row.get('owner', '')} «{row.get('label', '')}» {where} {note}")
            for issue in issues:
                print(f"       - {issue}")
            for key_name in dismiss:
                print(f"       - {key_name}: false")
    print(f"{failures} flagged" if failures else "all floating menus pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
