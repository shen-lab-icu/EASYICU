#!/usr/bin/env python3
"""Non-destructive button and control audit for the native FastAPI UI.

The audit opens each native route, enumerates visible clickable controls, and
clicks controls that are safe to exercise automatically. It records whether the
click produced an observable UI effect: route/hash change, active state change,
modal/picker open, DOM text change, focus change, or similar.

Dangerous or side-effect-heavy controls are explicitly skipped but still listed
in the report, so "not tested" stays visible.

Run from ``EASYICU/`` with a FastAPI server already listening:

    python tools/qa_native_fastapi_button_audit.py --base-url http://127.0.0.1:8780/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


ROUTES = [
    "entry",
    "extraction",
    "patient",
    "cohort",
    "crossdb",
    "agent",
    "settings",
    "dictionary",
    "states",
    "help",
    "guided",
]

VIEWPORTS = [
    ("desktop", 1440, 900),
    ("mobile", 393, 852),
]

SKIP_ATTRS = {
    "data-src-remove": "registry_remove_requires_user_confirm",
    "data-src-rename": "rename_requires_prompt",
    "data-ag-external-run": "external_provider_run_skipped",
    "data-ag-live-signoff": "human_signoff_skipped",
    "data-ag-signoff": "human_signoff_skipped",
    "data-ag-artifact-download": "download_skipped",
    "data-ag-bundle-download": "download_skipped",
}

SKIP_TEXT_RE = re.compile(
    r"\b("
    r"remove|delete|sign\s*off|write local sign-off|confirm all|"
    r"run full with provider|extract\b|convert raw|retry conversion|"
    r"download|export bundle|download json|export diagnostics|"
    r"add\b|use this folder|choose another folder"
    r")\b",
    re.IGNORECASE,
)

SAFE_RUN_TEXT_RE = re.compile(
    r"\b(run benchmark|re-run|render|generate and load demo workspace|"
    r"load selected exports|load local export|retry metadata check|"
    r"load real filter options|preview supported filters)\b",
    re.IGNORECASE,
)

REQUIRES_INPUT_TEXT_RE = re.compile(
    r"\b(change|add|use this folder|open workspace)\b",
    re.IGNORECASE,
)

COLLECT_JS = r"""
(scope) => {
  const selector = [
    'button',
    '[role="button"]',
    'a[href]',
    '.switch',
    '.icobtn',
    '.radio',
    '.src-row',
    '.studycard',
    '.outcard'
  ].join(',');
  const roots = scope === 'content'
    ? [
        document.querySelector('.content'),
        document.querySelector('.gd-shell'),
        document.querySelector('.entry-shell'),
      ].filter(Boolean)
    : [document.body];
  const seen = new Set();
  const out = [];
  function visible(el) {
    if (!el || seen.has(el)) return false;
    seen.add(el);
    if (el.closest('template, script, style')) return false;
    if (el.closest('#cpDock:not(.open), #cpBackdrop:not(.open)')) return false;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) return false;
    const r = el.getBoundingClientRect();
    if (r.width < 2 || r.height < 2) return false;
    if (r.right < 0 || r.left > window.innerWidth || r.bottom < 0 || r.top > window.innerHeight) return false;
    return true;
  }
  function dataAttrs(el) {
    const attrs = {};
    for (const a of el.attributes || []) {
      if (a.name.startsWith('data-')) attrs[a.name] = a.value || '';
      if (a.name === 'aria-disabled' || a.name === 'disabled' || a.name === 'href') attrs[a.name] = a.value || '';
    }
    return attrs;
  }
  function label(el) {
    const aria = el.getAttribute('aria-label') || el.getAttribute('title') || '';
    const txt = (el.innerText || el.textContent || '').replace(/\s+/g, ' ').trim();
    return (txt || aria || el.getAttribute('href') || el.className || el.tagName).slice(0, 140);
  }
  for (const root of roots) {
    for (const el of root.querySelectorAll(selector)) {
      if (!visible(el)) continue;
      const r = el.getBoundingClientRect();
      const id = `qa_click_${out.length}`;
      el.setAttribute('data-qa-click-id', id);
      out.push({
        id,
        label: label(el),
        tag: el.tagName.toLowerCase(),
        role: el.getAttribute('role') || '',
        classes: typeof el.className === 'string' ? el.className.replace(/\s+/g, ' ').trim().slice(0, 180) : '',
        attrs: dataAttrs(el),
        rect: {
          x: Math.round(r.x),
          y: Math.round(r.y),
          width: Math.round(r.width),
          height: Math.round(r.height)
        },
        disabled: !!el.disabled || el.getAttribute('aria-disabled') === 'true',
      });
    }
  }
  return out;
}
"""

STATE_JS = r"""
() => {
  function text(el) {
    return (el ? (el.innerText || el.textContent || '') : '').replace(/\s+/g, ' ').trim();
  }
  const active = [...document.querySelectorAll('.active,.on,[aria-checked="true"],[aria-selected="true"]')]
    .map(el => `${el.tagName.toLowerCase()}.${typeof el.className === 'string' ? el.className.replace(/\s+/g, '.') : ''}:${text(el).slice(0, 80)}`)
    .slice(0, 120);
  const main = document.querySelector('.content') || document.querySelector('.gd-main') || document.querySelector('#app') || document.body;
  const dialogs = [...document.querySelectorAll('[role="dialog"], .eu-modal-ov, .eu-pick-backdrop, #cpDock.open')]
    .filter(el => getComputedStyle(el).display !== 'none')
    .map(el => text(el).slice(0, 120));
  const focus = document.activeElement ? {
    tag: document.activeElement.tagName.toLowerCase(),
    label: text(document.activeElement).slice(0, 80),
    aria: document.activeElement.getAttribute('aria-label') || '',
    cls: typeof document.activeElement.className === 'string' ? document.activeElement.className.slice(0, 80) : '',
  } : null;
  const bodyText = text(main).slice(0, 12000);
  return {
    hash: location.hash,
    path: location.pathname + location.search,
    title: text(document.querySelector('h1') || document.querySelector('.gd-name') || document.querySelector('.home-h1')).slice(0, 120),
    active,
    dialogs,
    focus,
    bodyText,
    bodyLength: bodyText.length,
    bodyHtmlLength: main.innerHTML.length,
    viewportOverflowX: Math.max(document.documentElement.scrollWidth, document.body.scrollWidth) - window.innerWidth,
  };
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8780/", help="FastAPI frontend base URL")
    parser.add_argument("--out-dir", default="output/playwright", help="Directory for the JSON report")
    parser.add_argument("--routes", nargs="*", default=ROUTES, help="Routes to audit")
    parser.add_argument("--viewports", nargs="*", default=[name for name, _, _ in VIEWPORTS], help="Viewport names to audit")
    parser.add_argument("--scope", choices=["content", "all"], default="content", help="Clickable scope to audit")
    parser.add_argument("--max-clicks", type=int, default=0, help="Stop after N safe clicks; 0 means no cap")
    parser.add_argument("--fail-on-noop", action="store_true", help="Exit non-zero when safe controls have no observable effect")
    parser.add_argument("--collect-only", action="store_true", help="Only enumerate and classify controls; do not click")
    parser.add_argument("--progress", action="store_true", help="Print route progress while the audit runs")
    parser.add_argument("--initial-wait-ms", type=int, default=350, help="Wait after opening a route")
    parser.add_argument("--datamode-wait-ms", type=int, default=100, help="Wait after forcing Demo mode")
    parser.add_argument("--after-click-wait-ms", type=int, default=150, help="Wait after a click before comparing state")
    parser.add_argument("--networkidle-ms", type=int, default=200, help="Best-effort networkidle wait after a click")
    return parser.parse_args()


def normalize_base(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def assert_ready(base_url: str) -> None:
    try:
        payload = get_json(base_url + "api/health")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise SystemExit(f"FastAPI server is not ready at {base_url}: {exc}") from exc
    if payload.get("status") != "ok":
        raise SystemExit(f"Unexpected /api/health response at {base_url}: {payload}")


def digest(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def state_signature(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "hash": state.get("hash"),
        "path": state.get("path"),
        "title": state.get("title"),
        "active_hash": digest(state.get("active") or []),
        "dialog_hash": digest(state.get("dialogs") or []),
        "focus_hash": digest(state.get("focus") or {}),
        "body_hash": digest(state.get("bodyText") or ""),
        "body_html_len": state.get("bodyHtmlLength"),
    }


def classify(candidate: dict[str, Any]) -> tuple[str, str]:
    if candidate.get("disabled"):
        return "skip", "disabled"
    attrs = candidate.get("attrs") or {}
    classes = str(candidate.get("classes") or "")
    if re.search(r"(^| )(active|on)( |$)", classes) and (
        re.search(r"(^| )(radio|tab|chip|wsitem|nav-item|cp-entry)( |$)", classes)
        or any(attr in attrs for attr in ["data-nav", "data-lang", "data-datamode", "data-cohtab", "data-ptab"])
    ):
        return "skip", "already_selected"
    for attr, reason in SKIP_ATTRS.items():
        if attr in attrs:
            return "skip", reason
    label = candidate.get("label") or ""
    if SKIP_TEXT_RE.search(label) and not SAFE_RUN_TEXT_RE.search(label):
        if REQUIRES_INPUT_TEXT_RE.search(label):
            return "skip", "requires_user_input"
        return "skip", "side_effect_or_download"
    href = attrs.get("href") or ""
    if href.startswith("http"):
        return "skip", "external_link"
    if href.startswith("#set-"):
        return "click", "settings_anchor"
    return "click", "safe"


def new_page(
    context: Any,
    base_url: str,
    route: str,
    *,
    initial_wait_ms: int,
    datamode_wait_ms: int,
) -> Any:
    page = context.new_page()
    page.add_init_script(
        """() => {
          try {
            localStorage.setItem('easyicu_home_data', 'demo');
            localStorage.setItem('easyicu_lang', 'en');
            localStorage.setItem('easyicu_onboarded', '1');
          } catch (e) {}
        }"""
    )
    page.goto(base_url + "?_v=button-audit#" + route, wait_until="domcontentloaded", timeout=20000)
    # Native routes hydrate from several metadata APIs. Give the first paint a
    # short stable window so collected click ids are less likely to detach.
    page.wait_for_timeout(max(0, initial_wait_ms))
    try:
        page.evaluate("window.setDataMode && window.setDataMode('demo', {force:true})")
        page.wait_for_timeout(max(0, datamode_wait_ms))
    except Exception:
        pass
    return page


def collect_candidates(page: Any, scope: str) -> list[dict[str, Any]]:
    return page.evaluate(COLLECT_JS, scope)


def click_candidate(page: Any, candidate: dict[str, Any]) -> None:
    selector = f"[data-qa-click-id='{candidate['id']}']"
    locator = page.locator(selector)
    try:
        locator.scroll_into_view_if_needed(timeout=1000)
    except PlaywrightTimeoutError:
        pass
    locator.click(timeout=2000, force=False)


def audit_candidate(
    context: Any,
    base_url: str,
    route: str,
    candidate_index: int,
    candidate_hint: dict[str, Any],
    scope: str,
    initial_wait_ms: int,
    datamode_wait_ms: int,
    after_click_wait_ms: int,
    networkidle_ms: int,
) -> dict[str, Any]:
    page = new_page(
        context,
        base_url,
        route,
        initial_wait_ms=initial_wait_ms,
        datamode_wait_ms=datamode_wait_ms,
    )
    console_errors: list[str] = []
    page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: console_errors.append(str(exc)))
    try:
        candidates = collect_candidates(page, scope)
        if candidate_index >= len(candidates):
            return {
                "status": "error",
                "error": "candidate_missing_after_reload",
                "candidate_index": candidate_index,
                "candidate": candidate_hint,
                "reloaded_candidate_count": len(candidates),
            }
        candidate = candidates[candidate_index]
        status, reason = classify(candidate)
        result = {"candidate_index": candidate_index, "candidate": candidate, "status": status, "reason": reason}
        if status != "click":
            return result
        before = page.evaluate(STATE_JS)
        before_sig = state_signature(before)
        try:
            click_candidate(page, candidate)
            page.wait_for_timeout(max(0, after_click_wait_ms))
            # Give lightweight API-backed controls a short chance to repaint.
            if networkidle_ms > 0:
                try:
                    page.wait_for_load_state("networkidle", timeout=networkidle_ms)
                except PlaywrightTimeoutError:
                    pass
        except Exception as exc:  # Playwright wraps clickability failures in several exception types.
            after = page.evaluate(STATE_JS)
            result.update(
                {
                    "status": "click_error",
                    "error": str(exc),
                    "before": before_sig,
                    "after": state_signature(after),
                    "consoleErrors": console_errors,
                }
            )
            return result
        after = page.evaluate(STATE_JS)
        after_sig = state_signature(after)
        changed_keys = [key for key, value in before_sig.items() if after_sig.get(key) != value]
        result.update(
            {
                "status": "clicked",
                "effect": "changed" if changed_keys else "no_observable_effect",
                "changedKeys": changed_keys,
                "before": before_sig,
                "after": after_sig,
                "consoleErrors": console_errors,
            }
        )
        if console_errors:
            result["effect"] = "console_error"
        return result
    finally:
        page.close()


def audit_route(
    context: Any,
    base_url: str,
    route: str,
    max_clicks: int,
    scope: str,
    *,
    collect_only: bool,
    initial_wait_ms: int,
    datamode_wait_ms: int,
    after_click_wait_ms: int,
    networkidle_ms: int,
) -> dict[str, Any]:
    page = new_page(
        context,
        base_url,
        route,
        initial_wait_ms=initial_wait_ms,
        datamode_wait_ms=datamode_wait_ms,
    )
    try:
        candidates = collect_candidates(page, scope)
    finally:
        page.close()

    route_results: list[dict[str, Any]] = []
    safe_seen = 0
    for idx, candidate in enumerate(candidates):
        status, reason = classify(candidate)
        if status != "click":
            route_results.append({"candidate_index": idx, "candidate": candidate, "status": status, "reason": reason})
            continue
        if collect_only:
            route_results.append({"candidate_index": idx, "candidate": candidate, "status": "skip", "reason": "collect_only_safe"})
            continue
        if max_clicks and safe_seen >= max_clicks:
            route_results.append({"candidate_index": idx, "candidate": candidate, "status": "skip", "reason": "max_clicks_cap"})
            continue
        safe_seen += 1
        route_results.append(
            audit_candidate(
                context,
                base_url,
                route,
                idx,
                candidate,
                scope,
                initial_wait_ms,
                datamode_wait_ms,
                after_click_wait_ms,
                networkidle_ms,
            )
        )

    return {
        "route": route,
        "candidate_count": len(candidates),
        "clicked_count": sum(1 for item in route_results if item.get("status") == "clicked"),
        "skipped_count": sum(1 for item in route_results if item.get("status") == "skip"),
        "results": route_results,
    }


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    totals = {
        "candidates": 0,
        "clicked": 0,
        "changed": 0,
        "no_observable_effect": 0,
        "click_errors": 0,
        "errors": 0,
        "console_errors": 0,
        "skipped": 0,
    }
    failures: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for viewport in report["viewports"]:
        for route in viewport["routes"]:
            totals["candidates"] += route["candidate_count"]
            for item in route["results"]:
                status = item.get("status")
                effect = item.get("effect")
                if status == "clicked":
                    totals["clicked"] += 1
                    if effect == "changed":
                        totals["changed"] += 1
                    elif effect == "no_observable_effect":
                        totals["no_observable_effect"] += 1
                        failures.append(
                            {
                                "viewport": viewport["name"],
                                "route": route["route"],
                                "label": (item.get("candidate") or {}).get("label"),
                                "classes": (item.get("candidate") or {}).get("classes"),
                                "attrs": (item.get("candidate") or {}).get("attrs"),
                                "reason": effect,
                            }
                        )
                    elif effect == "console_error":
                        totals["console_errors"] += 1
                        failures.append(
                            {
                                "viewport": viewport["name"],
                                "route": route["route"],
                                "label": (item.get("candidate") or {}).get("label"),
                                "reason": "console_error",
                                "consoleErrors": item.get("consoleErrors"),
                            }
                        )
                elif status == "click_error":
                    totals["click_errors"] += 1
                    failures.append(
                        {
                            "viewport": viewport["name"],
                            "route": route["route"],
                            "label": (item.get("candidate") or {}).get("label"),
                            "reason": "click_error",
                            "error": item.get("error"),
                        }
                    )
                elif status == "error":
                    totals["errors"] += 1
                    failures.append(
                        {
                            "viewport": viewport["name"],
                            "route": route["route"],
                            "label": (item.get("candidate") or {}).get("label"),
                            "reason": item.get("error") or "error",
                            "candidate_index": item.get("candidate_index"),
                        }
                    )
                elif status == "skip":
                    totals["skipped"] += 1
                    skipped[item.get("reason") or "unknown"] = skipped.get(item.get("reason") or "unknown", 0) + 1
    return {"totals": totals, "failures": failures, "skipped_by_reason": skipped}


def main() -> int:
    args = parse_args()
    base_url = normalize_base(args.base_url)
    assert_ready(base_url)
    selected_viewports = {name for name in args.viewports}
    viewports = [vp for vp in VIEWPORTS if vp[0] in selected_viewports]
    if not viewports:
        raise SystemExit(f"No matching viewports selected: {args.viewports}")

    run_dir = Path(args.out_dir) / f"native_fastapi_button_audit_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "base_url": base_url,
        "routes_requested": args.routes,
        "scope": args.scope,
        "viewports": [],
        "safety": {
            "mode": "non_destructive",
            "skipped_attrs": sorted(SKIP_ATTRS),
            "skip_text_pattern": SKIP_TEXT_RE.pattern,
        },
    }

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for viewport_name, width, height in viewports:
            context = browser.new_context(viewport={"width": width, "height": height})
            viewport_report = {"name": viewport_name, "width": width, "height": height, "routes": []}
            for route in args.routes:
                if args.progress:
                    print(f"[button-audit] {viewport_name} #{route}", flush=True)
                route_report = audit_route(
                    context,
                    base_url,
                    route,
                    args.max_clicks,
                    args.scope,
                    collect_only=args.collect_only,
                    initial_wait_ms=args.initial_wait_ms,
                    datamode_wait_ms=args.datamode_wait_ms,
                    after_click_wait_ms=args.after_click_wait_ms,
                    networkidle_ms=args.networkidle_ms,
                )
                viewport_report["routes"].append(route_report)
                if args.progress:
                    print(
                        "[button-audit] "
                        f"{viewport_name} #{route}: candidates={route_report['candidate_count']} "
                        f"clicked={route_report['clicked_count']} skipped={route_report['skipped_count']}",
                        flush=True,
                    )
            report["viewports"].append(viewport_report)
            context.close()
        browser.close()

    summary = summarize(report)
    report["summary"] = summary
    report_path = run_dir / "button_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Button audit report: {report_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.fail_on_noop and summary["failures"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
