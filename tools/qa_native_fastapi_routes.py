#!/usr/bin/env python3
"""Playwright QA for the native FastAPI EasyICU frontend routes.

Run from ``EASYICU/`` with the FastAPI server already listening:

    python tools/qa_native_fastapi_routes.py --base-url http://127.0.0.1:8765/

The script writes screenshots and a JSON report under ``output/playwright/``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
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

QA_JS = r"""
() => {
  const vw = window.innerWidth;
  const vh = window.innerHeight;
  const doc = document.documentElement;
  const body = document.body;
  const main = document.querySelector('.content') || document.querySelector('.gd-shell') || document.querySelector('.entry-shell') || document.querySelector('#app');
  const walker = document.createTreeWalker(body, NodeFilter.SHOW_ELEMENT);
  const offscreen = [];
  const clipped = [];
  function label(el) {
    const cls = typeof el.className === 'string' ? el.className : '';
    return {
      tag: el.tagName.toLowerCase(),
      id: el.id || '',
      cls: cls.replace(/\s+/g, '.').slice(0, 96),
    };
  }
  while (walker.nextNode()) {
    const el = walker.currentNode;
    if (el.closest('#cpDock:not(.open), #cpBackdrop:not(.open), template, script, style')) continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || Number(cs.opacity) === 0) continue;
    const r = el.getBoundingClientRect();
    if (r.width < 1 || r.height < 1) continue;
    if (cs.position === 'fixed' && (r.left >= vw || r.right <= 0)) continue;
    if (r.right > vw + 1 || r.left < -1) {
      offscreen.push({ ...label(el), left: Math.round(r.left), right: Math.round(r.right), width: Math.round(r.width) });
    }
    const clipsX = el.scrollWidth > el.clientWidth + 2;
    const clipsY = el.scrollHeight > el.clientHeight + 2;
    if ((clipsX || clipsY) && /(hidden|clip)/.test(`${cs.overflow} ${cs.overflowX} ${cs.overflowY}`)) {
      clipped.push({ ...label(el), scrollWidth: el.scrollWidth, clientWidth: el.clientWidth, scrollHeight: el.scrollHeight, clientHeight: el.clientHeight });
    }
  }
  const title = (document.querySelector('h1') || document.querySelector('.home-h1') || document.querySelector('.gd-name') || {}).textContent || '';
  return {
    hash: location.hash,
    title: title.trim(),
    mainTextLength: main ? main.textContent.trim().length : 0,
    overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - vw,
    overflowY: Math.max(doc.scrollHeight, body.scrollHeight) - vh,
    offscreenCount: offscreen.length,
    offscreenSample: offscreen.slice(0, 10),
    clippedCount: clipped.length,
    clippedSample: clipped.slice(0, 10),
  };
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765/", help="FastAPI frontend base URL")
    parser.add_argument("--out-dir", default="output/playwright", help="Directory for screenshots and JSON report")
    parser.add_argument("--no-screenshots", action="store_true", help="Skip screenshot capture")
    parser.add_argument("--strict-offscreen", action="store_true", help="Also fail on offscreen/clipped samples")
    return parser.parse_args()


def normalize_base(url: str) -> str:
    return url if url.endswith("/") else url + "/"


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


def assert_server_ready(base_url: str) -> None:
    try:
      with urllib.request.urlopen(base_url + "api/health", timeout=5) as resp:
          payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
      raise SystemExit(f"FastAPI server is not ready at {base_url}: {exc}") from exc
    if payload.get("status") != "ok":
      raise SystemExit(f"Unexpected /api/health response at {base_url}: {payload}")


def collect_route(page: Any, base_url: str, route: str, viewport_name: str, shot_dir: Path, screenshots: bool) -> dict[str, Any]:
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(base_url + "#" + route, wait_until="networkidle")
    page.wait_for_timeout(250)
    result = page.evaluate(QA_JS)
    result.update({"route": route, "viewport": viewport_name, "consoleErrors": errors})
    if screenshots:
      shot = shot_dir / f"{viewport_name}_{slug(route)}.png"
      page.screenshot(path=str(shot), full_page=True)
      result["screenshot"] = str(shot)
    return result


def collect_unknown_hash(page: Any, base_url: str, shot_dir: Path, screenshots: bool) -> dict[str, Any]:
    errors: list[str] = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(base_url + "#settings", wait_until="networkidle")
    page.wait_for_timeout(150)
    page.evaluate("location.hash = '#__qa_unknown_hash__'")
    try:
      page.wait_for_function("location.hash === '#entry'", timeout=2000)
    except PlaywrightTimeoutError:
      pass
    page.wait_for_timeout(150)
    result = page.evaluate(QA_JS)
    back_hash = None
    try:
      page.go_back(wait_until="domcontentloaded", timeout=2000)
      page.wait_for_timeout(150)
      back_hash = page.evaluate("location.hash")
    except PlaywrightTimeoutError:
      back_hash = "timeout"
    result.update({
      "route": "__unknown_hash_runtime__",
      "viewport": "mobile",
      "consoleErrors": errors,
      "backHashAfterFallback": back_hash,
    })
    if screenshots:
      shot = shot_dir / "mobile_unknown_hash_runtime.png"
      page.screenshot(path=str(shot), full_page=True)
      result["screenshot"] = str(shot)
    return result


def validate(results: list[dict[str, Any]], strict_offscreen: bool) -> list[str]:
    failures: list[str] = []
    for item in results:
      label = f"{item['viewport']} #{item['route']}"
      if item.get("mainTextLength", 0) <= 0:
        failures.append(f"{label}: main container is empty")
      if item.get("consoleErrors"):
        failures.append(f"{label}: console errors: {item['consoleErrors']}")
      if item.get("overflowX", 0) > 1:
        failures.append(f"{label}: horizontal overflow {item['overflowX']}px")
      if strict_offscreen and (item.get("offscreenCount", 0) or item.get("clippedCount", 0)):
        failures.append(f"{label}: offscreen={item.get('offscreenCount')} clipped={item.get('clippedCount')}")

    help_results = [r for r in results if r["route"] == "help"]
    if not help_results or not any("reviewable path" in r.get("title", "").lower() for r in help_results):
      failures.append("#help did not render the Get Started/Help alias")

    unknown = next((r for r in results if r["route"] == "__unknown_hash_runtime__"), None)
    if not unknown:
      failures.append("unknown hash runtime assertion did not run")
    else:
      if unknown.get("hash") != "#entry":
        failures.append(f"unknown hash did not rewrite to #entry: {unknown.get('hash')}")
      if "Welcome to EasyICU" not in (unknown.get("title") or ""):
        failures.append(f"unknown hash did not render Entry: {unknown.get('title')}")
      if unknown.get("backHashAfterFallback") != "#settings":
        failures.append(f"history back after unknown fallback did not return to #settings: {unknown.get('backHashAfterFallback')}")
    return failures


def main() -> int:
    args = parse_args()
    base_url = normalize_base(args.base_url)
    assert_server_ready(base_url)

    run_dir = Path(args.out_dir) / f"native_fastapi_route_qa_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    with sync_playwright() as p:
      browser = p.chromium.launch(headless=True)
      for viewport_name, width, height in VIEWPORTS:
        for route in ROUTES:
          page = browser.new_page(viewport={"width": width, "height": height})
          try:
            results.append(collect_route(page, base_url, route, viewport_name, run_dir, not args.no_screenshots))
          finally:
            page.close()
      page = browser.new_page(viewport={"width": 393, "height": 852})
      try:
        results.append(collect_unknown_hash(page, base_url, run_dir, not args.no_screenshots))
      finally:
        page.close()
      browser.close()

    report = {
      "base_url": base_url,
      "routes": ROUTES,
      "viewports": [{"name": n, "width": w, "height": h} for n, w, h in VIEWPORTS],
      "results": results,
    }
    report_path = run_dir / "route_qa.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Route QA report: {report_path}")
    print("viewport route overflowX offscreen clipped consoleErrors hash title")
    for r in results:
      print(
        f"{r['viewport']} {r['route']} {r.get('overflowX')} "
        f"{r.get('offscreenCount')} {r.get('clippedCount')} "
        f"{len(r.get('consoleErrors') or [])} {r.get('hash')} {r.get('title')!r}"
      )

    failures = validate(results, args.strict_offscreen)
    if failures:
      print("FAILURES:", file=sys.stderr)
      for failure in failures:
        print(f"- {failure}", file=sys.stderr)
      return 1
    print("Route QA passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
