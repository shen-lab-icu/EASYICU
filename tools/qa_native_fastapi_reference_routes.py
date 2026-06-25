#!/usr/bin/env python3
"""Browser QA for native FastAPI Dictionary and Workspace States routes."""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

from qa_native_fastapi_extraction_job_flow import DEFAULT_OUT_ROOT, server_context


def _provider_status(base_url: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url}/api/agent-runs/provider-status", timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8")).get("provider_status", {})


def _row_count(page: Page) -> int:
    return page.locator(".dict-row:not(.dict-head)").count()


def _active_text(page: Page, selector: str) -> str:
    return page.locator(selector).first.inner_text(timeout=5000).strip()


def _wait_rows_nonzero(page: Page, timeout: float = 10.0) -> int:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = _row_count(page)
        if rows > 0:
            return rows
        time.sleep(0.2)
    raise AssertionError("dictionary rows did not render")


def _goto(page: Page, base_url: str, route: str, tag: str) -> None:
    page.goto(f"{base_url}/?_v={tag}#{route}", wait_until="domcontentloaded")


def _run_browser_flow(*, base_url: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []
    result: dict[str, Any] = {"base_url": base_url}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        _goto(page, base_url, "dictionary", "stage59-reference")
        page.locator("#dictSearchInput").first.wait_for(state="visible", timeout=10000)
        initial_rows = _wait_rows_nonzero(page)
        page.locator("#dictSearchInput").first.fill("lactate")
        page.wait_for_function("() => document.body.innerText.toLowerCase().includes('lactate')", timeout=5000)
        search_rows = _row_count(page)
        table_text = page.locator(".dict-table").first.inner_text(timeout=5000).lower()
        has_lactate = "lactate" in table_text or "乳酸" in table_text or re.search(r"\blact\b", table_text) is not None
        page.locator("[data-dict-clear]").first.click()
        page.wait_for_function("() => document.querySelector('#dictSearchInput') && document.querySelector('#dictSearchInput').value === ''", timeout=5000)
        cleared_rows = _row_count(page)
        page.locator("[data-dict-cat]").nth(1).click()
        page.wait_for_function("() => !!document.querySelector('[data-dict-cat].on:not([data-dict-cat=\"all\"])')", timeout=5000)
        category_rows = _row_count(page)
        active_category = _active_text(page, "[data-dict-cat].on")

        result["dictionary"] = {
            "initial_rows": initial_rows,
            "search_rows": search_rows,
            "has_lactate": has_lactate,
            "cleared_rows": cleared_rows,
            "category_rows": category_rows,
            "active_category": active_category,
        }

        _goto(page, base_url, "states", "stage59-reference")
        page.locator("#ctxSeg").first.wait_for(state="visible", timeout=10000)
        no_action_reference_button = page.locator("button", has_text="Reference").count() == 0
        disabled_reference_buttons = page.evaluate(
            """
            () => [...document.querySelectorAll('.st-stage button')]
              .filter(btn => !btn.closest('#ctxSeg,#modeSeg,#stateSeg'))
              .every(btn => btn.getAttribute('aria-disabled') === 'true')
            """
        )
        page.locator("#ctxSeg [data-ctx='crossdb']").first.click()
        page.wait_for_function("() => document.querySelector('#ctxSeg [data-ctx=\"crossdb\"]').classList.contains('active')", timeout=5000)
        page.locator("#modeSeg [data-mode='real']").first.click()
        page.wait_for_function("() => document.querySelector('#modeSeg [data-mode=\"real\"]').classList.contains('active')", timeout=5000)
        page.locator("#stateSeg [data-state='error']").first.click()
        page.wait_for_function("() => document.querySelector('#stateSeg [data-state=\"error\"]').classList.contains('active')", timeout=5000)
        error_text = page.locator("#stStage").first.inner_text(timeout=5000)
        page.locator("#stateSeg [data-state='success']").first.click()
        page.wait_for_function("() => document.querySelector('#stateSeg [data-state=\"success\"]').classList.contains('active')", timeout=5000)
        success_text = page.locator("#stStage").first.inner_text(timeout=5000)
        toast_visible = page.locator(".st-toast").first.is_visible()
        page.locator(".st-toast .x").first.click()
        page.wait_for_function("() => !document.querySelector('.st-toast')", timeout=5000)
        toast_dismissed = page.locator(".st-toast").count() == 0

        result["states"] = {
            "no_action_reference_button": no_action_reference_button,
            "disabled_reference_buttons": bool(disabled_reference_buttons),
            "context_crossdb": "Cross-DB Benchmark" in success_text,
            "mode_real": "Real · local" in success_text,
            "error_state_text": "Database connection failed" in error_text,
            "success_state_text": "Benchmark assembled" in success_text,
            "toast_visible": toast_visible,
            "toast_dismissed": toast_dismissed,
        }

        result["overflowX"] = page.evaluate(
            "() => Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth)"
        )
        page.screenshot(path=str(out_dir / "reference_routes.png"), full_page=True)
        result["screenshot"] = str(out_dir / "reference_routes.png")
        result["console_errors"] = console_errors
        result["page_errors"] = page_errors
        browser.close()

    result["provider_status"] = _provider_status(base_url)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Existing FastAPI server, for example http://127.0.0.1:8782")
    parser.add_argument("--port", type=int, default=8796)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"native_fastapi_reference_routes_{stamp}"

    with server_context(args.base_url, args.port) as server:
        result = _run_browser_flow(base_url=server.base_url, out_dir=out_dir)

    failures: list[str] = []
    dictionary = result.get("dictionary") if isinstance(result.get("dictionary"), dict) else {}
    if dictionary.get("initial_rows", 0) <= 0:
        failures.append("dictionary_initial_empty")
    if dictionary.get("search_rows", 0) <= 0 or dictionary.get("has_lactate") is not True:
        failures.append("dictionary_search_failed")
    if dictionary.get("cleared_rows", 0) < dictionary.get("search_rows", 0):
        failures.append("dictionary_clear_failed")
    if dictionary.get("category_rows", 0) <= 0:
        failures.append("dictionary_category_failed")
    states = result.get("states") if isinstance(result.get("states"), dict) else {}
    for key in [
        "no_action_reference_button",
        "disabled_reference_buttons",
        "context_crossdb",
        "mode_real",
        "error_state_text",
        "success_state_text",
        "toast_visible",
        "toast_dismissed",
    ]:
        if states.get(key) is not True:
            failures.append(f"states_{key}_failed")
    if result.get("overflowX") != 0:
        failures.append("horizontal_overflow")
    if result.get("console_errors"):
        failures.append("console_errors")
    if result.get("page_errors"):
        failures.append("page_errors")
    provider = result.get("provider_status") if isinstance(result.get("provider_status"), dict) else {}
    if provider.get("client_constructed") or provider.get("network_calls") or provider.get("secrets_returned"):
        failures.append("provider_not_dormant")

    result["failures"] = failures
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "reference_routes_qa.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
