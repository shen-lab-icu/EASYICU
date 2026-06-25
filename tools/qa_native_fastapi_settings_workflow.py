#!/usr/bin/env python3
"""Browser QA for native FastAPI Settings controls.

The script starts an isolated FastAPI server unless ``--base-url`` is passed,
drives the Settings screen, and verifies that visible controls persist through
the backend settings store. It uses a temporary HOME by default so the user's
real ``~/.easyicu`` settings are not modified.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.sync_api import Page, sync_playwright

from qa_native_fastapi_extraction_job_flow import DEFAULT_OUT_ROOT, server_context


def _json_request(url: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    if payload is None:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _wait_setting(base_url: str, key: str, expected: Any, timeout: float = 10.0) -> Any:
    deadline = time.time() + timeout
    latest: Any = None
    while time.time() < deadline:
        settings = _json_request(f"{base_url}/api/settings")
        latest = settings.get(key)
        if latest == expected:
            return latest
        time.sleep(0.2)
    raise AssertionError(f"setting {key!r} did not become {expected!r}; latest={latest!r}")


def _goto_hash(page: Page, route: str) -> None:
    page.evaluate("(r) => { window.location.hash = '#' + r; }", route)
    page.wait_for_function("(r) => window.location.hash === '#' + r", arg=route, timeout=5000)
    page.wait_for_timeout(250)


def _click_setting_button(page: Page, key: str, value: str) -> None:
    page.locator(f".seg[data-setting='{key}'] button[data-val='{value}']").first.click()


def _run_browser_flow(*, base_url: str, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _json_request(
        f"{base_url}/api/settings",
        {
            "working_dir": "/tmp",
            "export_dir": "/tmp",
            "data_mode": "demo",
            "density": "comfortable",
            "evidence_gate": "strict",
            "token_budget": 120000,
            "telemetry_enabled": False,
            "reduce_motion": False,
            "auto_repair": True,
        },
    )

    console_errors: list[str] = []
    page_errors: list[str] = []
    result: dict[str, Any] = {"base_url": base_url}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page = context.new_page()
        page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: page_errors.append(str(exc)))

        page.goto(f"{base_url}/?_v=stage58-settings#settings", wait_until="domcontentloaded")
        page.locator(".settings-page").first.wait_for(state="visible", timeout=10000)

        page.locator('[data-settings-jump="set-privacy"]').first.click()
        page.wait_for_function("() => window.location.hash === '#settings'", timeout=5000)
        page.wait_for_function(
            "() => document.getElementById('set-privacy') && Math.abs(document.getElementById('set-privacy').getBoundingClientRect().top) < 220",
            timeout=5000,
        )
        result["rail_jump_privacy_keeps_route"] = page.evaluate("window.location.hash") == "#settings"

        _click_setting_button(page, "data_mode", "real")
        result["data_mode_real"] = _wait_setting(base_url, "data_mode", "real")
        _click_setting_button(page, "density", "compact")
        result["density_compact"] = _wait_setting(base_url, "density", "compact")
        _click_setting_button(page, "language", "zh")
        result["language_zh"] = _wait_setting(base_url, "language", "zh")
        body_text = page.locator("body").inner_text(timeout=5000)
        result["evidence_gate_strict_enforced"] = "strict enforced" in body_text or "Strict evidence" in body_text

        page.locator('[data-setting="ai_enabled"]').first.click()
        result["ai_enabled_true"] = _wait_setting(base_url, "ai_enabled", True)
        page.locator('[data-setting="reduce_motion"]').first.click()
        result["reduce_motion_true"] = _wait_setting(base_url, "reduce_motion", True)

        result["working_dir_is_not_global_picker"] = page.locator('[data-setting-path="working_dir"]').count() == 0

        page.locator('[data-setting-path="export_dir"]').first.click()
        page.locator(".eu-pick").first.wait_for(state="visible", timeout=10000)
        page.locator("[data-pk-use]").first.click()
        export_dir = _json_request(f"{base_url}/api/settings").get("export_dir")
        result["export_dir_picked"] = export_dir in {"/tmp", "/private/tmp"}

        with page.expect_download(timeout=10000) as download_info:
            page.locator("[data-settings-diagnostics]").first.click()
        download = download_info.value
        download_path = out_dir / download.suggested_filename
        download.save_as(download_path)
        diagnostics = json.loads(download_path.read_text(encoding="utf-8"))
        diagnostics_raw = json.dumps(diagnostics, ensure_ascii=False)
        result["diagnostics_downloaded"] = download.suggested_filename == "easyicu_settings_diagnostics.json"
        result["diagnostics_scope"] = diagnostics.get("scope")
        result["diagnostics_no_secret_markers"] = all(
            marker not in diagnostics_raw
            for marker in ["OPENAI_API_KEY", "EASYICU_LLM_API_KEY", "sk-", "api_key"]
        )

        page.locator('[data-settings-doc="release"]').first.click()
        page.wait_for_function("() => document.body.innerText.includes('Release notes are tracked')", timeout=5000)
        result["release_notes_notice"] = True
        page.locator('[data-settings-doc="docs"]').first.click()
        page.wait_for_function(
            "() => window.location.hash === '#help' || window.location.hash === '#tutorial'",
            timeout=5000,
        )
        result["docs_button_navigates_help"] = page.evaluate("window.location.hash") in {"#help", "#tutorial"}
        _goto_hash(page, "settings")
        page.locator(".settings-page").first.wait_for(state="visible", timeout=10000)

        page.locator("button[data-settings-reset]:visible").first.click(timeout=10000)
        page.wait_for_function(
            "() => document.body.innerText.includes('Settings reset to backend defaults')",
            timeout=5000,
        )
        reset_settings = _json_request(f"{base_url}/api/settings")
        result["reset_defaults"] = {
            "data_mode": reset_settings.get("data_mode"),
            "density": reset_settings.get("density"),
            "evidence_gate": reset_settings.get("evidence_gate"),
            "token_budget": reset_settings.get("token_budget"),
            "telemetry_enabled": reset_settings.get("telemetry_enabled"),
            "reduce_motion": reset_settings.get("reduce_motion"),
            "auto_repair": reset_settings.get("auto_repair"),
            "ai_enabled": reset_settings.get("ai_enabled"),
        }

        result["overflowX"] = page.evaluate(
            "() => Math.max(0, document.documentElement.scrollWidth - document.documentElement.clientWidth)"
        )
        page.screenshot(path=str(out_dir / "settings_workflow.png"), full_page=True)
        result["screenshot"] = str(out_dir / "settings_workflow.png")
        result["console_errors"] = console_errors
        result["page_errors"] = page_errors
        browser.close()

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Existing FastAPI server, for example http://127.0.0.1:8782")
    parser.add_argument("--port", type=int, default=8794)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_root / f"native_fastapi_settings_workflow_{stamp}"

    with server_context(args.base_url, args.port) as server:
        result = _run_browser_flow(base_url=server.base_url, out_dir=out_dir)

    failures: list[str] = []
    expected_true = [
        "working_dir_is_not_global_picker",
        "export_dir_picked",
        "evidence_gate_strict_enforced",
        "diagnostics_downloaded",
        "diagnostics_no_secret_markers",
        "release_notes_notice",
        "docs_button_navigates_help",
    ]
    failures.extend(key for key in expected_true if result.get(key) is not True)
    if result.get("data_mode_real") != "real":
        failures.append("data_mode_not_saved")
    if result.get("density_compact") != "compact":
        failures.append("density_not_saved")
    if result.get("language_zh") != "zh":
        failures.append("language_not_saved")
    if result.get("ai_enabled_true") is not True:
        failures.append("ai_enabled_not_saved")
    if result.get("reduce_motion_true") is not True:
        failures.append("reduce_motion_not_saved")
    reset = result.get("reset_defaults") if isinstance(result.get("reset_defaults"), dict) else {}
    if reset.get("data_mode") != "demo" or reset.get("density") != "comfortable":
        failures.append("reset_core_defaults_failed")
    if reset.get("ai_enabled") is not False:
        failures.append("reset_ai_enabled_not_false")
    if result.get("overflowX") != 0:
        failures.append("horizontal_overflow")
    if result.get("console_errors"):
        failures.append("console_errors")
    if result.get("page_errors"):
        failures.append("page_errors")

    result["failures"] = failures
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "settings_workflow_qa.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
