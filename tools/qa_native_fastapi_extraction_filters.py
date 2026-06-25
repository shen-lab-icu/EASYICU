#!/usr/bin/env python3
"""Fixture E2E for native FastAPI extraction advanced filters.

The script starts an isolated FastAPI server with a temporary HOME, registers a
small EasyICU export fixture through the public API, opens the native extraction
page in Chromium, and verifies that the advanced filter card is backed by the
real `/api/extraction/*` endpoints.

Run from `EASYICU/`:

    python tools/qa_native_fastapi_extraction_filters.py
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--out-dir", default="output/playwright")
    parser.add_argument("--no-screenshots", action="store_true")
    return parser.parse_args()


def post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(base_url: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(base_url + path, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_ready(base_url: str, proc: subprocess.Popen[str], timeout: float = 20.0) -> None:
    deadline = time.time() + timeout
    last_error: str | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited early with {proc.returncode}: {proc.stderr.read() if proc.stderr else ''}")
        try:
            if get_json(base_url, "api/health").get("status") == "ok":
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"server did not become ready at {base_url}: {last_error}")


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) != 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_fixture_export(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    tables = {
        "demographics": [
            {"stay_id": 1, "age": 50, "sex": "F"},
            {"stay_id": 2, "age": 70, "sex": "M"},
            {"stay_id": 3, "age": 60, "sex": "F"},
        ],
        "outcome": [
            {"stay_id": 1, "death": 0, "los_icu": 2.0},
            {"stay_id": 2, "death": 1, "los_icu": 5.0},
            {"stay_id": 3, "death": 0, "los_icu": 1.0},
        ],
        "sofa2_score": [
            {"stay_id": 1, "sofa2": 4},
            {"stay_id": 1, "sofa2": 5},
            {"stay_id": 2, "sofa2": 8},
        ],
        "vitals": [
            {"stay_id": 1, "hr": 90, "map": 70},
            {"stay_id": 1, "hr": 95, "map": 72},
            {"stay_id": 2, "hr": 80, "map": 75},
        ],
    }
    manifest_files = []
    for module, rows in tables.items():
        file_name = f"{module}.csv"
        write_csv(root / file_name, rows)
        manifest_files.append({"file": file_name, "module": module, "rows": len(rows)})
    (root / "_manifest.json").write_text(
        json.dumps(
            {"database": "miiv", "generated": "2026-06-24T02:00:00", "files": manifest_files},
            indent=2,
        ),
        encoding="utf-8",
    )
    return root


def start_server(port: int, home: Path) -> subprocess.Popen[str]:
    python_bin = Path(".venv/bin/python")
    executable = str(python_bin) if python_bin.exists() else sys.executable
    env = os.environ.copy()
    env["HOME"] = str(home)
    env["EASYICU_DISABLE_PROVIDER_ENV_FILE"] = "1"
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.Popen(
        [
            executable,
            "-m",
            "uvicorn",
            "easyicu.webserver.app:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=str(Path.cwd()),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_browser(base_url: str, fixture: Path, run_dir: Path, screenshots: bool) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 860})
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(base_url + "#extraction", wait_until="domcontentloaded")
        page.evaluate("window.setDataMode && window.setDataMode('real', {force:true})")
        page.wait_for_timeout(250)
        page.locator("#exPathInput").fill(str(fixture))
        page.locator("[data-ex-manual]").click()
        page.wait_for_selector("[data-ex-manual-body]:not([hidden])", timeout=5000)
        page.locator("[data-ex-src='module']").click()
        page.wait_for_function("document.body.innerText.includes('Folder recognized')", timeout=5000)
        page.locator("[data-ex-usedata]").click()
        page.wait_for_selector("[data-ex-advc]", timeout=5000)
        page.locator("[data-ex-advc]").click()
        page.wait_for_function("document.body.innerText.includes('Real filter provenance')", timeout=5000)
        page.locator("[data-ex-filter-quality] button[data-val='warn']").click()
        page.wait_for_function("document.body.innerText.includes('sofa2_score') && document.body.innerText.includes('vitals')", timeout=5000)
        result = page.evaluate(
            """() => {
              const text = document.body.innerText;
              const doc = document.documentElement;
              const body = document.body;
              return {
                hasRealProvenance: text.includes('Real filter provenance'),
                hasUnsupported: text.includes('Unsupported filters stay blocked'),
                hasSourceLabel: text.includes('MIIV'),
                hasWarnModules: text.includes('sofa2_score') && text.includes('vitals'),
                hasSeededDemo: text.includes('Seeded demo filters'),
                overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - window.innerWidth,
              };
            }"""
        )
        result["consoleErrors"] = errors
        if screenshots:
            shot = run_dir / "extraction_advanced_filters_fixture.png"
            page.screenshot(path=str(shot), full_page=True)
            result["screenshot"] = str(shot)
        browser.close()
        return result


def validate_api_payload(payload: dict[str, Any]) -> None:
    text = json.dumps(payload, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        if marker in text:
            raise AssertionError(f"row-level marker leaked from API payload: {marker}")


def main() -> int:
    args = parse_args()
    if not port_free(args.port):
        raise SystemExit(f"Port {args.port} is already in use; pass --port with a free local port.")

    run_dir = Path(args.out_dir) / f"native_fastapi_extraction_filters_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    fixture = write_fixture_export(run_dir / "fixture_export")
    base_url = f"http://127.0.0.1:{args.port}/"
    proc = start_server(args.port, home)
    try:
        wait_ready(base_url, proc)
        registered = post_json(base_url, "api/workspaces/register", {"path": str(fixture), "label": "Fixture MIIV", "active": True})
        options = post_json(base_url, "api/extraction/filter-options", {})
        preview = post_json(base_url, "api/extraction/filter-preview", {"filters": {"quality_statuses": ["warn"], "min_coverage_pct": 50}})
        validate_api_payload(options)
        validate_api_payload(preview)
        browser = run_browser(base_url, fixture, run_dir, not args.no_screenshots)
        report = {
            "base_url": base_url,
            "run_dir": str(run_dir),
            "registered_active_path": registered.get("active_path"),
            "options_source": options.get("source"),
            "preview_match_count": preview.get("match_count"),
            "preview_modules": [row.get("module") for row in preview.get("matched_modules", [])],
            "browser": browser,
        }
        report_path = run_dir / "extraction_filter_qa.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        failures = []
        if preview.get("match_count") != 2:
            failures.append(f"expected 2 warn modules, got {preview.get('match_count')}")
        if set(report["preview_modules"]) != {"sofa2_score", "vitals"}:
            failures.append(f"unexpected preview modules: {report['preview_modules']}")
        if browser.get("consoleErrors"):
            failures.append(f"console errors: {browser['consoleErrors']}")
        if browser.get("overflowX", 0) > 1:
            failures.append(f"horizontal overflow: {browser.get('overflowX')}")
        for key in ["hasRealProvenance", "hasUnsupported", "hasSourceLabel", "hasWarnModules"]:
            if not browser.get(key):
                failures.append(f"browser assertion failed: {key}")
        if browser.get("hasSeededDemo"):
            failures.append("real mode still showed seeded-demo filter copy")
        print(f"Extraction filter QA report: {report_path}")
        print(json.dumps({
            "preview_modules": report["preview_modules"],
            "browser": browser,
        }, indent=2, ensure_ascii=False))
        if failures:
            print("FAILURES:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("Extraction filter QA passed.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


if __name__ == "__main__":
    raise SystemExit(main())
