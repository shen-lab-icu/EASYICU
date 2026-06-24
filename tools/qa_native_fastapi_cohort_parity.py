#!/usr/bin/env python3
"""Fixture E2E for native FastAPI Cohort Review parity.

The script starts an isolated FastAPI server with a temporary HOME, registers a
small EasyICU export fixture through the public API, validates the bounded
``/api/cohort-review/summary`` payload, and opens ``#cohort`` in Chromium real
mode to verify the native UI is showing real cohort aggregates rather than demo
copy.

Run from ``EASYICU/``:

    python tools/qa_native_fastapi_cohort_parity.py
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
    parser.add_argument("--port", type=int, default=8768)
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
            stderr = proc.stderr.read() if proc.stderr else ""
            raise RuntimeError(f"server exited early with {proc.returncode}: {stderr}")
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


def choose_port(start: int) -> int:
    for port in range(start, start + 50):
        if port_free(port):
            return port
    raise SystemExit(f"No free local port found in range {start}-{start + 49}.")


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
            {"stay_id": 1, "subject_id": 101, "hadm_id": 201, "age": 50, "sex": "F"},
            {"stay_id": 2, "subject_id": 102, "hadm_id": 202, "age": 70, "sex": "M"},
            {"stay_id": 3, "subject_id": 103, "hadm_id": 203, "age": 60, "sex": "F"},
        ],
        "outcome": [
            {"stay_id": 1, "death": 0, "los_icu": 2.0},
            {"stay_id": 2, "death": 1, "los_icu": 5.0},
            {"stay_id": 3, "death": 0, "los_icu": 1.0},
        ],
        "sofa2_score": [
            {"stay_id": 1, "charttime": "2026-01-01 00:00", "sofa2": 4},
            {"stay_id": 1, "charttime": "2026-01-01 01:00", "sofa2": 5},
            {"stay_id": 2, "charttime": "2026-01-01 00:00", "sofa2": 8},
        ],
        "sepsis3_sofa2": [
            {"stay_id": 1, "sep3_sofa2": "true"},
            {"stay_id": 2, "sep3_sofa2": ""},
        ],
        "vitals": [
            {"stay_id": 1, "charttime": "2026-01-01 00:00", "hr": 90, "map": 70, "spo2": 97, "temp": 37.0},
            {"stay_id": 1, "charttime": "2026-01-01 01:00", "hr": 95, "map": 72, "spo2": 98, "temp": 37.2},
            {"stay_id": 2, "charttime": "2026-01-01 00:00", "hr": 80, "map": 75, "spo2": 96, "temp": 36.8},
        ],
    }
    manifest_files = []
    for module, rows in tables.items():
        file_name = f"{module}.csv"
        write_csv(root / file_name, rows)
        manifest_files.append({"file": file_name, "module": module, "rows": len(rows)})
    (root / "_manifest.json").write_text(
        json.dumps(
            {"database": "miiv", "generated": "2026-06-24T04:00:00", "files": manifest_files},
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


def validate_cohort_payload(payload: dict[str, Any]) -> None:
    if payload.get("mode") != "real" or payload.get("demo") is not False:
        raise AssertionError(f"unexpected cohort mode: {payload.get('mode')} demo={payload.get('demo')}")
    summary = payload.get("summary") or {}
    if summary.get("cohort_size") != 3:
        raise AssertionError(f"unexpected cohort size: {summary}")
    if summary.get("mortality_pct") != 33.3:
        raise AssertionError(f"unexpected mortality: {summary}")
    if (summary.get("age") or {}).get("median") != 60.0:
        raise AssertionError(f"unexpected age summary: {summary.get('age')}")
    if (summary.get("sofa2") or {}).get("median") != 6.5:
        raise AssertionError(f"unexpected SOFA summary: {summary.get('sofa2')}")
    if (payload.get("groups") or {}).get("inferential_statistics_allowed") is not False:
        raise AssertionError("inferential statistics were not fail-closed")
    if (payload.get("table_one") or {}).get("status") != "blocked":
        raise AssertionError("table-one inferential preview was not blocked")
    if (payload.get("sofa_reclassification") or {}).get("status") != "blocked":
        raise AssertionError("paired SOFA reclassification was not blocked")
    text = json.dumps(payload, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        if marker in text:
            raise AssertionError(f"row-level marker leaked from cohort payload: {marker}")


def page_snapshot(page: Any, panel: str) -> dict[str, Any]:
    return page.evaluate(
        """(panel) => {
          const content = document.querySelector('.content') || document.body;
          const text = content.innerText || '';
          const lower = text.toLowerCase();
          const doc = document.documentElement;
          const body = document.body;
          return {
            panel,
            hash: location.hash,
            mainTextLength: content.textContent.trim().length,
            hasRealReady: lower.includes('local export cohort review ready'),
            hasAggregate: lower.includes('cohort size') && lower.includes('mortality') && lower.includes('median sofa-2'),
            hasProvenance: lower.includes('path hash') && lower.includes('aggregate-only payload'),
            hasCoverage: lower.includes('real module coverage and quality'),
            hasSnapshot: lower.includes('aggregate ranges') && lower.includes('source provenance'),
            hasSofa: lower.includes('sofa-2 aggregate review') && lower.includes('paired reclassification blocked'),
            hasBlockedScope: lower.includes('no row-level filters') || lower.includes('fail-closed scope'),
            hasDemoCopy: /demo cohort snapshot|demo \\/ seeded|group contrast table|total patients\\s+10|generate demo/.test(lower),
            hasRawMarkers: /stay_id|subject_id|hadm_id|tableRows/.test(text),
            overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - window.innerWidth,
          };
        }""",
        panel,
    )


def run_browser(base_url: str, run_dir: Path, screenshots: bool) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 393, "height": 852})
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(base_url + "#cohort", wait_until="networkidle")
        page.wait_for_function("window.EU_API && window.EU_API.loadCohortReviewSummary", timeout=5000)
        page.evaluate("window.setDataMode && window.setDataMode('real', {force:true})")
        page.wait_for_timeout(250)
        page.locator("[data-cohort-run]").last.click()
        page.wait_for_function("document.body.innerText.includes('Local export cohort review ready')", timeout=8000)
        panels = [page_snapshot(page, "groups")]

        for panel, marker in [
            ("coverage", "Real module coverage and quality"),
            ("snapshot", "Source provenance"),
            ("sofa", "Paired reclassification blocked"),
        ]:
            page.locator(f"[data-cohtab='{panel}']").click()
            try:
                page.wait_for_function(
                    "(text) => document.body.innerText.toLowerCase().includes(text.toLowerCase())",
                    arg=marker,
                    timeout=5000,
                )
            except Exception as exc:
                text = page.evaluate("document.body.innerText.slice(0, 2000)")
                raise RuntimeError(f"panel {panel} did not render marker {marker!r}; text={text!r}") from exc
            panels.append(page_snapshot(page, panel))

        if screenshots:
            for item in panels:
                page.locator(f"[data-cohtab='{item['panel']}']").click()
                page.wait_for_timeout(100)
                shot = run_dir / f"cohort_{item['panel']}_mobile.png"
                page.screenshot(path=str(shot), full_page=True)
                item["screenshot"] = str(shot)
        browser.close()
        return {"panels": panels, "consoleErrors": errors}


def main() -> int:
    args = parse_args()
    port = choose_port(args.port)
    run_dir = Path(args.out_dir) / f"native_fastapi_cohort_parity_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    fixture = write_fixture_export(run_dir / "fixture_export")
    base_url = f"http://127.0.0.1:{port}/"
    proc = start_server(port, home)
    try:
        wait_ready(base_url, proc)
        registered = post_json(
            base_url,
            "api/workspaces/register",
            {"path": str(fixture), "label": "Cohort Parity Fixture", "active": True},
        )
        payload = post_json(base_url, "api/cohort-review/summary", {})
        validate_cohort_payload(payload)
        browser = run_browser(base_url, run_dir, not args.no_screenshots)
        report = {
            "base_url": base_url,
            "run_dir": str(run_dir),
            "registered_active_path": registered.get("active_path"),
            "api": {
                "source": payload.get("source"),
                "summary": payload.get("summary"),
                "quality": payload.get("quality"),
                "blocked_features": payload.get("blocked_features"),
            },
            "browser": browser,
        }
        report_path = run_dir / "cohort_parity_qa.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        failures: list[str] = []
        if browser.get("consoleErrors"):
            failures.append(f"console errors: {browser['consoleErrors']}")
        for panel in browser["panels"]:
            if panel.get("mainTextLength", 0) <= 0:
                failures.append(f"{panel['panel']}: main container empty")
            if panel.get("overflowX", 0) > 1:
                failures.append(f"{panel['panel']}: horizontal overflow {panel.get('overflowX')}")
            if panel.get("hasDemoCopy"):
                failures.append(f"{panel['panel']}: real Cohort Review still showed demo copy")
            if panel.get("hasRawMarkers"):
                failures.append(f"{panel['panel']}: browser text leaked row-level markers")
        first = browser["panels"][0]
        for key in ["hasRealReady", "hasAggregate", "hasProvenance", "hasBlockedScope"]:
            if not first.get(key):
                failures.append(f"groups: browser assertion failed: {key}")
        checks = {
            "coverage": "hasCoverage",
            "snapshot": "hasSnapshot",
            "sofa": "hasSofa",
        }
        for panel_name, key in checks.items():
            item = next((row for row in browser["panels"] if row["panel"] == panel_name), None)
            if not item or not item.get(key):
                failures.append(f"{panel_name}: browser assertion failed: {key}")
        print(f"Cohort parity QA report: {report_path}")
        print(json.dumps({"api": report["api"], "browser": browser}, indent=2, ensure_ascii=False))
        if failures:
            print("FAILURES:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("Cohort parity QA passed.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
