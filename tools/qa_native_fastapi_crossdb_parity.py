#!/usr/bin/env python3
"""Fixture E2E for native FastAPI Cross-DB Review parity.

The script starts an isolated FastAPI server with a temporary HOME, registers
two small EasyICU export fixtures through the public API, validates the bounded
``/api/crossdb-review/summary`` payload, and opens ``#crossdb`` in Chromium real
mode to verify the native UI is showing real registered-source aggregates.

Run from ``EASYICU/``:

    python tools/qa_native_fastapi_crossdb_parity.py
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
    parser.add_argument("--port", type=int, default=8769)
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


def write_fixture_export(root: Path, database: str, variant: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if variant == "miiv":
        demographics = [
            {"stay_id": 1, "subject_id": 101, "hadm_id": 201, "age": 50, "sex": "F"},
            {"stay_id": 2, "subject_id": 102, "hadm_id": 202, "age": 70, "sex": "M"},
            {"stay_id": 3, "subject_id": 103, "hadm_id": 203, "age": 60, "sex": "F"},
        ]
        outcome = [
            {"stay_id": 1, "death": 0, "los_icu": 2.0},
            {"stay_id": 2, "death": 1, "los_icu": 5.0},
            {"stay_id": 3, "death": 0, "los_icu": 1.0},
        ]
        sofa = [
            {"stay_id": 1, "charttime": "2026-01-01 00:00", "sofa2": 4},
            {"stay_id": 1, "charttime": "2026-01-01 01:00", "sofa2": 5},
            {"stay_id": 2, "charttime": "2026-01-01 00:00", "sofa2": 8},
        ]
        sepsis = [
            {"stay_id": 1, "sep3_sofa2": "true"},
            {"stay_id": 2, "sep3_sofa2": ""},
        ]
        vitals = [
            {"stay_id": 1, "charttime": "2026-01-01 00:00", "hr": 90, "map": 70, "spo2": 97, "temp": 37.0},
            {"stay_id": 1, "charttime": "2026-01-01 01:00", "hr": 95, "map": 72, "spo2": 98, "temp": 37.2},
            {"stay_id": 2, "charttime": "2026-01-01 00:00", "hr": 80, "map": 75, "spo2": 96, "temp": 36.8},
        ]
    else:
        demographics = [
            {"stay_id": 10, "subject_id": 110, "hadm_id": 210, "age": 55, "sex": "F"},
            {"stay_id": 11, "subject_id": 111, "hadm_id": 211, "age": 65, "sex": "F"},
            {"stay_id": 12, "subject_id": 112, "hadm_id": 212, "age": 75, "sex": "M"},
            {"stay_id": 13, "subject_id": 113, "hadm_id": 213, "age": 80, "sex": "M"},
        ]
        outcome = [
            {"stay_id": 10, "death": 0, "los_icu": 2.5},
            {"stay_id": 11, "death": 0, "los_icu": 4.0},
            {"stay_id": 12, "death": 1, "los_icu": 6.0},
            {"stay_id": 13, "death": 1, "los_icu": 7.5},
        ]
        sofa = [
            {"stay_id": 10, "charttime": "2026-01-01 00:00", "sofa2": 6},
            {"stay_id": 11, "charttime": "2026-01-01 00:00", "sofa2": 7},
            {"stay_id": 12, "charttime": "2026-01-01 00:00", "sofa2": 9},
            {"stay_id": 13, "charttime": "2026-01-01 00:00", "sofa2": 10},
        ]
        sepsis = [
            {"stay_id": 10, "sep3_sofa2": ""},
            {"stay_id": 11, "sep3_sofa2": "true"},
            {"stay_id": 12, "sep3_sofa2": "true"},
        ]
        vitals = [
            {"stay_id": 10, "charttime": "2026-01-01 00:00", "hr": 88, "map": 78, "spo2": 97, "temp": 36.9},
            {"stay_id": 11, "charttime": "2026-01-01 00:00", "hr": 91, "map": 80, "spo2": 96, "temp": 37.3},
            {"stay_id": 12, "charttime": "2026-01-01 00:00", "hr": 99, "map": 74, "spo2": 94, "temp": 37.8},
        ]

    tables = {
        "demographics": demographics,
        "outcome": outcome,
        "sofa2_score": sofa,
        "sepsis3_sofa2": sepsis,
        "vitals": vitals,
    }
    manifest_files = []
    for module, rows in tables.items():
        file_name = f"{module}.csv"
        write_csv(root / file_name, rows)
        manifest_files.append({"file": file_name, "module": module, "rows": len(rows)})
    (root / "_manifest.json").write_text(
        json.dumps(
            {"database": database, "generated": "2026-06-24T06:00:00", "files": manifest_files},
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


def validate_crossdb_payload(payload: dict[str, Any]) -> None:
    if payload.get("mode") != "real" or payload.get("demo") is not False:
        raise AssertionError(f"unexpected Cross-DB mode: {payload.get('mode')} demo={payload.get('demo')}")
    if payload.get("source_count") != 2:
        raise AssertionError(f"unexpected source count: {payload.get('source_count')}")
    if payload.get("shared_modules") != ["demographics", "outcome", "sepsis3_sofa2", "sofa2_score", "vitals"]:
        raise AssertionError(f"unexpected shared modules: {payload.get('shared_modules')}")
    gate = payload.get("compatibility_gate") or {}
    if gate.get("status") != "compatible":
        raise AssertionError(f"unexpected compatibility gate: {gate}")
    if gate.get("matched_cohort") is not False or gate.get("inferential_statistics_allowed") is not False:
        raise AssertionError(f"unsupported Cross-DB claims were not fail-closed: {gate}")
    rows = {row.get("key"): row for row in payload.get("rows") or []}
    if (rows.get("cohort_size") or {}).get("values") != [3, 4]:
        raise AssertionError(f"unexpected cohort sizes: {rows.get('cohort_size')}")
    if (rows.get("mortality_pct") or {}).get("values") != [33.3, 50.0]:
        raise AssertionError(f"unexpected mortality values: {rows.get('mortality_pct')}")
    availability = {row.get("module"): row for row in payload.get("availability") or []}
    if not availability.get("demographics", {}).get("shared"):
        raise AssertionError(f"demographics not shared in availability: {availability.get('demographics')}")
    density = {row.get("module"): row for row in payload.get("feature_density") or []}
    if not density.get("vitals"):
        raise AssertionError(f"vitals feature density missing: {payload.get('feature_density')}")
    vitals_features = {row.get("feature"): row for row in density["vitals"].get("features") or []}
    for feature in ["hr", "map", "spo2", "temp"]:
        if feature not in vitals_features:
            raise AssertionError(f"vitals feature density did not include {feature}: {vitals_features}")
        values = vitals_features[feature].get("values") or []
        if not values or not all(value.get("present") for value in values):
            raise AssertionError(f"vitals density values not present for {feature}: {values}")
    distributions = {row.get("module"): row for row in payload.get("feature_distributions") or []}
    if not distributions.get("vitals"):
        raise AssertionError(f"vitals feature distributions missing: {payload.get('feature_distributions')}")
    hr_dist = next((row for row in distributions["vitals"].get("features") or [] if row.get("feature") == "hr"), None)
    if not hr_dist:
        raise AssertionError(f"hr value distribution missing from vitals: {distributions['vitals']}")
    if not all((value.get("points") or []) for value in hr_dist.get("values") or []):
        raise AssertionError(f"hr value distribution did not include density points: {hr_dist}")
    text = json.dumps(payload, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        if marker in text:
            raise AssertionError(f"row-level marker leaked from Cross-DB payload: {marker}")


def page_snapshot(page: Any) -> dict[str, Any]:
    return page.evaluate(
        """() => {
          const content = document.querySelector('.content') || document.body;
          const text = content.innerText || '';
          const lower = text.toLowerCase();
          const doc = document.documentElement;
          const body = document.body;
          return {
            hash: location.hash,
            mainTextLength: content.textContent.trim().length,
            hasRealReady: lower.includes('real cross-database benchmark ready'),
            hasRegisteredComparison: lower.includes('registered export comparison'),
            hasProvenance: lower.includes('source provenance') && lower.includes('path hash'),
            hasAggregate: lower.includes('cohort size') && lower.includes('mortality') && lower.includes('median sofa-2'),
            hasDistribution: (lower.includes('multi-database feature density grid') ||
              lower.includes('value density distribution by module and feature')) &&
              document.querySelectorAll('.xdb-density-panel').length >= 1 &&
              document.querySelectorAll('.xdb-density-module').length >= 1 &&
              document.querySelectorAll('.xdb-density-feature').length >= 1 &&
              document.querySelectorAll('.xdb-density-svg').length >= 1 &&
              document.querySelectorAll('.xdb-density-line').length >= 1,
            hasAvailability: lower.includes('module availability matrix') && lower.includes('demographics'),
            hasGate: lower.includes('compatibility gate') && lower.includes('inferential_statistics=false'),
            hasBlockedScope: lower.includes('fail-closed scope') && lower.includes('matched_cohort'),
            hasDemoCopy: /demo simulated data|seeded feature frames|144 rows|demo cohort/.test(lower),
            hasRawMarkers: /stay_id|subject_id|hadm_id|tableRows/.test(text),
            overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - window.innerWidth,
          };
        }"""
    )


def run_browser(base_url: str, run_dir: Path, screenshots: bool) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 393, "height": 852})
        page = context.new_page()
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(base_url + "#crossdb", wait_until="domcontentloaded")
        page.wait_for_function("window.EU_API && window.EU_API.loadCrossdbReviewSummary", timeout=5000)
        page.evaluate("window.setDataMode && window.setDataMode('real', {force:true})")
        page.wait_for_timeout(250)
        page.locator("[data-run]").last.click()
        page.wait_for_function(
            """() => document.querySelector('.xdb-density-panel') ||
              document.body.innerText.includes('Registered export comparison') ||
              document.body.innerText.includes('Module availability matrix') ||
              document.body.innerText.includes('Real cross-database benchmark ready')""",
            timeout=12000,
        )
        module_filters = page.locator("[data-density-module-filter]").count()
        if module_filters > 1:
            page.locator("[data-density-module-filter]").nth(1).click()
            page.wait_for_timeout(100)
        feature_cards = page.locator("[data-density-feature-key]").count()
        if feature_cards > 0:
            page.locator("[data-density-feature-key]").first.click()
            page.wait_for_timeout(100)
        snapshot = page_snapshot(page)
        snapshot["densityModuleFilterCount"] = module_filters
        snapshot["densityFeatureCount"] = feature_cards
        snapshot["densityDetailVisible"] = page.locator(".xdb-density-detail").count()
        with page.expect_download(timeout=8000) as export_info:
            page.locator(".loaded-bar [data-crossdb-export]").click()
        export_download = export_info.value
        export_path = run_dir / export_download.suggested_filename
        export_download.save_as(str(export_path))
        exported = json.loads(export_path.read_text(encoding="utf-8"))
        export_text = json.dumps(exported, ensure_ascii=False)
        snapshot["exportDownloadOk"] = (
            exported.get("payload_scope") == "bounded_crossdb_review"
            and isinstance(exported.get("crossdb_review"), dict)
            and not any(marker in export_text for marker in ["stay_id", "subject_id", "hadm_id", "tableRows"])
        )
        snapshot["exportDownloadPath"] = str(export_path)
        snapshot["exportDownloadFilename"] = export_download.suggested_filename
        page.locator(".loaded-bar [data-viz-reset]").click()
        page.wait_for_function("document.body.innerText.includes('Real raw database mode')", timeout=5000)
        snapshot["changeSelectionOk"] = page.evaluate(
            "() => document.body.innerText.includes('Raw ICU data root') && !!document.querySelector('[data-run]')"
        )
        page.locator("[data-run]").last.click()
        page.wait_for_function(
            """() => document.querySelector('.xdb-density-panel') &&
              document.body.innerText.includes('Real cross-database benchmark ready')""",
            timeout=12000,
        )
        snapshot["rerunOk"] = page.evaluate(
            "() => document.body.innerText.includes('Real cross-database benchmark ready') && document.querySelectorAll('.xdb-density-panel').length >= 1"
        )
        if screenshots:
            shot = run_dir / "crossdb_real_mobile.png"
            page.screenshot(path=str(shot), full_page=True)
            snapshot["screenshot"] = str(shot)
        context.close()
        browser.close()
        return {"page": snapshot, "consoleErrors": errors}


def main() -> int:
    args = parse_args()
    port = choose_port(args.port)
    run_dir = Path(args.out_dir) / f"native_fastapi_crossdb_parity_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    miiv = write_fixture_export(run_dir / "fixture_miiv", "miiv", "miiv")
    eicu = write_fixture_export(run_dir / "fixture_eicu", "eicu", "eicu")
    base_url = f"http://127.0.0.1:{port}/"
    proc = start_server(port, home)
    try:
        wait_ready(base_url, proc)
        first = post_json(
            base_url,
            "api/workspaces/register",
            {"path": str(miiv), "label": "CrossDB MIIV Fixture", "active": True, "crossdb": True},
        )
        second = post_json(
            base_url,
            "api/workspaces/register",
            {"path": str(eicu), "label": "CrossDB eICU Fixture", "active": False, "crossdb": True},
        )
        payload = post_json(base_url, "api/crossdb-review/summary", {})
        validate_crossdb_payload(payload)
        browser = run_browser(base_url, run_dir, not args.no_screenshots)
        report = {
            "base_url": base_url,
            "run_dir": str(run_dir),
            "registered_crossdb_paths": second.get("crossdb_paths") or first.get("crossdb_paths"),
            "api": {
                "sources": payload.get("sources"),
                "rows": payload.get("rows"),
                "availability": payload.get("availability"),
                "feature_density": payload.get("feature_density"),
                "feature_distributions": payload.get("feature_distributions"),
                "compatibility_gate": payload.get("compatibility_gate"),
                "privacy": payload.get("privacy"),
                "blocked_features": payload.get("blocked_features"),
            },
            "browser": browser,
        }
        report_path = run_dir / "crossdb_parity_qa.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        failures: list[str] = []
        page = browser["page"]
        if browser.get("consoleErrors"):
            failures.append(f"console errors: {browser['consoleErrors']}")
        if page.get("mainTextLength", 0) <= 0:
            failures.append("main container empty")
        if page.get("overflowX", 0) > 1:
            failures.append(f"horizontal overflow {page.get('overflowX')}")
        if page.get("hasDemoCopy"):
            failures.append("real Cross-DB still showed demo copy")
        if page.get("hasRawMarkers"):
            failures.append("browser text leaked row-level markers")
        for key in [
            "hasRealReady",
            "hasRegisteredComparison",
            "hasProvenance",
            "hasAggregate",
            "hasDistribution",
            "hasAvailability",
            "hasGate",
            "hasBlockedScope",
        ]:
            if not page.get(key):
                failures.append(f"browser assertion failed: {key}")
        if page.get("densityModuleFilterCount", 0) < 2:
            failures.append(f"density module filters missing: {page.get('densityModuleFilterCount')}")
        if page.get("densityFeatureCount", 0) < 1:
            failures.append(f"density feature cards missing: {page.get('densityFeatureCount')}")
        if page.get("densityDetailVisible", 0) < 1:
            failures.append("density feature detail did not open")
        for key in ["exportDownloadOk", "changeSelectionOk", "rerunOk"]:
            if not page.get(key):
                failures.append(f"browser action assertion failed: {key}")
        print(f"Cross-DB parity QA report: {report_path}")
        print(json.dumps({"api": report["api"], "browser": browser}, indent=2, ensure_ascii=False))
        if failures:
            print("FAILURES:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("Cross-DB parity QA passed.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
