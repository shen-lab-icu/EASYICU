#!/usr/bin/env python3
"""Fixture E2E for native FastAPI Patient Review drilldown.

The script starts an isolated FastAPI server with a temporary HOME, registers a
small EasyICU export fixture through the public API, validates the bounded
``/api/patient-review/drilldown`` payload, and opens ``#patient`` in Chromium
real mode to verify the native UI is not showing demo copy.

Run from ``EASYICU/``:

    python tools/qa_native_fastapi_patient_drilldown.py
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
    parser.add_argument("--port", type=int, default=8767)
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
            {"database": "miiv", "generated": "2026-06-24T03:00:00", "files": manifest_files},
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


def validate_drilldown_payload(payload: dict[str, Any]) -> None:
    if payload.get("mode") != "real" or payload.get("demo") is not False:
        raise AssertionError(f"unexpected drilldown mode: {payload.get('mode')} demo={payload.get('demo')}")
    if (payload.get("summary") or {}).get("entities") != 3:
        raise AssertionError(f"unexpected entity count: {payload.get('summary')}")
    selected = payload.get("selected") or {}
    if selected.get("label") != "Entity 1":
        raise AssertionError(f"unexpected selected entity: {selected}")
    if not (selected.get("signals") or []):
        raise AssertionError("selected entity did not return bounded signals")
    modules = {row.get("module"): row for row in payload.get("module_profiles") or []}
    if (modules.get("vitals") or {}).get("dynamic_features") != 4:
        raise AssertionError(f"vitals module profile is not backed by dynamic features: {modules.get('vitals')}")
    lanes = {row.get("lane"): row for row in payload.get("time_lanes") or []}
    if (lanes.get("vitals") or {}).get("status") != "ready":
        raise AssertionError(f"vitals clinical lane is not ready: {lanes.get('vitals')}")
    other_signals = (lanes.get("other") or {}).get("signals") or []
    if any(signal.get("feature") == "age" for signal in other_signals):
        raise AssertionError("static demographic age leaked into time-series lanes")
    quality_metrics = payload.get("quality_metrics") or {}
    if quality_metrics.get("payload_scope") != "aggregate_quality_metrics_no_row_payload":
        raise AssertionError(f"unexpected quality metrics scope: {quality_metrics.get('payload_scope')}")
    if ((quality_metrics.get("summary") or {}).get("concept_count") or 0) < 8:
        raise AssertionError(f"quality metrics did not cover expected concepts: {quality_metrics.get('summary')}")
    data_tables = payload.get("data_tables") or {}
    if data_tables.get("payload_scope") != "old_data_tables_semantics_without_row_payload":
        raise AssertionError(f"old Data Tables semantics missing: {data_tables}")
    if (data_tables.get("detail_gate") or {}).get("title") != "Source records are optional":
        raise AssertionError(f"Data Tables detail gate missing: {data_tables.get('detail_gate')}")
    trajectory = payload.get("trajectory_review") or {}
    if trajectory.get("payload_scope") != "old_timeseries_semantics_bounded":
        raise AssertionError(f"old Time Series semantics missing: {trajectory}")
    mode_ids = {row.get("id") for row in trajectory.get("modes") or []}
    if mode_ids != {"clinical_lanes", "single_entity", "multi_entity_comparison"}:
        raise AssertionError(f"unexpected trajectory modes: {trajectory.get('modes')}")
    overview = payload.get("patient_overview") or {}
    if overview.get("payload_scope") != "old_patient_overview_semantics_pseudonymous":
        raise AssertionError(f"old Patient Overview semantics missing: {overview}")
    if (overview.get("data_table") or {}).get("row_preview") != "blocked":
        raise AssertionError(f"Patient Overview row preview should stay blocked: {overview.get('data_table')}")
    quality_review = payload.get("quality_review") or {}
    if quality_review.get("payload_scope") != "old_quality_semantics_aggregate_only":
        raise AssertionError(f"old Quality semantics missing: {quality_review}")
    text = json.dumps(payload, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        if marker in text:
            raise AssertionError(f"row-level marker leaked from drilldown payload: {marker}")


def validate_sources_payload(payload: dict[str, Any]) -> None:
    if payload.get("mode") != "real" or payload.get("demo") is not False:
        raise AssertionError(f"unexpected sources mode: {payload.get('mode')} demo={payload.get('demo')}")
    if payload.get("source_count") != 1 or payload.get("can_load") is not True:
        raise AssertionError(f"source readiness did not pass: {payload}")
    active = payload.get("active_source") or {}
    if active.get("label") != "Patient Drilldown Fixture" or active.get("patient_ready") is not True:
        raise AssertionError(f"unexpected active source: {active}")
    summary = active.get("summary") or {}
    if summary.get("entities") != 3 or summary.get("modules") != 5:
        raise AssertionError(f"unexpected source summary: {summary}")
    text = json.dumps(payload, ensure_ascii=False)
    for marker in ["stay_id", "subject_id", "hadm_id", "tableRows", '"series"']:
        if marker in text:
            raise AssertionError(f"row-level marker leaked from source readiness payload: {marker}")


def run_browser(base_url: str, run_dir: Path, screenshots: bool) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 393, "height": 852})
        page = context.new_page()
        errors: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.goto(base_url + "#patient", wait_until="domcontentloaded")
        page.wait_for_function(
            "window.EU_API && window.EU_API.loadPatientReviewSources && window.EU_API.loadPatientReviewDrilldown",
            timeout=5000,
        )
        page.evaluate("window.setDataMode && window.setDataMode('real', {force:true})")
        page.wait_for_function("document.body.innerText.includes('Ready to load local export')", timeout=8000)
        source_ready = page.evaluate(
            """() => {
              const text = document.body.innerText;
              return {
                hasSourceReady: text.includes('Ready to load local export'),
                hasSourceHash: text.includes('path_hash='),
              };
            }"""
        )
        page.locator("button[data-gen]").last.click()
        page.wait_for_function("document.body.innerText.includes('Local export patient drilldown ready')", timeout=8000)
        tab_results: dict[str, bool] = {}
        tab_expectations = {
            "tables": "Source records are optional",
            "series": "Trajectory ledger",
            "patient": "Category View",
            "quality": "Quality dashboard",
        }
        tab_texts: dict[str, str] = {}
        for tab, marker in tab_expectations.items():
            page.locator(f"[data-ptab='{tab}']").first.click()
            page.wait_for_timeout(150)
            text = page.locator("#ptbody").inner_text(timeout=5000)
            tab_texts[tab] = text[:600]
            tab_results[tab] = marker.lower() in text.lower()
        page.locator("[data-ptab='patient']").first.click()
        page.wait_for_function("document.body.innerText.includes('Pseudonymous drilldown')", timeout=5000)
        entity_two = page.locator("[data-patient-entity]").nth(1)
        if entity_two.count():
            entity_two.click()
            page.wait_for_function("document.body.innerText.includes('Entity 2')", timeout=8000)
        result = page.evaluate(
            """() => {
              const text = document.body.innerText;
              const doc = document.documentElement;
              const body = document.body;
              return {
                hash: location.hash,
                mainTextLength: (document.querySelector('.content') || body).textContent.trim().length,
                hasRealReady: text.includes('Local export patient drilldown ready'),
                hasPseudonymous: text.includes('Pseudonymous drilldown'),
                hasEntity2: text.includes('Entity 2'),
                hasExportButton: !!document.querySelector('[data-patient-export]'),
                axisChartCount: document.querySelectorAll('[data-axis-chart="true"]').length,
                axisLabelCount: document.querySelectorAll('[data-axis-label]').length,
                hasDemoCopy: /Demo review workspace ready|seeded example|Generate a lightweight demo|Generate and load demo workspace/.test(text),
                hasRawMarkers: /stay_id|subject_id|hadm_id|tableRows/.test(text),
                overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - window.innerWidth,
              };
            }"""
        )

        with page.expect_download(timeout=5000) as download_info:
            page.locator("[data-patient-export]").click()
        download = download_info.value
        download_path = run_dir / download.suggested_filename
        download.save_as(str(download_path))
        exported = json.loads(download_path.read_text(encoding="utf-8"))
        export_text = json.dumps(exported, ensure_ascii=False)
        download_has_raw_markers = any(
            marker in export_text
            for marker in ["stay_id", "subject_id", "hadm_id", "tableRows"]
        )
        download_ok = (
            exported.get("payload_scope") == "bounded_patient_review_drilldown"
            and isinstance(exported.get("patient_review"), dict)
            and not download_has_raw_markers
        )

        page.locator(".nextbar [data-nav='cohort']").click()
        page.wait_for_function("location.hash === '#cohort'", timeout=5000)
        nav_cohort_ok = "#cohort" == page.evaluate("location.hash")
        page.evaluate("location.hash = '#patient'")
        page.wait_for_function("document.body.innerText.includes('Local export patient drilldown ready')", timeout=8000)
        page.locator(".nextbar [data-nav='agent']").click()
        page.wait_for_function("location.hash === '#agent'", timeout=5000)
        nav_agent_ok = "#agent" == page.evaluate("location.hash")
        page.evaluate("location.hash = '#patient'")
        page.wait_for_function("document.body.innerText.includes('Local export patient drilldown ready')", timeout=8000)
        page.locator(".loaded-bar [data-viz-reset]").click()
        page.wait_for_function("document.body.innerText.includes('Load a review workspace')", timeout=5000)
        reset_ok = page.evaluate("document.body.innerText.includes('Ready to load local export')")
        result.update(source_ready)
        result["tabResults"] = tab_results
        result["tabTextPreviews"] = tab_texts
        result["downloadOk"] = download_ok
        result["downloadFilename"] = download.suggested_filename
        result["downloadHasRawMarkers"] = download_has_raw_markers
        result["downloadPath"] = str(download_path)
        result["navCohortOk"] = nav_cohort_ok
        result["navAgentOk"] = nav_agent_ok
        result["resetOk"] = reset_ok
        result["consoleErrors"] = errors
        if screenshots:
            shot = run_dir / "patient_drilldown_mobile.png"
            page.screenshot(path=str(shot), full_page=True)
            result["screenshot"] = str(shot)
        context.close()
        browser.close()
        return result


def main() -> int:
    args = parse_args()
    if not port_free(args.port):
        raise SystemExit(f"Port {args.port} is already in use; pass --port with a free local port.")

    run_dir = Path(args.out_dir) / f"native_fastapi_patient_drilldown_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    home = run_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    fixture = write_fixture_export(run_dir / "fixture_export")
    base_url = f"http://127.0.0.1:{args.port}/"
    proc = start_server(args.port, home)
    try:
        wait_ready(base_url, proc)
        registered = post_json(
            base_url,
            "api/workspaces/register",
            {"path": str(fixture), "label": "Patient Drilldown Fixture", "active": True},
        )
        sources = post_json(base_url, "api/patient-review/sources", {})
        validate_sources_payload(sources)
        payload = post_json(base_url, "api/patient-review/drilldown", {})
        validate_drilldown_payload(payload)
        browser = run_browser(base_url, run_dir, not args.no_screenshots)
        report = {
            "base_url": base_url,
            "run_dir": str(run_dir),
            "registered_active_path": registered.get("active_path"),
            "api": {
                "source_ready": {
                    "source_count": sources.get("source_count"),
                    "can_load": sources.get("can_load"),
                    "active_source": sources.get("active_source"),
                },
                "source": payload.get("source"),
                "summary": payload.get("summary"),
                "selected_label": (payload.get("selected") or {}).get("label"),
                "signal_count": len((payload.get("selected") or {}).get("signals") or []),
            },
            "browser": browser,
        }
        report_path = run_dir / "patient_drilldown_qa.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        failures = []
        if browser.get("consoleErrors"):
            failures.append(f"console errors: {browser['consoleErrors']}")
        if browser.get("overflowX", 0) > 1:
            failures.append(f"horizontal overflow: {browser.get('overflowX')}")
        for key in ["hasSourceReady", "hasSourceHash", "hasRealReady", "hasPseudonymous", "hasEntity2", "hasExportButton"]:
            if not browser.get(key):
                failures.append(f"browser assertion failed: {key}")
        if (browser.get("axisChartCount") or 0) < 3:
            failures.append(f"Patient Review rendered too few axis-backed charts: {browser.get('axisChartCount')}")
        if (browser.get("axisLabelCount") or 0) < 3:
            failures.append(f"Patient Review rendered too few axis labels: {browser.get('axisLabelCount')}")
        for tab in ["tables", "series", "patient", "quality"]:
            if not (browser.get("tabResults") or {}).get(tab):
                failures.append(f"Patient Review tab did not render expected panel: {tab}")
        for key in ["downloadOk", "navCohortOk", "navAgentOk", "resetOk"]:
            if not browser.get(key):
                failures.append(f"browser action assertion failed: {key}")
        if browser.get("downloadHasRawMarkers"):
            failures.append("downloaded Patient Review JSON leaked raw row-level markers")
        if browser.get("hasDemoCopy"):
            failures.append("real Patient Review still showed demo copy")
        if browser.get("hasRawMarkers"):
            failures.append("browser text leaked raw row-level markers")
        print(f"Patient drilldown QA report: {report_path}")
        print(json.dumps({"api": report["api"], "browser": browser}, indent=2, ensure_ascii=False))
        if failures:
            print("FAILURES:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("Patient drilldown QA passed.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
