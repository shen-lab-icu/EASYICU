#!/usr/bin/env python3
"""Fixture E2E for native FastAPI Agent Projects workflow buttons.

The script starts an isolated FastAPI server with a temporary HOME, registers a
small EasyICU export fixture, drives the ``#agent`` page in Chromium real mode,
and verifies that visible controls call the real local backend:

* run analysis through ``/api/jobs/agent-run`` + SSE
* open artifact viewer through ``/api/agent-runs/artifact``
* download one artifact and the full whitelisted bundle
* write local human sign-off while keeping ``reportable=false``

Run from ``EASYICU/``:

    python tools/qa_native_fastapi_agent_workflow.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from qa_native_fastapi_patient_drilldown import (  # noqa: E402
    get_json,
    port_free,
    post_json,
    start_server,
    wait_ready,
    write_fixture_export,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8788)
    parser.add_argument("--out-dir", default="output/playwright")
    parser.add_argument("--no-screenshots", action="store_true")
    return parser.parse_args()


def _download_json(download_path: Path) -> dict[str, Any]:
    return json.loads(download_path.read_text(encoding="utf-8"))


def _zip_names(zip_path: Path) -> list[str]:
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        return sorted(zf.namelist())


def run_browser(base_url: str, run_dir: Path, screenshots: bool, study_id: str) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, viewport={"width": 1280, "height": 900})
        page = context.new_page()
        errors: list[str] = []
        requests: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("request", lambda req: requests.append(req.url) if "/api/" in req.url else None)
        page.goto(base_url + "#agent", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => !!(window.EU_API && typeof window.EU_API.startAgentRun === 'function' && typeof window.EU_API.signoffAgentRun === 'function')",
            timeout=8000,
        )
        page.evaluate("window.setDataMode && window.setDataMode('real', {force:true})")
        page.wait_for_function("document.body.innerText.includes('Agent Projects')", timeout=8000)

        page.wait_for_function(
            "(sid) => !!document.querySelector(`[data-ag-sel=\"${sid}\"]`)",
            arg=study_id,
            timeout=8000,
        )
        page.locator(f"[data-ag-sel='{study_id}']").click()
        page.wait_for_function(
            "(sid) => window.EU_AGENT_LAST_RUN ? true : !!document.querySelector(`.studycard.on[data-ag-sel=\"${sid}\"]`)",
            arg=study_id,
            timeout=5000,
        )
        page.locator(".nextbar [data-ag-runbtn]").click()
        page.wait_for_function(
            "() => document.body.innerText.includes('Real local artifacts read from') || document.body.innerText.includes('Evidence-bound preflight')",
            timeout=20000,
        )
        post_run = page.evaluate(
            """() => ({
              hasOutputs: document.body.innerText.includes('Real local artifacts read from'),
              hasEvidenceCopy: document.body.innerText.includes('Evidence-bound preflight'),
              hasBundleButton: !!document.querySelector('[data-ag-bundle-download]'),
              artifactCards: document.querySelectorAll('[data-ag-artifact-view]').length,
              projectDir: (window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.project_dir) || null,
              runId: (window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.run_id) || null,
              jobId: ((window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.run_id) || '').replace(/^run_/, ''),
              gateStatus: (window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.gate && window.EU_AGENT_LAST_RUN.gate.status) || null,
            })"""
        )

        page.wait_for_function(
            "() => !!(window.EU_AGENT_RUN_REVIEW && window.EU_AGENT_RUN_REVIEW.ok && document.querySelectorAll('[data-ag-artifact-view]').length)",
            timeout=8000,
        )
        artifact_names = page.evaluate(
            "() => [...document.querySelectorAll('[data-ag-artifact-view]')].map(x => x.dataset.agArtifactView || '')"
        )
        first_artifact = page.locator("[data-ag-artifact-view]").first
        with page.expect_response(lambda res: "/api/agent-runs/artifact" in res.url, timeout=8000) as artifact_response_info:
            first_artifact.click()
        artifact_response = artifact_response_info.value
        try:
            page.wait_for_function("() => document.body.innerText.toLowerCase().includes('artifact viewer')", timeout=8000)
            viewer_wait_ok = True
        except PlaywrightTimeoutError:
            viewer_wait_ok = False
        viewer = page.evaluate(
            """() => ({
              hasViewer: document.body.innerText.toLowerCase().includes('artifact viewer'),
              hasPrivacyScan: document.body.innerText.includes('privacy scan clean'),
              hasSha: /[a-f0-9]{12}/.test(document.body.innerText),
              artifactDownloadButtons: document.querySelectorAll('[data-ag-artifact-download]').length,
              bodyPreview: document.body.innerText.slice(0, 2200),
            })"""
        )
        viewer["waitOk"] = viewer_wait_ok

        artifact_path = None
        artifact_payload: Any = None
        artifact_raw_marker = False
        artifact_filename = None
        if viewer.get("artifactDownloadButtons"):
            with page.expect_download(timeout=8000) as artifact_download_info:
                page.locator("[data-ag-artifact-download]").first.click()
            artifact_download = artifact_download_info.value
            artifact_filename = artifact_download.suggested_filename
            artifact_path = run_dir / artifact_filename
            artifact_download.save_as(str(artifact_path))
            artifact_payload = _download_json(artifact_path)
            artifact_text = json.dumps(artifact_payload, ensure_ascii=False)
            artifact_raw_marker = any(marker in artifact_text for marker in ["stay_id", "subject_id", "hadm_id", "tableRows"])

        with page.expect_download(timeout=8000) as bundle_download_info:
            page.locator("[data-ag-bundle-download]").click()
        bundle_download = bundle_download_info.value
        bundle_path = run_dir / bundle_download.suggested_filename
        bundle_download.save_as(str(bundle_path))
        bundle_names = _zip_names(bundle_path)

        page.locator("[data-ag-tab='draft']").click()
        page.wait_for_function("document.body.innerText.includes('Local sign-off confirmations')", timeout=8000)
        initially_disabled = page.locator("[data-ag-live-signoff]").get_attribute("aria-disabled") == "true"
        for box in page.locator("[data-ag-live-confirm]").all():
            box.check()
        page.wait_for_function("document.querySelector('[data-ag-live-signoff]')?.getAttribute('aria-disabled') === 'false'", timeout=5000)
        enabled_after_checks = page.locator("[data-ag-live-signoff]").get_attribute("aria-disabled") == "false"
        with page.expect_response(lambda res: "/api/agent-runs/signoff" in res.url, timeout=8000) as signoff_response_info:
            page.locator("[data-ag-live-signoff]").click()
        signoff_response = signoff_response_info.value
        try:
            signoff_response_payload = signoff_response.json()
        except Exception:
            signoff_response_payload = {"_parse_error": True, "text": signoff_response.text()}
        if signoff_response.status == 200:
            page.wait_for_function("document.body.innerText.includes('human_signoff.json')", timeout=8000)
        signoff = page.evaluate(
            """() => {
              const review = window.EU_AGENT_RUN_REVIEW || {};
              const signoff = review.signoff || {};
              return {
                signed: !!review.signed,
                readinessStatus: review.readiness && review.readiness.status,
                signoffStatus: signoff.status,
                reportable: signoff.reportable,
                draftUnlocked: signoff.draft_unlocked,
                uploads: signoff.uploads,
                tokens: signoff.tokens,
                externalCalls: signoff.external_calls,
                signoffStale: !!review.signoff_stale,
                hasHumanSignoffArtifact: (review.artifacts || []).some(a => a.name === 'human_signoff.json'),
              };
            }"""
        )
        signoff["responseStatus"] = signoff_response.status
        signoff["responsePayload"] = signoff_response_payload

        restored = {"attempted": False}
        if post_run.get("jobId"):
            page.evaluate(
                """meta => {
                  localStorage.setItem('easyicu_last_idea_agent_project', JSON.stringify({
                    study_id: meta.studyId,
                    created_at: Date.now()
                  }));
                  localStorage.setItem('easyicu.agent.activeJob.v1', JSON.stringify({
                    job_id: meta.jobId,
                    study_id: meta.studyId,
                    source_path: (window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.source && window.EU_AGENT_LAST_RUN.source.path) || '',
                    run_type: 'preflight',
                    provider: 'mock',
                    created_at: Date.now()
                  }));
                }""",
                {"jobId": post_run["jobId"], "studyId": study_id},
            )
            page.goto(base_url + "?_v=stage68-agent-resume#agent", wait_until="domcontentloaded")
            page.wait_for_function(
                "() => !!(window.EU_API && typeof window.EU_API.loadJobSnapshot === 'function' && typeof window.EU_API.cancelJob === 'function')",
                timeout=8000,
            )
            page.evaluate("window.setDataMode && window.setDataMode('real', {force:true})")
            page.wait_for_function(
                "(sid) => document.body.innerText.includes('Real local artifacts read from') && window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.study_id === sid",
                arg=study_id,
                timeout=12000,
            )
            try:
                page.wait_for_function(
                    "() => !!(window.EU_AGENT_RUN_REVIEW && window.EU_AGENT_RUN_REVIEW.ok)",
                    timeout=8000,
                )
            except PlaywrightTimeoutError:
                pass
            restored = page.evaluate(
                """() => ({
                  attempted: true,
                  hash: location.hash,
                  studyId: window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.study_id,
                  runId: window.EU_AGENT_LAST_RUN && window.EU_AGENT_LAST_RUN.run_id,
                  hasOutputs: document.body.innerText.includes('Real local artifacts read from'),
                  hasReview: !!(window.EU_AGENT_RUN_REVIEW && window.EU_AGENT_RUN_REVIEW.ok),
                  snapshotApiPresent: !!(window.EU_API && window.EU_API.loadJobSnapshot),
                  cancelApiPresent: !!(window.EU_API && window.EU_API.cancelJob),
                  activeJobStorageCleared: !localStorage.getItem('easyicu.agent.activeJob.v1'),
                })"""
            )

        final = page.evaluate(
            """() => {
              const doc = document.documentElement;
              const body = document.body;
              const text = body.innerText;
              return {
                hash: location.hash,
                hasDemoCopy: /Seeded demo artifacts|Demo simulated|seeded example/i.test(text),
                hasRawMarkers: /stay_id|subject_id|hadm_id|tableRows/.test(text),
                overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - window.innerWidth,
              };
            }"""
        )
        if screenshots:
            shot = run_dir / "agent_workflow_desktop.png"
            page.screenshot(path=str(shot), full_page=True)
            final["screenshot"] = str(shot)

        context.close()
        browser.close()
        return {
            "postRun": post_run,
            "viewer": viewer,
            "artifactNames": artifact_names,
            "artifactResponseStatus": artifact_response.status,
            "artifactDownload": {
                "filename": artifact_filename,
                "path": str(artifact_path) if artifact_path else None,
                "json_object": isinstance(artifact_payload, dict),
                "raw_marker": artifact_raw_marker,
            },
            "bundleDownload": {
                "filename": bundle_download.suggested_filename,
                "path": str(bundle_path),
                "names": bundle_names,
                "has_quality_gate": "quality_gate.json" in bundle_names,
                "has_evidence_ledger": "evidence_ledger.json" in bundle_names,
            },
            "signoff": signoff,
            "restoredAfterReload": restored,
            "signoffInitiallyDisabled": initially_disabled,
            "signoffEnabledAfterChecks": enabled_after_checks,
            "consoleErrors": errors,
            "apiRequests": sorted(set(requests)),
            **final,
        }


def main() -> int:
    args = parse_args()
    if not port_free(args.port):
        raise SystemExit(f"Port {args.port} is already in use; pass --port with a free local port.")

    run_dir = Path(args.out_dir) / f"native_fastapi_agent_workflow_{time.strftime('%Y%m%d_%H%M%S')}"
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
            {"path": str(fixture), "label": "Agent Workflow Fixture", "active": True},
        )
        idea_run = post_json(
            base_url,
            "api/ideas/mine",
            {
                "topic": "Vasopressor-fluid resuscitation strategy and in-hospital mortality in adult ICU patients",
                "title": "Vasopressor or Fluids in Early Septic Shock",
                "journal": "New England Journal of Medicine",
                "year": 2026,
                "doi": "10.1056/NEJMoa2516225",
                "excerpt": (
                    "The article reports a randomized trial comparing restricted intravenous fluid "
                    "with earlier vasopressor use in septic shock. Use vasopressor, total input, "
                    "shock index, SOFA, lactate, and death as candidate EasyICU concepts."
                ),
            },
        )
        handoff = post_json(
            base_url,
            "api/ideas/handoff",
            {
                "run_id": idea_run.get("run_id"),
                "idea_id": idea_run.get("selected_idea_id"),
                "plan_edits": "QA seed: run local Agent preflight from the active fixture export.",
            },
        )
        project = post_json(
            base_url,
            "api/ideas/create-agent-project",
            {
                "run_id": idea_run.get("run_id"),
                "idea_id": idea_run.get("selected_idea_id"),
                "plan_edits": "QA seed: run local Agent preflight from the active fixture export.",
            },
        )
        study_id = ((project.get("project") or {}).get("study_id") or "").strip()
        if not study_id:
            raise SystemExit(f"Agent project seed was not created: {project}")
        provider_status = get_json(base_url, "api/agent-runs/provider-status").get("provider_status") or {}
        browser = run_browser(base_url, run_dir, not args.no_screenshots, study_id)
        report = {
            "base_url": base_url,
            "run_dir": str(run_dir),
            "registered_active_path": registered.get("active_path"),
            "idea_run_id": idea_run.get("run_id"),
            "handoff_run_id": handoff.get("run_id"),
            "agent_project_study_id": study_id,
            "provider_status": {
                "ai_enabled": provider_status.get("ai_enabled"),
                "ready": provider_status.get("ready"),
                "client_constructed": provider_status.get("client_constructed"),
                "network_calls": provider_status.get("network_calls"),
                "secrets_returned": provider_status.get("secrets_returned"),
            },
            "browser": browser,
        }
        report_path = run_dir / "agent_workflow_qa.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        failures: list[str] = []
        post_run = browser.get("postRun") or {}
        if not post_run.get("hasOutputs") or not post_run.get("hasBundleButton") or post_run.get("artifactCards", 0) < 4:
            failures.append(f"run did not land in real outputs: {post_run}")
        viewer = browser.get("viewer") or {}
        if not viewer.get("hasViewer") or not viewer.get("hasPrivacyScan") or not viewer.get("artifactDownloadButtons"):
            failures.append(f"artifact viewer did not expose expected safe controls: {viewer}")
        artifact = browser.get("artifactDownload") or {}
        if not artifact.get("json_object") or artifact.get("raw_marker"):
            failures.append(f"artifact download was invalid or leaked raw markers: {artifact}")
        bundle = browser.get("bundleDownload") or {}
        if not bundle.get("has_quality_gate") or not bundle.get("has_evidence_ledger"):
            failures.append(f"bundle download missed expected artifacts: {bundle}")
        signoff = browser.get("signoff") or {}
        if not browser.get("signoffInitiallyDisabled") or not browser.get("signoffEnabledAfterChecks"):
            failures.append("signoff button did not enforce three confirmations")
        if not signoff.get("signed") or signoff.get("signoffStatus") != "signed_analysis_only":
            failures.append(f"signoff was not written as signed_analysis_only: {signoff}")
        if signoff.get("reportable") is not False or signoff.get("draftUnlocked") is not False:
            failures.append(f"signoff unlocked/reportable unexpectedly: {signoff}")
        if signoff.get("externalCalls") not in (0, None):
            failures.append(f"signoff reported external calls: {signoff}")
        restored = browser.get("restoredAfterReload") or {}
        if not restored.get("attempted") or not restored.get("hasOutputs") or not restored.get("hasReview"):
            failures.append(f"agent job snapshot restore after reload failed: {restored}")
        if not restored.get("snapshotApiPresent") or not restored.get("cancelApiPresent"):
            failures.append(f"agent resume/cancel API wrappers missing in browser: {restored}")
        if not restored.get("activeJobStorageCleared"):
            failures.append(f"completed agent job was not cleared from localStorage: {restored}")
        if browser.get("hasDemoCopy"):
            failures.append("real Agent workflow showed seeded demo copy")
        if browser.get("hasRawMarkers"):
            failures.append("Agent workflow DOM leaked raw row-level markers")
        if browser.get("overflowX", 0) > 1:
            failures.append(f"horizontal overflow: {browser.get('overflowX')}")
        if browser.get("consoleErrors"):
            failures.append(f"console errors: {browser.get('consoleErrors')}")
        if provider_status.get("client_constructed") or provider_status.get("network_calls") or provider_status.get("secrets_returned"):
            failures.append(f"provider was not dormant: {provider_status}")
        print(f"Agent workflow QA report: {report_path}")
        print(json.dumps({"provider_status": report["provider_status"], "browser": browser}, indent=2, ensure_ascii=False))
        if failures:
            print("FAILURES:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("Agent workflow QA passed.")
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
