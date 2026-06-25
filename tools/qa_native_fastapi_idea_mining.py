#!/usr/bin/env python3
"""Browser QA for the native FastAPI Idea Mining workflow.

Starts an isolated FastAPI server, registers a small EasyICU export, drives
``#ideas`` in Chromium, and verifies the first-class local discovery path:

* user-supplied article/topic/excerpt -> ``/api/ideas/mine``
* source evidence table + idea ledger + pre-experiment statistics render
* ``/api/ideas/handoff`` freezes an Agent handoff without unlocking draft
* ``/api/ideas/create-agent-project`` creates a metadata-only Agent project seed
* no row-level markers or provider/network calls appear in the browser payload
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

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
    parser.add_argument("--port", type=int, default=8812)
    parser.add_argument("--out-dir", default="output/playwright")
    parser.add_argument("--no-screenshots", action="store_true")
    return parser.parse_args()


def run_browser(base_url: str, run_dir: Path, screenshots: bool) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1280, "height": 900})
        page = context.new_page()
        errors: list[str] = []
        requests: list[str] = []
        page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        page.on("pageerror", lambda exc: errors.append(str(exc)))
        page.on("request", lambda req: requests.append(req.url) if "/api/" in req.url else None)

        page.goto(base_url + "#ideas", wait_until="domcontentloaded")
        page.wait_for_function(
            "() => !!(window.EU_API && typeof window.EU_API.mineIdeas === 'function')",
            timeout=8000,
        )
        page.evaluate("window.setDataMode && window.setDataMode('real', {force:true})")
        page.fill(
            "#ideaTopic",
            "Early septic shock resuscitation comparing vasopressor-first or fluid-sparing strategy "
            "against fluid-forward resuscitation, with lactate, blood pressure, SOFA-2 severity, "
            "and mortality outcomes.",
        )
        page.locator("details.ideas-advanced summary").first.click()
        page.fill("#ideaTitle", "Vasopressors or Fluids in Early Septic Shock")
        page.fill("#ideaJournal", "New England Journal of Medicine")
        page.fill("#ideaYear", "2026")
        page.fill(
            "#ideaExcerpt",
            "Adult septic shock patients were assigned to restricted intravenous fluid and earlier "
            "vasopressor use or greater fluid volume and later vasopressors.",
        )

        with page.expect_response(lambda res: "/api/ideas/resolve-source" in res.url, timeout=10000) as resolve_info:
            page.locator("[data-idea-resolve]").click()
        resolve_response = resolve_info.value
        page.wait_for_function(
            "() => document.body.innerText.includes('metadata_ready') || document.body.innerText.includes('source resolved')",
            timeout=8000,
        )

        with page.expect_response(lambda res: "/api/ideas/mine" in res.url, timeout=10000) as mine_info:
            page.locator("[data-idea-mine]").click()
        mine_response = mine_info.value
        page.wait_for_function(
            "() => document.body.innerText.includes('Idea ledger') && document.body.innerText.includes('Pre-experiment')",
            timeout=10000,
        )
        mined = page.evaluate(
            """() => {
              const run = window.EU_IDEA_LAST_RUN || {};
              const idea = (run.idea_ledger || [])[0] || {};
              const concepts = (idea.mapped_concepts || []).map((row) => row.concept_id);
              return {
                runId: run.run_id || null,
                selectedIdeaId: run.selected_idea_id || null,
                ideaTitle: idea.idea_title || null,
                conceptIds: concepts,
                ledgerRows: document.querySelectorAll('table tbody tr').length,
                sourceEvidenceVisible: document.body.innerText.includes('SOURCE EVIDENCE'),
                preExperimentVisible: document.body.innerText.includes('Pre-experiment'),
                feasibilityText: document.body.innerText,
                goNoGo: idea.go_no_go || null,
                preStatus: run.pre_experiment && run.pre_experiment.status,
                missingRequired: run.pre_experiment && run.pre_experiment.missing_required_concepts,
                networkCalls: run.privacy && run.privacy.network_calls,
                externalLlmCalls: run.privacy && run.privacy.external_llm_calls,
            };
        }"""
        )

        page.locator("[data-idea-step='handoff']").first.click()
        page.wait_for_function(
            "() => document.body.innerText.includes('Plan handoff') && !!document.querySelector('#ideaPlanEdits')",
            timeout=8000,
        )
        page.fill("#ideaPlanEdits", "Use adult first ICU stay and add a missingness sensitivity check.")
        with page.expect_response(lambda res: "/api/ideas/handoff" in res.url, timeout=10000) as handoff_info:
            page.locator("[data-idea-handoff]").click()
        handoff_response = handoff_info.value
        page.wait_for_function("() => !!window.EU_IDEA_HANDOFF", timeout=8000)
        handoff = page.evaluate(
            """() => {
              const h = window.EU_IDEA_HANDOFF || {};
              return {
                ok: !!h.ok,
                runId: h.run_id || null,
                ideaId: h.idea_id || null,
                reportable: h.agent_seed && h.agent_seed.reportable,
                draftUnlocked: h.agent_seed && h.agent_seed.draft_unlocked,
                planNotes: h.handoff_plan && h.handoff_plan.human_plan_notes,
              };
            }"""
        )
        with page.expect_response(lambda res: "/api/ideas/create-agent-project" in res.url, timeout=10000) as project_info:
            page.locator("[data-idea-create-project]").click()
        project_response = project_info.value
        page.wait_for_function("() => !!window.EU_IDEA_AGENT_PROJECT", timeout=8000)
        project = page.evaluate(
            """() => {
              const p = window.EU_IDEA_AGENT_PROJECT || {};
              return {
                studyId: p.study_id || null,
                title: p.title || null,
                status: p.status || null,
                reportable: p.reportable,
                draftUnlocked: p.draft_unlocked,
              };
            }"""
        )

        page.locator("[data-idea-new]").first.click()
        page.wait_for_function(
            "() => document.body.innerText.includes('No idea ledger yet')",
            timeout=8000,
        )
        with page.expect_response(lambda res: "/api/ideas/run" in res.url, timeout=10000) as run_info:
            page.locator("[data-idea-record]").first.click()
        run_response = run_info.value
        page.wait_for_function(
            "() => document.body.innerText.includes('Idea ledger') && document.body.innerText.includes('Plan handoff')",
            timeout=10000,
        )
        history_load = page.evaluate(
            """() => ({
              ledgerVisible: document.body.innerText.includes('Idea ledger'),
              preExperimentVisible: document.body.innerText.includes('Pre-experiment'),
              handoffVisible: document.body.innerText.includes('Handoff written') || document.body.innerText.includes('Plan handoff'),
              projectVisible: document.body.innerText.includes('Agent project seed created') || document.body.innerText.includes('Create Agent project'),
            })"""
        )

        page.locator("[data-nav='agent']").first.click()
        page.wait_for_function(
            "() => location.hash === '#agent' && document.body.innerText.includes('Vasopressor-fluid resuscitation strategy')",
            timeout=10000,
        )
        agent = page.evaluate(
            """() => ({
              hash: location.hash,
              seedVisible: document.body.innerText.includes('Vasopressor-fluid resuscitation strategy'),
              seedModeVisible: /Idea seed|IDEA SEED|想法种子/.test(document.body.innerText),
              projectText: document.body.innerText,
            })"""
        )
        final = page.evaluate(
            """() => {
              const doc = document.documentElement;
              const body = document.body;
              const text = body.innerText;
              return {
                hash: location.hash,
                hasRawMarkers: /stay_id|subject_id|hadm_id|tableRows/.test(text),
                overflowX: Math.max(doc.scrollWidth, body.scrollWidth) - window.innerWidth,
              };
            }"""
        )
        if screenshots:
            shot = run_dir / "idea_mining_desktop.png"
            page.screenshot(path=str(shot), full_page=True)
            final["screenshot"] = str(shot)
        context.close()
        browser.close()
        return {
            "resolveResponseStatus": resolve_response.status,
            "mineResponseStatus": mine_response.status,
            "handoffResponseStatus": handoff_response.status,
            "projectResponseStatus": project_response.status,
            "historyRunResponseStatus": run_response.status,
            "mined": mined,
            "handoff": handoff,
            "project": project,
            "historyLoad": history_load,
            "agent": agent,
            "consoleErrors": errors,
            "apiRequests": sorted(set(requests)),
            **final,
        }


def main() -> int:
    args = parse_args()
    if not port_free(args.port):
        raise SystemExit(f"Port {args.port} is already in use; pass --port with a free local port.")

    run_dir = Path(args.out_dir) / f"native_fastapi_idea_mining_{time.strftime('%Y%m%d_%H%M%S')}"
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
            {"path": str(fixture), "label": "Idea Mining Fixture", "active": True},
        )
        provider_status = get_json(base_url, "api/agent-runs/provider-status").get("provider_status") or {}
        browser = run_browser(base_url, run_dir, not args.no_screenshots)
        report = {
            "base_url": base_url,
            "run_dir": str(run_dir),
            "registered_active_path": registered.get("active_path"),
            "provider_status": {
                "ai_enabled": provider_status.get("ai_enabled"),
                "ready": provider_status.get("ready"),
                "client_constructed": provider_status.get("client_constructed"),
                "network_calls": provider_status.get("network_calls"),
                "secrets_returned": provider_status.get("secrets_returned"),
            },
            "browser": browser,
        }
        report_path = run_dir / "idea_mining_qa.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        failures: list[str] = []
        if browser.get("resolveResponseStatus") != 200:
            failures.append(f"resolve response {browser.get('resolveResponseStatus')}")
        if browser.get("mineResponseStatus") != 200:
            failures.append(f"mine response {browser.get('mineResponseStatus')}")
        if browser.get("handoffResponseStatus") != 200:
            failures.append(f"handoff response {browser.get('handoffResponseStatus')}")
        if browser.get("projectResponseStatus") != 200:
            failures.append(f"project response {browser.get('projectResponseStatus')}")
        if browser.get("historyRunResponseStatus") != 200:
            failures.append(f"history run response {browser.get('historyRunResponseStatus')}")
        mined = browser.get("mined") or {}
        if mined.get("preStatus") != "partial":
            failures.append(f"expected partial pre-experiment, got {mined.get('preStatus')}")
        if "Vasopressor-fluid" not in str(mined.get("ideaTitle") or ""):
            failures.append(f"strategy idea title was not preserved: {mined.get('ideaTitle')}")
        concept_ids = set(mined.get("conceptIds") or [])
        if {"vaso_ind", "total_input_ml", "death"} - concept_ids:
            failures.append(f"strategy concept set was not preserved: {sorted(concept_ids)}")
        missing_required = set(mined.get("missingRequired") or [])
        if {"vaso_ind", "total_input_ml", "lact"} - missing_required:
            failures.append(f"missing required concepts incomplete: {sorted(missing_required)}")
        if mined.get("networkCalls") != 0 or mined.get("externalLlmCalls") != 0:
            failures.append("idea mining used network/external LLM")
        handoff = browser.get("handoff") or {}
        if handoff.get("reportable") is not False or handoff.get("draftUnlocked") is not False:
            failures.append(f"handoff unlocked reporting: {handoff}")
        project = browser.get("project") or {}
        if project.get("status") != "seeded_from_idea":
            failures.append(f"agent project seed was not created: {project}")
        if project.get("reportable") is not False or project.get("draftUnlocked") is not False:
            failures.append(f"project seed unlocked reporting: {project}")
        history_load = browser.get("historyLoad") or {}
        if not all(history_load.get(k) for k in ["ledgerVisible", "preExperimentVisible", "handoffVisible", "projectVisible"]):
            failures.append(f"history card did not restore the idea run: {history_load}")
        agent = browser.get("agent") or {}
        if not agent.get("seedVisible"):
            failures.append("agent page did not show the idea-derived project seed")
        if browser.get("hasRawMarkers"):
            failures.append("browser text leaked raw row markers")
        if browser.get("consoleErrors"):
            failures.append(f"console errors: {browser.get('consoleErrors')}")
        if (browser.get("overflowX") or 0) > 0:
            failures.append(f"horizontal overflow: {browser.get('overflowX')}")
        if provider_status.get("client_constructed") or provider_status.get("network_calls"):
            failures.append(f"provider was not dormant: {provider_status}")

        print(json.dumps(report, indent=2, ensure_ascii=False))
        if failures:
            raise SystemExit("; ".join(failures))
        return 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
