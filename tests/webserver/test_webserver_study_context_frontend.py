"""Static ownership and handoff contracts for the native StudyContext UI."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from easyicu.webserver import study_contexts
from tools.run_js_contracts import CONTRACTS


STATIC = Path(__file__).parents[1] / "src" / "easyicu" / "webserver" / "static"
ROOT = Path(__file__).parents[1]


def _read(relative: str) -> str:
    return (STATIC / relative).read_text(encoding="utf-8")


def _node_binary() -> str | None:
    direct = shutil.which("node")
    if direct:
        return direct
    candidates = sorted((Path.home() / ".nvm" / "versions" / "node").glob("*/bin/node"))
    return str(candidates[-1]) if candidates else None


def test_study_context_owner_is_wired_before_route_modules() -> None:
    index = _read("index.html")
    assert "js/study-context.js?v=20260824-extraction-roundtrip1" in index
    assert index.index("js/api.js?") < index.index("js/study-context.js?")
    assert index.index("js/study-context.js?") < index.index("js/screens-extraction.js?")
    assert index.index("js/screens-extraction.js?") < index.index(
        "js/screens-extraction-study-context.js?"
    )
    assert index.index("js/screens-viz.js?") < index.index(
        "js/screens-viz-study-context.js?"
    )
    assert index.index("js/screens-viz-study-context.js?") < index.index(
        "js/screens-agent-study-context.js?"
    )
    assert index.index("js/screens-agent-study-context.js?") < index.index(
        "js/screens-agent.js?"
    )
    assert index.index("js/screens-guided.js?") < index.index(
        "js/screens-guided-study-context.js?"
    ) < index.index("js/app.js?")


def test_study_context_transport_and_payload_use_backend_canonical_fields() -> None:
    api = _read("js/api.js")
    owner = _read("js/study-context.js")
    for endpoint in (
        "/api/study-contexts/active",
        "/api/study-contexts",
        "/api/study-contexts/",
        "/api/study-contexts/handoff",
    ):
        assert endpoint in api
    for method in (
        "loadActiveStudyContext",
        "listStudyContexts",
        "loadStudyContext",
        "saveStudyContext",
        "handoffStudyContext",
    ):
        assert f"window.EU_API.{method}" in api
    assert "const PERSISTED_FIELDS" in owner
    for field in (
        "'data_source'",
        "'cohort'",
        "'modules'",
        "'outcome'",
        "'covariate_selection'",
        "'execution_concepts'",
        "'analysis_design'",
        "'sensitivity_specs'",
        "'time_window'",
        "'comparator'",
        "'current_stage'",
        "'last_route'",
    ):
        assert field in owner
    for obsolete in (
        "data_sources",
        "cohort_spec",
        "feature_modules",
        "latest_artifacts",
    ):
        assert obsolete not in owner
    assert "const payload = persistPayload(context)" in owner
    assert "payload.expected_revision = context.revision" in owner
    assert "api.saveStudyContext(payload)" in owner
    assert "expected_revision: savedContext.revision" in owner
    assert "let revision = 0" in owner
    assert "requestRevision === revision" in owner
    assert "hydrateRevision === revision" in owner
    assert "function startNew(patch, options)" in owner
    assert "function activate(id)" in owner
    assert "api.listStudyContexts()" in owner
    assert "const serverIds = new Set(serverContexts.map" in owner
    assert "serverIds.has(context.id)" in owner
    assert "|| isDirty(context.id)" in owner
    assert "sourceBoundary" in owner
    assert "sourceIdentity" in owner
    assert "questionBoundary" in owner
    assert "ROW_LEVEL_KEYS" in owner
    assert "assertMetadataOnly(raw, 'context')" in owner
    assert "cleanSchemaObject(raw.cohort, COHORT_SCHEMA)" in owner
    assert "cleanSchemaObject(raw.execution_concepts, EXECUTION_CONCEPTS_SCHEMA)" in owner
    assert "cleanSchemaObject(raw.analysis_design, ANALYSIS_DESIGN_SCHEMA)" in owner
    assert "analysis_family: 'text'" in owner
    assert "sensitivity_specs: cleanSensitivitySpecs(raw.sensitivity_specs)" in owner
    assert "covariate_selection: ['planner_selectable', 'exact'].includes" in owner
    assert "api.saveStudyContext({ id: context.id })" in owner
    assert "requestContextRevision === contextRevision(context.id)" in owner
    assert "easyicu:study-context" in owner
    assert "localStorage.setItem(STORAGE_KEY" in owner


def test_route_handoffs_have_sources_and_viz_mapping_has_its_own_owner() -> None:
    extraction = _read("js/screens-extraction.js")
    extraction_owner = _read("js/screens-extraction-study-context.js")
    viz = _read("js/screens-viz.js")
    viz_owner = _read("js/screens-viz-study-context.js")
    shell = _read("js/app.js")
    assert "syncExtractionToCopilot" in extraction
    assert "bridge.matchesDatabase(currentDatabase, nextDatabase)" in extraction
    assert "continueExisting: true, allowSourceRebind" in extraction
    assert "window.EU_EXTRACTION_CONTEXT" in extraction
    assert "registerSource(" not in extraction
    assert "window.EU_EXTRACTION_CONTEXT" in extraction_owner
    assert "window.EU_STUDY_CONTEXT.registerSource('extraction'" in extraction_owner
    assert "window.EU_EXTRACTION_STUDY_CONTEXT" in extraction_owner
    assert "function project(context, expectedDatabase)" in extraction_owner
    assert "function hydrate(context, expectedDatabase)" in extraction_owner
    assert "function matchesDatabase(expected, actual)" in extraction_owner
    assert "question: existing.question || ''" in extraction_owner
    assert "purpose: existing.purpose || ''" in extraction_owner
    assert "analysis_goal: existing.analysis_goal || ''" in extraction_owner
    assert "Run an evidence-bound analysis of the prepared cohort." not in extraction_owner
    # crossdb moved out of screens-viz.js into its own owner files; its handoff
    # marker moved with it. Assert per-owner rather than in the shell file, or
    # the test drifts into demanding a layering violation.
    for route in ("patient", "cohort"):
        assert f'data-study-source="{route}" data-study-target="guided"' in viz
    assert (
        'data-study-source="crossdb" data-study-target="guided"'
        in _read("js/screens-viz-crossdb-results.js")
    )
    for route in ("patient", "cohort", "crossdb"):
        assert route in viz_owner
    assert "window.EU_VIZ_CONTEXT" in viz
    assert "registerSource(" not in viz
    assert "window.EU_VIZ_CONTEXT" in viz_owner
    assert "window.EU_STUDY_CONTEXT.registerSource" in viz_owner
    assert "EU_STUDY_CONTEXT" not in shell


def test_guided_owns_run_submission_and_monitor_reuses_the_same_context() -> None:
    guided = _read("js/screens-guided.js")
    guided_owner = _read("js/screens-guided-study-context.js")
    agent = _read("js/screens-agent.js")
    agent_owner = _read("js/screens-agent-study-context.js")
    assert "window.EU_GUIDED_CONTEXT" in guided
    assert "EU_STUDY_CONTEXT" not in guided
    assert "window.EU_STUDY_CONTEXT.registerSource('guided'" in guided_owner
    assert "sourceRoute: 'guided'" in guided_owner
    assert "continueExisting: true" in guided_owner
    assert "study_id: runToken.study_id" in guided
    assert "study_context_id: runToken.context_id" in guided
    assert "persistForRun('agent_preflight')" in guided
    assert "EU_STUDY_CONTEXT" not in agent
    assert "projectKind: 'study_context'" in agent_owner
    assert "function projects()" in agent_owner
    assert "function activate(id)" in agent_owner
    assert "persistForRun(s)" not in agent
    assert "window.EU_API.startAgentRun" not in agent
    assert "markContextRunning(runToken.context_id, runToken.job_id" in guided
    assert "markContextFinished(" in guided
    assert "markActiveRunning" not in guided
    assert "markActiveFinished" not in guided
    # The monitor may finish a reconnected stream, but cannot initiate a run.
    assert "window.EU_AGENT_STUDY_CONTEXT.markContextFinished(" in agent
    assert "result && result.study_context_revision" in agent
    assert "createRunChannel" in agent_owner
    assert "createJobMemory" in agent_owner
    assert "prepareGuidedHandoff" in agent_owner
    assert "takeGuidedHandoff" in agent_owner
    assert "easyicu.pi-project-binding-handoff/1" in agent_owner
    assert 'data-ag-guided' in agent
    assert 'data-nav="guided"' not in agent[agent.index('<div class="handoff">'):agent.index('</div>`;', agent.index('<div class="handoff">'))]
    assert "prepareGuidedHandoff(selected)" in agent
    assert "takeGuidedHandoff()" in guided
    assert "binding_receipt: guidedBinding.binding_receipt || null" in guided
    assert "agJobMemory.get(studyId)" in agent
    assert "agRunChannel.isCurrent(runToken)" in agent
    assert "guidedRunChannel.isCurrent(runToken)" in guided
    assert "StudyContext persistence is unavailable; the real Guided run was not submitted." in guided
    assert "No active registered export is selected; no real run was submitted." in guided
    assert "path: src.path,\n      study_id: branch || 'guided'" not in guided
    assert guided.count("window.EU_API.startAgentRun({") == guided.count(
        "study_context_id: runToken.context_id"
    )
    assert "Applied when the run starts" in agent_owner
    assert "Informational until the analysis pipeline consumes them" in agent_owner
    assert "window.addEventListener('easyicu:study-context'" in agent_owner
    assert "${esc(t(s.name[0], s.name[1]))}" in agent
    assert "${esc(s.cohort)}" in agent
    assert "${esc(linkedPath || linked)}" in agent


def test_crossdb_handoff_is_plan_only_and_non_crossdb_routes_clear_the_flag() -> None:
    viz = _read("js/screens-viz.js")
    viz_owner = _read("js/screens-viz-study-context.js")
    extraction_owner = _read("js/screens-extraction-study-context.js")
    guided_owner = _read("js/screens-guided-study-context.js")
    agent_owner = _read("js/screens-agent-study-context.js")
    agent = _read("js/screens-agent.js")
    # The plan-only handoff button lives with the crossdb results owner. It had
    # silently degraded to a bare data-nav during the owner split, so nothing
    # ever set crossdb_plan_only and the Agent-side gate below was unreachable.
    crossdb_results = _read("js/screens-viz-crossdb-results.js")
    assert "Plan in Guided Copilot" in crossdb_results
    assert 'data-study-handoff data-study-source="crossdb"' in crossdb_results
    assert 'data-nav="agent"' not in crossdb_results
    assert "crossdb_plan_only" in viz_owner
    assert "confirmations.crossdb_plan_only = route === 'crossdb'" in viz_owner
    assert "ROUTE_CONFIRMATION_FIELDS.forEach(key => delete confirmations[key])" in viz_owner
    assert "crossdb_plan_only: false" in extraction_owner
    assert "crossdb_plan_only: false" in guided_owner
    assert "currentStage === 'crossdb_plan_only'" in agent_owner
    assert "function runBlocker(study)" in agent_owner
    assert "crossdb_selection" in viz_owner
    assert "crossdb_selection" in agent_owner
    assert "EU_SOURCES.crossdbPaths" not in agent
    assert "study.planOnly || selectedSources.length > 1" in agent_owner
    assert "No single export path or stay count is substituted" in agent
    assert "Cross-DB selection receipt bound" in agent


def test_nonfatal_agent_submission_warnings_are_visible_in_both_surfaces() -> None:
    guided = _read("js/screens-guided.js")
    agent = _read("js/screens-agent.js")
    owner = _read("js/screens-agent-study-context.js")
    assert "audit_warning" in owner
    # A run whose active-job reservation failed is now refused outright, so
    # there is deliberately no "the pointer did not sync but it runs anyway"
    # warning left for this screen to render.
    assert "context_sync_warning" not in owner
    assert "submissionWarning(r)" not in agent
    assert "warningNote(agRun.warning)" in agent
    assert "submissionWarning(r)" in guided
    assert "warningNote(guidedAgent.warning)" in guided


def test_agent_blocked_gate_planning_copy_and_tabs_are_truthful() -> None:
    agent = _read("js/screens-agent.js")
    owner = _read("js/screens-agent-study-context.js")
    assert "review_blocked" in agent
    assert "Evidence verification blocked" in agent
    assert "result.gate && result.gate.status === 'blocked'" in agent
    assert "gate && Array.isArray(gate.checks)" in agent
    assert "Waiting for verification results" in agent
    assert "Denominators resolved" not in agent
    assert "markContextStage(boundId, terminalStage(status, result), null, jobId, revision)" in owner
    assert "if (!updated) return null" in owner
    assert "Planning Blocks" not in agent
    assert "规划块" not in agent
    assert "window.EU_API.startAgentRun" not in agent
    assert "data-ag-guided" in agent
    assert "Workflow Blocks" not in agent
    assert 'role="tablist"' in agent
    assert 'role="tab"' in agent
    assert 'role="tabpanel"' in agent
    assert 'aria-selected="${selected}"' in agent
    assert "remembered.study_id === selected.id" in agent


def test_agent_and_guided_run_tokens_reject_interleaved_callbacks() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests" / "js" / "run_context_race.test.js"),
            str(STATIC / "js" / "screens-agent-study-context.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["ui_events"] == 2
    assert payload["patches"] == 4


def test_study_context_source_boundary_and_history_activation_in_javascript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    node = _node_binary()
    if not node:
        pytest.skip("Node.js is unavailable")
    owner_paths = [
        str(STATIC / "js" / owner)
        for owner in CONTRACTS["study_context_lifecycle.test.js"]
    ]
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests" / "js" / "study_context_lifecycle.test.js"),
            *owner_paths,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    monkeypatch.setattr(study_contexts, "_CONFIG_PATH", tmp_path / "contexts.json")
    for route in ("patient", "guided"):
        context = {
            key: value
            for key, value in payload[route].items()
            if key in study_contexts._CONTEXT_FIELDS
        }
        saved = study_contexts.upsert_context(context)
        binding = study_contexts.build_agent_context_binding(
            saved,
            export_path=saved["data_source"]["path"],
        )
        assert binding["status"] == "bound"
        assert saved["confirmations"].get("crossdb_plan_only") is False
        if route == "patient":
            assert saved["cohort"]["entity_count"] == 94_458
            assert saved["cohort"]["full_entity_count"] == 94_458
            assert saved["cohort"]["review_entities"] == 500
            assert saved["cohort"]["review_entity_cap"] == 500
            assert saved["cohort"]["review_scope"] == "browser_bounded_entity_sample"
            assert saved["confirmations"]["patient_review_bounded_sample"] is True
            assert saved["confirmations"]["patient_review_full_entity_set"] is False
        assert not {
            "source_count",
            "source_type",
            "comparison_mode",
        }.intersection(saved["cohort"])


def test_patient_scope_truth_renderer_in_javascript() -> None:
    node = _node_binary()
    if not node:
        pytest.skip("Node.js is unavailable")
    result = subprocess.run(
        [
            node,
            str(ROOT / "tests" / "js" / "patient_scope_truth.test.js"),
            str(STATIC / "js" / "screens-viz-patient-overview.js"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    assert json.loads(result.stdout) == {"ok": True}
