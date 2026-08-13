# Nature publication skills default integration

> Date: 2026-08-12
> Task: `AGENT-NATURE-PUBLICATION-SKILLS`
> Modules: `agent` + `web`

## Outcome

EasyICU now exposes two governed, default-on publication skills through one typed registry:

- `nature-figure`: reuses the existing article figure strategy and deterministic `PublicationFigureSkill` renderer. Result-bearing figures remain source-data/code backed and retain figure-contract, editable-export, and visual-QA requirements; free-form image generation cannot create or alter numeric results.
- `nature-writing`: adds a versioned, case-neutral writing contract to the Writer prompt pack. It requires paragraph roles, calibrated claims, stable scientific terms, exact run-evidence bindings, and run-bound literature citations, while forbidding invented results, references, methods, novelty, mechanisms, or statistics.

Both skills are enabled by default. Web Settings now provides a built-in Skills master switch plus individual Nature Figure and Nature Writing switches. The Web runner reads the settings once when a new run is created and freezes the resolved flags into `PipelineConfig`; changing a switch does not mutate an already-running or historical run.

The Write phase compiles a deterministic `publication_skill_activation` receipt, registers it in the run evidence store, and records active/inactive skill IDs plus an activation SHA-256. The Science Workbench and capability API consume the same registry instead of maintaining copied policy lists.

This change adds governed built-ins, not an arbitrary user-upload/install marketplace. User-defined Skill packages, MCP servers, and third-party plugins still need a separate manifest, permission, validation, and lifecycle design.

## Owner contracts and implementation

- Registry and activation receipt owner: `src/easyicu/research_agent/publication_skills.py`
- Writing contract: `src/easyicu/research_agent/providers/prompts/v1/nature_writing.txt`
- Run-bound configuration: `src/easyicu/research_agent/orchestration/config.py`, `src/easyicu/research_agent/pipeline.py`
- Writer and Write-phase integration: `src/easyicu/research_agent/agents/core.py`, `src/easyicu/research_agent/reporting/write_phase.py`
- Web settings/capability/workbench/run binding: `src/easyicu/webserver/settings.py`, `capabilities.py`, `science_workbench.py`, `agent_pipeline_runs.py`
- Settings UI owner: `src/easyicu/webserver/static/js/screens-settings.js`

No Nature-specific CSS was added. The existing Settings owner styles remain responsible for the controls.

## Verification

Canonical interpreter: `.venv/bin/python` (Python 3.11.15).

Focused and adjacent gate:

```text
.venv/bin/python -m pytest -q \
  tests/research_agent/test_publication_skills.py \
  tests/test_webserver_capabilities.py \
  tests/test_webserver_science_workbench.py \
  tests/test_webserver_settings_contract.py \
  tests/test_pi_copilot_research_workflow.py::test_web_runner_delegates_to_research_agent_pipeline \
  tests/research_agent/test_pipeline.py::test_pipeline_end_to_end_synthetic_cohort

53 passed, 4 warnings in 105.24s
```

Additional checks:

- Ruff on the changed Python scope: passed.
- Python compile check: passed.
- `node --check` for `screens-settings.js`: passed.
- `git diff --check`: passed.
- All static CSS brace/comment scan: passed.
- Route ownership scan: no Nature-specific selector was added outside the Settings owner.

Desktop browser QA against the real FastAPI app:

- 1440×900 and 1280×800: no document/body horizontal overflow; all three Skills switches remained visible and unclipped.
- Both individual switches loaded as enabled, persisted an off/on round trip through `/api/settings`, and were restored to `true` after QA.
- Chinese save feedback renders as `Nature 图件已保存。` / the corresponding Nature Writing localized label, rather than leaking a raw settings key.
- The temporary QA server and browser sessions were stopped; pre-existing user services were not touched.

## Remaining boundary

A fresh real-provider Agent UAT may verify the activation receipt alongside real manuscript/figure artifacts. It must not be described as a Canonical9 result or paper-authority unlock: this task made the skills available and auditable, but did not run a provider, patient dataset, formal benchmark batch, or frozen publication experiment.
