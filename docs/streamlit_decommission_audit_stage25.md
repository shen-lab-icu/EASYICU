# Stage25 Streamlit WebApp decommission audit

Date: 2026-06-24

## Verdict

Do not delete `src/easyicu/webapp` in the current state.

The legacy route split CSS has been removed, and only `tokens.css` plus
`shell_overrides.css` remain. Stage26A migrated shared helpers out of the
Streamlit namespace, and Stage26B moved the default package entrypoint and
repository launchers to the native FastAPI WebApp. The Python package still
cannot be deleted because legacy UI tests and `src/easyicu/webapp` self-imports
remain.

## Current package size

- Tracked files under `src/easyicu/webapp`: 81
- Python files under `src/easyicu/webapp`: 76
- Python line count under `src/easyicu/webapp`: 93,922
- Remaining CSS files: 2
- Remaining CSS line count: 1,680
  - `tokens.css`: 200
  - `shell_overrides.css`: 1,480

## Scan evidence

Representative commands:

```bash
git merge-base --is-ancestor 7fb9f89 HEAD && echo contains_7fb9f89=yes
rg -n "easyicu\\.webapp|src/easyicu/webapp|easyicu-webapp|streamlit|Streamlit" src/easyicu -g '!src/easyicu/webapp/**'
rg -n "easyicu\\.webapp|streamlit|Streamlit" tests -g '!**/__pycache__/**'
rg -n "easyicu-webapp|project\\.scripts|streamlit|webapp" pyproject.toml
rg -n "easyicu\\[webapp\\]|streamlit|easyicu-webapp|webapp" scripts/launch_easyicu.py start_easyicu.sh start_easyicu.command README.md README_zh.md pyproject.toml
```

`7fb9f89 Remove inactive legacy Streamlit split CSS` is in the current branch history.

## Active dependency blockers

| Class | Evidence | Classification | Required Stage26 action |
|---|---|---|---|
| FastAPI provider gate | `src/easyicu/webserver/provider_gate.py` historically imported `easyicu.webapp.ai_optin` | resolved by Stage26A | Uses shared `easyicu.ai_optin` policy; legacy module is a compatibility shim. |
| FastAPI catalog endpoint | `src/easyicu/webserver/catalog.py` historically imported `easyicu.webapp.concept_catalog` | resolved by Stage26A | Uses shared `easyicu.concept_catalog`. |
| FastAPI export/data summaries | `src/easyicu/webserver/dataio.py` historically imported `CONCEPT_GROUPS_INTERNAL` from `easyicu.webapp.concept_catalog` | resolved by Stage26A | Uses shared `easyicu.concept_catalog`. |
| Core public API path detection | `src/easyicu/api.py` historically imported `easyicu.webapp.data_paths.find_database_path` | resolved by Stage26A | Uses shared `easyicu.data_paths`. |
| Core visualization helper | `src/easyicu/cohort_visualization.py` historically imported `easyicu.webapp.data_paths.find_database_path` | resolved by Stage26A | Uses shared `easyicu.data_paths`. |
| Export script | `scripts/full_export_modules.py` still imports `easyicu.webapp.concept_catalog` | remaining script dependency | Move to shared `easyicu.concept_catalog` before Stage27 package deletion, or archive the script if it is legacy-only. |
| Streamlit launcher | `scripts/launch_easyicu.py`, `start_easyicu.sh`, `start_easyicu.command` historically installed/ran the Streamlit app | resolved by Stage26B | Default launchers now start the native FastAPI server on port 8765. |
| Python package entrypoint | `pyproject.toml` historically had `easyicu-webapp = "easyicu.webapp.__main__:main"` | resolved by Stage26B | `easyicu-webapp` now points to `easyicu.webserver.__main__:main`; Streamlit is explicit as `easyicu-webapp-legacy`. |
| Optional dependencies | `pyproject.toml` historically defined `webapp` as Streamlit dependencies | resolved by Stage26B | `webapp` now installs native FastAPI dependencies; Streamlit dependencies moved to `webapp-legacy`. |

## Tests-only blockers

The following test files import `easyicu.webapp` or Streamlit-specific helpers and must be migrated, archived, skipped, or deleted before deleting the package:

- `tests/test_app_rendering.py`
- `tests/test_research_agent_web_helpers.py`
- `tests/test_webapp_launch.py`
- `tests/test_webapp_resume_panel.py`
- `tests/test_real_ui_smoke.py`
- `tests/test_repository_contract.py`
- `tests/test_llm_chat.py`
- `tests/test_cohort_workspace_bundle.py`
- `tests/test_concept_catalog_consistency.py`
- `tests/test_mock_data_catalog_coverage.py`
- `tests/test_batch2_medications.py`
- `tests/test_batch3_medications.py`
- `tests/test_batch4_medications.py`
- `tests/test_batch5_medications.py`
- `tests/test_batch6_medications.py`
- `tests/test_batch7_medications.py`
- `tests/test_batch8_medications.py`
- `tests/test_new_medication_concepts.py`
- `tests/test_propofol_rate_concept.py`
- `tests/test_furosemide_concept.py`
- `tests/test_mass_rate_concepts.py`
- `tests/test_uo_rate_concepts.py`
- `tests/webapp/test_copilot_classic_parity.py`
- `tests/webapp/test_copilot_classic_parity_synthetic.py`
- `tests/webapp/test_copilot_cli_agent.py`
- `tests/webapp/test_copilot_engine.py`
- `tests/webapp/test_copilot_keyword_route.py`

The medication/concept catalog tests are not inherently Streamlit tests; they should move to the shared concept catalog import after the catalog is extracted.

## Docs-only and archive references

Docs and historical tools still mention Streamlit as legacy/archive material:

- `src/easyicu/README.md`
- `docs/_internal/code_and_tools_map_20260523.txt`
- `docs/_internal/web_ux_rearchitecture_list.txt`
- `tools/qa_legacy_streamlit_css_guard.py`
- `tools/legacy_streamlit_fallback_baseline_stage21b.json`
- `tools/inventory_legacy_streamlit_css.py`
- `docs/legacy_streamlit_fallback_baseline_stage21b.md`

These are not runtime blockers. Public top-level README files and the native
FastAPI docs now point to the native WebApp by default; older internal maps and
legacy tools should be archived or removed with Stage26C/Stage27.

## Shared helper migration list

Done in Stage26A:

1. `easyicu.webapp.ai_optin`
   - Keep pure provider policy outside Streamlit.
   - Native FastAPI should not import any module that imports `streamlit` at module load.
2. `easyicu.webapp.concept_catalog`
   - Move concept names, groups, coverage, descriptions, and supported DB metadata into a shared importable module.
   - Update FastAPI catalog, extraction/export dataio, concept tests, medication tests, and scripts.
3. `easyicu.webapp.data_paths.find_database_path`
   - Extract path detection into a non-Streamlit module.
   - Leave `_directory_input`, folder picker UI, and `st.session_state` wrappers in legacy/archive code only.
4. Optional if preserving Copilot/Agent helper tests:
   - `workspace_snapshots.py`, `cohort_workspace.py`, `copilot/*`, and selected pure helpers from `agent_workbench.py` / `research_agent.py` need a decision: migrate pure pieces or retire the legacy tests with the app.

## Stage26 deletion plan

1. Create shared modules:
   - Done in Stage26A: `src/easyicu/ai_optin.py`.
   - Done in Stage26A: `src/easyicu/concept_catalog.py`.
   - Done in Stage26A: `src/easyicu/data_paths.py`.
2. Update active imports:
   - Done in Stage26A for `src/easyicu/webserver/provider_gate.py`.
   - Done in Stage26A for `src/easyicu/webserver/catalog.py`.
   - Done in Stage26A for `src/easyicu/webserver/dataio.py`.
   - Done in Stage26A for `src/easyicu/api.py`.
   - Done in Stage26A for `src/easyicu/cohort_visualization.py`.
   - Remaining follow-up: `scripts/full_export_modules.py`.
3. Update or remove package entrypoints:
   - Done in Stage26B: `easyicu-webapp` and `start_easyicu.*` are native FastAPI by default.
   - Done in Stage26B: Streamlit launch is explicit as `easyicu-webapp-legacy`.
   - Done in Stage26B: Streamlit dependencies moved to `webapp-legacy`.
4. Split tests:
   - Convert concept catalog tests to shared catalog imports.
   - Convert provider policy tests to the shared opt-in module.
   - Archive/delete legacy UI rendering tests once the Streamlit package is removed.
5. Update public docs:
   - `README.md`, `README_zh.md`, `src/easyicu/README.md`, and native FastAPI docs should no longer present Streamlit as the active no-code path.
6. Only after all imports are gone:
   - `git rm -r src/easyicu/webapp`
   - Remove legacy tools specific to Streamlit CSS/browser guard, or move them under archive.
   - Run full focused native checks plus import/package checks.

## Deletion readiness

Current readiness: no.

Blocking reasons:

- At least 27 test files import `easyicu.webapp` or Streamlit-specific helpers.
- `scripts/full_export_modules.py` still imports the legacy concept catalog shim.
- `src/easyicu/webapp` still has internal self-imports and Streamlit runtime code.
- Some docs and archive tools intentionally mention Streamlit as legacy material.
- Stage26C must split/skip/archive legacy tests before Stage27 can remove the package.

Stage25 is an audit-only milestone. It does not remove files.
