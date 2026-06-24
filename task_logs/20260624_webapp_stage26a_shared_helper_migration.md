# Stage26A Streamlit Decommission Shared Helper Migration

Date: 2026-06-24

## Scope

Stage26A migrates shared helper dependencies that still kept active FastAPI/API
code tied to the deprecated Streamlit package. It does not delete
`src/easyicu/webapp`, does not modify legacy Streamlit UI page logic, and does
not run or configure any external provider.

## Changes

- Added non-Streamlit shared modules:
  - `src/easyicu/ai_optin.py`
  - `src/easyicu/concept_catalog.py`
  - `src/easyicu/data_paths.py`
- Converted clean legacy modules to compatibility shims:
  - `src/easyicu/webapp/ai_optin.py` keeps Streamlit sidebar adaptation and reuses `easyicu.ai_optin`.
  - `src/easyicu/webapp/concept_catalog.py` forwards all metadata, including private legacy helpers, to `easyicu.concept_catalog`.
- Updated non-Streamlit production imports:
  - `src/easyicu/webserver/provider_gate.py`
  - `src/easyicu/webserver/catalog.py`
  - `src/easyicu/webserver/dataio.py`
  - `src/easyicu/api.py`
  - `src/easyicu/cohort_visualization.py`
  - `src/easyicu/comorbidity.py`
  - `src/easyicu/outcomes.py`
- Updated FastAPI catalog source comment:
  - `src/easyicu/webserver/static/js/data-catalog.js`
- Updated non-UI concept catalog tests and added migration regression:
  - `tests/test_shared_webapp_helper_migration.py`
  - medication/catalog tests now import `easyicu.concept_catalog`.
- Filled 10 missing shared catalog display labels for derived concepts already present in the extraction dictionary and catalog groups:
  `shock_index`, `modified_shock_index`, `diastolic_shock_index`,
  `oxygenation_index`, `corrected_calcium`, `nlr`, `plr`,
  `bun_creatinine_ratio`, `egfr`, `persistent_critical_illness`.

`src/easyicu/webapp/data_paths.py` was intentionally not edited or staged:
it already contains unrelated dirty Streamlit UI changes. Legacy compatibility
remains because the old function still exists there, while all non-Streamlit
production imports now use `easyicu.data_paths`.

`scripts/full_export_modules.py` is ignored by `scripts/*` and was not part of
the commit boundary. It should be treated with Stage26B/26C script and launcher
decommission work rather than forced into this tracked helper migration commit.

## Import Evidence

Command:

```bash
rg "easyicu.webapp.ai_optin|easyicu.webapp.concept_catalog|easyicu.webapp.data_paths" src tests
```

Remaining matches are limited to legacy Streamlit package files and the dirty
legacy UI test `tests/test_app_rendering.py`. The stricter tracked `src/easyicu`
non-WebApp production scan returned no matches:

```bash
rg -n "easyicu\\.webapp\\.(ai_optin|concept_catalog|data_paths)|from easyicu\\.webapp import (ai_optin|concept_catalog|data_paths)|import easyicu\\.webapp\\.(ai_optin|concept_catalog|data_paths)|\\.webapp\\.data_paths" src/easyicu --glob '!src/easyicu/webapp/**'
```

Result: no output.

## Validation

- `python -m py_compile src/easyicu/ai_optin.py src/easyicu/data_paths.py src/easyicu/concept_catalog.py src/easyicu/webapp/ai_optin.py src/easyicu/webapp/concept_catalog.py tests/test_shared_webapp_helper_migration.py` passed.
- `python -m compileall -q src/easyicu/webserver` passed.
- `python -m compileall -q src/easyicu` passed.
- `node --check src/easyicu/webserver/static/js/data-catalog.js` passed.
- `pytest -q tests/test_shared_webapp_helper_migration.py` passed: 4 passed.
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` passed: 60 passed.
- `pytest -q tests/test_concept_catalog_consistency.py tests/test_mock_data_catalog_coverage.py tests/test_batch2_medications.py tests/test_batch3_medications.py tests/test_batch4_medications.py tests/test_batch5_medications.py tests/test_batch6_medications.py tests/test_batch7_medications.py tests/test_batch8_medications.py tests/test_new_medication_concepts.py tests/test_propofol_rate_concept.py tests/test_furosemide_concept.py tests/test_mass_rate_concepts.py tests/test_uo_rate_concepts.py` passed: 289 passed.
- `pytest -q tests/test_api_cache.py tests/test_cohort_visualization_layout.py tests/test_shared_webapp_helper_migration.py` passed: 7 passed.
- `ruff check` on touched tracked production/test files passed.
- `git diff --check` passed.
- Provider dormant smoke passed with patched settings and provider env-file disabled:
  `ai_enabled=false`, `ready=false`, `client_constructed=false`,
  `network_calls=0`, `secrets_returned=false`.

## Remaining Stage26 Blockers

- `src/easyicu/webapp` is still a real legacy Streamlit package and contains UI
  imports of `easyicu.webapp.ai_optin`, `easyicu.webapp.concept_catalog`, and
  `easyicu.webapp.data_paths`.
- `pyproject.toml` still exposes `easyicu-webapp` and the `webapp` extra.
- Launch scripts and docs still reference Streamlit.
- Legacy UI tests, including dirty `tests/test_app_rendering.py`, still import
  `easyicu.webapp` and one dirty assertion still expects
  `cohort_visualization.py` to import `easyicu.webapp.data_paths`.

## Next Step

Stage26B should remove or downgrade Streamlit entrypoints, launcher defaults,
and docs. Stage26C should split or archive Streamlit UI tests so they are no
longer active FastAPI migration gates. Only after those references are cleared
should Stage27 delete `src/easyicu/webapp`.
