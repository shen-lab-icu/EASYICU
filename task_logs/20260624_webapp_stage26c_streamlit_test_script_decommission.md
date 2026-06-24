# Stage26C Streamlit tests/script decommission

Date: 2026-06-24

## Scope

FastAPI native is now the default WebApp entrypoint. This stage splits the
remaining Streamlit test/script references so Stage27 can decide whether to
archive or delete the legacy `src/easyicu/webapp` package.

## Changes

- `tests/conftest.py`
  - Added `--run-legacy-streamlit`.
  - Default pytest collection ignores legacy Streamlit UI files and
    `tests/webapp/`.
  - Explicitly collected legacy items are marked `legacy_streamlit`.
- `pytest.ini`
  - Declared the `legacy_streamlit` marker.
- `tests/test_repository_contract.py`
  - Updated the concept-catalog split contract to match Stage26A: the large
    catalog now lives in `easyicu.concept_catalog`; the legacy Streamlit module
    is a compatibility shim.
- `README.md`, `README_zh.md`, `CONTRIBUTING.md`,
  `docs/native_fastapi_webserver.md`
  - Documented that `pytest -q` is the default FastAPI/core gate and legacy
    Streamlit tests require `pytest --run-legacy-streamlit ...`.
- `docs/streamlit_decommission_audit_stage25.md`
  - Updated Stage25 audit status after Stage26C.
- `scripts/full_export_modules.py`
  - Local gitignored utility was changed from
    `easyicu.webapp.concept_catalog` to `easyicu.concept_catalog`.
  - It is ignored by `.gitignore` (`scripts/*`) and is not part of the tracked
    commit boundary unless explicitly force-added later.

## Legacy Streamlit test scope

Default `pytest -q` no longer collects:

- `tests/test_app_rendering.py`
- `tests/test_cohort_workspace_bundle.py`
- `tests/test_llm_chat.py`
- `tests/test_mock_data_catalog_coverage.py`
- `tests/test_real_ui_smoke.py`
- `tests/test_research_agent_web_helpers.py`
- `tests/test_shared_webapp_helper_migration.py`
- `tests/test_webapp_launch.py`
- `tests/test_webapp_resume_panel.py`
- `tests/webapp/`

Explicit legacy command:

```bash
pytest --run-legacy-streamlit tests/test_webapp_launch.py
pytest --run-legacy-streamlit tests/webapp
```

## Evidence

Reference scans:

```bash
rg -n "easyicu\\.webapp|from easyicu import webapp" tests scripts src --glob '!src/easyicu/webapp/**'
```

Remaining hits are in the explicit legacy test list above. The script hit is
gone in the local ignored script after the import migration.

Default gate collection sample:

```bash
pytest --collect-only -q | rg "test_app_rendering.py|test_webapp_launch.py|test_shared_webapp_helper_migration.py|tests/webapp|test_repository_contract.py|collected"
```

Result:

```text
collected 2298 items
    <Module test_repository_contract.py>
======================== 2298 tests collected in 2.75s =========================
```

Validation passed:

- `python -m py_compile tests/conftest.py scripts/full_export_modules.py`
- `python -m compileall -q src/easyicu`
- `ruff check tests/conftest.py`
- `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py` -> 60 passed
- `pytest -q tests/test_repository_contract.py tests/test_concept_catalog_consistency.py` -> 35 passed
- `pytest -q --run-legacy-streamlit tests/test_webapp_launch.py tests/test_shared_webapp_helper_migration.py` -> 5 passed
- `git diff --check`

Provider dormant smoke still reports:

```text
ai_enabled=false
ready=false
client_constructed=false
network_calls=0
secrets_returned=false
```

## Stage27 blockers

- Explicit legacy tests still import `easyicu.webapp` when run with
  `--run-legacy-streamlit`.
- `src/easyicu/webapp` still has internal self-imports and Streamlit runtime
  code.
- Stage27 must choose one of:
  - migrate useful pure helper coverage to shared/FastAPI modules, then delete
    the legacy package and tests; or
  - keep the package as an explicit legacy archive and stop short of deletion.
