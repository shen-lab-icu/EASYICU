# Stage27B Streamlit residual cleanup

Date: 2026-06-24

## Goal

After Stage27A removed the legacy Streamlit package boundary, clean the remaining non-functional Streamlit residue and run a CI-equivalent validation pass. This stage does not restore or modify the legacy Streamlit WebApp. FastAPI native remains the only Web UI.

## Scope

- Removed active metadata references to historical Streamlit paths in `src/easyicu/visualization_design.json`; the web UI metadata now points to `src/easyicu/webserver/` and the `easyicu-webapp` FastAPI command.
- Reworded shared/helper docstrings that still described Streamlit-specific avoidance or wrappers:
  - `src/easyicu/ai_optin.py`
  - `src/easyicu/data_paths.py`
  - `src/easyicu/hosted_llm_server.py`
  - `src/easyicu/webserver/app.py`
  - `src/easyicu/webserver/dataio.py`
  - `src/easyicu/webserver/jobs.py`
  - `src/easyicu/webserver/settings.py`
- Updated release-hardening tests so they assert the native FastAPI metadata contract instead of reading the deleted `src/easyicu/webapp/app.py`.
- Removed local scratch artifacts that were not user data:
  - `src/easyicu/data/concept-dict2.json` was an empty `{}` scratch file.
  - `src/easyicu/test-results/` contained generated test runner output.
- Added `.gitignore` entries to keep those package-local scratch artifacts out of future status noise.
- Fixed two unused local variables surfaced by full-suite `ruff`:
  - `src/easyicu/webserver/cohort_review.py`
  - `src/easyicu/webserver/numeric_evidence_audit.py`

## Evidence

| Check | Result |
|---|---|
| `find src/easyicu -maxdepth 3 \( -path '*/test-results*' -o -name 'concept-dict2.json' \) -print` | no output after cleanup |
| `rg "easyicu\.webapp\|easyicu-webapp-legacy\|webapp-legacy\|--run-legacy-streamlit\|legacy_streamlit" src tests scripts pyproject.toml README.md README_zh.md docs CONTRIBUTING.md pytest.ini` | only negative repository/release-hardening assertions remain |
| `rg "streamlit\|Streamlit" src/easyicu/visualization_design.json src/easyicu/ai_optin.py src/easyicu/data_paths.py src/easyicu/hosted_llm_server.py src/easyicu/webserver tests/test_release_hardening_p0.py` | only negative release-hardening assertion names/strings remain |
| `python -m json.tool src/easyicu/visualization_design.json` | pass |
| `python -m py_compile ...` for touched Python modules | pass |
| `python -m compileall -q src/easyicu` | pass |
| `ruff check src tests` | pass |
| `pytest -q tests/test_release_hardening_p0.py tests/test_repository_contract.py` | 25 passed |
| `pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py tests/test_repository_contract.py` | 78 passed, 1 warning |
| `pytest -q` | 2246 passed, 41 skipped, 950 warnings in 336.44s |
| provider dormant smoke with `EASYICU_DISABLE_PROVIDER_ENV_FILE=1` | `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false` |
| `git diff --check` | pass |

## Concurrency

Pre-flight `ps` showed a concurrent Claude provider/research-agent run and an existing uvicorn process on `127.0.0.1:8765`. This stage did not touch or kill either process and did not print or read provider secrets.

## Verdict

Stage27B is done. The legacy Streamlit WebApp package remains removed, active source metadata now points to FastAPI native, default tests and production code do not depend on `easyicu.webapp`, and local scratch artifacts are no longer present in the worktree. Remaining broad documentation mentions of Streamlit are historical/decommission context, not active package dependencies.

## Next

Return to FastAPI native and submission work. If future cleanup is needed, treat it as historical documentation polish, not WebApp migration work.
