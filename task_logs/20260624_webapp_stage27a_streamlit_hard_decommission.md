# Stage27A Streamlit legacy package hard decommission

Date: 2026-06-24

## Scope

FastAPI native is the maintained Web UI. This stage removes the legacy
Streamlit package and legacy Streamlit UI tests from the active package
boundary after first saving a recovery patch for the dirty legacy worktree.

## Archive patch

Before deleting tracked legacy files, the dirty Streamlit worktree was saved to:

```text
output/stage27_streamlit_archive_patch/legacy_streamlit_dirty.patch
```

Companion files:

```text
output/stage27_streamlit_archive_patch/pre_stage27_status.txt
output/stage27_streamlit_archive_patch/legacy_dirty_files.txt
```

Patch evidence:

```text
legacy_streamlit_dirty.patch: 37,618 lines
legacy_dirty_files.txt: 28 files
```

The archive patch is a local recovery artifact under `output/`; it is not part
of the package runtime.

## Changes

- Removed the tracked legacy Streamlit package:
  - `src/easyicu/webapp/`
- Removed legacy Streamlit UI tests:
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
- Removed the legacy Streamlit entrypoint and dependency extra from
  `pyproject.toml`:
  - `easyicu-webapp-legacy`
  - `webapp-legacy`
- Removed legacy Streamlit pytest collection controls:
  - `--run-legacy-streamlit`
  - `legacy_streamlit`
- Updated docs to state that native FastAPI is the only maintained Web UI and
  that legacy Streamlit recovery is from git history or the local Stage27
  archive patch.
- Added a repository contract asserting that `src/easyicu/webapp`,
  `easyicu-webapp-legacy`, and `webapp-legacy` remain absent.

Deletion size:

```text
95 tracked legacy files removed
105,538 deleted lines
```

No `src/easyicu/webserver/**` files were modified.

## Reference scans

Required scan:

```bash
rg "easyicu\\.webapp" src tests scripts pyproject.toml README.md README_zh.md docs
```

Result: no matches.

The remaining broad `Streamlit`/`webapp` wording is historical or generic
documentation terminology only; it does not import `easyicu.webapp` or expose a
default Streamlit runtime path.

## Validation

Passed:

```bash
python -m py_compile tests/conftest.py tests/test_repository_contract.py
python -m compileall -q src/easyicu
ruff check tests/conftest.py tests/test_repository_contract.py
pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py tests/test_repository_contract.py
git diff --check
git diff --cached --check
```

Focused pytest result:

```text
78 passed, 1 warning
```

Provider dormant smoke:

```text
ai_enabled=false
ready=false
client_constructed=false
network_calls=0
secrets_returned=false
```

## Current verdict

The Streamlit WebApp package is no longer part of production code, default
tests, scripts, or maintained docs. The maintained Web UI is FastAPI native.

Recovery path if legacy Streamlit is needed:

- Use git history before Stage27.
- Or apply the local archive patch:
  `output/stage27_streamlit_archive_patch/legacy_streamlit_dirty.patch`.

## Remaining follow-up

- Optional docs cleanup can reduce historical `Streamlit` wording in design
  notes, but it is no longer a runtime or package dependency.
- Keep `src/easyicu/data/concept-dict2.json` and `src/easyicu/test-results/`
  out of this commit; they are unrelated untracked local artifacts.
