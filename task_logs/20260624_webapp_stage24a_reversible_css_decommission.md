# Stage24A reversible legacy Streamlit CSS decommission

Date: 2026-06-24

## Scope

Stage24A changes the default Streamlit CSS runtime path without deleting legacy CSS files. The FastAPI native WebApp remains the maintained path; legacy Streamlit UI is deprecated/frozen and no longer receives visual parity work.

No FastAPI webserver files were modified. No provider was started, no provider secrets were read, and no user data/export artifacts were deleted.

## Changes

- `src/easyicu/webapp/shell_styles.py`
  - Keeps loading `tokens.css` and `shell_overrides.css` by default.
  - Stops loading legacy route split CSS by default.
  - Adds temporary opt-in switch: `EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1`.
  - With the switch enabled, the existing split CSS load order remains available for rollback.
- `tools/inventory_legacy_streamlit_css.py`
  - Distinguishes present CSS files, default-loaded CSS, active-loaded CSS, and legacy-env loaded CSS.
  - Reports inactive-by-default legacy CSS counts and line totals.
- `src/easyicu/webapp/LEGACY.md`
  - Marks the Streamlit WebApp as deprecated/frozen.
  - Documents the FastAPI native mainline and temporary CSS opt-in.

## Inventory evidence

Default inventory command:

```bash
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage24a_default
```

Result:

- Report path: `/tmp/easyicu_stage24a_default/inventory_20260624_172513`
- Present CSS files: 16
- Total present CSS lines: 54,699
- Active/default loaded CSS files: 2
- Active/default loaded CSS lines: 1,680
- Default loaded CSS: `tokens.css`, `shell_overrides.css`
- Legacy split CSS inactive by default: 14 files / 53,019 lines
- Delete-now candidates: 0

Legacy opt-in inventory command:

```bash
EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1 python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage24a_legacy_enabled
```

Result:

- Report path: `/tmp/easyicu_stage24a_legacy_enabled/inventory_20260624_172513`
- Active loaded CSS files with env enabled: 16
- Active loaded CSS lines with env enabled: 54,699
- Legacy env loaded files: 16
- Legacy env loaded lines: 54,699

## Validation

Passed:

```bash
python -m py_compile src/easyicu/webapp/shell_styles.py tools/inventory_legacy_streamlit_css.py
python -m compileall -q src/easyicu/webserver
pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py
ruff check tools/inventory_legacy_streamlit_css.py
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage24a_default
EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1 python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage24a_legacy_enabled
git diff --check
```

Pytest result: 60 passed, 1 warning.

Provider dormant smoke:

```text
ai_enabled=false
ready=false
client_constructed=false
network_calls=0
secrets_returned=false
```

## Boundaries

No CSS files were deleted in Stage24A. The 14 legacy split CSS files remain tracked and can be temporarily restored into the legacy Streamlit runtime with:

```bash
EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1 easyicu-webapp
```

Existing unrelated dirty Streamlit Python/test/data files were not staged for this task.

## Stage24B conditions

Stage24B may delete the 14 inactive-by-default split CSS files after one more default-runtime validation pass confirms no active path depends on the env-disabled legacy CSS. Keep `tokens.css` and `shell_overrides.css` unless a separate import/runtime audit proves they are no longer needed.

Do not delete the whole `src/easyicu/webapp` package yet. That requires a separate archive/import audit proving no CLI, packaging, docs, tests, or local fallback workflows still import its Python modules.
