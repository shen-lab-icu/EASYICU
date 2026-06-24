# Stage24B remove inactive legacy Streamlit split CSS

Date: 2026-06-24

## Scope

Stage24B deletes the 14 legacy Streamlit split CSS files that Stage24A already removed from the default runtime path. FastAPI native remains the maintained WebApp path.

No FastAPI webserver files were modified. No provider was started, no provider secrets were read, and no user data/export artifacts were deleted.

## Pre-delete evidence

Command:

```bash
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage24b_before_delete
```

Result:

- Report path: `/tmp/easyicu_stage24b_before_delete/inventory_20260624_175918`
- Present CSS files: 16
- Total present CSS lines: 54,699
- Active/default loaded CSS: `tokens.css`, `shell_overrides.css`
- Active/default loaded lines: 1,680
- Inactive-by-default legacy split CSS: 14 files / 53,019 lines

## Deleted CSS files

Deleted with `git rm`:

| File | Deleted lines |
|---|---:|
| `src/easyicu/webapp/agent_overrides.css` | 14,196 |
| `src/easyicu/webapp/alignment.css` | 21 |
| `src/easyicu/webapp/cohort_overrides.css` | 3,014 |
| `src/easyicu/webapp/crossdb_overrides.css` | 3,435 |
| `src/easyicu/webapp/dictionary_overrides.css` | 909 |
| `src/easyicu/webapp/entry_overrides.css` | 2,251 |
| `src/easyicu/webapp/extract_overrides.css` | 6,289 |
| `src/easyicu/webapp/guided_overrides.css` | 7,759 |
| `src/easyicu/webapp/patient_overrides.css` | 5,345 |
| `src/easyicu/webapp/settings_overrides.css` | 2,138 |
| `src/easyicu/webapp/shell_navigation_overrides.css` | 2,548 |
| `src/easyicu/webapp/states_overrides.css` | 2,437 |
| `src/easyicu/webapp/tutorial_overrides.css` | 2,536 |
| `src/easyicu/webapp/visualization_shell_overrides.css` | 141 |

Total deleted CSS lines: 53,019.

## Post-delete evidence

Command:

```bash
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage24b_after_delete
```

Result:

- Report path: `/tmp/easyicu_stage24b_after_delete/inventory_20260624_180057`
- Present CSS files: 2
- Present CSS lines: 1,680
- Active/default loaded CSS: `tokens.css`, `shell_overrides.css`
- Active/default loaded lines: 1,680
- Stage24B removed legacy split CSS files: 14
- Missing required default CSS: 0

Remaining CSS files:

- `src/easyicu/webapp/tokens.css`: 200 lines
- `src/easyicu/webapp/shell_overrides.css`: 1,480 lines

## Documentation and inventory updates

- `src/easyicu/webapp/LEGACY.md` now states the route split CSS has been removed and can only be recovered from Git history (`63bba1c` or earlier) for archive forensics.
- `tools/inventory_legacy_streamlit_css.py` now reports removed split CSS as Stage24B intentional removal and treats only missing default CSS (`tokens.css`, `shell_overrides.css`) as a runtime inventory issue.

## Validation

Passed:

```bash
python -m py_compile src/easyicu/webapp/shell_styles.py tools/inventory_legacy_streamlit_css.py
python -m compileall -q src/easyicu/webserver
pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py
ruff check tools/inventory_legacy_streamlit_css.py
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage24b_after_delete
git diff --check
git diff --name-only -- src/easyicu/webserver
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

`git diff --name-only -- src/easyicu/webserver` returned empty.

## Boundaries

This stage did not delete `tokens.css`, `shell_overrides.css`, or the `src/easyicu/webapp` Python package. Existing unrelated dirty Streamlit Python/test/data files were not staged for this task.

## Recovery path

If legacy Streamlit route CSS is needed for archive forensics, restore the removed files from Git history at `63bba1c` or an earlier commit, then run the old app with:

```bash
EASYICU_ENABLE_LEGACY_STREAMLIT_CSS=1 easyicu-webapp
```

Do not resume selector-level visual maintenance unless explicitly requested.
