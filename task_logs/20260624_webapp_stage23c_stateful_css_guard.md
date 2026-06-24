# Stage23C Legacy Streamlit CSS Stateful Guard

Date: 2026-06-24

## Scope

- Goal: extend the legacy Streamlit CSS browser/computed-style guard so stateful Extract, Patient, Guided, and Shell navigation states are actually reachable before further selector-level cleanup.
- Changed files: `tools/qa_legacy_streamlit_css_guard.py` only.
- CSS deletion: 0 lines. No CSS file was edited or deleted.
- Explicit non-scope: no FastAPI changes, no `shell_styles.py` import changes, no provider calls, no old Streamlit Python/test dirty line staging.

## Guard Changes

- Isolated every route/viewport in its own Playwright browser context to prevent Streamlit session-state bleed across routes.
- Added Streamlit idle waiting via `[data-testid="stApp"][data-test-script-state]` before evaluating DOM/computed styles.
- Made `guardBlockers` fail validation instead of being hidden in JSON.
- Added viewport-specific state contracts:
  - Extract step2/step3/step4/export preview use real `ex_action` query links and wait for route-owned visible DOM.
  - Patient loaded tables/time-series/overview/quality use real demo loaded state and include current renderer owners such as `.eu-ts-lane-head`, `.quality-summary-grid`, `.quality-issue-panel`, and `stPlotlyChart`.
  - Guided welcome/study workspace requires visible assistant panel/shell owners and validates mobile left-rail collapse.
  - Shell navigation validates desktop sidebar/topbar and mobile bottom nav; floating launcher is recorded but not used as a deletion guard because its visibility is not stable enough for cleanup decisions.
- Normalized report comparison to ignore known non-CSS volatility: dynamic rects, text length, subpixel grid tracks, width/height, button counts, and hidden optional selector existence.

## Browser QA Evidence

Primary passing guard:

`output/playwright/stage23_css_guard_20260624_155116_stage23c_stateful_guard_v5/computed_style_guard.json`

Routes:

| Route | Desktop | Mobile |
|---|---:|---:|
| extract_step2 | pass | pass |
| extract_step3 | pass | pass |
| extract_step4 | pass | pass |
| extract_export_preview | pass | pass |
| patient_loaded_tables | pass | pass |
| patient_loaded_time_series | pass | pass |
| patient_loaded_overview | pass | pass |
| patient_loaded_quality | pass | pass |
| guided_welcome | pass | pass |
| guided_study_workspace | pass | pass |
| shell_navigation | pass | pass |

Result: `failures=[]`, root `overflowX=0` for all 22 route/viewport records.

Repeat guard:

`output/playwright/stage23_css_guard_20260624_155346_stage23c_stateful_guard_repeat/computed_style_guard.json`

- Route validation failures excluding strict compare noise: `[]`.
- Normalized compare between v5 and repeat reports: `[]`.
- The original strict compare failed on dynamic items such as button counts, hidden optional `stRadio`, subpixel grid columns, and rect/text drift, so the tool now compares only stable style-contract fields.

## Ownership / Inventory

Owner scans:

| Scan | Result |
|---|---|
| `--check-extract-owner-guards` | `issues=[]`, `unclassified_marker=0` |
| `--check-patient-owner-guards` | `issues=[]` |
| `--check-guided-owner-guards` | `issues=[]`, `unclassified_marker=0` |

Inventory:

`python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage23c_inventory_check`

- Output: `/tmp/easyicu_stage23c_inventory_check/inventory_20260624_155758`
- CSS files: 16
- Imported: 16
- Untracked: 0
- `delete_now=[]`
- Total CSS lines: 55,714

Largest CSS files after this round:

| File | Lines |
|---|---:|
| `agent_overrides.css` | 14,196 |
| `guided_overrides.css` | 8,422 |
| `extract_overrides.css` | 6,641 |
| `patient_overrides.css` | 5,345 |
| `crossdb_overrides.css` | 3,435 |

## Validation

- `python -m py_compile tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py` passed.
- `ruff check tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py` passed.
- `git diff --check` passed.
- Provider dormant smoke: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.

## Remaining Guard Gaps / Blockers

- True export-progress state (`.eu-export-progress-shell`) was not driven because it may trigger real export/write behavior; needs a safe read-only fixture hook before CSS depending on export progress can be deleted.
- The shell floating launcher is recorded in computed-style output but is not a reliable cleanup guard yet; desktop/mobile visibility differs across stateful runs, so no deletion should depend on it.
- Offscreen/clipped samples remain nonzero in several legacy pages even with `overflowX=0`; this preserves Stage20/23 caution that high-specificity layout cascade cannot be bulk-deleted without more component-level visual guards.

## Decision

Stage23C is done as a stateful guard expansion. It deliberately deletes 0 CSS lines. Next cleanup should use this guard to test a very small guarded deletion batch, preferably one confirmed duplicate block in `extract_overrides.css` or `patient_overrides.css`, and should avoid Guided high-specificity cascade until component-level visual/computed guards are narrower.
