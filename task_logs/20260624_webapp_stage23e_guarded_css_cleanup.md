# Stage23E guarded legacy CSS cleanup

Date: 2026-06-24

## Scope

- Goal: continue legacy Streamlit CSS cleanup with stateful browser/computed-style guard.
- Edited files:
  - `src/easyicu/webapp/guided_overrides.css`
  - `tools/qa_legacy_streamlit_css_guard.py`
- Not touched:
  - `src/easyicu/webserver/**`
  - `src/easyicu/webapp/shell_styles.py`
  - legacy Streamlit Python page files
  - dirty tests/data artifacts
- No CSS file deletion and no import-order change.
- Provider stayed dormant; no provider secrets were read.

## Guard setup

The guard was extended with Patient loaded-bar/current-setup rail selectors so future Patient cleanup has direct component-level coverage:

- `.eu-qv-rail`
- `.eu-qv-rail-head`
- `.eu-qv-rail-sep`
- `.eu-qv-rail-edit`
- `.eu-qv-rail .setup-row`
- `[class*='st-key-eu_qv_loaded_bar']`
- `.eu-qv-loaded-copy-line`
- `.loaded-bar`
- `.eu-qv-loaded-export-visual`

Full before guard:

- `output/playwright/stage23_css_guard_20260624_165742_stage23e_before_extended_patient_guard/computed_style_guard.json`
- routes: Extract step2/step3/step4/export preview, Patient loaded tables/time-series/overview/quality, Guided welcome/study workspace, Shell navigation
- result: `failures=[]`

Full after guard:

- `output/playwright/stage23_css_guard_20260624_171052_stage23e_after_full/computed_style_guard.json`
- result: `failures=[]`

Earlier accepted guided responsive probe:

- `output/playwright/stage23_css_guard_20260624_165402_stage23e_after_guided_responsive_probe/computed_style_guard.json`
- result: `failures=[]`

## Accepted cleanup

All accepted deletions were in `src/easyicu/webapp/guided_overrides.css`.

| Original start line | Deleted lines | Block | Evidence |
|---:|---:|---|---|
| 554 | 16 | old viewport-fill helper for `.eu-copilot-page-marker` main container | Later guided structural wrapper correction and final cascade cover the visible state; guided guard after deletion passed. |
| 2805 | 269 | `Research Copilot final layout guardrails`, `Research Copilot polish parity pass`, and `Research Copilot composer polish` | Later guided design-source/final/terminal cascade owns topbar, rail, composer, and guided intent geometry; guided guard after deletion passed. |
| 3415 | 150 | `Research Copilot reference rail reduction` | Later guided design-source and terminal cascade own rail widths/backgrounds and study rail presentation; guided guard after deletion passed. |
| 5354 | 68 | `Research Copilot - responsive max-width parity` | Later EOF final guided page parity owns composer/intents max-width and submit button sizing; guided guard after deletion passed. |

Net CSS deletion this stage: 503 lines.

## Rolled back candidates

| File | Candidate | Guard result | Decision |
|---|---|---|---|
| `patient_overrides.css` | loaded mobile topbar current-page blocks plus early `.eu-qv-rail*` sidebar block | `output/playwright/stage23_css_guard_20260624_170158_stage23e_after_patient_topbar_rail_probe/computed_style_guard.json`; desktop `patient_loaded_tables`, `patient_loaded_time_series`, `patient_loaded_overview`, and `patient_loaded_quality` computed-style signatures changed | Reverted. The early Patient rail/topbar layer is still live. |
| `shell_navigation_overrides.css` | early `Sidebar footer (design .rail-foot)` block | `output/playwright/stage23_css_guard_20260624_170359_stage23e_after_shell_footer_probe/computed_style_guard.json`; mobile shell expected-hidden sidebar/topbar became visible and desktop/mobile signatures changed | Reverted. The early block still carries mobile hide fallback and footer geometry. |

## Validation

Commands and results:

- `python tools/qa_legacy_streamlit_css_guard.py --base-url http://127.0.0.1:8513 --label stage23e_after_full --compare-before output/playwright/stage23_css_guard_20260624_165742_stage23e_before_extended_patient_guard/computed_style_guard.json --routes extract_step2 extract_step3 extract_step4 extract_export_preview patient_loaded_tables patient_loaded_time_series patient_loaded_overview patient_loaded_quality guided_welcome guided_study_workspace shell_navigation --no-screenshots`
  - `failures=[]`
- `python tools/inventory_legacy_streamlit_css.py --check-extract-owner-guards`
  - `issues=[]`
- `python tools/inventory_legacy_streamlit_css.py --check-patient-owner-guards`
  - `issues=[]`
- `python tools/inventory_legacy_streamlit_css.py --check-guided-owner-guards`
  - `issues=[]`
- `python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage23e_inventory_check`
  - `files=16`, `imported=16`, `untracked=0`, `delete_now=[]`, `lines=54699`
- `python -m py_compile tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py`
  - passed
- `ruff check tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py`
  - passed
- `git diff --check`
  - passed
- provider dormant smoke:
  - `ai_enabled=false`
  - `ready=false`
  - `client_constructed=false`
  - `network_calls=0`
  - `secrets_returned=false`

## Current CSS inventory

- Total CSS lines: 54,699
- Largest files:
  - `agent_overrides.css`: 14,196
  - `guided_overrides.css`: 7,759
  - `extract_overrides.css`: 6,289
  - `patient_overrides.css`: 5,345
  - `crossdb_overrides.css`: 3,435

Baseline comparison:

- Stage21 baseline: 62,171 lines
- Stage23D after: 55,202 lines
- Stage23E after: 54,699 lines
- Total reduction since baseline: 7,472 lines

## Remaining blockers

- Patient loaded topbar/sidebar rail still has live desktop computed-style ownership; deletion requires finer replacement owner or markup cleanup.
- Shell footer/mobile hide fallback is still live; do not delete until shell mobile nav/floating launcher guard explicitly covers replacement owners.
- Extract completion/export-progress remains blocked until a safe read-only export-progress fixture hook exists.
- Guided remaining large blocks are mostly live terminal/final cascade. Future cleanup should target small guard-backed duplicate internals, not broad file-level deletion.

## Next action

Stage23F should continue with guarded cleanup only after adding replacement-owner guard for one of:

- Patient loaded bar/sidebar rail final owner
- Shell mobile nav/floating launcher and footer final owner
- Extract completion/export-progress fixture state
- Guided terminal/composer/rail subcomponent computed-style snapshots
