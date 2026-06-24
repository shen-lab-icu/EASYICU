# Stage23B Legacy CSS Guard Extension + Small Cleanup

Date: 2026-06-24

## Scope

- Continued legacy Streamlit fallback CSS cleanup after `45fd02a`.
- Did not touch `src/easyicu/webserver/**`.
- Did not modify `src/easyicu/webapp/shell_styles.py`.
- Did not delete any CSS file.
- Did not run or read external provider secrets.
- Did not stage existing dirty Streamlit Python/test/data changes.

## Changes

### Browser/computed-style guard

Updated `tools/qa_legacy_streamlit_css_guard.py` to cover finer fallback states:

- Extract: `extract_idle`, `extract_step2`, `extract_step3`, `extract_step4`.
- Patient: `patient_idle`, `patient_loaded_tables`, `patient_loaded_time_series`, `patient_loaded_overview`, `patient_loaded_quality`.
- Guided: `guided_welcome`, `guided_study_workspace` with optional intent-click interaction.
- Shell navigation: `shell_navigation` for desktop/sidebar and mobile shell visibility checks.

The guard now records `guardBlockers` when a requested deeper state cannot be reached in the current URL/query-driven fallback session. These blockers do not fail the basic route guard, but they block high-specificity cascade deletion.

## CSS cleanup accepted

| File | Deleted lines | Evidence | Guard result |
|---|---:|---|---|
| `src/easyicu/webapp/extract_overrides.css` | 6 | Removed an earlier duplicate real-source button recolor block; the later route-owned final block remains. | Accepted after Extract/Patient/Guided/Shell computed-style guard. |
| `src/easyicu/webapp/guided_overrides.css` | 39 net | Removed source-missing `route_fallback_*` fallback prompt block; `rg route_fallback src/easyicu/webapp/*.py` finds no live source owner. Also removed empty media residuals. | Accepted after Extract/Patient/Guided/Shell computed-style guard. |
| `src/easyicu/webapp/patient_overrides.css` | 7 | Removed empty media/comment residuals that carried no declarations. | Accepted after Extract/Patient/Guided/Shell computed-style guard. |

An initial full after-guard also reported an Agent desktop computed-style drift, but the only Agent change was an empty media rule. To keep the batch evidence precise, that empty Agent change was reverted and the accepted after-guard was restricted to the routes touched by this batch.

## Browser guard evidence

Before guard:

- `output/playwright/stage23_css_guard_20260624_150035_before_stage23b_guard_extension/computed_style_guard.json`

Accepted after guard:

- `output/playwright/stage23_css_guard_20260624_151632_after_stage23b_extract_patient_guided_shell/computed_style_guard.json`
- `failures=[]`

Guard blockers recorded in the accepted after guard:

| Viewport | Route | Blocker |
|---|---|---|
| desktop/mobile | `extract_step3` | `.eu-step3-design-marker` / `.eu-step3-modules-cfg` not visible from current query-driven fallback state. |
| desktop/mobile | `extract_step4` | `.eu-step4-design-marker` / `.eu-step4-run-link` not visible from current query-driven fallback state. |
| desktop/mobile | `patient_loaded_time_series` | `.eu-qv-series-grid` not visible from current query-driven fallback state. |
| desktop/mobile | `patient_loaded_quality` | `.eu-qv-quality-card` / `.eu-qv-quality-note` not visible from current query-driven fallback state. |
| mobile | `guided_welcome` | Guided left rail is intentionally hidden/collapsed on mobile. |
| mobile | `shell_navigation` | Sidebar/topbar nav is intentionally hidden/collapsed on mobile. |

These blockers explain why Stage23B did not delete the requested 1500-2500 lines: the remaining high-specificity Extract/Patient/Guided/Shell blocks have live owners or need state-driving guards before deletion.

## Line counts

- Stage23B start: 55,766 total CSS lines.
- After accepted cleanup: 55,714 total CSS lines.
- Net Stage23B CSS reduction: 52 lines.
- Baseline reduction so far: 62,171 -> 55,714, net -6,457 lines.

Current largest CSS files:

| File | Lines |
|---|---:|
| `agent_overrides.css` | 14,196 |
| `guided_overrides.css` | 8,422 |
| `extract_overrides.css` | 6,641 |
| `patient_overrides.css` | 5,345 |
| `crossdb_overrides.css` | 3,435 |

## Validation

Commands run:

```bash
python tools/qa_legacy_streamlit_css_guard.py --base-url http://127.0.0.1:8513 --label after_stage23b_extract_patient_guided_shell --compare-before output/playwright/stage23_css_guard_20260624_150035_before_stage23b_guard_extension/computed_style_guard.json --routes extract extract_idle extract_step2 extract_step3 extract_step4 patient patient_idle patient_loaded_tables patient_loaded_time_series patient_loaded_overview patient_loaded_quality guided guided_welcome guided_study_workspace shell shell_navigation --no-screenshots
python -m py_compile tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py
ruff check tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py
python tools/inventory_legacy_streamlit_css.py --check-extract-owner-guards
python tools/inventory_legacy_streamlit_css.py --check-patient-owner-guards
python tools/inventory_legacy_streamlit_css.py --check-guided-owner-guards
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage23b_inventory_check
git diff --check -- src/easyicu/webapp/extract_overrides.css src/easyicu/webapp/guided_overrides.css src/easyicu/webapp/patient_overrides.css tools/qa_legacy_streamlit_css_guard.py
```

Results:

- Browser/computed-style guard: `failures=[]`.
- Extract owner scan: `issues=[]`, `unclassified_marker=0`.
- Patient owner scan: `issues=[]`.
- Guided owner scan: `issues=[]`, `unclassified_marker=0`.
- Inventory no-copy: 16 CSS files, 16 imported, 0 untracked, `delete_now=[]`, 55,714 lines.
- `py_compile`: passed.
- `ruff`: passed.
- `git diff --check`: passed.
- Provider dormant smoke: `ai_enabled=false`, `ready=false`, `client_constructed=false`, `network_calls=0`, `secrets_returned=false`.

## Status

Stage23B is blocked for large-line cleanup, not failed:

- The guard extension is useful and passing for the accepted cleanup batch.
- The 1500-line deletion target was not met.
- Remaining large candidates require deeper state-driving browser guards or source-owner proof before deletion.

## Next Action

Stage23C should unblock larger cleanup by driving or fixture-seeding:

1. Extract step3/step4 visible state.
2. Patient time-series and data-quality loaded panels.
3. Guided desktop/mobile rail/composer/study workspace computed-style baselines.
4. Shell mobile expected-hidden nav baseline, separating intentional mobile collapse from real missing owner.

Only after those guards are green should the next batch delete high-specificity Extract/Patient/Guided/Shell cascade locks.
