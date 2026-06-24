# Stage23D legacy CSS stateful-guard cleanup

Date: 2026-06-24

## Scope

- Goal: use the Stage23C browser/computed-style guard to perform real legacy Streamlit CSS cleanup.
- Changed CSS only:
  - `src/easyicu/webapp/extract_overrides.css`
  - `src/easyicu/webapp/guided_overrides.css`
- Did not modify FastAPI webserver code, `shell_styles.py`, old Streamlit Python pages, dirty tests, provider code, or data artifacts.

## Accepted deletions

| File | Removed lines | Original block | Evidence |
|---|---:|---|---|
| `extract_overrides.css` | 152 | `Moved from classic_viz_overrides.css: extraction final cascade lock` / `DATA EXTRACTION - final cascade lock` | Extract step2/step3/step4/export preview compare passed: `output/playwright/stage23_css_guard_20260624_161849_stage23d_after_extract_final_lock_probe/computed_style_guard.json`, `failures=[]`. |
| `extract_overrides.css` | 200 | `Data Extraction default viewport parity pass` plus `extraction responsive default viewport` | Extract step2/step3/step4/export preview compare passed: `output/playwright/stage23_css_guard_20260624_162917_stage23d_after_extract_default_viewport_probe/computed_style_guard.json`, `failures=[]`. |
| `guided_overrides.css` | 160 | `Guided Copilot - r3/r4/r5/r6 residual fixes` | Guided welcome/study workspace compare passed: `output/playwright/stage23_css_guard_20260624_162133_stage23d_after_guided_r3_r6_probe/computed_style_guard.json`, `failures=[]`. |

Total accepted deletion: 512 CSS lines.

CSS inventory after cleanup: 55,202 total lines across 16 imported CSS files.

Largest files after cleanup:

| File | Lines |
|---|---:|
| `agent_overrides.css` | 14,196 |
| `guided_overrides.css` | 8,262 |
| `extract_overrides.css` | 6,289 |
| `patient_overrides.css` | 5,345 |
| `crossdb_overrides.css` | 3,435 |

## Failed candidates rolled back

| Candidate | Result | Reason |
|---|---|---|
| `patient_overrides.css` `Patient Review loaded Data Tables numeric migration` through secondary-tab EOF locks | Rolled back | Patient loaded tables/time-series/overview/quality desktop+mobile computed-style signatures changed. The block is live. Report: `output/playwright/stage23_css_guard_20260624_161513_stage23d_after_patient_loaded_migration/computed_style_guard.json`. |
| `guided_overrides.css` `Guided Copilot - residual component metrics from guided.css` | Rolled back | Guided welcome/study workspace desktop+mobile computed-style signatures changed. Report: `output/playwright/stage23_css_guard_20260624_162348_stage23d_after_guided_residual_metrics_probe/computed_style_guard.json`. |
| `guided_overrides.css` `Guided Copilot - final non-geometric style normalization` | Rolled back | Guided welcome/study workspace desktop+mobile computed-style signatures changed. Report: `output/playwright/stage23_css_guard_20260624_162630_stage23d_after_guided_final_normalization_probe/computed_style_guard.json`. |

An initial full after compare against the first before baseline reported Patient Quality differences caused by optional Plotly/DataFrame mount timing, while root overflow and CSS files unrelated to Patient were unchanged. To avoid accepting a flaky baseline, the final evidence used a rerun workflow: reverse the accepted CSS patch, collect a fresh full before baseline, reapply the same patch, then compare full after against that fresh baseline.

## Browser guard evidence

Final full before baseline:

- `output/playwright/stage23_css_guard_20260624_163314_stage23d_before_rerun_full/computed_style_guard.json`
- routes: `extract_step2`, `extract_step3`, `extract_step4`, `extract_export_preview`, `patient_loaded_tables`, `patient_loaded_time_series`, `patient_loaded_overview`, `patient_loaded_quality`, `guided_welcome`, `guided_study_workspace`, `shell_navigation`
- viewports: desktop 1440x900 and mobile 393x852
- result: `failures=[]`

Final full after compare:

- `output/playwright/stage23_css_guard_20260624_163524_stage23d_after_rerun_full/computed_style_guard.json`
- same routes and viewports
- result: `failures=[]`

## Validation

Commands passed:

```bash
python tools/inventory_legacy_streamlit_css.py --check-extract-owner-guards
python tools/inventory_legacy_streamlit_css.py --check-patient-owner-guards
python tools/inventory_legacy_streamlit_css.py --check-guided-owner-guards
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage23d_inventory_check
python -m py_compile tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py
ruff check tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py
git diff --check
```

Owner scan results:

- Extract: `issues=[]`, `confirmed_stale_source_missing_class=0`
- Patient: `issues=[]`, `confirmed_stale_source_missing_class=0`
- Guided: `issues=[]`, `confirmed_stale_selector_or_comment=0`

Inventory result:

- `/tmp/easyicu_stage23d_inventory_check/inventory_20260624_163735`
- `files=16`, `imported=16`, `untracked=0`, `delete_now=[]`, `lines=55202`

Provider dormant smoke:

```text
ai_enabled=false
ready=false
client_constructed=false
network_calls=0
secrets_returned=false
```

## Remaining blockers

- Patient loaded Data Tables migration/EOF locks are live and cannot be deleted without a more precise replacement or broader computed-style guard.
- Guided residual component metrics and final non-geometric style normalization are live. Deleting them changes guided welcome/study workspace desktop and mobile signatures.
- Export-progress and floating launcher cleanup remain blocked until dedicated fixture hooks can prove before/after style parity.
- Large remaining files are now dominated by live owner locks, not simple source-missing selectors.

## Next action

Stage23E should continue guard-backed cleanup, but should first add more targeted per-component guard signatures for:

- Extract completion/export-progress state.
- Guided terminal/alignment lock internals so safe sub-blocks can be isolated instead of deleting whole residual sections.
- Patient loaded quality/table/time-series panel replacement owners.
- Shell floating launcher and mobile nav intentional-collapse comparisons.
