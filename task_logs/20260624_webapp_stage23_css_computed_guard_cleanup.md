# 2026-06-24 WebApp Stage23 CSS computed-style guard cleanup

## Scope

Stage23 changed the legacy Streamlit fallback CSS cleanup method from static owner
scan only to browser/computed-style guarded cleanup. The FastAPI native
webserver was not modified. No CSS file was deleted and `shell_styles.py` import
order was not changed.

Files intentionally changed in this batch:

- `src/easyicu/webapp/agent_overrides.css`
- `src/easyicu/webapp/extract_overrides.css`
- `src/easyicu/webapp/guided_overrides.css`
- `src/easyicu/webapp/patient_overrides.css`
- `src/easyicu/webapp/shell_navigation_overrides.css`
- `tools/qa_legacy_streamlit_css_guard.py`

## Browser/computed-style guard

New guard tool:

```bash
python tools/qa_legacy_streamlit_css_guard.py --base-url http://127.0.0.1:8513 --label before_stage23_agent_subviews
python tools/qa_legacy_streamlit_css_guard.py --base-url http://127.0.0.1:8513 --label after_stage23_prop_subset_recovered --compare-before output/playwright/stage23_css_guard_20260624_143504_before_stage23_agent_subviews/computed_style_guard.json
```

The guard records desktop `1440x900` and mobile `393x852` snapshots for Extract,
Patient, Guided, Shell, and Agent subviews. It stores root overflow, visible
offscreen/clipped counts, console errors, required selectors, key computed styles,
and screenshots.

Accepted guard reports:

| Label | JSON | Result |
|---|---|---|
| before | `output/playwright/stage23_css_guard_20260624_143504_before_stage23_agent_subviews/computed_style_guard.json` | `failures=[]` |
| after accepted cleanup | `output/playwright/stage23_css_guard_20260624_144657_after_stage23_prop_subset_recovered/computed_style_guard.json` | `failures=[]` |

Rejected experiments that were restored:

| Attempt | JSON | Reason |
|---|---|---|
| Guided lines 1-2742 large trim | `output/playwright/stage23_css_guard_20260624_142607_after_stage23_guided_trim/computed_style_guard.json` | Guided desktop/mobile computed-style guard failed |
| Guided lines 2743-6171 trim | `output/playwright/stage23_css_guard_20260624_142735_after_stage23_guided_2743_6171/computed_style_guard.json` | Guided desktop/mobile computed-style guard failed |
| broader property-subset cascade trim | `output/playwright/stage23_css_guard_20260624_144502_after_stage23_property_subset_cascade/computed_style_guard.json` | Agent mobile, Agent history/mobile, Agent outputs/mobile, Agent summary/mobile, and Shell mobile computed-style signatures changed |

## Cleanup result

Only confirmed stale or browser-guarded duplicate cascade was kept:

- Removed stale Agent `_eu_ra_tabs` / `_eu_ra_view_` blocks. Current Agent markup
  uses the project tab route (`eu_agent_project_tabs` / `ag-tabs`); owner scan and
  browser subview guard remained clean.
- Removed exact duplicate or guarded subset cascade in Extract, Patient, Guided,
  and Shell navigation where the browser guard showed no key computed-style change.
- Kept all Guided large layout blocks that failed browser guard.
- Restored Agent/Shell property-subset removals that changed mobile signatures.

Line counts:

| File | Before | After | Net |
|---|---:|---:|---:|
| `agent_overrides.css` | 14,390 | 14,196 | -194 |
| `guided_overrides.css` | 9,022 | 8,461 | -561 |
| `extract_overrides.css` | 6,733 | 6,647 | -86 |
| `patient_overrides.css` | 5,529 | 5,352 | -177 |
| `shell_navigation_overrides.css` | 2,554 | 2,548 | -6 |
| **All imported CSS** | **56,790** | **55,766** | **-1,024** |

The requested 2,000-line deletion target was not safely reached. This batch is a
guarded partial cleanup, not a completed Stage23 target.

Current largest CSS files:

| Rank | File | Lines |
|---:|---|---:|
| 1 | `agent_overrides.css` | 14,196 |
| 2 | `guided_overrides.css` | 8,461 |
| 3 | `extract_overrides.css` | 6,647 |
| 4 | `patient_overrides.css` | 5,352 |
| 5 | `crossdb_overrides.css` | 3,435 |

## Validation

Commands run:

```bash
python -m py_compile tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py
ruff check tools/inventory_legacy_streamlit_css.py tools/qa_legacy_streamlit_css_guard.py
python tools/inventory_legacy_streamlit_css.py --check-agent-owner-guards
python tools/inventory_legacy_streamlit_css.py --check-extract-owner-guards
python tools/inventory_legacy_streamlit_css.py --check-patient-owner-guards
python tools/inventory_legacy_streamlit_css.py --check-guided-owner-guards
python tools/inventory_legacy_streamlit_css.py --no-copy --out-root /tmp/easyicu_stage23_current_inventory_check
git diff --check -- src/easyicu/webapp/agent_overrides.css src/easyicu/webapp/extract_overrides.css src/easyicu/webapp/guided_overrides.css src/easyicu/webapp/patient_overrides.css src/easyicu/webapp/shell_navigation_overrides.css tools/qa_legacy_streamlit_css_guard.py tools/inventory_legacy_streamlit_css.py
```

Results:

- `py_compile`: passed.
- `ruff check`: passed after removing one unused import.
- owner scans: `issues=[]` for Agent, Extract, Patient, and Guided.
- inventory no-copy: 16 CSS files, all imported, untracked imported CSS count `0`,
  `delete_now=[]`, total lines `55,766`.
- targeted `git diff --check`: passed.
- browser/computed-style guard after accepted cleanup: `failures=[]`.

Provider dormant smoke:

```json
{"ai_enabled": false, "ready": false, "client_constructed": false, "network_calls": 0, "secrets_returned": false}
```

## Blockers and next action

The next 1,000+ lines cannot be removed by selector-name/static-owner scan alone.
Known blockers:

- Guided large cascade blocks still have live computed-style ownership; two
  large trim attempts failed browser guard.
- Agent and Shell mobile cascade has real fallback behavior; property-subset
  deletion changed mobile signatures and was restored.
- Extract and Patient remaining large blocks need stateful page guards for later
  extraction steps and loaded patient subpanels before deleting high-specificity
  layout locks.

Next action: Stage23B should extend the guard to cover more stateful Extract
steps, Patient loaded/drilldown subviews, Guided rail/composer computed-style
contracts, and Agent/Shell mobile navigation before deleting any additional
high-specificity cascade.
