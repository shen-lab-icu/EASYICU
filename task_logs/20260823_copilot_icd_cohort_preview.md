# Copilot ICD cohort preview — 2026-08-23

## Scope and isolation

- Worktree: `/Users/haibo/Documents/GitHub/EASYICU-copilot-data-preview-20260823`
- Branch: `codex/copilot-bounded-data-preview-20260823`
- Base: `d4f7990`
- `main` was not checked out, edited, merged, or pushed.
- Scope was narrowed after code audit: arbitrary feature existence/distribution was already owned by `easyicu_review_cohort`; this patch adds only the missing pre-extraction ICD cohort count/filter-funnel path.

## Owner boundary

- New public Data Extraction adapter: `easyicu.webserver.dataio.preview_export_cohort`.
- It calls the same `_resolve_export_cohort` used by actual extraction and discards patient/stay ids, id column, loader arguments, and host path.
- New Copilot read tool: `easyicu_preview_icd_cohort`.
- Tool arguments are path-free and bounded to one optional registered source id, 1–16 include prefixes, and 0–16 exclude prefixes.
- The bound StudyContext demographic/stay settings are preserved; only the ICD clauses are overlaid for the preview.
- Snapshot view: `icd_cohort_preview`, immutable and project scoped.
- Right preview links to the original `#extraction` workspace and passes a bounded cohort-size handoff.

## Defects found by real-data/browser QA

1. ICD lists were previously stringified as Python list syntax by `_split_icd_tokens`; list and string inputs now share the same normalization.
2. Parquet column projection could request `stay_id` twice, producing duplicate columns and a DataFrame where a Series was expected. `_read_table_columns` now de-duplicates requested columns while preserving order.
3. The first right-preview layout overflowed its 321 px desktop aside by 41 px. The funnel grid now has a zero-overflow narrow-container layout.

## Evidence

### Real MIMIC-IV demo probe

- Source: official local MIMIC-IV Clinical Database Demo v2.2 raw folder.
- Contract: all ICU stays, age 18–100, repeated stays allowed, ICD include prefix `A41`, no exclude prefix.
- Owner result: source denominator `140`, before ICD `140`, final cohort `10`, include matches `10`.
- Privacy receipt: patient ids false, raw rows false, host path false.

### Focused automated checks

Final command covered the new tool surface, path-free snapshot, real extraction ICD regression, native extraction handoff, route-owned assets, and existing Data Workbench contracts:

- `18 passed, 5 warnings`
- Python Ruff lint: pass.
- Python compile: pass.
- Node syntax (`main.mjs`, `event-projection.mjs`, `screens-viz-embedded.js`): pass.
- CSS owner/foreign-marker/brace/comment scan: pass.
- `git diff --check`: pass.

The broader pre-existing `tests/test_pi_copilot_static.py` snapshot has two unrelated baseline failures on base `d4f7990`: its expected Guided Pi export string omits the already-present `rebind` member, and its expected `guided-pi-preview.css` cache version predates the already-present extraction-workspace version. Neither owning file was changed by this patch; these stale assertions were not rewritten as part of the ICD scope.

### Browser QA

- Server: isolated `EASYICU_HOME`, `127.0.0.1:8769`, stopped after QA.
- Browser: Playwright CLI, 1440×1000, Chinese Guided Copilot route.
- Copilot right preview: body horizontal overflow `0`, aside horizontal overflow `0`, overflowing descendants `0`.
- Native handoff: click `打开完整数据提取` reached `#extraction`; banner carried `队列 10 ICU stays`.
- Extraction page horizontal overflow: body `0`, main `0`.
- Console on both views: `0 errors`, `0 warnings`.
- Screenshots:
  - `task_logs/browser_qa_20260823/copilot-icd-cohort-preview-1440x1000.png`
  - `task_logs/browser_qa_20260823/copilot-icd-handoff-data-extraction-1440x1000.png`

The browser render used the exact real-data aggregate payload mounted through the production embedded owner in the actual Guided Copilot layout. A fresh formal-provider model-selection canary was not rerun because the isolated QA state intentionally contained no copied credential; host-tool schema, prompt routing, dispatch, projection, data owner, and browser resource contracts were verified directly.

