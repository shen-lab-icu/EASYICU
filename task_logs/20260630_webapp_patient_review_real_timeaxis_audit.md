# 2026-06-30 Patient Review legacy-vs-native audit + real time-axis fix

## Scope

Requested as an **audit-first, minimal-fix** pass: confirm the FastAPI native
Patient Review actually carries the useful semantics of the deleted Streamlit
Patient Review, verify demo + real states in a real browser (not code-only),
and fix any concrete gap.

## 1. Legacy logic recovered from git history

Source of truth: `git show d1da22e^:src/easyicu/webapp/patient_page.py`
(removed in `d1da22e "Remove legacy Streamlit WebApp package"`). The old page
had **three** `selectbox` view modes (not four):

- **Dashboard / 综合仪表盘** — summary card (demographics, LOS, death, vent,
  rrt, vaso flags) + compact trend panels + SOFA organ-contribution stacked
  trajectory over `Time (hours from ICU admission)` + organ-domain comparison.
- **Category View / 分类视图** — 10 clinical categories rendered as
  latest-value + trend + delta signal grids: Vital Signs, SOFA Score
  (`decimals=0` → integer), Sepsis-3 Status (binary status tiles, `include_chart=False`,
  explicitly *not* missingness), Laboratory Tests, Blood Gas, Vasopressors
  (`decimals=3` dose traces, `vaso_ind` boolean-max), Respiratory Support
  (vent flags), Neurological (GCS tone-coded), Renal Function, Other Scores.
- **Data Table / 数据表格** — per-concept expander, patient rows in a dataframe.

Key legacy semantics the user wanted preserved: integer total scores,
hour-based time axis, non-negative dose traces, sepsis/event flags as status
not missingness, clinical-domain grouping.

## 2. Current FastAPI native state (kept / migrated)

The four current tabs (数据表 / 时间序列 / 患者概览 / 数据质量) map onto and
extend the legacy three modes. Verified in-browser on a fresh server
(`easyicu-native`, port 8502) in **demo** mode:

- **数据表 (Data Tables)** — kept FastAPI strengths: cohort attrition flow
  (72→56→48→48), module pills with row counts, bounded pseudonymous previews,
  pagination (上一页/下一页), wide tables scroll horizontally
  (`table-scroll` scrollWidth 1108–1760 > clientWidth 1026), page
  `overflowX = 0`. Same patient `demo_ent_1` shown across charttimes
  `0.2, 1, 2, 3.4, 6, 8, 12` with integer SOFA-2 components.
- **时间序列 (Time Series)** — legacy intent restored: 3 modes (临床泳道 /
  单患者 / 多患者对比), module-grouped mini-charts, axis `0.2h→48h`, units +
  reference lines, no `t0/t1`. Multi-patient mode = same-feature traces across
  5 pseudonymous entities; exact value matrix demoted to a collapsed audit view.
- **患者概览 (Patient Overview)** — rich case profile (not age+LOS): header
  age/sex 58/M, SOFA-2 max 6, Sepsis-3 Positive, outcome Survived, ICU LOS
  4.2d; Vitals + Labs snapshots with deltas; integer scores; non-negative
  therapy doses; 19-module availability ledger.
- **数据质量 (Data Quality)** — coverage/missingness ranking + physiologic
  range + duplicate-time integrity, **plus** a separate "事件/暴露发生率 ·
  不是缺失率" panel where Sepsis-3/ventilation/vasopressors appear as
  prevalence (93/90/87%), not flagged as bad missingness.

Demo realism (user's prior partial edits in `screens-viz-demo.js`) confirmed
correct and kept as-is: `DEMO_CHART_HOURS = [0.2,1,2,3.4,6,8,12,18,24,30,36,48]`,
integer score normalization, non-negative dose/rate clamp, boolean handling,
"种子观测值 / seeded demo" labelling.

## 3. Concrete gap found and fixed (real-data path)

Audit of the **real** export path exposed one real regression the demo path
hid: `_time_lane_payloads()` in `src/easyicu/webserver/patient_drilldown.py`
sorted by `charttime` but **never emitted the time values**. Real lane signals
came back with `values` but no `times` key, so the front-end
(`screens-viz-patient-series.js`, which reads `sig.times` for the
"Time since ICU admission (hours)" axis) had no hours to render for real data —
only the demo path (which fabricates `times`) showed the hour axis.

Real exports store `charttime` as hours-from-admission floats
(e.g. `-1.0, 0.0, 1.0 … 987.0`), so the fix is to carry them through.

Fix: in `_time_lane_payloads`, pair `charttime` with each numeric value
row-by-row (same NaN-drop rule as `dataio._numeric_values`, so `times` stays
aligned with `values`) and emit `"times": times[:_MAX_SIGNAL_POINTS]`.

Result (real `/Users/haibo/easyicu/exports/miiv`, `demo=false`):
`hr` signal now returns `times: [0,1,2,…,11]` aligned with
`values: [109,103,…,113]`. In-browser, real loaded export Time Series renders
the `0h → 13h` axis on every vital chart (screenshot-confirmed).

This is a **backend-only** change. No static JS/CSS was modified (the
front-end was already `times`-aware), so no `index.html` cache-bust was needed.

## 4. Files changed

- `src/easyicu/webserver/patient_drilldown.py` — `_time_lane_payloads` now
  emits charttime-aligned `times` per signal.
- `tests/test_webserver_workspace_summary.py` — extended
  `test_patient_review_drilldown_uses_active_source_with_bounded_table_previews`
  to lock that the vitals `hr` lane signal carries `times` aligned with values.

## 5. Verification

- `node --check` on all four patient JS files — ok.
- `pytest -q tests/test_webserver_static_routes.py -k patient` → 2 passed.
- `pytest -q tests/test_webserver_workspace_summary.py -k patient` → 9 passed.
- `git diff --check` → clean.
- Browser (fresh `easyicu-native` :8502):
  - Demo mode: all 4 tabs verified (see section 2), console errors none.
  - Real mode: source picker lists real exports; loaded a real export
    (`demo=false`), Time Series renders real `0h→13h` hour axis. Screenshot
    captured.

## 6. Not done / remaining (honest)

- Only the smallest real export was exercised end-to-end in the browser; the
  94k-entity / 98M-row Guided full export on the external drive was not loaded
  (slow mount). The fixed code path is shared, so behavior should hold, but a
  large-export browser pass is still open.
- `screens-viz.js` remains ~5.5k lines (well over the 1,500-line budget). Not
  touched here to keep this a minimal fix; the patient renderers already live
  in their own owner files. A future split of `screens-viz.js` is still owed.
- `tools/qa_native_fastapi_patient_drilldown.py` still asserts the old Data
  Tables gate title and needs updating to the bounded-preview contract
  (carried over from earlier log, not addressed here).
