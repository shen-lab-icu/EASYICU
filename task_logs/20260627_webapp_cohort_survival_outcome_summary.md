# 2026-06-27 Cohort Survival Outcome Summary Fix

## Context

User feedback on the Cohort Statistics survival panel:

- Hospital mortality, ICU mortality, and 28-day mortality were rendered as selectable KM endpoint buttons.
- Hospital and ICU mortality are event-rate dimensions, not fixed time-window controls.
- ICU mortality must be counted if a registered export carries an ICU-specific event column; otherwise it must be explicitly unavailable, not silently blocked or inferred.
- 28-day mortality can drive the KM time window; event summaries should show count and percentage.

## Changes

- Backend: `src/easyicu/webserver/cohort_review.py`
  - Added outcome-level `event_summary` separate from KM/log-rank readiness.
  - Default survival endpoint now prefers `mort_28d` when available.
  - ICU mortality event rate is allowed when an ICU-specific event column exists, even if KM/log-rank is blocked due to missing ICU time columns.
  - Current active export is not allowed to fake ICU mortality from hospital death or ICU LOS.

- Frontend: `src/easyicu/webserver/static/js/screens-viz.js`
  - Replaced clickable mortality endpoint buttons with read-only outcome overview cards.
  - Removed `data-cohort-surv-outcome` button contract.
  - KM section now uses the default ready endpoint, normally `mort_28d`.
  - Added Chinese/English copy for unavailable ICU mortality and event-rate-only outcomes.

- CSS: `src/easyicu/webserver/static/css/cohort.css`
  - Added owner-scoped outcome overview card layout.

- Cache bust: `src/easyicu/webserver/static/index.html`
  - Bumped `cohort.css` and `screens-viz.js` asset versions to `20260627-survival-outcome-summary`.

- Tests:
  - `tests/test_webserver_workspace_summary.py`
  - `tests/test_webserver_static_routes.py`

## Evidence

Commands:

```bash
./.venv/bin/python -m py_compile src/easyicu/webserver/cohort_review.py
./.venv/bin/python -m compileall -q src/easyicu/webserver
find src/easyicu/webserver/static/js -name '*.js' -print0 | xargs -0 -n1 /Users/haibo/.nvm/versions/node/v24.11.0/bin/node --check
./.venv/bin/python -m pytest -q tests/test_webserver_workspace_summary.py -k 'cohort_review_summary_uses_active_source_without_row_payload or cohort_review_icu_death_event_rate_does_not_require_km_time or cohort_review_large_parquet_export_reuses_active_source_for_km_and_coverage'
./.venv/bin/python -m pytest -q tests/test_webserver_static_routes.py -k 'cohort'
./.venv/bin/python -m pytest -q tests/test_webserver_static_routes.py tests/test_webserver_workspace_summary.py tests/test_webserver_provider_tools.py
./.venv/bin/python -m ruff check src/easyicu/webserver/cohort_review.py tests/test_webserver_workspace_summary.py tests/test_webserver_static_routes.py
git diff --check
```

Results:

- Focused backend cohort tests: 3 passed.
- Focused static cohort tests: 6 passed.
- Broader webserver focused tests: 156 passed, 1 warning.
- Node syntax checks passed.
- Ruff passed on touched Python/test files.
- `git diff --check` passed.

API smoke against the active registered export:

```text
default_outcome: mort_28d
hospital_death: event_count=9466, denominator=94458, event_rate_pct=10.0, km_status=ready
icu_death: event_summary.status=missing, reason=ICU-specific event column is not present in the registered export.
mort_28d: event_count=14218, denominator=94458, event_rate_pct=15.1, km_status=ready
curve_outcomes: ["hospital_death", "mort_28d"]
row_markers: false
```

Browser DOM smoke on the Cohort survival tab:

```json
{
  "cards": [
    "院内死亡 10% 9,466 / 94,458 事件 事件率",
    "ICU 死亡 不可用 当前注册导出没有 ICU 专用死亡事件列。",
    "28 天死亡 15.1% 14,218 / 94,458 事件 KM 曲线结局 · 28 天窗口"
  ],
  "oldButtons": 0,
  "title": "28 天死亡 · 按 Sepsis vs 非 Sepsis 分组",
  "logrank": "Log-rank χ² 353.33 · p = 7.977 × 10^-79 df 1 · 仅探索",
  "overflowX": 0,
  "consoleErrors": []
}
```

## Notes

The current active export contains `death`, `los_icu`, `los_hosp`, `mort_28d`, `mort_90d`, and `mort_365d`, but does not contain an ICU-specific mortality event column such as `icu_death`, `death_icu`, or `icu_mortality`. ICU mortality is therefore shown as unavailable instead of being inferred from hospital mortality.
