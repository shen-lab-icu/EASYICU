# 2026-06-27 Cohort KM Survival Window Fix

## Scope

Addressed three manual-debug findings in FastAPI native Cohort Statistics survival analysis:

- Hospital mortality KM curve should not default to the full observed hospital LOS tail (>500 days). It now displays a bounded 30-day window by default; observations after the window are censored at day 30.
- Log-rank p-values should be shown as exact scientific notation instead of threshold text such as `p <0.001`.
- Outcome availability should be explicit. ICU mortality is marked unavailable only when the active export lacks ICU-specific event/time columns, while 28-day mortality can be derived from hospital death + hospital LOS when dedicated 28-day columns are absent.

## Changed Files

- `src/easyicu/webserver/cohort_review.py`
- `src/easyicu/webserver/static/js/screens-viz.js`
- `src/easyicu/webserver/static/index.html`
- `tests/test_webserver_workspace_summary.py`
- `tests/test_webserver_static_routes.py`

## Backend Evidence

Active export API smoke via `TestClient`:

```text
survival_status ready
outcome hospital_death ready 8762 30.0 None None
outcome icu_death blocked 0 None None ICU mortality is unavailable because this export does not include ICU-specific event and time columns.
outcome mort_28d ready 13255 28.0 hospital_mortality_time_window None
curve_window 30.0
risk_times [0.0, 1.0, 3.0, 7.0, 14.0, 28.0, 30.0]
p_value 8.744409560103507e-119
p_label 8.744 x 10^-119
```

The payload remains aggregate-only and does not expose patient rows.

## Browser Evidence

Verified in the in-app browser on `http://127.0.0.1:8786/?_v=cohort-km-window-20260627b#cohort` after loading the active real export and opening the Survival tab:

```json
{
  "hasKM": true,
  "kmTicks": ["0%", "25%", "50%", "75%", "100%", "0", "7.5", "15", "22.5", "30", "天", "生存概率"],
  "outcomeText": [
    "院内死亡\n8,762 事件 · 30 天显示窗口",
    "ICU 死亡\n不可用 · 缺少 ICU 事件/时间列",
    "28 天死亡\n13,255 事件 · 28 天窗口 · 由院内死亡 + 住院时长派生"
  ],
  "logrankText": "Log-rank\nχ² 536.94 · p = 8.74 × 10^-119\ndf 1 · 仅探索",
  "has515": false,
  "hasPThreshold": false,
  "hasScientificP": true,
  "has30Window": true,
  "has28Ready": true,
  "hasIcuUnavailableReason": true
}
```

Browser console errors: `[]`.

## Verification Commands

```bash
./.venv/bin/python -m py_compile src/easyicu/webserver/cohort_review.py
/Users/haibo/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node --check src/easyicu/webserver/static/js/screens-viz.js
./.venv/bin/ruff check src/easyicu/webserver/cohort_review.py tests/test_webserver_workspace_summary.py tests/test_webserver_static_routes.py
./.venv/bin/pytest -q tests/test_webserver_workspace_summary.py -k 'cohort_review'
./.venv/bin/pytest -q tests/test_webserver_static_routes.py -k 'cohort_comparison_radios or crossdb_restores_distribution_visuals or cohort_snapshot_renders_real_distribution_charts'
git diff --check
python3 EASYICU/tools/lint_main_plan.py
```

All checks passed.

## Notes

The live browser URL on port `8785` was backed by an older uvicorn process. The verified preview ran from the current checkout on port `8786`.
