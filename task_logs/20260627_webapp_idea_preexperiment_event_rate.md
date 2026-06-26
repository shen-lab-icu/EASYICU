# 2026-06-27 · Idea Mining pre-experiment event-rate fix

## Problem

The Idea Mining pre-experiment panel reported boolean/event concepts such as `sep3_sofa2` as coverage and missingness. For sparse event tables, this is clinically wrong: absent rows usually mean event-negative patients, not missing observations. In the real UI this made Sepsis-3 positives look like `41.8% coverage / 58.2% missing`, which could mislead downstream feasibility decisions.

## Fix

- `src/easyicu/webserver/ideas/mining.py`
  - Boolean concepts from the concept dictionary now use `metric_kind="event_rate"`.
  - Event rows report `event_entities`, `non_event_entities`, `event_rate_pct`, and `missing_pct=None`.
  - Event rows are excluded from low-coverage feasibility-risk counting.
  - The pre-experiment interpretation explicitly states that boolean/event indicators are positive rates and negative patients are not missing.
- `src/easyicu/webserver/static/js/screens-ideas.js`
  - Idea Mining cards render event indicators as event rate, events, and non-events instead of coverage/missingness.
- `src/easyicu/webserver/static/js/screens-guided.js`
  - Guided Copilot embedded pre-experiment uses the same event-rate wording.
- `src/easyicu/webserver/static/css/ideas.css`
  - Added event-rate bar styling inside the Idea Mining owner stylesheet.
- `tests/test_webserver_workspace_summary.py`
  - Regression test asserts `sep3_sofa2` is `metric_kind="event_rate"`, with `event_entities=1`, `non_event_entities=2`, `denominator_entities=3`, `event_rate_pct=33.3`, `missing_pct=None`, and `low_coverage=False`.

## Verification

- `./.venv/bin/python -m pytest -q tests/test_webserver_workspace_summary.py -k "idea_mining_web_preserves_vasopressor_fluid_strategy_concept_set or idea_mining_web_run_creates_ledger_preexperiment"`
  Result: `2 passed, 105 deselected`.
- `./.venv/bin/python -m pytest -q tests/test_webserver_workspace_summary.py tests/test_webserver_idea_sources.py -k "idea"`
  Result: `14 passed, 102 deselected`.
- `/Users/haibo/.nvm/versions/node/v24.11.0/bin/node --check src/easyicu/webserver/static/js/screens-ideas.js`
- `/Users/haibo/.nvm/versions/node/v24.11.0/bin/node --check src/easyicu/webserver/static/js/screens-guided.js`
- `./.venv/bin/python -m compileall -q src/easyicu/webserver/ideas src/easyicu/webserver`

## Boundary

No commit was made. Existing unrelated working-tree changes were left untouched.
