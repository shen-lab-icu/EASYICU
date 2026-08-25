# Extraction query cancellation

Date: 2026-08-24

Candidate: `codex/easyicu-unified-product-20260823` in `/Users/haibo/Documents/GitHub/EASYICU-unified-product-20260823`.

## User-visible failure

An all-patient MIMIC-IV extraction remained in the running state after the user requested cancellation. The UI truthfully recorded `cancel_requested`, but the backend checked that flag only between feature modules. A large DuckDB vitals aggregation therefore continued until the entire query returned.

## Runtime evidence and safety response

- The live job `0f353210f4e9` accepted `POST /api/jobs/{job_id}/cancel` and stored `cancel_requested=true`.
- Demographics had completed; the worker was inside the next DuckDB aggregation.
- DuckDB also reported `No space left on device` while spilling query state to its temporary directory.
- The already-cancelled live worker was stopped without deleting export files. DuckDB removed its own process-temporary spill files during shutdown, recovering approximately 3.2 GiB. No user dataset or export directory was deleted.

## Repair

- `Job` now owns a small cancel-callback contract. Callbacks run once on the first accepted cancellation, can be unregistered, run immediately if registration loses the cancellation race, and cannot block one another if one callback fails.
- The datasource owner exposes the current worker thread's DuckDB `interrupt()` callback and maps only DuckDB's explicit `InterruptException` to `DuckDBQueryInterrupted`. IO failures such as disk exhaustion remain real failures.
- The extraction runner registers the DuckDB interrupt only around the active concept load. An accepted cancellation breaks the module loop, preserves completed files, omits the incomplete manifest, and finishes the job as `cancelled`.
- The concept batch loaders re-raise the typed interruption instead of silently falling back to a second large read.
- The UI now says `正在停止抽取… / 已接受取消，正在停止当前数据库查询…` until the terminal cancelled event arrives.

## Verification

- Focused Python matrix: `28 passed`.
- Includes a real cross-thread DuckDB long-query interruption test, typed exception classification, callback race/error contracts, partial-export cancellation, adjacent bucket-loader regressions, job continuity, and cache-token/UI-copy contracts.
- Ruff, Python compile, JavaScript syntax, and `git diff --check` passed.
- Live isolated worktree service on `127.0.0.1:8897`: `/` 200, `/api/catalog` 200, unknown output job 404, and served index contains `20260824-query-cancel1`.
- In-app browser reload restored the existing Guided Copilot conversation and cleared the stale running card; the page title is `研究引导 — EasyICU` at `http://127.0.0.1:8897/#guided`.

## Scope boundary

This validates cancellation mechanics and local UI continuity. It does not validate the scientific content of a full MIMIC-IV extraction, all feature modules, or release readiness. The change remains isolated from `main`.
