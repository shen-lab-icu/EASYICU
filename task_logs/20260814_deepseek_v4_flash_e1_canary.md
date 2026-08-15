# DeepSeek V4 Flash Web E1 Planner canary — 2026-08-14

## Scope

- Development-only provider compatibility check; no formal Canonical9 batch.
- Endpoint: `https://api.zyai.online/v1`.
- Requested and reported model: `deepseek-v4-flash`.
- Credentials remained in the private local 0600 provider store and were not
  written to Git, artifacts, prompts, or this log.

## Provider verification

- `/v1/models`: HTTP 200; the requested model was present.
- Minimal Chat Completions JSON probe: HTTP 200, strict JSON returned.
- EasyICU governed `OpenAIClient`: constructed and authorized successfully;
  model provenance and call-scoped usage were returned; `secrets_returned=false`.

## Fresh Web E1 Planner canary

- StudyContext:
  `e1-deepseek-v4-flash-canary-20260814-2da0829`.
- Web job: `04d00d3a8b45`.
- Pipeline run: `run_20260815T014652_c11dbc`.
- Execution profile: server-owned `planner_canary`; execution remained blocked.

The run successfully completed provider authorization, typed data-foundation
materialization, runtime validation, ResearchContext construction, and the
initial cohort audit. All five bounded Planner drafts were rejected. Every
Planner response ended at exactly 8,192 completion tokens with
`finish_reason=length`; attempts 2, 4 and 5 were invalid/truncated JSON, while
attempts 1 and 3 reached the article-contract validator but still failed the
same digest-bound violation. The terminal result was
`research_pipeline_plan_contract_exhausted`; no analysis was executed.

Safe hard-stop receipt:

- provider attempts: 6 (1 acquisition + 5 Planner);
- accounted tokens: 130,971;
- conservative generic cost estimate: USD 2.16833 (not a reviewed bill for
  this provider);
- patient rows, prompts, raw model output, and secrets recorded: false.

## Adjudication

`deepseek-v4-flash` is transport-compatible with EasyICU but is not currently
Planner-compatible under the frozen 8,192-token structured-plan contract.
Do not widen the shared Planner output limit or weaken scientific validators to
rescue this model during E1. Keep the result as a development compatibility
failure and use the already proven `gpt-5.6-luna` path for the next E1 gate.

## Adjacent Web startup defect

Starting the current Web app exposed an unrelated stale-recovery regression:
`Path.iterdir()` raised only during lazy iteration when a historical seed
pointed to a deleted temporary pipeline directory. Commit `573c24d` now
materializes the bounded directory scan inside the `OSError` boundary. The
full owner test file passed (`17 passed`), Ruff passed, and the Web server then
started successfully on `127.0.0.1:8765`.
