# FastAPI route domain split and composition-root cleanup

Date: 2026-07-10  
Branch: `fix/easyicu-concept-bounds-enforcement`  
Status: done

## Objective

Replace the monolithic native WebServer application module with explicit,
domain-owned FastAPI routers while preserving every HTTP method, path,
operation name, error contract, registration order, and local-first behavior.

## Result

- `src/easyicu/webserver/app.py` is now a 122-line composition root. It owns
  host middleware, ordered router registration, and the root static mount only.
- The pre-extraction baseline immediately before the first route-owner commit
  was 1,155 lines (`git show a301072^:src/easyicu/webserver/app.py | wc -l`).
- Route adapters now live under `src/easyicu/webserver/routes/` by owner:
  `system`, `local_data`, `reviews`, `extraction`, `workspaces`, `jobs`,
  `agent`, `guided`, `copilot`, `page_guide`, and `ideas`.
- Jobs uses two routers so submission remains before Agent control while job
  lifecycle/SSE remains after Agent artifact routes. `job_store.MANAGER` is
  resolved dynamically, preventing split-brain manager state in tests and
  runtime.
- Agent uses separate control and artifact routers. Its Idea-derived seed gate,
  provider gate, review/signoff behavior, and download headers remain intact.
- Provider configuration now validates `enable_ai` before writing credentials;
  an invalid boolean returns HTTP 400 without creating either provider or
  Settings files.

## Commit batches

| Commit | Scope |
|---|---|
| `a301072` | system/settings/capability routes |
| `7597537` | Guided routes |
| `8e81514` | Copilot and Page Guide routes |
| `f3d4bee` | Idea Mining routes and shared request parsing |
| `f4a6ea7` | local data and Workspace registry routes |
| `e7fd6a7` | Patient/Cohort/Cross-DB review and extraction filters |
| `ad5c4dd` | Job submission/lifecycle routes and SSE regression |
| `f283896` | Agent control/artifact routes |
| `3ccc590` | explicit application composition root and public route exports |
| `08e6a7a` | pre-write Provider opt-in validation |

## Compatibility contracts

`tests/test_webserver_route_contracts.py` now locks:

- exact HTTP method, path, and operation name for every extracted router;
- eager and lazy FastAPI `include_router` behavior via endpoint identity;
- owner presence and foreign-route absence;
- the ordered sequence from system routes through job/Agent routers;
- the root `StaticFiles` mount as the final route.

## Verification

Per-batch focused tests were run before and after each move. The final impacted
WebServer integration suite was:

```text
pytest -q \
  tests/test_webserver_route_contracts.py \
  tests/test_webserver_security_hardening.py \
  tests/test_webserver_idea_sources.py \
  tests/test_webserver_science_workbench.py \
  tests/test_webserver_workspace_summary.py

194 passed, 1 warning in 85.69s
```

Scoped Ruff checks, `py_compile`, and `git diff --check` also passed throughout.
The 3,000+ full repository suite was intentionally not repeated for each
mechanical batch; direct owner contracts and affected/adjacent behavior ran per
commit, followed by the 194-test cross-domain integration gate above.

## Remaining WebApp work

- Run desktop browser QA with the 94k-entity real export.
- Complete real six-database Cross-DB density/n×n verification and freeze the
  related Figure 3 caption/source data.
- Push remains a user decision; no commits in this sequence were pushed.

