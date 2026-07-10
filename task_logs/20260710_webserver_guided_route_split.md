# 2026-07-10 WebServer Guided route split

Task ID: `WEBAPP-FASTAPI-NATIVE-QA`.

## Scope

This is the first post-review FastAPI route extraction batch. It moves only the
nine `/api/guided/*` HTTP adapters and their single private response helper from
`webserver/app.py` to the explicit owner
`webserver/routes/guided.py`. `/api/copilot/*`, `/api/page-guide/*`, Ideas,
workspaces, jobs, and Agent routes are unchanged.

The batch is intentionally behavior-neutral:

- request methods, paths, handler names, signatures, status/error behavior, and
  OpenAPI operation names are preserved;
- tests continue to monkeypatch the canonical `guided_sessions` module object,
  rather than an alias on `app.py`;
- the Guided router remains registered between Agent history and Copilot routes;
- the root `StaticFiles` mount remains last;
- the route contract handles both eager FastAPI route expansion and the current
  lazy `_IncludedRouter` registration model.

`app.py` decreased from 1,075 to 993 lines and from 65 to 56 directly registered
handlers. The new owner file is 99 lines.

## Ownership regression

`tests/test_webserver_route_contracts.py` now locks:

- exact `(method, path, operation name)` for all nine Guided routes, including
  the POST/DELETE compatibility pair;
- registration presence, no duplicate/foreign routes, and contiguous ordering;
- `/api/guided/*` absent from `app.py` and present in `routes/guided.py`;
- `/api/copilot/*` and `/api/page-guide/*` absent from the Guided owner;
- no reverse import from the Guided owner to `webserver.app`;
- router ordering and final static mount invariants.

## Verification

```text
Pre-change focused baseline:
12 passed, 136 deselected

Final route/ownership contracts:
5 passed

Final Guided + adjacent Copilot/Page Guide backend, security, and frontend wiring:
22 passed, 192 deselected

Scoped ruff:
All checks passed

py_compile:
passed

git diff --check:
no output
```

The 3,222-test repository suite was not rerun. This physical route move is
covered by its owner contract and adjacent behavior/security/wiring tests; real
data, LLM, pipeline, and statistical suites are outside its impact boundary.

## Next route split

The next lowest-risk batch is the eight `/api/copilot/*` and
`/api/page-guide/*` compatibility handlers in their own owner module. Ideas and
workspaces should follow separately. Jobs and Agent routes remain last because
tests currently monkeypatch `app.MANAGER`, `app.dataio`, and `app.source_store`,
and those domains share seed/job helpers.
