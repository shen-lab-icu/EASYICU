# Codex account as one Copilot and Research Agent model connection

Date: 2026-08-16

Branch: `codex/figure2-e1-h3-20260816`

## Outcome

New Pi Copilot sessions use one immutable model connection for both the
conversation shell and the Research Agent analysis handoff. A user can select a
browser-bound Codex account and an allowed Codex model once; the same account,
model, and binding digest are then used by both paths. API-backed sessions retain
the same one-connection contract.

The implementation does not merge the two runtime authorities. Pi remains the
conversation/orchestration owner, while the Research Agent remains the only
owner of scientific execution, numeric evidence, and publication gates.

Existing schema-v1 sessions are not silently migrated. They remain readable and
are projected as legacy conversation/analysis bindings. Newly created sessions
use schema v2 and expose one safe `model_connection` projection.

## Security boundary

- The browser Codex App Server remains the login, refresh, and logout owner.
- The Pi child receives only the exact private auth-file coordinate for the
  frozen browser account; tokens are not returned to the browser or persisted in
  the Pi session record.
- The auth file must be absolute, regular, non-symlinked, private, bounded in
  size, and backed by an OAuth access token that is not near expiration.
- Account-backed Pi children remove API-key environment variables and fail
  closed if the account, model, binding digest, or runtime installation drifts.
- One native Pi gateway is pooled per immutable account/model binding.

## Verification

Focused tests:

```text
195 passed, 2 skipped
```

Scope:

```text
tests/test_pi_copilot_contract.py
tests/test_pi_copilot_gateway.py
tests/test_pi_copilot_codex_gateway.py
tests/test_webserver_codex_account_sessions.py
tests/test_pi_copilot_routes.py
tests/test_webserver_static_routes.py
```

Additional checks:

```text
ruff: passed
node --check: passed
git diff --check: passed
secret-pattern scan of the scoped diff: clean
```

Live Chrome smoke used an isolated EasyICU home and the browser-bound Codex
account with `gpt-5.6-luna`. A new schema-v2 session displayed one
`codex · gpt-5.6-luna` connection and returned the requested text in 6.6 seconds
without tool use. The 2560x1262 desktop viewport had no horizontal overflow.

This proves account login, model routing, and the unified conversation path. It
does not prove Figure 2 E1 completion: no Planner, Execute, evidence audit,
figure, manuscript, or publication gate ran in this smoke.

## Next gate

Resume the E1 minimum canary from this exact source identity. Fix only the owner
responsible for any progressive Planner convergence failure. Do not start E2
until E1 completes all required stages and gates.
