# Codex browser auth and GPT-5.6 Luna runtime verification

Date: 2026-08-16

Task: make a user's Codex/ChatGPT account an alternative Research Agent
provider inside Copilot, while preserving the existing API-provider path and
EasyICU scientific authority boundaries.

## Implemented boundary

- Provider selection now belongs to the Copilot conversation rather than the
  Agent Projects screen. The browser can select a configured API provider or a
  browser-isolated Codex account before freezing an immutable
  `ResearchProviderBinding` for a scientific run.
- Codex browser authentication is scoped by an HttpOnly session cookie. Each
  browser session receives an isolated `HOME`/`CODEX_HOME`; the operator's
  Codex login, API keys, raw token files, and host paths are not projected to
  the browser or stored in scientific artifacts.
- Account logout, missing session state, a stale account directory, a model
  outside the reviewed catalog, or a provider/binding mismatch fails closed
  before a Research Agent call.
- API providers remain supported. Codex account auth is an additional
  transport credential, not a replacement for provider-independent typed
  validation, EvidenceStore, sandbox execution, numeric host authority, or
  publication gates.
- The provider UI has a dedicated Copilot owner module
  (`screens-guided-pi-provider.js`). No route-specific provider workflow was
  added to a catch-all shell file.

## Managed runtime resolution

The existing PATH runtime reported `codex-cli 0.139.0`. Authentication and
`thread/start(model="gpt-5.6-luna")` succeeded, but the first complete turn was
rejected by the service because that Codex runtime was too old.

The App Server owner now resolves the executable in this order:

1. an explicit, validated `EASYICU_CODEX_EXECUTABLE` absolute path;
2. the Codex runtime bundled by the official ChatGPT macOS application;
3. the reviewed PATH fallback.

An invalid explicit override fails closed. The override crosses only the Codex
subprocess boundary; unrelated API-key environment variables remain excluded.
On this host, the selected development runtime is the ChatGPT application
bundle (`codex-cli 0.148.0-alpha.9`).

## Real bounded verification

A real call was made through `CodexAppServerLLMClient` using the authenticated,
browser-isolated account session and the normal provider authorization factory.
The request used a strict JSON schema requiring `{"ready": true}`.

Observed receipt:

- runtime source: `chatgpt_app_bundle`
- runtime: `codex-cli 0.148.0-alpha.9`
- requested and actual model: `gpt-5.6-luna`
- generation completed: yes
- strict structured output validated: yes
- provider usage record persisted: yes
- raw credential material printed or persisted in this log: no

This proves the development account path can perform a governed generation; it
does not make a Figure 2 run scientifically successful or formal-ready.

## Focused verification

```text
332 passed, 5 warnings in 3.25s
ruff: all checks passed
git diff --check: passed
```

The test matrix covers the App Server account client and runtime resolver,
browser-session isolation, provider binding contracts, Copilot research
workflow, route contracts, static asset ownership/wiring, and UX fail-closed
behavior.

## Remaining formal boundary

The current ChatGPT application bundle is a mutable development dependency and
the App Server interface is still treated as a development integration. Before
any formal Figure 2 experiment, freeze the exact Codex runtime version/digest,
model identifier, provider receipt schema, source HEAD, and exact runner image.
The next development action is to combine this account transport with the
Progressive Planner branch, rebuild the exact-head runner image, and run a fresh
E1 canary. E1 must complete its governed 11-stage path before E2 begins.
