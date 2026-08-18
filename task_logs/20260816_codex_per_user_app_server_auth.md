# Codex per-user App Server authentication

Date: 2026-08-16
Task: expose a user-owned Codex/ChatGPT sign-in alongside API providers in the native EasyICU Web Research Agent.
Branch: `feat/figure2-dev9-heldout27-20260815`
Starting HEAD: `95d23ac`

## Outcome

The public Web `codex` provider no longer inherits or probes the server operator's Codex CLI login. Each browser receives an opaque HttpOnly cookie that binds to an isolated server-side `HOME` and `CODEX_HOME`. Authentication is owned by the official Codex App Server device-code flow, while the Research Agent run, EvidenceStore, scientific validation, human review, and publication gates remain unchanged.

API-key providers remain available through the same provider panel and run workflow. Claude remains an Anthropic API provider; no Claude CLI account login was added.

## Owner boundaries

- `research_agent/providers/codex_app_server.py`: minimal initialized JSON-RPC stdio transport for the official Codex App Server protocol.
- `webserver/codex_account_sessions.py`: per-browser cookie, isolated filesystem coordinates, device login/status/cancel/logout, account masking, and owned child-process shutdown.
- `research_agent/providers/llm.py::CodexAppServerLLMClient`: session-bound text/strict-schema generation through `thread/start` and `turn/start`; no tool, workspace, or network authority is granted to a model turn.
- `research_agent/providers/factory.py`: exact provider, model, endpoint, session digest, and subprocess-environment authorization receipt.
- `webserver/provider_adapter.py`: the only bridge from the Web provider identity to the Research Agent client.
- `webserver/routes/agent.py` and the Agent provider owner JS files: browser auth endpoints, run binding, status, and UI controls.

## Security and fail-closed behavior

- Missing or forged cookies never start App Server and never inspect host auth.
- Two browser sessions receive different cookies, session digests, and `CODEX_HOME` paths.
- API keys for OpenAI, DeepSeek, and unrelated providers are removed from the App Server subprocess environment.
- App Server initialization must report the exact isolated `codexHome`; mismatch fails closed.
- Session directories must be private real directories. Symlinked child directories are rejected before a provider process can start.
- Device verification links are accepted only for `https://auth.openai.com/codex/device`.
- Account email is masked in UI metadata; tokens, raw email, cookies, and user codes are not written to research artifacts or provider receipts.
- A paused run records the non-secret session binding and cannot be resumed from a different browser account session.
- Logout/cancel and concurrent-tab state changes are serialized per user session.

## Verification

Focused aggregate matrix:

```text
289 passed, 5 warnings in 10.73s
```

The matrix covered the App Server client, provider factory/authorization, legacy internal CLI boundary, Web account sessions, Web provider portability, durable review recovery, route contracts, native static frontend contracts, and Pi static contracts.

Additional directly invoked route checks:

```text
12 passed, 98 deselected in 5.04s
```

Static quality checks:

- Ruff passed for every changed Python source/test file.
- `node --check` passed for `api.js`, `screens-agent-provider.js`, and `screens-agent.js`.
- `git diff --check` passed.

Live desktop-browser QA used a temporary isolated `EASYICU_HOME` and the exact worktree source on `127.0.0.1:8877`:

- Codex selection showed a per-user sign-in button and explicitly stated that server-operator auth is not used.
- An official device-code ceremony started successfully and returned the allowlisted OpenAI URL; cancel returned to the signed-out state.
- Before authentication, both provider scaffold and Planner canary remained disabled.
- 1280 x 720: no horizontal overflow, login control not clipped, and no console warning/error.
- The temporary device ceremony was cancelled and the temporary server stopped.

## Remaining user action and claim boundary

No real ChatGPT account was authorized during automated QA, because completing that OAuth/device grant is a user-controlled security action. Therefore this task proves live App Server startup plus login/start/status/cancel and all mocked generation contracts, but it does not yet claim a successful real-account Planner/model turn or a speed advantage over API providers. The next interactive smoke test is: the user clicks **Sign in with Codex**, completes the official OpenAI page, then runs one bounded provider scaffold or Planner canary and compares its measured receipt with the API route.

Official protocol reference: <https://learn.chatgpt.com/docs/app-server>
