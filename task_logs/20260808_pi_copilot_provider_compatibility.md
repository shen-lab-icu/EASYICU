# Pi Copilot provider compatibility and CLIProxyAPI login diagnosis

- Date: 2026-08-08
- Branch: `feat/pi-copilot-shell`
- Commit: `c97dbff` (`fix(web): support provider-aware Pi setup`)
- Owner: `easyicu.webserver.pi_copilot` + shared credential URL policy

## User-visible failure

The reported setup page was opened as
`file:///.../static/index.html#guided`. That static preview had no FastAPI
backend, so the browser failed before a request could reach CLIProxyAPI and
surfaced the misleading generic message `Failed to fetch`.

Read-only host probes established that `cli-proxy` was listening on
`127.0.0.1:8317` and an unauthenticated `GET /v1/models` returned HTTP 401.
This proves the local service was reachable and enforcing authentication; no
user credential was used by the diagnostic run.

## Fix

1. `file://` is now an explicit static-preview state. Provider submission is
   disabled and the page directs the user to
   `http://127.0.0.1:8765/#guided`.
2. The UI separates service/provider choice from the wire protocol. It offers
   CLIProxyAPI/local proxy, other OpenAI-compatible gateways, OpenAI,
   Anthropic, and Google Gemini presets.
3. The governed Pi sidecar and FastAPI contract now accept all four custom
   provider protocols documented by the pinned Pi package:
   `openai-completions`, `openai-responses`, `anthropic-messages`, and
   `google-generative-ai`.
4. Verification uses protocol-specific authentication and catalog shapes:
   Bearer + `data[].id` for OpenAI-compatible services, `x-api-key` for
   Anthropic, and `x-goog-api-key` + `models[].name` for Google.
5. A model-name mismatch returns a bounded, secret-filtered list of model IDs
   reported by the service. Failed verification still writes no credential.
6. The shared URL policy permits the RFC 2544 `198.18.0.0/15` Fake-IP range
   only for exact official provider hostnames over HTTPS, preserving the
   private-network fail-closed rule for arbitrary endpoints.

## Verification

- `166 passed` across Pi contracts, install/gateway/provider/routes/static,
  Web static/route contracts, and provider privacy boundaries.
- Ruff, Node syntax, npm build, and `git diff --check` passed.
- Browser QA passed at 1440×900 and 1024×768 with zero document overflow.
- Browser interaction confirmed that selecting Anthropic updates provider ID,
  endpoint, transport, and model together.
- No real credential or real model prompt was used. The final authenticated
  CLIProxyAPI canary remains a user-performed step in the real WebApp.

## Upstream contracts checked

- Pi multi-provider API and the four custom-provider protocols:
  <https://github.com/earendil-works/pi/blob/main/packages/ai/README.md>
- CLIProxyAPI model registry and `/v1/models` publication:
  <https://github.com/router-for-me/CLIProxyAPI/blob/main/docs/sdk-advanced.md>
