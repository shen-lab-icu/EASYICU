# EasyICU Pi Copilot gateway

This private sidecar embeds the upstream Pi `AgentSession` SDK for the native
EasyICU WebApp. It is not a general coding agent and does not expose Pi's
built-in filesystem or shell tools.

## Pin and attribution

- upstream: <https://github.com/earendil-works/pi>
- reviewed source commit: `9dd90a49711d088b86fdd9b4aea575913a8328a8`
- npm package: `@earendil-works/pi-coding-agent@0.84.1`
- upstream license: MIT (see `THIRD_PARTY_NOTICES.md`)

The lockfile is authoritative. From an installed EasyICU wheel, create the
private versioned runtime explicitly:

```sh
easyicu copilot install
```

The installer copies only packaged runtime files and runs
`npm ci --ignore-scripts`; Web server startup never installs dependencies.

## Runtime configuration

The sidecar is launched by EasyICU, not by the browser. The normal product path
is Guided Copilot's first-use setup: EasyICU verifies `/models`, stores the
credential in `~/.easyicu/pi-provider.env` with mode `0600`, and opens chat only
while the matching verification receipt remains valid. The key is submitted
once and is never returned to the page or browser storage.

Operators can alternatively provide process-environment overrides:

```sh
export EASYICU_PI_API_KEY='...'
export EASYICU_PI_BASE_URL='http://127.0.0.1:8317/v1'
export EASYICU_PI_MODEL='gpt-5.6-luna'
```

Optional variables:

- `EASYICU_PI_PROVIDER` (default `easyicu-local`)
- `EASYICU_PI_API` (`openai-completions` or `openai-responses`)
- `EASYICU_PI_CONTEXT_WINDOW` (default `200000`)
- `EASYICU_PI_MAX_TOKENS` (default `16384`)
- `EASYICU_PI_SESSION_TOKEN_BUDGET` (default `1000000`; hard session stop)
- `EASYICU_PI_CWD` (normally a private empty workspace supplied by the host)
- `EASYICU_PI_SESSION_DIR` (normally supplied by the Python host)

The shell provider is independent of any provider selected for an EasyICU
scientific run. The child receives only a strict runtime/`EASYICU_PI_*`
environment allowlist; credentials are never stored in this package or
returned over the gateway protocol. Environment overrides must still match a
local verification receipt before the chat gate opens. Node `>=22.19.0` is
required.

The shell registers fifteen EasyICU-only tools. Study setup can be collected
and saved inside the conversation with a one-message Configure grant; the
existing typed StudyContext store remains authoritative. Generic Pi
filesystem, editing, network, and shell tools stay disabled.
Raw model reasoning is forced off and is not streamed or returned in session
transcripts.

See `docs/pi_copilot_integration_architecture.md` for authority, PHI, session,
failure, and upgrade contracts.
