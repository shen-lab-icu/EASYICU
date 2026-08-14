# EasyICU Pi Copilot gateway

This private sidecar embeds the upstream Pi `AgentSession` SDK for the native
EasyICU WebApp. Research conversations expose only governed EasyICU science
tools. Workspace conversations additionally expose host-governed project
artifact tools for bounded read, write, edit, static checking, and web preview;
Pi's unsandboxed built-in filesystem and shell tools remain disabled.

## Pin and attribution

- upstream: <https://github.com/earendil-works/pi>
- reviewed source commit: `9dd90a49711d088b86fdd9b4aea575913a8328a8`
- npm package: `@earendil-works/pi-coding-agent@0.84.1`
- upstream license: MIT (see `THIRD_PARTY_NOTICES.md`)

The lockfile is authoritative. From an installed EasyICU wheel, create the
private content-addressed runtime explicitly:

```sh
easyicu copilot install
```

The installer copies only packaged runtime files and runs
`npm ci --ignore-scripts`; Web server startup never installs dependencies.
`runtime-manifest.json` records SHA-256 digests for the packaged executable
files and exact Pi package versions, and the host revalidates them before it
starts Node.

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
- `EASYICU_PI_SESSION_TOKEN_BUDGET` (default `1000000`; provider-call checked cumulative ceiling)
- `EASYICU_PI_CWD` (normally a private empty workspace supplied by the host)
- `EASYICU_PI_MAX_PROVIDER_CALLS_PER_MESSAGE` (default `8`)
- `EASYICU_PI_MAX_PROVIDER_CALLS_PER_SESSION` (default `128`)
- `EASYICU_PI_INPUT_PRICE_USD_PER_1M_TOKENS`
- `EASYICU_PI_OUTPUT_PRICE_USD_PER_1M_TOKENS`
- `EASYICU_PI_MAX_COST_USD_PER_MESSAGE`
- `EASYICU_PI_MAX_COST_USD_PER_SESSION`
- `EASYICU_PI_SESSION_DIR` (normally supplied by the Python host)

The four pricing/cost variables are an all-or-none contract. When a provider
publishes reliable prices, the shell conservatively reserves the maximum input
and output cost before every call, persists the cumulative reservation, and
fails closed at either cost ceiling. Existing sessions cannot silently adopt a
new or changed pricing contract. Local proxies that do not publish trustworthy
pricing leave all four unset; the token and call ceilings remain the hard-stop
fallback and the UI reports that dollar pricing is unavailable.

The shell provider is independent of any provider selected for an EasyICU
scientific run. The child receives only a strict runtime/`EASYICU_PI_*`
environment allowlist; credentials are never stored in this package or
returned over the gateway protocol. Environment overrides must still match a
local verification receipt before the chat gate opens. Node `>=22.19.0` is
required.

The shell registers fifteen EasyICU research tools. Study setup can be collected
and saved inside the conversation with a one-message Configure grant; the
existing typed StudyContext store remains authoritative. Workspace mode adds
seven governed artifact tools plus the packaged `web-prototype` skill. Writes
require a reusable host-held capability for that message and remain inside the
project-specific private workspace. Generic Pi filesystem, network, and shell
tools stay disabled. The
private path is the AgentSession's logical workspace; it is not an
operating-system sandbox for the Node process itself.
Raw model reasoning is forced off and is not streamed or returned in session
transcripts.

See `docs/pi_copilot_integration_architecture.md` for authority, PHI, session,
failure, and upgrade contracts.
