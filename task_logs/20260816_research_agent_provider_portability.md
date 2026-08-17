# Research Agent provider portability and bounded latency probe

Date: 2026-08-16
Task: `FIG2-DEV9-HELDOUT27`
Authority: development-only; not paper-authorized

## Trigger

E1 Dev29 used exact pushed source `8bbd8be` and reached one Progressive
Planner strict-schema request. The 61,951-byte request reserved 20,000 output
tokens but returned no response before the 900-second deadline. The run closed
with `APITimeoutError`, made no structured retry, and never entered Execute.
That observation is transport-only: it neither validates nor falsifies the
Dev28 compiler containment repair.

The immediate engineering question was whether EasyICU could select another
provider without relabelling every OpenAI-compatible endpoint as `openai`,
losing provenance, leaking credentials across providers, or silently assuming
strict JSON Schema support.

## Owner and public contract

`research_agent.providers.capabilities` now owns the dependency-neutral
immutable provider profiles alongside capability discovery. Provider identity
is separate from wire protocol. The reviewed profile registry contains:

- `openai`;
- `openrouter`;
- `deepseek`;
- `custom`, for an operator-configured OpenAI Chat Completions-compatible
  gateway or local server;
- `anthropic`, using the native Anthropic Messages API rather than the
  OpenAI-compatibility layer.

Each profile owns the ordered API-key, base-URL, and model environment
coordinates plus its reviewed default endpoint. Strict JSON Schema remains an
explicit run-bound transport capability and is never inferred from the
provider or model name.

This closes the reviewed API family across the Research Agent CLI, replication
CLI, MCP schema, discovery launcher, benchmark launchers, Web Research Agent
bridge, provider readiness/configuration, authorization manifest, and
`build_llm_client` selection. The Anthropic adapter moves system text to the
native top-level field, maps strict contracts to `output_config.format`, binds
the exact Messages endpoint into authorization receipts, and treats provider
refusal as terminal. Gemini's native wire format is not implemented; a
reviewed OpenAI-compatible gateway remains available through `custom`.

The only user-facing account-backed transport is `codex`, with exact provider
identity `codex-cli` and endpoint `cli://codex`. The current local Web server
uses a bounded non-interactive Codex process behind the provider adapter, so a
Web user does not need to operate a CLI. It verifies `codex login status`, maps
a host-derived strict schema to `codex exec --output-schema`, and binds a full
pipeline run to the typed `local_account` credential source. Claude is exposed
only as the native `anthropic` API provider; Claude Code and Gemini CLI are not
selectable in the Research Agent CLI, replication CLI, MCP, benchmark, or Web
catalog.

The profile registry and CLI adapter reuse the existing capability and
provider owners rather than adding a new module. The architecture diff
therefore remains at the pre-existing Progressive Planner debt of 524 modules
versus the 519-module baseline; this change does not raise it to 525 and does
not refresh the lower-is-better baseline.

## Security and diagnostic boundaries

- Provider-specific credentials take precedence over the generic fallback and
  are not copied into another provider's environment snapshot.
- Official DeepSeek always uses `Authorization`; a stale local-Luna
  `x-api-key` setting cannot alter its request or provenance receipt.
- `custom` requires a server-owned endpoint and credential. A per-call remote
  URL override remains forbidden.
- Every real provider still passes the canonical external-LLM opt-in gate.
- The capability probe uses synthetic JSON only, makes at most two calls per
  invocation, disables retries and streaming, and writes no prompt, response,
  credential, clinical data, or patient data.
- Private env input must be a non-symlink regular file with mode 0600 or
  stricter. The report destination also rejects a final-component symlink and
  is written atomically at mode 0600.
- Requested and provider-returned model identities are recorded separately so
  a relay fallback cannot be mistaken for the requested model.
- The Codex account transport receives a frozen allowlist environment. It
  retains only the cache coordinates needed for login and explicitly excludes
  API keys, so it cannot silently change from account use to API billing.
- Each Codex account call runs in a private temporary directory with a
  read-only sandbox. It uses exactly one non-streaming physical attempt and
  never falls back to another provider.
- The local Web path may use only the machine owner's account session. A hosted
  multi-user deployment must create an isolated login session per user and
  must never read, copy, serialize or upload the server operator's CLI auth
  cache.
- Other vendors remain API-backed. A new native protocol must receive its own
  reviewed adapter; compatible gateways can use `custom` without being
  relabelled as OpenAI.

## Verification

The final provider/account/MCP/Web/benchmark focused matrix completed:

```text
325 passed, 5 warnings in 40.43s
```

It covered provider protocol and catalog boundaries, authorization identity,
CLI and replication opt-in, MCP schemas/transport, discovery launcher
security, execution identity, LLM structured transport and retry behavior,
Provider budget/hard-stop enforcement, value/privacy and endpoint controls,
Web provider configuration, the native Anthropic SDK contract, Codex account
isolation, public exclusion of Claude/Gemini account modes, and the frozen
Figure 2 provider protocol probe. The discovery, replication, provider hard
stop, execution-identity, and authority neighborhood separately completed
`180 passed, 1 skipped`. After the concurrent execution/figure refactor became
syntactically stable, the broader Agent/Web regression matrix completed
`469 passed, 10 warnings`. Ruff, the provider-scoped `git diff --check`, and
both modified JavaScript files' Node syntax checks passed.

The native browser could not repeat the final visual pass because its local-URL
policy blocked the temporary 127.0.0.1 port. The FastAPI app started and served
the new static resources over HTTP, and the route/owner tests verify the six
provider buttons, script wiring, absence of Claude/Gemini account choices, and
the explicit Codex-account versus Claude-API copy. This task therefore does not
claim a fresh screenshot or clipping measurement; the immediately preceding
provider-panel browser pass at 1440 x 900 remains the visual baseline.

## Live development-only comparison

All requests used the same synthetic exact-JSON contract, no retries,
non-streaming transport, and a requested output ceiling of 20,000 tokens. The
five-sample latency comparison interleaved provider order. Report:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/provider_comparison_20260816/provider_latency_comparison.json`

| Route | Requested / returned model | Passed | Median latency | Range |
|---|---|---:|---:|---:|
| local Luna gateway | `gpt-5.6-luna` / `gpt-5.6-luna` | 5/5 | 3.063 s | 2.558–3.737 s |
| official DeepSeek API | `deepseek-v4-flash` / `deepseek-v4-flash` | 5/5 | 0.766 s | 0.652–0.903 s |
| ZyAI relay (`custom`) | `deepseek-v4-flash` / `deepseek-v4-flash` | 5/5 | 1.394 s | 1.149–2.659 s |

For this small request only, official DeepSeek was about 4.0 times as fast as
local Luna and 1.8 times as fast as the relay; the relay was about 2.2 times as
fast as local Luna. Provider tokenization differs, so token counts are not a
cross-model throughput measure.

The separate two-call capability receipts are under the same external root.
Their first request explicitly used
`response_format={"type":"json_object"}` rather than relying on prompt-only
JSON. Luna, official DeepSeek, and the relay all passed that JSON-object request
and the host's exact typed-value check. The corresponding one-call latencies
were 3.775 s, 1.372 s, and 2.985 s; these individual observations do not replace
the five-sample comparison above. Luna also passed OpenAI strict JSON Schema.
Official DeepSeek and the relay rejected the OpenAI `json_schema` response
format with `BadRequestError`; both therefore record
`strict_json_schema=false` rather than silently degrading the same request.
DeepSeek's documented Beta strict function-call route was also probed with a
minimal closed schema on `deepseek-v4-flash` and returned HTTP 400. That single
probe does not prove every Beta schema/model combination is unavailable, but
it is sufficient to keep that unverified transport disabled for Dev30.

These are small synthetic transport probes, not a 61,951-byte Planner stress
test and not evidence that E1 can complete. The relay's returned model string
is provider-declared provenance, not independent proof that its weights or
serving configuration are identical to the official endpoint. Every artifact
remains development-only and `paper_authorized=false`.

### Account-backed Codex capability probe

This machine had `codex-cli 0.139.0` and `codex login status` confirmed a
ChatGPT account session. A requested `gpt-5.6-luna` call failed before inference
because that locally installed CLI reported that the model requires a newer
CLI; the failure was recorded as model/CLI incompatibility, not authentication
failure. The installed CLI's default resolved to `gpt-5.5` and completed, after
which the bounded two-call probe was repeated with explicit `gpt-5.5`.

Report:

`/Volumes/外置硬盘/easyicu_data/figure2_dev9_runs/provider_comparison_20260816/codex_account_capability_probe.json`

| Route | Model | Host JSON | Strict JSON Schema | Latencies |
|---|---|---:|---:|---:|
| Codex account | `gpt-5.5` | pass | pass | 13.113 s / 11.954 s |

For this tiny probe, account-backed Codex was substantially slower than the
official DeepSeek API; its demonstrated advantages are no API key, an existing
ChatGPT login and native strict-schema output, not lower single-call latency.
Each Codex call currently starts a fresh `codex exec` process, so a future
per-user App Server session may reduce process startup and repeated-context
overhead without changing scientific authority.

No Claude Code or Gemini CLI capability or latency claim is made. Claude is
supported through the native Anthropic API adapter, which was verified against
the installed SDK contract with a no-network transport double; no live
Anthropic credential was used for this development change.

## Next gate

Commit and push the provider portability boundary, build the exact-source
image, and run E1 Dev30 on official DeepSeek with `progressive_v2` and the
strict-schema flag explicitly off. Host parsing, typed compilation, suffix
repair, Provider budgets, EvidenceStore, and publication gates remain intact.
This is a new development transport policy, not a continuation that may reuse
Dev29 state. E2 remains blocked until E1 completes analysis, audit, figure, and
report.
