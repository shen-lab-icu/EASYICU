# Pi-based Copilot integration architecture

Status: accepted for the first Web-first vertical slice

Date: 2026-08-08

EasyICU baseline: `a9610bf5dea50dfced80eba804959f8ba0e086f9`

Pi package: `@earendil-works/pi-coding-agent@0.84.1`

Pi source reviewed at: `9dd90a49711d088b86fdd9b4aea575913a8328a8`

## Decision

EasyICU will use Pi's official `AgentSession` SDK as the conversational shell
for Guided Copilot. Pi is adapted to EasyICU; the scientific engine is not
redesigned around Pi.

The first slice is Web-first. A pinned Node sidecar owns Pi sessions and model
streaming. The existing local FastAPI process owns authentication gates,
study/run bindings, PHI-safe projections, action authorization, EasyICU tool
execution, and the browser API. The existing JobManager carries progress and
cancellation to the browser over its established SSE contract.

```text
Guided Copilot Web client
          |
          | typed local HTTP + existing job SSE
          v
FastAPI Pi Copilot gateway owner
  - opt-in and atomically consumed one-use action grants
  - authority binding and per-tool stale-session check
  - PHI-safe projections
          |
          | strict JSON-lines, local stdio
          v
Pinned Pi AgentSession sidecar
  - conversation/session/compaction
  - model streaming/tool loop
  - EasyICU custom tools only
          |
          | host tool requests over the same JSON-lines channel
          v
Existing EasyICU owners
  StudyContext / capabilities / JobManager / agent_runs / evidence artefacts
```

## What this replaces

The Pi surface replaces the free-form conversational shell inside the Guided
Copilot route: assistant turn streaming, tool-call presentation, session
history, cancellation, hidden-reasoning model turns, and eventual compaction or
fork controls. Conversational setup is persisted by
`easyicu_update_study_context`, which delegates to the existing typed
StudyContext owner and requires a host-held one-turn Configure grant. The
existing setup cards remain an optional review/edit surface, not a required
detour from the Pi conversation.

The legacy deterministic Guided conversation remains available as a bounded
fallback while the Pi path is disabled or unconfigured. A fallback is always
labelled as legacy/local; it must never be presented as an active Pi session.

## What remains untouched

Pi does not replace or gain direct access to:

- `ResearchContext` construction;
- Planner or current-plan authority;
- the internal Coder/repair loop;
- execution phases or DockerRunner policy;
- EvidenceStore internals, evidence gates, or publication readiness;
- provider hard stops, budgets, receipts, or scientific run identity;
- benchmark/manuscript authority.

Version 1 calls existing public Web/agent owners. Phase 2 may evaluate a more
agentic step loop, but it is explicitly outside this change.

## Why Pi

Pi already provides a stateful tool-calling agent, event streaming, persistent
JSONL sessions, model/thinking controls, compaction, and SDK embedding. Reusing
that maintained harness removes the need for EasyICU to grow another general
chat/session/tool runtime. EasyICU keeps only the clinical and scientific
adapter code that is specific to this product.

## SDK versus extension versus fork

The sidecar uses the official SDK (`createAgentSession`, `SessionManager`,
custom tools). It supplies a discovery-free ResourceLoader and an explicit
allowlist of EasyICU tools. This is more controllable than exposing Pi's raw RPC
surface, which includes generic shell commands, and fits the existing FastAPI
Web host better than a second standalone website.

No Pi fork is used. The isolated Node app and its transitive Pi packages are
pinned under `src/easyicu/webserver/pi_copilot/node_app/`, which also lets the
Python wheel carry the sidecar manifest, lockfile, entrypoint, and attribution.
A fork is considered only if a future documented requirement cannot be met by
the official SDK/extension contracts.

## Python to TypeScript mechanism

FastAPI launches one long-lived local Node subprocess. Requests, responses,
events, and host-tool calls use newline-delimited JSON with:

- `protocol_version`;
- `kind`;
- `request_id` (and `parent_request_id` for tool calls);
- strict method/field validation;
- stable error codes;
- bounded event payloads.

Unknown methods, unknown fields, malformed JSON, oversized lines, stale
session bindings, and unrecognized tool names fail closed. Tracebacks and
credentials are never protocol payloads. The subprocess inherits no scientific
authority: every host-tool call is re-authorized and executed by FastAPI.

## Tool permission model

Pi is created with an explicit EasyICU custom-tool allowlist. Built-in
`read`, `write`, `edit`, `bash`, `grep`, `find`, and `ls` are inactive. The
sidecar has no generic bridge method for arbitrary paths, commands, HTTP calls,
Python functions, or EvidenceStore mutation.

Read tools receive a session authority binding and return only host-produced
projections. Mutating tools require a one-turn capability supplied by the Web
request. The capability is held by FastAPI and is not a tool argument, so the
model cannot grant it to itself. Each named grant has exactly one atomic
consumption; a second same-action call returns `pi_action_grant_consumed`.
Every tool call revalidates the bound study revision, active job, and current
run. A successful authority mutation invalidates the rest of that turn until
the user explicitly rebinds. The first slice permits only the existing
deterministic local preflight submission and cooperative cancellation through
their current owners. Full provider runs, scientific crash-resume, and replan
requests return stable blocked codes instead of inventing a second
implementation or bypassing their dedicated confirmations.

The registered tool surface is exactly:

- read/inspect: `easyicu_workspace_status`, `easyicu_inspect_context`,
  `easyicu_inspect_plan`, `easyicu_inspect_capability`, `easyicu_inspect_run`,
  `easyicu_inspect_step`, `easyicu_inspect_validation`,
  `easyicu_list_artifacts`, `easyicu_inspect_evidence`, and
  `easyicu_explain_blocker`;
- typed setup: `easyicu_update_study_context`;
- control: `easyicu_run`, `easyicu_resume`, `easyicu_cancel`, and
  `easyicu_request_replan`.

There is no `easyicu_create_plan` in version 1 because the current Web layer
does not expose a public Planner authority contract. Adding a second planner
through Pi would violate the purpose of this integration.

## PHI and patient-data boundary

Model-visible tools never return patient rows, identifiers, timestamps, notes,
credentials, source paths, or raw files. Study data sources are represented by
type/database plus a 32-hex one-way path correlation digest; that digest is
never an authority or integrity receipt. Run and evidence tools return bounded
status, gate codes, aggregate counts, artifact names/digests, and concise
summaries from existing public artefacts.

Every owner builds a semantic allowlist rather than forwarding a generic
readiness/job object. The complete model-visible result, including summary and
authority fields, then passes a recursive value scan for absolute paths,
credential shapes, row identifiers, and size limits. Job labels and free-form
reasons are replaced with stable codes.

Both inbound chat text and outbound tool payloads pass a fail-closed marker
scan. The scan is defense in depth, not de-identification. The Web UI also tells
users not to paste row-level data; real patient data remains in the existing
local extraction and execution owners.

When conversational setup binds a data source, the model supplies only
`bind_active_export=true`. FastAPI resolves the active local path and gives it
to StudyContext internally; the Pi result receives only the existing one-way
path digest.

## Scientific authority boundary

The Pi session is UX state only. The authoritative identity remains EasyICU's
study context revision, run/job id, plan/evidence receipts, and persisted run
artefacts. A Pi session stores a binding to those values, never a replacement
copy. Scientific actions flow through existing EasyICU entry points and retain
their opt-in, capability, sandbox, gate, and evidence behavior.

Research projects are also authoritative namespaces, not browser labels. The
FastAPI host persists a one-to-one `project_id -> StudyContext.id` mapping. A
new project receives its own typed StudyContext; it never inherits whichever
context happens to be globally active. Every list/open/message/rebind/abort
operation carries `project_id`, and the host verifies both session ownership
and the project-to-study mapping before opening Pi or executing a tool. The
browser and model cannot select an arbitrary StudyContext id.

Tool results expose two surfaces: a concise model/user summary and bounded
machine details with an owner and stable code. A successful chat turn never
implies a successful scientific run.

## Session and resume semantics

Pi JSONL sessions are stored under EasyICU's private local state directory.
EasyICU stores a separate bounded metadata record linking each UI session to a
Pi session file and one scientific authority binding.

On reopen, FastAPI validates that the session file is inside the dedicated Pi
session root, then asks Pi to open it. Before every new prompt it reloads the
authoritative StudyContext. The binding uses **current-run semantics**: creation
or rebind records the latest run id, and a newer run makes the session stale.
A revision/job/run mismatch before a prompt or host tool returns
`pi_session_authority_stale`; the user must explicitly rebind before continuing.
The metadata/JSONL retention ceiling is 100 sessions. When an older record is
evicted, FastAPI best-effort disposes it and deletes only a `.jsonl` proven to
be inside the private Pi session root.

“Resume” has two distinct meanings:

- resuming the Pi conversation is supported by the persisted Pi JSONL file;
- resuming an in-progress Web job means reattaching to its existing JobManager
  id;
- crash-resuming the scientific pipeline is not exposed until an existing
  EasyICU owner provides a public resumable contract. The tool returns
  `scientific_resume_not_supported` rather than reconstructing one.

## Provider and cost implications

The Pi shell model and the EasyICU scientific-run provider are separate
authorities. Pi configuration is owned by the Pi boundary and is never copied
into a scientific run. On first use, the Guided route shows a model-service
setup gate instead of a chat composer. The browser submits the credential once
to local FastAPI; FastAPI validates the destination, calls the bounded
OpenAI-compatible `/models` endpoint without redirects, requires the selected
model to be present, and only then writes the configuration plus a matching
verification receipt under `~/.easyicu/` with mode `0600`. A failed
verification writes neither file. Changing any credential or endpoint field
invalidates the receipt and closes the chat gate again.

`EASYICU_PI_*` process variables remain an operator override and take
precedence over the private file. They do not bypass verification: their
fingerprint must match the local verification receipt before chat can open.
Starting a Pi session still requires both the server-wide EasyICU AI opt-in
and a per-session external-LLM opt-in before subprocess startup.

The local OpenAI-compatible defaults are `http://127.0.0.1:8317/v1` and model
`gpt5.6 luna`. The API key is never written to the repository, browser storage,
session JSONL metadata, logs, verification receipt, responses, or tool results;
only the dedicated private local credential file contains it. Automated tests
use injected transports and never make a paid provider call.

The Node child receives a strict environment allowlist containing only basic
process locale/runtime variables and `EASYICU_PI_*` shell configuration. It
does not inherit scientific-provider, search, cloud, or database credentials.
Its AgentSession uses a private empty logical workspace outside the repository;
this is not an OS sandbox for the Node process. Runtime status also requires
Node `>=22.19.0`.

Raw reasoning is forced off and is neither streamed nor returned in session
transcripts. Shell usage is separate from scientific ledgers. Before every
provider request, the shell checks cumulative provider-reported tokens plus a
conservative UTF-8 input-token upper bound and reserved output against
`EASYICU_PI_SESSION_TOKEN_BUDGET` (default 1,000,000). Hidden provider retries
are disabled, and separate per-message/per-session provider-call ceilings
default to 8/128. Provider prices are not inferred: `shell_usage.cost` is null
and `pricing_available=false` until a trusted rate table exists.

A wheel never installs Node dependencies during server startup. Installation
is an explicit user action using the packaged exact lockfile:

```sh
easyicu copilot install
```

This creates a private content-addressed runtime and executes only
`npm ci --ignore-scripts` with an installer environment allowlist. The runtime
directory identity derives from a SHA-256 manifest covering the packaged
lockfile, entrypoints, projection/budget code, notices, and exact installed Pi
package versions. Gateway startup re-hashes those files and fails with
`pi_runtime_integrity_mismatch` rather than running stale or modified bytes.

## Failure behavior

- Missing opt-in, verified model, or credential: report `setup_required` and
  show the first-use connection gate instead of a chat composer.
- Missing Node or pinned dependencies: report unavailable and keep the legacy
  local shell clearly labelled.
- Sidecar exit/protocol violation: fail the Pi message job with a stable gateway
  code; do not mark or mutate a scientific run.
- Prompt timeout: best-effort abort the Pi turn, refresh session state, then
  return `pi_gateway_timeout`.
- Tool projection violation: withhold the payload and return
  `pi_projection_blocked`.
- Cancellation: abort the Pi turn and cooperatively cancel only the specifically
  bound EasyICU job. Cancellation cannot publish evidence or become success.
- Stale authority: reject the prompt until explicit rebind.
- Unsupported operation: return a structured blocked result; never silently
  fall back to direct filesystem or process access.

## Upgrade strategy

Upgrades are deliberate dependency-review changes:

1. update the exact Pi version and reviewed upstream commit;
2. regenerate the lockfile with lifecycle scripts disabled;
3. rerun protocol, built-in-tool-denial, PHI, authority, session/resume, and UI
   ownership tests;
4. inspect SDK changes to session events, custom tools, model runtime, and
   persistence;
5. perform local browser QA before changing the documented pin.

Pi JSONL is treated as an upstream format. EasyICU metadata carries its own
schema version and can be migrated independently.

## Phase 2 proposal (not implemented)

Only after this shell is characterized should EasyICU consider letting Pi
drive a bounded `inspect -> request edit -> execute -> validate -> inspect`
loop. That design must use typed change proposals, the existing Coder/executor,
current-plan identity, capability receipts, and evidence gates. Pi must never
edit generated scientific code or run arbitrary commands directly. Replacing
Planner, Coder, or the scientific resume contract remains a separate ADR and
benchmark workstream.
