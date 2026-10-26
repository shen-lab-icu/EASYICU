# Durable research execution

EasyICU uses [DBOS Transact for Python](https://github.com/dbos-inc/dbos-transact-py)
3.3.0 (MIT) for local durable dispatch. Its SQLite database stores an opaque
command ID and a small machine-readable receipt using portable JSON. No cloud
service, pickle, provider credentials, patient rows, or result bodies are stored
in this scheduler. Install the declared `webapp` extra; server startup does not
download dependencies. Desktop Python 3.11 platform locks include its transitive
dependencies and hashes.

## Admission and recovery

The normal research submission and approved-review routes reserve the Web job,
consume existing authorization, bind the current study, and activate a private
command outbox before DBOS starts work. An active outbox repairs a crash between
admission and enqueue. The DBOS workflow ID and Web job ID remain stable; a
duplicate enqueue cannot create another scientific launch. Only new commands
admitted through these routes participate. Old study runs and queued chat
messages are not enrolled retrospectively. Report-only revisions retain their
existing owner and are not automatically replayed.

At startup, under the existing single-process deployment lease, active transport
jobs are reattached before serving requests. Their event cursors keep increasing.
DBOS repeats only the interrupted host-dispatch step; EasyICU reconstructs the
scientific continuation through its existing checkpoint owner. A DBOS `SUCCESS`
means the dispatcher returned a receipt, not scientific success.

Each recovery verifies installed source/dictionary identity, the unchanged study
configuration, opt-in, provider identity, exact checkpoint, original budget task
and declaration. A missing ledger cannot create a replacement. Calls, unknown
in-flight reservations, tokens, cost and wall time remain cumulative. The host
allows at most the initial dispatch plus three crash recoveries; scientific
failures do not trigger DBOS retries or acquire a fresh provider-recovery quota.

Planner continuation writes to a separate `-restart-N` wrapper while retaining
the original paid task. Approved execution uses only already persisted,
authenticated review decisions; the automatic path cannot invent or resubmit an
approval. Formal/paper-facing profiles remain outside this development recovery
policy. Existing independent review and signing requirements are unchanged.

## Supported interruption boundaries

| Interrupted state | Automatic behavior |
| --- | --- |
| Valid progressive Planner checkpoint with original ledger | Reconstruct exact checkpoint; continue under original budget |
| Approved, execution not started | Continue using original persisted approval |
| Executing with Docker backend | Existing Docker owner must confirm stale-container cleanup before reusing outputs |
| Executing with guarded HostRunner | Wait for confirmed process cleanup, then resume through the original scientific checkpoint |
| Legacy Host execution without a lifecycle receipt, unresolved cleanup, or an unverified backend | Preserve files and stop |
| Awaiting human approval | Remain paused; never approve automatically |
| Missing checkpoint, changed code/source/provider, revoked opt-in, exhausted budget | Stop with a typed refusal |
| Write/finalize in progress, ambiguous result state | Preserve artifacts and stop |
| Browser account credential session unavailable after restart | Stop; never substitute another identity |

DBOS does not make external effects exactly-once. Unknown provider calls retain
their conservative budget reservation. Worker completion alone never seals a
scientific step: the existing checkpoint/evidence owner reuses accepted completed
steps and reruns an incomplete step only after process cleanup is established.

## Local Host execution lifecycle

The Host backend reuses the existing portable OS locks and
[psutil](https://github.com/giampaolo/psutil) process identities. No additional
daemon or dependency is installed. Before replacing scripts or deleting outputs,
the parent reserves a run-wide `.host-execution` lease and persists a fresh
nonce. It registers an independent guardian before releasing the launch lock.
The guardian holds that lock, records the worker's PID and creation time, and
only then admits the exact authorized script bytes. A late worker with an old
nonce cannot execute after the reservation changes.

The guardian monitors the original parent's identity and the step timeout. On
normal script completion, failure, cancellation request or parent death, it
freezes the worker and descendants, terminates them, and confirms they have
stopped before publishing a quiescent receipt. Process-group changes such as
`setsid` do not escape the owned-tree walk. The parent retains a separate receipt
fence until log writing and output collection finish, so another runner cannot
delete files in that interval. The whole-run writer lock also checks the Host
receipt before checkpoint restoration can prune outputs.

Missing or corrupt receipts, permission errors, unexpected worker/guardian death,
and unconfirmed cleanup block continuation. A prepared attempt whose launching
processes died before worker admission can be safely superseded. No recovery
path kills a process merely because its numeric PID matches an old receipt.
Only trusted, host-rendered executors use this backend; model-generated code
still requires the existing sandbox. This is not a general supervisor for
arbitrary daemonizing programs or externally launched processes.

The protocol adds process safety, not research authority. Automatic continuation
still needs the original approved plan, source/code/provider identity, opt-in and
existing cumulative budget ledger. The original run and evidence checkpoints
remain authoritative; lifecycle receipts contain no patient rows or credentials.

Explicit cancellation is stored in both the job journal and command outbox.
Queued cancellation settles the job and releases its study binding. A failed
job-journal write is not acknowledged as durable cancellation, though the live
interrupt still runs. Graceful Web shutdown retains its existing cancellation
behavior; automatic continuation covers unexpected process exit once the
application is restarted. DBOS is not an OS boot/relaunch supervisor. A draining
callback keeps the deployment lease so a replacement cannot overlap it.

## Verification and packaging

Owner tests cover two real subprocesses and the real DBOS SDK, stable job/event
identity, original ledger accounting, duplicate delivery, cancellation,
authority/approval refusals, code drift, finite recovery attempts, JSON-only
scheduler rows, outbox admission races and terminal-state reconciliation.
Synthetic review tests exercise the actual review owner with a fake executor.
They do not establish real-provider or scientific acceptance.

Host tests additionally kill real synthetic parents/workers/guardians, launch
detached children and grandchildren, race competing runners during execution
and output collection, and verify that unresolved attempts preserve outputs.
Native Windows process-death behavior and frozen application packaging require
their platform checks; POSIX SIGKILL acceptance does not substitute for them.

Frozen builds must retain `psycopg_binary` binary assets, the dynamically loaded
SQLAlchemy SQLite dialect and DBOS package metadata. DBOS 3.3.0 migrations are
Python code. Both platform build commands and locks include these dependencies;
an actual frozen-application build remains a separate packaging check.

Upstream behavior: [integration](https://docs.dbos.dev/python/integrating-dbos),
[workflows](https://docs.dbos.dev/python/tutorials/workflow-tutorial),
[steps](https://docs.dbos.dev/python/tutorials/step-tutorial).
