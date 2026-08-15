# Canonical9 real-run authority P4 final seal

Date: 2026-07-22 (EDT)  
Task: `FIG2-CANONICAL9-GATE` P4 authority closure  
Commit: `0c880bdd8cd63a55f87bc448953c6c7050a40b4f`  
Branch: `refactor/agent-control-plane` (fast-forward merged locally; not pushed)

## Scope

This batch closes the remaining launcher-to-ledger authority gaps without running
Docker, a real Provider, patient data, or Canonical9.  It does not create a
production input authority and therefore cannot itself authorize a real batch.

## Changes

- An operator declaration now names one as-yet-absent batch directory.  After
  verification, the launcher atomically reserves that exact root with `mkdir`
  semantics and writes an immutable, fsynced receipt with exclusive creation.
  Replaying the same declaration cannot obtain a second batch root.
- `RealRunAuthorization` carries the already-verified batch ID and frozen
  task-to-input map.  The launcher no longer reopens the declaration or production
  authority after verification.
- Preflight binds each JSONL row's exact cohort/trajectory authority paths and
  references, hashes the referenced sidecars, and invokes the same verified typed
  cohort/trajectory loaders used by the runner.  Unpaired trajectory declarations,
  mismatched sidecar paths, and a loader that cannot bind an authority all block.
- The execution-config digest now includes request timeout.  A mutable `--case`
  selector and a model matrix are not valid Canonical9 semantics.
- The post-run ledger verifies its immutable receipt, parses each actual
  `manifest.json`, validates `ExecutionIdentity`, derives run ID/identity/input
  authority from that manifest, and cross-checks any score summary.  A score cannot
  self-report a different identity into the ledger.
- Strict absolute/non-symlink JSONL handling remains mandatory for Canonical9 and
  explicit authority requests; ordinary non-Canonical9 JSONL invocations keep their
  prior relative-path behavior.

## Evidence

Focused P4 and cross-contract regression (all offline):

```text
82 passed in 1.78s
237 collected: P4/EHRFlow/JSONL/resume/scoring cross-suite
```

The 237-test cross-suite exposed three existing failures outside this batch:

- `test_gate_ladder_prefers_run_status_status` returns `None` before the expected
  `publication_ready` state.
- Two resume tests stop earlier because the Docker runtime source hash expected by
  the test environment differs from the current inherited checkout.

Neither failure is in a file modified by `0c880bd`; `git diff 4751842 -- src/`
was empty and the source hashes at `4751842` match this checkout.  They are kept as
separate existing-environment/coverage work, not masked or weakened here.

Structural checks:

```text
Ruff, Black, py_compile, git diff --check: passed
arch_measure: no lower-is-better regression
module graph: no new cycle
```

The resource-context baseline still reports an inherited drift in
`agents/core.py` and `research_context/outbound.py`; neither file changed from
`4751842` in this batch, so its baseline was not rewritten.

## Honest state after P4

The transport/authority protocol is now ready for a real input freeze, but the
actual full6 data has not been sealed into the required typed production input
authority.  Canonical9 remains blocked and no paper-facing result exists.  The next
work is the case-neutral nine-question development repair framework and the
typed-input/seal decision; only then can the operator approve a fresh real
`--arms aware` batch.
