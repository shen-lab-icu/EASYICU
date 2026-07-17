# E3 legacy resume authority preflight fix

Date: 2026-07-17 EDT

Branch: `refactor/agent-control-plane`

Review base: `fc62b95`

Task: `AGENT-TRACK-A-PERF-REFACTOR`

## Decision

The online E3 acceptance run remains frozen. A read-only preflight of the
archived run found two deterministic authority blockers that would have stopped
the requested Step 02 resume before any useful provider call. This small batch
fixes those blockers without relaxing dictionary provenance, evidence
containment, deterministic gates, or the monotonic provider ledger.

No request was sent to `127.0.0.1:8317`, and the archived E3 run was not
modified.

## Blockers found

### 1. Package dictionary drift was checked too late

The archived run records concept dictionary SHA
`bc377779ce0f...`; the current package dictionary is
`095350e3...`. The old resume path could prepare the run and refresh mutable
run-level fingerprint files before comparing the checkpoint-selected
dictionary authority.

The pipeline now verifies the already selected resume manifest before writing
the resume environment receipt or refreshing the root dictionary fingerprint.
Mismatch is a `ConceptDictDriftError`. Historical manifests that predate
dictionary coordinates remain compatible; explicit replay verification still
requires a recorded fingerprint.

`EASYICU_DICT_PATH` is not a valid solution for this archived run: it is a
merge overlay, while the fingerprint is computed from the packaged dictionary.
The later one-off online acceptance must import the current engine from an
ephemeral mirror whose packaged `concept-dict.json` alone is restored from
`ea9fc98^`, with the canonical `.venv` and an explicitly verified import root.

### 2. Legacy Step 01 revalidation used sealed absolute output paths

The archived run predates StepAuthorityCapsule. Current validator drift
therefore correctly triggers deterministic revalidation of Step 01. Its sealed
summary contains absolute paths into the historical output directory, while
the revalidator intentionally evaluates digest-verified evidence copied into a
temporary output view. Passing the old paths unchanged caused false
`declared_product_missing` and output-containment errors.

The revalidator now builds an in-memory projection with these limits:

- only evidence explicitly listed by the step, owned by the same step, path
  contained in the run, and matching its sealed SHA is materialized;
- only an explicitly recognized output-path field, direct figure path, or raw
  output-container file entry whose basename is in that verified materialized
  map is projected;
- provenance metadata, descriptions, and unmatched paths are not rewritten;
- unmatched or unsealed paths continue to fail the existing containment and
  declared-product gates;
- sealed summary bytes and persisted checkpoint records are not mutated.

## Archived-run characterization

On a read-only clone of
`run_20260716T072721_7fd5c5`, with the archived dictionary authority pinned,
selective revalidation produced:

- revalidated steps: `00_probe`, `01_cohort_flow`;
- invalidated steps: none;
- Step 01 latest status: `ok` with
  `revalidated_without_execution=true`;
- Step 01 deterministic gate errors: zero;
- Step 01 script, output bytes, evidence digests, provider receipts, and prior
  audit evidence: unchanged;
- no temporary replay path persisted into the sealed summary.

An unbound `/outside/not_sealed.parquet` control remained unchanged and failed
closed.

This refines milestone invariant 3: validator drift may append a new
`revalidated_without_execution` checkpoint for Step 01. The immutable
code/output/evidence/provider/audit authorities must remain unchanged, and Step
01 must not execute or call a provider; the current record digest itself is not
expected to remain byte-identical after legitimate revalidation.

## Verification

Canonical environment: repository `.venv`, `PYTHONPATH=src`; nested sandbox
execution tests were run outside the Codex app sandbox because macOS
`sandbox-exec` cannot nest inside it.

- resume revalidation + concept dictionary focused tests: **32 passed**;
- authority/capsule/meta/pipeline-resume regression: **149 passed**;
- full resume/provider-budget/execute-contract regression: **186 passed** in
  142.99 s;
- total focused regression: **367 passed, 0 failed**;
- Ruff, Black, `git diff --check`: green.

## Next authorized action

1. Commit this small batch and obtain Claude's read-only review.
2. From the accepted commit, construct an ephemeral current-engine import tree
   with only the packaged concept dictionary restored to `bc377779...`.
3. Verify engine, validator, dictionary, SOFA2, import-root, and archived replay
   coordinates before probing the endpoint.
4. Snapshot immutable Step 01 authorities and the old performance ledger.
5. Probe `8317`; on success, perform exactly one same-run resume stopping after
   `02_exposure_derivation_and_qc`.
6. Do not reset the old receipt, grant a fourth logical repair, retry a failed
   online run, or convert a scientific/data failure into a deterministic pass.
