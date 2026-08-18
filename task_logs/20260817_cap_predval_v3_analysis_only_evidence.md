# CAP-PREDVAL-V3 upstream lineage and analysis-only evidence

Date: 2026-08-17
Task: `CAP-PREDVAL-V3`
Branch: `codex/cap-predval-v3-analysis-only-20260817`
Worktree: `/Users/haibo/Documents/GitHub/.worktrees/easyicu-cap-predval-v1-20260817`
Base HEAD: `f9b2c9dc3fc1450c8f4970e553ab078e339e6d67`

## Decision

Extend the isolated prediction-validation incubator through the first governed
EvidenceStore boundary without promoting it into Planner/runtime selection.
The new bridge may register a deterministic result bundle only at the closed
`analysis_only` authority ceiling. It cannot publish aliases, numeric claims,
scientific claims, or paper authority.

The execution owner remains EvidenceStore-free. It can issue a typed host seal
only after recomputing the complete receipt from the same digest-bound CSV.
The new authority adapter consumes that seal and exact upstream records; it
does not reverse-import execution.

## Failing baseline

The V2.1 owner/provenance baseline reproduced at 38 passed before editing.
V3 tests were added before the bridge implementation and failed during
collection with:

```text
ModuleNotFoundError: No module named
'easyicu.research_agent.authority.prediction_validation_evidence'
```

## Closed lineage contract

One analysis registration requires exactly one canonically ordered binding for
each of these upstream roles:

1. prediction table;
2. cohort;
3. subject-disjoint split assignment;
4. model artifact;
5. code snapshot;
6. environment lock;
7. runtime receipt.

Every binding names an exact EvidenceStore id, SHA-256, kind and producing
step. The bridge also requires the producing run id to match record metadata,
re-verifies current artifact bytes, and parses the runtime receipt back into
the exact declared clean git/source/environment/container-or-local identity.
The prediction-table digest and evaluation split must reconcile with the V2.1
source receipt and contract.

Stable owner diagnostics distinguish invalid schema, missing record, record
metadata/digest mismatch, stale bytes, runtime semantic mismatch, invalid host
seal, and authority-ceiling violation.

## Analysis-only EvidenceStore bridge

- Registration writes one deterministic JSON `statistic` record with
  `publish_aliases=False` and the seven upstream evidence ids as inputs.
- The bundle contains the typed validation spec, complete V2 receipt, host
  recomputation seal, upstream lineage and a policy whose authority fields are
  fixed to false.
- The registration receipt binds the bundle digest, lineage digest, stored-file
  digest, run id, validation step and upstream ids.
- Revalidation works after an EvidenceStore reload and checks the current bytes,
  bundle coordinates, metadata, lineage records and runtime receipt again.
- Revalidation reports `prediction_validation_authority_ceiling_violation` if
  the analysis record later acquires an alias, numeric claim or scientific
  claim.

General EvidenceStore administrative APIs remain capable of a deliberate later
mutation. V3 detects that drift when its registration validator is invoked; it
does not claim a global immutable-store enforcement mechanism.

## Verification

- New V3 lineage/bridge suite: 10 passed.
- Prediction owner/provenance, V3 bridge, EvidenceStore registration,
  scientific-claim authority, capability inventory, package-direction
  boundaries and module-graph suite: 121 passed.
- Capability inventory audit: OK; both the runner and bridge remain explicitly
  `experimental`.
- Research-agent module graph: 540 modules, 2,119 edges, 0 cyclic SCCs.
- Targeted Ruff lint, formatting and `git diff --check`: passed.
- No full exact-head CI was run because this remains an isolated experimental
  slice and is not a freeze, merge, release or formal-experiment checkpoint.
- The active Figure 2 worktree was not edited or used for these checks.

## Remaining gates

V3 is an analysis-only infrastructure milestone, not a production prediction
capability. It still lacks a production workflow that itself creates these
seven records, a governed model-fit/preprocessing owner that proves training-
only fitting, independent human review, release/full-CI evidence, and explicit
promotion policy. Planner/runtime selection and paper-facing claims remain
forbidden. DeLong, decision-curve analysis, model selection, external
validation and dynamic prediction remain separate owners.
