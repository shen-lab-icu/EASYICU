# Pre-push review of the 43-commit EasyICU batch

## Scope

- Remote base: `ec7b7d46d5ab67ce2cdd56fd55a0fc9c79959469`
- Reviewed batch head: `ae3ca0fdf630b68fb1b756137f7de57092125735`
- Range: the exact 43 commits in `ec7b7d46d5ab..ae3ca0fdf6`
- Review boundary: source, contracts, tests, browser assets, and deterministic
  architecture gates. No Provider call, research run, deployment, or push was
  authorized or performed.

## Findings repaired during independent review

1. An ordinary failed-run resume could reopen the terminal initial-generation
   budget without an explicit step rerun. The bootstrap now grants that epoch
   only from explicit host authority, while the Web retry owner binds the first
   failed step from the accepted `run_status.json` gate.
2. Web scientific-readiness projection trusted a claimed cohort digest and
   permissively coerced denominator values. It now recomputes canonical cohort
   and lock digests and accepts only non-boolean JSON integers with consistent
   nonnegative denominators.
3. Five changed browser owners retained old cache-version URLs. Their script
   pins now advance with the reviewed behavior.
4. Advisory manuscript-length reporting introduced a six-module import cycle.
   Shared reader projection and advisory targets now live in the leaf
   `reporting.manuscript_surface` owner; the production graph is acyclic again.

## Architecture adjudication

The checked-in architecture snapshots were already stale at the remote base:
`arch_measure` reported 17 lower-is-better deviations and the module graph
contained 688 modules against a 643-module snapshot. The reviewed range adds
seven explicit owners:

- `authority.plausibility_receipt_code`
- `authority.rmst_runtime`
- `contracts.functional_form_projection`
- `contracts.rmst`
- `execution.docker_locality`
- `execution.runners.rmst_executor`
- `planning.progressive_module_ids`

Those modules separate typed authority, execution, and planning contracts. The
range adds no top-level package and no dynamic import loss. The current graph
contains 695 modules, 2,946 internal edges, and zero cyclic modules. The size
ratchet growth in the range is attached to the reviewed prompt-projection,
hard-stop, validation, RMST, and Docker-locality contracts; it is accepted here
as the new lower-is-better ceiling rather than left as silent CI drift.

The baseline refresh records both the pre-existing remote drift and the seven
owners added by this batch. It does not grant scientific, launch, experiment,
or publication authority.
