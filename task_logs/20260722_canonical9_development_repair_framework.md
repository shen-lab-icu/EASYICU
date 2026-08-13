# Canonical9 development repair framework — fail-closed handoff

> Date: 2026-07-22 EDT  
> Code: `refactor/agent-control-plane@74b875b`  
> Scope: repository-only benchmark policy and tests; no provider, Docker, full6 read, materialization, or Canonical9 run.

## What changed

`benchmarks/figure2_canonical9/development_repair_framework.py` loads a canonical
exact-nine repair protocol and combines it with the frozen input selector. It is
a readiness report, not a runner: it distinguishes typed input/provenance
authority, human protocol decisions, scientific redesign, and reproduced shared
engine regressions. It never upgrades an input binding or authorizes a real run.

`tools/inspect_canonical9_repair_readiness.py` is an offline inspection CLI. It
does not load cohort data or instantiate a Provider.

The benchmark-local policy holds three non-engine stops explicitly: E2 needs a
reviewed lactate outlier/transform ProtocolCard; H2 needs an owner-defined
exposure-data contract and then demonstrated positivity; H3 needs a reviewed
scientific redesign rather than a seed, threshold, or cluster-count relaxation.

## Current report

The committed selector yields **0/9 launch-ready**. This is expected: no task has
a paper-facing frozen production input, and the additional E2/H2/H3 requirements
remain open. It is not evidence that the runtime itself has failed nine tasks.

The full6 source is still the prerequisite. Its legacy export lacks recorded
patient identity, so a structural metadata sidecar could restore column structure
but cannot become patient-level paper authority. The divergent
`claude/full6-typed-seal-20260722` branch was not merged: merging it wholesale
would remove current P4 real-run protections.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='src:.' /opt/anaconda3/bin/python -m pytest -q \
  -p no:cacheprovider -p no:randomly \
  tests/benchmarks/figure2_canonical9/test_development_repair_framework.py \
  tests/benchmarks/figure2_canonical9/evaluator/test_input_binding_v2.py \
  tests/benchmarks/figure2_canonical9/test_realrun_authority.py
```

Result: **91 passed**. Black, Ruff, `py_compile`, diff-check, architecture, and
module-graph gates also passed. The resource/context baseline still reports its
pre-existing `core.py`/`outbound.py` SHA drift; neither file was changed here, so
the baseline was not regenerated or weakened.

Follow-up `471a16f` closes the source-authority boundary discovered during the
P5 review: materialization now preserves a native source `seal_kind` in its
immutable semantic provenance, and P4 rejects any structural retrofit source
even if its typed cohort sidecar, cohort bytes, and materialized provenance all
verify. This is intentionally stricter than “valid metadata”: an attested,
patient-level source contract would need a separate future path. P4/export
package/materialized metadata regression: **165 passed**.

## Next gated work

1. Review and selectively port only structural typed-seal primitives needed for
   a synthetic/no-write proof; do not merge the stale branch wholesale.
2. Before any external full6 write, obtain a separate operation confirmation and
   prove whether a patient-level source authority can be supplied. If it cannot,
   Canonical9 remains correctly blocked.
3. Only after input authority and case ProtocolCards exist, present the exact
   `--arms aware` command, model/cost boundary, and request final live-run
   confirmation.
