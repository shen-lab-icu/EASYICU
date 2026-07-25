# Canonical9 offline preflight: bounded control-flow evidence

Date: 2026-07-22

## Scope

This increment extends only the synthetic E1--E3 graph preflight. It uses an
in-memory cohort and the reviewed built-in offline client; it does not call a
real Provider, Docker, patient data, or the frozen Canonical9 bindings. Its
result remains `diagnostic_only` and has no paper authority.

## Controls exercised through the real pipeline

- A persistent primary-step failure consumes exactly the configured runtime
  code-repair cap (`2`), with five actual coder prompts rather than a synthetic
  counter assertion.
- Both a sleeping generated program and an infinite CPU loop hit the real
  subprocess timeout/watchdog, return `timed_out=True` / `returncode=-1`, and
  do not begin a repair when the repair cap is zero.
- A primary result requests replanning. An unchanged candidate is stopped by
  the consecutive no-op cap; two materially distinct candidates consume the
  substantive replan budget and produce the pipeline's typed
  `replan_budget_exhausted` finding.
- Stop/resume continues after all persisted files have the same mtime, while
  retaining the completed deterministic Table 1 evidence IDs and SHA-256
  values. Altering a completed table artifact instead makes resume fail before
  any LLM prompt can be delivered with `EvidenceAuthorityIntegrityError`.

The preflight harness exposes these solely as closed, static test controls. It
continues to pass the exact reviewed `PatternScriptedMockLLMClient` into the
pipeline; no custom client or wrapper crosses the provider-authority boundary.

## Verification

```bash
MPLCONFIGDIR=/private/tmp/easyicu_preflight_controls_final PYTHONPATH="src:." \
  /opt/anaconda3/bin/python -m pytest -q \
  tests/benchmarks/figure2_canonical9/preflight/ -p no:randomly
# 65 passed

ruff check benchmarks/figure2_canonical9/preflight \
  tests/benchmarks/figure2_canonical9/preflight
black --check benchmarks/figure2_canonical9/preflight \
  tests/benchmarks/figure2_canonical9/preflight
python -m py_compile benchmarks/figure2_canonical9/preflight/*.py
git diff --check
# all passed
```

## Remaining boundary

The controls prove bounded behaviour of the offline diagnostic harness, not
the validity of real clinical outputs. Before a paper-facing Canonical9 run,
the approved full6 input authority, frozen execution identity, explicit
operator authorization for any real Provider, and the full `--arms aware`
workflow remain mandatory.
