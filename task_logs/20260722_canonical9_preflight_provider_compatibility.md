# Canonical9 offline preflight: provider-compatible runtime gate

Date: 2026-07-22

## Scope

This is a development-only E1--E3 graph preflight.  It uses in-memory
synthetic cohorts and does not call a real Provider, Docker, patient data, or
the frozen Canonical9 input bindings.  It remains `diagnostic_only`; it is not
evidence that any Canonical9 task is paper-ready.

## Problem found after provider hardening

The preflight's `ScriptedPreflightLLM` was a benchmark-defined subclass of
`MockLLMClient`.  The reviewed provider factory deliberately rejects arbitrary
subclasses at delivery time.  Consequently, after the reviewed PHI/outbound
hardening was merged, the earlier preflight suite failed at construction rather
than exercising its graph.

The fix does **not** loosen the factory registry.  The preflight controller now
constructs the exact reviewed `PatternScriptedMockLLMClient` and passes that
object to the pipeline.  The controller itself never receives a prompt.

## Additional fail-closed guarantees

- `runner_network="none"` and `allow_unsafe_host_fallback=False` are explicit
  preflight construction settings.
- `run_preflight()` writes a parent/subprocess/isolation runtime receipt before
  launch and returns a structured `isolation_backend_unavailable` or
  `runtime_incompatible` result without building a pipeline when it is unsafe.
- A post-run receipt reclassifies a newly observed nested `sandbox-exec` denial
  instead of leaving it as a generic repair failure.
- Every live formal E1/E2/E3 `expected_output` has a one-to-one scope map.
  Only deterministic Table 1 is claimed as produced; auxiliary work is
  `planned_only` and sealed figures are explicitly `not_produced_offline`.
- The production paper-acceptance gate supplies typed rejection reasons for a
  one-task offline smoke (`TASK_COVERAGE_INVALID` and missing frozen execution
  identity), rather than treating an `invalid` string as proof by itself.

## Verification

```bash
MPLCONFIGDIR=/private/tmp/easyicu_preflight_final PYTHONPATH="src:." \
  /opt/anaconda3/bin/python -m pytest -q \
  tests/benchmarks/figure2_canonical9/preflight/ -p no:randomly
# 58 passed, 18 warnings

ruff check benchmarks/figure2_canonical9/preflight \
  tests/benchmarks/figure2_canonical9/preflight
black --check benchmarks/figure2_canonical9/preflight \
  tests/benchmarks/figure2_canonical9/preflight
python -m py_compile benchmarks/figure2_canonical9/preflight/*.py
git diff --check
# all passed
```

## Remaining boundary

This preflight covers only diagnostic E1--E3 orchestration.  The real
paper-facing Canonical9 evaluation is still frozen and must later run the full
aware arm using the approved data/input authority, with a real Provider only
under explicit operator authorization.
