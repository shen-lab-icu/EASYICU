"""Zero-Provider, graph-level offline **partial-flow smoke** for Canonical9 (E1-E3).

This package is a **development-only diagnostic harness** and a *partial-flow
smoke*, **NOT** E1/E2/E3 publication readiness.  It drives the real
``ResearchAgentPipeline`` graph offline with a scripted, deterministic
``MockLLMClient`` (no external Provider, no patient data) over minimal synthetic
cohort fixtures, and records the graph-level behaviour: plan-contract validity,
expected graph stages, the deterministic-executor / deterministic-renderer vs
Coder division of labour, loop termination, the final fail-closed tristate, and
proof that a mock run cannot acquire paper authority.

Each formal suite ``expected_output`` is mapped item-by-item (see
:class:`.fixtures.ProductMapping`) to its plan step and an honest fulfillment
level, verified against the REAL run manifest.  Offline, the harness genuinely
**produces** only the deterministic Table 1 artifact; the sealed publication
figures are ``not_produced_offline`` and the data-dependent audit/cohort products
are ``planned_only`` (the offline Coder contract-fails them).  That gap is
exactly why this is a partial-flow smoke, not readiness.

Hard boundaries (enforced by :mod:`benchmarks.figure2_canonical9.preflight.harness`):

* **Zero external Provider — two boundaries.** (1) *Parent process:* every run
  executes inside a fail-closed transport spy that replaces
  ``httpx.Client.send`` / ``httpx.AsyncClient.send`` with a counter that raises
  on first use; ``external_provider_calls == 0`` (the *measured* spy count) is
  asserted on every run.  (2) *Subprocess/CLI:* the runner is pinned to
  ``network_policy='none'`` + ``allow_unsafe_host_fallback=False`` so no
  environment variable can relax it; every subprocess step records
  ``requested_network_policy='none'`` and ``isolation_degraded=False``.  The
  ``__easyicu_mock_client__`` / ``llm_is_mockish`` markers are forgeable and are
  recorded only as descriptive colour.  What a *real* model would plan or code —
  plan/code *quality* — is out of scope and is the documented Provider boundary.
* **Isolation fail-closed.** ``run_preflight`` builds + persists a
  ``RuntimeManifest``; when ``integration_ready`` is false (e.g. a nested macOS
  sandbox) it returns a unique structured blocked outcome and does not start the
  pipeline.  A per-step nested-sandbox denial is converted to the same reason via
  ``step_isolation_unavailable`` — never left as a generic ``repair_failed``.
* **Zero patient data.** Cohorts are tiny in-memory synthetic frames.
* **No paper authority.** These runs are diagnostic-only; the production
  Figure 2 acceptance gate rejects them (asserted).  The fixtures never modify a
  global/shared prompt and never touch the frozen Canonical9 input bindings or
  the production input-freeze gate.
* **Scorer-tree neutral.** This package is a *sibling* of ``evaluator/`` and is
  therefore outside ``scorer_tree_sha256`` (which rglobs ``evaluator/`` only) and
  outside the installed wheel (which excludes ``benchmarks/``).

The fixtures in :mod:`.fixtures` are derived from the formal task protocols in
:func:`benchmarks.figure2_canonical9.evaluator.suite.easyicu_evaluation_protocol_suite`
(``e1_sepsis3_prevalence_mortality`` / ``e2_lactate_mortality`` /
``e3_kdigo_gradient``) and are marked diagnostic-only.
"""

from __future__ import annotations

DIAGNOSTIC_ONLY = True
PROVIDER_BUDGET = 0

__all__ = ["DIAGNOSTIC_ONLY", "PROVIDER_BUDGET"]
