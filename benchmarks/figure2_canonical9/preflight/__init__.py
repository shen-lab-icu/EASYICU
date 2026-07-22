"""Zero-Provider, graph-level offline preflight for Canonical9 (E1-E3 batch).

This package is a **development-only diagnostic harness**.  It drives the real
``ResearchAgentPipeline`` graph offline with a scripted, deterministic
``MockLLMClient`` (no external Provider, no patient data) over minimal synthetic
cohort fixtures, and records the graph-level behaviour the manuscript's Figure 2
capability claims depend on: plan-contract validity, expected graph stages, the
deterministic-executor / deterministic-renderer vs Coder division of labour, the
repair/retry cap under fault injection, timeout wiring, stop/resume, loop
termination, the final fail-closed tristate, and proof that a mock run cannot
acquire paper authority.

Hard boundaries (enforced by :mod:`benchmarks.figure2_canonical9.preflight.harness`):

* **Zero external Provider.** Every run uses a ``MockLLMClient`` subclass
  (``__easyicu_mock_client__ is True``); ``external_provider_calls == 0`` is
  asserted on every run.  What a *real* model would plan or code — plan/code
  *quality* — is explicitly out of scope and is the documented Provider boundary.
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
