"""Deterministic statistical method implementations used by the agent runtime.

The public convenience symbols remain available from :mod:`easyicu.research_agent`.
Import method-specific APIs from their named modules, for example
``easyicu.research_agent.methods.rmst``.

Wrap-vs-rewrite rule (review checklist for every kernel added here).
Prefer a thin wrapper around a curated package; hand-roll only when one
of these holds, and say which one in the module docstring:

1. the package has no such quantity (e.g. DeLong variance, NRI/IDI);
2. the package's quantity is not the one we claim (e.g. lifelines'
   restricted-mean population variance is not the estimator's sampling
   SE, so RMST integrates its own variance);
3. auditing must see through every step (fixed seeds, no global RNG,
   per-stage fail-closed checks that a package call cannot expose).

Anything else -- including a slower-but-correct reimplementation of an
existing primitive -- is a regression even if tests pass. Kernels are
frozen once reviewed: fix behavior by filing a new Tool Card and version,
never by hot-patching this package at runtime.
"""
