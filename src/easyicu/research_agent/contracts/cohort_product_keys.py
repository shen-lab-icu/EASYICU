"""The published spellings of the closed primary-cohort product.

This is vocabulary, not execution: four pure predicates over the strings a
Planner may write, with no dependency on binding, evidence or the filesystem.
It was moved out of ``execution.runners.typed_input_binding`` (which re-exports
every name, so its callers are unchanged) so that the planning layer can ask
whether a step declares the host's cohort input without importing an executor.

The alternative -- letting plan validation reach down into the execution layer
-- is the dependency inversion this package's module graph exists to prevent,
and the cost of a second copy of the vocabulary is already recorded below.
"""

from __future__ import annotations

from typing import Any, Optional


__all__ = [
    "CLOSED_COHORT_PRODUCT_KEYS",
    "CLOSED_COHORT_PRODUCT_KIND",
    "closed_cohort_product_vocabulary",
    "is_closed_cohort_product_key",
    "sole_typed_cohort_input",
]

#: The exact spellings the Planner may use for the one materialised closed
#: primary-cohort product.  This is a *published* vocabulary: the planner
#: directive tells the Planner to pick one of these, so a step that picks any
#: of them has obeyed the host.  It is a single constant because the previous
#: arrangement stated the list in prose and enforced a narrower one in code --
#: the directive offered five spellings and the ownership predicate could read
#: two.  Measured over 194 recorded plans, 42 real declarations used
#: ``dataset:analysis_cohort``; every one of them was legal, and every one made
#: its step unownable, which sent the primary model to the Coder.
CLOSED_COHORT_PRODUCT_KEYS = frozenset(
    {
        "artifact:analysis_cohort",
        "dataset:analysis_cohort",
        "table:analysis_cohort",
    }
)

#: Anything under this kind names a cohort by construction, so the product part
#: is open (``cohort:analysis_set`` and ``cohort:<exact cohort.name>`` alike).
CLOSED_COHORT_PRODUCT_KIND = "cohort"


def closed_cohort_product_vocabulary() -> tuple[str, ...]:
    """The published spellings, for prompts that must state what is enforced.

    Rendering the directive from this tuple is the point: a sentence listing
    the legal spellings and a predicate accepting them cannot drift apart when
    they are the same object.
    """

    return tuple(sorted(CLOSED_COHORT_PRODUCT_KEYS)) + (
        f"{CLOSED_COHORT_PRODUCT_KIND}:analysis_set",
        f"{CLOSED_COHORT_PRODUCT_KIND}:<exact cohort.name>",
    )


def is_closed_cohort_product_key(input_key: str) -> bool:
    """Whether one typed input key names the closed primary-cohort product."""

    kind, separator, product = str(input_key or "").strip().partition(":")
    if not separator or not product:
        return False
    return kind == CLOSED_COHORT_PRODUCT_KIND or input_key in CLOSED_COHORT_PRODUCT_KEYS


def sole_typed_cohort_input(step: Any) -> Optional[str]:
    """Return the one typed row-membership authority a step declares.

    Three return values, and the caller must keep them apart:

    ``None``  no typed input at all, so ``COHORT_PARQUET`` is the row authority.
    a key     exactly one cohort-scoped typed input; read that digest-bound
              table rather than silently analysing another frame.
    ``""``    more than one typed input, or one this executor family does not
              support -- not owned, so an owner must decline the step.

    This rule was written out three times (cohort summary, Table 1, and in a
    tuple-returning variant for the missingness audit).  All of them now call
    this, so "which frame did the model actually read" has one answer; the
    callers keep only their own arity policy, which genuinely differs.
    """

    typed_inputs = {
        str(value or "").strip()
        for value in getattr(step, "inputs", None) or []
        if ":" in str(value or "").strip()
    }
    if not typed_inputs:
        return None
    if len(typed_inputs) != 1:
        return ""
    input_key = next(iter(typed_inputs))
    return input_key if is_closed_cohort_product_key(input_key) else ""
