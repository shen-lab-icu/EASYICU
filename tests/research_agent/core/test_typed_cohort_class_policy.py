"""A typed cohort's source class policy compares as the chain it denotes.

Column metadata records ``database`` followed by its class prefixes with
repeats removed, and a typed cohort recovers its prefixes as that chain minus
its head.  Both official demo sources list their own name among their class
prefixes, so their cohorts record a shorter list for the same policy.
"""

from __future__ import annotations

import pytest

from easyicu.config import load_src_cfg
from easyicu.research_agent.pipeline import typed_cohort_source_resolution_chain


def _registry(database: str) -> tuple[str, ...]:
    return tuple(
        str(value).strip().lower()
        for value in load_src_cfg(database).class_prefix
        if str(value).strip()
    )


@pytest.mark.parametrize(
    ("database", "recorded"),
    [("eicu_demo", ("eicu",)), ("mimic_demo", ("mimic",))],
)
def test_a_self_prefixed_demo_source_matches_its_registry_policy(
    database: str, recorded: tuple[str, ...]
) -> None:
    registry = _registry(database)
    assert registry[0] == database  # the case that loses a prefix

    assert typed_cohort_source_resolution_chain(
        database, recorded
    ) == typed_cohort_source_resolution_chain(database, registry)


def test_a_different_class_policy_still_fails_the_comparison() -> None:
    registry = _registry("eicu_demo")

    for recorded in ((), ("mimic",), ("eicu", "mimic")):
        assert typed_cohort_source_resolution_chain(
            "eicu_demo", recorded
        ) != typed_cohort_source_resolution_chain("eicu_demo", registry)
