"""A helper the prompt tells every step to call must publish what it returns.

Measured 2026-08-01 over all 633 generated scripts on disk: exactly 2 import
``easyicu.research_agent.methods.ordered_trends.wilson_interval``, and **both
invented a field name** -- ``ci.lower``/``ci.upper`` in canary32's E3 and
``ci.lower_bound``/``ci.upper_bound`` in the 2026-07-31 run of the same step.
Zero read ``ci.ci_low``/``ci.ci_high``.  Both died:

    AttributeError: 'WilsonInterval' object has no attribute 'lower'

The correct field names were in the prompt -- inside a block opening "when and
only when the step method is exactly ``ordinal_stratified_descriptive_analysis``".
Both scripts belong to a ``visualization`` step.  The bullet that tells *every*
step to use the helper named it and said nothing about its shape, so the
instruction's scope was narrower than the obligation's scope and the model was
left to guess.

The other 30 scripts naming ``wilson_interval`` call a different function of
the same name in the distribution executor, which returns a ``(low, high)``
tuple -- which is exactly why guessing is not safe here.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from easyicu.research_agent.methods.ordered_trends import (
    WilsonInterval,
    wilson_interval,
)

_CODER_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "easyicu"
    / "research_agent"
    / "providers"
    / "prompts"
    / "v1"
    / "coder.txt"
)
_HELPER = "easyicu.research_agent.methods.ordered_trends.wilson_interval"
_GATED_BLOCK_OPENER = "CONTROLLED ORDERED-STRATIFIED METHOD"


def _prompt() -> str:
    return _CODER_PROMPT.read_text(encoding="utf-8")


def _ungated_region() -> str:
    """Everything before the method-gated block."""

    text = _prompt()
    return text[: text.index(_GATED_BLOCK_OPENER)]


def test_the_bound_field_names_are_the_dataclasss_own():
    """Anchored on the class, not on a string the prompt happens to contain.

    Renaming the field has to break this test, or the prompt drifts silently
    back into the state that killed both recorded scripts.
    """

    fields = {field.name for field in dataclasses.fields(WilsonInterval)}

    assert {"ci_low", "ci_high"} <= fields
    assert "lower" not in fields
    assert "upper" not in fields


def test_the_ungated_bullet_that_names_the_helper_also_says_how_to_read_it():
    """The scope of the instruction must equal the scope of the obligation."""

    ungated = _ungated_region()

    assert _HELPER in ungated, "the helper is named outside the gated block"
    for field in ("ci_low", "ci_high"):
        assert (
            f"ci.{field}" in ungated
        ), f"the ungated bullet names the helper but never says ci.{field}"


def test_both_recorded_guesses_are_named_as_wrong():
    """The two field names real runs actually invented, both refused by name."""

    ungated = _ungated_region()

    for guess in ("ci.lower", "ci.upper", "ci.lower_bound", "ci.upper_bound"):
        assert guess in ungated, f"{guess} is not ruled out where the helper is named"


def test_the_collision_with_the_other_helper_of_the_same_name_is_declared():
    """Two functions, one name, two return types -- the reason guessing failed.

    The distribution executor's ``wilson_interval`` returns a plain tuple, so a
    model that has seen either one has no way to know which is in scope.
    """

    from easyicu.research_agent.execution.runners import (
        exposure_outcome_distribution_executor as distribution,
    )

    low, high = distribution.wilson_interval(5, 20, confidence_level=0.95)
    assert isinstance(low, float) and isinstance(high, float)

    interval = wilson_interval(event_n=5, n=20)
    assert isinstance(interval, WilsonInterval)
    assert not isinstance(interval, tuple)

    assert "tuple" in _ungated_region().rsplit(_HELPER, 1)[1][:700]


def test_the_gated_block_still_carries_its_own_statement():
    """The narrower block is not weakened by widening the general one.

    Deleting the gated sentence and relying on the general bullet would be a
    net loss for the one method that has extra obligations.
    """

    text = _prompt()
    gated = text[text.index(_GATED_BLOCK_OPENER) :]

    assert "ci.ci_low" in gated
    assert "ci.ci_high" in gated


def test_no_case_specific_token_entered_the_bullet():
    """Prompt hygiene: this is a host-contract fact, not a benchmark fact."""

    ungated = _ungated_region()
    window = ungated.rsplit(_HELPER, 1)[1][:800].casefold()
    for token in ("sep3", "kdigo", "aki_stage", "mimic", "sepsis", "e1", "e3"):
        assert not re.search(rf"\b{re.escape(token)}\b", window)
