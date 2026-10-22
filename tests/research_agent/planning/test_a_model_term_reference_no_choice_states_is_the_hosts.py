"""A model term's reference the coding already decides is not the Planner's to write.

pilot1 planning (10-22, 规划修复统计): 「binary/categorical terms require a
reference index; continuous/ordinal_linear terms must omit it」 was the
second most frequent repair.  A binary term without one compares its other
level with its first; a continuous or ordinal-linear term drops one written
by mistake.  A categorical term must still state its reference.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from easyicu.research_agent.planning.progressive_contract import ProgressiveModelTermIntent


def _term(coding: str, reference: object, *, role: str = "exposure") -> ProgressiveModelTermIntent:
    return ProgressiveModelTermIntent.model_validate({
        "name": "term_a", "role": role, "coding": coding, "reference_level_index": reference,
        "clinical_rationale": None if role == "exposure" else "A baseline confounder of exposure and outcome.",
    })


def test_a_binary_term_without_a_reference_takes_its_first_level() -> None:
    assert _term("binary", None).reference_level_index == 0
    assert _term("binary", 1).reference_level_index == 1


@pytest.mark.parametrize("coding", ["continuous", "ordinal_linear"])
def test_a_linear_term_drops_a_reference(coding: str) -> None:
    assert _term(coding, 0).reference_level_index is None


def test_a_categorical_term_still_states_its_reference() -> None:
    assert _term("categorical", 2).reference_level_index == 2
    with pytest.raises(ValidationError, match="require a reference index"):
        _term("categorical", None)
