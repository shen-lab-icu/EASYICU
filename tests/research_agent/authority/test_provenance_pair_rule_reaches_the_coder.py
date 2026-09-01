"""The host demanded a helper globally and exempted it in a branch that never fired.

canary9's E3 step 07 was quarantined BEFORE it ran: the Coder called
``measurement_provenance_receipt`` in a rendering-only figure, and the preflight
gate refused it with ``measurement_provenance_pair_undeclared`` --
``declared_pairs: []``, ``observed_pair: ["stage", "stage_n"]``.

The Coder was following the instruction it was given. ``coder.txt`` says, in
bold, that "every result step declaring a measured/count pair must call the host
``measurement_provenance_receipt`` ... this requirement is not limited to a
component-QC step". The sentence that exempts a step with no pair existed, but
it was emitted only when the step declared NO INPUTS AT ALL -- and every real
figure step has inputs. So no figure step was ever told, and the gate refused
what the prompt had asked for.
"""

from __future__ import annotations

import pytest

from easyicu.research_agent.agents.core import (
    _NO_PROVENANCE_PAIR_RULE,
    _declares_no_measurement_provenance_pair,
    _typed_input_scope_contract,
)
from easyicu.research_agent.schema import AnalysisStep


def _step(inputs: list[str]) -> AnalysisStep:
    return AnalysisStep.model_validate(
        {
            "step_id": "07_stage_stratified_outcome_figure",
            "planned_analysis_role": "auxiliary",
            "intent": "Draw the stage-stratified outcome from its bound tables.",
            "inputs": inputs,
            "expected_outputs": ["figure:stage_stratified_outcome"],
            "method": "visualization",
            "input_consumption_contracts": [
                {
                    "schema_version": "easyicu.artifact_consumption/1",
                    "input_key": key,
                    "mode": "all_rows",
                    "role_column": None,
                    "expected_roles": [],
                }
                for key in inputs
                if ":" in key
            ],
        }
    )


def _rule_reaches(inputs: list[str]) -> bool:
    return _NO_PROVENANCE_PAIR_RULE.strip() in _typed_input_scope_contract(
        _step(inputs)
    )


def test_a_figure_step_with_typed_inputs_is_told_not_to_call_the_helper() -> None:
    """The exact shape that was quarantined on canary9.

    A figure consumes ``table:``-style products, which are never a bare
    measured/count pair, so the gate would refuse any call -- and now the
    prompt says so before the Coder writes one.
    """

    assert _rule_reaches(["table:absolute_risk_context"]) is True


def test_a_step_with_no_inputs_is_still_told() -> None:
    """The branch that already worked keeps working, from the same one rule."""

    assert _rule_reaches([]) is True


@pytest.mark.parametrize(
    "inputs",
    [
        ["lact_measured", "lact_n"],
        ["x_measured_6h", "x_n_6h"],
        ["lact_measured", "lact_n", "table:absolute_risk_context"],
    ],
)
def test_a_step_that_really_declares_a_pair_is_not_told_to_skip_it(inputs) -> None:
    """The rule must not silence the steps the gate expects the call FROM.

    Emitting it everywhere would be the opposite defect: the audit that proves
    a measured flag agrees with its count would stop being written, and the
    step would fail the other way.
    """

    assert _rule_reaches(inputs) is False


@pytest.mark.parametrize(
    "inputs",
    [
        ["lact_measured"],  # a status column whose companion is not declared
        ["lact_n"],  # a count whose status column is not declared
        ["age", "sex"],  # bare columns that are not a pair at all
    ],
)
def test_bare_columns_that_are_not_a_pair_are_told(inputs) -> None:
    assert _rule_reaches(inputs) is True


def test_the_predicate_uses_the_gates_own_pairing_rule() -> None:
    """One rule, not two spellings of it.

    The gate pairs ``<stem>_measured`` with ``<stem>_n``, preserving a trailing
    window suffix. A second copy here would drift, and the prompt would start
    exempting steps the gate still refuses -- which is the defect this file is
    about, in the other direction.
    """

    from easyicu.research_agent.icu_rules import companion_count_column_for_measured

    assert companion_count_column_for_measured("lact_measured") == "lact_n"
    assert companion_count_column_for_measured("x_measured_6h") == "x_n_6h"
    assert companion_count_column_for_measured("lact_max") is None
    # ...and the predicate agrees with it on a name the naive guess gets wrong.
    assert _declares_no_measurement_provenance_pair(_step(["lact_max", "lact_max_n"]))


def test_the_coder_prompt_still_demands_the_call_where_it_applies() -> None:
    """The global demand is not weakened; only its exemption is delivered.

    If this stops being true, the fix has drifted into removing the audit
    rather than telling the Coder when it does not apply.
    """

    from pathlib import Path

    import easyicu.research_agent as package

    text = (
        Path(package.__file__).parent / "providers" / "prompts" / "v1" / "coder.txt"
    ).read_text(encoding="utf-8")
    assert "measurement_provenance_receipt` on the exact analyzed frame" in text
