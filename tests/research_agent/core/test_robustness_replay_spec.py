"""A method label must not decide whether the host runs the locked-grid replay.

The prespecified-robustness replay is claimed by a three-string method
allowlist (`prespecified_robustness`, `robustness_sensitivity`,
`sensitivity_comparison`).  Measured over 565 recorded plans: of 485
robustness-vocabulary steps, 263 are claimed and 182 are turned away while
being neither figures nor claimed by the agent-owned validation gate.  62 of
those say `prespecified_sensitivity_analysis` and 12 say the plainest possible
`sensitivity_analysis`, which the list simply does not contain.

The mirror of the measurement-audit case, where the *product names* were the
half that bled and the method string cost only 10 steps.  Two runners, two
allowlists, and in each one it was the other half that leaked -- which is the
argument against enumerating either.

``test_renaming_the_method_alone_loses_the_owner`` is the load-bearing one: the
real fresh19 robustness step, with nothing changed but its label.

Widening the allowlist would be worse than the gap, and
``test_a_scientifically_different_sensitivity_step_is_not_this_one`` records
why: the replay re-estimates an already-locked grid, so a causal-emulation or
weighting variant is different science the runner cannot produce.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.phase import (
    _robustness_sensitivity_runner_owns_step,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    ROBUSTNESS_REPLAY_OUTPUT_FILES,
    declared_robustness_product_registrations,
    robustness_replay_spec_has_kind_mismatch,
    robustness_replay_spec_is_emittable,
)
from easyicu.research_agent.schema import (
    ROBUSTNESS_REPLAY_OUTPUTS,
    AnalysisStep,
    RobustnessReplaySpec,
)

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "real_plan_steps_fresh17_fresh19.json"
_REAL_STEP_ID = "09_standard_robustness_sensitivity"

# The real declaration, minus the one product it spells twice (see
# ``test_the_same_product_under_two_kinds_is_refused``).
_REAL_SPEC = {
    "products": [
        {"product_id": "primary_or", "output": "primary_effect"},
        {"product_id": "complete_case_n", "output": "complete_case_n"},
        {"product_id": "robustness_summary", "output": "robustness_summary"},
        {
            "product_id": "missingness_strategy_notes",
            "output": "missingness_strategy_notes",
        },
        {"product_id": "robustness_matrix", "output": "robustness_matrix"},
    ]
}


def _real_payload() -> dict:
    document = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    plan = next(e for e in document["plans"] if e["label"] == "fresh19")["plan"]
    payload = next(s for s in plan["steps"] if s["step_id"] == _REAL_STEP_ID)
    return json.loads(json.dumps(payload))


def _step(*, declare: bool = True, **overrides) -> AnalysisStep:
    payload = _real_payload()
    if declare:
        payload["expected_outputs"] = [
            value
            for value in payload["expected_outputs"]
            if value != "statistic:robustness_summary"
        ]
        payload["robustness_replay_spec"] = json.loads(json.dumps(_REAL_SPEC))
    payload.update(overrides)
    return AnalysisStep.model_validate(payload)


def _owns(step: AnalysisStep) -> bool:
    return _robustness_sensitivity_runner_owns_step(
        str(step.method or ""),
        str(step.step_id or ""),
        step.expected_outputs or [],
        step=step,
    )


# --------------------------------------------------------------------------
# the real step


def test_renaming_the_method_alone_loses_the_owner() -> None:
    """Same products, same science, 62 corpus steps' spelling of the label."""

    as_planned = _step(declare=False)
    renamed = _step(declare=False, method="prespecified_sensitivity_analysis")

    assert _owns(as_planned) is True
    assert _owns(renamed) is False

    # The declaration says what the label was only hinting at.
    assert _owns(_step(method="prespecified_sensitivity_analysis")) is True


def test_the_plainest_possible_label_is_owned_once_declared() -> None:
    assert _owns(_step(declare=False, method="sensitivity_analysis")) is False
    assert _owns(_step(method="sensitivity_analysis")) is True


def test_a_declaration_cannot_make_the_replay_claim_a_figure_step() -> None:
    """The replay emits no figure, so a declaration cannot buy it one.

    This is a decline, not a rejected plan: the step is well formed, the host
    simply cannot produce one of its products.  Rejecting it in the schema is
    what made a real run's own sealed plan unreadable, so the answer moved to
    the layer that is allowed to say "not mine".
    """

    step = _step(
        expected_outputs=[
            "table:robustness_matrix",
            "figure:robustness_plot",
        ],
        robustness_replay_spec={
            "products": [
                {"product_id": "robustness_matrix", "output": "robustness_matrix"},
                {"product_id": "robustness_plot", "output": "robustness_summary"},
            ]
        },
    )

    assert robustness_replay_spec_is_emittable(step) is False

    undeclared_figure_step = _step(
        declare=False,
        expected_outputs=[
            "table:robustness_matrix",
            "figure:robustness_plot",
        ],
    )

    assert _owns(undeclared_figure_step) is False


# --------------------------------------------------------------------------
# what the declaration refuses


def test_a_scientifically_different_sensitivity_step_is_not_this_one() -> None:
    """The spec is a claim about the science, and the host cannot check it.

    So the guard that exists is capability: an output this replay does not
    produce is refused at the contract rather than claimed and then missing.
    """

    with pytest.raises(ValueError, match="unknown robustness replay output"):
        RobustnessReplaySpec.model_validate(
            {"products": [{"product_id": "e_value", "output": "e_value"}]}
        )


def test_two_products_may_not_name_the_same_output() -> None:
    """`sensitivity_comparison.csv` is byte-for-byte `robustness_matrix.csv`."""

    with pytest.raises(ValueError, match="one declared product"):
        RobustnessReplaySpec.model_validate(
            {
                "products": [
                    {"product_id": "robustness_matrix", "output": "robustness_matrix"},
                    {
                        "product_id": "sensitivity_comparison",
                        "output": "robustness_matrix",
                    },
                ]
            }
        )


def test_the_same_product_under_two_kinds_is_refused() -> None:
    """The real recorded contract declares this exact pair.

    The legacy path normalises them to one name and satisfies both from one
    CSV, so whoever asked for `statistic:` is handed a table.  Only new plans
    reach this check, so nothing already executing changes.
    """

    payload = _real_payload()
    payload["robustness_replay_spec"] = json.loads(json.dumps(_REAL_SPEC))

    assert "statistic:robustness_summary" in payload["expected_outputs"]
    assert "table:robustness_summary" in payload["expected_outputs"]
    # The plan stays readable -- fresh21 proved that rejecting it here kills
    # the run at re-parse, far from the cause -- and the host declines.
    step = AnalysisStep.model_validate(payload)
    assert robustness_replay_spec_is_emittable(step) is False


def test_a_replay_csv_cannot_be_retyped_as_a_json_statistic() -> None:
    """Regression for Web E1's second plan revision and two wasted repairs."""

    payload = _real_payload()
    payload["expected_outputs"] = [
        value
        for value in payload["expected_outputs"]
        if value != "table:robustness_summary"
    ]
    payload["robustness_replay_spec"] = json.loads(json.dumps(_REAL_SPEC))
    step = AnalysisStep.model_validate(payload)

    assert "statistic:robustness_summary" in step.expected_outputs
    assert robustness_replay_spec_has_kind_mismatch(step) is True
    assert robustness_replay_spec_is_emittable(step) is False
    assert _owns(step) is False
    registrations = declared_robustness_product_registrations(step)
    assert "statistic:robustness_summary" not in registrations
    assert registrations["table:robustness_matrix"] == "robustness_matrix.csv"


def test_a_declared_product_with_no_output_is_refused() -> None:
    """A product nothing backs would be produced by nobody while owned."""

    step = _step(
        expected_outputs=[
            "table:robustness_matrix",
            "table:something_else",
        ],
        robustness_replay_spec={
            "products": [
                {"product_id": "robustness_matrix", "output": "robustness_matrix"}
            ]
        },
    )

    assert robustness_replay_spec_is_emittable(step) is False


def test_an_output_naming_a_product_the_step_never_declares_is_refused() -> None:
    with pytest.raises(ValueError, match="does not declare"):
        _step(
            expected_outputs=["table:robustness_matrix"],
            robustness_replay_spec={
                "products": [
                    {"product_id": "robustness_matrix", "output": "robustness_matrix"},
                    {"product_id": "phantom", "output": "complete_case_n"},
                ]
            },
        )


def test_a_prefixed_product_id_is_refused() -> None:
    with pytest.raises(ValueError, match="bare product name"):
        RobustnessReplaySpec.model_validate(
            {
                "products": [
                    {
                        "product_id": "table:robustness_matrix",
                        "output": "robustness_matrix",
                    }
                ]
            }
        )


# --------------------------------------------------------------------------
# capability, declared once


def test_every_declarable_output_has_an_implementation() -> None:
    assert set(ROBUSTNESS_REPLAY_OUTPUT_FILES) == set(ROBUSTNESS_REPLAY_OUTPUTS)


def test_every_capability_file_is_one_the_runner_collects() -> None:
    """The capability map and the runner's own output_files cannot drift.

    Read from the runner module's source rather than a fixture: `product_files`
    is a literal built during a real replay, so this is the same text the run
    would use.  It is the bare-name dict; ``output_files`` is now the canonical
    ``kind:name`` registration compiled from it, and the filenames -- which is
    what this test compares -- are the same in both.
    """

    source = (
        Path(__file__).resolve().parents[3]
        / "src/easyicu/research_agent/execution/runners/deterministic_robustness.py"
    ).read_text(encoding="utf-8")
    anchor = "product_files = {"
    assert anchor in source, (
        "the runner no longer builds its products under a literal named "
        f"{anchor!r}; repoint this test at whatever replaced it"
    )
    block = source.split(anchor, 1)[1].split("\n    }", 1)[0]
    missing = sorted(
        filename
        for filename in ROBUSTNESS_REPLAY_OUTPUT_FILES.values()
        if f'"{filename}"' not in block
    )

    assert missing == []


def test_a_step_without_a_declaration_is_not_claimed_by_the_spec_path() -> None:
    assert robustness_replay_spec_is_emittable(_step(declare=False)) is False
