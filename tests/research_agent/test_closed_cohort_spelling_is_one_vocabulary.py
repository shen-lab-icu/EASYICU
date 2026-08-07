"""The host must be able to read every cohort spelling it offers the Planner.

canary18's E1 died here.  The Planner declared a complete primary model --
one ``model_requirement`` with an exposure, an outcome, a method family and a
declared adjustment set -- and named its row authority ``dataset:analysis_cohort``,
which is one of the five spellings the planner directive itself offers.  The
ownership predicate could read two of those five.  So::

    adjusted_association_executor_verdict(step).claimed -> False
    reason: "the step declares more than one typed input, or one this
             executor family does not support"

and the step fell through to the LLM Coder.  The Coder's model was right
(OR 1.57, 95% CI 1.02-2.39) but its ``step_summary.json`` was in its own
shape: ``coefficient_file`` where the robustness replay reads
``diagnostic_companions.coefficients``, and no ``primary_or`` at all.  The
robustness step then blocked on "A completed primary estimate with point
estimate and confidence interval is required", its figure was skipped, and the
prevalence figure blocked on the missing evidence.  One unreadable-but-legal
spelling cost four of nine steps.

Measured over 194 recorded plans / 2,000 distinct steps: 42 real declarations
used ``dataset:analysis_cohort``, and reading them gains 18 deterministically
owned steps while losing none.

The durable property is not "accept dataset:" -- it is that the sentence
offering the spellings and the predicate accepting them are one object.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.agents.core import _build_planner_user_prompt
from easyicu.research_agent.execution.runners.adjusted_association_executor import (
    adjusted_association_executor_verdict,
)
from easyicu.research_agent.execution.runners.deterministic_missingness import (
    _cohort_input_scope,
)
from easyicu.research_agent.execution.runners.exposure_outcome_distribution_executor import (
    _typed_cohort_input as _distribution_typed_cohort_input,
)
from easyicu.research_agent.execution.runners.typed_input_binding import (
    CLOSED_COHORT_PRODUCT_KEYS,
    CLOSED_COHORT_PRODUCT_KIND,
    closed_cohort_product_vocabulary,
    is_closed_cohort_product_key,
    sole_typed_cohort_input,
)
from easyicu.research_agent.schema import (
    AnalysisStep,
    CohortDescriptor,
    PlannedModelRequirement,
    ResearchContext,
)

# Written out, not derived from the vocabulary under test: a test that reads
# its cases off the mapping it checks shrinks when the mapping is emptied.
_SPELLINGS_THE_DIRECTIVE_OFFERS = (
    "artifact:analysis_cohort",
    "dataset:analysis_cohort",
    "table:analysis_cohort",
    "cohort:analysis_set",
)


def _step(*, inputs: list[str]) -> AnalysisStep:
    """A completely declared primary adjusted-association step."""

    return AnalysisStep(
        step_id="06_primary_adjusted_association",
        intent="Estimate the adjusted association.",
        method="adjusted_association_models",
        planned_analysis_role="primary",
        expected_outputs=["table:adjusted_association_estimates"],
        inputs=inputs,
        model_requirements=[
            PlannedModelRequirement(
                requirement_id="primary_logistic",
                outcome="death",
                outcome_type="binary",
                method_family="logistic_regression",
                exposure_source="exposure",
                covariates=["age", "sex"],
                analysis_role="primary",
                analysis_set="source_aware",
                required_for_step_success=True,
            )
        ],
    )


# --- the vocabulary is one object, and it is the one that is enforced --------


def test_every_spelling_the_directive_offers_is_one_the_predicate_reads():
    for spelling in _SPELLINGS_THE_DIRECTIVE_OFFERS:
        assert is_closed_cohort_product_key(spelling), spelling


def test_the_published_vocabulary_is_exactly_the_enforced_one():
    """Deleting an entry must break this, not quietly narrow the test."""

    published = set(closed_cohort_product_vocabulary())
    assert published == set(_SPELLINGS_THE_DIRECTIVE_OFFERS) | {
        f"{CLOSED_COHORT_PRODUCT_KIND}:<exact cohort.name>"
    }
    assert CLOSED_COHORT_PRODUCT_KEYS == {
        "artifact:analysis_cohort",
        "dataset:analysis_cohort",
        "table:analysis_cohort",
    }


def test_an_arbitrary_cohort_name_under_the_cohort_kind_is_readable():
    """``cohort:<exact cohort.name>`` is open by construction, so it is a kind."""

    assert is_closed_cohort_product_key("cohort:sepsis_icu_adults")
    assert is_closed_cohort_product_key("cohort:whatever_the_study_called_it")


# --- and it still fails closed on everything else ---------------------------


@pytest.mark.parametrize(
    "key",
    [
        "table:cohort_summary",  # a summary OF the cohort, not the cohort
        "table:cohort_attrition",
        "statistic:primary_or",
        "figure:forest",
        "artifact:cohort_defined",  # a status output, named in the directive
        "dataset:something_else",
        "artifact:analysis_cohort_v2",
        "cohort:",  # no product
        ":analysis_cohort",  # no kind
        "analysis_cohort",  # not typed at all
        "",
    ],
)
def test_a_key_outside_the_vocabulary_is_refused(key):
    assert not is_closed_cohort_product_key(key)


# --- the real declaration that lost canary18 --------------------------------


def test_the_canary18_primary_step_is_now_claimed():
    step = _step(
        inputs=[
            "dataset:analysis_cohort",
            "sep3_sofa2_max",
            "death",
            "age",
            "sex",
            "charlson_max",
        ]
    )

    verdict = adjusted_association_executor_verdict(step)
    assert verdict.claimed, verdict.reason


def test_the_same_step_spelled_artifact_was_always_claimed():
    """The control: the fix widened what is read, it did not weaken a clause."""

    step = _step(inputs=["artifact:analysis_cohort", "age"])
    assert adjusted_association_executor_verdict(step).claimed


def test_two_typed_inputs_are_still_refused():
    step = _step(inputs=["dataset:analysis_cohort", "table:cohort_summary"])
    verdict = adjusted_association_executor_verdict(step)
    assert not verdict.claimed


def test_a_typed_input_that_is_not_the_cohort_is_still_refused():
    step = _step(inputs=["table:cohort_summary", "age"])
    assert not adjusted_association_executor_verdict(step).claimed


# --- each caller keeps its own arity policy ---------------------------------


def test_no_typed_input_is_three_valued_at_the_shared_predicate():
    assert sole_typed_cohort_input(_step(inputs=["age", "sex"])) is None
    assert sole_typed_cohort_input(_step(inputs=["dataset:analysis_cohort"])) == (
        "dataset:analysis_cohort"
    )
    assert (
        sole_typed_cohort_input(
            _step(inputs=["dataset:analysis_cohort", "cohort:analysis_set"])
        )
        == ""
    )


def test_the_distribution_owner_still_treats_absent_as_disqualifying():
    """Its rows would have no digest, no contract and no named producer."""

    assert _distribution_typed_cohort_input(_step(inputs=["age"])) == ""
    assert (
        _distribution_typed_cohort_input(_step(inputs=["dataset:analysis_cohort"]))
        == "dataset:analysis_cohort"
    )


def test_the_missingness_audit_still_supports_running_without_one():
    assert _cohort_input_scope(_step(inputs=["age"])) == (True, None)
    assert _cohort_input_scope(_step(inputs=["dataset:analysis_cohort"])) == (
        True,
        "dataset:analysis_cohort",
    )
    assert _cohort_input_scope(
        _step(inputs=["dataset:analysis_cohort", "table:cohort_summary"])
    ) == (False, None)


# --- one implementation, so a fourth copy cannot drift ----------------------


def test_the_policy_is_written_once():
    """Three byte-identical copies existed; a fourth is the recurring defect.

    The literal below is the shape all three carried.  A new copy of it in the
    runners package means the vocabulary has two answers again.
    """

    runners = pathlib.Path("src/easyicu/research_agent/execution/runners").resolve()
    if not runners.is_dir():  # installed-package layout
        pytest.skip("source tree not available")

    copies = [
        path.name
        for path in runners.glob("*.py")
        if 'input_key == "artifact:analysis_cohort"' in path.read_text(encoding="utf-8")
        and path.name != "typed_input_binding.py"
    ]
    assert copies == [], copies


# --- the directive states what is enforced ----------------------------------


@pytest.fixture(scope="module")
def directive() -> str:
    return _build_planner_user_prompt(
        ResearchContext(
            research_question="Is the exposure associated with death?",
            cohort=CohortDescriptor(
                cohort_name="spelling",
                database="test",
                n_patients=10,
                n_stays=10,
                id_columns=["stay_id"],
            ),
            variables=[],
        )
    )


def test_the_directive_offers_every_readable_spelling(directive):
    for spelling in _SPELLINGS_THE_DIRECTIVE_OFFERS:
        assert f"`{spelling}`" in directive, spelling


def test_the_directive_says_downstream_steps_reuse_the_same_key(directive):
    assert "executed by nobody" in directive


def test_the_directive_lists_nothing_the_predicate_cannot_read(directive):
    """A spelling offered but unreadable is exactly what cost canary18."""

    import re

    offered = set(re.findall(r"`((?:artifact|dataset|table|cohort):[^`]+)`", directive))
    cohort_products = {
        key
        for key in offered
        if key.endswith(":analysis_cohort")
        or key.startswith("cohort:")
        or key == "artifact:cohort_defined"
    }
    unreadable = {
        key
        for key in cohort_products
        if key != "artifact:cohort_defined" and not is_closed_cohort_product_key(key)
    }
    assert unreadable == set(), unreadable


# --- the measured corpus number, kept honest --------------------------------


def test_the_corpus_measurement_is_recorded_not_asserted():
    """Documentation, not a gate: the run corpus is not in the repo.

    Measured 2026-07-31 over 194 recorded plans / 2,000 distinct steps:
    42 declarations used ``dataset:analysis_cohort``; reading them moved
    deterministic ownership 528 -> 546 with zero steps losing an owner.
    """

    assert json.dumps({"gained": 18, "lost": 0})
