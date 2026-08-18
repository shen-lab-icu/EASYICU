"""The host demanded a field the plan had already filled in.

A step promised six robustness products and declared a
``robustness_replay_spec`` naming two of them.  The host refused it -- correctly:
without an entry it cannot tell which replay output the other four are, and
handing that to the Coder is the fail-open the gate exists to close.

What it TOLD the Planner was ``robustness_replay_spec.products``.  That field
was already there, with two entries in it.  So the plan-time gate's own
instruction -- "Each finding names a step and the exact field(s) it left
undeclared; add those to the step that already exists" -- resolved to "add
``products``", which was present.  The forced replan changed nothing, the step
reached execution still under-declared, and it was blocked along with the two
steps downstream of it.  On canary30 that is what stopped E1.

The information was never missing: the verdict's ``reason`` already listed the
unbacked products by name.  Only the machine-readable ``missing`` list, which is
what both the plan-time directive and the execution-time refusal message are
built from, collapsed all four into one field path.

So the verdict now reports one entry per unbacked product.  Both gates share
that list, so both become actionable at once.

Safety, checked rather than assumed: ``_prohibited_choices`` subtracts every
demanded field from the "do not change the science to satisfy this" list, keyed
on the normalised leaf of the path.  ``products['primary_or']`` normalises to
``product`` -- ``_declared_choice`` cuts at ``[`` -- which is not a scientific
choice, so nothing this adds can quietly un-forbid one.  A dotted spelling would
have put the product's own name in that position.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.execution.runners.deterministic_robustness import (
    robustness_replay_declaration_verdict,
    robustness_replay_spec_is_emittable,
)
from easyicu.research_agent.execution.owner_declaration import (
    _declared_choice,
    _prohibited_choices,
)
from easyicu.research_agent.plan_utils import (
    _enforce_advanced_plan_contract,
    _split_table_and_figure_outputs_in_plan,
)
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ResearchContext,
    RobustnessReplaySpec,
    UserPreferences,
)


def _step(outputs, *, products) -> AnalysisStep:
    return AnalysisStep(
        step_id="09_robustness_replay",
        planned_analysis_role="sensitivity",
        intent="Replay the pre-specified robustness grid on the locked cohort.",
        inputs=["artifact:analysis_cohort"],
        expected_outputs=list(outputs),
        method="robustness_sensitivity",
        robustness_replay_spec=RobustnessReplaySpec(products=list(products)),
    )


#: canary30's real declaration: six promised, two backed.
_PROMISED = [
    "statistic:primary_or",
    "statistic:complete_case_n",
    "table:robustness_summary",
    "log:missingness_strategy_notes",
    "table:robustness_matrix",
]
_BACKED = [
    {"product_id": "robustness_matrix", "output": "robustness_matrix"},
    {"product_id": "robustness_summary", "output": "robustness_summary"},
]


def _missing() -> tuple[str, ...]:
    verdict = robustness_replay_declaration_verdict(_step(_PROMISED, products=_BACKED))
    return tuple(verdict.missing_declarations)


def test_every_unbacked_product_is_named_on_its_own() -> None:
    """The property that was false: four gaps reported as one field."""

    missing = _missing()
    assert len(missing) == 3, missing
    for product in ("primary_or", "complete_case_n", "missingness_strategy_notes"):
        assert any(product in name for name in missing), (product, missing)


def test_the_field_that_already_exists_is_not_demanded_on_its_own() -> None:
    """The defect itself.

    Asking for ``robustness_replay_spec.products`` when the plan has it is an
    instruction with nothing to do, and the recorded replan did nothing.
    """

    assert "robustness_replay_spec.products" not in _missing()


def test_a_product_the_spec_does_back_is_not_demanded() -> None:
    """Only the gaps. Naming a backed product would send the Planner to edit
    an entry that is already correct."""

    missing = _missing()
    assert not any("robustness_matrix" in name for name in missing), missing
    assert not any("robustness_summary" in name for name in missing), missing


def test_a_fully_backed_spec_reports_no_gap_at_all() -> None:
    """The gate must not start firing on complete declarations."""

    verdict = robustness_replay_declaration_verdict(
        _step(
            ["table:robustness_matrix", "table:robustness_summary"],
            products=_BACKED,
        )
    )
    assert not verdict.missing_declarations


def test_host_output_augmentation_updates_the_existing_replay_spec_atomically() -> None:
    step = _step(
        ["table:robustness_matrix", "table:robustness_summary"],
        products=_BACKED,
    )
    plan = AnalysisPlan(
        research_question="Replay the prespecified robustness grid.",
        analysis_type="association_study",
        steps=[step],
    )
    context = ResearchContext(
        research_question=plan.research_question,
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="synthetic",
            n_stays=100,
        ),
        variables=[],
        user_preferences=UserPreferences(inferred_analysis_family="robustness"),
    )

    revised, _findings = _enforce_advanced_plan_contract(
        plan=plan,
        context=context,
    )

    revised_step = revised.steps[0]
    assert revised_step.robustness_replay_spec is not None
    assert {
        item.product_id: item.output
        for item in revised_step.robustness_replay_spec.products
    } == {
        "robustness_matrix": "robustness_matrix",
        "robustness_summary": "robustness_summary",
        "primary_or": "primary_effect",
        "complete_case_n": "complete_case_n",
        "missingness_strategy_notes": "missingness_strategy_notes",
    }
    split, _split_findings = _split_table_and_figure_outputs_in_plan(revised)
    assert robustness_replay_spec_is_emittable(split.steps[0])


def test_naming_products_cannot_unforbid_a_scientific_choice() -> None:
    """The safety property, checked on the real helper.

    ``_prohibited_choices`` subtracts demanded fields from the prohibition. If a
    product name landed in the leaf position, a product called ``outcome_*``
    would delete ``outcome`` from the list and the directive would tell the
    Planner it may change the outcome to satisfy a bookkeeping check.
    """

    baseline = _prohibited_choices(["robustness_replay_spec.products"])
    assert _prohibited_choices(_missing()) == baseline
    for name in _missing():
        assert _declared_choice(name) == "product", (name, _declared_choice(name))


def test_a_product_named_for_a_scientific_choice_still_cannot_unforbid_it() -> None:
    """The same property where it would actually bite.

    Product names come from the Planner, so one CAN be called
    ``outcome_label_executability``. The bracket spelling is what keeps that in
    the index position rather than the leaf.
    """

    verdict = robustness_replay_declaration_verdict(
        _step(
            ["table:outcome_label_executability", "table:robustness_matrix"],
            products=[
                {"product_id": "robustness_matrix", "output": "robustness_matrix"}
            ],
        )
    )
    missing = tuple(verdict.missing_declarations)
    assert missing, "the unbacked product is no longer reported"
    assert "outcome" in _prohibited_choices(missing), (
        "a product named for a scientific choice deleted that choice from the "
        f"prohibition: {missing} -> {_prohibited_choices(missing)}"
    )


def test_the_reason_still_explains_what_the_entries_are_for() -> None:
    """The list says what to add; the prose still has to say why."""

    verdict = robustness_replay_declaration_verdict(_step(_PROMISED, products=_BACKED))
    assert "which replay output" in (verdict.reason or "")


# --- the recorded corpus ------------------------------------------------------

_CORPUS = Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")


@pytest.mark.skipif(
    not _CORPUS.exists(), reason="recorded runs are not on this machine"
)
def test_the_recorded_refusal_asked_for_a_field_the_plan_already_had() -> None:
    """Real bytes: the defect, where it actually blocked a run.

    Every recorded step told to add ``robustness_replay_spec.products`` must
    turn out to have had that field populated -- otherwise the demand was
    correct and this fix is answering a question nobody asked.
    """

    empty_field = []
    already_populated = 0
    for manifest_path in sorted(_CORPUS.glob("batch_*/*/aware/run_*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            continue
        for record in manifest.get("per_step_records") or []:
            if not isinstance(record, dict):
                continue
            missing_by_owner = record.get("owner_declaration_missing")
            if not isinstance(missing_by_owner, dict):
                continue
            demanded = {
                name for names in missing_by_owner.values() for name in (names or [])
            }
            if "robustness_replay_spec.products" not in demanded:
                continue
            step = ((record.get("analysis_request") or {}).get("step")) or {}
            spec = step.get("robustness_replay_spec") or {}
            if spec.get("products"):
                already_populated += 1
            else:
                empty_field.append(record.get("step_id"))

    if not already_populated and not empty_field:
        pytest.skip("no recorded step was refused for this field")
    assert not empty_field, (
        "a recorded step was told to add this field and genuinely did not have "
        f"it, so the old demand was right for that case: {empty_field[:5]}"
    )
