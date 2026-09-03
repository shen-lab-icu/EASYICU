"""The host tells the planner ``forest_plot`` renders, then refuses the step for it.

Two functions in this codebase answer the same question -- "is this step
rendering-only, not a model with a figure output?" -- and they disagree.

``research_context/prompt_scope.py::_figure_contract_applies`` shapes what the
Planner is told, and reads::

    method in _FIGURE_METHODS or (output_kinds and output_kinds <= figure_kinds)

-- a name **or** the typed structure.  Its ``_FIGURE_METHODS`` contains
``forest_plot`` and ``kaplan_meier_plot``.

``plan_utils.py::_effect_figure_source_authorized`` judges the finished step,
and reads::

    if _normalised_method_head(...) not in _FIGURE_METHODS or <structure> ...
        return False

-- a name **and** the structure, with the name as a veto.  Its own, separate
``_FIGURE_METHODS`` does not contain ``forest_plot``.  So a render child that
uses the standard name for exactly this figure is refused
``unauthorized_effect_product`` no matter how perfectly it is bound.

MEASURED on the recorded corpus, by replaying the real authority function over
1,016 real steps with real bindings: dropping the name veto flips exactly 3
steps from refused to authorized and 0 the other way.  All 3 declare a single
``figure:`` product whose single input is the parent's typed effect table:

    m1  06_adjusted_association_figure  forest_plot
    e1  08_primary_effect_figure        host_rendered_visualization

Nothing that analyses anything flips, because the structural conditions beside
the veto already carry the whole load: every declared output must be
``figure:``/``log:``, and every input must be a typed ``table:``/``statistic:``
bound by exact evidence id and sha256 to a successful effect parent.  A step
built that way has no cohort to re-analyse.

So the veto is replaced, not deleted.  What it was reaching for -- a render
child must not claim to be the analysis -- is kept and grounded in the host's
own vocabulary: ``_EFFECT_CONTRACT_METHODS``, the set
``effect_output_authorized`` itself uses to decide who owns an effect result.
A child naming one of those is still refused; a child naming a figure is not.

m1 is 8 of 9 steps green with this refusal as its only remaining failure.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from easyicu.research_agent.plan_utils import (
    _EFFECT_CONTRACT_METHODS,
    _ROBUSTNESS_EFFECT_CONTRACT_METHODS,
    _effect_figure_source_authorized,
    effect_output_authorized,
)
from easyicu.research_agent.research_context.prompt_scope import (
    _FIGURE_METHODS as _PLANNER_FACING_FIGURE_METHODS,
)
from easyicu.research_agent.schema import AnalysisStep

from tests.research_agent.gates.test_declared_product_contract import (
    _effect_parent_and_figure_child,
    _resolved_render_bindings,
)

_CORPUS = pathlib.Path("/Volumes/外置硬盘/easyicu_data/canonical9_runs")

#: The exact nine spellings the replaced veto accepted.  Transcribed rather
#: than imported, so that shrinking the production set cannot quietly shrink
#: what this file claims was refused.
_ACCEPTED_BY_THE_REPLACED_LIST = frozenset(
    {
        "figure",
        "visualization",
        "visualisation",
        "plotting",
        "publication_figure",
        "publication_figure_generation",
        "render_figure",
        "figure_generation",
        "chart_generation",
    }
)


def _VETOED_BY_THE_REPLACED_LIST(method: str) -> bool:  # noqa: N802
    return method not in _ACCEPTED_BY_THE_REPLACED_LIST


def _authorized(method: str) -> bool:
    parent, child, record = _effect_parent_and_figure_child()
    assert effect_output_authorized(parent) is True
    child = child.model_copy(update={"method": method})
    return _effect_figure_source_authorized(
        step=child,
        completed_step_records=[record],
        resolved_input_bindings=_resolved_render_bindings(child),
    )


def _m1_recorded_child() -> tuple[AnalysisStep, AnalysisStep, dict]:
    """m1's real step 06 and its real parent, transcribed from the run record.

    Values come from ``analysis_plan_revision_4.json`` and
    ``resolved_inputs/06_adjusted_association_figure.json`` of
    ``batch_20260803_..._verify10``. The parent is authorized through its
    planner-owned model roster, not a method name, which is why the child's
    name is the only thing left to refuse.
    """

    parent = AnalysisStep(
        step_id="05_primary_adjusted_association",
        method="adjusted_association_models",
        intent="Estimate the adjusted association between bilirubin and mortality.",
        inputs=["artifact:analysis_cohort", "bili_max", "death"],
        expected_outputs=["table:adjusted_association_estimates"],
        model_requirements=[
            {
                "requirement_id": "primary_bilirubin_mortality_logistic",
                "outcome": "death",
                "outcome_type": "binary",
                "method_family": "logistic_regression",
                "exposure_source": "bili_max",
                "analysis_role": "primary",
                "analysis_set": "complete_case",
                "required_for_step_success": True,
            }
        ],
    )
    child = AnalysisStep(
        step_id="06_adjusted_association_figure",
        method="forest_plot",
        intent="Display the primary adjusted association and its uncertainty.",
        inputs=["table:adjusted_association_estimates"],
        expected_outputs=["figure:adjusted_association"],
    )
    record = {
        "step_id": parent.step_id,
        "status": "ok",
        "analysis_request": {"step": parent.model_dump(mode="json")},
        "step_summary": {
            "status": "ok",
            "output_files": {
                "table:adjusted_association_estimates": (
                    "adjusted_association_estimates.csv"
                )
            },
        },
    }
    return parent, child, record


def test_the_recorded_m1_figure_step_is_authorized():
    """The exact shape that failed the live run, reconstructed from its record."""

    parent, child, record = _m1_recorded_child()

    assert effect_output_authorized(parent) is True, "the parent must own the effect"
    assert effect_output_authorized(child) is False, "the child must not own it"
    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(
                child, producer_step_id=parent.step_id
            ),
        )
        is True
    )


def test_every_method_the_planner_is_told_renders_can_render():
    """The anchor: the two host layers must not disagree about one word.

    ``prompt_scope`` decides what the Planner is told a rendering step is.  A
    name it accepts there and this layer vetoes is the host refusing its own
    instruction.
    """

    refused = sorted(
        method for method in _PLANNER_FACING_FIGURE_METHODS if not _authorized(method)
    )

    assert refused == [], refused


def test_a_child_naming_an_effect_owning_method_is_still_refused():
    """The property the deleted veto was reaching for, kept and grounded.

    Not a new list: this is the same set ``effect_output_authorized`` consults
    to decide who may own an effect result.
    """

    for method in sorted(_EFFECT_CONTRACT_METHODS):
        assert _authorized(method) is False, method
    for method in sorted(_ROBUSTNESS_EFFECT_CONTRACT_METHODS):
        assert _authorized(method) is False, method


def test_the_refusal_reads_the_shared_vocabulary_not_a_copy():
    """A second private list would drift the same way the first one did."""

    import inspect

    from easyicu.research_agent import plan_utils

    source = inspect.getsource(plan_utils._effect_figure_source_authorized)
    assert "_FIGURE_METHODS" not in source, source
    assert "_EFFECT_CONTRACT_METHODS" in source, source
    # And the vocabulary really is the one the shared predicate uses.
    assert "logistic_regression" in _EFFECT_CONTRACT_METHODS


def test_an_unknown_rendering_name_is_decided_by_structure_alone():
    """Two identical children differing only in a non-effect name agree.

    ``host_rendered_visualization`` is a name the Planner invented; it appears
    in the recorded corpus and no list anywhere contains it. Structure, not
    spelling, must decide it.
    """

    assert _authorized("visualization") is True
    assert _authorized("host_rendered_visualization") is True
    assert _authorized("a_name_no_list_will_ever_contain") is True


def test_a_child_that_could_re_analyse_is_still_refused():
    """The structural conditions must still carry the load they always did."""

    parent, child, record = _effect_parent_and_figure_child()

    with_cohort = child.model_copy(
        update={"inputs": [*(child.inputs or []), "artifact:analysis_cohort"]}
    )
    assert (
        _effect_figure_source_authorized(
            step=with_cohort,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(with_cohort),
        )
        is False
    )

    owning_a_table = child.model_copy(
        update={
            "expected_outputs": [
                *(child.expected_outputs or []),
                "table:primary_association",
            ]
        }
    )
    assert (
        _effect_figure_source_authorized(
            step=owning_a_table,
            completed_step_records=[record],
            resolved_input_bindings=_resolved_render_bindings(owning_a_table),
        )
        is False
    )


def test_a_failed_parent_still_refuses_its_render_child():
    """Dropping the name veto must not weaken the parent-success requirement."""

    parent, child, record = _effect_parent_and_figure_child()
    child = child.model_copy(update={"method": "forest_plot"})
    failed = {**record, "status": "contract_failed"}

    assert (
        _effect_figure_source_authorized(
            step=child,
            completed_step_records=[record, failed],
            resolved_input_bindings=_resolved_render_bindings(child),
        )
        is False
    )


def test_the_recorded_corpus_contains_the_vetoed_shape():
    """Re-measures rather than restating: the defect was real in real runs."""

    if not _CORPUS.exists():
        pytest.skip("recorded run corpus is not mounted")

    vetoed: set[tuple[str, str]] = set()
    for plan_path in _CORPUS.glob("batch_*/*/aware/run_*/analysis_plan*.json"):
        try:
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for raw in plan.get("steps", []):
            outputs = [str(item) for item in (raw.get("expected_outputs") or [])]
            if not outputs:
                continue
            if not all(
                item.lower().startswith(("figure:", "log:")) for item in outputs
            ):
                continue
            if not any(item.lower().startswith("figure:") for item in outputs):
                continue
            inputs = [str(item) for item in (raw.get("inputs") or [])]
            if not inputs or not all(
                item.lower().startswith(("table:", "statistic:")) for item in inputs
            ):
                continue
            method = str(raw.get("method") or "")
            if not method or not _VETOED_BY_THE_REPLACED_LIST(method):
                continue
            vetoed.add((method, str(raw.get("step_id") or "")))

    if not vetoed:
        pytest.skip("no recorded plan carries a figure-only step")
    names = sorted({method for method, _ in vetoed})
    # The name that failed m1's live run, and one the Planner invented.
    assert "forest_plot" in names, names
    assert "host_rendered_visualization" in names, names
