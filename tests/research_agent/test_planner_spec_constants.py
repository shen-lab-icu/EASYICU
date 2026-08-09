"""A field with one legal value is a constant; the plan must not die retyping it.

MEASURED 2026-07-30, batch ``..._88d3983_canonical9_full02``.  Three of the
nine canonical tasks produced **no plan at all** -- zero steps, zero results --
with ``call_failed:StructuredResponseFailure`` after five planner attempts.
Two of them, ``m3_sepsis_subphenotype`` (25 recorded error lines) and
``h3_trajectory_clustering`` (31), failed almost entirely inside one object:

* 21 of m3's 25 and 24 of h3's 31 error lines are ``trajectory_stability_spec``
  fields reported ``missing`` or ``literal_error``;
* every one of those fields is a ``Literal`` with **exactly one legal value**,
  or ``minimum_successful_resamples``, which a validator already forces to
  equal ``n_resamples``.

``TrajectoryStabilitySpec`` required twenty fields, sixteen of which the
Planner had no freedom over at all -- it had to retype
``label_alignment_tie_break="minimum_rank_distance_then_lexicographic_v1"``
exactly, sixteen times over, or the whole task was lost.  The planner directive
made this worse by spelling all sixteen out as ``key=value`` pairs, which is
what produced the ``literal_error`` half: asked to copy a constant, the model
sometimes wrote a plausible synonym instead.

Every other Planner-owned spec in ``schema.py`` already carried its
single-valued literals as defaults -- fifteen such fields across six classes,
including ``TableOneSpec``.  ``TrajectoryStabilitySpec`` was the only outlier,
so this is an oversight in one class, not a design position.

The fix is subtractive: the constants keep their ``Literal`` type (every other
spelling is still rejected) and are still serialised into the spec and hashed
into its digest -- they simply carry their one legal value as a default, and
the directive stops asking for them.  Required fields go from twenty to
``n_resamples`` plus exactly one of ``sample_fraction``/``sample_size``.

Two invariants below are load-bearing:

* one legal value implies a default -- checked over EVERY model in the schema,
  so the next spec cannot reintroduce this;
* the directive must not spell out a constant the Planner cannot choose, since
  a field it is told to copy is a field it can misspell.
"""

from __future__ import annotations

import inspect
import typing

import pytest
from pydantic import BaseModel, ValidationError

from easyicu.research_agent import schema as S
from easyicu.research_agent.trajectory.plan_contract import (
    trajectory_planner_contract_guide,
)


def _literal_options(annotation: object) -> list[object] | None:
    """Legal values of a ``Literal``, seeing through ``Optional[Literal[...]]``."""

    candidate = annotation
    if typing.get_origin(annotation) is typing.Union:
        for arg in typing.get_args(annotation):
            if typing.get_origin(arg) is typing.Literal:
                candidate = arg
                break
    if typing.get_origin(candidate) is not typing.Literal:
        return None
    return list(typing.get_args(candidate))


def _schema_models() -> list[type[BaseModel]]:
    models = []
    for name in dir(S):
        obj = getattr(S, name)
        if inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel:
            models.append(obj)
    return models


def _single_value_literal_fields() -> list[tuple[str, str, object, bool]]:
    rows = []
    for model in _schema_models():
        for field_name, field in model.model_fields.items():
            options = _literal_options(field.annotation)
            if options is not None and len(options) == 1:
                rows.append(
                    (model.__name__, field_name, options[0], field.is_required())
                )
    return rows


# ---------------------------------------------------------------------------
# The general invariant, over the whole schema
# ---------------------------------------------------------------------------


def test_the_schema_really_has_single_value_literal_fields():
    """Otherwise the invariant below is vacuous."""

    rows = _single_value_literal_fields()
    assert len(rows) >= 20, f"expected many one-value literals, found {len(rows)}"
    assert len({model for model, _, _, _ in rows}) >= 5


def test_one_legal_value_means_the_host_supplies_it():
    """A field the Planner cannot choose must not be a field it can omit.

    This is the whole defect: sixteen such fields were required, so a plan that
    was scientifically complete still failed to validate.
    """

    required = [
        f"{model}.{field}"
        for model, field, _, is_required in _single_value_literal_fields()
        if is_required
    ]
    assert not required, (
        "these fields have exactly one legal value but no default, so the "
        f"Planner must retype a constant to produce a valid plan: {required}"
    )


def test_the_default_is_that_one_legal_value():
    """A default that is not the only legal value would be a different bug."""

    for model_name, field_name, only_value, _ in _single_value_literal_fields():
        model = getattr(S, model_name)
        default = model.model_fields[field_name].default
        assert (
            default == only_value
        ), f"{model_name}.{field_name} defaults to {default!r}"


# ---------------------------------------------------------------------------
# The spec that killed m3 and h3
# ---------------------------------------------------------------------------

# Verbatim from the recorded validation errors of the two dead tasks.  Every
# one of these was reported `missing` or `literal_error`; each must now be
# optional, or the same plan dies the same way.
_FIELDS_THE_DEAD_PLANS_FAILED_ON = (
    "resampling_method",
    "sample_fraction_rounding",
    "base_seed",
    "seed_derivation",
    "cross_resample_membership",
    "stability_metric",
    "stability_aggregation",
    "metric_label_source",
    "evaluation_scope",
    "label_alignment",
    "label_alignment_reference",
    "label_alignment_tie_break",
    "final_assignment_policy",
    "minimum_successful_resamples",
    "failed_refit_policy",
    "refit_engine",
    "refit_initialization",
    "refit_max_iter",
    "refit_tolerance",
    "refit_regularization",
    "decision_mode",
    "threshold_failure_action",
)


@pytest.mark.parametrize("field_name", _FIELDS_THE_DEAD_PLANS_FAILED_ON)
def test_every_field_the_dead_plans_failed_on_is_now_optional(field_name: str):
    field = S.TrajectoryStabilitySpec.model_fields[field_name]
    assert not field.is_required()


def test_the_planner_now_declares_only_its_two_real_decisions():
    """How many resamples, and how large -- nothing else is the study's call."""

    required = {
        name
        for name, field in S.TrajectoryStabilitySpec.model_fields.items()
        if field.is_required()
    }
    assert required == {"n_resamples"}

    spec = S.TrajectoryStabilitySpec(n_resamples=25, sample_fraction=0.8)
    assert spec.resampling_method == "subsample_without_replacement"
    assert spec.stability_metric == "adjusted_rand_index"
    assert spec.minimum_successful_resamples == 25
    assert spec.decision_mode == "report_only"


def test_the_sample_size_choice_is_still_the_planners():
    """The one thing that is genuinely a design decision stays mandatory."""

    with pytest.raises(ValidationError, match="exactly one of sample_fraction"):
        S.TrajectoryStabilitySpec(n_resamples=25)
    with pytest.raises(ValidationError, match="exactly one of sample_fraction"):
        S.TrajectoryStabilitySpec(n_resamples=25, sample_fraction=0.8, sample_size=50)


def test_a_derived_field_still_refuses_a_disagreeing_declaration():
    """Defaulting is not accepting: an explicit wrong value is still an error."""

    with pytest.raises(ValidationError, match="must equal n_resamples"):
        S.TrajectoryStabilitySpec(
            n_resamples=25, sample_fraction=0.8, minimum_successful_resamples=20
        )


def test_a_constant_is_still_closed_to_every_other_spelling():
    """The defaults must not have turned the contract into a suggestion."""

    with pytest.raises(ValidationError):
        S.TrajectoryStabilitySpec(
            n_resamples=25, sample_fraction=0.8, stability_metric="silhouette"
        )
    with pytest.raises(ValidationError):
        S.TrajectoryStabilitySpec(
            n_resamples=25,
            sample_fraction=0.8,
            resampling_method="bootstrap_with_replacement",
        )


def test_an_invented_field_is_still_rejected_not_ignored():
    """m3 and h3 also invented fields; silently dropping them would be worse."""

    with pytest.raises(ValidationError):
        S.TrajectoryStabilitySpec(
            n_resamples=25, sample_fraction=0.8, candidate_cluster_counts=[2, 3, 4]
        )


def test_the_stability_threshold_is_still_a_scientific_choice():
    """decision_mode restates the threshold, so it is derived -- but the
    threshold itself, which does make a claim, is not invented for the study."""

    reported = S.TrajectoryStabilitySpec(n_resamples=25, sample_fraction=0.8)
    assert reported.minimum_mean_stability is None
    assert reported.decision_mode == "report_only"

    gated = S.TrajectoryStabilitySpec(
        n_resamples=25, sample_fraction=0.8, minimum_mean_stability=0.7
    )
    assert gated.decision_mode == "minimum_mean_threshold"

    with pytest.raises(ValidationError, match="must not declare a binary threshold"):
        S.TrajectoryStabilitySpec(
            n_resamples=25,
            sample_fraction=0.8,
            decision_mode="report_only",
            minimum_mean_stability=0.7,
        )
    with pytest.raises(ValidationError, match="requires minimum_mean_stability"):
        S.TrajectoryStabilitySpec(
            n_resamples=25,
            sample_fraction=0.8,
            decision_mode="minimum_mean_threshold",
        )


# ---------------------------------------------------------------------------
# The directive the Planner reads
# ---------------------------------------------------------------------------


def _trajectory_directive() -> str:
    context = S.ResearchContext(
        research_question="Cluster organ-dysfunction trajectories into subphenotypes.",
        cohort=S.CohortDescriptor(
            cohort_name="c", database="synthetic", n_patients=100, n_stays=100
        ),
        variables=[
            S.ConceptDescriptor(
                name="sofa2_cardio_w0",
                role=S.VariableRole.ORDINAL_SCORE,
                dtype="float64",
                fixed_window_trajectory=S.FixedWindowTrajectoryMetadata(
                    family="sofa2_cardio",
                    window_start_hours=0.0,
                    window_end_hours=24.0,
                    window_width_hours=24.0,
                    representation_kind="fractional_window_summary",
                ),
            )
        ],
        target_outcome="death",
    )
    guide = trajectory_planner_contract_guide(
        context=context, analysis_type="trajectory_clustering"
    )
    assert guide, "the directive must render, or the assertions below are vacuous"
    return guide


def test_the_directive_still_describes_the_stability_spec():
    assert "trajectory_stability_spec" in _trajectory_directive()


def test_the_directive_still_asks_for_the_real_decisions():
    guide = _trajectory_directive()
    for token in ("n_resamples", "sample_fraction", "sample_size"):
        assert token in guide


@pytest.mark.parametrize(
    "field_name,only_value",
    [
        (field, value)
        for model, field, value, _ in _single_value_literal_fields()
        if model == "TrajectoryStabilitySpec"
    ],
)
def test_the_directive_does_not_dictate_a_constant_it_could_misspell(
    field_name: str, only_value: object
):
    """The ``literal_error`` half of the defect.

    The old directive listed every constant as ``field=value``.  Told to copy
    ``seed_derivation=numpy_seedsequence_spawn_uint32_v1``, the Planner
    sometimes wrote something else -- and a near-miss on a field it never had a
    choice about cost the entire task.  A field the host fills in must not
    appear in the directive as an instruction to write.
    """

    guide = _trajectory_directive()
    assert f"{field_name}=" not in guide
    if "_" in str(only_value):
        # Skip plain English values like "floor"/"mean", which occur in prose
        # for unrelated reasons; the identifier-shaped ones are the constants
        # a model can only reproduce by copying, and those must be absent.
        assert str(only_value) not in guide
