"""A rank stored as a float is still a rank, and the record has to say so.

MEASURED on the five never-passing canonical tasks: 6 of 29 scientific blocking
findings are an ordinal score used as an interval measurement --

* "gcs_first is used as a continuous numeric covariate in propensity-score
  fitting and standardized mean differences. GCS is ordinal; this imposes
  unjustified equal-interval assumptions."
* "Ordinal SOFA summaries are passed to finite_summary, which reports arithmetic
  means for sofa2_first, component first/max values, and therefore averages
  ordinal scores rather than preserving rank summaries."
* "medians of ordinal levels can become fractional and are then emitted as
  invalid/non-rank-preserving SOFA representations"
* "Availability is based only on nonmissingness for ordinal SOFA fields; invalid
  ordinal levels would be counted as available without domain validation."

The auditor was RIGHT in every one of those. This is not a false-block class like
the censoring rule was; the defects were real. What was missing is on the other
side: the generated code had no machine-readable way to know.

The concept layer knows all of it. ``gcs_max`` arrives with
``role="ordinal_score"``, ``valid_range=[3.0, 15.0]``, and the pitfall "GCS is
ordinal; do not take its mean."

The record the script opens knows none of it. ``product_contract`` is derived
from the artifact file alone, so it publishes closed value sets for string
categoricals (``adm``, ``sex``) and lists every ordinal in ``numeric_columns``
next to lactate -- ``if name in numeric_set: continue`` is the line an ordinal
falls through. A script reading that record sees ``float32`` and averages it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from easyicu.research_agent.authority.typed_binding import (
    _RANK_SCALE_VARIABLE_ROLES,
    _write_resolved_inputs_manifest,
    rank_scale_columns_entry,
)
from easyicu.research_agent.schema import (
    AggregationRule,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context(*variables: ConceptDescriptor) -> ResearchContext:
    return ResearchContext(
        research_question="q",
        cohort=CohortDescriptor(
            cohort_name="c", database="miiv", n_patients=1, n_stays=1
        ),
        variables=list(variables),
    )


def _gcs() -> ConceptDescriptor:
    """The real descriptor shape, from a recorded run's context."""

    return ConceptDescriptor(
        name="gcs_max",
        description="Glasgow Coma Scale, worst in window",
        dtype="float32",
        role=VariableRole.ORDINAL_SCORE,
        valid_range=[3.0, 15.0],
        observed_domain={"n_unique": 13, "min": 3.0, "max": 15.0, "is_binary": False},
        aggregation_default=AggregationRule.MAX_LAST,
    )


def _lactate() -> ConceptDescriptor:
    return ConceptDescriptor(
        name="lact_max",
        description="Lactate, max in window",
        dtype="float64",
        role=VariableRole.LAB,
        unit="mmol/L",
        valid_range=[0.1, 30.0],
    )


def test_an_ordinal_column_is_declared_a_rank() -> None:
    entry = rank_scale_columns_entry(_context(_gcs()))
    assert entry is not None
    assert set(entry["columns"]) == {"gcs_max"}
    assert entry["columns"]["gcs_max"]["role"] == "ordinal_score"


def test_a_continuous_measurement_is_not_declared_a_rank() -> None:
    """The distinction is the whole point: both are float columns.

    A declaration that swept in every numeric column would tell the script not
    to average lactate, which is wrong and would be obeyed.
    """

    assert rank_scale_columns_entry(_context(_lactate())) is None
    entry = rank_scale_columns_entry(_context(_gcs(), _lactate()))
    assert entry is not None
    assert set(entry["columns"]) == {"gcs_max"}


def test_the_role_comes_from_the_concept_layer_not_a_name_list() -> None:
    """No score-name list anywhere: the dictionary already assigns the role.

    A name list here would be a second, divergent opinion about what GCS is --
    and it would silently miss every ordinal the dictionary knows about and this
    file's author did not think of.
    """

    assert _RANK_SCALE_VARIABLE_ROLES == frozenset({VariableRole.ORDINAL_SCORE.value})
    # A column NAMED like a score but typed as a lab is not swept in by its name.
    mislabelled = _lactate().model_copy(update={"name": "sofa_max"})
    assert rank_scale_columns_entry(_context(mislabelled)) is None


def test_both_the_legal_domain_and_the_observed_one_are_published() -> None:
    """Different facts, both needed.

    ``valid_range`` says which values are legal levels; the observed domain says
    which of them this cohort contains. A check written against only the second
    passes a cohort that happens to be clean and misses the invalid level the
    audit asked about.
    """

    entry = rank_scale_columns_entry(_context(_gcs()))
    assert entry is not None
    column = entry["columns"]["gcs_max"]
    assert column["valid_range"] == [3.0, 15.0]
    assert column["observed_min"] == 3.0
    assert column["observed_max"] == 15.0
    assert column["observed_n_unique"] == 13


def test_an_observed_level_set_is_published_when_the_context_has_one() -> None:
    """Measured: SOFA components arrive with `levels: [0,1,2,3,4]`.

    The exact list is what lets a script check its own output for a value
    between two levels -- the h3 defect where per-hour medians became fractional.
    """

    sofa = _gcs().model_copy(
        update={
            "name": "sofa2_resp_max",
            "valid_range": [0.0, 4.0],
            "observed_domain": {
                "n_unique": 5,
                "min": 0.0,
                "max": 4.0,
                "levels": [0.0, 1.0, 2.0, 3.0, 4.0],
            },
        }
    )
    entry = rank_scale_columns_entry(_context(sofa))
    assert entry is not None
    assert entry["columns"]["sofa2_resp_max"]["observed_levels"] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_the_aggregation_rule_is_published_by_value_not_by_repr() -> None:
    """An enum stringifies to `AggregationRule.MAX_LAST`.

    That is a Python identifier, not the vocabulary the rest of the record uses.
    A reader matching it against the aggregation names it knows finds no match
    and falls back to choosing one -- which is the behaviour being removed.
    """

    entry = rank_scale_columns_entry(_context(_gcs()))
    assert entry is not None
    published = entry["columns"]["gcs_max"]["aggregation_default"]
    assert published == AggregationRule.MAX_LAST.value
    assert "AggregationRule" not in published


def test_the_authorization_names_the_reason_the_dtype_misleads() -> None:
    entry = rank_scale_columns_entry(_context(_gcs()))
    assert entry is not None
    authorization = entry["authorization"]
    # Why the column is sitting among the numeric ones, so the script does not
    # read that placement as permission.
    assert "fact about storage and not about the scale" in authorization
    # The positive instruction, not only the prohibition.
    assert "rank-preservingly" in authorization
    # The specific failure the h3 finding described.
    assert "land between two levels" in authorization
    # And the fail-closed rule for an out-of-domain value.
    assert "must stop the step, not be counted as available" in authorization
    # The one permitted numeric use, so the record does not contradict the
    # system rule that allows a declared coding.
    assert "states that coding" in authorization


def test_nothing_is_published_when_no_column_is_a_rank() -> None:
    """`None`, not an empty declaration.

    An entry naming no column publishes "nothing here is a rank" with the host's
    authority. That exact shape shipped once this week -- an ambient-trajectory
    entry with `"concepts": []` under a sentence promising completeness -- and
    was reverted.
    """

    assert rank_scale_columns_entry(_context(_lactate())) is None
    assert rank_scale_columns_entry(_context()) is None
    assert rank_scale_columns_entry(None) is None


def test_an_empty_declaration_is_refused_by_the_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one column"):
        _write_resolved_inputs_manifest(
            run_dir=tmp_path,
            step_id="s",
            planner_declared_inputs=[],
            bindings={},
            rank_scale_columns={"columns": {}},
        )


def test_the_declaration_reaches_the_step_record(tmp_path: Path) -> None:
    manifest = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="04_primary_model",
        planner_declared_inputs=[],
        bindings={},
        rank_scale_columns=rank_scale_columns_entry(_context(_gcs(), _lactate())),
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    declared = payload["rank_scale_columns"]
    assert declared["schema_version"] == "easyicu.rank_scale_columns/1"
    assert set(declared["columns"]) == {"gcs_max"}


def test_the_key_is_absent_when_nothing_was_declared(tmp_path: Path) -> None:
    manifest = _write_resolved_inputs_manifest(
        run_dir=tmp_path,
        step_id="04_primary_model",
        planner_declared_inputs=[],
        bindings={},
        rank_scale_columns=rank_scale_columns_entry(_context(_lactate())),
    )
    assert "rank_scale_columns" not in json.loads(
        manifest.read_text(encoding="utf-8")
    )


def _phase_rank_call_arguments() -> list[str]:
    """What `phase.py` passes to the builder, located structurally.

    Every test above calls the builder itself, so they hold the content but say
    nothing about whether a run ever publishes it. A mutation that unwired the
    call site left an identically-shaped set of tests green earlier today.
    """

    import ast
    import inspect

    from easyicu.research_agent.execution import phase

    found: list[str] = []
    for node in ast.walk(ast.parse(inspect.getsource(phase))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "rank_scale_columns_entry":
            continue
        assert len(node.args) == 1
        found.append(ast.unparse(node.args[0]))
    return found


def test_the_declaration_covers_cohort_columns_and_says_nothing_about_long_rows() -> None:
    """The known boundary, stated rather than implied.

    MEASURED across the nine recorded contexts: this entry declares 18 rank
    columns for e1/h3/m1, 15 for e3, 12 for h1, 3 for h2 -- and **0** for m3,
    whose SOFA values arrive through the long trajectory table and therefore have
    no ``ConceptDescriptor`` in ``context.variables`` at all.

    So 5 of the 6 measured ordinal-as-interval findings are covered (the ones
    naming wide columns: ``gcs_first``, ``sofa2_first``, component
    ``_first``/``_max``) and the sixth -- "per-hour median aggregation" over long
    rows -- is not. The long table's ordinal facts are a property of its concept
    ids, not of a cohort column, and declaring them is separate work. This test
    exists so that gap is a recorded boundary rather than an assumption someone
    later reads out of the coverage number.
    """

    long_only = _context(
        _lactate(),
        ConceptDescriptor(
            name="stay_id", description="id", dtype="int64", role=VariableRole.ID
        ),
    )
    assert rank_scale_columns_entry(long_only) is None


def test_the_execute_phase_publishes_it_from_the_unscoped_context() -> None:
    """The roles are a property of the columns, not of one step's declarations.

    The step-scoped projection drops columns the step reads through an artifact
    rather than naming -- and a step consuming the analysis cohort names almost
    nothing. Building from the projection is how a sibling entry published an
    empty list under a promise of completeness.
    """

    arguments = _phase_rank_call_arguments()
    assert arguments == ["context"], arguments
