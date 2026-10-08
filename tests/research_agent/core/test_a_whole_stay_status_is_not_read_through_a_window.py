"""A window on a status the input records only for the whole stay is refused.

The cohort builder reads a predicate that tests an event over a finite window
by the event's own time, ``<concept>_time``.  An input that records the event
only as a whole-stay status, with no such column, was read over the whole stay
instead: an exclusion of the event within 24 h removed every stay with the
event, however late, and nothing said so.  Plan review refused it only when
the plan had a time zero, so any other plan, a benchmark's included, and the
materializer's own filter kept the other reading.

A column the context records an event over the whole stay in (an outcome
without its own analysis window, or the status of a typed event time) is now
read through a finite window only beside its time:

- the primary cohort is refused at planning, with a correction to state the
  criterion as not applied;
- the builder refuses it wherever a cohort is built from such an input: the
  locked analysis cohort, the materializer's own filter and every robustness
  override;
- a robustness override is a major review finding the Agent revises, and at
  execution its specification has no estimate;
- a specification without an estimate is in no count of sensitivity analyses.

Synthetic tables; the status is a generic event unless a test needs an export.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from easyicu.research_agent.cohort.materializer import (
    materialize_cohort,
    materialize_to_parquet,
)
from easyicu.research_agent.cohort.schema import (
    COHORT_EVENT_WINDOW_UNREADABLE,
    CohortDataError,
    CohortEventWindowUnreadableError,
    _build_cohort_with_flow,
    materialize_locked_analysis_cohort,
    predicates_read_over_the_whole_stay,
    validate_plan_typed_bindings_against_context,
)
from easyicu.research_agent.execution.runners.deterministic_robustness import (
    _replay_primary_model_for_cohort,
)
from easyicu.research_agent.intake.materialized_metadata import (
    MaterializedMetadataError,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    CohortSchemaError,
    ConceptPredicate,
    TimeWindow,
    cohort_concept_id_scope,
)
from easyicu.research_agent.research_context.stay_events import (
    event_time_status_column,
    stay_outcome_columns,
    whole_stay_event_columns,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.planning.scientific_review import (
    build_plan_scientific_review,
    remediation_route_for_finding,
    robustness_override_event_window_findings,
)
from easyicu.research_agent.providers.structured_diagnostics import (
    infer_validation_stage,
)
from easyicu.research_agent.reporting.writer_evidence import (
    _render_robustness_panel_block,
)
from easyicu.research_agent.robustness import runtime_panel
from easyicu.research_agent.robustness.estimators import (
    _data_with_predicate_aliases,
    _fit_one_row,
    fit_estimator,
)
from easyicu.research_agent.robustness.membership import _membership_audit
from easyicu.research_agent.robustness.panel import (
    RobustnessPanel,
    RobustnessPanelRow,
    numeric_digest_for_panel,
    row_was_estimated,
    unexecuted_locked_spec_ids,
)
from easyicu.research_agent.schema import (
    ConceptDescriptor,
    ObservationSemantics,
    VariableRole,
)
from tests.research_agent.planning.progressive_planner_fixtures import (
    _context as _planner_context,
)
from tests.support.native_outcome_export import (
    native_outcome,
    typed_native_export,
    untyped_native_export,
)

_EVENT = "event_flag"
_CODE = "ROBUSTNESS_OVERRIDE_EVENT_WINDOW_UNREADABLE"


@pytest.fixture(autouse=True)
def _generic_concepts():
    with cohort_concept_id_scope([_EVENT, "event_alias", "outcome_flag", "age_years"]):
        yield


def _predicate(concept: str, op: str, value, *, end: float = 24.0) -> ConceptPredicate:
    return ConceptPredicate(
        concept_id=concept,
        time_window=TimeWindow(
            anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=end
        ),
        aggregation="max",
        op=op,
        value=value,
    )


def _cohort(*, inclusion=(), exclusion=(), name: str = "primary") -> CohortDefinition:
    return CohortDefinition(
        name=name, inclusion=tuple(inclusion), exclusion=tuple(exclusion)
    )


def _universe(**columns) -> pd.DataFrame:
    """Stay 1 has the event at 2 h, stay 2 at 300 h; stays 3 and 4 never."""

    frame = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4],
            _EVENT: [1, 1, 0, 0],
            "exposure": [1.0, 0.0, 1.0, 0.0],
            "age": [61.0, 72.0, 55.0, 48.0],
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def _variable(name: str, role: str = "outcome", **fields) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        role=role,
        analysis_window=fields.get("analysis_window"),
        observation_semantics=fields.get("observation_semantics"),
        temporal_resolution=fields.get("temporal_resolution"),
        source_concept=fields.get("source_concept"),
    )


def _event_time(status: str, unit: str = "h") -> SimpleNamespace:
    return _variable(
        f"{status}_time",
        role="time",
        observation_semantics=SimpleNamespace(
            kind="conditional_event_time",
            event_status_column=status,
            time_origin="icu_admission",
            time_unit=unit,
        ),
    )


def _context(*, outcomes=(_EVENT,), variables=()) -> SimpleNamespace:
    return SimpleNamespace(
        cohort=SimpleNamespace(
            outcome_columns=list(outcomes), id_columns=["stay_id"], time_columns=[]
        ),
        target_outcome=None,
        variables=list(variables),
        materialized_inputs=None,
    )


def _kept(cohort: CohortDefinition, universe: pd.DataFrame, **kwargs) -> list[int]:
    selected, _ = _build_cohort_with_flow(cohort, universe, **kwargs)
    return sorted(int(stay) for stay in selected["stay_id"])


def test_the_columns_that_record_an_event_over_the_whole_stay() -> None:
    context = _context(
        outcomes=("roster_outcome",),
        variables=[
            _variable("outcome_variable"),
            _variable("windowed_outcome", analysis_window="icu_admission[0,24]h"),
            _variable("procedure_done", role="intervention"),
            _event_time("procedure_done"),
            _variable("lactate_max", role="lab"),
        ],
    )
    context.target_outcome = "target_outcome"

    assert stay_outcome_columns(context) == {
        "roster_outcome",
        "target_outcome",
        "outcome_variable",
        "windowed_outcome",
    }
    # An outcome with its own window records it over that window; the status
    # of a typed event time records it over the whole stay, outcome or not.
    assert whole_stay_event_columns(context) == {
        "roster_outcome",
        "target_outcome",
        "outcome_variable",
        "procedure_done",
    }
    assert whole_stay_event_columns(None) == frozenset()
    # One reading of the event-time descriptor serves this set and the
    # time-zero rule's reading of a status by its time.
    assert event_time_status_column(_event_time("procedure_done")) == "procedure_done"
    assert event_time_status_column(_variable("lactate_max", role="lab")) is None


@pytest.mark.parametrize(
    "kind, op, value",
    [
        ("exclusion", "==", 1),
        ("exclusion", "in", [1]),
        ("inclusion", "==", 0),
        ("inclusion", "!=", 1),
    ],
    ids=[
        "exclude occurrence",
        "exclude occurrence (in)",
        "keep absence",
        "keep absence (!=)",
    ],
)
def test_the_builder_refuses_a_window_on_a_whole_stay_status(kind, op, value) -> None:
    cohort = _cohort(**{kind: [_predicate(_EVENT, op, value)]})

    # Before: read over the whole stay, the event at 300 h left the cohort too.
    assert _kept(cohort, _universe()) == [3, 4]
    with pytest.raises(CohortEventWindowUnreadableError) as caught:
        _kept(cohort, _universe(), whole_stay_columns=[_EVENT])

    message = str(caught.value)
    assert message.startswith(f"{COHORT_EVENT_WINDOW_UNREADABLE}: cohort.{kind}[0] ")
    assert f"records '{_EVENT}' over the whole ICU stay" in message
    assert f"no '{_EVENT}_time'" in message
    assert caught.value.code == COHORT_EVENT_WINDOW_UNREADABLE
    assert [window.column for window in caught.value.windows] == [_EVENT]
    assert isinstance(caught.value, CohortDataError)


@pytest.mark.parametrize(
    "predicate_args, universe_columns, kept",
    [
        ((_EVENT, "==", 1, math.inf), {}, [3, 4]),
        ((_EVENT, ">=", 1, 24.0), {}, [3, 4]),
        ((_EVENT, "missing", None, 24.0), {}, [1, 2, 3, 4]),
        (
            (_EVENT, "==", 1, 24.0),
            {f"{_EVENT}_time": [2.0, 300.0, None, None]},
            [2, 3, 4],
        ),
    ],
    ids=["whole stay", "magnitude", "missingness", "read by its time"],
)
def test_what_the_builder_can_read_is_read_as_before(
    predicate_args, universe_columns, kept
) -> None:
    concept, op, value, end = predicate_args
    cohort = _cohort(exclusion=[_predicate(concept, op, value, end=end)])

    assert (
        _kept(cohort, _universe(**universe_columns), whole_stay_columns=[_EVENT])
        == kept
    )


def test_a_status_summarized_over_its_own_window_is_not_whole_stay() -> None:
    context = _context(
        outcomes=(_EVENT,),
        variables=[_variable(_EVENT, analysis_window="icu_admission[0,24]h")],
    )
    cohort = _cohort(exclusion=[_predicate(_EVENT, "==", 1)])

    assert whole_stay_event_columns(context) == frozenset()
    assert _kept(
        cohort, _universe(), whole_stay_columns=whole_stay_event_columns(context)
    ) == [3, 4]


def test_the_locked_analysis_cohort_is_refused_on_a_universe_without_the_time(
    tmp_path: Path,
) -> None:
    universe_path = tmp_path / "universe.parquet"
    _universe().to_parquet(universe_path, index=False)
    plan = SimpleNamespace(
        cohort=_cohort(exclusion=[_predicate(_EVENT, "==", 1)]), steps=[]
    )

    refused = materialize_locked_analysis_cohort(
        run_dir=tmp_path, plan=plan, universe_path=universe_path, context=_context()
    )
    # A context that types the event's time while the universe lacks it: the
    # plan passes review, and the builder still refuses.
    typed_time = _context(variables=[_event_time(_EVENT)])
    validate_plan_typed_bindings_against_context(plan=plan, context=typed_time)
    declared = materialize_locked_analysis_cohort(
        run_dir=tmp_path, plan=plan, universe_path=universe_path, context=typed_time
    )

    for result in (refused, declared):
        assert result["status"] == "error"
        assert (
            f"{COHORT_EVENT_WINDOW_UNREADABLE}: cohort.exclusion[0]" in result["error"]
        )


def test_an_export_without_the_time_is_refused_by_both_builders(
    tmp_path: Path,
) -> None:
    within_a_day = _cohort(exclusion=[_predicate("death", "==", 1)])
    whole_stay = _cohort(exclusion=[_predicate("death", "==", 1, end=math.inf)])
    outcome = native_outcome(death=[True, False, True, False])
    options = dict(
        feature_concepts=[],
        database="miiv",
        outcome_concepts=["death"],
        static_concepts=["age"],
    )

    # The materializer's own filter on an export whose outcome module issues
    # no time.  (On a typed export it refuses any predicate on an outcome
    # column before this.)
    untyped = untyped_native_export(
        tmp_path / "untyped", outcome=outcome, outcome_concepts=["death"]
    )
    with pytest.raises(CohortEventWindowUnreadableError, match="'death_time'"):
        materialize_cohort(
            cohort_definition=within_a_day, data_path=str(untyped), **options
        )
    kept, _ = materialize_cohort(
        cohort_definition=whole_stay, data_path=str(untyped), **options
    )
    assert sorted(kept["stay_id"]) == [2, 4]

    # The host's analysis cohort on a sealed universe of the same export.
    typed = typed_native_export(
        tmp_path / "typed", outcome=outcome, outcome_concepts=["death"]
    )
    paths = materialize_to_parquet(
        tmp_path / "run",
        stem="universe",
        cohort_window=(0.0, 24.0),
        data_path=str(typed),
        **options,
    )
    with pytest.raises(MaterializedMetadataError) as caught:
        materialize_locked_analysis_cohort(
            run_dir=tmp_path / "run",
            plan=SimpleNamespace(cohort=within_a_day, steps=[]),
            universe_path=Path(paths["parquet"]),
            context=_context(outcomes=("death",)),
        )
    assert isinstance(caught.value.__cause__, CohortEventWindowUnreadableError)


def test_the_planner_is_refused_a_primary_window_the_input_cannot_read() -> None:
    context = _planner_context()
    plan = SimpleNamespace(
        cohort=_cohort(exclusion=[_predicate("outcome_flag", "==", 1)]),
        steps=[],
        robustness_specs=[],
    )

    with pytest.raises(CohortSchemaError) as caught:
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    message = str(caught.value)
    assert message.startswith(f"{COHORT_EVENT_WINDOW_UNREADABLE}: cohort.exclusion[0] ")
    assert "records 'outcome_flag' over the whole ICU stay" in message
    assert "cohort.unapplied_population_criteria" in message
    # A Planner with strict structured output cannot write an unbounded end.
    assert 'end_offset_hours "inf"' not in message
    assert "death" not in message
    assert infer_validation_stage(caught.value) == "typed_context_binding"


def test_the_planner_keeps_the_windows_it_can_read() -> None:
    context = _planner_context()
    for cohort in (
        _cohort(exclusion=[_predicate("outcome_flag", "==", 1, end=math.inf)]),
        _cohort(exclusion=[_predicate("outcome_flag", "missing", None)]),
        _cohort(inclusion=[_predicate("age_years", ">=", 18)]),
    ):
        plan = SimpleNamespace(cohort=cohort, steps=[], robustness_specs=[])
        validate_plan_typed_bindings_against_context(plan=plan, context=context)

    # An override is the review's, not the Planner's binding gate.
    override = RobustnessSpec(
        spec_id="without_early_events",
        axis="cohort",
        description="Exclude the early events.",
        cohort_override=_cohort(
            name="override", exclusion=[_predicate("outcome_flag", "==", 1)]
        ),
    )
    plan = SimpleNamespace(cohort=None, steps=[], robustness_specs=[override])
    validate_plan_typed_bindings_against_context(plan=plan, context=context)


def test_the_checks_read_the_plan_as_it_was_built() -> None:
    # A plan's concepts were checked in its own concept scope when it was
    # built; neither check validates them again outside it.
    with cohort_concept_id_scope(["scoped_flag"]):
        primary = _cohort(inclusion=[_predicate("scoped_flag", ">=", 1)])
        override = _override("scoped_flag", spec_id="scoped")
    plan = SimpleNamespace(cohort=primary, steps=[], robustness_specs=[override])

    validate_plan_typed_bindings_against_context(plan=plan, context=_planner_context())
    assert robustness_override_event_window_findings(_planner_context(), plan) == []


def _override(
    concept: str = _EVENT, *, spec_id: str = "without_early_events"
) -> RobustnessSpec:
    return RobustnessSpec(
        spec_id=spec_id,
        axis="cohort",
        description="Exclude the stays with the event in the first day.",
        cohort_override=_cohort(name=spec_id, exclusion=[_predicate(concept, "==", 1)]),
    )


@pytest.mark.parametrize("concept", [_EVENT, "event_alias"], ids=["named", "aliased"])
def test_every_robustness_path_refuses_the_override(concept, tmp_path: Path) -> None:
    spec = _override(concept)
    context = _context()

    with pytest.raises(
        CohortEventWindowUnreadableError, match="cohort_override.exclusion"
    ):
        _data_with_predicate_aliases(
            data=_universe(),
            cohort_definition=spec.cohort_override,
            exposure="exposure",
            context=context,
        )
    replay = _replay_primary_model_for_cohort(
        spec=spec,
        source={"primary_contract": {"exposure_source": "exposure"}},
        data=_universe(),
        context=context,
        out_dir=tmp_path,
    )
    membership = _membership_audit(
        specs=[spec], cohort=_universe(), universe=_universe(), context=context
    )
    refit = _fit_one_row(
        spec_id=spec.spec_id,
        axis=spec.axis,
        spec=spec,
        data=_universe(),
        primary_cohort=None,
        exposure="exposure",
        outcome="age",
        outcome_columns={},
        kind="linear",
        default_missing="complete_case",
        context=context,
        evidence_id="robustness_panel",
    )

    assert replay["index"]["status"] == "blocked"
    assert replay["row"].point_estimate is None and replay["row"].n == 0
    assert replay["row"].notes.startswith(
        "locked cohort override could not be materialised: "
        f"{COHORT_EVENT_WINDOW_UNREADABLE}: cohort_override.exclusion[0] "
    )
    assert membership[-1]["membership_executable"] is False
    assert COHORT_EVENT_WINDOW_UNREADABLE in membership[-1]["notes"]
    assert refit.point_estimate is None and not refit.converged
    assert refit.notes.startswith(COHORT_EVENT_WINDOW_UNREADABLE)


def test_an_override_the_input_can_read_is_built() -> None:
    timed = _universe(**{f"{_EVENT}_time": [2.0, 300.0, None, None]})
    for data, spec in (
        (timed, _override()),
        (
            _universe(),
            RobustnessSpec(
                spec_id="adults",
                axis="cohort",
                description="Adults only.",
                cohort_override=_cohort(
                    name="adults", inclusion=[_predicate("age", ">=", 50)]
                ),
            ),
        ),
    ):
        out = _data_with_predicate_aliases(
            data=data,
            cohort_definition=spec.cohort_override,
            exposure="exposure",
            context=_context(),
        )
        assert out is data


def test_the_review_routes_an_unreadable_override_to_the_agent() -> None:
    context = _planner_context()
    plan = SimpleNamespace(
        robustness_specs=[
            _override("outcome_flag"),
            _override("age_years", spec_id="adults_only"),
        ]
    )

    findings = robustness_override_event_window_findings(context, plan)

    assert [item.code for item in findings] == [_CODE]
    finding = findings[0]
    assert (finding.severity, finding.dimension) == ("major", "robustness")
    assert remediation_route_for_finding(finding) == "agent_plan_revision"
    assert (
        "robustness_specs[without_early_events].cohort_override.exclusion[0]"
        in finding.message
    )
    assert "adults_only" not in finding.message
    assert "fails the run closed at the robustness panel" in finding.message
    assert 'end_offset_hours "inf"' not in finding.remediation
    assert "or remove the specification" in finding.remediation
    assert "death" not in finding.message + finding.remediation
    # A context that types the event's time reads the override by it.
    timed = _planner_context()
    timed.variables.append(
        ConceptDescriptor(
            name="outcome_flag_time",
            role=VariableRole.TIME,
            dtype="float64",
            observation_semantics=ObservationSemantics(
                kind="conditional_event_time",
                event_status_column="outcome_flag",
                time_origin="icu_admission",
                time_unit="h",
            ),
        )
    )
    assert robustness_override_event_window_findings(timed, plan) == []


def test_the_finding_reads_in_chinese() -> None:
    vocab = (
        Path(inspect.getfile(robustness_override_event_window_findings))
        .resolve()
        .parents[2]
        / "webserver/static/js/screens-agent-reader-vocab.js"
    ).read_text(encoding="utf-8")

    (entry,) = [
        line for line in vocab.splitlines() if line.strip().startswith(f"{_CODE}: [")
    ]
    assert f"{_CODE}: ['稳健性分析的事件窗口读不出'" in entry
    assert "没有该事件的时间列" in entry
    assert "由 Agent 改写或删除" in entry


def test_the_plan_review_reports_it() -> None:
    from easyicu.research_agent.agents.progressive_planner import (
        ProgressivePlannerAgent,
    )
    from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
    from tests.research_agent.planning.progressive_planner_fixtures import (
        _foundation_payload,
        _materialization_payloads,
        _outline_payload,
    )

    responses = [
        _outline_payload(),
        _foundation_payload(),
        *_materialization_payloads(),
    ]
    llm = ScriptedMockLLMClient([json.dumps(item) for item in responses])
    llm.supports_strict_json_schema = True
    plan = ProgressivePlannerAgent(llm).run(_planner_context())
    with_override = plan.model_copy(
        update={"robustness_specs": [*plan.robustness_specs, _override("outcome_flag")]}
    )

    def codes(reviewed) -> list[str]:
        review = build_plan_scientific_review(
            context=_planner_context(), plan=reviewed, literature=None
        )
        return [item.code for item in review.findings if item.code == _CODE]

    assert codes(plan) == []
    assert codes(with_override) == [_CODE]


def _row(
    spec_id: str, *, point=1.2, converged=True, axis="cohort"
) -> RobustnessPanelRow:
    return RobustnessPanelRow(
        spec_id=spec_id,
        axis="primary" if spec_id == "primary" else axis,
        n=0 if point is None else 120,
        point_estimate=point,
        ci_low=None if point is None else 0.9,
        ci_high=None if point is None else 1.6,
        se=None,
        evidence_id="" if point is None else f"stat_{spec_id}",
        converged=converged,
    )


def test_a_specification_without_an_estimate_is_counted_nowhere(tmp_path: Path) -> None:
    blocked = _replay_primary_model_for_cohort(
        spec=_override(),
        source={"primary_contract": {"exposure_source": "exposure"}},
        data=_universe(),
        context=_context(),
        out_dir=tmp_path,
    )["row"]
    rows = [
        _row("primary"),
        _row("alt_cohort", point=1.1),
        _row("alt_missing", point=1.3, axis="missing"),
        blocked,
    ]
    panel = RobustnessPanel.from_rows(rows)

    # Before: three variants were counted, the blank row among them.
    assert panel.n_variants == 2
    assert unexecuted_locked_spec_ids(panel) == ["without_early_events"]
    digest = numeric_digest_for_panel(panel)
    assert digest["n_variants"] == 2
    assert 3 not in [value for value in digest.values() if isinstance(value, int)]

    (tmp_path / "robustness_panel.json").write_text(
        json.dumps(panel.to_dict()), encoding="utf-8"
    )
    block = "\n".join(_render_robustness_panel_block(run_dir=tmp_path))
    assert "variants: n_variants=2, " in block
    assert "pre-specified but not estimated: without_early_events." in block
    assert "none of them is among the n_variants analyses" in block
    assert (
        f"notes=locked cohort override could not be materialised: {COHORT_EVENT_WINDOW_UNREADABLE}"
        in block
    )


def test_the_run_reports_the_count_and_the_unexecuted_specification(
    monkeypatch, tmp_path: Path
) -> None:
    specs = [_override(), SimpleNamespace(spec_id="alt_cohort", axis="cohort")]
    blocked = _replay_primary_model_for_cohort(
        spec=specs[0],
        source={"primary_contract": {"exposure_source": "exposure"}},
        data=_universe(),
        context=_context(),
        out_dir=tmp_path,
    )["row"]
    monkeypatch.setattr(
        runtime_panel, "robustness_specs_for_execution", lambda **_: specs
    )
    monkeypatch.setattr(
        runtime_panel,
        "fit_robustness_rows_from_records",
        lambda **_: ([_row("alt_cohort", point=1.1), blocked], []),
    )
    monkeypatch.setattr(runtime_panel, "write_robustness_panel", lambda **_: None)

    result = runtime_panel.finalize_run_robustness_panel(
        run_dir=tmp_path,
        plan=SimpleNamespace(robustness_specs=specs, cohort=None),
        per_step_records=[],
        cohort_path=None,
        context=_context(),
        evidence=SimpleNamespace(),
        prompt_pack_version="prompt-v1",
    )

    assert result.manifest_update()["robustness_n_variants"] == 1
    errors = [item for item in result.findings if item.severity == "error"]
    assert [item.detail["unexecuted_spec_ids"] for item in errors] == [
        ["without_early_events"]
    ]


def test_an_estimate_is_a_finite_converged_number() -> None:
    frame = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], "y": [1.0, 2.9, 5.2, 7.1, 8.8, 11.0]}
    )
    closed_form = fit_estimator(
        cohort=_cohort(), X=frame[["x"]], y=frame["y"], kind="linear", term="x"
    )
    free_form = RobustnessPanelRow.from_dict(
        {"spec_id": "free_form", "axis": "cohort", "n": 120, "point_estimate": 1.4}
    )

    # A closed-form fit has no optimizer, and its producer records it converged.
    assert row_was_estimated(
        _row("ols", point=closed_form.point_estimate, converged=closed_form.converged)
    )
    assert not row_was_estimated(_row("nan", point=float("nan")))
    assert not row_was_estimated(_row("diverged", point=1.4, converged=False))
    # A result row must record ``converged``; a free-form row without it has none.
    assert not row_was_estimated(free_form)
    panel = RobustnessPanel.from_rows([_row("primary"), free_form])
    assert panel.n_variants == 0 and unexecuted_locked_spec_ids(panel) == ["free_form"]


def test_the_refusal_names_the_predicate_its_column_and_the_missing_time() -> None:
    windows = predicates_read_over_the_whole_stay(
        _cohort(exclusion=[_predicate(_EVENT, "==", 1)]),
        columns=["stay_id", _EVENT],
        whole_stay_columns=[_EVENT],
        label="robustness_specs[x].cohort_override",
    )

    assert [window.description() for window in windows] == [
        "robustness_specs[x].cohort_override.exclusion[0] reads whether the event "
        f"of '{_EVENT}' happened within icu_admission[0, 24) h, but this input "
        f"records '{_EVENT}' over the whole ICU stay and has no '{_EVENT}_time' "
        "to place the event in that window"
    ]
