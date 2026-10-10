"""A study planned on metadata alone forms its exposure groups, and keeps them.

A Web planning run reads an input with no row that names each concept once
(``webserver.agent_pipeline_runs._metadata_only_planning_acquisition``).
Its grouping sources and compile read that input as the prepared data will
hold it -- a concept measured over time as its first, maximum, mean and
minimum over the study's window -- so the Planner is asked, and each applied
grouping is declared on the empty input rather than derived.  The run that
follows the accepted candidate on the prepared data forms the same groups
without asking (``PipelineConfig.bound_exposure_groupings``), from a record
checked against the input the candidate sealed; a grouping that reads
otherwise there stops planning.  A template that fixes the exposure's
reference and contrast compares the groups the study compares, never the
unmeasured level.  Synthetic inputs and scripted answers only.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.contracts.exposure_group_rules import (
    ExposureGroupRuleError,
    read_planned_group_columns,
)
from easyicu.research_agent.intake.materialized_metadata import (
    EXPOSURE_GROUP_STAGE_PRODUCER,
    MaterializedMetadataError,
    load_verified_materialized_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.orchestration.config import PipelineConfig
from easyicu.research_agent.orchestration.exposure_grouping_phase import (
    EXPOSURE_GROUPINGS_FILENAME,
    planning_run_groupings,
    run_exposure_grouping_phase,
)
from easyicu.research_agent.planning.exposure_group_compile import (
    CandidateExposureGroupings,
    exposure_group_contrast,
    grouping_sources,
)
from easyicu.research_agent.planning.family_spec.contract import FamilySpecError
from easyicu.research_agent.planning.family_spec.request import (
    _stated_contrast,
    build_family_spec_request,
)
from easyicu.research_agent.planning.progressive_contract import (
    ProgressivePlanCompileError,
)
from easyicu.research_agent.providers.mocks import ScriptedMockLLMClient
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.typed import declared_domain_for_variable
from tests.support.typed_export import typed_export

_QUESTION = "Compare hospital death across lactate groups in the first day."
_ANSWER = json.dumps(
    {
        "groupings": [
            {
                "id": "x1",
                "concept": "lact",
                "window": {"start_hours": 0, "end_hours": 24},
                "scale": "nominal",
                "groups": [
                    {
                        "id": "g1",
                        "label": "lactate below 2.5",
                        "rule": {
                            "summary": "max",
                            "op": "<",
                            "value": 2.5,
                            "unit": "mmol/L",
                        },
                    },
                    {"id": "g2", "label": "lactate 2.5 or above", "rule": "otherwise"},
                ],
                "unmeasured": {"handling": "own_group", "label": "not measured"},
                "reference": "g1",
                "contrast": "g2",
                "quote": "lactate groups",
                "source": "question",
            }
        ]
    }
)
_PLANNING_AUTHORITY = "easyicu_planning_authority"
#: The descriptor facts a grouped column's planning and prepared data share.
_TYPED_FACTS = (
    "unit",
    "valid_range",
    "unit_normalization",
    "source_concept",
    "derived_from_concepts",
    "analysis_window",
    "description",
    "role",
    "is_ordinal",
    "ordinal_levels",
)


def _context(cohort: Path, *, hours: float | None = 24):
    window = {"role": "outer_observation_window", "anchor": "icu_admission"}
    return build_research_context(
        research_question=_QUESTION,
        cohort=cohort,
        cohort_name="synthetic",
        database="miiv",
        target_outcome="death",
        id_columns=("stay_id",),
        outcome_columns=("death",),
        user_preferences=(
            {
                "data_constraints": json.dumps(
                    {"materialization_window": {**window, "hours": hours}}
                )
            }
            if hours is not None
            else None
        ),
    )


def _catalog(root: Path, *columns: str) -> Path:
    """An input planned on metadata alone, as the Web host writes it."""

    frame = pd.DataFrame(
        {
            "stay_id": pd.Series(dtype="int64"),
            **{name: pd.Series(dtype="float64") for name in columns},
            "death": pd.Series(dtype="float64"),
        }
    )
    frame.attrs[_PLANNING_AUTHORITY] = {
        "kind": "metadata_only_planning_catalog",
        "patient_rows_read": False,
    }
    path = root / "run" / "cohort.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    return path


def _prepared(root: Path) -> Path:
    """The run's exact typed copy of the study's prepared data."""

    root.mkdir(parents=True, exist_ok=True)
    paths = cohort_materializer.materialize_to_parquet(
        root / "materialized",
        stem="universe",
        data_path=typed_export(root / "export"),
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    copy = root / "run" / "cohort.parquet"
    stage_materialized_cohort_authority(
        paths["parquet"], copy, producer_implementation_sha256="a" * 64
    )
    return copy


def _phase(cohort: Path, client: Any, **changes: Any):
    options = {
        "context": _context(cohort),
        "cohort_path": cohort,
        "run_dir": cohort.parent,
        "planner": client,
        "rebuild_context": _context,
        "capability_review_pending": False,
        "trajectory_staged": False,
        "emit_progress": lambda *_args, **_kwargs: None,
    }
    options.update(changes)
    return run_exposure_grouping_phase(**options)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _planned(tmp_path: Path):
    """A planning run that formed the lactate groups; its input and record."""

    cohort = _catalog(tmp_path / "candidate", "lact", "age")
    client = ScriptedMockLLMClient([_ANSWER])
    phase = _phase(cohort, client)
    assert len(client.calls) == 1
    return phase, cohort


def test_a_planning_input_offers_the_summaries_its_prepared_data_will_hold(
    tmp_path: Path,
) -> None:
    catalog = _catalog(tmp_path / "a", "lact", "age")
    lines = [source.line() for source in grouping_sources(_context(catalog))]
    assert lines == [
        "- age: one value per stay (summary value, no window); unit years; "
        "declared values from 0 to 120",
        "- lact: first, max, mean, min over hours [0, 24) after ICU admission; "
        "unit mmol/L; declared values from 0 to 30",
    ]
    # Without the study's window, no summary over time can be planned.
    assert [
        source.concept for source in grouping_sources(_context(catalog, hours=None))
    ] == ["age"]
    # A column the input already names as one summary is read as that summary.
    operationalized = _catalog(tmp_path / "b", "lact_max")
    assert [
        source.line() for source in grouping_sources(_context(operationalized))
    ] == [
        "- lact: max over hours [0, 24) after ICU admission; unit mmol/L; "
        "declared values from 0 to 30"
    ]


def test_the_prepared_data_hold_each_summary_under_the_name_planned(
    tmp_path: Path,
) -> None:
    # One owner names a summary column for the planning input and the
    # prepared data alike.
    held = set(pd.read_parquet(_prepared(tmp_path)).columns)
    planned = {
        cohort_materializer.summary_column_name("lact", summary)
        for summary in cohort_materializer.EVENT_SUMMARIES
    }

    assert planned == {"lact_first", "lact_max", "lact_mean", "lact_min"}
    assert planned <= held
    with pytest.raises(ValueError):
        cohort_materializer.summary_column_name("lact", "median")


def test_a_grouping_planned_on_metadata_is_declared_on_the_empty_input(
    tmp_path: Path,
) -> None:
    phase, cohort = _planned(tmp_path)

    frame = pd.read_parquet(cohort)
    assert len(frame) == 0
    assert str(frame["lact_group_x1"].dtype) == "int64"
    record = json.loads((cohort.parent / EXPOSURE_GROUPINGS_FILENAME).read_text())
    assert record["candidate"] is None
    assert record["planner"]["stated"]["groupings"][0]["quote"] == "lactate groups"
    (declared,) = frame.attrs[_PLANNING_AUTHORITY]["exposure_groups"]
    assert declared["groupings_record_sha256"] == _sha256(phase.record_path)
    assert (declared["variable"], declared["levels"]) == ("lact_group_x1", 3)
    assert declared["compared"] == {"reference": 1, "contrast": 2}
    assert record["compiled"]["groupings"][0]["compared"] == declared["compared"]
    variable = phase.context.variable("lact_group_x1")
    assert declared_domain_for_variable(variable) == (
        [1, 2, 3],
        "declared_exposure_group_levels",
    )
    assert (variable.unit, variable.unit_normalization, variable.analysis_window) == (
        "category",
        "exposure_group",
        "icu_admission[0,24]h",
    )


def test_the_prepared_data_form_the_candidates_groups_without_asking(
    tmp_path: Path,
) -> None:
    planned, candidate_cohort = _planned(tmp_path)
    accepted = planning_run_groupings(
        candidate_cohort.parent, cohort_sha256=_sha256(candidate_cohort)
    )
    assert accepted is not None
    assert (accepted.variables, accepted.concepts) == (("lact_group_x1",), ("lact",))

    copy = _prepared(tmp_path / "package")
    client = ScriptedMockLLMClient([])
    phase = _phase(copy, client, candidate=accepted.candidate.model_dump(mode="json"))

    assert client.calls == []
    staged = load_verified_materialized_cohort_authority(copy)
    assert staged is not None
    assert staged.authority.producer == EXPOSURE_GROUP_STAGE_PRODUCER
    record = json.loads(phase.record_path.read_text())
    assert record["planner"] is None
    assert record["candidate"]["record_sha256"] == _sha256(planned.record_path)
    candidate_record = json.loads(planned.record_path.read_text())
    assert [item["derivation_sha256"] for item in record["compiled"]["groupings"]] == [
        item["derivation_sha256"] for item in candidate_record["compiled"]["groupings"]
    ]
    typed = phase.context.variable("lact_group_x1")
    declared = planned.context.variable("lact_group_x1")
    assert {name: getattr(typed, name) for name in _TYPED_FACTS} == {
        name: getattr(declared, name) for name in _TYPED_FACTS
    }


def test_a_candidate_grouping_that_reads_otherwise_here_stops_planning(
    tmp_path: Path,
) -> None:
    planned, candidate_cohort = _planned(tmp_path)
    accepted = planning_run_groupings(
        candidate_cohort.parent, cohort_sha256=_sha256(candidate_cohort)
    )
    assert accepted is not None
    drifted = accepted.candidate.model_dump(mode="json")
    drifted["derivations"] = {"x1": "0" * 64}
    copy = _prepared(tmp_path / "package")
    client = ScriptedMockLLMClient([])

    with pytest.raises(ProgressivePlanCompileError) as stopped:
        _phase(copy, client, candidate=drifted)
    assert stopped.value.reason_code == "exposure_group_candidate_drift"
    assert "x1 reads max from 'lact_max' here" in str(stopped.value)
    with pytest.raises(ProgressivePlanCompileError) as staged_trajectory:
        _phase(
            copy,
            client,
            candidate=accepted.candidate.model_dump(mode="json"),
            trajectory_staged=True,
        )
    assert staged_trajectory.value.reason_code == "exposure_group_candidate_drift"
    assert client.calls == []
    staged = load_verified_materialized_cohort_authority(copy)
    assert staged is not None
    assert staged.authority.producer == "research_agent_run_stage"


def test_a_candidates_group_no_prepared_stay_falls_in_stops_planning(
    tmp_path: Path,
) -> None:
    answer = json.loads(_ANSWER)
    low, rest = answer["groupings"][0]["groups"]
    low["label"], low["rule"]["value"] = "lactate below 2.2", 2.2
    # Between the prepared data's maxima, but none lies between 2.2 and 2.8.
    middle = {"id": "g2", "label": "lactate 2.2 to 2.8", "rule": {**low["rule"]}}
    middle["rule"]["value"] = 2.8
    rest.update(id="g3", label="lactate 2.8 or above")
    answer["groupings"][0]["groups"] = [low, middle, rest]
    answer["groupings"][0]["contrast"] = "g3"
    cohort = _catalog(tmp_path / "candidate", "lact", "age")
    # The planning input holds no row, so no group of it is empty.
    _phase(cohort, ScriptedMockLLMClient([json.dumps(answer)]))
    accepted = planning_run_groupings(cohort.parent, cohort_sha256=_sha256(cohort))
    assert accepted is not None
    copy = _prepared(tmp_path / "package")

    with pytest.raises(ProgressivePlanCompileError) as stopped:
        _phase(
            copy,
            ScriptedMockLLMClient([]),
            candidate=accepted.candidate.model_dump(mode="json"),
        )

    assert stopped.value.reason_code == "exposure_group_level_empty"
    assert "x1 'lactate 2.2 to 2.8'" in str(stopped.value)
    staged = load_verified_materialized_cohort_authority(copy)
    assert staged is not None
    assert staged.authority.producer == "research_agent_run_stage"


def test_a_candidate_that_formed_no_groups_asks_nothing(tmp_path: Path) -> None:
    copy = _prepared(tmp_path)
    client = ScriptedMockLLMClient([])
    none = CandidateExposureGroupings(
        record_sha256="a" * 64, stated={"groupings": []}, derivations={}
    )

    phase = _phase(copy, client, candidate=none.model_dump(mode="json"))

    assert client.calls == []
    record = json.loads(phase.record_path.read_text(encoding="utf-8"))
    assert record["not_asked"] == "candidate_stated_none"


def _declare(cohort: Path, columns: list[dict[str, Any]]) -> str:
    """``cohort``'s planning input declaring ``columns``; its new digest."""

    frame = pd.read_parquet(cohort)
    frame.attrs[_PLANNING_AUTHORITY] = {
        **frame.attrs[_PLANNING_AUTHORITY],
        "exposure_groups": columns,
    }
    frame.to_parquet(cohort, index=False)
    return _sha256(cohort)


def test_a_planning_run_asked_nothing_is_followed_without_groups(
    tmp_path: Path,
) -> None:
    cohort = _catalog(tmp_path / "asked-nothing", "lact", "age")
    client = ScriptedMockLLMClient([])

    phase = _phase(cohort, client, capability_review_pending=True)

    assert client.calls == []
    assert json.loads(phase.record_path.read_text())["not_asked"] == (
        "capability_review_pending"
    )
    assert planning_run_groupings(cohort.parent, cohort_sha256=_sha256(cohort)) is None
    # Its input cannot declare a group column no one asked for.
    _grouped_run, grouped = _planned(tmp_path)
    declared = pd.read_parquet(grouped).attrs[_PLANNING_AUTHORITY]["exposure_groups"]
    sealed = _declare(cohort, declared)
    with pytest.raises(ValueError, match="asked for no grouping"):
        planning_run_groupings(cohort.parent, cohort_sha256=sealed)


@pytest.mark.parametrize(
    "changed",
    [
        pytest.param({"compared": {"reference": 2, "contrast": 1}}, id="comparison"),
        pytest.param({"levels": 2}, id="levels"),
        pytest.param({"description": "Groups of lact"}, id="labels"),
        pytest.param({"transform": "exposure_group_ordinal"}, id="scale"),
    ],
)
def test_a_planning_input_declares_each_column_as_its_record_states_it(
    tmp_path: Path, changed: dict[str, Any]
) -> None:
    _planned_run, cohort = _planned(tmp_path)
    (declared,) = pd.read_parquet(cohort).attrs[_PLANNING_AUTHORITY]["exposure_groups"]

    sealed = _declare(cohort, [{**declared, **changed}])

    with pytest.raises(ValueError, match="does not declare the group columns"):
        planning_run_groupings(cohort.parent, cohort_sha256=sealed)


def test_a_planning_runs_groups_are_read_only_from_the_input_it_sealed(
    tmp_path: Path,
) -> None:
    assert planning_run_groupings(tmp_path, cohort_sha256="0" * 64) is None, (
        "a run that was not asked kept no record"
    )
    planned, cohort = _planned(tmp_path)
    sealed = _sha256(cohort)

    with pytest.raises(ValueError, match="not the one its capsule sealed"):
        planning_run_groupings(cohort.parent, cohort_sha256="0" * 64)
    record = json.loads(planned.record_path.read_text())
    record["compiled"]["groupings"][0]["derivation_sha256"] = "1" * 64
    planned.record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="does not declare the group columns"):
        planning_run_groupings(cohort.parent, cohort_sha256=sealed)


def test_a_bound_grouping_is_held_by_a_reviewed_run_that_plans_groupings(
    tmp_path: Path,
) -> None:
    bound = CandidateExposureGroupings(
        record_sha256="a" * 64,
        stated=json.loads(_ANSWER),
        derivations={"x1": "b" * 64},
    ).model_dump(mode="json")

    with pytest.raises(ValueError, match="requires require_human_plan_review"):
        PipelineConfig(workdir=tmp_path, bound_exposure_groupings=bound)
    with pytest.raises(ValueError, match="enable_exposure_grouping"):
        PipelineConfig(
            workdir=tmp_path,
            require_human_plan_review=True,
            bound_exposure_groupings=bound,
        )
    with pytest.raises(ValueError):
        PipelineConfig(
            workdir=tmp_path,
            require_human_plan_review=True,
            enable_exposure_grouping=True,
            bound_exposure_groupings={**bound, "derivations": {}},
        )
    config = PipelineConfig(
        workdir=tmp_path,
        require_human_plan_review=True,
        enable_exposure_grouping=True,
        bound_exposure_groupings=bound,
    )
    assert CandidateExposureGroupings.model_validate(
        config.bound_exposure_groupings
    ) == CandidateExposureGroupings.model_validate(bound)
    assert (
        "bound_exposure_groupings"
        not in PipelineConfig(workdir=tmp_path).canonical_payload()
    )


def test_planned_group_columns_are_read_strictly(tmp_path: Path) -> None:
    declared = {
        "variable": "lact_group_x1",
        "transform": "exposure_group",
        "levels": 3,
        "concept": "lact",
        "window": {"start_hours": 0, "end_hours": 24},
        "description": "Nominal groups of lact stated by the study, as level codes",
        "compared": {"reference": 1, "contrast": 2},
        "groupings_record_sha256": "a" * 64,
    }
    assert read_planned_group_columns([declared])[0].record() == declared
    unstated = {key: value for key, value in declared.items() if key != "compared"}
    for refused in (
        [],
        [{**declared, "extra": 1}],
        [unstated],
        [{**declared, "compared": {"reference": 2, "contrast": 2}}],
        [{**declared, "compared": {"reference": 1, "contrast": 4}}],
        [{**declared, "levels": 1}],
        [{**declared, "levels": 8}],
        [{**declared, "transform": "identity"}],
        [{**declared, "groupings_record_sha256": "A" * 64}],
        [declared, declared],
        [
            declared,
            {
                **declared,
                "variable": "lact_group_x2",
                "groupings_record_sha256": "b" * 64,
            },
        ],
    ):
        with pytest.raises(ExposureGroupRuleError):
            read_planned_group_columns(refused)

    # The context refuses an input declaring a column it does not hold.
    cohort = _catalog(tmp_path, "lact")
    frame = pd.read_parquet(cohort)
    frame.attrs[_PLANNING_AUTHORITY] = {
        **frame.attrs[_PLANNING_AUTHORITY],
        "exposure_groups": [declared],
    }
    frame.to_parquet(cohort, index=False)
    with pytest.raises(MaterializedMetadataError, match="exposure groups"):
        _context(cohort)


def _grouped(root: Path, **stated: Any):
    """A planning run that formed the lactate groups, as ``stated`` changes them."""

    answer = json.loads(_ANSWER)
    answer["groupings"][0].update(stated)
    cohort = _catalog(root, "lact", "age")
    return _phase(cohort, ScriptedMockLLMClient([json.dumps(answer)]))


@pytest.mark.parametrize(
    ("stated", "codes", "indices"),
    [
        ({}, (1, 2), (0, 1)),
        ({"reference": "g2", "contrast": "g1"}, (2, 1), (1, 0)),
        ({"reference": None, "contrast": None}, (1, 2), (0, 1)),
    ],
)
def test_a_template_compares_the_groups_the_study_compares(
    tmp_path: Path,
    stated: dict[str, Any],
    codes: tuple[int, int],
    indices: tuple[int, int],
) -> None:
    phase = _grouped(tmp_path, **stated)
    contrast = exposure_group_contrast(phase.context, "lact_group_x1")
    assert contrast is not None
    assert (contrast.reference, contrast.contrast) == codes

    context = phase.context.model_copy(update={"primary_exposure": "lact_group_x1"})
    request = build_family_spec_request(
        context,
        analysis_types=["descriptive_epidemiology"],
        variable_roster=[item.name for item in context.variables],
        allowed_literature_citation_keys=[],
    )
    # The unmeasured level, the last code, is not a group the study compares.
    assert request.exposure_levels == ["1", "2", "3"]
    assert (
        request.reference_level_index,
        request.primary_contrast_level_index,
    ) == indices


def test_a_template_that_cannot_offer_the_stated_groups_stops(tmp_path: Path) -> None:
    phase = _grouped(tmp_path)

    for levels in ([], ["1"]):
        with pytest.raises(FamilySpecError) as refused:
            _stated_contrast(
                phase.context, "lact_group_x1", levels, reference=0, contrast=0
            )
        assert refused.value.reason_code == (
            "family_spec_exposure_group_contrast_unavailable"
        )
    # Any other exposure keeps the template's own levels.
    assert _stated_contrast(
        phase.context, "age", ["1", "2"], reference=0, contrast=1
    ) == (
        0,
        1,
    )
