"""A run stages the exposure groups its study formed, and its authority vouches for them.

A study may form its exposure by grouping one measured value
(``planning/exposure_group_spec.py``).  The host derives each grouping's
column of level codes from the columns its compiled rules read
(``contracts.exposure_group_rules``) and stages it beside every column of the
source (``intake.materialized_metadata.stage_exposure_grouped_cohort_authority``):
the run's exact copy of the source is restaged in place, under the name the
run's input capsule seals.  The staged authority's parent is the source.
Loading it proves the columns are the parent's plus the groupings, and
recomputes every code from the staged rows; an interrupted restage leaves no
cohort that loads.  A research context built on it declares each grouping's levels
before any row exists, nominal or ordinal.  Synthetic exports only.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from easyicu.research_agent.cohort import materializer as cohort_materializer
from easyicu.research_agent.contracts.exposure_group_rules import (
    EXPOSURE_GROUP_CONTRASTS_KEY,
    EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
    EXPOSURE_GROUP_TRANSFORM_ID,
    ExposureGroupRuleError,
    evaluate_grouping,
    read_group_contrast,
    read_grouping_rules,
)
from easyicu.research_agent.intake import materialized_metadata as materialized
from easyicu.research_agent.intake.materialized_metadata import (
    EXPOSURE_GROUP_STAGE_PRODUCER,
    MaterializedMetadataError,
    canonical_parameters_sha256,
    load_verified_materialized_cohort_authority,
    stage_exposure_grouped_cohort_authority,
    stage_materialized_cohort_authority,
)
from easyicu.research_agent.planning.exposure_group_compile import (
    CompiledGrouping,
    group_contrast,
)
from easyicu.research_agent.planning.exposure_group_spec import (
    ExposureGroupSpec,
    grouping_levels,
)
from easyicu.research_agent.research_context.builder import build_research_context
from easyicu.research_agent.research_context.typed import declared_domain_for_variable
from easyicu.research_agent.schema import VariableRole
from tests.support.typed_export import typed_export

_DAY = {"start_hours": 0.0, "end_hours": 24.0}
_RUN_COHORT = ("run", "cohort.parquet")


def _rule(summary: str, op: str, value: float) -> dict[str, Any]:
    return {"summary": summary, "op": op, "value": value, "unit": None}


def _nominal(**changes: Any) -> dict[str, Any]:
    """Low, high only, or neither, by the window's minimum and maximum."""

    grouping = {
        "variable": "lact_group_x1",
        "parameters": {
            "concept": "lact",
            "window": _DAY,
            "scale": "nominal",
            "groups": [
                {"id": "g1", "rule": _rule("min", "<", 1.0)},
                {"id": "g2", "rule": _rule("max", ">", 2.5)},
                {"id": "g3", "rule": "otherwise"},
            ],
            "unmeasured": "own_group",
            "source_columns": {"min": "lact_min", "max": "lact_max"},
        },
        "labels": {
            "g1": "low",
            "g2": "high only",
            "g3": "neither",
            "gU": "not measured",
        },
        "compared": {"reference": 1, "contrast": 3},
    }
    return {**grouping, **changes}


def _ordinal() -> dict[str, Any]:
    return {
        "variable": "lact_group_x2",
        "parameters": {
            "concept": "lact",
            "window": _DAY,
            "scale": "ordinal",
            "groups": [
                {"id": "g1", "rule": _rule("max", "<", 2.5)},
                {"id": "g2", "rule": "otherwise"},
            ],
            "unmeasured": "exclude",
            "source_columns": {"max": "lact_max"},
        },
        "labels": {"g1": "below 2.5", "g2": "2.5 or above"},
        "compared": {"reference": 1, "contrast": 2},
    }


def _universe(tmp_path: Path) -> Path:
    paths = cohort_materializer.materialize_to_parquet(
        tmp_path / "materialized",
        stem="universe",
        data_path=typed_export(tmp_path / "export"),
        database="miiv",
        static_concepts=("age",),
        feature_concepts=("lact",),
        outcome_concepts=("death",),
    )
    return paths["parquet"]


def _run_copy(tmp_path: Path) -> Path:
    copy = tmp_path / "run" / "cohort.parquet"
    stage_materialized_cohort_authority(
        _universe(tmp_path), copy, producer_implementation_sha256="a" * 64
    )
    return copy


def _stage(tmp_path: Path, *groupings: dict[str, Any]):
    return stage_exposure_grouped_cohort_authority(
        _run_copy(tmp_path),
        groupings=list(groupings),
        groupings_record_sha256="c" * 64,
        producer_implementation_sha256="a" * 64,
    )


def test_each_grouping_adds_its_codes_beside_every_source_column(
    tmp_path: Path,
) -> None:
    staged = _stage(tmp_path, _nominal(), _ordinal())

    parent = load_verified_materialized_cohort_authority(
        tmp_path / "materialized" / "universe.parquet"
    )
    assert staged is not None and parent is not None
    assert staged.authority.producer == EXPOSURE_GROUP_STAGE_PRODUCER
    assert staged.authority.parent_authority_sha256 == parent.reference.sha256
    assert staged.authority.cohort_columns == (
        *parent.authority.cohort_columns,
        "lact_group_x1",
        "lact_group_x2",
    )
    assert staged.authority.row_identity_sha256 == parent.authority.row_identity_sha256
    table = pq.read_table(tmp_path.joinpath(*_RUN_COHORT))
    source = pq.read_table(tmp_path / "materialized" / "universe.parquet")
    # Every source column is carried as it is.
    assert table.select(source.column_names).equals(source)
    maxima = table.column("lact_max").to_pylist()
    minima = table.column("lact_min").to_pylist()
    expected_nominal = [
        1 if low < 1.0 else 2 if high > 2.5 else 3 for low, high in zip(minima, maxima)
    ]
    assert table.column("lact_group_x1").to_pylist() == expected_nominal
    assert table.column("lact_group_x2").to_pylist() == [
        1 if high < 2.5 else 2 for high in maxima
    ]
    transforms = {
        item.output_column: item.transform_id
        for item in staged.authority.output_derivations
    }
    assert transforms["lact_group_x1"] == EXPOSURE_GROUP_TRANSFORM_ID
    assert transforms["lact_group_x2"] == EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID
    assert set(transforms.values()) - {
        EXPOSURE_GROUP_TRANSFORM_ID,
        EXPOSURE_GROUP_ORDINAL_TRANSFORM_ID,
    } == {"identity_stage_copy"}
    # Loading it again proves it from the run directory alone.
    again = load_verified_materialized_cohort_authority(tmp_path.joinpath(*_RUN_COHORT))
    assert again is not None and again.reference == staged.reference


def test_a_stay_takes_the_first_group_whose_rule_it_meets() -> None:
    table = pa.table(
        {
            "lact_min": [0.5, 0.5, 2.0, None, 1.5],
            "lact_max": [3.0, 0.9, 3.0, None, 2.0],
        }
    )

    own = read_grouping_rules(_nominal()["parameters"])
    excluded = read_grouping_rules(
        {**_nominal()["parameters"], "unmeasured": "exclude"}
    )

    # Low wins over high; otherwise takes the measured rest; an unmeasured
    # stay is its own level, or none when it leaves the study.
    assert evaluate_grouping(own, table).to_pylist() == [1, 1, 2, 4, 3]
    assert evaluate_grouping(excluded, table).to_pylist() == [1, 1, 2, None, 3]
    assert dict(own.codes) == {"g1": 1, "g2": 2, "g3": 3, "gU": 4}


def test_a_nominal_grouping_codes_its_groups_in_the_order_stated() -> None:
    table = pa.table(
        {
            "lact_min": [0.5, 0.5, 2.0, None, 1.5],
            "lact_max": [3.0, 0.9, 3.0, None, 2.0],
        }
    )
    stated = [
        {"id": "g2", "rule": _rule("max", ">", 2.5)},
        {"id": "g1", "rule": _rule("min", "<", 1.0)},
        {"id": "g3", "rule": "otherwise"},
    ]
    nominal = read_grouping_rules({**_nominal()["parameters"], "groups": stated})
    ordinal = read_grouping_rules(
        {
            **_ordinal()["parameters"],
            "groups": [
                {"id": "g2", "rule": _rule("max", ">=", 2.5)},
                {"id": "g1", "rule": "otherwise"},
            ],
        }
    )

    # Nominal groups have no scale: the study's order names them.  An
    # ordinal grouping's ids are numbered along its scale.
    assert dict(nominal.codes) == {"g2": 1, "g1": 2, "g3": 3, "gU": 4}
    assert evaluate_grouping(nominal, table).to_pylist() == [1, 2, 1, 4, 3]
    assert dict(ordinal.codes) == {"g1": 1, "g2": 2}


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param({"reference": 1, "contrast": 4}, id="the-unmeasured-level"),
        pytest.param({"reference": 2, "contrast": 2}, id="one-group"),
        pytest.param({"reference": 0, "contrast": 1}, id="no-such-code"),
        pytest.param({"reference": True, "contrast": 2}, id="not-a-code"),
        pytest.param({"reference": 1}, id="no-contrast"),
        pytest.param({"reference": 1, "contrast": 2, "order": 3}, id="more"),
        pytest.param([1, 2], id="not-a-mapping"),
    ],
)
def test_a_grouping_compares_two_of_its_groups(raw: Any) -> None:
    with pytest.raises(ExposureGroupRuleError):
        read_group_contrast(raw, variable="lact_group_x1", groups=3)


def test_a_stage_seals_the_groups_its_study_compares(tmp_path: Path) -> None:
    refused, sealed = tmp_path / "refused", tmp_path / "sealed"
    for root in (refused, sealed):
        root.mkdir()
    with pytest.raises(MaterializedMetadataError, match="unreadable"):
        _stage(refused, _nominal(compared={"reference": 1, "contrast": 4}))

    staged = _stage(sealed, _nominal())
    assert staged is not None
    parameters = json.loads(
        json.dumps(materialized._thaw_json(staged.authority.producer_parameters))
    )
    assert parameters["groupings"][0]["compared"] == {"reference": 1, "contrast": 3}
    # A sealed comparison is read again on load.
    parameters["groupings"][0]["compared"] = {"reference": 3, "contrast": 3}
    forged = replace(
        staged.authority,
        producer_parameters=parameters,
        producer_parameters_sha256=canonical_parameters_sha256(parameters),
    )
    _resign(sealed.joinpath(*_RUN_COHORT), forged)

    with pytest.raises(MaterializedMetadataError, match="unreadable"):
        load_verified_materialized_cohort_authority(sealed.joinpath(*_RUN_COHORT))


def test_without_otherwise_only_a_measured_stay_can_meet_no_group() -> None:
    def rules(*groups: dict[str, Any]):
        return read_grouping_rules({**_ordinal()["parameters"], "groups": list(groups)})

    covering = rules(
        {"id": "g1", "rule": _rule("max", "<", 2.5)},
        {"id": "g2", "rule": _rule("max", ">=", 2.5)},
    )
    gapped = rules(
        {"id": "g1", "rule": _rule("max", "<", 2.0)},
        {"id": "g2", "rule": _rule("max", ">", 3.0)},
    )

    # An unmeasured stay meets no threshold either, but it is left to the
    # unmeasured handling; only a measured value between the rules is refused.
    table = pa.table({"lact_max": [1.0, None, 3.0]})
    assert evaluate_grouping(covering, table).to_pylist() == [1, None, 2]
    with pytest.raises(ExposureGroupRuleError, match="^1 measured stays meet no group"):
        evaluate_grouping(gapped, pa.table({"lact_max": [2.5, None]}))


def _resign(cohort_path: Path, authority) -> None:
    authority_ref = materialized._write_authority(cohort_path.parent, authority)
    provenance_path = materialized.materialized_provenance_path(cohort_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    sidecar_ref = authority.column_metadata
    sidecar = materialized.read_content_addressed_sidecar(
        cohort_path.parent / sidecar_ref.file,
        expected_sha256=sidecar_ref.sha256,
        expected_size=sidecar_ref.size,
    )
    provenance["column_metadata"] = materialized._descriptor(
        authority=authority_ref,
        sidecar=sidecar_ref,
        file_binding=sidecar.files[0],
    )
    materialized._atomic_write_json(provenance_path, provenance)


def test_an_authority_whose_rules_the_rows_do_not_follow_is_refused(
    tmp_path: Path,
) -> None:
    staged = _stage(tmp_path, _ordinal())
    assert staged is not None
    parameters = json.loads(
        json.dumps(materialized._thaw_json(staged.authority.producer_parameters))
    )
    # Another threshold, the staged codes unchanged.
    parameters["groupings"][0]["parameters"]["groups"][0]["rule"]["value"] = 9.5
    forged = replace(
        staged.authority,
        producer_parameters=parameters,
        producer_parameters_sha256=canonical_parameters_sha256(parameters),
    )
    _resign(tmp_path.joinpath(*_RUN_COHORT), forged)

    with pytest.raises(MaterializedMetadataError, match="codes its rules give"):
        load_verified_materialized_cohort_authority(tmp_path.joinpath(*_RUN_COHORT))


@pytest.mark.parametrize(
    "source_columns",
    [
        pytest.param({"max": "lact_min"}, id="another-summary"),
        pytest.param({"max": "age"}, id="another-concept"),
        pytest.param({"max": "lact_absent"}, id="a-column-the-cohort-lacks"),
    ],
)
def test_a_grouping_reads_only_the_stated_summary_of_its_concept(
    tmp_path: Path, source_columns: dict[str, str]
) -> None:
    grouping = _ordinal()
    grouping["parameters"] = {
        **grouping["parameters"],
        "source_columns": source_columns,
    }

    with pytest.raises(MaterializedMetadataError, match="an exposure grouping reads"):
        _stage(tmp_path, grouping)
    # The run's exact copy stays as it was.
    copy = load_verified_materialized_cohort_authority(tmp_path.joinpath(*_RUN_COHORT))
    assert copy is not None and copy.authority.producer == "research_agent_run_stage"


def test_a_grouping_over_another_window_is_refused(tmp_path: Path) -> None:
    grouping = _ordinal()
    grouping["parameters"] = {
        **grouping["parameters"],
        "window": {"start_hours": 0.0, "end_hours": 12.0},
    }

    with pytest.raises(MaterializedMetadataError, match="an exposure grouping reads"):
        _stage(tmp_path, grouping)


def test_a_grouping_cannot_take_a_name_the_cohort_holds(tmp_path: Path) -> None:
    with pytest.raises(MaterializedMetadataError, match="already holds"):
        _stage(tmp_path, _nominal(variable="age"))


def test_exposure_groups_are_staged_only_on_the_runs_exact_copy(
    tmp_path: Path,
) -> None:
    stage = {
        "groupings": [_ordinal()],
        "groupings_record_sha256": "c" * 64,
        "producer_implementation_sha256": "a" * 64,
    }
    _stage(tmp_path, _nominal())

    # Neither the source nor a cohort already grouped is a run's exact copy.
    for cohort in (
        tmp_path / "materialized" / "universe.parquet",
        tmp_path.joinpath(*_RUN_COHORT),
    ):
        with pytest.raises(MaterializedMetadataError, match="exact typed copy"):
            stage_exposure_grouped_cohort_authority(cohort, **stage)


def test_an_interrupted_restage_leaves_no_cohort_that_loads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copy = _run_copy(tmp_path)

    def interrupted(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("interrupted")

    monkeypatch.setattr(materialized.pq, "write_table", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        stage_exposure_grouped_cohort_authority(
            copy,
            groupings=[_ordinal()],
            groupings_record_sha256="c" * 64,
            producer_implementation_sha256="a" * 64,
        )

    with pytest.raises(MaterializedMetadataError):
        load_verified_materialized_cohort_authority(copy)


def test_the_context_declares_each_groupings_levels(tmp_path: Path) -> None:
    _stage(tmp_path, _nominal(), _ordinal())

    context = build_research_context(
        research_question="Compare death across lactate groups.",
        cohort=tmp_path.joinpath(*_RUN_COHORT),
        cohort_name="grouped",
        database="miiv",
        target_outcome="death",
        primary_exposure="lact_group_x1",
        id_columns=("stay_id",),
        outcome_columns=("death",),
    )

    variables = {item.name: item for item in context.variables}
    nominal, ordinal = variables["lact_group_x1"], variables["lact_group_x2"]
    assert declared_domain_for_variable(nominal) == (
        [1, 2, 3, 4],
        "declared_exposure_group_levels",
    )
    assert declared_domain_for_variable(ordinal) == ([1, 2], "declared_ordinal_levels")
    assert (nominal.is_ordinal, nominal.ordinal_levels) == (False, None)
    assert (ordinal.is_ordinal, ordinal.ordinal_levels) == (True, [1, 2])
    for variable in (nominal, ordinal):
        assert (variable.role, variable.unit) == (VariableRole.OTHER, "category")
        # Its concept is the grouping's own, derived from the grouped one.
        assert variable.source_concept == variable.name
        assert "lact" in variable.derived_from_concepts
    assert "1 = low; 2 = high only; 3 = neither; 4 = not measured" in (
        nominal.description or ""
    )
    # The groups each grouping compares are its sealed ones, as level codes.
    assert context.cohort.provenance[EXPOSURE_GROUP_CONTRASTS_KEY] == [
        {"variable": "lact_group_x1", "reference": 1, "contrast": 3},
        {"variable": "lact_group_x2", "reference": 1, "contrast": 2},
    ]


def test_an_ordinal_grouping_lets_its_unmeasured_stays_leave() -> None:
    with pytest.raises(ExposureGroupRuleError, match="on no scale"):
        read_grouping_rules({**_ordinal()["parameters"], "unmeasured": "own_group"})


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        pytest.param(
            {
                "id": "x1",
                "concept": "glu",
                "window": _DAY,
                "scale": "nominal",
                "groups": [
                    {
                        "id": "g1",
                        "label": "hypoglycaemia",
                        "rule": _rule("min", "<", 70),
                    },
                    {
                        "id": "g2",
                        "label": "hyperglycaemia only",
                        "rule": _rule("max", ">", 180),
                    },
                    {"id": "g3", "label": "normoglycaemia", "rule": "otherwise"},
                ],
                "unmeasured": {"handling": "own_group", "label": "no glucose"},
                "quote": "glucose below 70 or above 180",
                "source": "question",
            },
            {"reference": 1, "contrast": 3},
            id="nominal",
        ),
        pytest.param(
            {
                "id": "x2",
                "concept": "bmi",
                "window": None,
                "scale": "ordinal",
                "groups": [
                    {
                        "id": "g1",
                        "label": "under 18.5",
                        "rule": _rule("value", "<", 18.5),
                    },
                    {
                        "id": "g3",
                        "label": "30 or more",
                        "rule": _rule("value", ">=", 30),
                    },
                    {"id": "g2", "label": "18.5 to 29.9", "rule": "otherwise"},
                ],
                "unmeasured": {"handling": "exclude"},
                "reference": "g2",
                "quote": "body mass index category",
                "source": "outline",
            },
            {"reference": 2, "contrast": 3},
            id="ordinal",
        ),
    ],
)
def test_the_host_compile_states_parameters_the_kernel_reads(
    spec: dict[str, Any], expected: dict[str, int]
) -> None:
    grouping = ExposureGroupSpec.model_validate(spec)
    summaries = {
        group.rule.summary for group in grouping.groups if group.rule != "otherwise"
    }
    compiled = CompiledGrouping(
        grouping=grouping,
        disposition="applied",
        reason=None,
        detail="",
        source_columns=MappingProxyType(
            {summary: f"{grouping.concept}_{summary}" for summary in summaries}
        ),
    )

    rules = read_grouping_rules(compiled.derivation())

    assert rules.levels == grouping_levels(grouping)
    assert rules.scale == grouping.scale
    compared = group_contrast(compiled).compared()
    assert (
        read_group_contrast(
            compared, variable="lact_group_x1", groups=len(rules.groups)
        ).compared()
        == compared
    )
    assert compared == expected
