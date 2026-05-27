"""CTAS cohort time-aggregation schema tests."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest


def _age_predicate(start: float, end: float):
    from easyicu.research_agent.cohort_schema import ConceptPredicate, TimeWindow

    return ConceptPredicate(
        concept_id="age",
        time_window=TimeWindow(
            anchor="icu_admit",
            start_offset_hours=start,
            end_offset_hours=end,
        ),
        aggregation="max",
        op=">=",
        value=18,
    )


def test_concept_predicate_rejects_missing_time_window() -> None:
    from easyicu.research_agent.cohort_schema import ConceptPredicate, CohortSchemaError

    with pytest.raises(CohortSchemaError, match="time_window"):
        ConceptPredicate.from_dict({"concept_id": "age", "aggregation": "max", "op": ">="})


def test_concept_predicate_rejects_missing_aggregation() -> None:
    from easyicu.research_agent.cohort_schema import ConceptPredicate, CohortSchemaError

    with pytest.raises(CohortSchemaError, match="aggregation"):
        ConceptPredicate.from_dict(
            {
                "concept_id": "age",
                "time_window": {
                    "anchor": "icu_admit",
                    "start_offset_hours": 0,
                    "end_offset_hours": 24,
                },
                "op": ">=",
            }
        )


def test_aggregation_op_incompatibility_rejected() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortSchemaError,
        ConceptPredicate,
        TimeWindow,
    )

    with pytest.raises(CohortSchemaError, match="only supports"):
        ConceptPredicate(
            concept_id="mech_vent",
            time_window=TimeWindow("icu_admit", 0, 24),
            aggregation="any",
            op=">=",
            value=1,
        )


def test_unknown_concept_id_rejected() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortSchemaError,
        ConceptPredicate,
        TimeWindow,
    )

    with pytest.raises(CohortSchemaError, match="unknown concept_id"):
        ConceptPredicate(
            concept_id="not_a_real_easyicu_concept",
            time_window=TimeWindow("icu_admit", 0, 24),
            aggregation="max",
            op=">=",
            value=1,
        )


def test_time_window_accepts_case_owned_anchor_string() -> None:
    from easyicu.research_agent.cohort_schema import TimeWindow, UNIVERSAL_ANCHORS

    window = TimeWindow("delirium_onset", 0, 24)

    assert window.anchor == "delirium_onset"
    assert "delirium_onset" not in UNIVERSAL_ANCHORS


def test_registered_pattern_expansion_is_deterministic() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        expand_named_cohort,
        register_pattern,
        reset_pattern_registry,
    )

    reset_pattern_registry()
    try:
        register_pattern(
            "adult_admission_window",
            CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),)),
            provenance="test fixture",
        )
        first = expand_named_cohort("adult_admission_window").to_dict()
        second = expand_named_cohort("adult_admission_window").to_dict()
        assert first == second
        assert first["derived_from_named"] == "adult_admission_window"
    finally:
        reset_pattern_registry()


def test_two_registered_patterns_with_different_windows_have_different_hash() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        cohort_definition_sha,
        expand_named_cohort,
        register_pattern,
        reset_pattern_registry,
    )

    reset_pattern_registry()
    try:
        register_pattern(
            "adult_first_day",
            CohortDefinition(name="adult_first_day", inclusion=(_age_predicate(0, 24),)),
            provenance="test fixture",
        )
        register_pattern(
            "adult_first_hour",
            CohortDefinition(name="adult_first_hour", inclusion=(_age_predicate(0, 1),)),
            provenance="test fixture",
        )
        assert cohort_definition_sha(
            expand_named_cohort("adult_first_day")
        ) != cohort_definition_sha(expand_named_cohort("adult_first_hour"))
    finally:
        reset_pattern_registry()


def test_unknown_named_pattern_rejected() -> None:
    from easyicu.research_agent.cohort_schema import reset_pattern_registry

    from easyicu.research_agent.schema import AnalysisPlan

    reset_pattern_registry()
    with pytest.raises(ValueError, match="unknown named cohort pattern"):
        AnalysisPlan(
            research_question="Does a predictor associate with mortality?",
            cohort={"from_named": "case_specific_pattern_not_registered"},
            steps=[],
        )


def test_planner_string_cohort_rejected() -> None:
    from easyicu.research_agent.schema import AnalysisPlan

    with pytest.raises(ValueError, match="free-text cohort strings"):
        AnalysisPlan(
            research_question="Does a predictor associate with mortality?",
            cohort="SOFA-2 = 0 patients",
            steps=[],
        )


def test_planner_named_cohort_accepted_and_expanded() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        register_pattern,
        reset_pattern_registry,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    reset_pattern_registry()
    try:
        register_pattern(
            "adult_admission_window",
            CohortDefinition(name="adult", inclusion=(_age_predicate(0, 24),)),
            provenance="test fixture",
        )
        plan = AnalysisPlan(
            research_question="Does a predictor associate with mortality?",
            cohort={"from_named": "adult_admission_window"},
            steps=[],
        )
        assert plan.cohort is not None
        assert plan.cohort.derived_from_named == "adult_admission_window"
        assert plan.cohort.inclusion[0].time_window.end_offset_hours == 24
    finally:
        reset_pattern_registry()


def test_robustness_spec_cohort_override_schema_validated() -> None:
    from easyicu.research_agent.cohort_schema import CohortSchemaError
    from easyicu.research_agent.robustness_panel import RobustnessSpec

    with pytest.raises(CohortSchemaError, match="unknown concept_id"):
        RobustnessSpec.from_dict(
            {
                "spec_id": "bad_cohort",
                "axis": "cohort",
                "description": "Invalid concept.",
                "cohort_override": {
                    "name": "bad",
                    "inclusion": [
                        {
                            "concept_id": "not_a_real_easyicu_concept",
                            "time_window": {
                                "anchor": "icu_admit",
                                "start_offset_hours": 0,
                                "end_offset_hours": 24,
                            },
                            "aggregation": "max",
                            "op": ">=",
                            "value": 1,
                        }
                    ],
                    "exclusion": [],
                },
            }
        )


def test_cohort_locked_recorded_in_manifest(ra, tmp_path: Path) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        COHORT_LOCK_FILENAME,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisManifest, AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with mortality?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    evidence = ra.EvidenceStore(tmp_path)
    path = write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = AnalysisManifest(
        run_id="r1",
        research_question=plan.research_question,
        started_at=datetime.now(timezone.utc),
        context_path="context.json",
        cohort_locked_path=COHORT_LOCK_FILENAME,
        cohort_locked_sha=digest,
    )
    assert manifest.cohort_locked_path == COHORT_LOCK_FILENAME
    assert manifest.cohort_locked_sha == digest


def test_assert_cohort_definition_locked_catches_post_lock_mutation(
    ra,
    tmp_path: Path,
) -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        CohortSchemaError,
        assert_cohort_definition_locked,
        write_locked_cohort_definition,
    )
    from easyicu.research_agent.schema import AnalysisPlan

    plan = AnalysisPlan(
        research_question="Does a predictor associate with mortality?",
        cohort=CohortDefinition(name="primary", inclusion=(_age_predicate(0, 24),)),
        steps=[],
    )
    evidence = ra.EvidenceStore(tmp_path)
    write_locked_cohort_definition(
        run_dir=tmp_path,
        plan=plan,
        evidence=evidence,
        prompt_pack_version="test",
        llm_signature="mock",
    )

    mutated_plan = plan.model_copy(
        update={
            "cohort": CohortDefinition(
                name="primary",
                inclusion=(_age_predicate(0, 1),),
            )
        }
    )
    with pytest.raises(CohortSchemaError, match="changed after plan lock"):
        assert_cohort_definition_locked(run_dir=tmp_path, plan=mutated_plan)


def test_builder_rejects_unsupported_aggregation_with_not_implemented() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
        build_cohort,
    )

    definition = CohortDefinition(
        name="mean_age",
        inclusion=(
            ConceptPredicate(
                "age",
                TimeWindow("icu_admit", 0, 24),
                "mean",
                ">=",
                18,
            ),
        ),
    )
    with pytest.raises(NotImplementedError, match="aggregation 'mean'"):
        build_cohort(definition, pd.DataFrame({"age": [21, 17]}))


def test_builder_missing_materialised_column_is_data_error() -> None:
    from easyicu.research_agent.cohort_schema import (
        CohortDataError,
        CohortDefinition,
        build_cohort,
    )

    definition = CohortDefinition(
        name="adult",
        inclusion=(_age_predicate(0, 24),),
    )
    with pytest.raises(CohortDataError, match="missing concept column 'age'"):
        build_cohort(definition, pd.DataFrame({"other_column": [1, 2]}))
