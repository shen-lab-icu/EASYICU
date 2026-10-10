"""A column without values stops the owner that reads it, by one rule.

A column no row holds a value of reads nothing: a cohort predicate over it
keeps no stay or excludes none, and an imputer fitted on it drops it with only
a warning, so the analysis is not the one planned.  Each owner that filters or
fits rows refuses such a column in the rows it uses
(``contracts.concept_values.columns_without_values``).  Synthetic data only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from easyicu.research_agent.cohort.schema import (
    CohortPredicateColumnWithoutValuesError,
    build_cohort,
)
from easyicu.research_agent.execution.runners.cross_sectional_phenotyping_executor import (
    run_primary_phenotyping,
)
from easyicu.research_agent.methods.subtype_assignment import (
    SubtypeAssignmentError,
    fit_and_evaluate_early_subtype_assignment,
)
from easyicu.research_agent.planning.cohort_contract import (
    CohortDefinition,
    ConceptPredicate,
    TimeWindow,
)
from easyicu.research_agent.planning.robustness_contract import RobustnessSpec
from easyicu.research_agent.robustness.panel import robustness_specs_sha
from easyicu.research_agent.schema import (
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)

_DAY = TimeWindow(anchor="icu_admission", start_offset_hours=0.0, end_offset_hours=24.0)


def _excluding(concept: str) -> CohortDefinition:
    return CohortDefinition(
        name="primary",
        inclusion=(
            ConceptPredicate(
                concept_id="age", time_window=_DAY, aggregation="first", op=">=", value=18
            ),
        ),
        exclusion=(
            ConceptPredicate(
                concept_id=concept, time_window=_DAY, aggregation="max", op=">", value=4
            ),
        ),
    )


def test_an_exclusion_over_a_column_without_values_stops_the_cohort() -> None:
    data = pd.DataFrame(
        {
            "stay_id": [1, 2, 3, 4, 5],
            "age": [40, 55, 17, 70, 66],
            "lact": [5.0, 1.0, 1.0, np.nan, 6.0],
            "crea": [np.nan] * 5,
        }
    )
    kept = build_cohort(_excluding("lact"), data)
    assert sorted(kept["stay_id"]) == [2, 4]

    with pytest.raises(CohortPredicateColumnWithoutValuesError) as stopped:
        build_cohort(_excluding("crea"), data)

    assert stopped.value.code == "cohort_predicate_column_without_values"
    assert (stopped.value.kind, stopped.value.column, stopped.value.stays) == (
        "exclusion",
        "crea",
        5,
    )
    # A value the stays left at the criterion happen to miss is a missing value.
    sparse = data.assign(crea=[1.0, np.nan, np.nan, np.nan, np.nan])
    assert sorted(build_cohort(_excluding("crea"), sparse)["stay_id"]) == [1, 2, 4, 5]
    # Rows decide it only when there are rows to decide it.
    assert build_cohort(_excluding("crea"), data.iloc[0:0]).empty


def _phenotyping_context(n: int) -> ResearchContext:
    return ResearchContext(
        research_question="Discover early cross-sectional phenotypes.",
        cohort=CohortDescriptor(
            cohort_name="phenotype_fixture",
            database="synthetic",
            n_stays=n,
            id_columns=["stay_id"],
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(name="stay_id", role=VariableRole.ID, dtype="str"),
            ConceptDescriptor(name="marker_a", role=VariableRole.LAB, dtype="float64"),
            ConceptDescriptor(name="marker_b", role=VariableRole.VITAL, dtype="float64"),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
        ],
        target_outcome="death",
    )


def test_a_complete_case_refit_that_would_drop_a_feature_stops(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    n = 360
    group = np.repeat(np.arange(3), n // 3)
    frame = pd.DataFrame(
        {
            "stay_id": [f"stay_{index}" for index in range(n)],
            "marker_a": rng.normal(group * 2.5, 0.5),
            "marker_b": rng.normal((2 - group) * 2.0, 0.6),
            "death": rng.binomial(1, 0.2 + group * 0.1),
        }
    )
    # marker_b is recorded only where marker_a is not, so the stays complete
    # on marker_a hold no marker_b: the refit's imputer would drop it.
    recorded_a = rng.random(n) < 0.7
    frame.loc[~recorded_a, "marker_a"] = np.nan
    frame.loc[recorded_a, "marker_b"] = np.nan
    (tmp_path / "research_context.json").write_text(
        _phenotyping_context(n).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    spec = RobustnessSpec(
        spec_id="complete_case_marker_a",
        axis="missing",
        description="Refit on the stays with marker A recorded.",
        missing_override={"strategy": "complete_case", "variables": ["marker_a"]},
    )
    (tmp_path / "robustness_specs_locked.json").write_text(
        json.dumps(
            {
                "schema_version": "easyicu.robustness_specs/1",
                "locked_at": "2026-10-10T00:00:00+00:00",
                "spec_sha256": robustness_specs_sha([spec]),
                "specs": [spec.to_dict()],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="phenotyping_feature_unobserved") as stopped:
        run_primary_phenotyping(
            frame=frame,
            declared_columns=("stay_id", "marker_a", "marker_b", "death"),
            feature_columns=("marker_a", "marker_b"),
            typed_cohort_input="artifact:analysis_cohort",
            source_cohort=cohort_path,
            out_dir=tmp_path / "primary",
            run_dir=tmp_path,
            step_id="primary_phenotypes",
        )
    assert "'marker_b'" in str(stopped.value)


def test_a_subtype_feature_no_development_patient_holds_is_refused() -> None:
    rng = np.random.default_rng(7)

    def features(n: int) -> pd.DataFrame:
        return pd.DataFrame({"lact": rng.normal(2, 1, n), "map": rng.normal(70, 8, n)})

    development = features(60)
    development["map"] = np.nan
    with pytest.raises(SubtypeAssignmentError, match="at least one observed value"):
        fit_and_evaluate_early_subtype_assignment(
            development,
            [f"s{index % 2}" for index in range(60)],
            features(40),
            [f"s{index % 2}" for index in range(40)],
            development_patient_ids=[f"d{index}" for index in range(60)],
            validation_patient_ids=[f"v{index}" for index in range(40)],
            feature_window_end=6.0,
            phenotype_window_start=24.0,
        )
