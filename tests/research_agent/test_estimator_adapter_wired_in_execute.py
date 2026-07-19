"""Regression tests for execute-phase deterministic robustness fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def test_opt_in_adapter_primary_cannot_enter_final_panel(tmp_path) -> None:
    from easyicu.research_agent.cohort.schema import (
        CohortDefinition,
        ConceptPredicate,
        TimeWindow,
    )
    from easyicu.research_agent.estimators import fit_robustness_rows_from_records
    from easyicu.research_agent.robustness_panel import (
        RobustnessSpec,
        build_robustness_panel_from_records,
    )
    from easyicu.research_agent.schema import (
        CohortDescriptor,
        ConceptDescriptor,
        ResearchContext,
        VariableRole,
    )

    rng = np.random.default_rng(13)
    sofa = rng.normal(2.0, 1.0, size=900)
    p = 1 / (1 + np.exp(-(-0.5 + 0.55 * sofa)))
    df = pd.DataFrame(
        {
            "sofa2_admission": sofa,
            "death": rng.binomial(1, p),
            "age": rng.normal(65, 12, size=900),
        }
    )
    cohort_path = tmp_path / "cohort.parquet"
    df.to_parquet(cohort_path)

    context = ResearchContext(
        research_question="Is admission SOFA-2 associated with ICU mortality?",
        cohort=CohortDescriptor(
            cohort_name="synthetic",
            database="mock",
            n_patients=900,
            n_stays=900,
            outcome_columns=["death"],
        ),
        variables=[
            ConceptDescriptor(
                name="sofa2_admission",
                role=VariableRole.COMPOSITE_SCORE,
                dtype="float64",
            ),
            ConceptDescriptor(name="death", role=VariableRole.OUTCOME, dtype="int64"),
            ConceptDescriptor(name="age", role=VariableRole.DEMOGRAPHIC, dtype="float64"),
        ],
        target_outcome="death",
        cohort_parquet=str(cohort_path),
    )
    window = TimeWindow(anchor="icu_admit", start_offset_hours=0, end_offset_hours=24)
    primary = CohortDefinition(
        name="primary",
        inclusion=(
            ConceptPredicate(
                concept_id="sofa",
                time_window=window,
                aggregation="max",
                op=">=",
                value=0,
            ),
        ),
    )
    specs = [
        RobustnessSpec(
            spec_id="alt_sofa_positive",
            axis="cohort",
            description="Exclude the zero-score stratum.",
            cohort_override=CohortDefinition(
                name="sofa_positive",
                inclusion=(
                    ConceptPredicate(
                        concept_id="sofa",
                        time_window=window,
                        aggregation="max",
                        op=">",
                        value=0,
                    ),
                ),
            ),
        )
    ]

    rows, warnings = fit_robustness_rows_from_records(
        specs=specs,
        per_step_records=[],
        primary_cohort=primary,
        cohort_path=cohort_path,
        context=context,
        allow_implicit_cohort_refit=True,
    )
    panel = build_robustness_panel_from_records(
        specs=specs,
        per_step_records=[],
        adapter_rows=rows,
    )

    assert any("cohort parquet" in warning for warning in warnings)
    assert {row.spec_id for row in rows} == {"primary", "alt_sofa_positive"}
    assert {row.spec_id for row in panel.rows} == {"alt_sofa_positive"}
    assert sum(row.converged for row in panel.rows) >= 1
    assert all(row.point_estimate is not None for row in panel.rows)
