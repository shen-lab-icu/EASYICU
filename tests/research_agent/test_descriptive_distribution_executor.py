from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.execution.runners.descriptive_distribution_executor import (
    DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND,
    descriptive_distribution_executor_owns_step,
    run_descriptive_distribution_summary,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.authority.plausibility import FlagOnlyPlausibilityScope
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "04_distribution",
        "planned_analysis_role": "auxiliary",
        "intent": "Summarize a continuous variable by a closed category.",
        "inputs": ["artifact:analysis_cohort", "sex", "los_icu"],
        "expected_outputs": ["table:distribution_prevalence"],
        "method": "descriptive_distribution_summary",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _plan(step: AnalysisStep) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Describe the distribution.",
            "analysis_type": "descriptive_epidemiology",
            "steps": [step.model_dump(mode="json")],
            "cohort": {"name": "primary"},
            "display_labels": {},
            "rationale": "Descriptive only.",
        }
    )


def test_distribution_owner_is_exact_and_selected() -> None:
    step = _step()
    assert descriptive_distribution_executor_owns_step(step)
    selection = select_standard_executor(step, plan=_plan(step))
    assert selection is not None
    assert selection.analysis_kind == DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND
    assert selection.consumed_input_keys == ("artifact:analysis_cohort",)
    assert "run_descriptive_distribution_summary" in selection.code

    assert descriptive_distribution_executor_owns_step(
        _step(method="descriptive_distribution")
    )
    assert descriptive_distribution_executor_owns_step(
        _step(planned_analysis_role="primary")
    )
    assert descriptive_distribution_executor_owns_step(
        _step(planned_analysis_role="secondary")
    )

    assert not descriptive_distribution_executor_owns_step(
        _step(method="visualization")
    )
    assert not descriptive_distribution_executor_owns_step(
        _step(inputs=["artifact:analysis_cohort", "los_icu"])
    )


def test_distribution_selection_keeps_plausibility_receipt_attributable() -> None:
    step = _step()
    scope = FlagOnlyPlausibilityScope(
        step_id=step.step_id,
        expected_columns=("los_icu",),
        source_contracts_sha256="a" * 64,
        authority_kind="resolved_raw_input_contracts",
    )
    selection = select_standard_executor(
        step,
        plan=_plan(step),
        plausibility_scope=scope,
    )
    assert selection is not None
    compile(selection.code, "<distribution-executor>", "exec")
    assert 'plausibility_frame = frame' not in selection.code
    assert "plausibility_audit[column]" in selection.code
    assert 'summary["plausibility_audit"] = plausibility_audit' in selection.code


def test_distribution_rows_are_explicit_partition_and_digest_ready(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    out_dir = run_dir / "steps" / "04_distribution" / "outputs"
    run_dir.mkdir(parents=True)
    (run_dir / "research_context.json").write_text(
        json.dumps(
            {
                "variables": [
                    {
                        "name": "sex",
                        "observed_domain": {"levels": ["Female", "Male"]},
                    },
                    {"name": "los_icu", "unit": "days"},
                ]
            }
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "sex": ["Female", "Female", "Male", "Male"],
            "los_icu": [1.0, 3.0, 2.0, 6.0],
        }
    )
    summary = run_descriptive_distribution_summary(
        frame=frame,
        grouping_variable="sex",
        value_variable="los_icu",
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=run_dir / "cohort.parquet",
        out_dir=out_dir,
        run_dir=run_dir,
    )
    table = pd.read_csv(out_dir / "distribution_prevalence.csv")

    assert summary["deterministic_standard_analysis"] == (
        DESCRIPTIVE_DISTRIBUTION_ANALYSIS_KIND
    )
    assert table["row_role"].tolist() == [
        "overall",
        "exposure_level",
        "exposure_level",
    ]
    assert table["group_n"].tolist() == [4, 2, 2]
    assert table.loc[0, "group_n"] == table.loc[1:, "group_n"].sum()
    assert table["n_nonmissing"].tolist() == [4, 2, 2]
    assert table["unit"].tolist() == ["days", "days", "days"]
    assert table.loc[table["group"] == "Female", "median"].item() == 2.0
