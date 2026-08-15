from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from easyicu.research_agent.execution.runners.descriptive_association_executor import (
    DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND,
    descriptive_association_executor_owns_step,
    run_descriptive_association,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import AnalysisPlan, AnalysisStep


def _step(**updates) -> AnalysisStep:
    payload = {
        "step_id": "05_age_los_association",
        "planned_analysis_role": "auxiliary",
        "intent": "Describe the monotonic association without a causal claim.",
        "inputs": ["artifact:analysis_cohort", "age", "los_icu"],
        "expected_outputs": ["statistic:age_los_spearman"],
        "method": "descriptive_association",
    }
    payload.update(updates)
    return AnalysisStep.model_validate(payload)


def _plan(step: AnalysisStep) -> AnalysisPlan:
    return AnalysisPlan.model_validate(
        {
            "research_question": "Describe age and ICU stay length.",
            "analysis_type": "descriptive_epidemiology",
            "steps": [step.model_dump(mode="json")],
            "cohort": {"name": "primary"},
            "display_labels": {},
            "rationale": "Descriptive only.",
        }
    )


def test_descriptive_association_owner_is_exact_and_selected() -> None:
    step = _step()
    assert descriptive_association_executor_owns_step(step)
    selection = select_standard_executor(step, plan=_plan(step))
    assert selection is not None
    assert selection.analysis_kind == DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND
    assert selection.consumed_input_keys == ("artifact:analysis_cohort",)
    assert "run_descriptive_association" in selection.code

    assert descriptive_association_executor_owns_step(
        _step(planned_analysis_role="primary")
    )
    assert descriptive_association_executor_owns_step(
        _step(planned_analysis_role="secondary")
    )
    assert not descriptive_association_executor_owns_step(
        _step(method="adjusted_association_models")
    )
    assert not descriptive_association_executor_owns_step(
        _step(inputs=["artifact:analysis_cohort", "age", "sex", "los_icu"])
    )
    assert not descriptive_association_executor_owns_step(
        _step(expected_outputs=["table:age_los_spearman"])
    )


def test_descriptive_association_writes_one_digest_ready_statistic(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0, None],
            "los_icu": [1.0, 2.0, 4.0, 8.0, 3.0],
        }
    )
    summary = run_descriptive_association(
        frame=frame,
        predictor="age",
        outcome="los_icu",
        statistic_product="age_los_spearman",
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=tmp_path / "cohort.parquet",
        out_dir=tmp_path / "outputs",
    )
    payload = json.loads(
        (tmp_path / "outputs" / "age_los_spearman.json").read_text("utf-8")
    )

    assert summary["deterministic_standard_analysis"] == (
        DESCRIPTIVE_ASSOCIATION_ANALYSIS_KIND
    )
    assert summary["output_files"] == {
        "statistic:age_los_spearman": "age_los_spearman.json"
    }
    assert payload["name"] == "age_los_spearman"
    assert payload["value"] == 1.0
    assert payload["n_complete_case"] == 4
    assert payload["n_total"] == 5
    assert payload["interpretation_class"] == "descriptive_noncausal_association"


def test_descriptive_association_fails_closed_on_nonnumeric_or_constant(
    tmp_path: Path,
) -> None:
    bad = pd.DataFrame({"age": [20, "unknown", 40], "los_icu": [1, 2, 3]})
    try:
        run_descriptive_association(
            frame=bad,
            predictor="age",
            outcome="los_icu",
            statistic_product="age_los_spearman",
            typed_cohort_input="artifact:analysis_cohort",
            source_cohort=tmp_path / "cohort.parquet",
            out_dir=tmp_path / "bad",
        )
    except RuntimeError as exc:
        assert "non-numeric" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("nonnumeric association input must fail closed")

    constant = pd.DataFrame({"age": [20, 20, 20], "los_icu": [1, 2, 3]})
    try:
        run_descriptive_association(
            frame=constant,
            predictor="age",
            outcome="los_icu",
            statistic_product="age_los_spearman",
            typed_cohort_input="artifact:analysis_cohort",
            source_cohort=tmp_path / "cohort.parquet",
            out_dir=tmp_path / "constant",
        )
    except RuntimeError as exc:
        assert "constant" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("constant association input must fail closed")
