from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from easyicu.research_agent.execution.runners.prediction_figure_executor import (
    PREDICTION_COMPOSITE_FIGURE_INPUTS,
    PREDICTION_FIGURE_ANALYSIS_KIND,
    prediction_figure_executor_owns_step,
    run_prediction_figure,
)
from easyicu.research_agent.execution.runners.prediction_model_executor import (
    PREDICTION_MODEL_ANALYSIS_KIND,
    PREDICTION_SCORES_PRODUCT,
    prediction_model_executor_owns_step,
    run_prediction_model,
    run_prediction_score_analysis,
)
from easyicu.research_agent.execution.runners.selection import select_standard_executor
from easyicu.research_agent.schema import (
    AnalysisPlan,
    AnalysisStep,
    CohortDescriptor,
    ConceptDescriptor,
    ResearchContext,
    VariableRole,
)


def _context(row_count: int) -> ResearchContext:
    return ResearchContext(
        research_question="Predict a binary outcome from prespecified variables.",
        cohort=CohortDescriptor(
            cohort_name="prediction_fixture",
            database="synthetic",
            n_stays=row_count,
            id_columns=["patient_stay_id"],
            outcome_columns=["death"],
            provenance={
                "replacement_row_identity": {
                    "output_identity_column": "patient_stay_id",
                    "mapping_file_sha256": "a" * 64,
                    "patient_group_derivation": {
                        "algorithm": "prefix_before_:s",
                        "delimiter": ":s",
                    },
                }
            },
        ),
        variables=[
            ConceptDescriptor(name="age", dtype="float64"),
            ConceptDescriptor(
                name="sex",
                dtype="object",
                observed_domain={"n_unique": 2, "levels": ["F", "M"]},
            ),
            ConceptDescriptor(name="marker", dtype="float64"),
            ConceptDescriptor(
                name="death",
                role=VariableRole.OUTCOME,
                dtype="int64",
                observed_domain={
                    "n_unique": 2,
                    "is_binary": True,
                    "levels": [0, 1],
                },
            ),
        ],
        target_outcome="death",
    )


def _frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    subject_count = 90
    stays_per_subject = np.where(np.arange(subject_count) % 4 == 0, 2, 1)
    subjects = np.repeat(np.arange(subject_count), stays_per_subject)
    stay_number = np.concatenate([np.arange(count) + 1 for count in stays_per_subject])
    age = rng.normal(64, 12, len(subjects))
    marker = rng.normal(0, 1, len(subjects))
    sex = np.where(subjects % 2 == 0, "F", "M")
    logit = -2.0 + 0.025 * (age - 60) + 0.7 * marker + 0.25 * (sex == "M")
    probability = 1.0 / (1.0 + np.exp(-logit))
    death = rng.binomial(1, probability)
    # Keep both classes abundant so the one fixed group split is executable.
    death[:10] = np.arange(10) % 2
    frame = pd.DataFrame(
        {
            "patient_stay_id": [
                f"p{subject}:s{stay}"
                for subject, stay in zip(subjects, stay_number, strict=True)
            ],
            "age": age,
            "sex": sex,
            "marker": marker,
            "death": death,
        }
    )
    frame.loc[frame.index[::13], "marker"] = np.nan
    frame.loc[frame.index[::17], "sex"] = None
    return frame


def _binding(key: str, frame: pd.DataFrame, path: Path) -> dict[str, object]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    product = key.partition(":")[2]
    return {
        "declared_kind": "table",
        "evidence_kind": "table",
        "product": product,
        "relative_path": str(path.relative_to(path.parents[1])),
        "sha256": digest,
        "evidence_id": f"evidence_{product}",
        "product_contract": {"columns": list(frame.columns), "row_count": len(frame)},
        "consumption_contract": {
            "input_key": key,
            "mode": "all_rows",
            "artifact_sha256": digest,
        },
        "identity_row": {
            "input_key": key,
            "declared_kind": "table",
            "product": product,
            "evidence_id": f"evidence_{product}",
            "sha256": digest,
        },
    }


def _primary_step() -> AnalysisStep:
    return AnalysisStep(
        step_id="primary_model",
        planned_analysis_role="primary",
        intent="Fit the prespecified static mortality model.",
        inputs=["age", "sex", "marker", "death", "artifact:analysis_cohort"],
        expected_outputs=["table:prediction_scores", "table:model_performance"],
        method="logistic prediction model",
        scientific_action_id="prediction.discrimination_calibration",
    )


def test_prediction_owner_selects_only_exact_action_contract() -> None:
    step = _primary_step()
    assert prediction_model_executor_owns_step(step)
    selection = select_standard_executor(
        step,
        plan=AnalysisPlan(research_question="Predict mortality.", steps=[step]),
    )
    assert selection is not None
    assert selection.analysis_kind == PREDICTION_MODEL_ANALYSIS_KIND
    assert selection.consumed_input_keys == ("artifact:analysis_cohort",)
    widened = step.model_copy(
        update={"expected_outputs": [*step.expected_outputs, "table:extra"]}
    )
    assert not prediction_model_executor_owns_step(widened)


def test_prediction_workflow_is_group_safe_source_bound_and_renderable(
    tmp_path: Path,
) -> None:
    frame = _frame()
    (tmp_path / "research_context.json").write_text(
        _context(len(frame)).model_dump_json(indent=2), encoding="utf-8"
    )
    cohort_path = tmp_path / "cohort.csv"
    frame.to_csv(cohort_path, index=False)
    primary_dir = tmp_path / "primary"
    primary = run_prediction_model(
        frame=frame,
        declared_columns=("age", "sex", "marker", "death"),
        typed_cohort_input="artifact:analysis_cohort",
        source_cohort=cohort_path,
        out_dir=primary_dir,
        run_dir=tmp_path,
        step_id="primary_model",
    )
    assert primary["authority_scope"] == "analysis_only"
    scores = pd.read_csv(primary_dir / "prediction_scores.csv")
    assert scores.groupby("subject_id")["split"].nunique().max() == 1
    assert set(scores["split"]) == {"development", "validation"}

    score_binding = _binding(
        PREDICTION_SCORES_PRODUCT,
        scores,
        primary_dir / "prediction_scores.csv",
    )
    product_paths = {
        "table:prediction_scores": primary_dir / "prediction_scores.csv",
        "table:model_performance": primary_dir / "prediction_performance.csv",
    }
    for action, product, folder in (
        ("prediction.internal_validation", "table:validation", "validation"),
        ("prediction.calibration_metrics", "table:calibration", "calibration"),
        ("prediction.decision_curve", "table:clinical_utility", "utility"),
    ):
        out_dir = tmp_path / folder
        summary = run_prediction_score_analysis(
            action_id=action,
            out_dir=out_dir,
            run_dir=tmp_path,
            resolved_inputs={"step_id": folder, "inputs": {PREDICTION_SCORES_PRODUCT: score_binding}},
            step_id=folder,
        )
        assert summary["output_files"].keys() == {product}
        product_paths[product] = out_dir / next(iter(summary["output_files"].values()))

    figure_bindings = {}
    for key in PREDICTION_COMPOSITE_FIGURE_INPUTS:
        source = pd.read_csv(product_paths[key])
        figure_bindings[key] = _binding(key, source, product_paths[key])
    figure_step = AnalysisStep(
        step_id="prediction_figure",
        planned_analysis_role="auxiliary",
        intent="Render prediction performance and calibration.",
        inputs=list(PREDICTION_COMPOSITE_FIGURE_INPUTS),
        expected_outputs=["figure:prediction_figure"],
        method="visualization",
        input_consumption_contracts=[
            {"input_key": key, "mode": "all_rows"}
            for key in PREDICTION_COMPOSITE_FIGURE_INPUTS
        ],
    )
    assert prediction_figure_executor_owns_step(
        figure_step, resolved_bindings=figure_bindings
    )
    selection = select_standard_executor(
        figure_step,
        plan=AnalysisPlan(research_question="Predict mortality.", steps=[figure_step]),
        resolved_bindings=figure_bindings,
    )
    assert selection is not None
    assert selection.analysis_kind == PREDICTION_FIGURE_ANALYSIS_KIND
    assert selection.host_sealed_renderer is True
    figure_dir = tmp_path / "figure"
    summary = run_prediction_figure(
        out_dir=figure_dir,
        run_dir=tmp_path,
        resolved_inputs={"step_id": figure_step.step_id, "inputs": figure_bindings},
        step_id=figure_step.step_id,
        figure_product="prediction_figure",
    )
    assert summary["status"] == "ok"
    assert summary["paper_authorization_allowed"] is False
    for suffix in ("png", "svg", "pdf", "tiff", "figure_contract.json"):
        assert (figure_dir / f"prediction_figure.{suffix}").is_file()
    contract = json.loads(
        (figure_dir / "prediction_figure.figure_contract.json").read_text("utf-8")
    )
    assert [panel["role"] for panel in contract["panels"]] == [
        "model_performance",
        "model_performance",
        "calibration",
        "validation",
    ]
